
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.ccompiler
2: 
3: Contains CCompiler, an abstract base class that defines the interface
4: for the Distutils compiler abstraction model.'''
5: 
6: __revision__ = "$Id$"
7: 
8: import sys
9: import os
10: import re
11: 
12: from distutils.errors import (CompileError, LinkError, UnknownFileError,
13:                               DistutilsPlatformError, DistutilsModuleError)
14: from distutils.spawn import spawn
15: from distutils.file_util import move_file
16: from distutils.dir_util import mkpath
17: from distutils.dep_util import newer_group
18: from distutils.util import split_quoted, execute
19: from distutils import log
20: # following import is for backward compatibility
21: from distutils.sysconfig import customize_compiler
22: 
23: class CCompiler:
24:     '''Abstract base class to define the interface that must be implemented
25:     by real compiler classes.  Also has some utility methods used by
26:     several compiler classes.
27: 
28:     The basic idea behind a compiler abstraction class is that each
29:     instance can be used for all the compile/link steps in building a
30:     single project.  Thus, attributes common to all of those compile and
31:     link steps -- include directories, macros to define, libraries to link
32:     against, etc. -- are attributes of the compiler instance.  To allow for
33:     variability in how individual files are treated, most of those
34:     attributes may be varied on a per-compilation or per-link basis.
35:     '''
36: 
37:     # 'compiler_type' is a class attribute that identifies this class.  It
38:     # keeps code that wants to know what kind of compiler it's dealing with
39:     # from having to import all possible compiler classes just to do an
40:     # 'isinstance'.  In concrete CCompiler subclasses, 'compiler_type'
41:     # should really, really be one of the keys of the 'compiler_class'
42:     # dictionary (see below -- used by the 'new_compiler()' factory
43:     # function) -- authors of new compiler interface classes are
44:     # responsible for updating 'compiler_class'!
45:     compiler_type = None
46: 
47:     # XXX things not handled by this compiler abstraction model:
48:     #   * client can't provide additional options for a compiler,
49:     #     e.g. warning, optimization, debugging flags.  Perhaps this
50:     #     should be the domain of concrete compiler abstraction classes
51:     #     (UnixCCompiler, MSVCCompiler, etc.) -- or perhaps the base
52:     #     class should have methods for the common ones.
53:     #   * can't completely override the include or library searchg
54:     #     path, ie. no "cc -I -Idir1 -Idir2" or "cc -L -Ldir1 -Ldir2".
55:     #     I'm not sure how widely supported this is even by Unix
56:     #     compilers, much less on other platforms.  And I'm even less
57:     #     sure how useful it is; maybe for cross-compiling, but
58:     #     support for that is a ways off.  (And anyways, cross
59:     #     compilers probably have a dedicated binary with the
60:     #     right paths compiled in.  I hope.)
61:     #   * can't do really freaky things with the library list/library
62:     #     dirs, e.g. "-Ldir1 -lfoo -Ldir2 -lfoo" to link against
63:     #     different versions of libfoo.a in different locations.  I
64:     #     think this is useless without the ability to null out the
65:     #     library search path anyways.
66: 
67: 
68:     # Subclasses that rely on the standard filename generation methods
69:     # implemented below should override these; see the comment near
70:     # those methods ('object_filenames()' et. al.) for details:
71:     src_extensions = None               # list of strings
72:     obj_extension = None                # string
73:     static_lib_extension = None
74:     shared_lib_extension = None         # string
75:     static_lib_format = None            # format string
76:     shared_lib_format = None            # prob. same as static_lib_format
77:     exe_extension = None                # string
78: 
79:     # Default language settings. language_map is used to detect a source
80:     # file or Extension target language, checking source filenames.
81:     # language_order is used to detect the language precedence, when deciding
82:     # what language to use when mixing source types. For example, if some
83:     # extension has two files with ".c" extension, and one with ".cpp", it
84:     # is still linked as c++.
85:     language_map = {".c"   : "c",
86:                     ".cc"  : "c++",
87:                     ".cpp" : "c++",
88:                     ".cxx" : "c++",
89:                     ".m"   : "objc",
90:                    }
91:     language_order = ["c++", "objc", "c"]
92: 
93:     def __init__ (self, verbose=0, dry_run=0, force=0):
94:         self.dry_run = dry_run
95:         self.force = force
96:         self.verbose = verbose
97: 
98:         # 'output_dir': a common output directory for object, library,
99:         # shared object, and shared library files
100:         self.output_dir = None
101: 
102:         # 'macros': a list of macro definitions (or undefinitions).  A
103:         # macro definition is a 2-tuple (name, value), where the value is
104:         # either a string or None (no explicit value).  A macro
105:         # undefinition is a 1-tuple (name,).
106:         self.macros = []
107: 
108:         # 'include_dirs': a list of directories to search for include files
109:         self.include_dirs = []
110: 
111:         # 'libraries': a list of libraries to include in any link
112:         # (library names, not filenames: eg. "foo" not "libfoo.a")
113:         self.libraries = []
114: 
115:         # 'library_dirs': a list of directories to search for libraries
116:         self.library_dirs = []
117: 
118:         # 'runtime_library_dirs': a list of directories to search for
119:         # shared libraries/objects at runtime
120:         self.runtime_library_dirs = []
121: 
122:         # 'objects': a list of object files (or similar, such as explicitly
123:         # named library files) to include on any link
124:         self.objects = []
125: 
126:         for key in self.executables.keys():
127:             self.set_executable(key, self.executables[key])
128: 
129:     def set_executables(self, **args):
130:         '''Define the executables (and options for them) that will be run
131:         to perform the various stages of compilation.  The exact set of
132:         executables that may be specified here depends on the compiler
133:         class (via the 'executables' class attribute), but most will have:
134:           compiler      the C/C++ compiler
135:           linker_so     linker used to create shared objects and libraries
136:           linker_exe    linker used to create binary executables
137:           archiver      static library creator
138: 
139:         On platforms with a command-line (Unix, DOS/Windows), each of these
140:         is a string that will be split into executable name and (optional)
141:         list of arguments.  (Splitting the string is done similarly to how
142:         Unix shells operate: words are delimited by spaces, but quotes and
143:         backslashes can override this.  See
144:         'distutils.util.split_quoted()'.)
145:         '''
146: 
147:         # Note that some CCompiler implementation classes will define class
148:         # attributes 'cpp', 'cc', etc. with hard-coded executable names;
149:         # this is appropriate when a compiler class is for exactly one
150:         # compiler/OS combination (eg. MSVCCompiler).  Other compiler
151:         # classes (UnixCCompiler, in particular) are driven by information
152:         # discovered at run-time, since there are many different ways to do
153:         # basically the same things with Unix C compilers.
154: 
155:         for key in args.keys():
156:             if key not in self.executables:
157:                 raise ValueError, \
158:                       "unknown executable '%s' for class %s" % \
159:                       (key, self.__class__.__name__)
160:             self.set_executable(key, args[key])
161: 
162:     def set_executable(self, key, value):
163:         if isinstance(value, str):
164:             setattr(self, key, split_quoted(value))
165:         else:
166:             setattr(self, key, value)
167: 
168:     def _find_macro(self, name):
169:         i = 0
170:         for defn in self.macros:
171:             if defn[0] == name:
172:                 return i
173:             i = i + 1
174:         return None
175: 
176:     def _check_macro_definitions(self, definitions):
177:         '''Ensures that every element of 'definitions' is a valid macro
178:         definition, ie. either (name,value) 2-tuple or a (name,) tuple.  Do
179:         nothing if all definitions are OK, raise TypeError otherwise.
180:         '''
181:         for defn in definitions:
182:             if not (isinstance(defn, tuple) and
183:                     (len (defn) == 1 or
184:                      (len (defn) == 2 and
185:                       (isinstance(defn[1], str) or defn[1] is None))) and
186:                     isinstance(defn[0], str)):
187:                 raise TypeError, \
188:                       ("invalid macro definition '%s': " % defn) + \
189:                       "must be tuple (string,), (string, string), or " + \
190:                       "(string, None)"
191: 
192: 
193:     # -- Bookkeeping methods -------------------------------------------
194: 
195:     def define_macro(self, name, value=None):
196:         '''Define a preprocessor macro for all compilations driven by this
197:         compiler object.  The optional parameter 'value' should be a
198:         string; if it is not supplied, then the macro will be defined
199:         without an explicit value and the exact outcome depends on the
200:         compiler used (XXX true? does ANSI say anything about this?)
201:         '''
202:         # Delete from the list of macro definitions/undefinitions if
203:         # already there (so that this one will take precedence).
204:         i = self._find_macro (name)
205:         if i is not None:
206:             del self.macros[i]
207: 
208:         defn = (name, value)
209:         self.macros.append (defn)
210: 
211:     def undefine_macro(self, name):
212:         '''Undefine a preprocessor macro for all compilations driven by
213:         this compiler object.  If the same macro is defined by
214:         'define_macro()' and undefined by 'undefine_macro()' the last call
215:         takes precedence (including multiple redefinitions or
216:         undefinitions).  If the macro is redefined/undefined on a
217:         per-compilation basis (ie. in the call to 'compile()'), then that
218:         takes precedence.
219:         '''
220:         # Delete from the list of macro definitions/undefinitions if
221:         # already there (so that this one will take precedence).
222:         i = self._find_macro (name)
223:         if i is not None:
224:             del self.macros[i]
225: 
226:         undefn = (name,)
227:         self.macros.append (undefn)
228: 
229:     def add_include_dir(self, dir):
230:         '''Add 'dir' to the list of directories that will be searched for
231:         header files.  The compiler is instructed to search directories in
232:         the order in which they are supplied by successive calls to
233:         'add_include_dir()'.
234:         '''
235:         self.include_dirs.append (dir)
236: 
237:     def set_include_dirs(self, dirs):
238:         '''Set the list of directories that will be searched to 'dirs' (a
239:         list of strings).  Overrides any preceding calls to
240:         'add_include_dir()'; subsequence calls to 'add_include_dir()' add
241:         to the list passed to 'set_include_dirs()'.  This does not affect
242:         any list of standard include directories that the compiler may
243:         search by default.
244:         '''
245:         self.include_dirs = dirs[:]
246: 
247:     def add_library(self, libname):
248:         '''Add 'libname' to the list of libraries that will be included in
249:         all links driven by this compiler object.  Note that 'libname'
250:         should *not* be the name of a file containing a library, but the
251:         name of the library itself: the actual filename will be inferred by
252:         the linker, the compiler, or the compiler class (depending on the
253:         platform).
254: 
255:         The linker will be instructed to link against libraries in the
256:         order they were supplied to 'add_library()' and/or
257:         'set_libraries()'.  It is perfectly valid to duplicate library
258:         names; the linker will be instructed to link against libraries as
259:         many times as they are mentioned.
260:         '''
261:         self.libraries.append (libname)
262: 
263:     def set_libraries(self, libnames):
264:         '''Set the list of libraries to be included in all links driven by
265:         this compiler object to 'libnames' (a list of strings).  This does
266:         not affect any standard system libraries that the linker may
267:         include by default.
268:         '''
269:         self.libraries = libnames[:]
270: 
271: 
272:     def add_library_dir(self, dir):
273:         '''Add 'dir' to the list of directories that will be searched for
274:         libraries specified to 'add_library()' and 'set_libraries()'.  The
275:         linker will be instructed to search for libraries in the order they
276:         are supplied to 'add_library_dir()' and/or 'set_library_dirs()'.
277:         '''
278:         self.library_dirs.append(dir)
279: 
280:     def set_library_dirs(self, dirs):
281:         '''Set the list of library search directories to 'dirs' (a list of
282:         strings).  This does not affect any standard library search path
283:         that the linker may search by default.
284:         '''
285:         self.library_dirs = dirs[:]
286: 
287:     def add_runtime_library_dir(self, dir):
288:         '''Add 'dir' to the list of directories that will be searched for
289:         shared libraries at runtime.
290:         '''
291:         self.runtime_library_dirs.append(dir)
292: 
293:     def set_runtime_library_dirs(self, dirs):
294:         '''Set the list of directories to search for shared libraries at
295:         runtime to 'dirs' (a list of strings).  This does not affect any
296:         standard search path that the runtime linker may search by
297:         default.
298:         '''
299:         self.runtime_library_dirs = dirs[:]
300: 
301:     def add_link_object(self, object):
302:         '''Add 'object' to the list of object files (or analogues, such as
303:         explicitly named library files or the output of "resource
304:         compilers") to be included in every link driven by this compiler
305:         object.
306:         '''
307:         self.objects.append(object)
308: 
309:     def set_link_objects(self, objects):
310:         '''Set the list of object files (or analogues) to be included in
311:         every link to 'objects'.  This does not affect any standard object
312:         files that the linker may include by default (such as system
313:         libraries).
314:         '''
315:         self.objects = objects[:]
316: 
317: 
318:     # -- Private utility methods --------------------------------------
319:     # (here for the convenience of subclasses)
320: 
321:     # Helper method to prep compiler in subclass compile() methods
322: 
323:     def _setup_compile(self, outdir, macros, incdirs, sources, depends,
324:                        extra):
325:         '''Process arguments and decide which source files to compile.'''
326:         if outdir is None:
327:             outdir = self.output_dir
328:         elif not isinstance(outdir, str):
329:             raise TypeError, "'output_dir' must be a string or None"
330: 
331:         if macros is None:
332:             macros = self.macros
333:         elif isinstance(macros, list):
334:             macros = macros + (self.macros or [])
335:         else:
336:             raise TypeError, "'macros' (if supplied) must be a list of tuples"
337: 
338:         if incdirs is None:
339:             incdirs = self.include_dirs
340:         elif isinstance(incdirs, (list, tuple)):
341:             incdirs = list(incdirs) + (self.include_dirs or [])
342:         else:
343:             raise TypeError, \
344:                   "'include_dirs' (if supplied) must be a list of strings"
345: 
346:         if extra is None:
347:             extra = []
348: 
349:         # Get the list of expected output (object) files
350:         objects = self.object_filenames(sources,
351:                                         strip_dir=0,
352:                                         output_dir=outdir)
353:         assert len(objects) == len(sources)
354: 
355:         pp_opts = gen_preprocess_options(macros, incdirs)
356: 
357:         build = {}
358:         for i in range(len(sources)):
359:             src = sources[i]
360:             obj = objects[i]
361:             ext = os.path.splitext(src)[1]
362:             self.mkpath(os.path.dirname(obj))
363:             build[obj] = (src, ext)
364: 
365:         return macros, objects, extra, pp_opts, build
366: 
367:     def _get_cc_args(self, pp_opts, debug, before):
368:         # works for unixccompiler, emxccompiler, cygwinccompiler
369:         cc_args = pp_opts + ['-c']
370:         if debug:
371:             cc_args[:0] = ['-g']
372:         if before:
373:             cc_args[:0] = before
374:         return cc_args
375: 
376:     def _fix_compile_args(self, output_dir, macros, include_dirs):
377:         '''Typecheck and fix-up some of the arguments to the 'compile()'
378:         method, and return fixed-up values.  Specifically: if 'output_dir'
379:         is None, replaces it with 'self.output_dir'; ensures that 'macros'
380:         is a list, and augments it with 'self.macros'; ensures that
381:         'include_dirs' is a list, and augments it with 'self.include_dirs'.
382:         Guarantees that the returned values are of the correct type,
383:         i.e. for 'output_dir' either string or None, and for 'macros' and
384:         'include_dirs' either list or None.
385:         '''
386:         if output_dir is None:
387:             output_dir = self.output_dir
388:         elif not isinstance(output_dir, str):
389:             raise TypeError, "'output_dir' must be a string or None"
390: 
391:         if macros is None:
392:             macros = self.macros
393:         elif isinstance(macros, list):
394:             macros = macros + (self.macros or [])
395:         else:
396:             raise TypeError, "'macros' (if supplied) must be a list of tuples"
397: 
398:         if include_dirs is None:
399:             include_dirs = self.include_dirs
400:         elif isinstance(include_dirs, (list, tuple)):
401:             include_dirs = list (include_dirs) + (self.include_dirs or [])
402:         else:
403:             raise TypeError, \
404:                   "'include_dirs' (if supplied) must be a list of strings"
405: 
406:         return output_dir, macros, include_dirs
407: 
408:     def _fix_object_args(self, objects, output_dir):
409:         '''Typecheck and fix up some arguments supplied to various methods.
410:         Specifically: ensure that 'objects' is a list; if output_dir is
411:         None, replace with self.output_dir.  Return fixed versions of
412:         'objects' and 'output_dir'.
413:         '''
414:         if not isinstance(objects, (list, tuple)):
415:             raise TypeError, \
416:                   "'objects' must be a list or tuple of strings"
417:         objects = list (objects)
418: 
419:         if output_dir is None:
420:             output_dir = self.output_dir
421:         elif not isinstance(output_dir, str):
422:             raise TypeError, "'output_dir' must be a string or None"
423: 
424:         return (objects, output_dir)
425: 
426:     def _fix_lib_args(self, libraries, library_dirs, runtime_library_dirs):
427:         '''Typecheck and fix up some of the arguments supplied to the
428:         'link_*' methods.  Specifically: ensure that all arguments are
429:         lists, and augment them with their permanent versions
430:         (eg. 'self.libraries' augments 'libraries').  Return a tuple with
431:         fixed versions of all arguments.
432:         '''
433:         if libraries is None:
434:             libraries = self.libraries
435:         elif isinstance(libraries, (list, tuple)):
436:             libraries = list (libraries) + (self.libraries or [])
437:         else:
438:             raise TypeError, \
439:                   "'libraries' (if supplied) must be a list of strings"
440: 
441:         if library_dirs is None:
442:             library_dirs = self.library_dirs
443:         elif isinstance(library_dirs, (list, tuple)):
444:             library_dirs = list (library_dirs) + (self.library_dirs or [])
445:         else:
446:             raise TypeError, \
447:                   "'library_dirs' (if supplied) must be a list of strings"
448: 
449:         if runtime_library_dirs is None:
450:             runtime_library_dirs = self.runtime_library_dirs
451:         elif isinstance(runtime_library_dirs, (list, tuple)):
452:             runtime_library_dirs = (list (runtime_library_dirs) +
453:                                     (self.runtime_library_dirs or []))
454:         else:
455:             raise TypeError, \
456:                   "'runtime_library_dirs' (if supplied) " + \
457:                   "must be a list of strings"
458: 
459:         return (libraries, library_dirs, runtime_library_dirs)
460: 
461:     def _need_link(self, objects, output_file):
462:         '''Return true if we need to relink the files listed in 'objects'
463:         to recreate 'output_file'.
464:         '''
465:         if self.force:
466:             return 1
467:         else:
468:             if self.dry_run:
469:                 newer = newer_group (objects, output_file, missing='newer')
470:             else:
471:                 newer = newer_group (objects, output_file)
472:             return newer
473: 
474:     def detect_language(self, sources):
475:         '''Detect the language of a given file, or list of files. Uses
476:         language_map, and language_order to do the job.
477:         '''
478:         if not isinstance(sources, list):
479:             sources = [sources]
480:         lang = None
481:         index = len(self.language_order)
482:         for source in sources:
483:             base, ext = os.path.splitext(source)
484:             extlang = self.language_map.get(ext)
485:             try:
486:                 extindex = self.language_order.index(extlang)
487:                 if extindex < index:
488:                     lang = extlang
489:                     index = extindex
490:             except ValueError:
491:                 pass
492:         return lang
493: 
494:     # -- Worker methods ------------------------------------------------
495:     # (must be implemented by subclasses)
496: 
497:     def preprocess(self, source, output_file=None, macros=None,
498:                    include_dirs=None, extra_preargs=None, extra_postargs=None):
499:         '''Preprocess a single C/C++ source file, named in 'source'.
500:         Output will be written to file named 'output_file', or stdout if
501:         'output_file' not supplied.  'macros' is a list of macro
502:         definitions as for 'compile()', which will augment the macros set
503:         with 'define_macro()' and 'undefine_macro()'.  'include_dirs' is a
504:         list of directory names that will be added to the default list.
505: 
506:         Raises PreprocessError on failure.
507:         '''
508:         pass
509: 
510:     def compile(self, sources, output_dir=None, macros=None,
511:                 include_dirs=None, debug=0, extra_preargs=None,
512:                 extra_postargs=None, depends=None):
513:         '''Compile one or more source files.
514: 
515:         'sources' must be a list of filenames, most likely C/C++
516:         files, but in reality anything that can be handled by a
517:         particular compiler and compiler class (eg. MSVCCompiler can
518:         handle resource files in 'sources').  Return a list of object
519:         filenames, one per source filename in 'sources'.  Depending on
520:         the implementation, not all source files will necessarily be
521:         compiled, but all corresponding object filenames will be
522:         returned.
523: 
524:         If 'output_dir' is given, object files will be put under it, while
525:         retaining their original path component.  That is, "foo/bar.c"
526:         normally compiles to "foo/bar.o" (for a Unix implementation); if
527:         'output_dir' is "build", then it would compile to
528:         "build/foo/bar.o".
529: 
530:         'macros', if given, must be a list of macro definitions.  A macro
531:         definition is either a (name, value) 2-tuple or a (name,) 1-tuple.
532:         The former defines a macro; if the value is None, the macro is
533:         defined without an explicit value.  The 1-tuple case undefines a
534:         macro.  Later definitions/redefinitions/ undefinitions take
535:         precedence.
536: 
537:         'include_dirs', if given, must be a list of strings, the
538:         directories to add to the default include file search path for this
539:         compilation only.
540: 
541:         'debug' is a boolean; if true, the compiler will be instructed to
542:         output debug symbols in (or alongside) the object file(s).
543: 
544:         'extra_preargs' and 'extra_postargs' are implementation- dependent.
545:         On platforms that have the notion of a command-line (e.g. Unix,
546:         DOS/Windows), they are most likely lists of strings: extra
547:         command-line arguments to prepand/append to the compiler command
548:         line.  On other platforms, consult the implementation class
549:         documentation.  In any event, they are intended as an escape hatch
550:         for those occasions when the abstract compiler framework doesn't
551:         cut the mustard.
552: 
553:         'depends', if given, is a list of filenames that all targets
554:         depend on.  If a source file is older than any file in
555:         depends, then the source file will be recompiled.  This
556:         supports dependency tracking, but only at a coarse
557:         granularity.
558: 
559:         Raises CompileError on failure.
560:         '''
561:         # A concrete compiler class can either override this method
562:         # entirely or implement _compile().
563: 
564:         macros, objects, extra_postargs, pp_opts, build = \
565:                 self._setup_compile(output_dir, macros, include_dirs, sources,
566:                                     depends, extra_postargs)
567:         cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
568: 
569:         for obj in objects:
570:             try:
571:                 src, ext = build[obj]
572:             except KeyError:
573:                 continue
574:             self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
575: 
576:         # Return *all* object filenames, not just the ones we just built.
577:         return objects
578: 
579:     def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
580:         '''Compile 'src' to product 'obj'.'''
581: 
582:         # A concrete compiler class that does not override compile()
583:         # should implement _compile().
584:         pass
585: 
586:     def create_static_lib(self, objects, output_libname, output_dir=None,
587:                           debug=0, target_lang=None):
588:         '''Link a bunch of stuff together to create a static library file.
589:         The "bunch of stuff" consists of the list of object files supplied
590:         as 'objects', the extra object files supplied to
591:         'add_link_object()' and/or 'set_link_objects()', the libraries
592:         supplied to 'add_library()' and/or 'set_libraries()', and the
593:         libraries supplied as 'libraries' (if any).
594: 
595:         'output_libname' should be a library name, not a filename; the
596:         filename will be inferred from the library name.  'output_dir' is
597:         the directory where the library file will be put.
598: 
599:         'debug' is a boolean; if true, debugging information will be
600:         included in the library (note that on most platforms, it is the
601:         compile step where this matters: the 'debug' flag is included here
602:         just for consistency).
603: 
604:         'target_lang' is the target language for which the given objects
605:         are being compiled. This allows specific linkage time treatment of
606:         certain languages.
607: 
608:         Raises LibError on failure.
609:         '''
610:         pass
611: 
612:     # values for target_desc parameter in link()
613:     SHARED_OBJECT = "shared_object"
614:     SHARED_LIBRARY = "shared_library"
615:     EXECUTABLE = "executable"
616: 
617:     def link(self, target_desc, objects, output_filename, output_dir=None,
618:              libraries=None, library_dirs=None, runtime_library_dirs=None,
619:              export_symbols=None, debug=0, extra_preargs=None,
620:              extra_postargs=None, build_temp=None, target_lang=None):
621:         '''Link a bunch of stuff together to create an executable or
622:         shared library file.
623: 
624:         The "bunch of stuff" consists of the list of object files supplied
625:         as 'objects'.  'output_filename' should be a filename.  If
626:         'output_dir' is supplied, 'output_filename' is relative to it
627:         (i.e. 'output_filename' can provide directory components if
628:         needed).
629: 
630:         'libraries' is a list of libraries to link against.  These are
631:         library names, not filenames, since they're translated into
632:         filenames in a platform-specific way (eg. "foo" becomes "libfoo.a"
633:         on Unix and "foo.lib" on DOS/Windows).  However, they can include a
634:         directory component, which means the linker will look in that
635:         specific directory rather than searching all the normal locations.
636: 
637:         'library_dirs', if supplied, should be a list of directories to
638:         search for libraries that were specified as bare library names
639:         (ie. no directory component).  These are on top of the system
640:         default and those supplied to 'add_library_dir()' and/or
641:         'set_library_dirs()'.  'runtime_library_dirs' is a list of
642:         directories that will be embedded into the shared library and used
643:         to search for other shared libraries that *it* depends on at
644:         run-time.  (This may only be relevant on Unix.)
645: 
646:         'export_symbols' is a list of symbols that the shared library will
647:         export.  (This appears to be relevant only on Windows.)
648: 
649:         'debug' is as for 'compile()' and 'create_static_lib()', with the
650:         slight distinction that it actually matters on most platforms (as
651:         opposed to 'create_static_lib()', which includes a 'debug' flag
652:         mostly for form's sake).
653: 
654:         'extra_preargs' and 'extra_postargs' are as for 'compile()' (except
655:         of course that they supply command-line arguments for the
656:         particular linker being used).
657: 
658:         'target_lang' is the target language for which the given objects
659:         are being compiled. This allows specific linkage time treatment of
660:         certain languages.
661: 
662:         Raises LinkError on failure.
663:         '''
664:         raise NotImplementedError
665: 
666: 
667:     # Old 'link_*()' methods, rewritten to use the new 'link()' method.
668: 
669:     def link_shared_lib(self, objects, output_libname, output_dir=None,
670:                         libraries=None, library_dirs=None,
671:                         runtime_library_dirs=None, export_symbols=None,
672:                         debug=0, extra_preargs=None, extra_postargs=None,
673:                         build_temp=None, target_lang=None):
674:         self.link(CCompiler.SHARED_LIBRARY, objects,
675:                   self.library_filename(output_libname, lib_type='shared'),
676:                   output_dir,
677:                   libraries, library_dirs, runtime_library_dirs,
678:                   export_symbols, debug,
679:                   extra_preargs, extra_postargs, build_temp, target_lang)
680: 
681: 
682:     def link_shared_object(self, objects, output_filename, output_dir=None,
683:                            libraries=None, library_dirs=None,
684:                            runtime_library_dirs=None, export_symbols=None,
685:                            debug=0, extra_preargs=None, extra_postargs=None,
686:                            build_temp=None, target_lang=None):
687:         self.link(CCompiler.SHARED_OBJECT, objects,
688:                   output_filename, output_dir,
689:                   libraries, library_dirs, runtime_library_dirs,
690:                   export_symbols, debug,
691:                   extra_preargs, extra_postargs, build_temp, target_lang)
692: 
693:     def link_executable(self, objects, output_progname, output_dir=None,
694:                         libraries=None, library_dirs=None,
695:                         runtime_library_dirs=None, debug=0, extra_preargs=None,
696:                         extra_postargs=None, target_lang=None):
697:         self.link(CCompiler.EXECUTABLE, objects,
698:                   self.executable_filename(output_progname), output_dir,
699:                   libraries, library_dirs, runtime_library_dirs, None,
700:                   debug, extra_preargs, extra_postargs, None, target_lang)
701: 
702: 
703:     # -- Miscellaneous methods -----------------------------------------
704:     # These are all used by the 'gen_lib_options() function; there is
705:     # no appropriate default implementation so subclasses should
706:     # implement all of these.
707: 
708:     def library_dir_option(self, dir):
709:         '''Return the compiler option to add 'dir' to the list of
710:         directories searched for libraries.
711:         '''
712:         raise NotImplementedError
713: 
714:     def runtime_library_dir_option(self, dir):
715:         '''Return the compiler option to add 'dir' to the list of
716:         directories searched for runtime libraries.
717:         '''
718:         raise NotImplementedError
719: 
720:     def library_option(self, lib):
721:         '''Return the compiler option to add 'lib' to the list of libraries
722:         linked into the shared library or executable.
723:         '''
724:         raise NotImplementedError
725: 
726:     def has_function(self, funcname, includes=None, include_dirs=None,
727:                      libraries=None, library_dirs=None):
728:         '''Return a boolean indicating whether funcname is supported on
729:         the current platform.  The optional arguments can be used to
730:         augment the compilation environment.
731:         '''
732: 
733:         # this can't be included at module scope because it tries to
734:         # import math which might not be available at that point - maybe
735:         # the necessary logic should just be inlined?
736:         import tempfile
737:         if includes is None:
738:             includes = []
739:         if include_dirs is None:
740:             include_dirs = []
741:         if libraries is None:
742:             libraries = []
743:         if library_dirs is None:
744:             library_dirs = []
745:         fd, fname = tempfile.mkstemp(".c", funcname, text=True)
746:         f = os.fdopen(fd, "w")
747:         try:
748:             for incl in includes:
749:                 f.write('''#include "%s"\n''' % incl)
750:             f.write('''\
751: main (int argc, char **argv) {
752:     %s();
753: }
754: ''' % funcname)
755:         finally:
756:             f.close()
757:         try:
758:             objects = self.compile([fname], include_dirs=include_dirs)
759:         except CompileError:
760:             return False
761: 
762:         try:
763:             self.link_executable(objects, "a.out",
764:                                  libraries=libraries,
765:                                  library_dirs=library_dirs)
766:         except (LinkError, TypeError):
767:             return False
768:         return True
769: 
770:     def find_library_file (self, dirs, lib, debug=0):
771:         '''Search the specified list of directories for a static or shared
772:         library file 'lib' and return the full path to that file.  If
773:         'debug' true, look for a debugging version (if that makes sense on
774:         the current platform).  Return None if 'lib' wasn't found in any of
775:         the specified directories.
776:         '''
777:         raise NotImplementedError
778: 
779:     # -- Filename generation methods -----------------------------------
780: 
781:     # The default implementation of the filename generating methods are
782:     # prejudiced towards the Unix/DOS/Windows view of the world:
783:     #   * object files are named by replacing the source file extension
784:     #     (eg. .c/.cpp -> .o/.obj)
785:     #   * library files (shared or static) are named by plugging the
786:     #     library name and extension into a format string, eg.
787:     #     "lib%s.%s" % (lib_name, ".a") for Unix static libraries
788:     #   * executables are named by appending an extension (possibly
789:     #     empty) to the program name: eg. progname + ".exe" for
790:     #     Windows
791:     #
792:     # To reduce redundant code, these methods expect to find
793:     # several attributes in the current object (presumably defined
794:     # as class attributes):
795:     #   * src_extensions -
796:     #     list of C/C++ source file extensions, eg. ['.c', '.cpp']
797:     #   * obj_extension -
798:     #     object file extension, eg. '.o' or '.obj'
799:     #   * static_lib_extension -
800:     #     extension for static library files, eg. '.a' or '.lib'
801:     #   * shared_lib_extension -
802:     #     extension for shared library/object files, eg. '.so', '.dll'
803:     #   * static_lib_format -
804:     #     format string for generating static library filenames,
805:     #     eg. 'lib%s.%s' or '%s.%s'
806:     #   * shared_lib_format
807:     #     format string for generating shared library filenames
808:     #     (probably same as static_lib_format, since the extension
809:     #     is one of the intended parameters to the format string)
810:     #   * exe_extension -
811:     #     extension for executable files, eg. '' or '.exe'
812: 
813:     def object_filenames(self, source_filenames, strip_dir=0, output_dir=''):
814:         if output_dir is None:
815:             output_dir = ''
816:         obj_names = []
817:         for src_name in source_filenames:
818:             base, ext = os.path.splitext(src_name)
819:             base = os.path.splitdrive(base)[1] # Chop off the drive
820:             base = base[os.path.isabs(base):]  # If abs, chop off leading /
821:             if ext not in self.src_extensions:
822:                 raise UnknownFileError, \
823:                       "unknown file type '%s' (from '%s')" % (ext, src_name)
824:             if strip_dir:
825:                 base = os.path.basename(base)
826:             obj_names.append(os.path.join(output_dir,
827:                                           base + self.obj_extension))
828:         return obj_names
829: 
830:     def shared_object_filename(self, basename, strip_dir=0, output_dir=''):
831:         assert output_dir is not None
832:         if strip_dir:
833:             basename = os.path.basename (basename)
834:         return os.path.join(output_dir, basename + self.shared_lib_extension)
835: 
836:     def executable_filename(self, basename, strip_dir=0, output_dir=''):
837:         assert output_dir is not None
838:         if strip_dir:
839:             basename = os.path.basename (basename)
840:         return os.path.join(output_dir, basename + (self.exe_extension or ''))
841: 
842:     def library_filename(self, libname, lib_type='static',     # or 'shared'
843:                          strip_dir=0, output_dir=''):
844:         assert output_dir is not None
845:         if lib_type not in ("static", "shared", "dylib", "xcode_stub"):
846:             raise ValueError, (''''lib_type' must be "static", "shared", '''
847:                                '''"dylib", or "xcode_stub".''')
848:         fmt = getattr(self, lib_type + "_lib_format")
849:         ext = getattr(self, lib_type + "_lib_extension")
850: 
851:         dir, base = os.path.split (libname)
852:         filename = fmt % (base, ext)
853:         if strip_dir:
854:             dir = ''
855: 
856:         return os.path.join(output_dir, dir, filename)
857: 
858: 
859:     # -- Utility methods -----------------------------------------------
860: 
861:     def announce(self, msg, level=1):
862:         log.debug(msg)
863: 
864:     def debug_print(self, msg):
865:         from distutils.debug import DEBUG
866:         if DEBUG:
867:             print msg
868: 
869:     def warn(self, msg):
870:         sys.stderr.write("warning: %s\n" % msg)
871: 
872:     def execute(self, func, args, msg=None, level=1):
873:         execute(func, args, msg, self.dry_run)
874: 
875:     def spawn(self, cmd):
876:         spawn(cmd, dry_run=self.dry_run)
877: 
878:     def move_file(self, src, dst):
879:         return move_file(src, dst, dry_run=self.dry_run)
880: 
881:     def mkpath(self, name, mode=0777):
882:         mkpath(name, mode, dry_run=self.dry_run)
883: 
884: 
885: # class CCompiler
886: 
887: 
888: # Map a sys.platform/os.name ('posix', 'nt') to the default compiler
889: # type for that platform. Keys are interpreted as re match
890: # patterns. Order is important; platform mappings are preferred over
891: # OS names.
892: _default_compilers = (
893: 
894:     # Platform string mappings
895: 
896:     # on a cygwin built python we can use gcc like an ordinary UNIXish
897:     # compiler
898:     ('cygwin.*', 'unix'),
899:     ('os2emx', 'emx'),
900: 
901:     # OS name mappings
902:     ('posix', 'unix'),
903:     ('nt', 'msvc'),
904: 
905:     )
906: 
907: def get_default_compiler(osname=None, platform=None):
908:     ''' Determine the default compiler to use for the given platform.
909: 
910:         osname should be one of the standard Python OS names (i.e. the
911:         ones returned by os.name) and platform the common value
912:         returned by sys.platform for the platform in question.
913: 
914:         The default values are os.name and sys.platform in case the
915:         parameters are not given.
916: 
917:     '''
918:     if osname is None:
919:         osname = os.name
920:     if platform is None:
921:         platform = sys.platform
922:     for pattern, compiler in _default_compilers:
923:         if re.match(pattern, platform) is not None or \
924:            re.match(pattern, osname) is not None:
925:             return compiler
926:     # Default to Unix compiler
927:     return 'unix'
928: 
929: # Map compiler types to (module_name, class_name) pairs -- ie. where to
930: # find the code that implements an interface to this compiler.  (The module
931: # is assumed to be in the 'distutils' package.)
932: compiler_class = { 'unix':    ('unixccompiler', 'UnixCCompiler',
933:                                "standard UNIX-style compiler"),
934:                    'msvc':    ('msvccompiler', 'MSVCCompiler',
935:                                "Microsoft Visual C++"),
936:                    'cygwin':  ('cygwinccompiler', 'CygwinCCompiler',
937:                                "Cygwin port of GNU C Compiler for Win32"),
938:                    'mingw32': ('cygwinccompiler', 'Mingw32CCompiler',
939:                                "Mingw32 port of GNU C Compiler for Win32"),
940:                    'bcpp':    ('bcppcompiler', 'BCPPCompiler',
941:                                "Borland C++ Compiler"),
942:                    'emx':     ('emxccompiler', 'EMXCCompiler',
943:                                "EMX port of GNU C Compiler for OS/2"),
944:                  }
945: 
946: def show_compilers():
947:     '''Print list of available compilers (used by the "--help-compiler"
948:     options to "build", "build_ext", "build_clib").
949:     '''
950:     # XXX this "knows" that the compiler option it's describing is
951:     # "--compiler", which just happens to be the case for the three
952:     # commands that use it.
953:     from distutils.fancy_getopt import FancyGetopt
954:     compilers = []
955:     for compiler in compiler_class.keys():
956:         compilers.append(("compiler="+compiler, None,
957:                           compiler_class[compiler][2]))
958:     compilers.sort()
959:     pretty_printer = FancyGetopt(compilers)
960:     pretty_printer.print_help("List of available compilers:")
961: 
962: 
963: def new_compiler(plat=None, compiler=None, verbose=0, dry_run=0, force=0):
964:     '''Generate an instance of some CCompiler subclass for the supplied
965:     platform/compiler combination.  'plat' defaults to 'os.name'
966:     (eg. 'posix', 'nt'), and 'compiler' defaults to the default compiler
967:     for that platform.  Currently only 'posix' and 'nt' are supported, and
968:     the default compilers are "traditional Unix interface" (UnixCCompiler
969:     class) and Visual C++ (MSVCCompiler class).  Note that it's perfectly
970:     possible to ask for a Unix compiler object under Windows, and a
971:     Microsoft compiler object under Unix -- if you supply a value for
972:     'compiler', 'plat' is ignored.
973:     '''
974:     if plat is None:
975:         plat = os.name
976: 
977:     try:
978:         if compiler is None:
979:             compiler = get_default_compiler(plat)
980: 
981:         (module_name, class_name, long_description) = compiler_class[compiler]
982:     except KeyError:
983:         msg = "don't know how to compile C/C++ code on platform '%s'" % plat
984:         if compiler is not None:
985:             msg = msg + " with '%s' compiler" % compiler
986:         raise DistutilsPlatformError, msg
987: 
988:     try:
989:         module_name = "distutils." + module_name
990:         __import__ (module_name)
991:         module = sys.modules[module_name]
992:         klass = vars(module)[class_name]
993:     except ImportError:
994:         raise DistutilsModuleError, \
995:               "can't compile C/C++ code: unable to load module '%s'" % \
996:               module_name
997:     except KeyError:
998:         raise DistutilsModuleError, \
999:               ("can't compile C/C++ code: unable to find class '%s' " +
1000:                "in module '%s'") % (class_name, module_name)
1001: 
1002:     # XXX The None is necessary to preserve backwards compatibility
1003:     # with classes that expect verbose to be the first positional
1004:     # argument.
1005:     return klass(None, dry_run, force)
1006: 
1007: 
1008: def gen_preprocess_options(macros, include_dirs):
1009:     '''Generate C pre-processor options (-D, -U, -I) as used by at least
1010:     two types of compilers: the typical Unix compiler and Visual C++.
1011:     'macros' is the usual thing, a list of 1- or 2-tuples, where (name,)
1012:     means undefine (-U) macro 'name', and (name,value) means define (-D)
1013:     macro 'name' to 'value'.  'include_dirs' is just a list of directory
1014:     names to be added to the header file search path (-I).  Returns a list
1015:     of command-line options suitable for either Unix compilers or Visual
1016:     C++.
1017:     '''
1018:     # XXX it would be nice (mainly aesthetic, and so we don't generate
1019:     # stupid-looking command lines) to go over 'macros' and eliminate
1020:     # redundant definitions/undefinitions (ie. ensure that only the
1021:     # latest mention of a particular macro winds up on the command
1022:     # line).  I don't think it's essential, though, since most (all?)
1023:     # Unix C compilers only pay attention to the latest -D or -U
1024:     # mention of a macro on their command line.  Similar situation for
1025:     # 'include_dirs'.  I'm punting on both for now.  Anyways, weeding out
1026:     # redundancies like this should probably be the province of
1027:     # CCompiler, since the data structures used are inherited from it
1028:     # and therefore common to all CCompiler classes.
1029: 
1030:     pp_opts = []
1031:     for macro in macros:
1032: 
1033:         if not (isinstance(macro, tuple) and
1034:                 1 <= len (macro) <= 2):
1035:             raise TypeError, \
1036:                   ("bad macro definition '%s': " +
1037:                    "each element of 'macros' list must be a 1- or 2-tuple") % \
1038:                   macro
1039: 
1040:         if len (macro) == 1:        # undefine this macro
1041:             pp_opts.append ("-U%s" % macro[0])
1042:         elif len (macro) == 2:
1043:             if macro[1] is None:    # define with no explicit value
1044:                 pp_opts.append ("-D%s" % macro[0])
1045:             else:
1046:                 # XXX *don't* need to be clever about quoting the
1047:                 # macro value here, because we're going to avoid the
1048:                 # shell at all costs when we spawn the command!
1049:                 pp_opts.append ("-D%s=%s" % macro)
1050: 
1051:     for dir in include_dirs:
1052:         pp_opts.append ("-I%s" % dir)
1053: 
1054:     return pp_opts
1055: 
1056: 
1057: def gen_lib_options(compiler, library_dirs, runtime_library_dirs, libraries):
1058:     '''Generate linker options for searching library directories and
1059:     linking with specific libraries.
1060: 
1061:     'libraries' and 'library_dirs' are, respectively, lists of library names
1062:     (not filenames!) and search directories.  Returns a list of command-line
1063:     options suitable for use with some compiler (depending on the two format
1064:     strings passed in).
1065:     '''
1066:     lib_opts = []
1067: 
1068:     for dir in library_dirs:
1069:         lib_opts.append(compiler.library_dir_option(dir))
1070: 
1071:     for dir in runtime_library_dirs:
1072:         opt = compiler.runtime_library_dir_option(dir)
1073:         if isinstance(opt, list):
1074:             lib_opts.extend(opt)
1075:         else:
1076:             lib_opts.append(opt)
1077: 
1078:     # XXX it's important that we *not* remove redundant library mentions!
1079:     # sometimes you really do have to say "-lfoo -lbar -lfoo" in order to
1080:     # resolve all symbols.  I just hope we never have to say "-lfoo obj.o
1081:     # -lbar" to get things to work -- that's certainly a possibility, but a
1082:     # pretty nasty way to arrange your C code.
1083: 
1084:     for lib in libraries:
1085:         lib_dir, lib_name = os.path.split(lib)
1086:         if lib_dir != '':
1087:             lib_file = compiler.find_library_file([lib_dir], lib_name)
1088:             if lib_file is not None:
1089:                 lib_opts.append(lib_file)
1090:             else:
1091:                 compiler.warn("no library file corresponding to "
1092:                               "'%s' found (skipping)" % lib)
1093:         else:
1094:             lib_opts.append(compiler.library_option(lib))
1095: 
1096:     return lib_opts
1097: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_303834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', 'distutils.ccompiler\n\nContains CCompiler, an abstract base class that defines the interface\nfor the Distutils compiler abstraction model.')

# Assigning a Str to a Name (line 6):

# Assigning a Str to a Name (line 6):
str_303835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__revision__', str_303835)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import sys' statement (line 8)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import os' statement (line 9)
import os

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import re' statement (line 10)
import re

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.errors import CompileError, LinkError, UnknownFileError, DistutilsPlatformError, DistutilsModuleError' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_303836 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors')

if (type(import_303836) is not StypyTypeError):

    if (import_303836 != 'pyd_module'):
        __import__(import_303836)
        sys_modules_303837 = sys.modules[import_303836]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', sys_modules_303837.module_type_store, module_type_store, ['CompileError', 'LinkError', 'UnknownFileError', 'DistutilsPlatformError', 'DistutilsModuleError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_303837, sys_modules_303837.module_type_store, module_type_store)
    else:
        from distutils.errors import CompileError, LinkError, UnknownFileError, DistutilsPlatformError, DistutilsModuleError

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', None, module_type_store, ['CompileError', 'LinkError', 'UnknownFileError', 'DistutilsPlatformError', 'DistutilsModuleError'], [CompileError, LinkError, UnknownFileError, DistutilsPlatformError, DistutilsModuleError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', import_303836)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.spawn import spawn' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_303838 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.spawn')

if (type(import_303838) is not StypyTypeError):

    if (import_303838 != 'pyd_module'):
        __import__(import_303838)
        sys_modules_303839 = sys.modules[import_303838]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.spawn', sys_modules_303839.module_type_store, module_type_store, ['spawn'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_303839, sys_modules_303839.module_type_store, module_type_store)
    else:
        from distutils.spawn import spawn

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.spawn', None, module_type_store, ['spawn'], [spawn])

else:
    # Assigning a type to the variable 'distutils.spawn' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.spawn', import_303838)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.file_util import move_file' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_303840 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.file_util')

if (type(import_303840) is not StypyTypeError):

    if (import_303840 != 'pyd_module'):
        __import__(import_303840)
        sys_modules_303841 = sys.modules[import_303840]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.file_util', sys_modules_303841.module_type_store, module_type_store, ['move_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_303841, sys_modules_303841.module_type_store, module_type_store)
    else:
        from distutils.file_util import move_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.file_util', None, module_type_store, ['move_file'], [move_file])

else:
    # Assigning a type to the variable 'distutils.file_util' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.file_util', import_303840)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils.dir_util import mkpath' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_303842 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.dir_util')

if (type(import_303842) is not StypyTypeError):

    if (import_303842 != 'pyd_module'):
        __import__(import_303842)
        sys_modules_303843 = sys.modules[import_303842]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.dir_util', sys_modules_303843.module_type_store, module_type_store, ['mkpath'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_303843, sys_modules_303843.module_type_store, module_type_store)
    else:
        from distutils.dir_util import mkpath

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.dir_util', None, module_type_store, ['mkpath'], [mkpath])

else:
    # Assigning a type to the variable 'distutils.dir_util' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.dir_util', import_303842)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from distutils.dep_util import newer_group' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_303844 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.dep_util')

if (type(import_303844) is not StypyTypeError):

    if (import_303844 != 'pyd_module'):
        __import__(import_303844)
        sys_modules_303845 = sys.modules[import_303844]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.dep_util', sys_modules_303845.module_type_store, module_type_store, ['newer_group'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_303845, sys_modules_303845.module_type_store, module_type_store)
    else:
        from distutils.dep_util import newer_group

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.dep_util', None, module_type_store, ['newer_group'], [newer_group])

else:
    # Assigning a type to the variable 'distutils.dep_util' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.dep_util', import_303844)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from distutils.util import split_quoted, execute' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_303846 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.util')

if (type(import_303846) is not StypyTypeError):

    if (import_303846 != 'pyd_module'):
        __import__(import_303846)
        sys_modules_303847 = sys.modules[import_303846]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.util', sys_modules_303847.module_type_store, module_type_store, ['split_quoted', 'execute'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_303847, sys_modules_303847.module_type_store, module_type_store)
    else:
        from distutils.util import split_quoted, execute

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.util', None, module_type_store, ['split_quoted', 'execute'], [split_quoted, execute])

else:
    # Assigning a type to the variable 'distutils.util' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.util', import_303846)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from distutils import log' statement (line 19)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils', None, module_type_store, ['log'], [log])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from distutils.sysconfig import customize_compiler' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_303848 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.sysconfig')

if (type(import_303848) is not StypyTypeError):

    if (import_303848 != 'pyd_module'):
        __import__(import_303848)
        sys_modules_303849 = sys.modules[import_303848]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.sysconfig', sys_modules_303849.module_type_store, module_type_store, ['customize_compiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_303849, sys_modules_303849.module_type_store, module_type_store)
    else:
        from distutils.sysconfig import customize_compiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.sysconfig', None, module_type_store, ['customize_compiler'], [customize_compiler])

else:
    # Assigning a type to the variable 'distutils.sysconfig' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.sysconfig', import_303848)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

# Declaration of the 'CCompiler' class

class CCompiler:
    str_303850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'str', 'Abstract base class to define the interface that must be implemented\n    by real compiler classes.  Also has some utility methods used by\n    several compiler classes.\n\n    The basic idea behind a compiler abstraction class is that each\n    instance can be used for all the compile/link steps in building a\n    single project.  Thus, attributes common to all of those compile and\n    link steps -- include directories, macros to define, libraries to link\n    against, etc. -- are attributes of the compiler instance.  To allow for\n    variability in how individual files are treated, most of those\n    attributes may be varied on a per-compilation or per-link basis.\n    ')
    
    # Assigning a Name to a Name (line 45):
    
    # Assigning a Name to a Name (line 71):
    
    # Assigning a Name to a Name (line 72):
    
    # Assigning a Name to a Name (line 73):
    
    # Assigning a Name to a Name (line 74):
    
    # Assigning a Name to a Name (line 75):
    
    # Assigning a Name to a Name (line 76):
    
    # Assigning a Name to a Name (line 77):
    
    # Assigning a Dict to a Name (line 85):
    
    # Assigning a List to a Name (line 91):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_303851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 32), 'int')
        int_303852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 43), 'int')
        int_303853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 52), 'int')
        defaults = [int_303851, int_303852, int_303853]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.__init__', ['verbose', 'dry_run', 'force'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['verbose', 'dry_run', 'force'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 94):
        
        # Assigning a Name to a Attribute (line 94):
        # Getting the type of 'dry_run' (line 94)
        dry_run_303854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 23), 'dry_run')
        # Getting the type of 'self' (line 94)
        self_303855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self')
        # Setting the type of the member 'dry_run' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_303855, 'dry_run', dry_run_303854)
        
        # Assigning a Name to a Attribute (line 95):
        
        # Assigning a Name to a Attribute (line 95):
        # Getting the type of 'force' (line 95)
        force_303856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'force')
        # Getting the type of 'self' (line 95)
        self_303857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self')
        # Setting the type of the member 'force' of a type (line 95)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_303857, 'force', force_303856)
        
        # Assigning a Name to a Attribute (line 96):
        
        # Assigning a Name to a Attribute (line 96):
        # Getting the type of 'verbose' (line 96)
        verbose_303858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'verbose')
        # Getting the type of 'self' (line 96)
        self_303859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self')
        # Setting the type of the member 'verbose' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_303859, 'verbose', verbose_303858)
        
        # Assigning a Name to a Attribute (line 100):
        
        # Assigning a Name to a Attribute (line 100):
        # Getting the type of 'None' (line 100)
        None_303860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 26), 'None')
        # Getting the type of 'self' (line 100)
        self_303861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'self')
        # Setting the type of the member 'output_dir' of a type (line 100)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), self_303861, 'output_dir', None_303860)
        
        # Assigning a List to a Attribute (line 106):
        
        # Assigning a List to a Attribute (line 106):
        
        # Obtaining an instance of the builtin type 'list' (line 106)
        list_303862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 106)
        
        # Getting the type of 'self' (line 106)
        self_303863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'self')
        # Setting the type of the member 'macros' of a type (line 106)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), self_303863, 'macros', list_303862)
        
        # Assigning a List to a Attribute (line 109):
        
        # Assigning a List to a Attribute (line 109):
        
        # Obtaining an instance of the builtin type 'list' (line 109)
        list_303864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 109)
        
        # Getting the type of 'self' (line 109)
        self_303865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self')
        # Setting the type of the member 'include_dirs' of a type (line 109)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_303865, 'include_dirs', list_303864)
        
        # Assigning a List to a Attribute (line 113):
        
        # Assigning a List to a Attribute (line 113):
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_303866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        
        # Getting the type of 'self' (line 113)
        self_303867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self')
        # Setting the type of the member 'libraries' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_303867, 'libraries', list_303866)
        
        # Assigning a List to a Attribute (line 116):
        
        # Assigning a List to a Attribute (line 116):
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_303868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        
        # Getting the type of 'self' (line 116)
        self_303869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'self')
        # Setting the type of the member 'library_dirs' of a type (line 116)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), self_303869, 'library_dirs', list_303868)
        
        # Assigning a List to a Attribute (line 120):
        
        # Assigning a List to a Attribute (line 120):
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_303870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        
        # Getting the type of 'self' (line 120)
        self_303871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self')
        # Setting the type of the member 'runtime_library_dirs' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_303871, 'runtime_library_dirs', list_303870)
        
        # Assigning a List to a Attribute (line 124):
        
        # Assigning a List to a Attribute (line 124):
        
        # Obtaining an instance of the builtin type 'list' (line 124)
        list_303872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 124)
        
        # Getting the type of 'self' (line 124)
        self_303873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'self')
        # Setting the type of the member 'objects' of a type (line 124)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), self_303873, 'objects', list_303872)
        
        
        # Call to keys(...): (line 126)
        # Processing the call keyword arguments (line 126)
        kwargs_303877 = {}
        # Getting the type of 'self' (line 126)
        self_303874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 19), 'self', False)
        # Obtaining the member 'executables' of a type (line 126)
        executables_303875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 19), self_303874, 'executables')
        # Obtaining the member 'keys' of a type (line 126)
        keys_303876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 19), executables_303875, 'keys')
        # Calling keys(args, kwargs) (line 126)
        keys_call_result_303878 = invoke(stypy.reporting.localization.Localization(__file__, 126, 19), keys_303876, *[], **kwargs_303877)
        
        # Testing the type of a for loop iterable (line 126)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 126, 8), keys_call_result_303878)
        # Getting the type of the for loop variable (line 126)
        for_loop_var_303879 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 126, 8), keys_call_result_303878)
        # Assigning a type to the variable 'key' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'key', for_loop_var_303879)
        # SSA begins for a for statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to set_executable(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'key' (line 127)
        key_303882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 32), 'key', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 127)
        key_303883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 54), 'key', False)
        # Getting the type of 'self' (line 127)
        self_303884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 37), 'self', False)
        # Obtaining the member 'executables' of a type (line 127)
        executables_303885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 37), self_303884, 'executables')
        # Obtaining the member '__getitem__' of a type (line 127)
        getitem___303886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 37), executables_303885, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
        subscript_call_result_303887 = invoke(stypy.reporting.localization.Localization(__file__, 127, 37), getitem___303886, key_303883)
        
        # Processing the call keyword arguments (line 127)
        kwargs_303888 = {}
        # Getting the type of 'self' (line 127)
        self_303880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'self', False)
        # Obtaining the member 'set_executable' of a type (line 127)
        set_executable_303881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), self_303880, 'set_executable')
        # Calling set_executable(args, kwargs) (line 127)
        set_executable_call_result_303889 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), set_executable_303881, *[key_303882, subscript_call_result_303887], **kwargs_303888)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_executables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_executables'
        module_type_store = module_type_store.open_function_context('set_executables', 129, 4, False)
        # Assigning a type to the variable 'self' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.set_executables.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.set_executables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.set_executables.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.set_executables.__dict__.__setitem__('stypy_function_name', 'CCompiler.set_executables')
        CCompiler.set_executables.__dict__.__setitem__('stypy_param_names_list', [])
        CCompiler.set_executables.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.set_executables.__dict__.__setitem__('stypy_kwargs_param_name', 'args')
        CCompiler.set_executables.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.set_executables.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.set_executables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.set_executables.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.set_executables', [], None, 'args', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_executables', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_executables(...)' code ##################

        str_303890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, (-1)), 'str', "Define the executables (and options for them) that will be run\n        to perform the various stages of compilation.  The exact set of\n        executables that may be specified here depends on the compiler\n        class (via the 'executables' class attribute), but most will have:\n          compiler      the C/C++ compiler\n          linker_so     linker used to create shared objects and libraries\n          linker_exe    linker used to create binary executables\n          archiver      static library creator\n\n        On platforms with a command-line (Unix, DOS/Windows), each of these\n        is a string that will be split into executable name and (optional)\n        list of arguments.  (Splitting the string is done similarly to how\n        Unix shells operate: words are delimited by spaces, but quotes and\n        backslashes can override this.  See\n        'distutils.util.split_quoted()'.)\n        ")
        
        
        # Call to keys(...): (line 155)
        # Processing the call keyword arguments (line 155)
        kwargs_303893 = {}
        # Getting the type of 'args' (line 155)
        args_303891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 19), 'args', False)
        # Obtaining the member 'keys' of a type (line 155)
        keys_303892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 19), args_303891, 'keys')
        # Calling keys(args, kwargs) (line 155)
        keys_call_result_303894 = invoke(stypy.reporting.localization.Localization(__file__, 155, 19), keys_303892, *[], **kwargs_303893)
        
        # Testing the type of a for loop iterable (line 155)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 155, 8), keys_call_result_303894)
        # Getting the type of the for loop variable (line 155)
        for_loop_var_303895 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 155, 8), keys_call_result_303894)
        # Assigning a type to the variable 'key' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'key', for_loop_var_303895)
        # SSA begins for a for statement (line 155)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'key' (line 156)
        key_303896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'key')
        # Getting the type of 'self' (line 156)
        self_303897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 26), 'self')
        # Obtaining the member 'executables' of a type (line 156)
        executables_303898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 26), self_303897, 'executables')
        # Applying the binary operator 'notin' (line 156)
        result_contains_303899 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 15), 'notin', key_303896, executables_303898)
        
        # Testing the type of an if condition (line 156)
        if_condition_303900 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 12), result_contains_303899)
        # Assigning a type to the variable 'if_condition_303900' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'if_condition_303900', if_condition_303900)
        # SSA begins for if statement (line 156)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'ValueError' (line 157)
        ValueError_303901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 22), 'ValueError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 157, 16), ValueError_303901, 'raise parameter', BaseException)
        # SSA join for if statement (line 156)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_executable(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'key' (line 160)
        key_303904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 32), 'key', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 160)
        key_303905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 42), 'key', False)
        # Getting the type of 'args' (line 160)
        args_303906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 37), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___303907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 37), args_303906, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_303908 = invoke(stypy.reporting.localization.Localization(__file__, 160, 37), getitem___303907, key_303905)
        
        # Processing the call keyword arguments (line 160)
        kwargs_303909 = {}
        # Getting the type of 'self' (line 160)
        self_303902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'self', False)
        # Obtaining the member 'set_executable' of a type (line 160)
        set_executable_303903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), self_303902, 'set_executable')
        # Calling set_executable(args, kwargs) (line 160)
        set_executable_call_result_303910 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), set_executable_303903, *[key_303904, subscript_call_result_303908], **kwargs_303909)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_executables(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_executables' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_303911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_303911)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_executables'
        return stypy_return_type_303911


    @norecursion
    def set_executable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_executable'
        module_type_store = module_type_store.open_function_context('set_executable', 162, 4, False)
        # Assigning a type to the variable 'self' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.set_executable.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.set_executable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.set_executable.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.set_executable.__dict__.__setitem__('stypy_function_name', 'CCompiler.set_executable')
        CCompiler.set_executable.__dict__.__setitem__('stypy_param_names_list', ['key', 'value'])
        CCompiler.set_executable.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.set_executable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.set_executable.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.set_executable.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.set_executable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.set_executable.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.set_executable', ['key', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_executable', localization, ['key', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_executable(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 163)
        # Getting the type of 'str' (line 163)
        str_303912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'str')
        # Getting the type of 'value' (line 163)
        value_303913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 'value')
        
        (may_be_303914, more_types_in_union_303915) = may_be_subtype(str_303912, value_303913)

        if may_be_303914:

            if more_types_in_union_303915:
                # Runtime conditional SSA (line 163)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'value' (line 163)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'value', remove_not_subtype_from_union(value_303913, str))
            
            # Call to setattr(...): (line 164)
            # Processing the call arguments (line 164)
            # Getting the type of 'self' (line 164)
            self_303917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'self', False)
            # Getting the type of 'key' (line 164)
            key_303918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 26), 'key', False)
            
            # Call to split_quoted(...): (line 164)
            # Processing the call arguments (line 164)
            # Getting the type of 'value' (line 164)
            value_303920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 44), 'value', False)
            # Processing the call keyword arguments (line 164)
            kwargs_303921 = {}
            # Getting the type of 'split_quoted' (line 164)
            split_quoted_303919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 31), 'split_quoted', False)
            # Calling split_quoted(args, kwargs) (line 164)
            split_quoted_call_result_303922 = invoke(stypy.reporting.localization.Localization(__file__, 164, 31), split_quoted_303919, *[value_303920], **kwargs_303921)
            
            # Processing the call keyword arguments (line 164)
            kwargs_303923 = {}
            # Getting the type of 'setattr' (line 164)
            setattr_303916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'setattr', False)
            # Calling setattr(args, kwargs) (line 164)
            setattr_call_result_303924 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), setattr_303916, *[self_303917, key_303918, split_quoted_call_result_303922], **kwargs_303923)
            

            if more_types_in_union_303915:
                # Runtime conditional SSA for else branch (line 163)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_303914) or more_types_in_union_303915):
            # Assigning a type to the variable 'value' (line 163)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'value', remove_subtype_from_union(value_303913, str))
            
            # Call to setattr(...): (line 166)
            # Processing the call arguments (line 166)
            # Getting the type of 'self' (line 166)
            self_303926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'self', False)
            # Getting the type of 'key' (line 166)
            key_303927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 26), 'key', False)
            # Getting the type of 'value' (line 166)
            value_303928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 31), 'value', False)
            # Processing the call keyword arguments (line 166)
            kwargs_303929 = {}
            # Getting the type of 'setattr' (line 166)
            setattr_303925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'setattr', False)
            # Calling setattr(args, kwargs) (line 166)
            setattr_call_result_303930 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), setattr_303925, *[self_303926, key_303927, value_303928], **kwargs_303929)
            

            if (may_be_303914 and more_types_in_union_303915):
                # SSA join for if statement (line 163)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'set_executable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_executable' in the type store
        # Getting the type of 'stypy_return_type' (line 162)
        stypy_return_type_303931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_303931)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_executable'
        return stypy_return_type_303931


    @norecursion
    def _find_macro(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_find_macro'
        module_type_store = module_type_store.open_function_context('_find_macro', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler._find_macro.__dict__.__setitem__('stypy_localization', localization)
        CCompiler._find_macro.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler._find_macro.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler._find_macro.__dict__.__setitem__('stypy_function_name', 'CCompiler._find_macro')
        CCompiler._find_macro.__dict__.__setitem__('stypy_param_names_list', ['name'])
        CCompiler._find_macro.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler._find_macro.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler._find_macro.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler._find_macro.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler._find_macro.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler._find_macro.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler._find_macro', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_find_macro', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_find_macro(...)' code ##################

        
        # Assigning a Num to a Name (line 169):
        
        # Assigning a Num to a Name (line 169):
        int_303932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 12), 'int')
        # Assigning a type to the variable 'i' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'i', int_303932)
        
        # Getting the type of 'self' (line 170)
        self_303933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 20), 'self')
        # Obtaining the member 'macros' of a type (line 170)
        macros_303934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 20), self_303933, 'macros')
        # Testing the type of a for loop iterable (line 170)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 170, 8), macros_303934)
        # Getting the type of the for loop variable (line 170)
        for_loop_var_303935 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 170, 8), macros_303934)
        # Assigning a type to the variable 'defn' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'defn', for_loop_var_303935)
        # SSA begins for a for statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Obtaining the type of the subscript
        int_303936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 20), 'int')
        # Getting the type of 'defn' (line 171)
        defn_303937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'defn')
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___303938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 15), defn_303937, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_303939 = invoke(stypy.reporting.localization.Localization(__file__, 171, 15), getitem___303938, int_303936)
        
        # Getting the type of 'name' (line 171)
        name_303940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'name')
        # Applying the binary operator '==' (line 171)
        result_eq_303941 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 15), '==', subscript_call_result_303939, name_303940)
        
        # Testing the type of an if condition (line 171)
        if_condition_303942 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 12), result_eq_303941)
        # Assigning a type to the variable 'if_condition_303942' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'if_condition_303942', if_condition_303942)
        # SSA begins for if statement (line 171)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'i' (line 172)
        i_303943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 23), 'i')
        # Assigning a type to the variable 'stypy_return_type' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'stypy_return_type', i_303943)
        # SSA join for if statement (line 171)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 173):
        
        # Assigning a BinOp to a Name (line 173):
        # Getting the type of 'i' (line 173)
        i_303944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 16), 'i')
        int_303945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 20), 'int')
        # Applying the binary operator '+' (line 173)
        result_add_303946 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 16), '+', i_303944, int_303945)
        
        # Assigning a type to the variable 'i' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'i', result_add_303946)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'None' (line 174)
        None_303947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'stypy_return_type', None_303947)
        
        # ################# End of '_find_macro(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_find_macro' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_303948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_303948)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_find_macro'
        return stypy_return_type_303948


    @norecursion
    def _check_macro_definitions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_macro_definitions'
        module_type_store = module_type_store.open_function_context('_check_macro_definitions', 176, 4, False)
        # Assigning a type to the variable 'self' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler._check_macro_definitions.__dict__.__setitem__('stypy_localization', localization)
        CCompiler._check_macro_definitions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler._check_macro_definitions.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler._check_macro_definitions.__dict__.__setitem__('stypy_function_name', 'CCompiler._check_macro_definitions')
        CCompiler._check_macro_definitions.__dict__.__setitem__('stypy_param_names_list', ['definitions'])
        CCompiler._check_macro_definitions.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler._check_macro_definitions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler._check_macro_definitions.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler._check_macro_definitions.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler._check_macro_definitions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler._check_macro_definitions.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler._check_macro_definitions', ['definitions'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_macro_definitions', localization, ['definitions'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_macro_definitions(...)' code ##################

        str_303949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, (-1)), 'str', "Ensures that every element of 'definitions' is a valid macro\n        definition, ie. either (name,value) 2-tuple or a (name,) tuple.  Do\n        nothing if all definitions are OK, raise TypeError otherwise.\n        ")
        
        # Getting the type of 'definitions' (line 181)
        definitions_303950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 20), 'definitions')
        # Testing the type of a for loop iterable (line 181)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 181, 8), definitions_303950)
        # Getting the type of the for loop variable (line 181)
        for_loop_var_303951 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 181, 8), definitions_303950)
        # Assigning a type to the variable 'defn' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'defn', for_loop_var_303951)
        # SSA begins for a for statement (line 181)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'defn' (line 182)
        defn_303953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 31), 'defn', False)
        # Getting the type of 'tuple' (line 182)
        tuple_303954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 37), 'tuple', False)
        # Processing the call keyword arguments (line 182)
        kwargs_303955 = {}
        # Getting the type of 'isinstance' (line 182)
        isinstance_303952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 182)
        isinstance_call_result_303956 = invoke(stypy.reporting.localization.Localization(__file__, 182, 20), isinstance_303952, *[defn_303953, tuple_303954], **kwargs_303955)
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'defn' (line 183)
        defn_303958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 26), 'defn', False)
        # Processing the call keyword arguments (line 183)
        kwargs_303959 = {}
        # Getting the type of 'len' (line 183)
        len_303957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 21), 'len', False)
        # Calling len(args, kwargs) (line 183)
        len_call_result_303960 = invoke(stypy.reporting.localization.Localization(__file__, 183, 21), len_303957, *[defn_303958], **kwargs_303959)
        
        int_303961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 35), 'int')
        # Applying the binary operator '==' (line 183)
        result_eq_303962 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 21), '==', len_call_result_303960, int_303961)
        
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'defn' (line 184)
        defn_303964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 27), 'defn', False)
        # Processing the call keyword arguments (line 184)
        kwargs_303965 = {}
        # Getting the type of 'len' (line 184)
        len_303963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 22), 'len', False)
        # Calling len(args, kwargs) (line 184)
        len_call_result_303966 = invoke(stypy.reporting.localization.Localization(__file__, 184, 22), len_303963, *[defn_303964], **kwargs_303965)
        
        int_303967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 36), 'int')
        # Applying the binary operator '==' (line 184)
        result_eq_303968 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 22), '==', len_call_result_303966, int_303967)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 185)
        # Processing the call arguments (line 185)
        
        # Obtaining the type of the subscript
        int_303970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 39), 'int')
        # Getting the type of 'defn' (line 185)
        defn_303971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 34), 'defn', False)
        # Obtaining the member '__getitem__' of a type (line 185)
        getitem___303972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 34), defn_303971, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 185)
        subscript_call_result_303973 = invoke(stypy.reporting.localization.Localization(__file__, 185, 34), getitem___303972, int_303970)
        
        # Getting the type of 'str' (line 185)
        str_303974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 43), 'str', False)
        # Processing the call keyword arguments (line 185)
        kwargs_303975 = {}
        # Getting the type of 'isinstance' (line 185)
        isinstance_303969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 185)
        isinstance_call_result_303976 = invoke(stypy.reporting.localization.Localization(__file__, 185, 23), isinstance_303969, *[subscript_call_result_303973, str_303974], **kwargs_303975)
        
        
        
        # Obtaining the type of the subscript
        int_303977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 56), 'int')
        # Getting the type of 'defn' (line 185)
        defn_303978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 51), 'defn')
        # Obtaining the member '__getitem__' of a type (line 185)
        getitem___303979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 51), defn_303978, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 185)
        subscript_call_result_303980 = invoke(stypy.reporting.localization.Localization(__file__, 185, 51), getitem___303979, int_303977)
        
        # Getting the type of 'None' (line 185)
        None_303981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 62), 'None')
        # Applying the binary operator 'is' (line 185)
        result_is__303982 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 51), 'is', subscript_call_result_303980, None_303981)
        
        # Applying the binary operator 'or' (line 185)
        result_or_keyword_303983 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 23), 'or', isinstance_call_result_303976, result_is__303982)
        
        # Applying the binary operator 'and' (line 184)
        result_and_keyword_303984 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 22), 'and', result_eq_303968, result_or_keyword_303983)
        
        # Applying the binary operator 'or' (line 183)
        result_or_keyword_303985 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 21), 'or', result_eq_303962, result_and_keyword_303984)
        
        # Applying the binary operator 'and' (line 182)
        result_and_keyword_303986 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 20), 'and', isinstance_call_result_303956, result_or_keyword_303985)
        
        # Call to isinstance(...): (line 186)
        # Processing the call arguments (line 186)
        
        # Obtaining the type of the subscript
        int_303988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 36), 'int')
        # Getting the type of 'defn' (line 186)
        defn_303989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 31), 'defn', False)
        # Obtaining the member '__getitem__' of a type (line 186)
        getitem___303990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 31), defn_303989, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 186)
        subscript_call_result_303991 = invoke(stypy.reporting.localization.Localization(__file__, 186, 31), getitem___303990, int_303988)
        
        # Getting the type of 'str' (line 186)
        str_303992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 40), 'str', False)
        # Processing the call keyword arguments (line 186)
        kwargs_303993 = {}
        # Getting the type of 'isinstance' (line 186)
        isinstance_303987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 20), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 186)
        isinstance_call_result_303994 = invoke(stypy.reporting.localization.Localization(__file__, 186, 20), isinstance_303987, *[subscript_call_result_303991, str_303992], **kwargs_303993)
        
        # Applying the binary operator 'and' (line 182)
        result_and_keyword_303995 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 20), 'and', result_and_keyword_303986, isinstance_call_result_303994)
        
        # Applying the 'not' unary operator (line 182)
        result_not__303996 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 15), 'not', result_and_keyword_303995)
        
        # Testing the type of an if condition (line 182)
        if_condition_303997 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 12), result_not__303996)
        # Assigning a type to the variable 'if_condition_303997' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'if_condition_303997', if_condition_303997)
        # SSA begins for if statement (line 182)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'TypeError' (line 187)
        TypeError_303998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 22), 'TypeError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 187, 16), TypeError_303998, 'raise parameter', BaseException)
        # SSA join for if statement (line 182)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_check_macro_definitions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_macro_definitions' in the type store
        # Getting the type of 'stypy_return_type' (line 176)
        stypy_return_type_303999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_303999)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_macro_definitions'
        return stypy_return_type_303999


    @norecursion
    def define_macro(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 195)
        None_304000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 39), 'None')
        defaults = [None_304000]
        # Create a new context for function 'define_macro'
        module_type_store = module_type_store.open_function_context('define_macro', 195, 4, False)
        # Assigning a type to the variable 'self' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.define_macro.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.define_macro.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.define_macro.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.define_macro.__dict__.__setitem__('stypy_function_name', 'CCompiler.define_macro')
        CCompiler.define_macro.__dict__.__setitem__('stypy_param_names_list', ['name', 'value'])
        CCompiler.define_macro.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.define_macro.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.define_macro.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.define_macro.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.define_macro.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.define_macro.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.define_macro', ['name', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'define_macro', localization, ['name', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'define_macro(...)' code ##################

        str_304001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, (-1)), 'str', "Define a preprocessor macro for all compilations driven by this\n        compiler object.  The optional parameter 'value' should be a\n        string; if it is not supplied, then the macro will be defined\n        without an explicit value and the exact outcome depends on the\n        compiler used (XXX true? does ANSI say anything about this?)\n        ")
        
        # Assigning a Call to a Name (line 204):
        
        # Assigning a Call to a Name (line 204):
        
        # Call to _find_macro(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'name' (line 204)
        name_304004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 30), 'name', False)
        # Processing the call keyword arguments (line 204)
        kwargs_304005 = {}
        # Getting the type of 'self' (line 204)
        self_304002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'self', False)
        # Obtaining the member '_find_macro' of a type (line 204)
        _find_macro_304003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), self_304002, '_find_macro')
        # Calling _find_macro(args, kwargs) (line 204)
        _find_macro_call_result_304006 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), _find_macro_304003, *[name_304004], **kwargs_304005)
        
        # Assigning a type to the variable 'i' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'i', _find_macro_call_result_304006)
        
        # Type idiom detected: calculating its left and rigth part (line 205)
        # Getting the type of 'i' (line 205)
        i_304007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'i')
        # Getting the type of 'None' (line 205)
        None_304008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'None')
        
        (may_be_304009, more_types_in_union_304010) = may_not_be_none(i_304007, None_304008)

        if may_be_304009:

            if more_types_in_union_304010:
                # Runtime conditional SSA (line 205)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Deleting a member
            # Getting the type of 'self' (line 206)
            self_304011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'self')
            # Obtaining the member 'macros' of a type (line 206)
            macros_304012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 16), self_304011, 'macros')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 206)
            i_304013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 28), 'i')
            # Getting the type of 'self' (line 206)
            self_304014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'self')
            # Obtaining the member 'macros' of a type (line 206)
            macros_304015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 16), self_304014, 'macros')
            # Obtaining the member '__getitem__' of a type (line 206)
            getitem___304016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 16), macros_304015, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 206)
            subscript_call_result_304017 = invoke(stypy.reporting.localization.Localization(__file__, 206, 16), getitem___304016, i_304013)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 12), macros_304012, subscript_call_result_304017)

            if more_types_in_union_304010:
                # SSA join for if statement (line 205)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Tuple to a Name (line 208):
        
        # Assigning a Tuple to a Name (line 208):
        
        # Obtaining an instance of the builtin type 'tuple' (line 208)
        tuple_304018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 208)
        # Adding element type (line 208)
        # Getting the type of 'name' (line 208)
        name_304019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 16), tuple_304018, name_304019)
        # Adding element type (line 208)
        # Getting the type of 'value' (line 208)
        value_304020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 22), 'value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 16), tuple_304018, value_304020)
        
        # Assigning a type to the variable 'defn' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'defn', tuple_304018)
        
        # Call to append(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'defn' (line 209)
        defn_304024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 28), 'defn', False)
        # Processing the call keyword arguments (line 209)
        kwargs_304025 = {}
        # Getting the type of 'self' (line 209)
        self_304021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'self', False)
        # Obtaining the member 'macros' of a type (line 209)
        macros_304022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), self_304021, 'macros')
        # Obtaining the member 'append' of a type (line 209)
        append_304023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), macros_304022, 'append')
        # Calling append(args, kwargs) (line 209)
        append_call_result_304026 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), append_304023, *[defn_304024], **kwargs_304025)
        
        
        # ################# End of 'define_macro(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'define_macro' in the type store
        # Getting the type of 'stypy_return_type' (line 195)
        stypy_return_type_304027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304027)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'define_macro'
        return stypy_return_type_304027


    @norecursion
    def undefine_macro(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'undefine_macro'
        module_type_store = module_type_store.open_function_context('undefine_macro', 211, 4, False)
        # Assigning a type to the variable 'self' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.undefine_macro.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.undefine_macro.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.undefine_macro.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.undefine_macro.__dict__.__setitem__('stypy_function_name', 'CCompiler.undefine_macro')
        CCompiler.undefine_macro.__dict__.__setitem__('stypy_param_names_list', ['name'])
        CCompiler.undefine_macro.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.undefine_macro.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.undefine_macro.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.undefine_macro.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.undefine_macro.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.undefine_macro.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.undefine_macro', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'undefine_macro', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'undefine_macro(...)' code ##################

        str_304028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, (-1)), 'str', "Undefine a preprocessor macro for all compilations driven by\n        this compiler object.  If the same macro is defined by\n        'define_macro()' and undefined by 'undefine_macro()' the last call\n        takes precedence (including multiple redefinitions or\n        undefinitions).  If the macro is redefined/undefined on a\n        per-compilation basis (ie. in the call to 'compile()'), then that\n        takes precedence.\n        ")
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to _find_macro(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'name' (line 222)
        name_304031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 30), 'name', False)
        # Processing the call keyword arguments (line 222)
        kwargs_304032 = {}
        # Getting the type of 'self' (line 222)
        self_304029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'self', False)
        # Obtaining the member '_find_macro' of a type (line 222)
        _find_macro_304030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 12), self_304029, '_find_macro')
        # Calling _find_macro(args, kwargs) (line 222)
        _find_macro_call_result_304033 = invoke(stypy.reporting.localization.Localization(__file__, 222, 12), _find_macro_304030, *[name_304031], **kwargs_304032)
        
        # Assigning a type to the variable 'i' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'i', _find_macro_call_result_304033)
        
        # Type idiom detected: calculating its left and rigth part (line 223)
        # Getting the type of 'i' (line 223)
        i_304034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'i')
        # Getting the type of 'None' (line 223)
        None_304035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), 'None')
        
        (may_be_304036, more_types_in_union_304037) = may_not_be_none(i_304034, None_304035)

        if may_be_304036:

            if more_types_in_union_304037:
                # Runtime conditional SSA (line 223)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Deleting a member
            # Getting the type of 'self' (line 224)
            self_304038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'self')
            # Obtaining the member 'macros' of a type (line 224)
            macros_304039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 16), self_304038, 'macros')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 224)
            i_304040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 28), 'i')
            # Getting the type of 'self' (line 224)
            self_304041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'self')
            # Obtaining the member 'macros' of a type (line 224)
            macros_304042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 16), self_304041, 'macros')
            # Obtaining the member '__getitem__' of a type (line 224)
            getitem___304043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 16), macros_304042, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 224)
            subscript_call_result_304044 = invoke(stypy.reporting.localization.Localization(__file__, 224, 16), getitem___304043, i_304040)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 12), macros_304039, subscript_call_result_304044)

            if more_types_in_union_304037:
                # SSA join for if statement (line 223)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Tuple to a Name (line 226):
        
        # Assigning a Tuple to a Name (line 226):
        
        # Obtaining an instance of the builtin type 'tuple' (line 226)
        tuple_304045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 226)
        # Adding element type (line 226)
        # Getting the type of 'name' (line 226)
        name_304046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 18), 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 18), tuple_304045, name_304046)
        
        # Assigning a type to the variable 'undefn' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'undefn', tuple_304045)
        
        # Call to append(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'undefn' (line 227)
        undefn_304050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 28), 'undefn', False)
        # Processing the call keyword arguments (line 227)
        kwargs_304051 = {}
        # Getting the type of 'self' (line 227)
        self_304047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'self', False)
        # Obtaining the member 'macros' of a type (line 227)
        macros_304048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), self_304047, 'macros')
        # Obtaining the member 'append' of a type (line 227)
        append_304049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), macros_304048, 'append')
        # Calling append(args, kwargs) (line 227)
        append_call_result_304052 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), append_304049, *[undefn_304050], **kwargs_304051)
        
        
        # ################# End of 'undefine_macro(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'undefine_macro' in the type store
        # Getting the type of 'stypy_return_type' (line 211)
        stypy_return_type_304053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304053)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'undefine_macro'
        return stypy_return_type_304053


    @norecursion
    def add_include_dir(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_include_dir'
        module_type_store = module_type_store.open_function_context('add_include_dir', 229, 4, False)
        # Assigning a type to the variable 'self' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.add_include_dir.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.add_include_dir.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.add_include_dir.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.add_include_dir.__dict__.__setitem__('stypy_function_name', 'CCompiler.add_include_dir')
        CCompiler.add_include_dir.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        CCompiler.add_include_dir.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.add_include_dir.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.add_include_dir.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.add_include_dir.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.add_include_dir.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.add_include_dir.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.add_include_dir', ['dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_include_dir', localization, ['dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_include_dir(...)' code ##################

        str_304054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, (-1)), 'str', "Add 'dir' to the list of directories that will be searched for\n        header files.  The compiler is instructed to search directories in\n        the order in which they are supplied by successive calls to\n        'add_include_dir()'.\n        ")
        
        # Call to append(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'dir' (line 235)
        dir_304058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 34), 'dir', False)
        # Processing the call keyword arguments (line 235)
        kwargs_304059 = {}
        # Getting the type of 'self' (line 235)
        self_304055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'self', False)
        # Obtaining the member 'include_dirs' of a type (line 235)
        include_dirs_304056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), self_304055, 'include_dirs')
        # Obtaining the member 'append' of a type (line 235)
        append_304057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), include_dirs_304056, 'append')
        # Calling append(args, kwargs) (line 235)
        append_call_result_304060 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), append_304057, *[dir_304058], **kwargs_304059)
        
        
        # ################# End of 'add_include_dir(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_include_dir' in the type store
        # Getting the type of 'stypy_return_type' (line 229)
        stypy_return_type_304061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304061)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_include_dir'
        return stypy_return_type_304061


    @norecursion
    def set_include_dirs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_include_dirs'
        module_type_store = module_type_store.open_function_context('set_include_dirs', 237, 4, False)
        # Assigning a type to the variable 'self' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.set_include_dirs.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.set_include_dirs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.set_include_dirs.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.set_include_dirs.__dict__.__setitem__('stypy_function_name', 'CCompiler.set_include_dirs')
        CCompiler.set_include_dirs.__dict__.__setitem__('stypy_param_names_list', ['dirs'])
        CCompiler.set_include_dirs.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.set_include_dirs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.set_include_dirs.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.set_include_dirs.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.set_include_dirs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.set_include_dirs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.set_include_dirs', ['dirs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_include_dirs', localization, ['dirs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_include_dirs(...)' code ##################

        str_304062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, (-1)), 'str', "Set the list of directories that will be searched to 'dirs' (a\n        list of strings).  Overrides any preceding calls to\n        'add_include_dir()'; subsequence calls to 'add_include_dir()' add\n        to the list passed to 'set_include_dirs()'.  This does not affect\n        any list of standard include directories that the compiler may\n        search by default.\n        ")
        
        # Assigning a Subscript to a Attribute (line 245):
        
        # Assigning a Subscript to a Attribute (line 245):
        
        # Obtaining the type of the subscript
        slice_304063 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 245, 28), None, None, None)
        # Getting the type of 'dirs' (line 245)
        dirs_304064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 28), 'dirs')
        # Obtaining the member '__getitem__' of a type (line 245)
        getitem___304065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 28), dirs_304064, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 245)
        subscript_call_result_304066 = invoke(stypy.reporting.localization.Localization(__file__, 245, 28), getitem___304065, slice_304063)
        
        # Getting the type of 'self' (line 245)
        self_304067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'self')
        # Setting the type of the member 'include_dirs' of a type (line 245)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), self_304067, 'include_dirs', subscript_call_result_304066)
        
        # ################# End of 'set_include_dirs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_include_dirs' in the type store
        # Getting the type of 'stypy_return_type' (line 237)
        stypy_return_type_304068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304068)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_include_dirs'
        return stypy_return_type_304068


    @norecursion
    def add_library(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_library'
        module_type_store = module_type_store.open_function_context('add_library', 247, 4, False)
        # Assigning a type to the variable 'self' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.add_library.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.add_library.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.add_library.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.add_library.__dict__.__setitem__('stypy_function_name', 'CCompiler.add_library')
        CCompiler.add_library.__dict__.__setitem__('stypy_param_names_list', ['libname'])
        CCompiler.add_library.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.add_library.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.add_library.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.add_library.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.add_library.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.add_library.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.add_library', ['libname'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_library', localization, ['libname'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_library(...)' code ##################

        str_304069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, (-1)), 'str', "Add 'libname' to the list of libraries that will be included in\n        all links driven by this compiler object.  Note that 'libname'\n        should *not* be the name of a file containing a library, but the\n        name of the library itself: the actual filename will be inferred by\n        the linker, the compiler, or the compiler class (depending on the\n        platform).\n\n        The linker will be instructed to link against libraries in the\n        order they were supplied to 'add_library()' and/or\n        'set_libraries()'.  It is perfectly valid to duplicate library\n        names; the linker will be instructed to link against libraries as\n        many times as they are mentioned.\n        ")
        
        # Call to append(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'libname' (line 261)
        libname_304073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 31), 'libname', False)
        # Processing the call keyword arguments (line 261)
        kwargs_304074 = {}
        # Getting the type of 'self' (line 261)
        self_304070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'self', False)
        # Obtaining the member 'libraries' of a type (line 261)
        libraries_304071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), self_304070, 'libraries')
        # Obtaining the member 'append' of a type (line 261)
        append_304072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), libraries_304071, 'append')
        # Calling append(args, kwargs) (line 261)
        append_call_result_304075 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), append_304072, *[libname_304073], **kwargs_304074)
        
        
        # ################# End of 'add_library(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_library' in the type store
        # Getting the type of 'stypy_return_type' (line 247)
        stypy_return_type_304076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304076)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_library'
        return stypy_return_type_304076


    @norecursion
    def set_libraries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_libraries'
        module_type_store = module_type_store.open_function_context('set_libraries', 263, 4, False)
        # Assigning a type to the variable 'self' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.set_libraries.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.set_libraries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.set_libraries.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.set_libraries.__dict__.__setitem__('stypy_function_name', 'CCompiler.set_libraries')
        CCompiler.set_libraries.__dict__.__setitem__('stypy_param_names_list', ['libnames'])
        CCompiler.set_libraries.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.set_libraries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.set_libraries.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.set_libraries.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.set_libraries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.set_libraries.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.set_libraries', ['libnames'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_libraries', localization, ['libnames'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_libraries(...)' code ##################

        str_304077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, (-1)), 'str', "Set the list of libraries to be included in all links driven by\n        this compiler object to 'libnames' (a list of strings).  This does\n        not affect any standard system libraries that the linker may\n        include by default.\n        ")
        
        # Assigning a Subscript to a Attribute (line 269):
        
        # Assigning a Subscript to a Attribute (line 269):
        
        # Obtaining the type of the subscript
        slice_304078 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 269, 25), None, None, None)
        # Getting the type of 'libnames' (line 269)
        libnames_304079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 25), 'libnames')
        # Obtaining the member '__getitem__' of a type (line 269)
        getitem___304080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 25), libnames_304079, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 269)
        subscript_call_result_304081 = invoke(stypy.reporting.localization.Localization(__file__, 269, 25), getitem___304080, slice_304078)
        
        # Getting the type of 'self' (line 269)
        self_304082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'self')
        # Setting the type of the member 'libraries' of a type (line 269)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 8), self_304082, 'libraries', subscript_call_result_304081)
        
        # ################# End of 'set_libraries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_libraries' in the type store
        # Getting the type of 'stypy_return_type' (line 263)
        stypy_return_type_304083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304083)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_libraries'
        return stypy_return_type_304083


    @norecursion
    def add_library_dir(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_library_dir'
        module_type_store = module_type_store.open_function_context('add_library_dir', 272, 4, False)
        # Assigning a type to the variable 'self' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.add_library_dir.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.add_library_dir.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.add_library_dir.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.add_library_dir.__dict__.__setitem__('stypy_function_name', 'CCompiler.add_library_dir')
        CCompiler.add_library_dir.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        CCompiler.add_library_dir.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.add_library_dir.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.add_library_dir.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.add_library_dir.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.add_library_dir.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.add_library_dir.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.add_library_dir', ['dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_library_dir', localization, ['dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_library_dir(...)' code ##################

        str_304084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, (-1)), 'str', "Add 'dir' to the list of directories that will be searched for\n        libraries specified to 'add_library()' and 'set_libraries()'.  The\n        linker will be instructed to search for libraries in the order they\n        are supplied to 'add_library_dir()' and/or 'set_library_dirs()'.\n        ")
        
        # Call to append(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'dir' (line 278)
        dir_304088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 33), 'dir', False)
        # Processing the call keyword arguments (line 278)
        kwargs_304089 = {}
        # Getting the type of 'self' (line 278)
        self_304085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 278)
        library_dirs_304086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), self_304085, 'library_dirs')
        # Obtaining the member 'append' of a type (line 278)
        append_304087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), library_dirs_304086, 'append')
        # Calling append(args, kwargs) (line 278)
        append_call_result_304090 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), append_304087, *[dir_304088], **kwargs_304089)
        
        
        # ################# End of 'add_library_dir(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_library_dir' in the type store
        # Getting the type of 'stypy_return_type' (line 272)
        stypy_return_type_304091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304091)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_library_dir'
        return stypy_return_type_304091


    @norecursion
    def set_library_dirs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_library_dirs'
        module_type_store = module_type_store.open_function_context('set_library_dirs', 280, 4, False)
        # Assigning a type to the variable 'self' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.set_library_dirs.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.set_library_dirs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.set_library_dirs.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.set_library_dirs.__dict__.__setitem__('stypy_function_name', 'CCompiler.set_library_dirs')
        CCompiler.set_library_dirs.__dict__.__setitem__('stypy_param_names_list', ['dirs'])
        CCompiler.set_library_dirs.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.set_library_dirs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.set_library_dirs.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.set_library_dirs.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.set_library_dirs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.set_library_dirs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.set_library_dirs', ['dirs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_library_dirs', localization, ['dirs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_library_dirs(...)' code ##################

        str_304092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, (-1)), 'str', "Set the list of library search directories to 'dirs' (a list of\n        strings).  This does not affect any standard library search path\n        that the linker may search by default.\n        ")
        
        # Assigning a Subscript to a Attribute (line 285):
        
        # Assigning a Subscript to a Attribute (line 285):
        
        # Obtaining the type of the subscript
        slice_304093 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 285, 28), None, None, None)
        # Getting the type of 'dirs' (line 285)
        dirs_304094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 28), 'dirs')
        # Obtaining the member '__getitem__' of a type (line 285)
        getitem___304095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 28), dirs_304094, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 285)
        subscript_call_result_304096 = invoke(stypy.reporting.localization.Localization(__file__, 285, 28), getitem___304095, slice_304093)
        
        # Getting the type of 'self' (line 285)
        self_304097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'self')
        # Setting the type of the member 'library_dirs' of a type (line 285)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), self_304097, 'library_dirs', subscript_call_result_304096)
        
        # ################# End of 'set_library_dirs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_library_dirs' in the type store
        # Getting the type of 'stypy_return_type' (line 280)
        stypy_return_type_304098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304098)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_library_dirs'
        return stypy_return_type_304098


    @norecursion
    def add_runtime_library_dir(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_runtime_library_dir'
        module_type_store = module_type_store.open_function_context('add_runtime_library_dir', 287, 4, False)
        # Assigning a type to the variable 'self' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.add_runtime_library_dir.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.add_runtime_library_dir.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.add_runtime_library_dir.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.add_runtime_library_dir.__dict__.__setitem__('stypy_function_name', 'CCompiler.add_runtime_library_dir')
        CCompiler.add_runtime_library_dir.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        CCompiler.add_runtime_library_dir.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.add_runtime_library_dir.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.add_runtime_library_dir.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.add_runtime_library_dir.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.add_runtime_library_dir.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.add_runtime_library_dir.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.add_runtime_library_dir', ['dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_runtime_library_dir', localization, ['dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_runtime_library_dir(...)' code ##################

        str_304099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, (-1)), 'str', "Add 'dir' to the list of directories that will be searched for\n        shared libraries at runtime.\n        ")
        
        # Call to append(...): (line 291)
        # Processing the call arguments (line 291)
        # Getting the type of 'dir' (line 291)
        dir_304103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 41), 'dir', False)
        # Processing the call keyword arguments (line 291)
        kwargs_304104 = {}
        # Getting the type of 'self' (line 291)
        self_304100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'self', False)
        # Obtaining the member 'runtime_library_dirs' of a type (line 291)
        runtime_library_dirs_304101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), self_304100, 'runtime_library_dirs')
        # Obtaining the member 'append' of a type (line 291)
        append_304102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), runtime_library_dirs_304101, 'append')
        # Calling append(args, kwargs) (line 291)
        append_call_result_304105 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), append_304102, *[dir_304103], **kwargs_304104)
        
        
        # ################# End of 'add_runtime_library_dir(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_runtime_library_dir' in the type store
        # Getting the type of 'stypy_return_type' (line 287)
        stypy_return_type_304106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304106)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_runtime_library_dir'
        return stypy_return_type_304106


    @norecursion
    def set_runtime_library_dirs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_runtime_library_dirs'
        module_type_store = module_type_store.open_function_context('set_runtime_library_dirs', 293, 4, False)
        # Assigning a type to the variable 'self' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.set_runtime_library_dirs.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.set_runtime_library_dirs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.set_runtime_library_dirs.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.set_runtime_library_dirs.__dict__.__setitem__('stypy_function_name', 'CCompiler.set_runtime_library_dirs')
        CCompiler.set_runtime_library_dirs.__dict__.__setitem__('stypy_param_names_list', ['dirs'])
        CCompiler.set_runtime_library_dirs.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.set_runtime_library_dirs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.set_runtime_library_dirs.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.set_runtime_library_dirs.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.set_runtime_library_dirs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.set_runtime_library_dirs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.set_runtime_library_dirs', ['dirs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_runtime_library_dirs', localization, ['dirs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_runtime_library_dirs(...)' code ##################

        str_304107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, (-1)), 'str', "Set the list of directories to search for shared libraries at\n        runtime to 'dirs' (a list of strings).  This does not affect any\n        standard search path that the runtime linker may search by\n        default.\n        ")
        
        # Assigning a Subscript to a Attribute (line 299):
        
        # Assigning a Subscript to a Attribute (line 299):
        
        # Obtaining the type of the subscript
        slice_304108 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 299, 36), None, None, None)
        # Getting the type of 'dirs' (line 299)
        dirs_304109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 36), 'dirs')
        # Obtaining the member '__getitem__' of a type (line 299)
        getitem___304110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 36), dirs_304109, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 299)
        subscript_call_result_304111 = invoke(stypy.reporting.localization.Localization(__file__, 299, 36), getitem___304110, slice_304108)
        
        # Getting the type of 'self' (line 299)
        self_304112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'self')
        # Setting the type of the member 'runtime_library_dirs' of a type (line 299)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 8), self_304112, 'runtime_library_dirs', subscript_call_result_304111)
        
        # ################# End of 'set_runtime_library_dirs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_runtime_library_dirs' in the type store
        # Getting the type of 'stypy_return_type' (line 293)
        stypy_return_type_304113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304113)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_runtime_library_dirs'
        return stypy_return_type_304113


    @norecursion
    def add_link_object(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_link_object'
        module_type_store = module_type_store.open_function_context('add_link_object', 301, 4, False)
        # Assigning a type to the variable 'self' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.add_link_object.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.add_link_object.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.add_link_object.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.add_link_object.__dict__.__setitem__('stypy_function_name', 'CCompiler.add_link_object')
        CCompiler.add_link_object.__dict__.__setitem__('stypy_param_names_list', ['object'])
        CCompiler.add_link_object.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.add_link_object.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.add_link_object.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.add_link_object.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.add_link_object.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.add_link_object.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.add_link_object', ['object'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_link_object', localization, ['object'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_link_object(...)' code ##################

        str_304114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, (-1)), 'str', 'Add \'object\' to the list of object files (or analogues, such as\n        explicitly named library files or the output of "resource\n        compilers") to be included in every link driven by this compiler\n        object.\n        ')
        
        # Call to append(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'object' (line 307)
        object_304118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 28), 'object', False)
        # Processing the call keyword arguments (line 307)
        kwargs_304119 = {}
        # Getting the type of 'self' (line 307)
        self_304115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'self', False)
        # Obtaining the member 'objects' of a type (line 307)
        objects_304116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), self_304115, 'objects')
        # Obtaining the member 'append' of a type (line 307)
        append_304117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), objects_304116, 'append')
        # Calling append(args, kwargs) (line 307)
        append_call_result_304120 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), append_304117, *[object_304118], **kwargs_304119)
        
        
        # ################# End of 'add_link_object(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_link_object' in the type store
        # Getting the type of 'stypy_return_type' (line 301)
        stypy_return_type_304121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304121)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_link_object'
        return stypy_return_type_304121


    @norecursion
    def set_link_objects(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_link_objects'
        module_type_store = module_type_store.open_function_context('set_link_objects', 309, 4, False)
        # Assigning a type to the variable 'self' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.set_link_objects.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.set_link_objects.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.set_link_objects.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.set_link_objects.__dict__.__setitem__('stypy_function_name', 'CCompiler.set_link_objects')
        CCompiler.set_link_objects.__dict__.__setitem__('stypy_param_names_list', ['objects'])
        CCompiler.set_link_objects.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.set_link_objects.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.set_link_objects.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.set_link_objects.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.set_link_objects.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.set_link_objects.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.set_link_objects', ['objects'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_link_objects', localization, ['objects'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_link_objects(...)' code ##################

        str_304122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, (-1)), 'str', "Set the list of object files (or analogues) to be included in\n        every link to 'objects'.  This does not affect any standard object\n        files that the linker may include by default (such as system\n        libraries).\n        ")
        
        # Assigning a Subscript to a Attribute (line 315):
        
        # Assigning a Subscript to a Attribute (line 315):
        
        # Obtaining the type of the subscript
        slice_304123 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 315, 23), None, None, None)
        # Getting the type of 'objects' (line 315)
        objects_304124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 23), 'objects')
        # Obtaining the member '__getitem__' of a type (line 315)
        getitem___304125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 23), objects_304124, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 315)
        subscript_call_result_304126 = invoke(stypy.reporting.localization.Localization(__file__, 315, 23), getitem___304125, slice_304123)
        
        # Getting the type of 'self' (line 315)
        self_304127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'self')
        # Setting the type of the member 'objects' of a type (line 315)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), self_304127, 'objects', subscript_call_result_304126)
        
        # ################# End of 'set_link_objects(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_link_objects' in the type store
        # Getting the type of 'stypy_return_type' (line 309)
        stypy_return_type_304128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304128)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_link_objects'
        return stypy_return_type_304128


    @norecursion
    def _setup_compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_setup_compile'
        module_type_store = module_type_store.open_function_context('_setup_compile', 323, 4, False)
        # Assigning a type to the variable 'self' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler._setup_compile.__dict__.__setitem__('stypy_localization', localization)
        CCompiler._setup_compile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler._setup_compile.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler._setup_compile.__dict__.__setitem__('stypy_function_name', 'CCompiler._setup_compile')
        CCompiler._setup_compile.__dict__.__setitem__('stypy_param_names_list', ['outdir', 'macros', 'incdirs', 'sources', 'depends', 'extra'])
        CCompiler._setup_compile.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler._setup_compile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler._setup_compile.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler._setup_compile.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler._setup_compile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler._setup_compile.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler._setup_compile', ['outdir', 'macros', 'incdirs', 'sources', 'depends', 'extra'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_setup_compile', localization, ['outdir', 'macros', 'incdirs', 'sources', 'depends', 'extra'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_setup_compile(...)' code ##################

        str_304129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 8), 'str', 'Process arguments and decide which source files to compile.')
        
        # Type idiom detected: calculating its left and rigth part (line 326)
        # Getting the type of 'outdir' (line 326)
        outdir_304130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 11), 'outdir')
        # Getting the type of 'None' (line 326)
        None_304131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 21), 'None')
        
        (may_be_304132, more_types_in_union_304133) = may_be_none(outdir_304130, None_304131)

        if may_be_304132:

            if more_types_in_union_304133:
                # Runtime conditional SSA (line 326)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 327):
            
            # Assigning a Attribute to a Name (line 327):
            # Getting the type of 'self' (line 327)
            self_304134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 21), 'self')
            # Obtaining the member 'output_dir' of a type (line 327)
            output_dir_304135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 21), self_304134, 'output_dir')
            # Assigning a type to the variable 'outdir' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'outdir', output_dir_304135)

            if more_types_in_union_304133:
                # Runtime conditional SSA for else branch (line 326)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_304132) or more_types_in_union_304133):
            
            # Type idiom detected: calculating its left and rigth part (line 328)
            # Getting the type of 'str' (line 328)
            str_304136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 36), 'str')
            # Getting the type of 'outdir' (line 328)
            outdir_304137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 28), 'outdir')
            
            (may_be_304138, more_types_in_union_304139) = may_not_be_subtype(str_304136, outdir_304137)

            if may_be_304138:

                if more_types_in_union_304139:
                    # Runtime conditional SSA (line 328)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'outdir' (line 328)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 13), 'outdir', remove_subtype_from_union(outdir_304137, str))
                # Getting the type of 'TypeError' (line 329)
                TypeError_304140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 18), 'TypeError')
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 329, 12), TypeError_304140, 'raise parameter', BaseException)

                if more_types_in_union_304139:
                    # SSA join for if statement (line 328)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_304132 and more_types_in_union_304133):
                # SSA join for if statement (line 326)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 331)
        # Getting the type of 'macros' (line 331)
        macros_304141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 11), 'macros')
        # Getting the type of 'None' (line 331)
        None_304142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 21), 'None')
        
        (may_be_304143, more_types_in_union_304144) = may_be_none(macros_304141, None_304142)

        if may_be_304143:

            if more_types_in_union_304144:
                # Runtime conditional SSA (line 331)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 332):
            
            # Assigning a Attribute to a Name (line 332):
            # Getting the type of 'self' (line 332)
            self_304145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 21), 'self')
            # Obtaining the member 'macros' of a type (line 332)
            macros_304146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 21), self_304145, 'macros')
            # Assigning a type to the variable 'macros' (line 332)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'macros', macros_304146)

            if more_types_in_union_304144:
                # Runtime conditional SSA for else branch (line 331)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_304143) or more_types_in_union_304144):
            
            # Type idiom detected: calculating its left and rigth part (line 333)
            # Getting the type of 'list' (line 333)
            list_304147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 32), 'list')
            # Getting the type of 'macros' (line 333)
            macros_304148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 24), 'macros')
            
            (may_be_304149, more_types_in_union_304150) = may_be_subtype(list_304147, macros_304148)

            if may_be_304149:

                if more_types_in_union_304150:
                    # Runtime conditional SSA (line 333)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'macros' (line 333)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 13), 'macros', remove_not_subtype_from_union(macros_304148, list))
                
                # Assigning a BinOp to a Name (line 334):
                
                # Assigning a BinOp to a Name (line 334):
                # Getting the type of 'macros' (line 334)
                macros_304151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 21), 'macros')
                
                # Evaluating a boolean operation
                # Getting the type of 'self' (line 334)
                self_304152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 31), 'self')
                # Obtaining the member 'macros' of a type (line 334)
                macros_304153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 31), self_304152, 'macros')
                
                # Obtaining an instance of the builtin type 'list' (line 334)
                list_304154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 46), 'list')
                # Adding type elements to the builtin type 'list' instance (line 334)
                
                # Applying the binary operator 'or' (line 334)
                result_or_keyword_304155 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 31), 'or', macros_304153, list_304154)
                
                # Applying the binary operator '+' (line 334)
                result_add_304156 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 21), '+', macros_304151, result_or_keyword_304155)
                
                # Assigning a type to the variable 'macros' (line 334)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 12), 'macros', result_add_304156)

                if more_types_in_union_304150:
                    # Runtime conditional SSA for else branch (line 333)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_304149) or more_types_in_union_304150):
                # Assigning a type to the variable 'macros' (line 333)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 13), 'macros', remove_subtype_from_union(macros_304148, list))
                # Getting the type of 'TypeError' (line 336)
                TypeError_304157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 18), 'TypeError')
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 336, 12), TypeError_304157, 'raise parameter', BaseException)

                if (may_be_304149 and more_types_in_union_304150):
                    # SSA join for if statement (line 333)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_304143 and more_types_in_union_304144):
                # SSA join for if statement (line 331)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 338)
        # Getting the type of 'incdirs' (line 338)
        incdirs_304158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 11), 'incdirs')
        # Getting the type of 'None' (line 338)
        None_304159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 22), 'None')
        
        (may_be_304160, more_types_in_union_304161) = may_be_none(incdirs_304158, None_304159)

        if may_be_304160:

            if more_types_in_union_304161:
                # Runtime conditional SSA (line 338)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 339):
            
            # Assigning a Attribute to a Name (line 339):
            # Getting the type of 'self' (line 339)
            self_304162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 22), 'self')
            # Obtaining the member 'include_dirs' of a type (line 339)
            include_dirs_304163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 22), self_304162, 'include_dirs')
            # Assigning a type to the variable 'incdirs' (line 339)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'incdirs', include_dirs_304163)

            if more_types_in_union_304161:
                # Runtime conditional SSA for else branch (line 338)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_304160) or more_types_in_union_304161):
            
            
            # Call to isinstance(...): (line 340)
            # Processing the call arguments (line 340)
            # Getting the type of 'incdirs' (line 340)
            incdirs_304165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 24), 'incdirs', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 340)
            tuple_304166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 34), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 340)
            # Adding element type (line 340)
            # Getting the type of 'list' (line 340)
            list_304167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 34), 'list', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 34), tuple_304166, list_304167)
            # Adding element type (line 340)
            # Getting the type of 'tuple' (line 340)
            tuple_304168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 40), 'tuple', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 340, 34), tuple_304166, tuple_304168)
            
            # Processing the call keyword arguments (line 340)
            kwargs_304169 = {}
            # Getting the type of 'isinstance' (line 340)
            isinstance_304164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 340)
            isinstance_call_result_304170 = invoke(stypy.reporting.localization.Localization(__file__, 340, 13), isinstance_304164, *[incdirs_304165, tuple_304166], **kwargs_304169)
            
            # Testing the type of an if condition (line 340)
            if_condition_304171 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 13), isinstance_call_result_304170)
            # Assigning a type to the variable 'if_condition_304171' (line 340)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 13), 'if_condition_304171', if_condition_304171)
            # SSA begins for if statement (line 340)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 341):
            
            # Assigning a BinOp to a Name (line 341):
            
            # Call to list(...): (line 341)
            # Processing the call arguments (line 341)
            # Getting the type of 'incdirs' (line 341)
            incdirs_304173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 27), 'incdirs', False)
            # Processing the call keyword arguments (line 341)
            kwargs_304174 = {}
            # Getting the type of 'list' (line 341)
            list_304172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 22), 'list', False)
            # Calling list(args, kwargs) (line 341)
            list_call_result_304175 = invoke(stypy.reporting.localization.Localization(__file__, 341, 22), list_304172, *[incdirs_304173], **kwargs_304174)
            
            
            # Evaluating a boolean operation
            # Getting the type of 'self' (line 341)
            self_304176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 39), 'self')
            # Obtaining the member 'include_dirs' of a type (line 341)
            include_dirs_304177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 39), self_304176, 'include_dirs')
            
            # Obtaining an instance of the builtin type 'list' (line 341)
            list_304178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 60), 'list')
            # Adding type elements to the builtin type 'list' instance (line 341)
            
            # Applying the binary operator 'or' (line 341)
            result_or_keyword_304179 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 39), 'or', include_dirs_304177, list_304178)
            
            # Applying the binary operator '+' (line 341)
            result_add_304180 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 22), '+', list_call_result_304175, result_or_keyword_304179)
            
            # Assigning a type to the variable 'incdirs' (line 341)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 12), 'incdirs', result_add_304180)
            # SSA branch for the else part of an if statement (line 340)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'TypeError' (line 343)
            TypeError_304181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 18), 'TypeError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 343, 12), TypeError_304181, 'raise parameter', BaseException)
            # SSA join for if statement (line 340)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_304160 and more_types_in_union_304161):
                # SSA join for if statement (line 338)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 346)
        # Getting the type of 'extra' (line 346)
        extra_304182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 11), 'extra')
        # Getting the type of 'None' (line 346)
        None_304183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 20), 'None')
        
        (may_be_304184, more_types_in_union_304185) = may_be_none(extra_304182, None_304183)

        if may_be_304184:

            if more_types_in_union_304185:
                # Runtime conditional SSA (line 346)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Name (line 347):
            
            # Assigning a List to a Name (line 347):
            
            # Obtaining an instance of the builtin type 'list' (line 347)
            list_304186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 20), 'list')
            # Adding type elements to the builtin type 'list' instance (line 347)
            
            # Assigning a type to the variable 'extra' (line 347)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'extra', list_304186)

            if more_types_in_union_304185:
                # SSA join for if statement (line 346)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 350):
        
        # Assigning a Call to a Name (line 350):
        
        # Call to object_filenames(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of 'sources' (line 350)
        sources_304189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 40), 'sources', False)
        # Processing the call keyword arguments (line 350)
        int_304190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 50), 'int')
        keyword_304191 = int_304190
        # Getting the type of 'outdir' (line 352)
        outdir_304192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 51), 'outdir', False)
        keyword_304193 = outdir_304192
        kwargs_304194 = {'output_dir': keyword_304193, 'strip_dir': keyword_304191}
        # Getting the type of 'self' (line 350)
        self_304187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 18), 'self', False)
        # Obtaining the member 'object_filenames' of a type (line 350)
        object_filenames_304188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 18), self_304187, 'object_filenames')
        # Calling object_filenames(args, kwargs) (line 350)
        object_filenames_call_result_304195 = invoke(stypy.reporting.localization.Localization(__file__, 350, 18), object_filenames_304188, *[sources_304189], **kwargs_304194)
        
        # Assigning a type to the variable 'objects' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'objects', object_filenames_call_result_304195)
        # Evaluating assert statement condition
        
        
        # Call to len(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'objects' (line 353)
        objects_304197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 19), 'objects', False)
        # Processing the call keyword arguments (line 353)
        kwargs_304198 = {}
        # Getting the type of 'len' (line 353)
        len_304196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 15), 'len', False)
        # Calling len(args, kwargs) (line 353)
        len_call_result_304199 = invoke(stypy.reporting.localization.Localization(__file__, 353, 15), len_304196, *[objects_304197], **kwargs_304198)
        
        
        # Call to len(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'sources' (line 353)
        sources_304201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 35), 'sources', False)
        # Processing the call keyword arguments (line 353)
        kwargs_304202 = {}
        # Getting the type of 'len' (line 353)
        len_304200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 31), 'len', False)
        # Calling len(args, kwargs) (line 353)
        len_call_result_304203 = invoke(stypy.reporting.localization.Localization(__file__, 353, 31), len_304200, *[sources_304201], **kwargs_304202)
        
        # Applying the binary operator '==' (line 353)
        result_eq_304204 = python_operator(stypy.reporting.localization.Localization(__file__, 353, 15), '==', len_call_result_304199, len_call_result_304203)
        
        
        # Assigning a Call to a Name (line 355):
        
        # Assigning a Call to a Name (line 355):
        
        # Call to gen_preprocess_options(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'macros' (line 355)
        macros_304206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 41), 'macros', False)
        # Getting the type of 'incdirs' (line 355)
        incdirs_304207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 49), 'incdirs', False)
        # Processing the call keyword arguments (line 355)
        kwargs_304208 = {}
        # Getting the type of 'gen_preprocess_options' (line 355)
        gen_preprocess_options_304205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 18), 'gen_preprocess_options', False)
        # Calling gen_preprocess_options(args, kwargs) (line 355)
        gen_preprocess_options_call_result_304209 = invoke(stypy.reporting.localization.Localization(__file__, 355, 18), gen_preprocess_options_304205, *[macros_304206, incdirs_304207], **kwargs_304208)
        
        # Assigning a type to the variable 'pp_opts' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'pp_opts', gen_preprocess_options_call_result_304209)
        
        # Assigning a Dict to a Name (line 357):
        
        # Assigning a Dict to a Name (line 357):
        
        # Obtaining an instance of the builtin type 'dict' (line 357)
        dict_304210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 357)
        
        # Assigning a type to the variable 'build' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'build', dict_304210)
        
        
        # Call to range(...): (line 358)
        # Processing the call arguments (line 358)
        
        # Call to len(...): (line 358)
        # Processing the call arguments (line 358)
        # Getting the type of 'sources' (line 358)
        sources_304213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 27), 'sources', False)
        # Processing the call keyword arguments (line 358)
        kwargs_304214 = {}
        # Getting the type of 'len' (line 358)
        len_304212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 23), 'len', False)
        # Calling len(args, kwargs) (line 358)
        len_call_result_304215 = invoke(stypy.reporting.localization.Localization(__file__, 358, 23), len_304212, *[sources_304213], **kwargs_304214)
        
        # Processing the call keyword arguments (line 358)
        kwargs_304216 = {}
        # Getting the type of 'range' (line 358)
        range_304211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 17), 'range', False)
        # Calling range(args, kwargs) (line 358)
        range_call_result_304217 = invoke(stypy.reporting.localization.Localization(__file__, 358, 17), range_304211, *[len_call_result_304215], **kwargs_304216)
        
        # Testing the type of a for loop iterable (line 358)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 358, 8), range_call_result_304217)
        # Getting the type of the for loop variable (line 358)
        for_loop_var_304218 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 358, 8), range_call_result_304217)
        # Assigning a type to the variable 'i' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'i', for_loop_var_304218)
        # SSA begins for a for statement (line 358)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 359):
        
        # Assigning a Subscript to a Name (line 359):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 359)
        i_304219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 26), 'i')
        # Getting the type of 'sources' (line 359)
        sources_304220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 18), 'sources')
        # Obtaining the member '__getitem__' of a type (line 359)
        getitem___304221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 18), sources_304220, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 359)
        subscript_call_result_304222 = invoke(stypy.reporting.localization.Localization(__file__, 359, 18), getitem___304221, i_304219)
        
        # Assigning a type to the variable 'src' (line 359)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'src', subscript_call_result_304222)
        
        # Assigning a Subscript to a Name (line 360):
        
        # Assigning a Subscript to a Name (line 360):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 360)
        i_304223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 26), 'i')
        # Getting the type of 'objects' (line 360)
        objects_304224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 18), 'objects')
        # Obtaining the member '__getitem__' of a type (line 360)
        getitem___304225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 18), objects_304224, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 360)
        subscript_call_result_304226 = invoke(stypy.reporting.localization.Localization(__file__, 360, 18), getitem___304225, i_304223)
        
        # Assigning a type to the variable 'obj' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'obj', subscript_call_result_304226)
        
        # Assigning a Subscript to a Name (line 361):
        
        # Assigning a Subscript to a Name (line 361):
        
        # Obtaining the type of the subscript
        int_304227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 40), 'int')
        
        # Call to splitext(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'src' (line 361)
        src_304231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 35), 'src', False)
        # Processing the call keyword arguments (line 361)
        kwargs_304232 = {}
        # Getting the type of 'os' (line 361)
        os_304228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 361)
        path_304229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 18), os_304228, 'path')
        # Obtaining the member 'splitext' of a type (line 361)
        splitext_304230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 18), path_304229, 'splitext')
        # Calling splitext(args, kwargs) (line 361)
        splitext_call_result_304233 = invoke(stypy.reporting.localization.Localization(__file__, 361, 18), splitext_304230, *[src_304231], **kwargs_304232)
        
        # Obtaining the member '__getitem__' of a type (line 361)
        getitem___304234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 18), splitext_call_result_304233, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 361)
        subscript_call_result_304235 = invoke(stypy.reporting.localization.Localization(__file__, 361, 18), getitem___304234, int_304227)
        
        # Assigning a type to the variable 'ext' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'ext', subscript_call_result_304235)
        
        # Call to mkpath(...): (line 362)
        # Processing the call arguments (line 362)
        
        # Call to dirname(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'obj' (line 362)
        obj_304241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 40), 'obj', False)
        # Processing the call keyword arguments (line 362)
        kwargs_304242 = {}
        # Getting the type of 'os' (line 362)
        os_304238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 362)
        path_304239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 24), os_304238, 'path')
        # Obtaining the member 'dirname' of a type (line 362)
        dirname_304240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 24), path_304239, 'dirname')
        # Calling dirname(args, kwargs) (line 362)
        dirname_call_result_304243 = invoke(stypy.reporting.localization.Localization(__file__, 362, 24), dirname_304240, *[obj_304241], **kwargs_304242)
        
        # Processing the call keyword arguments (line 362)
        kwargs_304244 = {}
        # Getting the type of 'self' (line 362)
        self_304236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 362)
        mkpath_304237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), self_304236, 'mkpath')
        # Calling mkpath(args, kwargs) (line 362)
        mkpath_call_result_304245 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), mkpath_304237, *[dirname_call_result_304243], **kwargs_304244)
        
        
        # Assigning a Tuple to a Subscript (line 363):
        
        # Assigning a Tuple to a Subscript (line 363):
        
        # Obtaining an instance of the builtin type 'tuple' (line 363)
        tuple_304246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 363)
        # Adding element type (line 363)
        # Getting the type of 'src' (line 363)
        src_304247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 26), 'src')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 26), tuple_304246, src_304247)
        # Adding element type (line 363)
        # Getting the type of 'ext' (line 363)
        ext_304248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 31), 'ext')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 26), tuple_304246, ext_304248)
        
        # Getting the type of 'build' (line 363)
        build_304249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'build')
        # Getting the type of 'obj' (line 363)
        obj_304250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 18), 'obj')
        # Storing an element on a container (line 363)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 12), build_304249, (obj_304250, tuple_304246))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 365)
        tuple_304251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 365)
        # Adding element type (line 365)
        # Getting the type of 'macros' (line 365)
        macros_304252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 15), 'macros')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 15), tuple_304251, macros_304252)
        # Adding element type (line 365)
        # Getting the type of 'objects' (line 365)
        objects_304253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 23), 'objects')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 15), tuple_304251, objects_304253)
        # Adding element type (line 365)
        # Getting the type of 'extra' (line 365)
        extra_304254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 32), 'extra')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 15), tuple_304251, extra_304254)
        # Adding element type (line 365)
        # Getting the type of 'pp_opts' (line 365)
        pp_opts_304255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 39), 'pp_opts')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 15), tuple_304251, pp_opts_304255)
        # Adding element type (line 365)
        # Getting the type of 'build' (line 365)
        build_304256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 48), 'build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 365, 15), tuple_304251, build_304256)
        
        # Assigning a type to the variable 'stypy_return_type' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'stypy_return_type', tuple_304251)
        
        # ################# End of '_setup_compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_setup_compile' in the type store
        # Getting the type of 'stypy_return_type' (line 323)
        stypy_return_type_304257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304257)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_setup_compile'
        return stypy_return_type_304257


    @norecursion
    def _get_cc_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_cc_args'
        module_type_store = module_type_store.open_function_context('_get_cc_args', 367, 4, False)
        # Assigning a type to the variable 'self' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler._get_cc_args.__dict__.__setitem__('stypy_localization', localization)
        CCompiler._get_cc_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler._get_cc_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler._get_cc_args.__dict__.__setitem__('stypy_function_name', 'CCompiler._get_cc_args')
        CCompiler._get_cc_args.__dict__.__setitem__('stypy_param_names_list', ['pp_opts', 'debug', 'before'])
        CCompiler._get_cc_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler._get_cc_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler._get_cc_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler._get_cc_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler._get_cc_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler._get_cc_args.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler._get_cc_args', ['pp_opts', 'debug', 'before'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_cc_args', localization, ['pp_opts', 'debug', 'before'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_cc_args(...)' code ##################

        
        # Assigning a BinOp to a Name (line 369):
        
        # Assigning a BinOp to a Name (line 369):
        # Getting the type of 'pp_opts' (line 369)
        pp_opts_304258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 18), 'pp_opts')
        
        # Obtaining an instance of the builtin type 'list' (line 369)
        list_304259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 369)
        # Adding element type (line 369)
        str_304260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 29), 'str', '-c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 369, 28), list_304259, str_304260)
        
        # Applying the binary operator '+' (line 369)
        result_add_304261 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 18), '+', pp_opts_304258, list_304259)
        
        # Assigning a type to the variable 'cc_args' (line 369)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'cc_args', result_add_304261)
        
        # Getting the type of 'debug' (line 370)
        debug_304262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 11), 'debug')
        # Testing the type of an if condition (line 370)
        if_condition_304263 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 370, 8), debug_304262)
        # Assigning a type to the variable 'if_condition_304263' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'if_condition_304263', if_condition_304263)
        # SSA begins for if statement (line 370)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Subscript (line 371):
        
        # Assigning a List to a Subscript (line 371):
        
        # Obtaining an instance of the builtin type 'list' (line 371)
        list_304264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 371)
        # Adding element type (line 371)
        str_304265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 27), 'str', '-g')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 26), list_304264, str_304265)
        
        # Getting the type of 'cc_args' (line 371)
        cc_args_304266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'cc_args')
        int_304267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 21), 'int')
        slice_304268 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 371, 12), None, int_304267, None)
        # Storing an element on a container (line 371)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 12), cc_args_304266, (slice_304268, list_304264))
        # SSA join for if statement (line 370)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'before' (line 372)
        before_304269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 11), 'before')
        # Testing the type of an if condition (line 372)
        if_condition_304270 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 372, 8), before_304269)
        # Assigning a type to the variable 'if_condition_304270' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'if_condition_304270', if_condition_304270)
        # SSA begins for if statement (line 372)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 373):
        
        # Assigning a Name to a Subscript (line 373):
        # Getting the type of 'before' (line 373)
        before_304271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 26), 'before')
        # Getting the type of 'cc_args' (line 373)
        cc_args_304272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 12), 'cc_args')
        int_304273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 21), 'int')
        slice_304274 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 373, 12), None, int_304273, None)
        # Storing an element on a container (line 373)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 12), cc_args_304272, (slice_304274, before_304271))
        # SSA join for if statement (line 372)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'cc_args' (line 374)
        cc_args_304275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 15), 'cc_args')
        # Assigning a type to the variable 'stypy_return_type' (line 374)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'stypy_return_type', cc_args_304275)
        
        # ################# End of '_get_cc_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_cc_args' in the type store
        # Getting the type of 'stypy_return_type' (line 367)
        stypy_return_type_304276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304276)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_cc_args'
        return stypy_return_type_304276


    @norecursion
    def _fix_compile_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_fix_compile_args'
        module_type_store = module_type_store.open_function_context('_fix_compile_args', 376, 4, False)
        # Assigning a type to the variable 'self' (line 377)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler._fix_compile_args.__dict__.__setitem__('stypy_localization', localization)
        CCompiler._fix_compile_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler._fix_compile_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler._fix_compile_args.__dict__.__setitem__('stypy_function_name', 'CCompiler._fix_compile_args')
        CCompiler._fix_compile_args.__dict__.__setitem__('stypy_param_names_list', ['output_dir', 'macros', 'include_dirs'])
        CCompiler._fix_compile_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler._fix_compile_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler._fix_compile_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler._fix_compile_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler._fix_compile_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler._fix_compile_args.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler._fix_compile_args', ['output_dir', 'macros', 'include_dirs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fix_compile_args', localization, ['output_dir', 'macros', 'include_dirs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fix_compile_args(...)' code ##################

        str_304277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, (-1)), 'str', "Typecheck and fix-up some of the arguments to the 'compile()'\n        method, and return fixed-up values.  Specifically: if 'output_dir'\n        is None, replaces it with 'self.output_dir'; ensures that 'macros'\n        is a list, and augments it with 'self.macros'; ensures that\n        'include_dirs' is a list, and augments it with 'self.include_dirs'.\n        Guarantees that the returned values are of the correct type,\n        i.e. for 'output_dir' either string or None, and for 'macros' and\n        'include_dirs' either list or None.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 386)
        # Getting the type of 'output_dir' (line 386)
        output_dir_304278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 11), 'output_dir')
        # Getting the type of 'None' (line 386)
        None_304279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 25), 'None')
        
        (may_be_304280, more_types_in_union_304281) = may_be_none(output_dir_304278, None_304279)

        if may_be_304280:

            if more_types_in_union_304281:
                # Runtime conditional SSA (line 386)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 387):
            
            # Assigning a Attribute to a Name (line 387):
            # Getting the type of 'self' (line 387)
            self_304282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 25), 'self')
            # Obtaining the member 'output_dir' of a type (line 387)
            output_dir_304283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 25), self_304282, 'output_dir')
            # Assigning a type to the variable 'output_dir' (line 387)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'output_dir', output_dir_304283)

            if more_types_in_union_304281:
                # Runtime conditional SSA for else branch (line 386)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_304280) or more_types_in_union_304281):
            
            # Type idiom detected: calculating its left and rigth part (line 388)
            # Getting the type of 'str' (line 388)
            str_304284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 40), 'str')
            # Getting the type of 'output_dir' (line 388)
            output_dir_304285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 28), 'output_dir')
            
            (may_be_304286, more_types_in_union_304287) = may_not_be_subtype(str_304284, output_dir_304285)

            if may_be_304286:

                if more_types_in_union_304287:
                    # Runtime conditional SSA (line 388)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'output_dir' (line 388)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 13), 'output_dir', remove_subtype_from_union(output_dir_304285, str))
                # Getting the type of 'TypeError' (line 389)
                TypeError_304288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 18), 'TypeError')
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 389, 12), TypeError_304288, 'raise parameter', BaseException)

                if more_types_in_union_304287:
                    # SSA join for if statement (line 388)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_304280 and more_types_in_union_304281):
                # SSA join for if statement (line 386)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 391)
        # Getting the type of 'macros' (line 391)
        macros_304289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 11), 'macros')
        # Getting the type of 'None' (line 391)
        None_304290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 21), 'None')
        
        (may_be_304291, more_types_in_union_304292) = may_be_none(macros_304289, None_304290)

        if may_be_304291:

            if more_types_in_union_304292:
                # Runtime conditional SSA (line 391)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 392):
            
            # Assigning a Attribute to a Name (line 392):
            # Getting the type of 'self' (line 392)
            self_304293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 21), 'self')
            # Obtaining the member 'macros' of a type (line 392)
            macros_304294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 21), self_304293, 'macros')
            # Assigning a type to the variable 'macros' (line 392)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 12), 'macros', macros_304294)

            if more_types_in_union_304292:
                # Runtime conditional SSA for else branch (line 391)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_304291) or more_types_in_union_304292):
            
            # Type idiom detected: calculating its left and rigth part (line 393)
            # Getting the type of 'list' (line 393)
            list_304295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 32), 'list')
            # Getting the type of 'macros' (line 393)
            macros_304296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 24), 'macros')
            
            (may_be_304297, more_types_in_union_304298) = may_be_subtype(list_304295, macros_304296)

            if may_be_304297:

                if more_types_in_union_304298:
                    # Runtime conditional SSA (line 393)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'macros' (line 393)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 13), 'macros', remove_not_subtype_from_union(macros_304296, list))
                
                # Assigning a BinOp to a Name (line 394):
                
                # Assigning a BinOp to a Name (line 394):
                # Getting the type of 'macros' (line 394)
                macros_304299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 21), 'macros')
                
                # Evaluating a boolean operation
                # Getting the type of 'self' (line 394)
                self_304300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 31), 'self')
                # Obtaining the member 'macros' of a type (line 394)
                macros_304301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 31), self_304300, 'macros')
                
                # Obtaining an instance of the builtin type 'list' (line 394)
                list_304302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 46), 'list')
                # Adding type elements to the builtin type 'list' instance (line 394)
                
                # Applying the binary operator 'or' (line 394)
                result_or_keyword_304303 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 31), 'or', macros_304301, list_304302)
                
                # Applying the binary operator '+' (line 394)
                result_add_304304 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 21), '+', macros_304299, result_or_keyword_304303)
                
                # Assigning a type to the variable 'macros' (line 394)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'macros', result_add_304304)

                if more_types_in_union_304298:
                    # Runtime conditional SSA for else branch (line 393)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_304297) or more_types_in_union_304298):
                # Assigning a type to the variable 'macros' (line 393)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 13), 'macros', remove_subtype_from_union(macros_304296, list))
                # Getting the type of 'TypeError' (line 396)
                TypeError_304305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 18), 'TypeError')
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 396, 12), TypeError_304305, 'raise parameter', BaseException)

                if (may_be_304297 and more_types_in_union_304298):
                    # SSA join for if statement (line 393)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_304291 and more_types_in_union_304292):
                # SSA join for if statement (line 391)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 398)
        # Getting the type of 'include_dirs' (line 398)
        include_dirs_304306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 11), 'include_dirs')
        # Getting the type of 'None' (line 398)
        None_304307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 27), 'None')
        
        (may_be_304308, more_types_in_union_304309) = may_be_none(include_dirs_304306, None_304307)

        if may_be_304308:

            if more_types_in_union_304309:
                # Runtime conditional SSA (line 398)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 399):
            
            # Assigning a Attribute to a Name (line 399):
            # Getting the type of 'self' (line 399)
            self_304310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 27), 'self')
            # Obtaining the member 'include_dirs' of a type (line 399)
            include_dirs_304311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 27), self_304310, 'include_dirs')
            # Assigning a type to the variable 'include_dirs' (line 399)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 12), 'include_dirs', include_dirs_304311)

            if more_types_in_union_304309:
                # Runtime conditional SSA for else branch (line 398)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_304308) or more_types_in_union_304309):
            
            
            # Call to isinstance(...): (line 400)
            # Processing the call arguments (line 400)
            # Getting the type of 'include_dirs' (line 400)
            include_dirs_304313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 24), 'include_dirs', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 400)
            tuple_304314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 39), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 400)
            # Adding element type (line 400)
            # Getting the type of 'list' (line 400)
            list_304315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 39), 'list', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 39), tuple_304314, list_304315)
            # Adding element type (line 400)
            # Getting the type of 'tuple' (line 400)
            tuple_304316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 45), 'tuple', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 39), tuple_304314, tuple_304316)
            
            # Processing the call keyword arguments (line 400)
            kwargs_304317 = {}
            # Getting the type of 'isinstance' (line 400)
            isinstance_304312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 13), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 400)
            isinstance_call_result_304318 = invoke(stypy.reporting.localization.Localization(__file__, 400, 13), isinstance_304312, *[include_dirs_304313, tuple_304314], **kwargs_304317)
            
            # Testing the type of an if condition (line 400)
            if_condition_304319 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 400, 13), isinstance_call_result_304318)
            # Assigning a type to the variable 'if_condition_304319' (line 400)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 13), 'if_condition_304319', if_condition_304319)
            # SSA begins for if statement (line 400)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 401):
            
            # Assigning a BinOp to a Name (line 401):
            
            # Call to list(...): (line 401)
            # Processing the call arguments (line 401)
            # Getting the type of 'include_dirs' (line 401)
            include_dirs_304321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 33), 'include_dirs', False)
            # Processing the call keyword arguments (line 401)
            kwargs_304322 = {}
            # Getting the type of 'list' (line 401)
            list_304320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 27), 'list', False)
            # Calling list(args, kwargs) (line 401)
            list_call_result_304323 = invoke(stypy.reporting.localization.Localization(__file__, 401, 27), list_304320, *[include_dirs_304321], **kwargs_304322)
            
            
            # Evaluating a boolean operation
            # Getting the type of 'self' (line 401)
            self_304324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 50), 'self')
            # Obtaining the member 'include_dirs' of a type (line 401)
            include_dirs_304325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 50), self_304324, 'include_dirs')
            
            # Obtaining an instance of the builtin type 'list' (line 401)
            list_304326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 71), 'list')
            # Adding type elements to the builtin type 'list' instance (line 401)
            
            # Applying the binary operator 'or' (line 401)
            result_or_keyword_304327 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 50), 'or', include_dirs_304325, list_304326)
            
            # Applying the binary operator '+' (line 401)
            result_add_304328 = python_operator(stypy.reporting.localization.Localization(__file__, 401, 27), '+', list_call_result_304323, result_or_keyword_304327)
            
            # Assigning a type to the variable 'include_dirs' (line 401)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 12), 'include_dirs', result_add_304328)
            # SSA branch for the else part of an if statement (line 400)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'TypeError' (line 403)
            TypeError_304329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 18), 'TypeError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 403, 12), TypeError_304329, 'raise parameter', BaseException)
            # SSA join for if statement (line 400)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_304308 and more_types_in_union_304309):
                # SSA join for if statement (line 398)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Obtaining an instance of the builtin type 'tuple' (line 406)
        tuple_304330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 406)
        # Adding element type (line 406)
        # Getting the type of 'output_dir' (line 406)
        output_dir_304331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 15), 'output_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 15), tuple_304330, output_dir_304331)
        # Adding element type (line 406)
        # Getting the type of 'macros' (line 406)
        macros_304332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 27), 'macros')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 15), tuple_304330, macros_304332)
        # Adding element type (line 406)
        # Getting the type of 'include_dirs' (line 406)
        include_dirs_304333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 35), 'include_dirs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 15), tuple_304330, include_dirs_304333)
        
        # Assigning a type to the variable 'stypy_return_type' (line 406)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'stypy_return_type', tuple_304330)
        
        # ################# End of '_fix_compile_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fix_compile_args' in the type store
        # Getting the type of 'stypy_return_type' (line 376)
        stypy_return_type_304334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304334)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fix_compile_args'
        return stypy_return_type_304334


    @norecursion
    def _fix_object_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_fix_object_args'
        module_type_store = module_type_store.open_function_context('_fix_object_args', 408, 4, False)
        # Assigning a type to the variable 'self' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler._fix_object_args.__dict__.__setitem__('stypy_localization', localization)
        CCompiler._fix_object_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler._fix_object_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler._fix_object_args.__dict__.__setitem__('stypy_function_name', 'CCompiler._fix_object_args')
        CCompiler._fix_object_args.__dict__.__setitem__('stypy_param_names_list', ['objects', 'output_dir'])
        CCompiler._fix_object_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler._fix_object_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler._fix_object_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler._fix_object_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler._fix_object_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler._fix_object_args.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler._fix_object_args', ['objects', 'output_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fix_object_args', localization, ['objects', 'output_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fix_object_args(...)' code ##################

        str_304335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, (-1)), 'str', "Typecheck and fix up some arguments supplied to various methods.\n        Specifically: ensure that 'objects' is a list; if output_dir is\n        None, replace with self.output_dir.  Return fixed versions of\n        'objects' and 'output_dir'.\n        ")
        
        
        
        # Call to isinstance(...): (line 414)
        # Processing the call arguments (line 414)
        # Getting the type of 'objects' (line 414)
        objects_304337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 26), 'objects', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 414)
        tuple_304338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 414)
        # Adding element type (line 414)
        # Getting the type of 'list' (line 414)
        list_304339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 36), 'list', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 36), tuple_304338, list_304339)
        # Adding element type (line 414)
        # Getting the type of 'tuple' (line 414)
        tuple_304340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 42), 'tuple', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 414, 36), tuple_304338, tuple_304340)
        
        # Processing the call keyword arguments (line 414)
        kwargs_304341 = {}
        # Getting the type of 'isinstance' (line 414)
        isinstance_304336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 414)
        isinstance_call_result_304342 = invoke(stypy.reporting.localization.Localization(__file__, 414, 15), isinstance_304336, *[objects_304337, tuple_304338], **kwargs_304341)
        
        # Applying the 'not' unary operator (line 414)
        result_not__304343 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 11), 'not', isinstance_call_result_304342)
        
        # Testing the type of an if condition (line 414)
        if_condition_304344 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 414, 8), result_not__304343)
        # Assigning a type to the variable 'if_condition_304344' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'if_condition_304344', if_condition_304344)
        # SSA begins for if statement (line 414)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'TypeError' (line 415)
        TypeError_304345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 18), 'TypeError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 415, 12), TypeError_304345, 'raise parameter', BaseException)
        # SSA join for if statement (line 414)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 417):
        
        # Assigning a Call to a Name (line 417):
        
        # Call to list(...): (line 417)
        # Processing the call arguments (line 417)
        # Getting the type of 'objects' (line 417)
        objects_304347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 24), 'objects', False)
        # Processing the call keyword arguments (line 417)
        kwargs_304348 = {}
        # Getting the type of 'list' (line 417)
        list_304346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 18), 'list', False)
        # Calling list(args, kwargs) (line 417)
        list_call_result_304349 = invoke(stypy.reporting.localization.Localization(__file__, 417, 18), list_304346, *[objects_304347], **kwargs_304348)
        
        # Assigning a type to the variable 'objects' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'objects', list_call_result_304349)
        
        # Type idiom detected: calculating its left and rigth part (line 419)
        # Getting the type of 'output_dir' (line 419)
        output_dir_304350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 11), 'output_dir')
        # Getting the type of 'None' (line 419)
        None_304351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 25), 'None')
        
        (may_be_304352, more_types_in_union_304353) = may_be_none(output_dir_304350, None_304351)

        if may_be_304352:

            if more_types_in_union_304353:
                # Runtime conditional SSA (line 419)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 420):
            
            # Assigning a Attribute to a Name (line 420):
            # Getting the type of 'self' (line 420)
            self_304354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 25), 'self')
            # Obtaining the member 'output_dir' of a type (line 420)
            output_dir_304355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 25), self_304354, 'output_dir')
            # Assigning a type to the variable 'output_dir' (line 420)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 12), 'output_dir', output_dir_304355)

            if more_types_in_union_304353:
                # Runtime conditional SSA for else branch (line 419)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_304352) or more_types_in_union_304353):
            
            # Type idiom detected: calculating its left and rigth part (line 421)
            # Getting the type of 'str' (line 421)
            str_304356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 40), 'str')
            # Getting the type of 'output_dir' (line 421)
            output_dir_304357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 28), 'output_dir')
            
            (may_be_304358, more_types_in_union_304359) = may_not_be_subtype(str_304356, output_dir_304357)

            if may_be_304358:

                if more_types_in_union_304359:
                    # Runtime conditional SSA (line 421)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'output_dir' (line 421)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 13), 'output_dir', remove_subtype_from_union(output_dir_304357, str))
                # Getting the type of 'TypeError' (line 422)
                TypeError_304360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 18), 'TypeError')
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 422, 12), TypeError_304360, 'raise parameter', BaseException)

                if more_types_in_union_304359:
                    # SSA join for if statement (line 421)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_304352 and more_types_in_union_304353):
                # SSA join for if statement (line 419)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Obtaining an instance of the builtin type 'tuple' (line 424)
        tuple_304361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 424)
        # Adding element type (line 424)
        # Getting the type of 'objects' (line 424)
        objects_304362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 16), 'objects')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 16), tuple_304361, objects_304362)
        # Adding element type (line 424)
        # Getting the type of 'output_dir' (line 424)
        output_dir_304363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 25), 'output_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 16), tuple_304361, output_dir_304363)
        
        # Assigning a type to the variable 'stypy_return_type' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'stypy_return_type', tuple_304361)
        
        # ################# End of '_fix_object_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fix_object_args' in the type store
        # Getting the type of 'stypy_return_type' (line 408)
        stypy_return_type_304364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304364)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fix_object_args'
        return stypy_return_type_304364


    @norecursion
    def _fix_lib_args(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_fix_lib_args'
        module_type_store = module_type_store.open_function_context('_fix_lib_args', 426, 4, False)
        # Assigning a type to the variable 'self' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler._fix_lib_args.__dict__.__setitem__('stypy_localization', localization)
        CCompiler._fix_lib_args.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler._fix_lib_args.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler._fix_lib_args.__dict__.__setitem__('stypy_function_name', 'CCompiler._fix_lib_args')
        CCompiler._fix_lib_args.__dict__.__setitem__('stypy_param_names_list', ['libraries', 'library_dirs', 'runtime_library_dirs'])
        CCompiler._fix_lib_args.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler._fix_lib_args.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler._fix_lib_args.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler._fix_lib_args.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler._fix_lib_args.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler._fix_lib_args.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler._fix_lib_args', ['libraries', 'library_dirs', 'runtime_library_dirs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_fix_lib_args', localization, ['libraries', 'library_dirs', 'runtime_library_dirs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_fix_lib_args(...)' code ##################

        str_304365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, (-1)), 'str', "Typecheck and fix up some of the arguments supplied to the\n        'link_*' methods.  Specifically: ensure that all arguments are\n        lists, and augment them with their permanent versions\n        (eg. 'self.libraries' augments 'libraries').  Return a tuple with\n        fixed versions of all arguments.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 433)
        # Getting the type of 'libraries' (line 433)
        libraries_304366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 11), 'libraries')
        # Getting the type of 'None' (line 433)
        None_304367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 24), 'None')
        
        (may_be_304368, more_types_in_union_304369) = may_be_none(libraries_304366, None_304367)

        if may_be_304368:

            if more_types_in_union_304369:
                # Runtime conditional SSA (line 433)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 434):
            
            # Assigning a Attribute to a Name (line 434):
            # Getting the type of 'self' (line 434)
            self_304370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 24), 'self')
            # Obtaining the member 'libraries' of a type (line 434)
            libraries_304371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 24), self_304370, 'libraries')
            # Assigning a type to the variable 'libraries' (line 434)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 'libraries', libraries_304371)

            if more_types_in_union_304369:
                # Runtime conditional SSA for else branch (line 433)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_304368) or more_types_in_union_304369):
            
            
            # Call to isinstance(...): (line 435)
            # Processing the call arguments (line 435)
            # Getting the type of 'libraries' (line 435)
            libraries_304373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 24), 'libraries', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 435)
            tuple_304374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 36), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 435)
            # Adding element type (line 435)
            # Getting the type of 'list' (line 435)
            list_304375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 36), 'list', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 36), tuple_304374, list_304375)
            # Adding element type (line 435)
            # Getting the type of 'tuple' (line 435)
            tuple_304376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 42), 'tuple', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 435, 36), tuple_304374, tuple_304376)
            
            # Processing the call keyword arguments (line 435)
            kwargs_304377 = {}
            # Getting the type of 'isinstance' (line 435)
            isinstance_304372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 13), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 435)
            isinstance_call_result_304378 = invoke(stypy.reporting.localization.Localization(__file__, 435, 13), isinstance_304372, *[libraries_304373, tuple_304374], **kwargs_304377)
            
            # Testing the type of an if condition (line 435)
            if_condition_304379 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 435, 13), isinstance_call_result_304378)
            # Assigning a type to the variable 'if_condition_304379' (line 435)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 13), 'if_condition_304379', if_condition_304379)
            # SSA begins for if statement (line 435)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 436):
            
            # Assigning a BinOp to a Name (line 436):
            
            # Call to list(...): (line 436)
            # Processing the call arguments (line 436)
            # Getting the type of 'libraries' (line 436)
            libraries_304381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 30), 'libraries', False)
            # Processing the call keyword arguments (line 436)
            kwargs_304382 = {}
            # Getting the type of 'list' (line 436)
            list_304380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 24), 'list', False)
            # Calling list(args, kwargs) (line 436)
            list_call_result_304383 = invoke(stypy.reporting.localization.Localization(__file__, 436, 24), list_304380, *[libraries_304381], **kwargs_304382)
            
            
            # Evaluating a boolean operation
            # Getting the type of 'self' (line 436)
            self_304384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 44), 'self')
            # Obtaining the member 'libraries' of a type (line 436)
            libraries_304385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 44), self_304384, 'libraries')
            
            # Obtaining an instance of the builtin type 'list' (line 436)
            list_304386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 62), 'list')
            # Adding type elements to the builtin type 'list' instance (line 436)
            
            # Applying the binary operator 'or' (line 436)
            result_or_keyword_304387 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 44), 'or', libraries_304385, list_304386)
            
            # Applying the binary operator '+' (line 436)
            result_add_304388 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 24), '+', list_call_result_304383, result_or_keyword_304387)
            
            # Assigning a type to the variable 'libraries' (line 436)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 12), 'libraries', result_add_304388)
            # SSA branch for the else part of an if statement (line 435)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'TypeError' (line 438)
            TypeError_304389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 18), 'TypeError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 438, 12), TypeError_304389, 'raise parameter', BaseException)
            # SSA join for if statement (line 435)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_304368 and more_types_in_union_304369):
                # SSA join for if statement (line 433)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 441)
        # Getting the type of 'library_dirs' (line 441)
        library_dirs_304390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 11), 'library_dirs')
        # Getting the type of 'None' (line 441)
        None_304391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 27), 'None')
        
        (may_be_304392, more_types_in_union_304393) = may_be_none(library_dirs_304390, None_304391)

        if may_be_304392:

            if more_types_in_union_304393:
                # Runtime conditional SSA (line 441)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 442):
            
            # Assigning a Attribute to a Name (line 442):
            # Getting the type of 'self' (line 442)
            self_304394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 27), 'self')
            # Obtaining the member 'library_dirs' of a type (line 442)
            library_dirs_304395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 27), self_304394, 'library_dirs')
            # Assigning a type to the variable 'library_dirs' (line 442)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 12), 'library_dirs', library_dirs_304395)

            if more_types_in_union_304393:
                # Runtime conditional SSA for else branch (line 441)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_304392) or more_types_in_union_304393):
            
            
            # Call to isinstance(...): (line 443)
            # Processing the call arguments (line 443)
            # Getting the type of 'library_dirs' (line 443)
            library_dirs_304397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 24), 'library_dirs', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 443)
            tuple_304398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 39), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 443)
            # Adding element type (line 443)
            # Getting the type of 'list' (line 443)
            list_304399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 39), 'list', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 39), tuple_304398, list_304399)
            # Adding element type (line 443)
            # Getting the type of 'tuple' (line 443)
            tuple_304400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 45), 'tuple', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 443, 39), tuple_304398, tuple_304400)
            
            # Processing the call keyword arguments (line 443)
            kwargs_304401 = {}
            # Getting the type of 'isinstance' (line 443)
            isinstance_304396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 13), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 443)
            isinstance_call_result_304402 = invoke(stypy.reporting.localization.Localization(__file__, 443, 13), isinstance_304396, *[library_dirs_304397, tuple_304398], **kwargs_304401)
            
            # Testing the type of an if condition (line 443)
            if_condition_304403 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 443, 13), isinstance_call_result_304402)
            # Assigning a type to the variable 'if_condition_304403' (line 443)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 13), 'if_condition_304403', if_condition_304403)
            # SSA begins for if statement (line 443)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 444):
            
            # Assigning a BinOp to a Name (line 444):
            
            # Call to list(...): (line 444)
            # Processing the call arguments (line 444)
            # Getting the type of 'library_dirs' (line 444)
            library_dirs_304405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 33), 'library_dirs', False)
            # Processing the call keyword arguments (line 444)
            kwargs_304406 = {}
            # Getting the type of 'list' (line 444)
            list_304404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 27), 'list', False)
            # Calling list(args, kwargs) (line 444)
            list_call_result_304407 = invoke(stypy.reporting.localization.Localization(__file__, 444, 27), list_304404, *[library_dirs_304405], **kwargs_304406)
            
            
            # Evaluating a boolean operation
            # Getting the type of 'self' (line 444)
            self_304408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 50), 'self')
            # Obtaining the member 'library_dirs' of a type (line 444)
            library_dirs_304409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 50), self_304408, 'library_dirs')
            
            # Obtaining an instance of the builtin type 'list' (line 444)
            list_304410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 71), 'list')
            # Adding type elements to the builtin type 'list' instance (line 444)
            
            # Applying the binary operator 'or' (line 444)
            result_or_keyword_304411 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 50), 'or', library_dirs_304409, list_304410)
            
            # Applying the binary operator '+' (line 444)
            result_add_304412 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 27), '+', list_call_result_304407, result_or_keyword_304411)
            
            # Assigning a type to the variable 'library_dirs' (line 444)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 12), 'library_dirs', result_add_304412)
            # SSA branch for the else part of an if statement (line 443)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'TypeError' (line 446)
            TypeError_304413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 18), 'TypeError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 446, 12), TypeError_304413, 'raise parameter', BaseException)
            # SSA join for if statement (line 443)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_304392 and more_types_in_union_304393):
                # SSA join for if statement (line 441)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 449)
        # Getting the type of 'runtime_library_dirs' (line 449)
        runtime_library_dirs_304414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 11), 'runtime_library_dirs')
        # Getting the type of 'None' (line 449)
        None_304415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 35), 'None')
        
        (may_be_304416, more_types_in_union_304417) = may_be_none(runtime_library_dirs_304414, None_304415)

        if may_be_304416:

            if more_types_in_union_304417:
                # Runtime conditional SSA (line 449)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 450):
            
            # Assigning a Attribute to a Name (line 450):
            # Getting the type of 'self' (line 450)
            self_304418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 35), 'self')
            # Obtaining the member 'runtime_library_dirs' of a type (line 450)
            runtime_library_dirs_304419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 35), self_304418, 'runtime_library_dirs')
            # Assigning a type to the variable 'runtime_library_dirs' (line 450)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 12), 'runtime_library_dirs', runtime_library_dirs_304419)

            if more_types_in_union_304417:
                # Runtime conditional SSA for else branch (line 449)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_304416) or more_types_in_union_304417):
            
            
            # Call to isinstance(...): (line 451)
            # Processing the call arguments (line 451)
            # Getting the type of 'runtime_library_dirs' (line 451)
            runtime_library_dirs_304421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 24), 'runtime_library_dirs', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 451)
            tuple_304422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 47), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 451)
            # Adding element type (line 451)
            # Getting the type of 'list' (line 451)
            list_304423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 47), 'list', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 47), tuple_304422, list_304423)
            # Adding element type (line 451)
            # Getting the type of 'tuple' (line 451)
            tuple_304424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 53), 'tuple', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 451, 47), tuple_304422, tuple_304424)
            
            # Processing the call keyword arguments (line 451)
            kwargs_304425 = {}
            # Getting the type of 'isinstance' (line 451)
            isinstance_304420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 13), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 451)
            isinstance_call_result_304426 = invoke(stypy.reporting.localization.Localization(__file__, 451, 13), isinstance_304420, *[runtime_library_dirs_304421, tuple_304422], **kwargs_304425)
            
            # Testing the type of an if condition (line 451)
            if_condition_304427 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 451, 13), isinstance_call_result_304426)
            # Assigning a type to the variable 'if_condition_304427' (line 451)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 13), 'if_condition_304427', if_condition_304427)
            # SSA begins for if statement (line 451)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 452):
            
            # Assigning a BinOp to a Name (line 452):
            
            # Call to list(...): (line 452)
            # Processing the call arguments (line 452)
            # Getting the type of 'runtime_library_dirs' (line 452)
            runtime_library_dirs_304429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 42), 'runtime_library_dirs', False)
            # Processing the call keyword arguments (line 452)
            kwargs_304430 = {}
            # Getting the type of 'list' (line 452)
            list_304428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 36), 'list', False)
            # Calling list(args, kwargs) (line 452)
            list_call_result_304431 = invoke(stypy.reporting.localization.Localization(__file__, 452, 36), list_304428, *[runtime_library_dirs_304429], **kwargs_304430)
            
            
            # Evaluating a boolean operation
            # Getting the type of 'self' (line 453)
            self_304432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 37), 'self')
            # Obtaining the member 'runtime_library_dirs' of a type (line 453)
            runtime_library_dirs_304433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 37), self_304432, 'runtime_library_dirs')
            
            # Obtaining an instance of the builtin type 'list' (line 453)
            list_304434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 66), 'list')
            # Adding type elements to the builtin type 'list' instance (line 453)
            
            # Applying the binary operator 'or' (line 453)
            result_or_keyword_304435 = python_operator(stypy.reporting.localization.Localization(__file__, 453, 37), 'or', runtime_library_dirs_304433, list_304434)
            
            # Applying the binary operator '+' (line 452)
            result_add_304436 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 36), '+', list_call_result_304431, result_or_keyword_304435)
            
            # Assigning a type to the variable 'runtime_library_dirs' (line 452)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 12), 'runtime_library_dirs', result_add_304436)
            # SSA branch for the else part of an if statement (line 451)
            module_type_store.open_ssa_branch('else')
            # Getting the type of 'TypeError' (line 455)
            TypeError_304437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 18), 'TypeError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 455, 12), TypeError_304437, 'raise parameter', BaseException)
            # SSA join for if statement (line 451)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_304416 and more_types_in_union_304417):
                # SSA join for if statement (line 449)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Obtaining an instance of the builtin type 'tuple' (line 459)
        tuple_304438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 459)
        # Adding element type (line 459)
        # Getting the type of 'libraries' (line 459)
        libraries_304439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 16), 'libraries')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 16), tuple_304438, libraries_304439)
        # Adding element type (line 459)
        # Getting the type of 'library_dirs' (line 459)
        library_dirs_304440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 27), 'library_dirs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 16), tuple_304438, library_dirs_304440)
        # Adding element type (line 459)
        # Getting the type of 'runtime_library_dirs' (line 459)
        runtime_library_dirs_304441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 41), 'runtime_library_dirs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 459, 16), tuple_304438, runtime_library_dirs_304441)
        
        # Assigning a type to the variable 'stypy_return_type' (line 459)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 8), 'stypy_return_type', tuple_304438)
        
        # ################# End of '_fix_lib_args(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_fix_lib_args' in the type store
        # Getting the type of 'stypy_return_type' (line 426)
        stypy_return_type_304442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304442)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_fix_lib_args'
        return stypy_return_type_304442


    @norecursion
    def _need_link(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_need_link'
        module_type_store = module_type_store.open_function_context('_need_link', 461, 4, False)
        # Assigning a type to the variable 'self' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler._need_link.__dict__.__setitem__('stypy_localization', localization)
        CCompiler._need_link.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler._need_link.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler._need_link.__dict__.__setitem__('stypy_function_name', 'CCompiler._need_link')
        CCompiler._need_link.__dict__.__setitem__('stypy_param_names_list', ['objects', 'output_file'])
        CCompiler._need_link.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler._need_link.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler._need_link.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler._need_link.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler._need_link.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler._need_link.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler._need_link', ['objects', 'output_file'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_need_link', localization, ['objects', 'output_file'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_need_link(...)' code ##################

        str_304443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, (-1)), 'str', "Return true if we need to relink the files listed in 'objects'\n        to recreate 'output_file'.\n        ")
        
        # Getting the type of 'self' (line 465)
        self_304444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 11), 'self')
        # Obtaining the member 'force' of a type (line 465)
        force_304445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 11), self_304444, 'force')
        # Testing the type of an if condition (line 465)
        if_condition_304446 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 465, 8), force_304445)
        # Assigning a type to the variable 'if_condition_304446' (line 465)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'if_condition_304446', if_condition_304446)
        # SSA begins for if statement (line 465)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_304447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 12), 'stypy_return_type', int_304447)
        # SSA branch for the else part of an if statement (line 465)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 468)
        self_304448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 15), 'self')
        # Obtaining the member 'dry_run' of a type (line 468)
        dry_run_304449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 15), self_304448, 'dry_run')
        # Testing the type of an if condition (line 468)
        if_condition_304450 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 468, 12), dry_run_304449)
        # Assigning a type to the variable 'if_condition_304450' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 12), 'if_condition_304450', if_condition_304450)
        # SSA begins for if statement (line 468)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 469):
        
        # Assigning a Call to a Name (line 469):
        
        # Call to newer_group(...): (line 469)
        # Processing the call arguments (line 469)
        # Getting the type of 'objects' (line 469)
        objects_304452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 37), 'objects', False)
        # Getting the type of 'output_file' (line 469)
        output_file_304453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 46), 'output_file', False)
        # Processing the call keyword arguments (line 469)
        str_304454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 67), 'str', 'newer')
        keyword_304455 = str_304454
        kwargs_304456 = {'missing': keyword_304455}
        # Getting the type of 'newer_group' (line 469)
        newer_group_304451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 24), 'newer_group', False)
        # Calling newer_group(args, kwargs) (line 469)
        newer_group_call_result_304457 = invoke(stypy.reporting.localization.Localization(__file__, 469, 24), newer_group_304451, *[objects_304452, output_file_304453], **kwargs_304456)
        
        # Assigning a type to the variable 'newer' (line 469)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 16), 'newer', newer_group_call_result_304457)
        # SSA branch for the else part of an if statement (line 468)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 471):
        
        # Assigning a Call to a Name (line 471):
        
        # Call to newer_group(...): (line 471)
        # Processing the call arguments (line 471)
        # Getting the type of 'objects' (line 471)
        objects_304459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 37), 'objects', False)
        # Getting the type of 'output_file' (line 471)
        output_file_304460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 46), 'output_file', False)
        # Processing the call keyword arguments (line 471)
        kwargs_304461 = {}
        # Getting the type of 'newer_group' (line 471)
        newer_group_304458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 24), 'newer_group', False)
        # Calling newer_group(args, kwargs) (line 471)
        newer_group_call_result_304462 = invoke(stypy.reporting.localization.Localization(__file__, 471, 24), newer_group_304458, *[objects_304459, output_file_304460], **kwargs_304461)
        
        # Assigning a type to the variable 'newer' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 16), 'newer', newer_group_call_result_304462)
        # SSA join for if statement (line 468)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'newer' (line 472)
        newer_304463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 19), 'newer')
        # Assigning a type to the variable 'stypy_return_type' (line 472)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'stypy_return_type', newer_304463)
        # SSA join for if statement (line 465)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_need_link(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_need_link' in the type store
        # Getting the type of 'stypy_return_type' (line 461)
        stypy_return_type_304464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304464)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_need_link'
        return stypy_return_type_304464


    @norecursion
    def detect_language(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'detect_language'
        module_type_store = module_type_store.open_function_context('detect_language', 474, 4, False)
        # Assigning a type to the variable 'self' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.detect_language.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.detect_language.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.detect_language.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.detect_language.__dict__.__setitem__('stypy_function_name', 'CCompiler.detect_language')
        CCompiler.detect_language.__dict__.__setitem__('stypy_param_names_list', ['sources'])
        CCompiler.detect_language.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.detect_language.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.detect_language.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.detect_language.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.detect_language.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.detect_language.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.detect_language', ['sources'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'detect_language', localization, ['sources'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'detect_language(...)' code ##################

        str_304465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, (-1)), 'str', 'Detect the language of a given file, or list of files. Uses\n        language_map, and language_order to do the job.\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 478)
        # Getting the type of 'list' (line 478)
        list_304466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 35), 'list')
        # Getting the type of 'sources' (line 478)
        sources_304467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 26), 'sources')
        
        (may_be_304468, more_types_in_union_304469) = may_not_be_subtype(list_304466, sources_304467)

        if may_be_304468:

            if more_types_in_union_304469:
                # Runtime conditional SSA (line 478)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'sources' (line 478)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'sources', remove_subtype_from_union(sources_304467, list))
            
            # Assigning a List to a Name (line 479):
            
            # Assigning a List to a Name (line 479):
            
            # Obtaining an instance of the builtin type 'list' (line 479)
            list_304470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 479, 22), 'list')
            # Adding type elements to the builtin type 'list' instance (line 479)
            # Adding element type (line 479)
            # Getting the type of 'sources' (line 479)
            sources_304471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 23), 'sources')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 479, 22), list_304470, sources_304471)
            
            # Assigning a type to the variable 'sources' (line 479)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'sources', list_304470)

            if more_types_in_union_304469:
                # SSA join for if statement (line 478)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Name (line 480):
        
        # Assigning a Name to a Name (line 480):
        # Getting the type of 'None' (line 480)
        None_304472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 15), 'None')
        # Assigning a type to the variable 'lang' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'lang', None_304472)
        
        # Assigning a Call to a Name (line 481):
        
        # Assigning a Call to a Name (line 481):
        
        # Call to len(...): (line 481)
        # Processing the call arguments (line 481)
        # Getting the type of 'self' (line 481)
        self_304474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 20), 'self', False)
        # Obtaining the member 'language_order' of a type (line 481)
        language_order_304475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 20), self_304474, 'language_order')
        # Processing the call keyword arguments (line 481)
        kwargs_304476 = {}
        # Getting the type of 'len' (line 481)
        len_304473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 16), 'len', False)
        # Calling len(args, kwargs) (line 481)
        len_call_result_304477 = invoke(stypy.reporting.localization.Localization(__file__, 481, 16), len_304473, *[language_order_304475], **kwargs_304476)
        
        # Assigning a type to the variable 'index' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'index', len_call_result_304477)
        
        # Getting the type of 'sources' (line 482)
        sources_304478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 22), 'sources')
        # Testing the type of a for loop iterable (line 482)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 482, 8), sources_304478)
        # Getting the type of the for loop variable (line 482)
        for_loop_var_304479 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 482, 8), sources_304478)
        # Assigning a type to the variable 'source' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'source', for_loop_var_304479)
        # SSA begins for a for statement (line 482)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 483):
        
        # Assigning a Call to a Name:
        
        # Call to splitext(...): (line 483)
        # Processing the call arguments (line 483)
        # Getting the type of 'source' (line 483)
        source_304483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 41), 'source', False)
        # Processing the call keyword arguments (line 483)
        kwargs_304484 = {}
        # Getting the type of 'os' (line 483)
        os_304480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 483)
        path_304481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 24), os_304480, 'path')
        # Obtaining the member 'splitext' of a type (line 483)
        splitext_304482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 24), path_304481, 'splitext')
        # Calling splitext(args, kwargs) (line 483)
        splitext_call_result_304485 = invoke(stypy.reporting.localization.Localization(__file__, 483, 24), splitext_304482, *[source_304483], **kwargs_304484)
        
        # Assigning a type to the variable 'call_assignment_303808' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'call_assignment_303808', splitext_call_result_304485)
        
        # Assigning a Call to a Name (line 483):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_304488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 12), 'int')
        # Processing the call keyword arguments
        kwargs_304489 = {}
        # Getting the type of 'call_assignment_303808' (line 483)
        call_assignment_303808_304486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'call_assignment_303808', False)
        # Obtaining the member '__getitem__' of a type (line 483)
        getitem___304487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 12), call_assignment_303808_304486, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_304490 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___304487, *[int_304488], **kwargs_304489)
        
        # Assigning a type to the variable 'call_assignment_303809' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'call_assignment_303809', getitem___call_result_304490)
        
        # Assigning a Name to a Name (line 483):
        # Getting the type of 'call_assignment_303809' (line 483)
        call_assignment_303809_304491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'call_assignment_303809')
        # Assigning a type to the variable 'base' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'base', call_assignment_303809_304491)
        
        # Assigning a Call to a Name (line 483):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_304494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 12), 'int')
        # Processing the call keyword arguments
        kwargs_304495 = {}
        # Getting the type of 'call_assignment_303808' (line 483)
        call_assignment_303808_304492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'call_assignment_303808', False)
        # Obtaining the member '__getitem__' of a type (line 483)
        getitem___304493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 12), call_assignment_303808_304492, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_304496 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___304493, *[int_304494], **kwargs_304495)
        
        # Assigning a type to the variable 'call_assignment_303810' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'call_assignment_303810', getitem___call_result_304496)
        
        # Assigning a Name to a Name (line 483):
        # Getting the type of 'call_assignment_303810' (line 483)
        call_assignment_303810_304497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 12), 'call_assignment_303810')
        # Assigning a type to the variable 'ext' (line 483)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 18), 'ext', call_assignment_303810_304497)
        
        # Assigning a Call to a Name (line 484):
        
        # Assigning a Call to a Name (line 484):
        
        # Call to get(...): (line 484)
        # Processing the call arguments (line 484)
        # Getting the type of 'ext' (line 484)
        ext_304501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 44), 'ext', False)
        # Processing the call keyword arguments (line 484)
        kwargs_304502 = {}
        # Getting the type of 'self' (line 484)
        self_304498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 22), 'self', False)
        # Obtaining the member 'language_map' of a type (line 484)
        language_map_304499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 22), self_304498, 'language_map')
        # Obtaining the member 'get' of a type (line 484)
        get_304500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 22), language_map_304499, 'get')
        # Calling get(args, kwargs) (line 484)
        get_call_result_304503 = invoke(stypy.reporting.localization.Localization(__file__, 484, 22), get_304500, *[ext_304501], **kwargs_304502)
        
        # Assigning a type to the variable 'extlang' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 12), 'extlang', get_call_result_304503)
        
        
        # SSA begins for try-except statement (line 485)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 486):
        
        # Assigning a Call to a Name (line 486):
        
        # Call to index(...): (line 486)
        # Processing the call arguments (line 486)
        # Getting the type of 'extlang' (line 486)
        extlang_304507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 53), 'extlang', False)
        # Processing the call keyword arguments (line 486)
        kwargs_304508 = {}
        # Getting the type of 'self' (line 486)
        self_304504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 27), 'self', False)
        # Obtaining the member 'language_order' of a type (line 486)
        language_order_304505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 27), self_304504, 'language_order')
        # Obtaining the member 'index' of a type (line 486)
        index_304506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 27), language_order_304505, 'index')
        # Calling index(args, kwargs) (line 486)
        index_call_result_304509 = invoke(stypy.reporting.localization.Localization(__file__, 486, 27), index_304506, *[extlang_304507], **kwargs_304508)
        
        # Assigning a type to the variable 'extindex' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 16), 'extindex', index_call_result_304509)
        
        
        # Getting the type of 'extindex' (line 487)
        extindex_304510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 19), 'extindex')
        # Getting the type of 'index' (line 487)
        index_304511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 30), 'index')
        # Applying the binary operator '<' (line 487)
        result_lt_304512 = python_operator(stypy.reporting.localization.Localization(__file__, 487, 19), '<', extindex_304510, index_304511)
        
        # Testing the type of an if condition (line 487)
        if_condition_304513 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 487, 16), result_lt_304512)
        # Assigning a type to the variable 'if_condition_304513' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 16), 'if_condition_304513', if_condition_304513)
        # SSA begins for if statement (line 487)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 488):
        
        # Assigning a Name to a Name (line 488):
        # Getting the type of 'extlang' (line 488)
        extlang_304514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 27), 'extlang')
        # Assigning a type to the variable 'lang' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 20), 'lang', extlang_304514)
        
        # Assigning a Name to a Name (line 489):
        
        # Assigning a Name to a Name (line 489):
        # Getting the type of 'extindex' (line 489)
        extindex_304515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 28), 'extindex')
        # Assigning a type to the variable 'index' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 20), 'index', extindex_304515)
        # SSA join for if statement (line 487)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 485)
        # SSA branch for the except 'ValueError' branch of a try statement (line 485)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 485)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'lang' (line 492)
        lang_304516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 15), 'lang')
        # Assigning a type to the variable 'stypy_return_type' (line 492)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'stypy_return_type', lang_304516)
        
        # ################# End of 'detect_language(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'detect_language' in the type store
        # Getting the type of 'stypy_return_type' (line 474)
        stypy_return_type_304517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304517)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'detect_language'
        return stypy_return_type_304517


    @norecursion
    def preprocess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 497)
        None_304518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 45), 'None')
        # Getting the type of 'None' (line 497)
        None_304519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 58), 'None')
        # Getting the type of 'None' (line 498)
        None_304520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 32), 'None')
        # Getting the type of 'None' (line 498)
        None_304521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 52), 'None')
        # Getting the type of 'None' (line 498)
        None_304522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 73), 'None')
        defaults = [None_304518, None_304519, None_304520, None_304521, None_304522]
        # Create a new context for function 'preprocess'
        module_type_store = module_type_store.open_function_context('preprocess', 497, 4, False)
        # Assigning a type to the variable 'self' (line 498)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.preprocess.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.preprocess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.preprocess.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.preprocess.__dict__.__setitem__('stypy_function_name', 'CCompiler.preprocess')
        CCompiler.preprocess.__dict__.__setitem__('stypy_param_names_list', ['source', 'output_file', 'macros', 'include_dirs', 'extra_preargs', 'extra_postargs'])
        CCompiler.preprocess.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.preprocess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.preprocess.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.preprocess.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.preprocess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.preprocess.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.preprocess', ['source', 'output_file', 'macros', 'include_dirs', 'extra_preargs', 'extra_postargs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'preprocess', localization, ['source', 'output_file', 'macros', 'include_dirs', 'extra_preargs', 'extra_postargs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'preprocess(...)' code ##################

        str_304523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, (-1)), 'str', "Preprocess a single C/C++ source file, named in 'source'.\n        Output will be written to file named 'output_file', or stdout if\n        'output_file' not supplied.  'macros' is a list of macro\n        definitions as for 'compile()', which will augment the macros set\n        with 'define_macro()' and 'undefine_macro()'.  'include_dirs' is a\n        list of directory names that will be added to the default list.\n\n        Raises PreprocessError on failure.\n        ")
        pass
        
        # ################# End of 'preprocess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'preprocess' in the type store
        # Getting the type of 'stypy_return_type' (line 497)
        stypy_return_type_304524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304524)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'preprocess'
        return stypy_return_type_304524


    @norecursion
    def compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 510)
        None_304525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 42), 'None')
        # Getting the type of 'None' (line 510)
        None_304526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 55), 'None')
        # Getting the type of 'None' (line 511)
        None_304527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 29), 'None')
        int_304528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 41), 'int')
        # Getting the type of 'None' (line 511)
        None_304529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 58), 'None')
        # Getting the type of 'None' (line 512)
        None_304530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 31), 'None')
        # Getting the type of 'None' (line 512)
        None_304531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 45), 'None')
        defaults = [None_304525, None_304526, None_304527, int_304528, None_304529, None_304530, None_304531]
        # Create a new context for function 'compile'
        module_type_store = module_type_store.open_function_context('compile', 510, 4, False)
        # Assigning a type to the variable 'self' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.compile.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.compile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.compile.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.compile.__dict__.__setitem__('stypy_function_name', 'CCompiler.compile')
        CCompiler.compile.__dict__.__setitem__('stypy_param_names_list', ['sources', 'output_dir', 'macros', 'include_dirs', 'debug', 'extra_preargs', 'extra_postargs', 'depends'])
        CCompiler.compile.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.compile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.compile.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.compile.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.compile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.compile.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.compile', ['sources', 'output_dir', 'macros', 'include_dirs', 'debug', 'extra_preargs', 'extra_postargs', 'depends'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'compile', localization, ['sources', 'output_dir', 'macros', 'include_dirs', 'debug', 'extra_preargs', 'extra_postargs', 'depends'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'compile(...)' code ##################

        str_304532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 560, (-1)), 'str', 'Compile one or more source files.\n\n        \'sources\' must be a list of filenames, most likely C/C++\n        files, but in reality anything that can be handled by a\n        particular compiler and compiler class (eg. MSVCCompiler can\n        handle resource files in \'sources\').  Return a list of object\n        filenames, one per source filename in \'sources\'.  Depending on\n        the implementation, not all source files will necessarily be\n        compiled, but all corresponding object filenames will be\n        returned.\n\n        If \'output_dir\' is given, object files will be put under it, while\n        retaining their original path component.  That is, "foo/bar.c"\n        normally compiles to "foo/bar.o" (for a Unix implementation); if\n        \'output_dir\' is "build", then it would compile to\n        "build/foo/bar.o".\n\n        \'macros\', if given, must be a list of macro definitions.  A macro\n        definition is either a (name, value) 2-tuple or a (name,) 1-tuple.\n        The former defines a macro; if the value is None, the macro is\n        defined without an explicit value.  The 1-tuple case undefines a\n        macro.  Later definitions/redefinitions/ undefinitions take\n        precedence.\n\n        \'include_dirs\', if given, must be a list of strings, the\n        directories to add to the default include file search path for this\n        compilation only.\n\n        \'debug\' is a boolean; if true, the compiler will be instructed to\n        output debug symbols in (or alongside) the object file(s).\n\n        \'extra_preargs\' and \'extra_postargs\' are implementation- dependent.\n        On platforms that have the notion of a command-line (e.g. Unix,\n        DOS/Windows), they are most likely lists of strings: extra\n        command-line arguments to prepand/append to the compiler command\n        line.  On other platforms, consult the implementation class\n        documentation.  In any event, they are intended as an escape hatch\n        for those occasions when the abstract compiler framework doesn\'t\n        cut the mustard.\n\n        \'depends\', if given, is a list of filenames that all targets\n        depend on.  If a source file is older than any file in\n        depends, then the source file will be recompiled.  This\n        supports dependency tracking, but only at a coarse\n        granularity.\n\n        Raises CompileError on failure.\n        ')
        
        # Assigning a Call to a Tuple (line 564):
        
        # Assigning a Call to a Name:
        
        # Call to _setup_compile(...): (line 565)
        # Processing the call arguments (line 565)
        # Getting the type of 'output_dir' (line 565)
        output_dir_304535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 36), 'output_dir', False)
        # Getting the type of 'macros' (line 565)
        macros_304536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 48), 'macros', False)
        # Getting the type of 'include_dirs' (line 565)
        include_dirs_304537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 56), 'include_dirs', False)
        # Getting the type of 'sources' (line 565)
        sources_304538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 70), 'sources', False)
        # Getting the type of 'depends' (line 566)
        depends_304539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 36), 'depends', False)
        # Getting the type of 'extra_postargs' (line 566)
        extra_postargs_304540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 45), 'extra_postargs', False)
        # Processing the call keyword arguments (line 565)
        kwargs_304541 = {}
        # Getting the type of 'self' (line 565)
        self_304533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 16), 'self', False)
        # Obtaining the member '_setup_compile' of a type (line 565)
        _setup_compile_304534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 16), self_304533, '_setup_compile')
        # Calling _setup_compile(args, kwargs) (line 565)
        _setup_compile_call_result_304542 = invoke(stypy.reporting.localization.Localization(__file__, 565, 16), _setup_compile_304534, *[output_dir_304535, macros_304536, include_dirs_304537, sources_304538, depends_304539, extra_postargs_304540], **kwargs_304541)
        
        # Assigning a type to the variable 'call_assignment_303811' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_303811', _setup_compile_call_result_304542)
        
        # Assigning a Call to a Name (line 564):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_304545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 8), 'int')
        # Processing the call keyword arguments
        kwargs_304546 = {}
        # Getting the type of 'call_assignment_303811' (line 564)
        call_assignment_303811_304543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_303811', False)
        # Obtaining the member '__getitem__' of a type (line 564)
        getitem___304544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 8), call_assignment_303811_304543, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_304547 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___304544, *[int_304545], **kwargs_304546)
        
        # Assigning a type to the variable 'call_assignment_303812' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_303812', getitem___call_result_304547)
        
        # Assigning a Name to a Name (line 564):
        # Getting the type of 'call_assignment_303812' (line 564)
        call_assignment_303812_304548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_303812')
        # Assigning a type to the variable 'macros' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'macros', call_assignment_303812_304548)
        
        # Assigning a Call to a Name (line 564):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_304551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 8), 'int')
        # Processing the call keyword arguments
        kwargs_304552 = {}
        # Getting the type of 'call_assignment_303811' (line 564)
        call_assignment_303811_304549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_303811', False)
        # Obtaining the member '__getitem__' of a type (line 564)
        getitem___304550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 8), call_assignment_303811_304549, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_304553 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___304550, *[int_304551], **kwargs_304552)
        
        # Assigning a type to the variable 'call_assignment_303813' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_303813', getitem___call_result_304553)
        
        # Assigning a Name to a Name (line 564):
        # Getting the type of 'call_assignment_303813' (line 564)
        call_assignment_303813_304554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_303813')
        # Assigning a type to the variable 'objects' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 16), 'objects', call_assignment_303813_304554)
        
        # Assigning a Call to a Name (line 564):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_304557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 8), 'int')
        # Processing the call keyword arguments
        kwargs_304558 = {}
        # Getting the type of 'call_assignment_303811' (line 564)
        call_assignment_303811_304555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_303811', False)
        # Obtaining the member '__getitem__' of a type (line 564)
        getitem___304556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 8), call_assignment_303811_304555, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_304559 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___304556, *[int_304557], **kwargs_304558)
        
        # Assigning a type to the variable 'call_assignment_303814' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_303814', getitem___call_result_304559)
        
        # Assigning a Name to a Name (line 564):
        # Getting the type of 'call_assignment_303814' (line 564)
        call_assignment_303814_304560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_303814')
        # Assigning a type to the variable 'extra_postargs' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 25), 'extra_postargs', call_assignment_303814_304560)
        
        # Assigning a Call to a Name (line 564):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_304563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 8), 'int')
        # Processing the call keyword arguments
        kwargs_304564 = {}
        # Getting the type of 'call_assignment_303811' (line 564)
        call_assignment_303811_304561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_303811', False)
        # Obtaining the member '__getitem__' of a type (line 564)
        getitem___304562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 8), call_assignment_303811_304561, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_304565 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___304562, *[int_304563], **kwargs_304564)
        
        # Assigning a type to the variable 'call_assignment_303815' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_303815', getitem___call_result_304565)
        
        # Assigning a Name to a Name (line 564):
        # Getting the type of 'call_assignment_303815' (line 564)
        call_assignment_303815_304566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_303815')
        # Assigning a type to the variable 'pp_opts' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 41), 'pp_opts', call_assignment_303815_304566)
        
        # Assigning a Call to a Name (line 564):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_304569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 8), 'int')
        # Processing the call keyword arguments
        kwargs_304570 = {}
        # Getting the type of 'call_assignment_303811' (line 564)
        call_assignment_303811_304567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_303811', False)
        # Obtaining the member '__getitem__' of a type (line 564)
        getitem___304568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 8), call_assignment_303811_304567, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_304571 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___304568, *[int_304569], **kwargs_304570)
        
        # Assigning a type to the variable 'call_assignment_303816' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_303816', getitem___call_result_304571)
        
        # Assigning a Name to a Name (line 564):
        # Getting the type of 'call_assignment_303816' (line 564)
        call_assignment_303816_304572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'call_assignment_303816')
        # Assigning a type to the variable 'build' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 50), 'build', call_assignment_303816_304572)
        
        # Assigning a Call to a Name (line 567):
        
        # Assigning a Call to a Name (line 567):
        
        # Call to _get_cc_args(...): (line 567)
        # Processing the call arguments (line 567)
        # Getting the type of 'pp_opts' (line 567)
        pp_opts_304575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 36), 'pp_opts', False)
        # Getting the type of 'debug' (line 567)
        debug_304576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 45), 'debug', False)
        # Getting the type of 'extra_preargs' (line 567)
        extra_preargs_304577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 52), 'extra_preargs', False)
        # Processing the call keyword arguments (line 567)
        kwargs_304578 = {}
        # Getting the type of 'self' (line 567)
        self_304573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 18), 'self', False)
        # Obtaining the member '_get_cc_args' of a type (line 567)
        _get_cc_args_304574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 18), self_304573, '_get_cc_args')
        # Calling _get_cc_args(args, kwargs) (line 567)
        _get_cc_args_call_result_304579 = invoke(stypy.reporting.localization.Localization(__file__, 567, 18), _get_cc_args_304574, *[pp_opts_304575, debug_304576, extra_preargs_304577], **kwargs_304578)
        
        # Assigning a type to the variable 'cc_args' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'cc_args', _get_cc_args_call_result_304579)
        
        # Getting the type of 'objects' (line 569)
        objects_304580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 19), 'objects')
        # Testing the type of a for loop iterable (line 569)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 569, 8), objects_304580)
        # Getting the type of the for loop variable (line 569)
        for_loop_var_304581 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 569, 8), objects_304580)
        # Assigning a type to the variable 'obj' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'obj', for_loop_var_304581)
        # SSA begins for a for statement (line 569)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 570)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Tuple (line 571):
        
        # Assigning a Subscript to a Name (line 571):
        
        # Obtaining the type of the subscript
        int_304582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 16), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'obj' (line 571)
        obj_304583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 33), 'obj')
        # Getting the type of 'build' (line 571)
        build_304584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 27), 'build')
        # Obtaining the member '__getitem__' of a type (line 571)
        getitem___304585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 27), build_304584, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 571)
        subscript_call_result_304586 = invoke(stypy.reporting.localization.Localization(__file__, 571, 27), getitem___304585, obj_304583)
        
        # Obtaining the member '__getitem__' of a type (line 571)
        getitem___304587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 16), subscript_call_result_304586, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 571)
        subscript_call_result_304588 = invoke(stypy.reporting.localization.Localization(__file__, 571, 16), getitem___304587, int_304582)
        
        # Assigning a type to the variable 'tuple_var_assignment_303817' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 16), 'tuple_var_assignment_303817', subscript_call_result_304588)
        
        # Assigning a Subscript to a Name (line 571):
        
        # Obtaining the type of the subscript
        int_304589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 16), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'obj' (line 571)
        obj_304590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 33), 'obj')
        # Getting the type of 'build' (line 571)
        build_304591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 27), 'build')
        # Obtaining the member '__getitem__' of a type (line 571)
        getitem___304592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 27), build_304591, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 571)
        subscript_call_result_304593 = invoke(stypy.reporting.localization.Localization(__file__, 571, 27), getitem___304592, obj_304590)
        
        # Obtaining the member '__getitem__' of a type (line 571)
        getitem___304594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 16), subscript_call_result_304593, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 571)
        subscript_call_result_304595 = invoke(stypy.reporting.localization.Localization(__file__, 571, 16), getitem___304594, int_304589)
        
        # Assigning a type to the variable 'tuple_var_assignment_303818' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 16), 'tuple_var_assignment_303818', subscript_call_result_304595)
        
        # Assigning a Name to a Name (line 571):
        # Getting the type of 'tuple_var_assignment_303817' (line 571)
        tuple_var_assignment_303817_304596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 16), 'tuple_var_assignment_303817')
        # Assigning a type to the variable 'src' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 16), 'src', tuple_var_assignment_303817_304596)
        
        # Assigning a Name to a Name (line 571):
        # Getting the type of 'tuple_var_assignment_303818' (line 571)
        tuple_var_assignment_303818_304597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 16), 'tuple_var_assignment_303818')
        # Assigning a type to the variable 'ext' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 21), 'ext', tuple_var_assignment_303818_304597)
        # SSA branch for the except part of a try statement (line 570)
        # SSA branch for the except 'KeyError' branch of a try statement (line 570)
        module_type_store.open_ssa_branch('except')
        # SSA join for try-except statement (line 570)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to _compile(...): (line 574)
        # Processing the call arguments (line 574)
        # Getting the type of 'obj' (line 574)
        obj_304600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 26), 'obj', False)
        # Getting the type of 'src' (line 574)
        src_304601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 31), 'src', False)
        # Getting the type of 'ext' (line 574)
        ext_304602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 36), 'ext', False)
        # Getting the type of 'cc_args' (line 574)
        cc_args_304603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 41), 'cc_args', False)
        # Getting the type of 'extra_postargs' (line 574)
        extra_postargs_304604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 50), 'extra_postargs', False)
        # Getting the type of 'pp_opts' (line 574)
        pp_opts_304605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 66), 'pp_opts', False)
        # Processing the call keyword arguments (line 574)
        kwargs_304606 = {}
        # Getting the type of 'self' (line 574)
        self_304598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'self', False)
        # Obtaining the member '_compile' of a type (line 574)
        _compile_304599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 12), self_304598, '_compile')
        # Calling _compile(args, kwargs) (line 574)
        _compile_call_result_304607 = invoke(stypy.reporting.localization.Localization(__file__, 574, 12), _compile_304599, *[obj_304600, src_304601, ext_304602, cc_args_304603, extra_postargs_304604, pp_opts_304605], **kwargs_304606)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'objects' (line 577)
        objects_304608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 15), 'objects')
        # Assigning a type to the variable 'stypy_return_type' (line 577)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'stypy_return_type', objects_304608)
        
        # ################# End of 'compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'compile' in the type store
        # Getting the type of 'stypy_return_type' (line 510)
        stypy_return_type_304609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304609)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'compile'
        return stypy_return_type_304609


    @norecursion
    def _compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_compile'
        module_type_store = module_type_store.open_function_context('_compile', 579, 4, False)
        # Assigning a type to the variable 'self' (line 580)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler._compile.__dict__.__setitem__('stypy_localization', localization)
        CCompiler._compile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler._compile.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler._compile.__dict__.__setitem__('stypy_function_name', 'CCompiler._compile')
        CCompiler._compile.__dict__.__setitem__('stypy_param_names_list', ['obj', 'src', 'ext', 'cc_args', 'extra_postargs', 'pp_opts'])
        CCompiler._compile.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler._compile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler._compile.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler._compile.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler._compile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler._compile.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler._compile', ['obj', 'src', 'ext', 'cc_args', 'extra_postargs', 'pp_opts'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_compile', localization, ['obj', 'src', 'ext', 'cc_args', 'extra_postargs', 'pp_opts'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_compile(...)' code ##################

        str_304610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 580, 8), 'str', "Compile 'src' to product 'obj'.")
        pass
        
        # ################# End of '_compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_compile' in the type store
        # Getting the type of 'stypy_return_type' (line 579)
        stypy_return_type_304611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304611)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_compile'
        return stypy_return_type_304611


    @norecursion
    def create_static_lib(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 586)
        None_304612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 68), 'None')
        int_304613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 32), 'int')
        # Getting the type of 'None' (line 587)
        None_304614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 47), 'None')
        defaults = [None_304612, int_304613, None_304614]
        # Create a new context for function 'create_static_lib'
        module_type_store = module_type_store.open_function_context('create_static_lib', 586, 4, False)
        # Assigning a type to the variable 'self' (line 587)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.create_static_lib.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.create_static_lib.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.create_static_lib.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.create_static_lib.__dict__.__setitem__('stypy_function_name', 'CCompiler.create_static_lib')
        CCompiler.create_static_lib.__dict__.__setitem__('stypy_param_names_list', ['objects', 'output_libname', 'output_dir', 'debug', 'target_lang'])
        CCompiler.create_static_lib.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.create_static_lib.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.create_static_lib.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.create_static_lib.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.create_static_lib.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.create_static_lib.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.create_static_lib', ['objects', 'output_libname', 'output_dir', 'debug', 'target_lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_static_lib', localization, ['objects', 'output_libname', 'output_dir', 'debug', 'target_lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_static_lib(...)' code ##################

        str_304615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, (-1)), 'str', 'Link a bunch of stuff together to create a static library file.\n        The "bunch of stuff" consists of the list of object files supplied\n        as \'objects\', the extra object files supplied to\n        \'add_link_object()\' and/or \'set_link_objects()\', the libraries\n        supplied to \'add_library()\' and/or \'set_libraries()\', and the\n        libraries supplied as \'libraries\' (if any).\n\n        \'output_libname\' should be a library name, not a filename; the\n        filename will be inferred from the library name.  \'output_dir\' is\n        the directory where the library file will be put.\n\n        \'debug\' is a boolean; if true, debugging information will be\n        included in the library (note that on most platforms, it is the\n        compile step where this matters: the \'debug\' flag is included here\n        just for consistency).\n\n        \'target_lang\' is the target language for which the given objects\n        are being compiled. This allows specific linkage time treatment of\n        certain languages.\n\n        Raises LibError on failure.\n        ')
        pass
        
        # ################# End of 'create_static_lib(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_static_lib' in the type store
        # Getting the type of 'stypy_return_type' (line 586)
        stypy_return_type_304616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304616)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_static_lib'
        return stypy_return_type_304616

    
    # Assigning a Str to a Name (line 613):
    
    # Assigning a Str to a Name (line 614):
    
    # Assigning a Str to a Name (line 615):

    @norecursion
    def link(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 617)
        None_304617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 69), 'None')
        # Getting the type of 'None' (line 618)
        None_304618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 23), 'None')
        # Getting the type of 'None' (line 618)
        None_304619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 42), 'None')
        # Getting the type of 'None' (line 618)
        None_304620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 69), 'None')
        # Getting the type of 'None' (line 619)
        None_304621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 28), 'None')
        int_304622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 40), 'int')
        # Getting the type of 'None' (line 619)
        None_304623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 57), 'None')
        # Getting the type of 'None' (line 620)
        None_304624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 28), 'None')
        # Getting the type of 'None' (line 620)
        None_304625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 45), 'None')
        # Getting the type of 'None' (line 620)
        None_304626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 63), 'None')
        defaults = [None_304617, None_304618, None_304619, None_304620, None_304621, int_304622, None_304623, None_304624, None_304625, None_304626]
        # Create a new context for function 'link'
        module_type_store = module_type_store.open_function_context('link', 617, 4, False)
        # Assigning a type to the variable 'self' (line 618)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.link.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.link.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.link.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.link.__dict__.__setitem__('stypy_function_name', 'CCompiler.link')
        CCompiler.link.__dict__.__setitem__('stypy_param_names_list', ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'])
        CCompiler.link.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.link.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.link.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.link.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.link.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.link.__dict__.__setitem__('stypy_declared_arg_number', 14)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.link', ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'link', localization, ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'link(...)' code ##################

        str_304627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, (-1)), 'str', 'Link a bunch of stuff together to create an executable or\n        shared library file.\n\n        The "bunch of stuff" consists of the list of object files supplied\n        as \'objects\'.  \'output_filename\' should be a filename.  If\n        \'output_dir\' is supplied, \'output_filename\' is relative to it\n        (i.e. \'output_filename\' can provide directory components if\n        needed).\n\n        \'libraries\' is a list of libraries to link against.  These are\n        library names, not filenames, since they\'re translated into\n        filenames in a platform-specific way (eg. "foo" becomes "libfoo.a"\n        on Unix and "foo.lib" on DOS/Windows).  However, they can include a\n        directory component, which means the linker will look in that\n        specific directory rather than searching all the normal locations.\n\n        \'library_dirs\', if supplied, should be a list of directories to\n        search for libraries that were specified as bare library names\n        (ie. no directory component).  These are on top of the system\n        default and those supplied to \'add_library_dir()\' and/or\n        \'set_library_dirs()\'.  \'runtime_library_dirs\' is a list of\n        directories that will be embedded into the shared library and used\n        to search for other shared libraries that *it* depends on at\n        run-time.  (This may only be relevant on Unix.)\n\n        \'export_symbols\' is a list of symbols that the shared library will\n        export.  (This appears to be relevant only on Windows.)\n\n        \'debug\' is as for \'compile()\' and \'create_static_lib()\', with the\n        slight distinction that it actually matters on most platforms (as\n        opposed to \'create_static_lib()\', which includes a \'debug\' flag\n        mostly for form\'s sake).\n\n        \'extra_preargs\' and \'extra_postargs\' are as for \'compile()\' (except\n        of course that they supply command-line arguments for the\n        particular linker being used).\n\n        \'target_lang\' is the target language for which the given objects\n        are being compiled. This allows specific linkage time treatment of\n        certain languages.\n\n        Raises LinkError on failure.\n        ')
        # Getting the type of 'NotImplementedError' (line 664)
        NotImplementedError_304628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 664, 8), NotImplementedError_304628, 'raise parameter', BaseException)
        
        # ################# End of 'link(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'link' in the type store
        # Getting the type of 'stypy_return_type' (line 617)
        stypy_return_type_304629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304629)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'link'
        return stypy_return_type_304629


    @norecursion
    def link_shared_lib(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 669)
        None_304630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 66), 'None')
        # Getting the type of 'None' (line 670)
        None_304631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 34), 'None')
        # Getting the type of 'None' (line 670)
        None_304632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 53), 'None')
        # Getting the type of 'None' (line 671)
        None_304633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 45), 'None')
        # Getting the type of 'None' (line 671)
        None_304634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 66), 'None')
        int_304635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 30), 'int')
        # Getting the type of 'None' (line 672)
        None_304636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 47), 'None')
        # Getting the type of 'None' (line 672)
        None_304637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 68), 'None')
        # Getting the type of 'None' (line 673)
        None_304638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 35), 'None')
        # Getting the type of 'None' (line 673)
        None_304639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 53), 'None')
        defaults = [None_304630, None_304631, None_304632, None_304633, None_304634, int_304635, None_304636, None_304637, None_304638, None_304639]
        # Create a new context for function 'link_shared_lib'
        module_type_store = module_type_store.open_function_context('link_shared_lib', 669, 4, False)
        # Assigning a type to the variable 'self' (line 670)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.link_shared_lib.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.link_shared_lib.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.link_shared_lib.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.link_shared_lib.__dict__.__setitem__('stypy_function_name', 'CCompiler.link_shared_lib')
        CCompiler.link_shared_lib.__dict__.__setitem__('stypy_param_names_list', ['objects', 'output_libname', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'])
        CCompiler.link_shared_lib.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.link_shared_lib.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.link_shared_lib.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.link_shared_lib.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.link_shared_lib.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.link_shared_lib.__dict__.__setitem__('stypy_declared_arg_number', 13)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.link_shared_lib', ['objects', 'output_libname', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'link_shared_lib', localization, ['objects', 'output_libname', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'link_shared_lib(...)' code ##################

        
        # Call to link(...): (line 674)
        # Processing the call arguments (line 674)
        # Getting the type of 'CCompiler' (line 674)
        CCompiler_304642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 18), 'CCompiler', False)
        # Obtaining the member 'SHARED_LIBRARY' of a type (line 674)
        SHARED_LIBRARY_304643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 18), CCompiler_304642, 'SHARED_LIBRARY')
        # Getting the type of 'objects' (line 674)
        objects_304644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 44), 'objects', False)
        
        # Call to library_filename(...): (line 675)
        # Processing the call arguments (line 675)
        # Getting the type of 'output_libname' (line 675)
        output_libname_304647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 40), 'output_libname', False)
        # Processing the call keyword arguments (line 675)
        str_304648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 65), 'str', 'shared')
        keyword_304649 = str_304648
        kwargs_304650 = {'lib_type': keyword_304649}
        # Getting the type of 'self' (line 675)
        self_304645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 18), 'self', False)
        # Obtaining the member 'library_filename' of a type (line 675)
        library_filename_304646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 18), self_304645, 'library_filename')
        # Calling library_filename(args, kwargs) (line 675)
        library_filename_call_result_304651 = invoke(stypy.reporting.localization.Localization(__file__, 675, 18), library_filename_304646, *[output_libname_304647], **kwargs_304650)
        
        # Getting the type of 'output_dir' (line 676)
        output_dir_304652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 18), 'output_dir', False)
        # Getting the type of 'libraries' (line 677)
        libraries_304653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 18), 'libraries', False)
        # Getting the type of 'library_dirs' (line 677)
        library_dirs_304654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 29), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 677)
        runtime_library_dirs_304655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 43), 'runtime_library_dirs', False)
        # Getting the type of 'export_symbols' (line 678)
        export_symbols_304656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 18), 'export_symbols', False)
        # Getting the type of 'debug' (line 678)
        debug_304657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 34), 'debug', False)
        # Getting the type of 'extra_preargs' (line 679)
        extra_preargs_304658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 18), 'extra_preargs', False)
        # Getting the type of 'extra_postargs' (line 679)
        extra_postargs_304659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 33), 'extra_postargs', False)
        # Getting the type of 'build_temp' (line 679)
        build_temp_304660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 49), 'build_temp', False)
        # Getting the type of 'target_lang' (line 679)
        target_lang_304661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 61), 'target_lang', False)
        # Processing the call keyword arguments (line 674)
        kwargs_304662 = {}
        # Getting the type of 'self' (line 674)
        self_304640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 674, 8), 'self', False)
        # Obtaining the member 'link' of a type (line 674)
        link_304641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 674, 8), self_304640, 'link')
        # Calling link(args, kwargs) (line 674)
        link_call_result_304663 = invoke(stypy.reporting.localization.Localization(__file__, 674, 8), link_304641, *[SHARED_LIBRARY_304643, objects_304644, library_filename_call_result_304651, output_dir_304652, libraries_304653, library_dirs_304654, runtime_library_dirs_304655, export_symbols_304656, debug_304657, extra_preargs_304658, extra_postargs_304659, build_temp_304660, target_lang_304661], **kwargs_304662)
        
        
        # ################# End of 'link_shared_lib(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'link_shared_lib' in the type store
        # Getting the type of 'stypy_return_type' (line 669)
        stypy_return_type_304664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304664)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'link_shared_lib'
        return stypy_return_type_304664


    @norecursion
    def link_shared_object(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 682)
        None_304665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 70), 'None')
        # Getting the type of 'None' (line 683)
        None_304666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 37), 'None')
        # Getting the type of 'None' (line 683)
        None_304667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 56), 'None')
        # Getting the type of 'None' (line 684)
        None_304668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 48), 'None')
        # Getting the type of 'None' (line 684)
        None_304669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 69), 'None')
        int_304670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 33), 'int')
        # Getting the type of 'None' (line 685)
        None_304671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 50), 'None')
        # Getting the type of 'None' (line 685)
        None_304672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 71), 'None')
        # Getting the type of 'None' (line 686)
        None_304673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 38), 'None')
        # Getting the type of 'None' (line 686)
        None_304674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 56), 'None')
        defaults = [None_304665, None_304666, None_304667, None_304668, None_304669, int_304670, None_304671, None_304672, None_304673, None_304674]
        # Create a new context for function 'link_shared_object'
        module_type_store = module_type_store.open_function_context('link_shared_object', 682, 4, False)
        # Assigning a type to the variable 'self' (line 683)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.link_shared_object.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.link_shared_object.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.link_shared_object.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.link_shared_object.__dict__.__setitem__('stypy_function_name', 'CCompiler.link_shared_object')
        CCompiler.link_shared_object.__dict__.__setitem__('stypy_param_names_list', ['objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'])
        CCompiler.link_shared_object.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.link_shared_object.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.link_shared_object.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.link_shared_object.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.link_shared_object.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.link_shared_object.__dict__.__setitem__('stypy_declared_arg_number', 13)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.link_shared_object', ['objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'link_shared_object', localization, ['objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'link_shared_object(...)' code ##################

        
        # Call to link(...): (line 687)
        # Processing the call arguments (line 687)
        # Getting the type of 'CCompiler' (line 687)
        CCompiler_304677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 18), 'CCompiler', False)
        # Obtaining the member 'SHARED_OBJECT' of a type (line 687)
        SHARED_OBJECT_304678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 18), CCompiler_304677, 'SHARED_OBJECT')
        # Getting the type of 'objects' (line 687)
        objects_304679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 43), 'objects', False)
        # Getting the type of 'output_filename' (line 688)
        output_filename_304680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 18), 'output_filename', False)
        # Getting the type of 'output_dir' (line 688)
        output_dir_304681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 35), 'output_dir', False)
        # Getting the type of 'libraries' (line 689)
        libraries_304682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 18), 'libraries', False)
        # Getting the type of 'library_dirs' (line 689)
        library_dirs_304683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 29), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 689)
        runtime_library_dirs_304684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 43), 'runtime_library_dirs', False)
        # Getting the type of 'export_symbols' (line 690)
        export_symbols_304685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 18), 'export_symbols', False)
        # Getting the type of 'debug' (line 690)
        debug_304686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 34), 'debug', False)
        # Getting the type of 'extra_preargs' (line 691)
        extra_preargs_304687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 18), 'extra_preargs', False)
        # Getting the type of 'extra_postargs' (line 691)
        extra_postargs_304688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 33), 'extra_postargs', False)
        # Getting the type of 'build_temp' (line 691)
        build_temp_304689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 49), 'build_temp', False)
        # Getting the type of 'target_lang' (line 691)
        target_lang_304690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 61), 'target_lang', False)
        # Processing the call keyword arguments (line 687)
        kwargs_304691 = {}
        # Getting the type of 'self' (line 687)
        self_304675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 8), 'self', False)
        # Obtaining the member 'link' of a type (line 687)
        link_304676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 8), self_304675, 'link')
        # Calling link(args, kwargs) (line 687)
        link_call_result_304692 = invoke(stypy.reporting.localization.Localization(__file__, 687, 8), link_304676, *[SHARED_OBJECT_304678, objects_304679, output_filename_304680, output_dir_304681, libraries_304682, library_dirs_304683, runtime_library_dirs_304684, export_symbols_304685, debug_304686, extra_preargs_304687, extra_postargs_304688, build_temp_304689, target_lang_304690], **kwargs_304691)
        
        
        # ################# End of 'link_shared_object(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'link_shared_object' in the type store
        # Getting the type of 'stypy_return_type' (line 682)
        stypy_return_type_304693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304693)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'link_shared_object'
        return stypy_return_type_304693


    @norecursion
    def link_executable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 693)
        None_304694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 67), 'None')
        # Getting the type of 'None' (line 694)
        None_304695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 34), 'None')
        # Getting the type of 'None' (line 694)
        None_304696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 53), 'None')
        # Getting the type of 'None' (line 695)
        None_304697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 45), 'None')
        int_304698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 57), 'int')
        # Getting the type of 'None' (line 695)
        None_304699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 74), 'None')
        # Getting the type of 'None' (line 696)
        None_304700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 39), 'None')
        # Getting the type of 'None' (line 696)
        None_304701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 57), 'None')
        defaults = [None_304694, None_304695, None_304696, None_304697, int_304698, None_304699, None_304700, None_304701]
        # Create a new context for function 'link_executable'
        module_type_store = module_type_store.open_function_context('link_executable', 693, 4, False)
        # Assigning a type to the variable 'self' (line 694)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.link_executable.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.link_executable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.link_executable.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.link_executable.__dict__.__setitem__('stypy_function_name', 'CCompiler.link_executable')
        CCompiler.link_executable.__dict__.__setitem__('stypy_param_names_list', ['objects', 'output_progname', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'debug', 'extra_preargs', 'extra_postargs', 'target_lang'])
        CCompiler.link_executable.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.link_executable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.link_executable.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.link_executable.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.link_executable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.link_executable.__dict__.__setitem__('stypy_declared_arg_number', 11)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.link_executable', ['objects', 'output_progname', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'debug', 'extra_preargs', 'extra_postargs', 'target_lang'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'link_executable', localization, ['objects', 'output_progname', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'debug', 'extra_preargs', 'extra_postargs', 'target_lang'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'link_executable(...)' code ##################

        
        # Call to link(...): (line 697)
        # Processing the call arguments (line 697)
        # Getting the type of 'CCompiler' (line 697)
        CCompiler_304704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 18), 'CCompiler', False)
        # Obtaining the member 'EXECUTABLE' of a type (line 697)
        EXECUTABLE_304705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 18), CCompiler_304704, 'EXECUTABLE')
        # Getting the type of 'objects' (line 697)
        objects_304706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 40), 'objects', False)
        
        # Call to executable_filename(...): (line 698)
        # Processing the call arguments (line 698)
        # Getting the type of 'output_progname' (line 698)
        output_progname_304709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 43), 'output_progname', False)
        # Processing the call keyword arguments (line 698)
        kwargs_304710 = {}
        # Getting the type of 'self' (line 698)
        self_304707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 18), 'self', False)
        # Obtaining the member 'executable_filename' of a type (line 698)
        executable_filename_304708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 18), self_304707, 'executable_filename')
        # Calling executable_filename(args, kwargs) (line 698)
        executable_filename_call_result_304711 = invoke(stypy.reporting.localization.Localization(__file__, 698, 18), executable_filename_304708, *[output_progname_304709], **kwargs_304710)
        
        # Getting the type of 'output_dir' (line 698)
        output_dir_304712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 61), 'output_dir', False)
        # Getting the type of 'libraries' (line 699)
        libraries_304713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 18), 'libraries', False)
        # Getting the type of 'library_dirs' (line 699)
        library_dirs_304714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 29), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 699)
        runtime_library_dirs_304715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 43), 'runtime_library_dirs', False)
        # Getting the type of 'None' (line 699)
        None_304716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 65), 'None', False)
        # Getting the type of 'debug' (line 700)
        debug_304717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 18), 'debug', False)
        # Getting the type of 'extra_preargs' (line 700)
        extra_preargs_304718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 25), 'extra_preargs', False)
        # Getting the type of 'extra_postargs' (line 700)
        extra_postargs_304719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 40), 'extra_postargs', False)
        # Getting the type of 'None' (line 700)
        None_304720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 56), 'None', False)
        # Getting the type of 'target_lang' (line 700)
        target_lang_304721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 62), 'target_lang', False)
        # Processing the call keyword arguments (line 697)
        kwargs_304722 = {}
        # Getting the type of 'self' (line 697)
        self_304702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 8), 'self', False)
        # Obtaining the member 'link' of a type (line 697)
        link_304703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 8), self_304702, 'link')
        # Calling link(args, kwargs) (line 697)
        link_call_result_304723 = invoke(stypy.reporting.localization.Localization(__file__, 697, 8), link_304703, *[EXECUTABLE_304705, objects_304706, executable_filename_call_result_304711, output_dir_304712, libraries_304713, library_dirs_304714, runtime_library_dirs_304715, None_304716, debug_304717, extra_preargs_304718, extra_postargs_304719, None_304720, target_lang_304721], **kwargs_304722)
        
        
        # ################# End of 'link_executable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'link_executable' in the type store
        # Getting the type of 'stypy_return_type' (line 693)
        stypy_return_type_304724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304724)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'link_executable'
        return stypy_return_type_304724


    @norecursion
    def library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'library_dir_option'
        module_type_store = module_type_store.open_function_context('library_dir_option', 708, 4, False)
        # Assigning a type to the variable 'self' (line 709)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.library_dir_option.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.library_dir_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.library_dir_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.library_dir_option.__dict__.__setitem__('stypy_function_name', 'CCompiler.library_dir_option')
        CCompiler.library_dir_option.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        CCompiler.library_dir_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.library_dir_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.library_dir_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.library_dir_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.library_dir_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.library_dir_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.library_dir_option', ['dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'library_dir_option', localization, ['dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'library_dir_option(...)' code ##################

        str_304725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, (-1)), 'str', "Return the compiler option to add 'dir' to the list of\n        directories searched for libraries.\n        ")
        # Getting the type of 'NotImplementedError' (line 712)
        NotImplementedError_304726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 712, 8), NotImplementedError_304726, 'raise parameter', BaseException)
        
        # ################# End of 'library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 708)
        stypy_return_type_304727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304727)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'library_dir_option'
        return stypy_return_type_304727


    @norecursion
    def runtime_library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runtime_library_dir_option'
        module_type_store = module_type_store.open_function_context('runtime_library_dir_option', 714, 4, False)
        # Assigning a type to the variable 'self' (line 715)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 715, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_function_name', 'CCompiler.runtime_library_dir_option')
        CCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        CCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.runtime_library_dir_option', ['dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'runtime_library_dir_option', localization, ['dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'runtime_library_dir_option(...)' code ##################

        str_304728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, (-1)), 'str', "Return the compiler option to add 'dir' to the list of\n        directories searched for runtime libraries.\n        ")
        # Getting the type of 'NotImplementedError' (line 718)
        NotImplementedError_304729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 718, 8), NotImplementedError_304729, 'raise parameter', BaseException)
        
        # ################# End of 'runtime_library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runtime_library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 714)
        stypy_return_type_304730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304730)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runtime_library_dir_option'
        return stypy_return_type_304730


    @norecursion
    def library_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'library_option'
        module_type_store = module_type_store.open_function_context('library_option', 720, 4, False)
        # Assigning a type to the variable 'self' (line 721)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.library_option.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.library_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.library_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.library_option.__dict__.__setitem__('stypy_function_name', 'CCompiler.library_option')
        CCompiler.library_option.__dict__.__setitem__('stypy_param_names_list', ['lib'])
        CCompiler.library_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.library_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.library_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.library_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.library_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.library_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.library_option', ['lib'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'library_option', localization, ['lib'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'library_option(...)' code ##################

        str_304731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 723, (-1)), 'str', "Return the compiler option to add 'lib' to the list of libraries\n        linked into the shared library or executable.\n        ")
        # Getting the type of 'NotImplementedError' (line 724)
        NotImplementedError_304732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 724, 8), NotImplementedError_304732, 'raise parameter', BaseException)
        
        # ################# End of 'library_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'library_option' in the type store
        # Getting the type of 'stypy_return_type' (line 720)
        stypy_return_type_304733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 720, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304733)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'library_option'
        return stypy_return_type_304733


    @norecursion
    def has_function(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 726)
        None_304734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 46), 'None')
        # Getting the type of 'None' (line 726)
        None_304735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 65), 'None')
        # Getting the type of 'None' (line 727)
        None_304736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 31), 'None')
        # Getting the type of 'None' (line 727)
        None_304737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 50), 'None')
        defaults = [None_304734, None_304735, None_304736, None_304737]
        # Create a new context for function 'has_function'
        module_type_store = module_type_store.open_function_context('has_function', 726, 4, False)
        # Assigning a type to the variable 'self' (line 727)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.has_function.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.has_function.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.has_function.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.has_function.__dict__.__setitem__('stypy_function_name', 'CCompiler.has_function')
        CCompiler.has_function.__dict__.__setitem__('stypy_param_names_list', ['funcname', 'includes', 'include_dirs', 'libraries', 'library_dirs'])
        CCompiler.has_function.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.has_function.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.has_function.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.has_function.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.has_function.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.has_function.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.has_function', ['funcname', 'includes', 'include_dirs', 'libraries', 'library_dirs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_function', localization, ['funcname', 'includes', 'include_dirs', 'libraries', 'library_dirs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_function(...)' code ##################

        str_304738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, (-1)), 'str', 'Return a boolean indicating whether funcname is supported on\n        the current platform.  The optional arguments can be used to\n        augment the compilation environment.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 736, 8))
        
        # 'import tempfile' statement (line 736)
        import tempfile

        import_module(stypy.reporting.localization.Localization(__file__, 736, 8), 'tempfile', tempfile, module_type_store)
        
        
        # Type idiom detected: calculating its left and rigth part (line 737)
        # Getting the type of 'includes' (line 737)
        includes_304739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 11), 'includes')
        # Getting the type of 'None' (line 737)
        None_304740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 23), 'None')
        
        (may_be_304741, more_types_in_union_304742) = may_be_none(includes_304739, None_304740)

        if may_be_304741:

            if more_types_in_union_304742:
                # Runtime conditional SSA (line 737)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Name (line 738):
            
            # Assigning a List to a Name (line 738):
            
            # Obtaining an instance of the builtin type 'list' (line 738)
            list_304743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 23), 'list')
            # Adding type elements to the builtin type 'list' instance (line 738)
            
            # Assigning a type to the variable 'includes' (line 738)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'includes', list_304743)

            if more_types_in_union_304742:
                # SSA join for if statement (line 737)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 739)
        # Getting the type of 'include_dirs' (line 739)
        include_dirs_304744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 11), 'include_dirs')
        # Getting the type of 'None' (line 739)
        None_304745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 27), 'None')
        
        (may_be_304746, more_types_in_union_304747) = may_be_none(include_dirs_304744, None_304745)

        if may_be_304746:

            if more_types_in_union_304747:
                # Runtime conditional SSA (line 739)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Name (line 740):
            
            # Assigning a List to a Name (line 740):
            
            # Obtaining an instance of the builtin type 'list' (line 740)
            list_304748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 740, 27), 'list')
            # Adding type elements to the builtin type 'list' instance (line 740)
            
            # Assigning a type to the variable 'include_dirs' (line 740)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 740, 12), 'include_dirs', list_304748)

            if more_types_in_union_304747:
                # SSA join for if statement (line 739)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 741)
        # Getting the type of 'libraries' (line 741)
        libraries_304749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 11), 'libraries')
        # Getting the type of 'None' (line 741)
        None_304750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 24), 'None')
        
        (may_be_304751, more_types_in_union_304752) = may_be_none(libraries_304749, None_304750)

        if may_be_304751:

            if more_types_in_union_304752:
                # Runtime conditional SSA (line 741)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Name (line 742):
            
            # Assigning a List to a Name (line 742):
            
            # Obtaining an instance of the builtin type 'list' (line 742)
            list_304753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 24), 'list')
            # Adding type elements to the builtin type 'list' instance (line 742)
            
            # Assigning a type to the variable 'libraries' (line 742)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 12), 'libraries', list_304753)

            if more_types_in_union_304752:
                # SSA join for if statement (line 741)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 743)
        # Getting the type of 'library_dirs' (line 743)
        library_dirs_304754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 11), 'library_dirs')
        # Getting the type of 'None' (line 743)
        None_304755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 27), 'None')
        
        (may_be_304756, more_types_in_union_304757) = may_be_none(library_dirs_304754, None_304755)

        if may_be_304756:

            if more_types_in_union_304757:
                # Runtime conditional SSA (line 743)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Name (line 744):
            
            # Assigning a List to a Name (line 744):
            
            # Obtaining an instance of the builtin type 'list' (line 744)
            list_304758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 744, 27), 'list')
            # Adding type elements to the builtin type 'list' instance (line 744)
            
            # Assigning a type to the variable 'library_dirs' (line 744)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 12), 'library_dirs', list_304758)

            if more_types_in_union_304757:
                # SSA join for if statement (line 743)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Tuple (line 745):
        
        # Assigning a Call to a Name:
        
        # Call to mkstemp(...): (line 745)
        # Processing the call arguments (line 745)
        str_304761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 37), 'str', '.c')
        # Getting the type of 'funcname' (line 745)
        funcname_304762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 43), 'funcname', False)
        # Processing the call keyword arguments (line 745)
        # Getting the type of 'True' (line 745)
        True_304763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 58), 'True', False)
        keyword_304764 = True_304763
        kwargs_304765 = {'text': keyword_304764}
        # Getting the type of 'tempfile' (line 745)
        tempfile_304759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 20), 'tempfile', False)
        # Obtaining the member 'mkstemp' of a type (line 745)
        mkstemp_304760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 20), tempfile_304759, 'mkstemp')
        # Calling mkstemp(args, kwargs) (line 745)
        mkstemp_call_result_304766 = invoke(stypy.reporting.localization.Localization(__file__, 745, 20), mkstemp_304760, *[str_304761, funcname_304762], **kwargs_304765)
        
        # Assigning a type to the variable 'call_assignment_303819' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'call_assignment_303819', mkstemp_call_result_304766)
        
        # Assigning a Call to a Name (line 745):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_304769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 8), 'int')
        # Processing the call keyword arguments
        kwargs_304770 = {}
        # Getting the type of 'call_assignment_303819' (line 745)
        call_assignment_303819_304767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'call_assignment_303819', False)
        # Obtaining the member '__getitem__' of a type (line 745)
        getitem___304768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 8), call_assignment_303819_304767, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_304771 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___304768, *[int_304769], **kwargs_304770)
        
        # Assigning a type to the variable 'call_assignment_303820' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'call_assignment_303820', getitem___call_result_304771)
        
        # Assigning a Name to a Name (line 745):
        # Getting the type of 'call_assignment_303820' (line 745)
        call_assignment_303820_304772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'call_assignment_303820')
        # Assigning a type to the variable 'fd' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'fd', call_assignment_303820_304772)
        
        # Assigning a Call to a Name (line 745):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_304775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 8), 'int')
        # Processing the call keyword arguments
        kwargs_304776 = {}
        # Getting the type of 'call_assignment_303819' (line 745)
        call_assignment_303819_304773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'call_assignment_303819', False)
        # Obtaining the member '__getitem__' of a type (line 745)
        getitem___304774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 8), call_assignment_303819_304773, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_304777 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___304774, *[int_304775], **kwargs_304776)
        
        # Assigning a type to the variable 'call_assignment_303821' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'call_assignment_303821', getitem___call_result_304777)
        
        # Assigning a Name to a Name (line 745):
        # Getting the type of 'call_assignment_303821' (line 745)
        call_assignment_303821_304778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 8), 'call_assignment_303821')
        # Assigning a type to the variable 'fname' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 12), 'fname', call_assignment_303821_304778)
        
        # Assigning a Call to a Name (line 746):
        
        # Assigning a Call to a Name (line 746):
        
        # Call to fdopen(...): (line 746)
        # Processing the call arguments (line 746)
        # Getting the type of 'fd' (line 746)
        fd_304781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 22), 'fd', False)
        str_304782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 746, 26), 'str', 'w')
        # Processing the call keyword arguments (line 746)
        kwargs_304783 = {}
        # Getting the type of 'os' (line 746)
        os_304779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 12), 'os', False)
        # Obtaining the member 'fdopen' of a type (line 746)
        fdopen_304780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 12), os_304779, 'fdopen')
        # Calling fdopen(args, kwargs) (line 746)
        fdopen_call_result_304784 = invoke(stypy.reporting.localization.Localization(__file__, 746, 12), fdopen_304780, *[fd_304781, str_304782], **kwargs_304783)
        
        # Assigning a type to the variable 'f' (line 746)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 746, 8), 'f', fdopen_call_result_304784)
        
        # Try-finally block (line 747)
        
        # Getting the type of 'includes' (line 748)
        includes_304785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 24), 'includes')
        # Testing the type of a for loop iterable (line 748)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 748, 12), includes_304785)
        # Getting the type of the for loop variable (line 748)
        for_loop_var_304786 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 748, 12), includes_304785)
        # Assigning a type to the variable 'incl' (line 748)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 12), 'incl', for_loop_var_304786)
        # SSA begins for a for statement (line 748)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to write(...): (line 749)
        # Processing the call arguments (line 749)
        str_304789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 24), 'str', '#include "%s"\n')
        # Getting the type of 'incl' (line 749)
        incl_304790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 48), 'incl', False)
        # Applying the binary operator '%' (line 749)
        result_mod_304791 = python_operator(stypy.reporting.localization.Localization(__file__, 749, 24), '%', str_304789, incl_304790)
        
        # Processing the call keyword arguments (line 749)
        kwargs_304792 = {}
        # Getting the type of 'f' (line 749)
        f_304787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 16), 'f', False)
        # Obtaining the member 'write' of a type (line 749)
        write_304788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 749, 16), f_304787, 'write')
        # Calling write(args, kwargs) (line 749)
        write_call_result_304793 = invoke(stypy.reporting.localization.Localization(__file__, 749, 16), write_304788, *[result_mod_304791], **kwargs_304792)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 750)
        # Processing the call arguments (line 750)
        str_304796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 754, (-1)), 'str', 'main (int argc, char **argv) {\n    %s();\n}\n')
        # Getting the type of 'funcname' (line 754)
        funcname_304797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 6), 'funcname', False)
        # Applying the binary operator '%' (line 754)
        result_mod_304798 = python_operator(stypy.reporting.localization.Localization(__file__, 754, (-1)), '%', str_304796, funcname_304797)
        
        # Processing the call keyword arguments (line 750)
        kwargs_304799 = {}
        # Getting the type of 'f' (line 750)
        f_304794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 750)
        write_304795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 12), f_304794, 'write')
        # Calling write(args, kwargs) (line 750)
        write_call_result_304800 = invoke(stypy.reporting.localization.Localization(__file__, 750, 12), write_304795, *[result_mod_304798], **kwargs_304799)
        
        
        # finally branch of the try-finally block (line 747)
        
        # Call to close(...): (line 756)
        # Processing the call keyword arguments (line 756)
        kwargs_304803 = {}
        # Getting the type of 'f' (line 756)
        f_304801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 756)
        close_304802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 12), f_304801, 'close')
        # Calling close(args, kwargs) (line 756)
        close_call_result_304804 = invoke(stypy.reporting.localization.Localization(__file__, 756, 12), close_304802, *[], **kwargs_304803)
        
        
        
        
        # SSA begins for try-except statement (line 757)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 758):
        
        # Assigning a Call to a Name (line 758):
        
        # Call to compile(...): (line 758)
        # Processing the call arguments (line 758)
        
        # Obtaining an instance of the builtin type 'list' (line 758)
        list_304807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 758, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 758)
        # Adding element type (line 758)
        # Getting the type of 'fname' (line 758)
        fname_304808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 36), 'fname', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 758, 35), list_304807, fname_304808)
        
        # Processing the call keyword arguments (line 758)
        # Getting the type of 'include_dirs' (line 758)
        include_dirs_304809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 57), 'include_dirs', False)
        keyword_304810 = include_dirs_304809
        kwargs_304811 = {'include_dirs': keyword_304810}
        # Getting the type of 'self' (line 758)
        self_304805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 22), 'self', False)
        # Obtaining the member 'compile' of a type (line 758)
        compile_304806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 22), self_304805, 'compile')
        # Calling compile(args, kwargs) (line 758)
        compile_call_result_304812 = invoke(stypy.reporting.localization.Localization(__file__, 758, 22), compile_304806, *[list_304807], **kwargs_304811)
        
        # Assigning a type to the variable 'objects' (line 758)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 12), 'objects', compile_call_result_304812)
        # SSA branch for the except part of a try statement (line 757)
        # SSA branch for the except 'CompileError' branch of a try statement (line 757)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'False' (line 760)
        False_304813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 12), 'stypy_return_type', False_304813)
        # SSA join for try-except statement (line 757)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 762)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to link_executable(...): (line 763)
        # Processing the call arguments (line 763)
        # Getting the type of 'objects' (line 763)
        objects_304816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 33), 'objects', False)
        str_304817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 42), 'str', 'a.out')
        # Processing the call keyword arguments (line 763)
        # Getting the type of 'libraries' (line 764)
        libraries_304818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 43), 'libraries', False)
        keyword_304819 = libraries_304818
        # Getting the type of 'library_dirs' (line 765)
        library_dirs_304820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 46), 'library_dirs', False)
        keyword_304821 = library_dirs_304820
        kwargs_304822 = {'libraries': keyword_304819, 'library_dirs': keyword_304821}
        # Getting the type of 'self' (line 763)
        self_304814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 12), 'self', False)
        # Obtaining the member 'link_executable' of a type (line 763)
        link_executable_304815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 763, 12), self_304814, 'link_executable')
        # Calling link_executable(args, kwargs) (line 763)
        link_executable_call_result_304823 = invoke(stypy.reporting.localization.Localization(__file__, 763, 12), link_executable_304815, *[objects_304816, str_304817], **kwargs_304822)
        
        # SSA branch for the except part of a try statement (line 762)
        # SSA branch for the except 'Tuple' branch of a try statement (line 762)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'False' (line 767)
        False_304824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 767)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'stypy_return_type', False_304824)
        # SSA join for try-except statement (line 762)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'True' (line 768)
        True_304825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 768)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'stypy_return_type', True_304825)
        
        # ################# End of 'has_function(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_function' in the type store
        # Getting the type of 'stypy_return_type' (line 726)
        stypy_return_type_304826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304826)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_function'
        return stypy_return_type_304826


    @norecursion
    def find_library_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_304827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 770, 50), 'int')
        defaults = [int_304827]
        # Create a new context for function 'find_library_file'
        module_type_store = module_type_store.open_function_context('find_library_file', 770, 4, False)
        # Assigning a type to the variable 'self' (line 771)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.find_library_file.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.find_library_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.find_library_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.find_library_file.__dict__.__setitem__('stypy_function_name', 'CCompiler.find_library_file')
        CCompiler.find_library_file.__dict__.__setitem__('stypy_param_names_list', ['dirs', 'lib', 'debug'])
        CCompiler.find_library_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.find_library_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.find_library_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.find_library_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.find_library_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.find_library_file.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.find_library_file', ['dirs', 'lib', 'debug'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_library_file', localization, ['dirs', 'lib', 'debug'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_library_file(...)' code ##################

        str_304828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, (-1)), 'str', "Search the specified list of directories for a static or shared\n        library file 'lib' and return the full path to that file.  If\n        'debug' true, look for a debugging version (if that makes sense on\n        the current platform).  Return None if 'lib' wasn't found in any of\n        the specified directories.\n        ")
        # Getting the type of 'NotImplementedError' (line 777)
        NotImplementedError_304829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 14), 'NotImplementedError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 777, 8), NotImplementedError_304829, 'raise parameter', BaseException)
        
        # ################# End of 'find_library_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_library_file' in the type store
        # Getting the type of 'stypy_return_type' (line 770)
        stypy_return_type_304830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304830)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_library_file'
        return stypy_return_type_304830


    @norecursion
    def object_filenames(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_304831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 59), 'int')
        str_304832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 73), 'str', '')
        defaults = [int_304831, str_304832]
        # Create a new context for function 'object_filenames'
        module_type_store = module_type_store.open_function_context('object_filenames', 813, 4, False)
        # Assigning a type to the variable 'self' (line 814)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.object_filenames.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.object_filenames.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.object_filenames.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.object_filenames.__dict__.__setitem__('stypy_function_name', 'CCompiler.object_filenames')
        CCompiler.object_filenames.__dict__.__setitem__('stypy_param_names_list', ['source_filenames', 'strip_dir', 'output_dir'])
        CCompiler.object_filenames.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.object_filenames.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.object_filenames.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.object_filenames.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.object_filenames.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.object_filenames.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.object_filenames', ['source_filenames', 'strip_dir', 'output_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'object_filenames', localization, ['source_filenames', 'strip_dir', 'output_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'object_filenames(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 814)
        # Getting the type of 'output_dir' (line 814)
        output_dir_304833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 11), 'output_dir')
        # Getting the type of 'None' (line 814)
        None_304834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 25), 'None')
        
        (may_be_304835, more_types_in_union_304836) = may_be_none(output_dir_304833, None_304834)

        if may_be_304835:

            if more_types_in_union_304836:
                # Runtime conditional SSA (line 814)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 815):
            
            # Assigning a Str to a Name (line 815):
            str_304837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 25), 'str', '')
            # Assigning a type to the variable 'output_dir' (line 815)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 12), 'output_dir', str_304837)

            if more_types_in_union_304836:
                # SSA join for if statement (line 814)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Name (line 816):
        
        # Assigning a List to a Name (line 816):
        
        # Obtaining an instance of the builtin type 'list' (line 816)
        list_304838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 816, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 816)
        
        # Assigning a type to the variable 'obj_names' (line 816)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 816, 8), 'obj_names', list_304838)
        
        # Getting the type of 'source_filenames' (line 817)
        source_filenames_304839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 24), 'source_filenames')
        # Testing the type of a for loop iterable (line 817)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 817, 8), source_filenames_304839)
        # Getting the type of the for loop variable (line 817)
        for_loop_var_304840 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 817, 8), source_filenames_304839)
        # Assigning a type to the variable 'src_name' (line 817)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 8), 'src_name', for_loop_var_304840)
        # SSA begins for a for statement (line 817)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 818):
        
        # Assigning a Call to a Name:
        
        # Call to splitext(...): (line 818)
        # Processing the call arguments (line 818)
        # Getting the type of 'src_name' (line 818)
        src_name_304844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 41), 'src_name', False)
        # Processing the call keyword arguments (line 818)
        kwargs_304845 = {}
        # Getting the type of 'os' (line 818)
        os_304841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 818)
        path_304842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 818, 24), os_304841, 'path')
        # Obtaining the member 'splitext' of a type (line 818)
        splitext_304843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 818, 24), path_304842, 'splitext')
        # Calling splitext(args, kwargs) (line 818)
        splitext_call_result_304846 = invoke(stypy.reporting.localization.Localization(__file__, 818, 24), splitext_304843, *[src_name_304844], **kwargs_304845)
        
        # Assigning a type to the variable 'call_assignment_303822' (line 818)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'call_assignment_303822', splitext_call_result_304846)
        
        # Assigning a Call to a Name (line 818):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_304849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 12), 'int')
        # Processing the call keyword arguments
        kwargs_304850 = {}
        # Getting the type of 'call_assignment_303822' (line 818)
        call_assignment_303822_304847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'call_assignment_303822', False)
        # Obtaining the member '__getitem__' of a type (line 818)
        getitem___304848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 818, 12), call_assignment_303822_304847, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_304851 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___304848, *[int_304849], **kwargs_304850)
        
        # Assigning a type to the variable 'call_assignment_303823' (line 818)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'call_assignment_303823', getitem___call_result_304851)
        
        # Assigning a Name to a Name (line 818):
        # Getting the type of 'call_assignment_303823' (line 818)
        call_assignment_303823_304852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'call_assignment_303823')
        # Assigning a type to the variable 'base' (line 818)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'base', call_assignment_303823_304852)
        
        # Assigning a Call to a Name (line 818):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_304855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 12), 'int')
        # Processing the call keyword arguments
        kwargs_304856 = {}
        # Getting the type of 'call_assignment_303822' (line 818)
        call_assignment_303822_304853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'call_assignment_303822', False)
        # Obtaining the member '__getitem__' of a type (line 818)
        getitem___304854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 818, 12), call_assignment_303822_304853, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_304857 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___304854, *[int_304855], **kwargs_304856)
        
        # Assigning a type to the variable 'call_assignment_303824' (line 818)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'call_assignment_303824', getitem___call_result_304857)
        
        # Assigning a Name to a Name (line 818):
        # Getting the type of 'call_assignment_303824' (line 818)
        call_assignment_303824_304858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'call_assignment_303824')
        # Assigning a type to the variable 'ext' (line 818)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 818, 18), 'ext', call_assignment_303824_304858)
        
        # Assigning a Subscript to a Name (line 819):
        
        # Assigning a Subscript to a Name (line 819):
        
        # Obtaining the type of the subscript
        int_304859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 819, 44), 'int')
        
        # Call to splitdrive(...): (line 819)
        # Processing the call arguments (line 819)
        # Getting the type of 'base' (line 819)
        base_304863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 38), 'base', False)
        # Processing the call keyword arguments (line 819)
        kwargs_304864 = {}
        # Getting the type of 'os' (line 819)
        os_304860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 819)
        path_304861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 819, 19), os_304860, 'path')
        # Obtaining the member 'splitdrive' of a type (line 819)
        splitdrive_304862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 819, 19), path_304861, 'splitdrive')
        # Calling splitdrive(args, kwargs) (line 819)
        splitdrive_call_result_304865 = invoke(stypy.reporting.localization.Localization(__file__, 819, 19), splitdrive_304862, *[base_304863], **kwargs_304864)
        
        # Obtaining the member '__getitem__' of a type (line 819)
        getitem___304866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 819, 19), splitdrive_call_result_304865, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 819)
        subscript_call_result_304867 = invoke(stypy.reporting.localization.Localization(__file__, 819, 19), getitem___304866, int_304859)
        
        # Assigning a type to the variable 'base' (line 819)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 12), 'base', subscript_call_result_304867)
        
        # Assigning a Subscript to a Name (line 820):
        
        # Assigning a Subscript to a Name (line 820):
        
        # Obtaining the type of the subscript
        
        # Call to isabs(...): (line 820)
        # Processing the call arguments (line 820)
        # Getting the type of 'base' (line 820)
        base_304871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 38), 'base', False)
        # Processing the call keyword arguments (line 820)
        kwargs_304872 = {}
        # Getting the type of 'os' (line 820)
        os_304868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 820)
        path_304869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 24), os_304868, 'path')
        # Obtaining the member 'isabs' of a type (line 820)
        isabs_304870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 24), path_304869, 'isabs')
        # Calling isabs(args, kwargs) (line 820)
        isabs_call_result_304873 = invoke(stypy.reporting.localization.Localization(__file__, 820, 24), isabs_304870, *[base_304871], **kwargs_304872)
        
        slice_304874 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 820, 19), isabs_call_result_304873, None, None)
        # Getting the type of 'base' (line 820)
        base_304875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 19), 'base')
        # Obtaining the member '__getitem__' of a type (line 820)
        getitem___304876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 820, 19), base_304875, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 820)
        subscript_call_result_304877 = invoke(stypy.reporting.localization.Localization(__file__, 820, 19), getitem___304876, slice_304874)
        
        # Assigning a type to the variable 'base' (line 820)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 820, 12), 'base', subscript_call_result_304877)
        
        
        # Getting the type of 'ext' (line 821)
        ext_304878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 15), 'ext')
        # Getting the type of 'self' (line 821)
        self_304879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 26), 'self')
        # Obtaining the member 'src_extensions' of a type (line 821)
        src_extensions_304880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 26), self_304879, 'src_extensions')
        # Applying the binary operator 'notin' (line 821)
        result_contains_304881 = python_operator(stypy.reporting.localization.Localization(__file__, 821, 15), 'notin', ext_304878, src_extensions_304880)
        
        # Testing the type of an if condition (line 821)
        if_condition_304882 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 821, 12), result_contains_304881)
        # Assigning a type to the variable 'if_condition_304882' (line 821)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 12), 'if_condition_304882', if_condition_304882)
        # SSA begins for if statement (line 821)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'UnknownFileError' (line 822)
        UnknownFileError_304883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 22), 'UnknownFileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 822, 16), UnknownFileError_304883, 'raise parameter', BaseException)
        # SSA join for if statement (line 821)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'strip_dir' (line 824)
        strip_dir_304884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 15), 'strip_dir')
        # Testing the type of an if condition (line 824)
        if_condition_304885 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 824, 12), strip_dir_304884)
        # Assigning a type to the variable 'if_condition_304885' (line 824)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 824, 12), 'if_condition_304885', if_condition_304885)
        # SSA begins for if statement (line 824)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 825):
        
        # Assigning a Call to a Name (line 825):
        
        # Call to basename(...): (line 825)
        # Processing the call arguments (line 825)
        # Getting the type of 'base' (line 825)
        base_304889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 40), 'base', False)
        # Processing the call keyword arguments (line 825)
        kwargs_304890 = {}
        # Getting the type of 'os' (line 825)
        os_304886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 825)
        path_304887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 23), os_304886, 'path')
        # Obtaining the member 'basename' of a type (line 825)
        basename_304888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 23), path_304887, 'basename')
        # Calling basename(args, kwargs) (line 825)
        basename_call_result_304891 = invoke(stypy.reporting.localization.Localization(__file__, 825, 23), basename_304888, *[base_304889], **kwargs_304890)
        
        # Assigning a type to the variable 'base' (line 825)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 825, 16), 'base', basename_call_result_304891)
        # SSA join for if statement (line 824)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 826)
        # Processing the call arguments (line 826)
        
        # Call to join(...): (line 826)
        # Processing the call arguments (line 826)
        # Getting the type of 'output_dir' (line 826)
        output_dir_304897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 42), 'output_dir', False)
        # Getting the type of 'base' (line 827)
        base_304898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 42), 'base', False)
        # Getting the type of 'self' (line 827)
        self_304899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 49), 'self', False)
        # Obtaining the member 'obj_extension' of a type (line 827)
        obj_extension_304900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 49), self_304899, 'obj_extension')
        # Applying the binary operator '+' (line 827)
        result_add_304901 = python_operator(stypy.reporting.localization.Localization(__file__, 827, 42), '+', base_304898, obj_extension_304900)
        
        # Processing the call keyword arguments (line 826)
        kwargs_304902 = {}
        # Getting the type of 'os' (line 826)
        os_304894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 29), 'os', False)
        # Obtaining the member 'path' of a type (line 826)
        path_304895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 29), os_304894, 'path')
        # Obtaining the member 'join' of a type (line 826)
        join_304896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 29), path_304895, 'join')
        # Calling join(args, kwargs) (line 826)
        join_call_result_304903 = invoke(stypy.reporting.localization.Localization(__file__, 826, 29), join_304896, *[output_dir_304897, result_add_304901], **kwargs_304902)
        
        # Processing the call keyword arguments (line 826)
        kwargs_304904 = {}
        # Getting the type of 'obj_names' (line 826)
        obj_names_304892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 12), 'obj_names', False)
        # Obtaining the member 'append' of a type (line 826)
        append_304893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 826, 12), obj_names_304892, 'append')
        # Calling append(args, kwargs) (line 826)
        append_call_result_304905 = invoke(stypy.reporting.localization.Localization(__file__, 826, 12), append_304893, *[join_call_result_304903], **kwargs_304904)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'obj_names' (line 828)
        obj_names_304906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 15), 'obj_names')
        # Assigning a type to the variable 'stypy_return_type' (line 828)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 8), 'stypy_return_type', obj_names_304906)
        
        # ################# End of 'object_filenames(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'object_filenames' in the type store
        # Getting the type of 'stypy_return_type' (line 813)
        stypy_return_type_304907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304907)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'object_filenames'
        return stypy_return_type_304907


    @norecursion
    def shared_object_filename(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_304908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 57), 'int')
        str_304909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 71), 'str', '')
        defaults = [int_304908, str_304909]
        # Create a new context for function 'shared_object_filename'
        module_type_store = module_type_store.open_function_context('shared_object_filename', 830, 4, False)
        # Assigning a type to the variable 'self' (line 831)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.shared_object_filename.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.shared_object_filename.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.shared_object_filename.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.shared_object_filename.__dict__.__setitem__('stypy_function_name', 'CCompiler.shared_object_filename')
        CCompiler.shared_object_filename.__dict__.__setitem__('stypy_param_names_list', ['basename', 'strip_dir', 'output_dir'])
        CCompiler.shared_object_filename.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.shared_object_filename.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.shared_object_filename.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.shared_object_filename.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.shared_object_filename.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.shared_object_filename.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.shared_object_filename', ['basename', 'strip_dir', 'output_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'shared_object_filename', localization, ['basename', 'strip_dir', 'output_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'shared_object_filename(...)' code ##################

        # Evaluating assert statement condition
        
        # Getting the type of 'output_dir' (line 831)
        output_dir_304910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 15), 'output_dir')
        # Getting the type of 'None' (line 831)
        None_304911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 33), 'None')
        # Applying the binary operator 'isnot' (line 831)
        result_is_not_304912 = python_operator(stypy.reporting.localization.Localization(__file__, 831, 15), 'isnot', output_dir_304910, None_304911)
        
        
        # Getting the type of 'strip_dir' (line 832)
        strip_dir_304913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 11), 'strip_dir')
        # Testing the type of an if condition (line 832)
        if_condition_304914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 832, 8), strip_dir_304913)
        # Assigning a type to the variable 'if_condition_304914' (line 832)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 8), 'if_condition_304914', if_condition_304914)
        # SSA begins for if statement (line 832)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 833):
        
        # Assigning a Call to a Name (line 833):
        
        # Call to basename(...): (line 833)
        # Processing the call arguments (line 833)
        # Getting the type of 'basename' (line 833)
        basename_304918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 41), 'basename', False)
        # Processing the call keyword arguments (line 833)
        kwargs_304919 = {}
        # Getting the type of 'os' (line 833)
        os_304915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 833)
        path_304916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 23), os_304915, 'path')
        # Obtaining the member 'basename' of a type (line 833)
        basename_304917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 833, 23), path_304916, 'basename')
        # Calling basename(args, kwargs) (line 833)
        basename_call_result_304920 = invoke(stypy.reporting.localization.Localization(__file__, 833, 23), basename_304917, *[basename_304918], **kwargs_304919)
        
        # Assigning a type to the variable 'basename' (line 833)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 12), 'basename', basename_call_result_304920)
        # SSA join for if statement (line 832)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to join(...): (line 834)
        # Processing the call arguments (line 834)
        # Getting the type of 'output_dir' (line 834)
        output_dir_304924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 28), 'output_dir', False)
        # Getting the type of 'basename' (line 834)
        basename_304925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 40), 'basename', False)
        # Getting the type of 'self' (line 834)
        self_304926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 51), 'self', False)
        # Obtaining the member 'shared_lib_extension' of a type (line 834)
        shared_lib_extension_304927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 51), self_304926, 'shared_lib_extension')
        # Applying the binary operator '+' (line 834)
        result_add_304928 = python_operator(stypy.reporting.localization.Localization(__file__, 834, 40), '+', basename_304925, shared_lib_extension_304927)
        
        # Processing the call keyword arguments (line 834)
        kwargs_304929 = {}
        # Getting the type of 'os' (line 834)
        os_304921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 834)
        path_304922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 15), os_304921, 'path')
        # Obtaining the member 'join' of a type (line 834)
        join_304923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 15), path_304922, 'join')
        # Calling join(args, kwargs) (line 834)
        join_call_result_304930 = invoke(stypy.reporting.localization.Localization(__file__, 834, 15), join_304923, *[output_dir_304924, result_add_304928], **kwargs_304929)
        
        # Assigning a type to the variable 'stypy_return_type' (line 834)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 8), 'stypy_return_type', join_call_result_304930)
        
        # ################# End of 'shared_object_filename(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'shared_object_filename' in the type store
        # Getting the type of 'stypy_return_type' (line 830)
        stypy_return_type_304931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304931)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'shared_object_filename'
        return stypy_return_type_304931


    @norecursion
    def executable_filename(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_304932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 54), 'int')
        str_304933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 68), 'str', '')
        defaults = [int_304932, str_304933]
        # Create a new context for function 'executable_filename'
        module_type_store = module_type_store.open_function_context('executable_filename', 836, 4, False)
        # Assigning a type to the variable 'self' (line 837)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.executable_filename.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.executable_filename.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.executable_filename.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.executable_filename.__dict__.__setitem__('stypy_function_name', 'CCompiler.executable_filename')
        CCompiler.executable_filename.__dict__.__setitem__('stypy_param_names_list', ['basename', 'strip_dir', 'output_dir'])
        CCompiler.executable_filename.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.executable_filename.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.executable_filename.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.executable_filename.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.executable_filename.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.executable_filename.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.executable_filename', ['basename', 'strip_dir', 'output_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'executable_filename', localization, ['basename', 'strip_dir', 'output_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'executable_filename(...)' code ##################

        # Evaluating assert statement condition
        
        # Getting the type of 'output_dir' (line 837)
        output_dir_304934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 15), 'output_dir')
        # Getting the type of 'None' (line 837)
        None_304935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 33), 'None')
        # Applying the binary operator 'isnot' (line 837)
        result_is_not_304936 = python_operator(stypy.reporting.localization.Localization(__file__, 837, 15), 'isnot', output_dir_304934, None_304935)
        
        
        # Getting the type of 'strip_dir' (line 838)
        strip_dir_304937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 11), 'strip_dir')
        # Testing the type of an if condition (line 838)
        if_condition_304938 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 838, 8), strip_dir_304937)
        # Assigning a type to the variable 'if_condition_304938' (line 838)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 838, 8), 'if_condition_304938', if_condition_304938)
        # SSA begins for if statement (line 838)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 839):
        
        # Assigning a Call to a Name (line 839):
        
        # Call to basename(...): (line 839)
        # Processing the call arguments (line 839)
        # Getting the type of 'basename' (line 839)
        basename_304942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 41), 'basename', False)
        # Processing the call keyword arguments (line 839)
        kwargs_304943 = {}
        # Getting the type of 'os' (line 839)
        os_304939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 839)
        path_304940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 23), os_304939, 'path')
        # Obtaining the member 'basename' of a type (line 839)
        basename_304941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 23), path_304940, 'basename')
        # Calling basename(args, kwargs) (line 839)
        basename_call_result_304944 = invoke(stypy.reporting.localization.Localization(__file__, 839, 23), basename_304941, *[basename_304942], **kwargs_304943)
        
        # Assigning a type to the variable 'basename' (line 839)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 12), 'basename', basename_call_result_304944)
        # SSA join for if statement (line 838)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to join(...): (line 840)
        # Processing the call arguments (line 840)
        # Getting the type of 'output_dir' (line 840)
        output_dir_304948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 28), 'output_dir', False)
        # Getting the type of 'basename' (line 840)
        basename_304949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 40), 'basename', False)
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 840)
        self_304950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 52), 'self', False)
        # Obtaining the member 'exe_extension' of a type (line 840)
        exe_extension_304951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 52), self_304950, 'exe_extension')
        str_304952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 840, 74), 'str', '')
        # Applying the binary operator 'or' (line 840)
        result_or_keyword_304953 = python_operator(stypy.reporting.localization.Localization(__file__, 840, 52), 'or', exe_extension_304951, str_304952)
        
        # Applying the binary operator '+' (line 840)
        result_add_304954 = python_operator(stypy.reporting.localization.Localization(__file__, 840, 40), '+', basename_304949, result_or_keyword_304953)
        
        # Processing the call keyword arguments (line 840)
        kwargs_304955 = {}
        # Getting the type of 'os' (line 840)
        os_304945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 840)
        path_304946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 15), os_304945, 'path')
        # Obtaining the member 'join' of a type (line 840)
        join_304947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 840, 15), path_304946, 'join')
        # Calling join(args, kwargs) (line 840)
        join_call_result_304956 = invoke(stypy.reporting.localization.Localization(__file__, 840, 15), join_304947, *[output_dir_304948, result_add_304954], **kwargs_304955)
        
        # Assigning a type to the variable 'stypy_return_type' (line 840)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 8), 'stypy_return_type', join_call_result_304956)
        
        # ################# End of 'executable_filename(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'executable_filename' in the type store
        # Getting the type of 'stypy_return_type' (line 836)
        stypy_return_type_304957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 836, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_304957)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'executable_filename'
        return stypy_return_type_304957


    @norecursion
    def library_filename(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_304958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 49), 'str', 'static')
        int_304959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 35), 'int')
        str_304960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 49), 'str', '')
        defaults = [str_304958, int_304959, str_304960]
        # Create a new context for function 'library_filename'
        module_type_store = module_type_store.open_function_context('library_filename', 842, 4, False)
        # Assigning a type to the variable 'self' (line 843)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.library_filename.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.library_filename.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.library_filename.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.library_filename.__dict__.__setitem__('stypy_function_name', 'CCompiler.library_filename')
        CCompiler.library_filename.__dict__.__setitem__('stypy_param_names_list', ['libname', 'lib_type', 'strip_dir', 'output_dir'])
        CCompiler.library_filename.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.library_filename.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.library_filename.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.library_filename.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.library_filename.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.library_filename.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.library_filename', ['libname', 'lib_type', 'strip_dir', 'output_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'library_filename', localization, ['libname', 'lib_type', 'strip_dir', 'output_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'library_filename(...)' code ##################

        # Evaluating assert statement condition
        
        # Getting the type of 'output_dir' (line 844)
        output_dir_304961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 15), 'output_dir')
        # Getting the type of 'None' (line 844)
        None_304962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 33), 'None')
        # Applying the binary operator 'isnot' (line 844)
        result_is_not_304963 = python_operator(stypy.reporting.localization.Localization(__file__, 844, 15), 'isnot', output_dir_304961, None_304962)
        
        
        
        # Getting the type of 'lib_type' (line 845)
        lib_type_304964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 11), 'lib_type')
        
        # Obtaining an instance of the builtin type 'tuple' (line 845)
        tuple_304965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 845)
        # Adding element type (line 845)
        str_304966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 28), 'str', 'static')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 845, 28), tuple_304965, str_304966)
        # Adding element type (line 845)
        str_304967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 38), 'str', 'shared')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 845, 28), tuple_304965, str_304967)
        # Adding element type (line 845)
        str_304968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 48), 'str', 'dylib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 845, 28), tuple_304965, str_304968)
        # Adding element type (line 845)
        str_304969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 57), 'str', 'xcode_stub')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 845, 28), tuple_304965, str_304969)
        
        # Applying the binary operator 'notin' (line 845)
        result_contains_304970 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 11), 'notin', lib_type_304964, tuple_304965)
        
        # Testing the type of an if condition (line 845)
        if_condition_304971 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 845, 8), result_contains_304970)
        # Assigning a type to the variable 'if_condition_304971' (line 845)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 845, 8), 'if_condition_304971', if_condition_304971)
        # SSA begins for if statement (line 845)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'ValueError' (line 846)
        ValueError_304972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 18), 'ValueError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 846, 12), ValueError_304972, 'raise parameter', BaseException)
        # SSA join for if statement (line 845)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 848):
        
        # Assigning a Call to a Name (line 848):
        
        # Call to getattr(...): (line 848)
        # Processing the call arguments (line 848)
        # Getting the type of 'self' (line 848)
        self_304974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 22), 'self', False)
        # Getting the type of 'lib_type' (line 848)
        lib_type_304975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 28), 'lib_type', False)
        str_304976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 39), 'str', '_lib_format')
        # Applying the binary operator '+' (line 848)
        result_add_304977 = python_operator(stypy.reporting.localization.Localization(__file__, 848, 28), '+', lib_type_304975, str_304976)
        
        # Processing the call keyword arguments (line 848)
        kwargs_304978 = {}
        # Getting the type of 'getattr' (line 848)
        getattr_304973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 14), 'getattr', False)
        # Calling getattr(args, kwargs) (line 848)
        getattr_call_result_304979 = invoke(stypy.reporting.localization.Localization(__file__, 848, 14), getattr_304973, *[self_304974, result_add_304977], **kwargs_304978)
        
        # Assigning a type to the variable 'fmt' (line 848)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 8), 'fmt', getattr_call_result_304979)
        
        # Assigning a Call to a Name (line 849):
        
        # Assigning a Call to a Name (line 849):
        
        # Call to getattr(...): (line 849)
        # Processing the call arguments (line 849)
        # Getting the type of 'self' (line 849)
        self_304981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 22), 'self', False)
        # Getting the type of 'lib_type' (line 849)
        lib_type_304982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 28), 'lib_type', False)
        str_304983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 849, 39), 'str', '_lib_extension')
        # Applying the binary operator '+' (line 849)
        result_add_304984 = python_operator(stypy.reporting.localization.Localization(__file__, 849, 28), '+', lib_type_304982, str_304983)
        
        # Processing the call keyword arguments (line 849)
        kwargs_304985 = {}
        # Getting the type of 'getattr' (line 849)
        getattr_304980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 14), 'getattr', False)
        # Calling getattr(args, kwargs) (line 849)
        getattr_call_result_304986 = invoke(stypy.reporting.localization.Localization(__file__, 849, 14), getattr_304980, *[self_304981, result_add_304984], **kwargs_304985)
        
        # Assigning a type to the variable 'ext' (line 849)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 849, 8), 'ext', getattr_call_result_304986)
        
        # Assigning a Call to a Tuple (line 851):
        
        # Assigning a Call to a Name:
        
        # Call to split(...): (line 851)
        # Processing the call arguments (line 851)
        # Getting the type of 'libname' (line 851)
        libname_304990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 35), 'libname', False)
        # Processing the call keyword arguments (line 851)
        kwargs_304991 = {}
        # Getting the type of 'os' (line 851)
        os_304987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 851)
        path_304988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 20), os_304987, 'path')
        # Obtaining the member 'split' of a type (line 851)
        split_304989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 20), path_304988, 'split')
        # Calling split(args, kwargs) (line 851)
        split_call_result_304992 = invoke(stypy.reporting.localization.Localization(__file__, 851, 20), split_304989, *[libname_304990], **kwargs_304991)
        
        # Assigning a type to the variable 'call_assignment_303825' (line 851)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 8), 'call_assignment_303825', split_call_result_304992)
        
        # Assigning a Call to a Name (line 851):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_304995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, 8), 'int')
        # Processing the call keyword arguments
        kwargs_304996 = {}
        # Getting the type of 'call_assignment_303825' (line 851)
        call_assignment_303825_304993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 8), 'call_assignment_303825', False)
        # Obtaining the member '__getitem__' of a type (line 851)
        getitem___304994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 8), call_assignment_303825_304993, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_304997 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___304994, *[int_304995], **kwargs_304996)
        
        # Assigning a type to the variable 'call_assignment_303826' (line 851)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 8), 'call_assignment_303826', getitem___call_result_304997)
        
        # Assigning a Name to a Name (line 851):
        # Getting the type of 'call_assignment_303826' (line 851)
        call_assignment_303826_304998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 8), 'call_assignment_303826')
        # Assigning a type to the variable 'dir' (line 851)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 8), 'dir', call_assignment_303826_304998)
        
        # Assigning a Call to a Name (line 851):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_305001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, 8), 'int')
        # Processing the call keyword arguments
        kwargs_305002 = {}
        # Getting the type of 'call_assignment_303825' (line 851)
        call_assignment_303825_304999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 8), 'call_assignment_303825', False)
        # Obtaining the member '__getitem__' of a type (line 851)
        getitem___305000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 8), call_assignment_303825_304999, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_305003 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___305000, *[int_305001], **kwargs_305002)
        
        # Assigning a type to the variable 'call_assignment_303827' (line 851)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 8), 'call_assignment_303827', getitem___call_result_305003)
        
        # Assigning a Name to a Name (line 851):
        # Getting the type of 'call_assignment_303827' (line 851)
        call_assignment_303827_305004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 8), 'call_assignment_303827')
        # Assigning a type to the variable 'base' (line 851)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 13), 'base', call_assignment_303827_305004)
        
        # Assigning a BinOp to a Name (line 852):
        
        # Assigning a BinOp to a Name (line 852):
        # Getting the type of 'fmt' (line 852)
        fmt_305005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 19), 'fmt')
        
        # Obtaining an instance of the builtin type 'tuple' (line 852)
        tuple_305006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 852)
        # Adding element type (line 852)
        # Getting the type of 'base' (line 852)
        base_305007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 26), 'base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 852, 26), tuple_305006, base_305007)
        # Adding element type (line 852)
        # Getting the type of 'ext' (line 852)
        ext_305008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 32), 'ext')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 852, 26), tuple_305006, ext_305008)
        
        # Applying the binary operator '%' (line 852)
        result_mod_305009 = python_operator(stypy.reporting.localization.Localization(__file__, 852, 19), '%', fmt_305005, tuple_305006)
        
        # Assigning a type to the variable 'filename' (line 852)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 8), 'filename', result_mod_305009)
        
        # Getting the type of 'strip_dir' (line 853)
        strip_dir_305010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 11), 'strip_dir')
        # Testing the type of an if condition (line 853)
        if_condition_305011 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 853, 8), strip_dir_305010)
        # Assigning a type to the variable 'if_condition_305011' (line 853)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 853, 8), 'if_condition_305011', if_condition_305011)
        # SSA begins for if statement (line 853)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 854):
        
        # Assigning a Str to a Name (line 854):
        str_305012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 854, 18), 'str', '')
        # Assigning a type to the variable 'dir' (line 854)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 12), 'dir', str_305012)
        # SSA join for if statement (line 853)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to join(...): (line 856)
        # Processing the call arguments (line 856)
        # Getting the type of 'output_dir' (line 856)
        output_dir_305016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 28), 'output_dir', False)
        # Getting the type of 'dir' (line 856)
        dir_305017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 40), 'dir', False)
        # Getting the type of 'filename' (line 856)
        filename_305018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 45), 'filename', False)
        # Processing the call keyword arguments (line 856)
        kwargs_305019 = {}
        # Getting the type of 'os' (line 856)
        os_305013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 856)
        path_305014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 856, 15), os_305013, 'path')
        # Obtaining the member 'join' of a type (line 856)
        join_305015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 856, 15), path_305014, 'join')
        # Calling join(args, kwargs) (line 856)
        join_call_result_305020 = invoke(stypy.reporting.localization.Localization(__file__, 856, 15), join_305015, *[output_dir_305016, dir_305017, filename_305018], **kwargs_305019)
        
        # Assigning a type to the variable 'stypy_return_type' (line 856)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 8), 'stypy_return_type', join_call_result_305020)
        
        # ################# End of 'library_filename(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'library_filename' in the type store
        # Getting the type of 'stypy_return_type' (line 842)
        stypy_return_type_305021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305021)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'library_filename'
        return stypy_return_type_305021


    @norecursion
    def announce(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_305022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 861, 34), 'int')
        defaults = [int_305022]
        # Create a new context for function 'announce'
        module_type_store = module_type_store.open_function_context('announce', 861, 4, False)
        # Assigning a type to the variable 'self' (line 862)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 862, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.announce.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.announce.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.announce.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.announce.__dict__.__setitem__('stypy_function_name', 'CCompiler.announce')
        CCompiler.announce.__dict__.__setitem__('stypy_param_names_list', ['msg', 'level'])
        CCompiler.announce.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.announce.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.announce.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.announce.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.announce.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.announce.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.announce', ['msg', 'level'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'announce', localization, ['msg', 'level'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'announce(...)' code ##################

        
        # Call to debug(...): (line 862)
        # Processing the call arguments (line 862)
        # Getting the type of 'msg' (line 862)
        msg_305025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 18), 'msg', False)
        # Processing the call keyword arguments (line 862)
        kwargs_305026 = {}
        # Getting the type of 'log' (line 862)
        log_305023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 8), 'log', False)
        # Obtaining the member 'debug' of a type (line 862)
        debug_305024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 862, 8), log_305023, 'debug')
        # Calling debug(args, kwargs) (line 862)
        debug_call_result_305027 = invoke(stypy.reporting.localization.Localization(__file__, 862, 8), debug_305024, *[msg_305025], **kwargs_305026)
        
        
        # ################# End of 'announce(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'announce' in the type store
        # Getting the type of 'stypy_return_type' (line 861)
        stypy_return_type_305028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305028)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'announce'
        return stypy_return_type_305028


    @norecursion
    def debug_print(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'debug_print'
        module_type_store = module_type_store.open_function_context('debug_print', 864, 4, False)
        # Assigning a type to the variable 'self' (line 865)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.debug_print.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.debug_print.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.debug_print.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.debug_print.__dict__.__setitem__('stypy_function_name', 'CCompiler.debug_print')
        CCompiler.debug_print.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        CCompiler.debug_print.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.debug_print.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.debug_print.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.debug_print.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.debug_print.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.debug_print.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.debug_print', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'debug_print', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'debug_print(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 865, 8))
        
        # 'from distutils.debug import DEBUG' statement (line 865)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/')
        import_305029 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 865, 8), 'distutils.debug')

        if (type(import_305029) is not StypyTypeError):

            if (import_305029 != 'pyd_module'):
                __import__(import_305029)
                sys_modules_305030 = sys.modules[import_305029]
                import_from_module(stypy.reporting.localization.Localization(__file__, 865, 8), 'distutils.debug', sys_modules_305030.module_type_store, module_type_store, ['DEBUG'])
                nest_module(stypy.reporting.localization.Localization(__file__, 865, 8), __file__, sys_modules_305030, sys_modules_305030.module_type_store, module_type_store)
            else:
                from distutils.debug import DEBUG

                import_from_module(stypy.reporting.localization.Localization(__file__, 865, 8), 'distutils.debug', None, module_type_store, ['DEBUG'], [DEBUG])

        else:
            # Assigning a type to the variable 'distutils.debug' (line 865)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 8), 'distutils.debug', import_305029)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
        
        
        # Getting the type of 'DEBUG' (line 866)
        DEBUG_305031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 866, 11), 'DEBUG')
        # Testing the type of an if condition (line 866)
        if_condition_305032 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 866, 8), DEBUG_305031)
        # Assigning a type to the variable 'if_condition_305032' (line 866)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 866, 8), 'if_condition_305032', if_condition_305032)
        # SSA begins for if statement (line 866)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'msg' (line 867)
        msg_305033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 18), 'msg')
        # SSA join for if statement (line 866)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'debug_print(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'debug_print' in the type store
        # Getting the type of 'stypy_return_type' (line 864)
        stypy_return_type_305034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 864, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305034)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'debug_print'
        return stypy_return_type_305034


    @norecursion
    def warn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'warn'
        module_type_store = module_type_store.open_function_context('warn', 869, 4, False)
        # Assigning a type to the variable 'self' (line 870)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.warn.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.warn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.warn.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.warn.__dict__.__setitem__('stypy_function_name', 'CCompiler.warn')
        CCompiler.warn.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        CCompiler.warn.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.warn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.warn.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.warn.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.warn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.warn.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.warn', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'warn', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'warn(...)' code ##################

        
        # Call to write(...): (line 870)
        # Processing the call arguments (line 870)
        str_305038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 870, 25), 'str', 'warning: %s\n')
        # Getting the type of 'msg' (line 870)
        msg_305039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 43), 'msg', False)
        # Applying the binary operator '%' (line 870)
        result_mod_305040 = python_operator(stypy.reporting.localization.Localization(__file__, 870, 25), '%', str_305038, msg_305039)
        
        # Processing the call keyword arguments (line 870)
        kwargs_305041 = {}
        # Getting the type of 'sys' (line 870)
        sys_305035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 8), 'sys', False)
        # Obtaining the member 'stderr' of a type (line 870)
        stderr_305036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 8), sys_305035, 'stderr')
        # Obtaining the member 'write' of a type (line 870)
        write_305037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 8), stderr_305036, 'write')
        # Calling write(args, kwargs) (line 870)
        write_call_result_305042 = invoke(stypy.reporting.localization.Localization(__file__, 870, 8), write_305037, *[result_mod_305040], **kwargs_305041)
        
        
        # ################# End of 'warn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'warn' in the type store
        # Getting the type of 'stypy_return_type' (line 869)
        stypy_return_type_305043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305043)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'warn'
        return stypy_return_type_305043


    @norecursion
    def execute(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 872)
        None_305044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 38), 'None')
        int_305045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 50), 'int')
        defaults = [None_305044, int_305045]
        # Create a new context for function 'execute'
        module_type_store = module_type_store.open_function_context('execute', 872, 4, False)
        # Assigning a type to the variable 'self' (line 873)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 873, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.execute.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.execute.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.execute.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.execute.__dict__.__setitem__('stypy_function_name', 'CCompiler.execute')
        CCompiler.execute.__dict__.__setitem__('stypy_param_names_list', ['func', 'args', 'msg', 'level'])
        CCompiler.execute.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.execute.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.execute.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.execute.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.execute.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.execute.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.execute', ['func', 'args', 'msg', 'level'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'execute', localization, ['func', 'args', 'msg', 'level'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'execute(...)' code ##################

        
        # Call to execute(...): (line 873)
        # Processing the call arguments (line 873)
        # Getting the type of 'func' (line 873)
        func_305047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 16), 'func', False)
        # Getting the type of 'args' (line 873)
        args_305048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 22), 'args', False)
        # Getting the type of 'msg' (line 873)
        msg_305049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 28), 'msg', False)
        # Getting the type of 'self' (line 873)
        self_305050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 33), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 873)
        dry_run_305051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 873, 33), self_305050, 'dry_run')
        # Processing the call keyword arguments (line 873)
        kwargs_305052 = {}
        # Getting the type of 'execute' (line 873)
        execute_305046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 8), 'execute', False)
        # Calling execute(args, kwargs) (line 873)
        execute_call_result_305053 = invoke(stypy.reporting.localization.Localization(__file__, 873, 8), execute_305046, *[func_305047, args_305048, msg_305049, dry_run_305051], **kwargs_305052)
        
        
        # ################# End of 'execute(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'execute' in the type store
        # Getting the type of 'stypy_return_type' (line 872)
        stypy_return_type_305054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305054)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'execute'
        return stypy_return_type_305054


    @norecursion
    def spawn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'spawn'
        module_type_store = module_type_store.open_function_context('spawn', 875, 4, False)
        # Assigning a type to the variable 'self' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.spawn.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.spawn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.spawn.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.spawn.__dict__.__setitem__('stypy_function_name', 'CCompiler.spawn')
        CCompiler.spawn.__dict__.__setitem__('stypy_param_names_list', ['cmd'])
        CCompiler.spawn.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.spawn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.spawn.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.spawn.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.spawn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.spawn.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.spawn', ['cmd'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'spawn', localization, ['cmd'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'spawn(...)' code ##################

        
        # Call to spawn(...): (line 876)
        # Processing the call arguments (line 876)
        # Getting the type of 'cmd' (line 876)
        cmd_305056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 14), 'cmd', False)
        # Processing the call keyword arguments (line 876)
        # Getting the type of 'self' (line 876)
        self_305057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 27), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 876)
        dry_run_305058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 876, 27), self_305057, 'dry_run')
        keyword_305059 = dry_run_305058
        kwargs_305060 = {'dry_run': keyword_305059}
        # Getting the type of 'spawn' (line 876)
        spawn_305055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'spawn', False)
        # Calling spawn(args, kwargs) (line 876)
        spawn_call_result_305061 = invoke(stypy.reporting.localization.Localization(__file__, 876, 8), spawn_305055, *[cmd_305056], **kwargs_305060)
        
        
        # ################# End of 'spawn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'spawn' in the type store
        # Getting the type of 'stypy_return_type' (line 875)
        stypy_return_type_305062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305062)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'spawn'
        return stypy_return_type_305062


    @norecursion
    def move_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'move_file'
        module_type_store = module_type_store.open_function_context('move_file', 878, 4, False)
        # Assigning a type to the variable 'self' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.move_file.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.move_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.move_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.move_file.__dict__.__setitem__('stypy_function_name', 'CCompiler.move_file')
        CCompiler.move_file.__dict__.__setitem__('stypy_param_names_list', ['src', 'dst'])
        CCompiler.move_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.move_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.move_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.move_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.move_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.move_file.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.move_file', ['src', 'dst'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'move_file', localization, ['src', 'dst'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'move_file(...)' code ##################

        
        # Call to move_file(...): (line 879)
        # Processing the call arguments (line 879)
        # Getting the type of 'src' (line 879)
        src_305064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 25), 'src', False)
        # Getting the type of 'dst' (line 879)
        dst_305065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 30), 'dst', False)
        # Processing the call keyword arguments (line 879)
        # Getting the type of 'self' (line 879)
        self_305066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 43), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 879)
        dry_run_305067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 43), self_305066, 'dry_run')
        keyword_305068 = dry_run_305067
        kwargs_305069 = {'dry_run': keyword_305068}
        # Getting the type of 'move_file' (line 879)
        move_file_305063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 15), 'move_file', False)
        # Calling move_file(args, kwargs) (line 879)
        move_file_call_result_305070 = invoke(stypy.reporting.localization.Localization(__file__, 879, 15), move_file_305063, *[src_305064, dst_305065], **kwargs_305069)
        
        # Assigning a type to the variable 'stypy_return_type' (line 879)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'stypy_return_type', move_file_call_result_305070)
        
        # ################# End of 'move_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'move_file' in the type store
        # Getting the type of 'stypy_return_type' (line 878)
        stypy_return_type_305071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305071)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'move_file'
        return stypy_return_type_305071


    @norecursion
    def mkpath(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_305072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 881, 32), 'int')
        defaults = [int_305072]
        # Create a new context for function 'mkpath'
        module_type_store = module_type_store.open_function_context('mkpath', 881, 4, False)
        # Assigning a type to the variable 'self' (line 882)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CCompiler.mkpath.__dict__.__setitem__('stypy_localization', localization)
        CCompiler.mkpath.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CCompiler.mkpath.__dict__.__setitem__('stypy_type_store', module_type_store)
        CCompiler.mkpath.__dict__.__setitem__('stypy_function_name', 'CCompiler.mkpath')
        CCompiler.mkpath.__dict__.__setitem__('stypy_param_names_list', ['name', 'mode'])
        CCompiler.mkpath.__dict__.__setitem__('stypy_varargs_param_name', None)
        CCompiler.mkpath.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CCompiler.mkpath.__dict__.__setitem__('stypy_call_defaults', defaults)
        CCompiler.mkpath.__dict__.__setitem__('stypy_call_varargs', varargs)
        CCompiler.mkpath.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CCompiler.mkpath.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompiler.mkpath', ['name', 'mode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mkpath', localization, ['name', 'mode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mkpath(...)' code ##################

        
        # Call to mkpath(...): (line 882)
        # Processing the call arguments (line 882)
        # Getting the type of 'name' (line 882)
        name_305074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 15), 'name', False)
        # Getting the type of 'mode' (line 882)
        mode_305075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 21), 'mode', False)
        # Processing the call keyword arguments (line 882)
        # Getting the type of 'self' (line 882)
        self_305076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 35), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 882)
        dry_run_305077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 882, 35), self_305076, 'dry_run')
        keyword_305078 = dry_run_305077
        kwargs_305079 = {'dry_run': keyword_305078}
        # Getting the type of 'mkpath' (line 882)
        mkpath_305073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 8), 'mkpath', False)
        # Calling mkpath(args, kwargs) (line 882)
        mkpath_call_result_305080 = invoke(stypy.reporting.localization.Localization(__file__, 882, 8), mkpath_305073, *[name_305074, mode_305075], **kwargs_305079)
        
        
        # ################# End of 'mkpath(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mkpath' in the type store
        # Getting the type of 'stypy_return_type' (line 881)
        stypy_return_type_305081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_305081)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mkpath'
        return stypy_return_type_305081


# Assigning a type to the variable 'CCompiler' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'CCompiler', CCompiler)

# Assigning a Name to a Name (line 45):
# Getting the type of 'None' (line 45)
None_305082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'None')
# Getting the type of 'CCompiler'
CCompiler_305083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CCompiler_305083, 'compiler_type', None_305082)

# Assigning a Name to a Name (line 71):
# Getting the type of 'None' (line 71)
None_305084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'None')
# Getting the type of 'CCompiler'
CCompiler_305085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CCompiler')
# Setting the type of the member 'src_extensions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CCompiler_305085, 'src_extensions', None_305084)

# Assigning a Name to a Name (line 72):
# Getting the type of 'None' (line 72)
None_305086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'None')
# Getting the type of 'CCompiler'
CCompiler_305087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CCompiler')
# Setting the type of the member 'obj_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CCompiler_305087, 'obj_extension', None_305086)

# Assigning a Name to a Name (line 73):
# Getting the type of 'None' (line 73)
None_305088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 27), 'None')
# Getting the type of 'CCompiler'
CCompiler_305089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CCompiler')
# Setting the type of the member 'static_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CCompiler_305089, 'static_lib_extension', None_305088)

# Assigning a Name to a Name (line 74):
# Getting the type of 'None' (line 74)
None_305090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 27), 'None')
# Getting the type of 'CCompiler'
CCompiler_305091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CCompiler')
# Setting the type of the member 'shared_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CCompiler_305091, 'shared_lib_extension', None_305090)

# Assigning a Name to a Name (line 75):
# Getting the type of 'None' (line 75)
None_305092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 24), 'None')
# Getting the type of 'CCompiler'
CCompiler_305093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CCompiler')
# Setting the type of the member 'static_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CCompiler_305093, 'static_lib_format', None_305092)

# Assigning a Name to a Name (line 76):
# Getting the type of 'None' (line 76)
None_305094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'None')
# Getting the type of 'CCompiler'
CCompiler_305095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CCompiler')
# Setting the type of the member 'shared_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CCompiler_305095, 'shared_lib_format', None_305094)

# Assigning a Name to a Name (line 77):
# Getting the type of 'None' (line 77)
None_305096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'None')
# Getting the type of 'CCompiler'
CCompiler_305097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CCompiler')
# Setting the type of the member 'exe_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CCompiler_305097, 'exe_extension', None_305096)

# Assigning a Dict to a Name (line 85):

# Obtaining an instance of the builtin type 'dict' (line 85)
dict_305098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 85)
# Adding element type (key, value) (line 85)
str_305099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 20), 'str', '.c')
str_305100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 29), 'str', 'c')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 19), dict_305098, (str_305099, str_305100))
# Adding element type (key, value) (line 85)
str_305101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 20), 'str', '.cc')
str_305102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 29), 'str', 'c++')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 19), dict_305098, (str_305101, str_305102))
# Adding element type (key, value) (line 85)
str_305103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 20), 'str', '.cpp')
str_305104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 29), 'str', 'c++')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 19), dict_305098, (str_305103, str_305104))
# Adding element type (key, value) (line 85)
str_305105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 20), 'str', '.cxx')
str_305106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 29), 'str', 'c++')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 19), dict_305098, (str_305105, str_305106))
# Adding element type (key, value) (line 85)
str_305107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 20), 'str', '.m')
str_305108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 29), 'str', 'objc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 19), dict_305098, (str_305107, str_305108))

# Getting the type of 'CCompiler'
CCompiler_305109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CCompiler')
# Setting the type of the member 'language_map' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CCompiler_305109, 'language_map', dict_305098)

# Assigning a List to a Name (line 91):

# Obtaining an instance of the builtin type 'list' (line 91)
list_305110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 91)
# Adding element type (line 91)
str_305111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 22), 'str', 'c++')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 21), list_305110, str_305111)
# Adding element type (line 91)
str_305112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 29), 'str', 'objc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 21), list_305110, str_305112)
# Adding element type (line 91)
str_305113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 37), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 21), list_305110, str_305113)

# Getting the type of 'CCompiler'
CCompiler_305114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CCompiler')
# Setting the type of the member 'language_order' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CCompiler_305114, 'language_order', list_305110)

# Assigning a Str to a Name (line 613):
str_305115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 20), 'str', 'shared_object')
# Getting the type of 'CCompiler'
CCompiler_305116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CCompiler')
# Setting the type of the member 'SHARED_OBJECT' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CCompiler_305116, 'SHARED_OBJECT', str_305115)

# Assigning a Str to a Name (line 614):
str_305117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 21), 'str', 'shared_library')
# Getting the type of 'CCompiler'
CCompiler_305118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CCompiler')
# Setting the type of the member 'SHARED_LIBRARY' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CCompiler_305118, 'SHARED_LIBRARY', str_305117)

# Assigning a Str to a Name (line 615):
str_305119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 17), 'str', 'executable')
# Getting the type of 'CCompiler'
CCompiler_305120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CCompiler')
# Setting the type of the member 'EXECUTABLE' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CCompiler_305120, 'EXECUTABLE', str_305119)

# Assigning a Tuple to a Name (line 892):

# Assigning a Tuple to a Name (line 892):

# Obtaining an instance of the builtin type 'tuple' (line 898)
tuple_305121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 898, 4), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 898)
# Adding element type (line 898)

# Obtaining an instance of the builtin type 'tuple' (line 898)
tuple_305122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 898, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 898)
# Adding element type (line 898)
str_305123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 898, 5), 'str', 'cygwin.*')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 898, 5), tuple_305122, str_305123)
# Adding element type (line 898)
str_305124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 898, 17), 'str', 'unix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 898, 5), tuple_305122, str_305124)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 898, 4), tuple_305121, tuple_305122)
# Adding element type (line 898)

# Obtaining an instance of the builtin type 'tuple' (line 899)
tuple_305125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 899)
# Adding element type (line 899)
str_305126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 5), 'str', 'os2emx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 899, 5), tuple_305125, str_305126)
# Adding element type (line 899)
str_305127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 899, 15), 'str', 'emx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 899, 5), tuple_305125, str_305127)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 898, 4), tuple_305121, tuple_305125)
# Adding element type (line 898)

# Obtaining an instance of the builtin type 'tuple' (line 902)
tuple_305128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 902)
# Adding element type (line 902)
str_305129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 5), 'str', 'posix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 902, 5), tuple_305128, str_305129)
# Adding element type (line 902)
str_305130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 14), 'str', 'unix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 902, 5), tuple_305128, str_305130)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 898, 4), tuple_305121, tuple_305128)
# Adding element type (line 898)

# Obtaining an instance of the builtin type 'tuple' (line 903)
tuple_305131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 903)
# Adding element type (line 903)
str_305132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, 5), 'str', 'nt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 903, 5), tuple_305131, str_305132)
# Adding element type (line 903)
str_305133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 903, 11), 'str', 'msvc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 903, 5), tuple_305131, str_305133)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 898, 4), tuple_305121, tuple_305131)

# Assigning a type to the variable '_default_compilers' (line 892)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 0), '_default_compilers', tuple_305121)

@norecursion
def get_default_compiler(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 907)
    None_305134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 32), 'None')
    # Getting the type of 'None' (line 907)
    None_305135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 47), 'None')
    defaults = [None_305134, None_305135]
    # Create a new context for function 'get_default_compiler'
    module_type_store = module_type_store.open_function_context('get_default_compiler', 907, 0, False)
    
    # Passed parameters checking function
    get_default_compiler.stypy_localization = localization
    get_default_compiler.stypy_type_of_self = None
    get_default_compiler.stypy_type_store = module_type_store
    get_default_compiler.stypy_function_name = 'get_default_compiler'
    get_default_compiler.stypy_param_names_list = ['osname', 'platform']
    get_default_compiler.stypy_varargs_param_name = None
    get_default_compiler.stypy_kwargs_param_name = None
    get_default_compiler.stypy_call_defaults = defaults
    get_default_compiler.stypy_call_varargs = varargs
    get_default_compiler.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_default_compiler', ['osname', 'platform'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_default_compiler', localization, ['osname', 'platform'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_default_compiler(...)' code ##################

    str_305136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 917, (-1)), 'str', ' Determine the default compiler to use for the given platform.\n\n        osname should be one of the standard Python OS names (i.e. the\n        ones returned by os.name) and platform the common value\n        returned by sys.platform for the platform in question.\n\n        The default values are os.name and sys.platform in case the\n        parameters are not given.\n\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 918)
    # Getting the type of 'osname' (line 918)
    osname_305137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 7), 'osname')
    # Getting the type of 'None' (line 918)
    None_305138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 17), 'None')
    
    (may_be_305139, more_types_in_union_305140) = may_be_none(osname_305137, None_305138)

    if may_be_305139:

        if more_types_in_union_305140:
            # Runtime conditional SSA (line 918)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 919):
        
        # Assigning a Attribute to a Name (line 919):
        # Getting the type of 'os' (line 919)
        os_305141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 17), 'os')
        # Obtaining the member 'name' of a type (line 919)
        name_305142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 919, 17), os_305141, 'name')
        # Assigning a type to the variable 'osname' (line 919)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 919, 8), 'osname', name_305142)

        if more_types_in_union_305140:
            # SSA join for if statement (line 918)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 920)
    # Getting the type of 'platform' (line 920)
    platform_305143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 7), 'platform')
    # Getting the type of 'None' (line 920)
    None_305144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 920, 19), 'None')
    
    (may_be_305145, more_types_in_union_305146) = may_be_none(platform_305143, None_305144)

    if may_be_305145:

        if more_types_in_union_305146:
            # Runtime conditional SSA (line 920)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 921):
        
        # Assigning a Attribute to a Name (line 921):
        # Getting the type of 'sys' (line 921)
        sys_305147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 19), 'sys')
        # Obtaining the member 'platform' of a type (line 921)
        platform_305148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 921, 19), sys_305147, 'platform')
        # Assigning a type to the variable 'platform' (line 921)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 921, 8), 'platform', platform_305148)

        if more_types_in_union_305146:
            # SSA join for if statement (line 920)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of '_default_compilers' (line 922)
    _default_compilers_305149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 29), '_default_compilers')
    # Testing the type of a for loop iterable (line 922)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 922, 4), _default_compilers_305149)
    # Getting the type of the for loop variable (line 922)
    for_loop_var_305150 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 922, 4), _default_compilers_305149)
    # Assigning a type to the variable 'pattern' (line 922)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 4), 'pattern', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 922, 4), for_loop_var_305150))
    # Assigning a type to the variable 'compiler' (line 922)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 4), 'compiler', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 922, 4), for_loop_var_305150))
    # SSA begins for a for statement (line 922)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    
    # Call to match(...): (line 923)
    # Processing the call arguments (line 923)
    # Getting the type of 'pattern' (line 923)
    pattern_305153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 20), 'pattern', False)
    # Getting the type of 'platform' (line 923)
    platform_305154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 29), 'platform', False)
    # Processing the call keyword arguments (line 923)
    kwargs_305155 = {}
    # Getting the type of 're' (line 923)
    re_305151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 11), 're', False)
    # Obtaining the member 'match' of a type (line 923)
    match_305152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 11), re_305151, 'match')
    # Calling match(args, kwargs) (line 923)
    match_call_result_305156 = invoke(stypy.reporting.localization.Localization(__file__, 923, 11), match_305152, *[pattern_305153, platform_305154], **kwargs_305155)
    
    # Getting the type of 'None' (line 923)
    None_305157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 46), 'None')
    # Applying the binary operator 'isnot' (line 923)
    result_is_not_305158 = python_operator(stypy.reporting.localization.Localization(__file__, 923, 11), 'isnot', match_call_result_305156, None_305157)
    
    
    
    # Call to match(...): (line 924)
    # Processing the call arguments (line 924)
    # Getting the type of 'pattern' (line 924)
    pattern_305161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 20), 'pattern', False)
    # Getting the type of 'osname' (line 924)
    osname_305162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 29), 'osname', False)
    # Processing the call keyword arguments (line 924)
    kwargs_305163 = {}
    # Getting the type of 're' (line 924)
    re_305159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 11), 're', False)
    # Obtaining the member 'match' of a type (line 924)
    match_305160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 11), re_305159, 'match')
    # Calling match(args, kwargs) (line 924)
    match_call_result_305164 = invoke(stypy.reporting.localization.Localization(__file__, 924, 11), match_305160, *[pattern_305161, osname_305162], **kwargs_305163)
    
    # Getting the type of 'None' (line 924)
    None_305165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 44), 'None')
    # Applying the binary operator 'isnot' (line 924)
    result_is_not_305166 = python_operator(stypy.reporting.localization.Localization(__file__, 924, 11), 'isnot', match_call_result_305164, None_305165)
    
    # Applying the binary operator 'or' (line 923)
    result_or_keyword_305167 = python_operator(stypy.reporting.localization.Localization(__file__, 923, 11), 'or', result_is_not_305158, result_is_not_305166)
    
    # Testing the type of an if condition (line 923)
    if_condition_305168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 923, 8), result_or_keyword_305167)
    # Assigning a type to the variable 'if_condition_305168' (line 923)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 8), 'if_condition_305168', if_condition_305168)
    # SSA begins for if statement (line 923)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'compiler' (line 925)
    compiler_305169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 19), 'compiler')
    # Assigning a type to the variable 'stypy_return_type' (line 925)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 925, 12), 'stypy_return_type', compiler_305169)
    # SSA join for if statement (line 923)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    str_305170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 927, 11), 'str', 'unix')
    # Assigning a type to the variable 'stypy_return_type' (line 927)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 4), 'stypy_return_type', str_305170)
    
    # ################# End of 'get_default_compiler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_default_compiler' in the type store
    # Getting the type of 'stypy_return_type' (line 907)
    stypy_return_type_305171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_305171)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_default_compiler'
    return stypy_return_type_305171

# Assigning a type to the variable 'get_default_compiler' (line 907)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 0), 'get_default_compiler', get_default_compiler)

# Assigning a Dict to a Name (line 932):

# Assigning a Dict to a Name (line 932):

# Obtaining an instance of the builtin type 'dict' (line 932)
dict_305172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 932)
# Adding element type (key, value) (line 932)
str_305173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 19), 'str', 'unix')

# Obtaining an instance of the builtin type 'tuple' (line 932)
tuple_305174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 31), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 932)
# Adding element type (line 932)
str_305175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 31), 'str', 'unixccompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 932, 31), tuple_305174, str_305175)
# Adding element type (line 932)
str_305176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 48), 'str', 'UnixCCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 932, 31), tuple_305174, str_305176)
# Adding element type (line 932)
str_305177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, 31), 'str', 'standard UNIX-style compiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 932, 31), tuple_305174, str_305177)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 932, 17), dict_305172, (str_305173, tuple_305174))
# Adding element type (key, value) (line 932)
str_305178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 19), 'str', 'msvc')

# Obtaining an instance of the builtin type 'tuple' (line 934)
tuple_305179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 31), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 934)
# Adding element type (line 934)
str_305180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 31), 'str', 'msvccompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 934, 31), tuple_305179, str_305180)
# Adding element type (line 934)
str_305181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 934, 47), 'str', 'MSVCCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 934, 31), tuple_305179, str_305181)
# Adding element type (line 934)
str_305182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 935, 31), 'str', 'Microsoft Visual C++')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 934, 31), tuple_305179, str_305182)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 932, 17), dict_305172, (str_305178, tuple_305179))
# Adding element type (key, value) (line 932)
str_305183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 19), 'str', 'cygwin')

# Obtaining an instance of the builtin type 'tuple' (line 936)
tuple_305184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 31), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 936)
# Adding element type (line 936)
str_305185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 31), 'str', 'cygwinccompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 936, 31), tuple_305184, str_305185)
# Adding element type (line 936)
str_305186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 50), 'str', 'CygwinCCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 936, 31), tuple_305184, str_305186)
# Adding element type (line 936)
str_305187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 937, 31), 'str', 'Cygwin port of GNU C Compiler for Win32')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 936, 31), tuple_305184, str_305187)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 932, 17), dict_305172, (str_305183, tuple_305184))
# Adding element type (key, value) (line 932)
str_305188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 19), 'str', 'mingw32')

# Obtaining an instance of the builtin type 'tuple' (line 938)
tuple_305189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 31), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 938)
# Adding element type (line 938)
str_305190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 31), 'str', 'cygwinccompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 938, 31), tuple_305189, str_305190)
# Adding element type (line 938)
str_305191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 938, 50), 'str', 'Mingw32CCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 938, 31), tuple_305189, str_305191)
# Adding element type (line 938)
str_305192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 31), 'str', 'Mingw32 port of GNU C Compiler for Win32')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 938, 31), tuple_305189, str_305192)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 932, 17), dict_305172, (str_305188, tuple_305189))
# Adding element type (key, value) (line 932)
str_305193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 19), 'str', 'bcpp')

# Obtaining an instance of the builtin type 'tuple' (line 940)
tuple_305194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 31), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 940)
# Adding element type (line 940)
str_305195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 31), 'str', 'bcppcompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 940, 31), tuple_305194, str_305195)
# Adding element type (line 940)
str_305196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 47), 'str', 'BCPPCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 940, 31), tuple_305194, str_305196)
# Adding element type (line 940)
str_305197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 31), 'str', 'Borland C++ Compiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 940, 31), tuple_305194, str_305197)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 932, 17), dict_305172, (str_305193, tuple_305194))
# Adding element type (key, value) (line 932)
str_305198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 19), 'str', 'emx')

# Obtaining an instance of the builtin type 'tuple' (line 942)
tuple_305199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 31), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 942)
# Adding element type (line 942)
str_305200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 31), 'str', 'emxccompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 942, 31), tuple_305199, str_305200)
# Adding element type (line 942)
str_305201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 47), 'str', 'EMXCCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 942, 31), tuple_305199, str_305201)
# Adding element type (line 942)
str_305202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 943, 31), 'str', 'EMX port of GNU C Compiler for OS/2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 942, 31), tuple_305199, str_305202)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 932, 17), dict_305172, (str_305198, tuple_305199))

# Assigning a type to the variable 'compiler_class' (line 932)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 932, 0), 'compiler_class', dict_305172)

@norecursion
def show_compilers(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'show_compilers'
    module_type_store = module_type_store.open_function_context('show_compilers', 946, 0, False)
    
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

    str_305203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 949, (-1)), 'str', 'Print list of available compilers (used by the "--help-compiler"\n    options to "build", "build_ext", "build_clib").\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 953, 4))
    
    # 'from distutils.fancy_getopt import FancyGetopt' statement (line 953)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/')
    import_305204 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 953, 4), 'distutils.fancy_getopt')

    if (type(import_305204) is not StypyTypeError):

        if (import_305204 != 'pyd_module'):
            __import__(import_305204)
            sys_modules_305205 = sys.modules[import_305204]
            import_from_module(stypy.reporting.localization.Localization(__file__, 953, 4), 'distutils.fancy_getopt', sys_modules_305205.module_type_store, module_type_store, ['FancyGetopt'])
            nest_module(stypy.reporting.localization.Localization(__file__, 953, 4), __file__, sys_modules_305205, sys_modules_305205.module_type_store, module_type_store)
        else:
            from distutils.fancy_getopt import FancyGetopt

            import_from_module(stypy.reporting.localization.Localization(__file__, 953, 4), 'distutils.fancy_getopt', None, module_type_store, ['FancyGetopt'], [FancyGetopt])

    else:
        # Assigning a type to the variable 'distutils.fancy_getopt' (line 953)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 953, 4), 'distutils.fancy_getopt', import_305204)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
    
    
    # Assigning a List to a Name (line 954):
    
    # Assigning a List to a Name (line 954):
    
    # Obtaining an instance of the builtin type 'list' (line 954)
    list_305206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 954, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 954)
    
    # Assigning a type to the variable 'compilers' (line 954)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 954, 4), 'compilers', list_305206)
    
    
    # Call to keys(...): (line 955)
    # Processing the call keyword arguments (line 955)
    kwargs_305209 = {}
    # Getting the type of 'compiler_class' (line 955)
    compiler_class_305207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 20), 'compiler_class', False)
    # Obtaining the member 'keys' of a type (line 955)
    keys_305208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 20), compiler_class_305207, 'keys')
    # Calling keys(args, kwargs) (line 955)
    keys_call_result_305210 = invoke(stypy.reporting.localization.Localization(__file__, 955, 20), keys_305208, *[], **kwargs_305209)
    
    # Testing the type of a for loop iterable (line 955)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 955, 4), keys_call_result_305210)
    # Getting the type of the for loop variable (line 955)
    for_loop_var_305211 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 955, 4), keys_call_result_305210)
    # Assigning a type to the variable 'compiler' (line 955)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 4), 'compiler', for_loop_var_305211)
    # SSA begins for a for statement (line 955)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 956)
    # Processing the call arguments (line 956)
    
    # Obtaining an instance of the builtin type 'tuple' (line 956)
    tuple_305214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 956, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 956)
    # Adding element type (line 956)
    str_305215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 956, 26), 'str', 'compiler=')
    # Getting the type of 'compiler' (line 956)
    compiler_305216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 38), 'compiler', False)
    # Applying the binary operator '+' (line 956)
    result_add_305217 = python_operator(stypy.reporting.localization.Localization(__file__, 956, 26), '+', str_305215, compiler_305216)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 956, 26), tuple_305214, result_add_305217)
    # Adding element type (line 956)
    # Getting the type of 'None' (line 956)
    None_305218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 48), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 956, 26), tuple_305214, None_305218)
    # Adding element type (line 956)
    
    # Obtaining the type of the subscript
    int_305219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 957, 51), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'compiler' (line 957)
    compiler_305220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 41), 'compiler', False)
    # Getting the type of 'compiler_class' (line 957)
    compiler_class_305221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 26), 'compiler_class', False)
    # Obtaining the member '__getitem__' of a type (line 957)
    getitem___305222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 957, 26), compiler_class_305221, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 957)
    subscript_call_result_305223 = invoke(stypy.reporting.localization.Localization(__file__, 957, 26), getitem___305222, compiler_305220)
    
    # Obtaining the member '__getitem__' of a type (line 957)
    getitem___305224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 957, 26), subscript_call_result_305223, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 957)
    subscript_call_result_305225 = invoke(stypy.reporting.localization.Localization(__file__, 957, 26), getitem___305224, int_305219)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 956, 26), tuple_305214, subscript_call_result_305225)
    
    # Processing the call keyword arguments (line 956)
    kwargs_305226 = {}
    # Getting the type of 'compilers' (line 956)
    compilers_305212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 8), 'compilers', False)
    # Obtaining the member 'append' of a type (line 956)
    append_305213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 956, 8), compilers_305212, 'append')
    # Calling append(args, kwargs) (line 956)
    append_call_result_305227 = invoke(stypy.reporting.localization.Localization(__file__, 956, 8), append_305213, *[tuple_305214], **kwargs_305226)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to sort(...): (line 958)
    # Processing the call keyword arguments (line 958)
    kwargs_305230 = {}
    # Getting the type of 'compilers' (line 958)
    compilers_305228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 4), 'compilers', False)
    # Obtaining the member 'sort' of a type (line 958)
    sort_305229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 958, 4), compilers_305228, 'sort')
    # Calling sort(args, kwargs) (line 958)
    sort_call_result_305231 = invoke(stypy.reporting.localization.Localization(__file__, 958, 4), sort_305229, *[], **kwargs_305230)
    
    
    # Assigning a Call to a Name (line 959):
    
    # Assigning a Call to a Name (line 959):
    
    # Call to FancyGetopt(...): (line 959)
    # Processing the call arguments (line 959)
    # Getting the type of 'compilers' (line 959)
    compilers_305233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 33), 'compilers', False)
    # Processing the call keyword arguments (line 959)
    kwargs_305234 = {}
    # Getting the type of 'FancyGetopt' (line 959)
    FancyGetopt_305232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 21), 'FancyGetopt', False)
    # Calling FancyGetopt(args, kwargs) (line 959)
    FancyGetopt_call_result_305235 = invoke(stypy.reporting.localization.Localization(__file__, 959, 21), FancyGetopt_305232, *[compilers_305233], **kwargs_305234)
    
    # Assigning a type to the variable 'pretty_printer' (line 959)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 959, 4), 'pretty_printer', FancyGetopt_call_result_305235)
    
    # Call to print_help(...): (line 960)
    # Processing the call arguments (line 960)
    str_305238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 30), 'str', 'List of available compilers:')
    # Processing the call keyword arguments (line 960)
    kwargs_305239 = {}
    # Getting the type of 'pretty_printer' (line 960)
    pretty_printer_305236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 4), 'pretty_printer', False)
    # Obtaining the member 'print_help' of a type (line 960)
    print_help_305237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 960, 4), pretty_printer_305236, 'print_help')
    # Calling print_help(args, kwargs) (line 960)
    print_help_call_result_305240 = invoke(stypy.reporting.localization.Localization(__file__, 960, 4), print_help_305237, *[str_305238], **kwargs_305239)
    
    
    # ################# End of 'show_compilers(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'show_compilers' in the type store
    # Getting the type of 'stypy_return_type' (line 946)
    stypy_return_type_305241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_305241)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'show_compilers'
    return stypy_return_type_305241

# Assigning a type to the variable 'show_compilers' (line 946)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 0), 'show_compilers', show_compilers)

@norecursion
def new_compiler(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 963)
    None_305242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 22), 'None')
    # Getting the type of 'None' (line 963)
    None_305243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 37), 'None')
    int_305244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 963, 51), 'int')
    int_305245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 963, 62), 'int')
    int_305246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 963, 71), 'int')
    defaults = [None_305242, None_305243, int_305244, int_305245, int_305246]
    # Create a new context for function 'new_compiler'
    module_type_store = module_type_store.open_function_context('new_compiler', 963, 0, False)
    
    # Passed parameters checking function
    new_compiler.stypy_localization = localization
    new_compiler.stypy_type_of_self = None
    new_compiler.stypy_type_store = module_type_store
    new_compiler.stypy_function_name = 'new_compiler'
    new_compiler.stypy_param_names_list = ['plat', 'compiler', 'verbose', 'dry_run', 'force']
    new_compiler.stypy_varargs_param_name = None
    new_compiler.stypy_kwargs_param_name = None
    new_compiler.stypy_call_defaults = defaults
    new_compiler.stypy_call_varargs = varargs
    new_compiler.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'new_compiler', ['plat', 'compiler', 'verbose', 'dry_run', 'force'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'new_compiler', localization, ['plat', 'compiler', 'verbose', 'dry_run', 'force'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'new_compiler(...)' code ##################

    str_305247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, (-1)), 'str', 'Generate an instance of some CCompiler subclass for the supplied\n    platform/compiler combination.  \'plat\' defaults to \'os.name\'\n    (eg. \'posix\', \'nt\'), and \'compiler\' defaults to the default compiler\n    for that platform.  Currently only \'posix\' and \'nt\' are supported, and\n    the default compilers are "traditional Unix interface" (UnixCCompiler\n    class) and Visual C++ (MSVCCompiler class).  Note that it\'s perfectly\n    possible to ask for a Unix compiler object under Windows, and a\n    Microsoft compiler object under Unix -- if you supply a value for\n    \'compiler\', \'plat\' is ignored.\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 974)
    # Getting the type of 'plat' (line 974)
    plat_305248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 7), 'plat')
    # Getting the type of 'None' (line 974)
    None_305249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 15), 'None')
    
    (may_be_305250, more_types_in_union_305251) = may_be_none(plat_305248, None_305249)

    if may_be_305250:

        if more_types_in_union_305251:
            # Runtime conditional SSA (line 974)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 975):
        
        # Assigning a Attribute to a Name (line 975):
        # Getting the type of 'os' (line 975)
        os_305252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 15), 'os')
        # Obtaining the member 'name' of a type (line 975)
        name_305253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 975, 15), os_305252, 'name')
        # Assigning a type to the variable 'plat' (line 975)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 975, 8), 'plat', name_305253)

        if more_types_in_union_305251:
            # SSA join for if statement (line 974)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # SSA begins for try-except statement (line 977)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Type idiom detected: calculating its left and rigth part (line 978)
    # Getting the type of 'compiler' (line 978)
    compiler_305254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 11), 'compiler')
    # Getting the type of 'None' (line 978)
    None_305255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 23), 'None')
    
    (may_be_305256, more_types_in_union_305257) = may_be_none(compiler_305254, None_305255)

    if may_be_305256:

        if more_types_in_union_305257:
            # Runtime conditional SSA (line 978)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 979):
        
        # Assigning a Call to a Name (line 979):
        
        # Call to get_default_compiler(...): (line 979)
        # Processing the call arguments (line 979)
        # Getting the type of 'plat' (line 979)
        plat_305259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 44), 'plat', False)
        # Processing the call keyword arguments (line 979)
        kwargs_305260 = {}
        # Getting the type of 'get_default_compiler' (line 979)
        get_default_compiler_305258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 23), 'get_default_compiler', False)
        # Calling get_default_compiler(args, kwargs) (line 979)
        get_default_compiler_call_result_305261 = invoke(stypy.reporting.localization.Localization(__file__, 979, 23), get_default_compiler_305258, *[plat_305259], **kwargs_305260)
        
        # Assigning a type to the variable 'compiler' (line 979)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 979, 12), 'compiler', get_default_compiler_call_result_305261)

        if more_types_in_union_305257:
            # SSA join for if statement (line 978)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Subscript to a Tuple (line 981):
    
    # Assigning a Subscript to a Name (line 981):
    
    # Obtaining the type of the subscript
    int_305262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'compiler' (line 981)
    compiler_305263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 69), 'compiler')
    # Getting the type of 'compiler_class' (line 981)
    compiler_class_305264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 54), 'compiler_class')
    # Obtaining the member '__getitem__' of a type (line 981)
    getitem___305265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 54), compiler_class_305264, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 981)
    subscript_call_result_305266 = invoke(stypy.reporting.localization.Localization(__file__, 981, 54), getitem___305265, compiler_305263)
    
    # Obtaining the member '__getitem__' of a type (line 981)
    getitem___305267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 8), subscript_call_result_305266, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 981)
    subscript_call_result_305268 = invoke(stypy.reporting.localization.Localization(__file__, 981, 8), getitem___305267, int_305262)
    
    # Assigning a type to the variable 'tuple_var_assignment_303828' (line 981)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 981, 8), 'tuple_var_assignment_303828', subscript_call_result_305268)
    
    # Assigning a Subscript to a Name (line 981):
    
    # Obtaining the type of the subscript
    int_305269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'compiler' (line 981)
    compiler_305270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 69), 'compiler')
    # Getting the type of 'compiler_class' (line 981)
    compiler_class_305271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 54), 'compiler_class')
    # Obtaining the member '__getitem__' of a type (line 981)
    getitem___305272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 54), compiler_class_305271, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 981)
    subscript_call_result_305273 = invoke(stypy.reporting.localization.Localization(__file__, 981, 54), getitem___305272, compiler_305270)
    
    # Obtaining the member '__getitem__' of a type (line 981)
    getitem___305274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 8), subscript_call_result_305273, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 981)
    subscript_call_result_305275 = invoke(stypy.reporting.localization.Localization(__file__, 981, 8), getitem___305274, int_305269)
    
    # Assigning a type to the variable 'tuple_var_assignment_303829' (line 981)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 981, 8), 'tuple_var_assignment_303829', subscript_call_result_305275)
    
    # Assigning a Subscript to a Name (line 981):
    
    # Obtaining the type of the subscript
    int_305276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'compiler' (line 981)
    compiler_305277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 69), 'compiler')
    # Getting the type of 'compiler_class' (line 981)
    compiler_class_305278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 54), 'compiler_class')
    # Obtaining the member '__getitem__' of a type (line 981)
    getitem___305279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 54), compiler_class_305278, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 981)
    subscript_call_result_305280 = invoke(stypy.reporting.localization.Localization(__file__, 981, 54), getitem___305279, compiler_305277)
    
    # Obtaining the member '__getitem__' of a type (line 981)
    getitem___305281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 8), subscript_call_result_305280, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 981)
    subscript_call_result_305282 = invoke(stypy.reporting.localization.Localization(__file__, 981, 8), getitem___305281, int_305276)
    
    # Assigning a type to the variable 'tuple_var_assignment_303830' (line 981)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 981, 8), 'tuple_var_assignment_303830', subscript_call_result_305282)
    
    # Assigning a Name to a Name (line 981):
    # Getting the type of 'tuple_var_assignment_303828' (line 981)
    tuple_var_assignment_303828_305283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 8), 'tuple_var_assignment_303828')
    # Assigning a type to the variable 'module_name' (line 981)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 981, 9), 'module_name', tuple_var_assignment_303828_305283)
    
    # Assigning a Name to a Name (line 981):
    # Getting the type of 'tuple_var_assignment_303829' (line 981)
    tuple_var_assignment_303829_305284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 8), 'tuple_var_assignment_303829')
    # Assigning a type to the variable 'class_name' (line 981)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 981, 22), 'class_name', tuple_var_assignment_303829_305284)
    
    # Assigning a Name to a Name (line 981):
    # Getting the type of 'tuple_var_assignment_303830' (line 981)
    tuple_var_assignment_303830_305285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 8), 'tuple_var_assignment_303830')
    # Assigning a type to the variable 'long_description' (line 981)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 981, 34), 'long_description', tuple_var_assignment_303830_305285)
    # SSA branch for the except part of a try statement (line 977)
    # SSA branch for the except 'KeyError' branch of a try statement (line 977)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a BinOp to a Name (line 983):
    
    # Assigning a BinOp to a Name (line 983):
    str_305286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 983, 14), 'str', "don't know how to compile C/C++ code on platform '%s'")
    # Getting the type of 'plat' (line 983)
    plat_305287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 72), 'plat')
    # Applying the binary operator '%' (line 983)
    result_mod_305288 = python_operator(stypy.reporting.localization.Localization(__file__, 983, 14), '%', str_305286, plat_305287)
    
    # Assigning a type to the variable 'msg' (line 983)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 983, 8), 'msg', result_mod_305288)
    
    # Type idiom detected: calculating its left and rigth part (line 984)
    # Getting the type of 'compiler' (line 984)
    compiler_305289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 8), 'compiler')
    # Getting the type of 'None' (line 984)
    None_305290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 27), 'None')
    
    (may_be_305291, more_types_in_union_305292) = may_not_be_none(compiler_305289, None_305290)

    if may_be_305291:

        if more_types_in_union_305292:
            # Runtime conditional SSA (line 984)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 985):
        
        # Assigning a BinOp to a Name (line 985):
        # Getting the type of 'msg' (line 985)
        msg_305293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 18), 'msg')
        str_305294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 985, 24), 'str', " with '%s' compiler")
        # Getting the type of 'compiler' (line 985)
        compiler_305295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 48), 'compiler')
        # Applying the binary operator '%' (line 985)
        result_mod_305296 = python_operator(stypy.reporting.localization.Localization(__file__, 985, 24), '%', str_305294, compiler_305295)
        
        # Applying the binary operator '+' (line 985)
        result_add_305297 = python_operator(stypy.reporting.localization.Localization(__file__, 985, 18), '+', msg_305293, result_mod_305296)
        
        # Assigning a type to the variable 'msg' (line 985)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 985, 12), 'msg', result_add_305297)

        if more_types_in_union_305292:
            # SSA join for if statement (line 984)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'DistutilsPlatformError' (line 986)
    DistutilsPlatformError_305298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 986, 14), 'DistutilsPlatformError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 986, 8), DistutilsPlatformError_305298, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 977)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 988)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a BinOp to a Name (line 989):
    
    # Assigning a BinOp to a Name (line 989):
    str_305299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 989, 22), 'str', 'distutils.')
    # Getting the type of 'module_name' (line 989)
    module_name_305300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 37), 'module_name')
    # Applying the binary operator '+' (line 989)
    result_add_305301 = python_operator(stypy.reporting.localization.Localization(__file__, 989, 22), '+', str_305299, module_name_305300)
    
    # Assigning a type to the variable 'module_name' (line 989)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 989, 8), 'module_name', result_add_305301)
    
    # Call to __import__(...): (line 990)
    # Processing the call arguments (line 990)
    # Getting the type of 'module_name' (line 990)
    module_name_305303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 20), 'module_name', False)
    # Processing the call keyword arguments (line 990)
    kwargs_305304 = {}
    # Getting the type of '__import__' (line 990)
    import___305302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 8), '__import__', False)
    # Calling __import__(args, kwargs) (line 990)
    import___call_result_305305 = invoke(stypy.reporting.localization.Localization(__file__, 990, 8), import___305302, *[module_name_305303], **kwargs_305304)
    
    
    # Assigning a Subscript to a Name (line 991):
    
    # Assigning a Subscript to a Name (line 991):
    
    # Obtaining the type of the subscript
    # Getting the type of 'module_name' (line 991)
    module_name_305306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 29), 'module_name')
    # Getting the type of 'sys' (line 991)
    sys_305307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 17), 'sys')
    # Obtaining the member 'modules' of a type (line 991)
    modules_305308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 991, 17), sys_305307, 'modules')
    # Obtaining the member '__getitem__' of a type (line 991)
    getitem___305309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 991, 17), modules_305308, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 991)
    subscript_call_result_305310 = invoke(stypy.reporting.localization.Localization(__file__, 991, 17), getitem___305309, module_name_305306)
    
    # Assigning a type to the variable 'module' (line 991)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 991, 8), 'module', subscript_call_result_305310)
    
    # Assigning a Subscript to a Name (line 992):
    
    # Assigning a Subscript to a Name (line 992):
    
    # Obtaining the type of the subscript
    # Getting the type of 'class_name' (line 992)
    class_name_305311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 992, 29), 'class_name')
    
    # Call to vars(...): (line 992)
    # Processing the call arguments (line 992)
    # Getting the type of 'module' (line 992)
    module_305313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 992, 21), 'module', False)
    # Processing the call keyword arguments (line 992)
    kwargs_305314 = {}
    # Getting the type of 'vars' (line 992)
    vars_305312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 992, 16), 'vars', False)
    # Calling vars(args, kwargs) (line 992)
    vars_call_result_305315 = invoke(stypy.reporting.localization.Localization(__file__, 992, 16), vars_305312, *[module_305313], **kwargs_305314)
    
    # Obtaining the member '__getitem__' of a type (line 992)
    getitem___305316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 992, 16), vars_call_result_305315, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 992)
    subscript_call_result_305317 = invoke(stypy.reporting.localization.Localization(__file__, 992, 16), getitem___305316, class_name_305311)
    
    # Assigning a type to the variable 'klass' (line 992)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 992, 8), 'klass', subscript_call_result_305317)
    # SSA branch for the except part of a try statement (line 988)
    # SSA branch for the except 'ImportError' branch of a try statement (line 988)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'DistutilsModuleError' (line 994)
    DistutilsModuleError_305318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 994, 14), 'DistutilsModuleError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 994, 8), DistutilsModuleError_305318, 'raise parameter', BaseException)
    # SSA branch for the except 'KeyError' branch of a try statement (line 988)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'DistutilsModuleError' (line 998)
    DistutilsModuleError_305319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 998, 14), 'DistutilsModuleError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 998, 8), DistutilsModuleError_305319, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 988)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to klass(...): (line 1005)
    # Processing the call arguments (line 1005)
    # Getting the type of 'None' (line 1005)
    None_305321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 17), 'None', False)
    # Getting the type of 'dry_run' (line 1005)
    dry_run_305322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 23), 'dry_run', False)
    # Getting the type of 'force' (line 1005)
    force_305323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 32), 'force', False)
    # Processing the call keyword arguments (line 1005)
    kwargs_305324 = {}
    # Getting the type of 'klass' (line 1005)
    klass_305320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1005, 11), 'klass', False)
    # Calling klass(args, kwargs) (line 1005)
    klass_call_result_305325 = invoke(stypy.reporting.localization.Localization(__file__, 1005, 11), klass_305320, *[None_305321, dry_run_305322, force_305323], **kwargs_305324)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1005)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1005, 4), 'stypy_return_type', klass_call_result_305325)
    
    # ################# End of 'new_compiler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'new_compiler' in the type store
    # Getting the type of 'stypy_return_type' (line 963)
    stypy_return_type_305326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_305326)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'new_compiler'
    return stypy_return_type_305326

# Assigning a type to the variable 'new_compiler' (line 963)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 0), 'new_compiler', new_compiler)

@norecursion
def gen_preprocess_options(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gen_preprocess_options'
    module_type_store = module_type_store.open_function_context('gen_preprocess_options', 1008, 0, False)
    
    # Passed parameters checking function
    gen_preprocess_options.stypy_localization = localization
    gen_preprocess_options.stypy_type_of_self = None
    gen_preprocess_options.stypy_type_store = module_type_store
    gen_preprocess_options.stypy_function_name = 'gen_preprocess_options'
    gen_preprocess_options.stypy_param_names_list = ['macros', 'include_dirs']
    gen_preprocess_options.stypy_varargs_param_name = None
    gen_preprocess_options.stypy_kwargs_param_name = None
    gen_preprocess_options.stypy_call_defaults = defaults
    gen_preprocess_options.stypy_call_varargs = varargs
    gen_preprocess_options.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gen_preprocess_options', ['macros', 'include_dirs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gen_preprocess_options', localization, ['macros', 'include_dirs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gen_preprocess_options(...)' code ##################

    str_305327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1017, (-1)), 'str', "Generate C pre-processor options (-D, -U, -I) as used by at least\n    two types of compilers: the typical Unix compiler and Visual C++.\n    'macros' is the usual thing, a list of 1- or 2-tuples, where (name,)\n    means undefine (-U) macro 'name', and (name,value) means define (-D)\n    macro 'name' to 'value'.  'include_dirs' is just a list of directory\n    names to be added to the header file search path (-I).  Returns a list\n    of command-line options suitable for either Unix compilers or Visual\n    C++.\n    ")
    
    # Assigning a List to a Name (line 1030):
    
    # Assigning a List to a Name (line 1030):
    
    # Obtaining an instance of the builtin type 'list' (line 1030)
    list_305328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1030, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1030)
    
    # Assigning a type to the variable 'pp_opts' (line 1030)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1030, 4), 'pp_opts', list_305328)
    
    # Getting the type of 'macros' (line 1031)
    macros_305329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 17), 'macros')
    # Testing the type of a for loop iterable (line 1031)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1031, 4), macros_305329)
    # Getting the type of the for loop variable (line 1031)
    for_loop_var_305330 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1031, 4), macros_305329)
    # Assigning a type to the variable 'macro' (line 1031)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1031, 4), 'macro', for_loop_var_305330)
    # SSA begins for a for statement (line 1031)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 1033)
    # Processing the call arguments (line 1033)
    # Getting the type of 'macro' (line 1033)
    macro_305332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 27), 'macro', False)
    # Getting the type of 'tuple' (line 1033)
    tuple_305333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 34), 'tuple', False)
    # Processing the call keyword arguments (line 1033)
    kwargs_305334 = {}
    # Getting the type of 'isinstance' (line 1033)
    isinstance_305331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 16), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 1033)
    isinstance_call_result_305335 = invoke(stypy.reporting.localization.Localization(__file__, 1033, 16), isinstance_305331, *[macro_305332, tuple_305333], **kwargs_305334)
    
    
    int_305336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 16), 'int')
    
    # Call to len(...): (line 1034)
    # Processing the call arguments (line 1034)
    # Getting the type of 'macro' (line 1034)
    macro_305338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 26), 'macro', False)
    # Processing the call keyword arguments (line 1034)
    kwargs_305339 = {}
    # Getting the type of 'len' (line 1034)
    len_305337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 21), 'len', False)
    # Calling len(args, kwargs) (line 1034)
    len_call_result_305340 = invoke(stypy.reporting.localization.Localization(__file__, 1034, 21), len_305337, *[macro_305338], **kwargs_305339)
    
    # Applying the binary operator '<=' (line 1034)
    result_le_305341 = python_operator(stypy.reporting.localization.Localization(__file__, 1034, 16), '<=', int_305336, len_call_result_305340)
    int_305342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1034, 36), 'int')
    # Applying the binary operator '<=' (line 1034)
    result_le_305343 = python_operator(stypy.reporting.localization.Localization(__file__, 1034, 16), '<=', len_call_result_305340, int_305342)
    # Applying the binary operator '&' (line 1034)
    result_and__305344 = python_operator(stypy.reporting.localization.Localization(__file__, 1034, 16), '&', result_le_305341, result_le_305343)
    
    # Applying the binary operator 'and' (line 1033)
    result_and_keyword_305345 = python_operator(stypy.reporting.localization.Localization(__file__, 1033, 16), 'and', isinstance_call_result_305335, result_and__305344)
    
    # Applying the 'not' unary operator (line 1033)
    result_not__305346 = python_operator(stypy.reporting.localization.Localization(__file__, 1033, 11), 'not', result_and_keyword_305345)
    
    # Testing the type of an if condition (line 1033)
    if_condition_305347 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1033, 8), result_not__305346)
    # Assigning a type to the variable 'if_condition_305347' (line 1033)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1033, 8), 'if_condition_305347', if_condition_305347)
    # SSA begins for if statement (line 1033)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'TypeError' (line 1035)
    TypeError_305348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1035, 18), 'TypeError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1035, 12), TypeError_305348, 'raise parameter', BaseException)
    # SSA join for if statement (line 1033)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 1040)
    # Processing the call arguments (line 1040)
    # Getting the type of 'macro' (line 1040)
    macro_305350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 16), 'macro', False)
    # Processing the call keyword arguments (line 1040)
    kwargs_305351 = {}
    # Getting the type of 'len' (line 1040)
    len_305349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 11), 'len', False)
    # Calling len(args, kwargs) (line 1040)
    len_call_result_305352 = invoke(stypy.reporting.localization.Localization(__file__, 1040, 11), len_305349, *[macro_305350], **kwargs_305351)
    
    int_305353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1040, 26), 'int')
    # Applying the binary operator '==' (line 1040)
    result_eq_305354 = python_operator(stypy.reporting.localization.Localization(__file__, 1040, 11), '==', len_call_result_305352, int_305353)
    
    # Testing the type of an if condition (line 1040)
    if_condition_305355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1040, 8), result_eq_305354)
    # Assigning a type to the variable 'if_condition_305355' (line 1040)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1040, 8), 'if_condition_305355', if_condition_305355)
    # SSA begins for if statement (line 1040)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 1041)
    # Processing the call arguments (line 1041)
    str_305358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 28), 'str', '-U%s')
    
    # Obtaining the type of the subscript
    int_305359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1041, 43), 'int')
    # Getting the type of 'macro' (line 1041)
    macro_305360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1041, 37), 'macro', False)
    # Obtaining the member '__getitem__' of a type (line 1041)
    getitem___305361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1041, 37), macro_305360, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1041)
    subscript_call_result_305362 = invoke(stypy.reporting.localization.Localization(__file__, 1041, 37), getitem___305361, int_305359)
    
    # Applying the binary operator '%' (line 1041)
    result_mod_305363 = python_operator(stypy.reporting.localization.Localization(__file__, 1041, 28), '%', str_305358, subscript_call_result_305362)
    
    # Processing the call keyword arguments (line 1041)
    kwargs_305364 = {}
    # Getting the type of 'pp_opts' (line 1041)
    pp_opts_305356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1041, 12), 'pp_opts', False)
    # Obtaining the member 'append' of a type (line 1041)
    append_305357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1041, 12), pp_opts_305356, 'append')
    # Calling append(args, kwargs) (line 1041)
    append_call_result_305365 = invoke(stypy.reporting.localization.Localization(__file__, 1041, 12), append_305357, *[result_mod_305363], **kwargs_305364)
    
    # SSA branch for the else part of an if statement (line 1040)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Call to len(...): (line 1042)
    # Processing the call arguments (line 1042)
    # Getting the type of 'macro' (line 1042)
    macro_305367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 18), 'macro', False)
    # Processing the call keyword arguments (line 1042)
    kwargs_305368 = {}
    # Getting the type of 'len' (line 1042)
    len_305366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 13), 'len', False)
    # Calling len(args, kwargs) (line 1042)
    len_call_result_305369 = invoke(stypy.reporting.localization.Localization(__file__, 1042, 13), len_305366, *[macro_305367], **kwargs_305368)
    
    int_305370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1042, 28), 'int')
    # Applying the binary operator '==' (line 1042)
    result_eq_305371 = python_operator(stypy.reporting.localization.Localization(__file__, 1042, 13), '==', len_call_result_305369, int_305370)
    
    # Testing the type of an if condition (line 1042)
    if_condition_305372 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1042, 13), result_eq_305371)
    # Assigning a type to the variable 'if_condition_305372' (line 1042)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1042, 13), 'if_condition_305372', if_condition_305372)
    # SSA begins for if statement (line 1042)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 1043)
    
    # Obtaining the type of the subscript
    int_305373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 21), 'int')
    # Getting the type of 'macro' (line 1043)
    macro_305374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 15), 'macro')
    # Obtaining the member '__getitem__' of a type (line 1043)
    getitem___305375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1043, 15), macro_305374, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 1043)
    subscript_call_result_305376 = invoke(stypy.reporting.localization.Localization(__file__, 1043, 15), getitem___305375, int_305373)
    
    # Getting the type of 'None' (line 1043)
    None_305377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 27), 'None')
    
    (may_be_305378, more_types_in_union_305379) = may_be_none(subscript_call_result_305376, None_305377)

    if may_be_305378:

        if more_types_in_union_305379:
            # Runtime conditional SSA (line 1043)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to append(...): (line 1044)
        # Processing the call arguments (line 1044)
        str_305382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1044, 32), 'str', '-D%s')
        
        # Obtaining the type of the subscript
        int_305383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1044, 47), 'int')
        # Getting the type of 'macro' (line 1044)
        macro_305384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 41), 'macro', False)
        # Obtaining the member '__getitem__' of a type (line 1044)
        getitem___305385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1044, 41), macro_305384, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 1044)
        subscript_call_result_305386 = invoke(stypy.reporting.localization.Localization(__file__, 1044, 41), getitem___305385, int_305383)
        
        # Applying the binary operator '%' (line 1044)
        result_mod_305387 = python_operator(stypy.reporting.localization.Localization(__file__, 1044, 32), '%', str_305382, subscript_call_result_305386)
        
        # Processing the call keyword arguments (line 1044)
        kwargs_305388 = {}
        # Getting the type of 'pp_opts' (line 1044)
        pp_opts_305380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 16), 'pp_opts', False)
        # Obtaining the member 'append' of a type (line 1044)
        append_305381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1044, 16), pp_opts_305380, 'append')
        # Calling append(args, kwargs) (line 1044)
        append_call_result_305389 = invoke(stypy.reporting.localization.Localization(__file__, 1044, 16), append_305381, *[result_mod_305387], **kwargs_305388)
        

        if more_types_in_union_305379:
            # Runtime conditional SSA for else branch (line 1043)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_305378) or more_types_in_union_305379):
        
        # Call to append(...): (line 1049)
        # Processing the call arguments (line 1049)
        str_305392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1049, 32), 'str', '-D%s=%s')
        # Getting the type of 'macro' (line 1049)
        macro_305393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 44), 'macro', False)
        # Applying the binary operator '%' (line 1049)
        result_mod_305394 = python_operator(stypy.reporting.localization.Localization(__file__, 1049, 32), '%', str_305392, macro_305393)
        
        # Processing the call keyword arguments (line 1049)
        kwargs_305395 = {}
        # Getting the type of 'pp_opts' (line 1049)
        pp_opts_305390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 16), 'pp_opts', False)
        # Obtaining the member 'append' of a type (line 1049)
        append_305391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1049, 16), pp_opts_305390, 'append')
        # Calling append(args, kwargs) (line 1049)
        append_call_result_305396 = invoke(stypy.reporting.localization.Localization(__file__, 1049, 16), append_305391, *[result_mod_305394], **kwargs_305395)
        

        if (may_be_305378 and more_types_in_union_305379):
            # SSA join for if statement (line 1043)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 1042)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 1040)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'include_dirs' (line 1051)
    include_dirs_305397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1051, 15), 'include_dirs')
    # Testing the type of a for loop iterable (line 1051)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1051, 4), include_dirs_305397)
    # Getting the type of the for loop variable (line 1051)
    for_loop_var_305398 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1051, 4), include_dirs_305397)
    # Assigning a type to the variable 'dir' (line 1051)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1051, 4), 'dir', for_loop_var_305398)
    # SSA begins for a for statement (line 1051)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 1052)
    # Processing the call arguments (line 1052)
    str_305401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1052, 24), 'str', '-I%s')
    # Getting the type of 'dir' (line 1052)
    dir_305402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 33), 'dir', False)
    # Applying the binary operator '%' (line 1052)
    result_mod_305403 = python_operator(stypy.reporting.localization.Localization(__file__, 1052, 24), '%', str_305401, dir_305402)
    
    # Processing the call keyword arguments (line 1052)
    kwargs_305404 = {}
    # Getting the type of 'pp_opts' (line 1052)
    pp_opts_305399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 8), 'pp_opts', False)
    # Obtaining the member 'append' of a type (line 1052)
    append_305400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1052, 8), pp_opts_305399, 'append')
    # Calling append(args, kwargs) (line 1052)
    append_call_result_305405 = invoke(stypy.reporting.localization.Localization(__file__, 1052, 8), append_305400, *[result_mod_305403], **kwargs_305404)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'pp_opts' (line 1054)
    pp_opts_305406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 11), 'pp_opts')
    # Assigning a type to the variable 'stypy_return_type' (line 1054)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1054, 4), 'stypy_return_type', pp_opts_305406)
    
    # ################# End of 'gen_preprocess_options(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gen_preprocess_options' in the type store
    # Getting the type of 'stypy_return_type' (line 1008)
    stypy_return_type_305407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1008, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_305407)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gen_preprocess_options'
    return stypy_return_type_305407

# Assigning a type to the variable 'gen_preprocess_options' (line 1008)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1008, 0), 'gen_preprocess_options', gen_preprocess_options)

@norecursion
def gen_lib_options(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gen_lib_options'
    module_type_store = module_type_store.open_function_context('gen_lib_options', 1057, 0, False)
    
    # Passed parameters checking function
    gen_lib_options.stypy_localization = localization
    gen_lib_options.stypy_type_of_self = None
    gen_lib_options.stypy_type_store = module_type_store
    gen_lib_options.stypy_function_name = 'gen_lib_options'
    gen_lib_options.stypy_param_names_list = ['compiler', 'library_dirs', 'runtime_library_dirs', 'libraries']
    gen_lib_options.stypy_varargs_param_name = None
    gen_lib_options.stypy_kwargs_param_name = None
    gen_lib_options.stypy_call_defaults = defaults
    gen_lib_options.stypy_call_varargs = varargs
    gen_lib_options.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gen_lib_options', ['compiler', 'library_dirs', 'runtime_library_dirs', 'libraries'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gen_lib_options', localization, ['compiler', 'library_dirs', 'runtime_library_dirs', 'libraries'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gen_lib_options(...)' code ##################

    str_305408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1065, (-1)), 'str', "Generate linker options for searching library directories and\n    linking with specific libraries.\n\n    'libraries' and 'library_dirs' are, respectively, lists of library names\n    (not filenames!) and search directories.  Returns a list of command-line\n    options suitable for use with some compiler (depending on the two format\n    strings passed in).\n    ")
    
    # Assigning a List to a Name (line 1066):
    
    # Assigning a List to a Name (line 1066):
    
    # Obtaining an instance of the builtin type 'list' (line 1066)
    list_305409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1066, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1066)
    
    # Assigning a type to the variable 'lib_opts' (line 1066)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1066, 4), 'lib_opts', list_305409)
    
    # Getting the type of 'library_dirs' (line 1068)
    library_dirs_305410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1068, 15), 'library_dirs')
    # Testing the type of a for loop iterable (line 1068)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1068, 4), library_dirs_305410)
    # Getting the type of the for loop variable (line 1068)
    for_loop_var_305411 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1068, 4), library_dirs_305410)
    # Assigning a type to the variable 'dir' (line 1068)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1068, 4), 'dir', for_loop_var_305411)
    # SSA begins for a for statement (line 1068)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 1069)
    # Processing the call arguments (line 1069)
    
    # Call to library_dir_option(...): (line 1069)
    # Processing the call arguments (line 1069)
    # Getting the type of 'dir' (line 1069)
    dir_305416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 52), 'dir', False)
    # Processing the call keyword arguments (line 1069)
    kwargs_305417 = {}
    # Getting the type of 'compiler' (line 1069)
    compiler_305414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 24), 'compiler', False)
    # Obtaining the member 'library_dir_option' of a type (line 1069)
    library_dir_option_305415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1069, 24), compiler_305414, 'library_dir_option')
    # Calling library_dir_option(args, kwargs) (line 1069)
    library_dir_option_call_result_305418 = invoke(stypy.reporting.localization.Localization(__file__, 1069, 24), library_dir_option_305415, *[dir_305416], **kwargs_305417)
    
    # Processing the call keyword arguments (line 1069)
    kwargs_305419 = {}
    # Getting the type of 'lib_opts' (line 1069)
    lib_opts_305412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 8), 'lib_opts', False)
    # Obtaining the member 'append' of a type (line 1069)
    append_305413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1069, 8), lib_opts_305412, 'append')
    # Calling append(args, kwargs) (line 1069)
    append_call_result_305420 = invoke(stypy.reporting.localization.Localization(__file__, 1069, 8), append_305413, *[library_dir_option_call_result_305418], **kwargs_305419)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'runtime_library_dirs' (line 1071)
    runtime_library_dirs_305421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1071, 15), 'runtime_library_dirs')
    # Testing the type of a for loop iterable (line 1071)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1071, 4), runtime_library_dirs_305421)
    # Getting the type of the for loop variable (line 1071)
    for_loop_var_305422 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1071, 4), runtime_library_dirs_305421)
    # Assigning a type to the variable 'dir' (line 1071)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1071, 4), 'dir', for_loop_var_305422)
    # SSA begins for a for statement (line 1071)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 1072):
    
    # Assigning a Call to a Name (line 1072):
    
    # Call to runtime_library_dir_option(...): (line 1072)
    # Processing the call arguments (line 1072)
    # Getting the type of 'dir' (line 1072)
    dir_305425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 50), 'dir', False)
    # Processing the call keyword arguments (line 1072)
    kwargs_305426 = {}
    # Getting the type of 'compiler' (line 1072)
    compiler_305423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 14), 'compiler', False)
    # Obtaining the member 'runtime_library_dir_option' of a type (line 1072)
    runtime_library_dir_option_305424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1072, 14), compiler_305423, 'runtime_library_dir_option')
    # Calling runtime_library_dir_option(args, kwargs) (line 1072)
    runtime_library_dir_option_call_result_305427 = invoke(stypy.reporting.localization.Localization(__file__, 1072, 14), runtime_library_dir_option_305424, *[dir_305425], **kwargs_305426)
    
    # Assigning a type to the variable 'opt' (line 1072)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1072, 8), 'opt', runtime_library_dir_option_call_result_305427)
    
    # Type idiom detected: calculating its left and rigth part (line 1073)
    # Getting the type of 'list' (line 1073)
    list_305428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 27), 'list')
    # Getting the type of 'opt' (line 1073)
    opt_305429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1073, 22), 'opt')
    
    (may_be_305430, more_types_in_union_305431) = may_be_subtype(list_305428, opt_305429)

    if may_be_305430:

        if more_types_in_union_305431:
            # Runtime conditional SSA (line 1073)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'opt' (line 1073)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1073, 8), 'opt', remove_not_subtype_from_union(opt_305429, list))
        
        # Call to extend(...): (line 1074)
        # Processing the call arguments (line 1074)
        # Getting the type of 'opt' (line 1074)
        opt_305434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 28), 'opt', False)
        # Processing the call keyword arguments (line 1074)
        kwargs_305435 = {}
        # Getting the type of 'lib_opts' (line 1074)
        lib_opts_305432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1074, 12), 'lib_opts', False)
        # Obtaining the member 'extend' of a type (line 1074)
        extend_305433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1074, 12), lib_opts_305432, 'extend')
        # Calling extend(args, kwargs) (line 1074)
        extend_call_result_305436 = invoke(stypy.reporting.localization.Localization(__file__, 1074, 12), extend_305433, *[opt_305434], **kwargs_305435)
        

        if more_types_in_union_305431:
            # Runtime conditional SSA for else branch (line 1073)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_305430) or more_types_in_union_305431):
        # Assigning a type to the variable 'opt' (line 1073)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1073, 8), 'opt', remove_subtype_from_union(opt_305429, list))
        
        # Call to append(...): (line 1076)
        # Processing the call arguments (line 1076)
        # Getting the type of 'opt' (line 1076)
        opt_305439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 28), 'opt', False)
        # Processing the call keyword arguments (line 1076)
        kwargs_305440 = {}
        # Getting the type of 'lib_opts' (line 1076)
        lib_opts_305437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1076, 12), 'lib_opts', False)
        # Obtaining the member 'append' of a type (line 1076)
        append_305438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1076, 12), lib_opts_305437, 'append')
        # Calling append(args, kwargs) (line 1076)
        append_call_result_305441 = invoke(stypy.reporting.localization.Localization(__file__, 1076, 12), append_305438, *[opt_305439], **kwargs_305440)
        

        if (may_be_305430 and more_types_in_union_305431):
            # SSA join for if statement (line 1073)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'libraries' (line 1084)
    libraries_305442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1084, 15), 'libraries')
    # Testing the type of a for loop iterable (line 1084)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 1084, 4), libraries_305442)
    # Getting the type of the for loop variable (line 1084)
    for_loop_var_305443 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 1084, 4), libraries_305442)
    # Assigning a type to the variable 'lib' (line 1084)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1084, 4), 'lib', for_loop_var_305443)
    # SSA begins for a for statement (line 1084)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 1085):
    
    # Assigning a Call to a Name:
    
    # Call to split(...): (line 1085)
    # Processing the call arguments (line 1085)
    # Getting the type of 'lib' (line 1085)
    lib_305447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 42), 'lib', False)
    # Processing the call keyword arguments (line 1085)
    kwargs_305448 = {}
    # Getting the type of 'os' (line 1085)
    os_305444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 28), 'os', False)
    # Obtaining the member 'path' of a type (line 1085)
    path_305445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 28), os_305444, 'path')
    # Obtaining the member 'split' of a type (line 1085)
    split_305446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 28), path_305445, 'split')
    # Calling split(args, kwargs) (line 1085)
    split_call_result_305449 = invoke(stypy.reporting.localization.Localization(__file__, 1085, 28), split_305446, *[lib_305447], **kwargs_305448)
    
    # Assigning a type to the variable 'call_assignment_303831' (line 1085)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1085, 8), 'call_assignment_303831', split_call_result_305449)
    
    # Assigning a Call to a Name (line 1085):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_305452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1085, 8), 'int')
    # Processing the call keyword arguments
    kwargs_305453 = {}
    # Getting the type of 'call_assignment_303831' (line 1085)
    call_assignment_303831_305450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 8), 'call_assignment_303831', False)
    # Obtaining the member '__getitem__' of a type (line 1085)
    getitem___305451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 8), call_assignment_303831_305450, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_305454 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___305451, *[int_305452], **kwargs_305453)
    
    # Assigning a type to the variable 'call_assignment_303832' (line 1085)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1085, 8), 'call_assignment_303832', getitem___call_result_305454)
    
    # Assigning a Name to a Name (line 1085):
    # Getting the type of 'call_assignment_303832' (line 1085)
    call_assignment_303832_305455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 8), 'call_assignment_303832')
    # Assigning a type to the variable 'lib_dir' (line 1085)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1085, 8), 'lib_dir', call_assignment_303832_305455)
    
    # Assigning a Call to a Name (line 1085):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_305458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1085, 8), 'int')
    # Processing the call keyword arguments
    kwargs_305459 = {}
    # Getting the type of 'call_assignment_303831' (line 1085)
    call_assignment_303831_305456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 8), 'call_assignment_303831', False)
    # Obtaining the member '__getitem__' of a type (line 1085)
    getitem___305457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1085, 8), call_assignment_303831_305456, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_305460 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___305457, *[int_305458], **kwargs_305459)
    
    # Assigning a type to the variable 'call_assignment_303833' (line 1085)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1085, 8), 'call_assignment_303833', getitem___call_result_305460)
    
    # Assigning a Name to a Name (line 1085):
    # Getting the type of 'call_assignment_303833' (line 1085)
    call_assignment_303833_305461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1085, 8), 'call_assignment_303833')
    # Assigning a type to the variable 'lib_name' (line 1085)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1085, 17), 'lib_name', call_assignment_303833_305461)
    
    
    # Getting the type of 'lib_dir' (line 1086)
    lib_dir_305462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1086, 11), 'lib_dir')
    str_305463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1086, 22), 'str', '')
    # Applying the binary operator '!=' (line 1086)
    result_ne_305464 = python_operator(stypy.reporting.localization.Localization(__file__, 1086, 11), '!=', lib_dir_305462, str_305463)
    
    # Testing the type of an if condition (line 1086)
    if_condition_305465 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1086, 8), result_ne_305464)
    # Assigning a type to the variable 'if_condition_305465' (line 1086)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1086, 8), 'if_condition_305465', if_condition_305465)
    # SSA begins for if statement (line 1086)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 1087):
    
    # Assigning a Call to a Name (line 1087):
    
    # Call to find_library_file(...): (line 1087)
    # Processing the call arguments (line 1087)
    
    # Obtaining an instance of the builtin type 'list' (line 1087)
    list_305468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1087, 50), 'list')
    # Adding type elements to the builtin type 'list' instance (line 1087)
    # Adding element type (line 1087)
    # Getting the type of 'lib_dir' (line 1087)
    lib_dir_305469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1087, 51), 'lib_dir', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1087, 50), list_305468, lib_dir_305469)
    
    # Getting the type of 'lib_name' (line 1087)
    lib_name_305470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1087, 61), 'lib_name', False)
    # Processing the call keyword arguments (line 1087)
    kwargs_305471 = {}
    # Getting the type of 'compiler' (line 1087)
    compiler_305466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1087, 23), 'compiler', False)
    # Obtaining the member 'find_library_file' of a type (line 1087)
    find_library_file_305467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1087, 23), compiler_305466, 'find_library_file')
    # Calling find_library_file(args, kwargs) (line 1087)
    find_library_file_call_result_305472 = invoke(stypy.reporting.localization.Localization(__file__, 1087, 23), find_library_file_305467, *[list_305468, lib_name_305470], **kwargs_305471)
    
    # Assigning a type to the variable 'lib_file' (line 1087)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1087, 12), 'lib_file', find_library_file_call_result_305472)
    
    # Type idiom detected: calculating its left and rigth part (line 1088)
    # Getting the type of 'lib_file' (line 1088)
    lib_file_305473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 12), 'lib_file')
    # Getting the type of 'None' (line 1088)
    None_305474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1088, 31), 'None')
    
    (may_be_305475, more_types_in_union_305476) = may_not_be_none(lib_file_305473, None_305474)

    if may_be_305475:

        if more_types_in_union_305476:
            # Runtime conditional SSA (line 1088)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to append(...): (line 1089)
        # Processing the call arguments (line 1089)
        # Getting the type of 'lib_file' (line 1089)
        lib_file_305479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 32), 'lib_file', False)
        # Processing the call keyword arguments (line 1089)
        kwargs_305480 = {}
        # Getting the type of 'lib_opts' (line 1089)
        lib_opts_305477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1089, 16), 'lib_opts', False)
        # Obtaining the member 'append' of a type (line 1089)
        append_305478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1089, 16), lib_opts_305477, 'append')
        # Calling append(args, kwargs) (line 1089)
        append_call_result_305481 = invoke(stypy.reporting.localization.Localization(__file__, 1089, 16), append_305478, *[lib_file_305479], **kwargs_305480)
        

        if more_types_in_union_305476:
            # Runtime conditional SSA for else branch (line 1088)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_305475) or more_types_in_union_305476):
        
        # Call to warn(...): (line 1091)
        # Processing the call arguments (line 1091)
        str_305484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1091, 30), 'str', "no library file corresponding to '%s' found (skipping)")
        # Getting the type of 'lib' (line 1092)
        lib_305485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1092, 56), 'lib', False)
        # Applying the binary operator '%' (line 1091)
        result_mod_305486 = python_operator(stypy.reporting.localization.Localization(__file__, 1091, 30), '%', str_305484, lib_305485)
        
        # Processing the call keyword arguments (line 1091)
        kwargs_305487 = {}
        # Getting the type of 'compiler' (line 1091)
        compiler_305482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1091, 16), 'compiler', False)
        # Obtaining the member 'warn' of a type (line 1091)
        warn_305483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1091, 16), compiler_305482, 'warn')
        # Calling warn(args, kwargs) (line 1091)
        warn_call_result_305488 = invoke(stypy.reporting.localization.Localization(__file__, 1091, 16), warn_305483, *[result_mod_305486], **kwargs_305487)
        

        if (may_be_305475 and more_types_in_union_305476):
            # SSA join for if statement (line 1088)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the else part of an if statement (line 1086)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 1094)
    # Processing the call arguments (line 1094)
    
    # Call to library_option(...): (line 1094)
    # Processing the call arguments (line 1094)
    # Getting the type of 'lib' (line 1094)
    lib_305493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 52), 'lib', False)
    # Processing the call keyword arguments (line 1094)
    kwargs_305494 = {}
    # Getting the type of 'compiler' (line 1094)
    compiler_305491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 28), 'compiler', False)
    # Obtaining the member 'library_option' of a type (line 1094)
    library_option_305492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 28), compiler_305491, 'library_option')
    # Calling library_option(args, kwargs) (line 1094)
    library_option_call_result_305495 = invoke(stypy.reporting.localization.Localization(__file__, 1094, 28), library_option_305492, *[lib_305493], **kwargs_305494)
    
    # Processing the call keyword arguments (line 1094)
    kwargs_305496 = {}
    # Getting the type of 'lib_opts' (line 1094)
    lib_opts_305489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1094, 12), 'lib_opts', False)
    # Obtaining the member 'append' of a type (line 1094)
    append_305490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1094, 12), lib_opts_305489, 'append')
    # Calling append(args, kwargs) (line 1094)
    append_call_result_305497 = invoke(stypy.reporting.localization.Localization(__file__, 1094, 12), append_305490, *[library_option_call_result_305495], **kwargs_305496)
    
    # SSA join for if statement (line 1086)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'lib_opts' (line 1096)
    lib_opts_305498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1096, 11), 'lib_opts')
    # Assigning a type to the variable 'stypy_return_type' (line 1096)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1096, 4), 'stypy_return_type', lib_opts_305498)
    
    # ################# End of 'gen_lib_options(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gen_lib_options' in the type store
    # Getting the type of 'stypy_return_type' (line 1057)
    stypy_return_type_305499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_305499)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gen_lib_options'
    return stypy_return_type_305499

# Assigning a type to the variable 'gen_lib_options' (line 1057)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1057, 0), 'gen_lib_options', gen_lib_options)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
