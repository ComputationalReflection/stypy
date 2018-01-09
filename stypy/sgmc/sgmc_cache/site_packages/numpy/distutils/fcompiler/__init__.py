
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''numpy.distutils.fcompiler
2: 
3: Contains FCompiler, an abstract base class that defines the interface
4: for the numpy.distutils Fortran compiler abstraction model.
5: 
6: Terminology:
7: 
8: To be consistent, where the term 'executable' is used, it means the single
9: file, like 'gcc', that is executed, and should be a string. In contrast,
10: 'command' means the entire command line, like ['gcc', '-c', 'file.c'], and
11: should be a list.
12: 
13: But note that FCompiler.executables is actually a dictionary of commands.
14: 
15: '''
16: from __future__ import division, absolute_import, print_function
17: 
18: __all__ = ['FCompiler', 'new_fcompiler', 'show_fcompilers',
19:            'dummy_fortran_file']
20: 
21: import os
22: import sys
23: import re
24: import types
25: try:
26:     set
27: except NameError:
28:     from sets import Set as set
29: 
30: from numpy.compat import open_latin1
31: 
32: from distutils.sysconfig import get_python_lib
33: from distutils.fancy_getopt import FancyGetopt
34: from distutils.errors import DistutilsModuleError, \
35:      DistutilsExecError, CompileError, LinkError, DistutilsPlatformError
36: from distutils.util import split_quoted, strtobool
37: 
38: from numpy.distutils.ccompiler import CCompiler, gen_lib_options
39: from numpy.distutils import log
40: from numpy.distutils.misc_util import is_string, all_strings, is_sequence, \
41:     make_temp_file, get_shared_lib_extension
42: from numpy.distutils.environment import EnvironmentConfig
43: from numpy.distutils.exec_command import find_executable
44: from numpy.distutils.compat import get_exception
45: 
46: __metaclass__ = type
47: 
48: class CompilerNotFound(Exception):
49:     pass
50: 
51: def flaglist(s):
52:     if is_string(s):
53:         return split_quoted(s)
54:     else:
55:         return s
56: 
57: def str2bool(s):
58:     if is_string(s):
59:         return strtobool(s)
60:     return bool(s)
61: 
62: def is_sequence_of_strings(seq):
63:     return is_sequence(seq) and all_strings(seq)
64: 
65: class FCompiler(CCompiler):
66:     '''Abstract base class to define the interface that must be implemented
67:     by real Fortran compiler classes.
68: 
69:     Methods that subclasses may redefine:
70: 
71:         update_executables(), find_executables(), get_version()
72:         get_flags(), get_flags_opt(), get_flags_arch(), get_flags_debug()
73:         get_flags_f77(), get_flags_opt_f77(), get_flags_arch_f77(),
74:         get_flags_debug_f77(), get_flags_f90(), get_flags_opt_f90(),
75:         get_flags_arch_f90(), get_flags_debug_f90(),
76:         get_flags_fix(), get_flags_linker_so()
77: 
78:     DON'T call these methods (except get_version) after
79:     constructing a compiler instance or inside any other method.
80:     All methods, except update_executables() and find_executables(),
81:     may call the get_version() method.
82: 
83:     After constructing a compiler instance, always call customize(dist=None)
84:     method that finalizes compiler construction and makes the following
85:     attributes available:
86:       compiler_f77
87:       compiler_f90
88:       compiler_fix
89:       linker_so
90:       archiver
91:       ranlib
92:       libraries
93:       library_dirs
94:     '''
95: 
96:     # These are the environment variables and distutils keys used.
97:     # Each configuration descripition is
98:     # (<hook name>, <environment variable>, <key in distutils.cfg>, <convert>)
99:     # The hook names are handled by the self._environment_hook method.
100:     #  - names starting with 'self.' call methods in this class
101:     #  - names starting with 'exe.' return the key in the executables dict
102:     #  - names like 'flags.YYY' return self.get_flag_YYY()
103:     # convert is either None or a function to convert a string to the
104:     # appropiate type used.
105: 
106:     distutils_vars = EnvironmentConfig(
107:         distutils_section='config_fc',
108:         noopt = (None, None, 'noopt', str2bool),
109:         noarch = (None, None, 'noarch', str2bool),
110:         debug = (None, None, 'debug', str2bool),
111:         verbose = (None, None, 'verbose', str2bool),
112:     )
113: 
114:     command_vars = EnvironmentConfig(
115:         distutils_section='config_fc',
116:         compiler_f77 = ('exe.compiler_f77', 'F77', 'f77exec', None),
117:         compiler_f90 = ('exe.compiler_f90', 'F90', 'f90exec', None),
118:         compiler_fix = ('exe.compiler_fix', 'F90', 'f90exec', None),
119:         version_cmd = ('exe.version_cmd', None, None, None),
120:         linker_so = ('exe.linker_so', 'LDSHARED', 'ldshared', None),
121:         linker_exe = ('exe.linker_exe', 'LD', 'ld', None),
122:         archiver = (None, 'AR', 'ar', None),
123:         ranlib = (None, 'RANLIB', 'ranlib', None),
124:     )
125: 
126:     flag_vars = EnvironmentConfig(
127:         distutils_section='config_fc',
128:         f77 = ('flags.f77', 'F77FLAGS', 'f77flags', flaglist),
129:         f90 = ('flags.f90', 'F90FLAGS', 'f90flags', flaglist),
130:         free = ('flags.free', 'FREEFLAGS', 'freeflags', flaglist),
131:         fix = ('flags.fix', None, None, flaglist),
132:         opt = ('flags.opt', 'FOPT', 'opt', flaglist),
133:         opt_f77 = ('flags.opt_f77', None, None, flaglist),
134:         opt_f90 = ('flags.opt_f90', None, None, flaglist),
135:         arch = ('flags.arch', 'FARCH', 'arch', flaglist),
136:         arch_f77 = ('flags.arch_f77', None, None, flaglist),
137:         arch_f90 = ('flags.arch_f90', None, None, flaglist),
138:         debug = ('flags.debug', 'FDEBUG', 'fdebug', flaglist),
139:         debug_f77 = ('flags.debug_f77', None, None, flaglist),
140:         debug_f90 = ('flags.debug_f90', None, None, flaglist),
141:         flags = ('self.get_flags', 'FFLAGS', 'fflags', flaglist),
142:         linker_so = ('flags.linker_so', 'LDFLAGS', 'ldflags', flaglist),
143:         linker_exe = ('flags.linker_exe', 'LDFLAGS', 'ldflags', flaglist),
144:         ar = ('flags.ar', 'ARFLAGS', 'arflags', flaglist),
145:     )
146: 
147:     language_map = {'.f': 'f77',
148:                     '.for': 'f77',
149:                     '.F': 'f77',    # XXX: needs preprocessor
150:                     '.ftn': 'f77',
151:                     '.f77': 'f77',
152:                     '.f90': 'f90',
153:                     '.F90': 'f90',  # XXX: needs preprocessor
154:                     '.f95': 'f90',
155:                     }
156:     language_order = ['f90', 'f77']
157: 
158: 
159:     # These will be set by the subclass
160: 
161:     compiler_type = None
162:     compiler_aliases = ()
163:     version_pattern = None
164: 
165:     possible_executables = []
166:     executables = {
167:         'version_cmd': ["f77", "-v"],
168:         'compiler_f77': ["f77"],
169:         'compiler_f90': ["f90"],
170:         'compiler_fix': ["f90", "-fixed"],
171:         'linker_so': ["f90", "-shared"],
172:         'linker_exe': ["f90"],
173:         'archiver': ["ar", "-cr"],
174:         'ranlib': None,
175:         }
176: 
177:     # If compiler does not support compiling Fortran 90 then it can
178:     # suggest using another compiler. For example, gnu would suggest
179:     # gnu95 compiler type when there are F90 sources.
180:     suggested_f90_compiler = None
181: 
182:     compile_switch = "-c"
183:     object_switch = "-o "   # Ending space matters! It will be stripped
184:                             # but if it is missing then object_switch
185:                             # will be prefixed to object file name by
186:                             # string concatenation.
187:     library_switch = "-o "  # Ditto!
188: 
189:     # Switch to specify where module files are created and searched
190:     # for USE statement.  Normally it is a string and also here ending
191:     # space matters. See above.
192:     module_dir_switch = None
193: 
194:     # Switch to specify where module files are searched for USE statement.
195:     module_include_switch = '-I'
196: 
197:     pic_flags = []           # Flags to create position-independent code
198: 
199:     src_extensions = ['.for', '.ftn', '.f77', '.f', '.f90', '.f95', '.F', '.F90', '.FOR']
200:     obj_extension = ".o"
201: 
202:     shared_lib_extension = get_shared_lib_extension()
203:     static_lib_extension = ".a"  # or .lib
204:     static_lib_format = "lib%s%s" # or %s%s
205:     shared_lib_format = "%s%s"
206:     exe_extension = ""
207: 
208:     _exe_cache = {}
209: 
210:     _executable_keys = ['version_cmd', 'compiler_f77', 'compiler_f90',
211:                         'compiler_fix', 'linker_so', 'linker_exe', 'archiver',
212:                         'ranlib']
213: 
214:     # This will be set by new_fcompiler when called in
215:     # command/{build_ext.py, build_clib.py, config.py} files.
216:     c_compiler = None
217: 
218:     # extra_{f77,f90}_compile_args are set by build_ext.build_extension method
219:     extra_f77_compile_args = []
220:     extra_f90_compile_args = []
221: 
222:     def __init__(self, *args, **kw):
223:         CCompiler.__init__(self, *args, **kw)
224:         self.distutils_vars = self.distutils_vars.clone(self._environment_hook)
225:         self.command_vars = self.command_vars.clone(self._environment_hook)
226:         self.flag_vars = self.flag_vars.clone(self._environment_hook)
227:         self.executables = self.executables.copy()
228:         for e in self._executable_keys:
229:             if e not in self.executables:
230:                 self.executables[e] = None
231: 
232:         # Some methods depend on .customize() being called first, so
233:         # this keeps track of whether that's happened yet.
234:         self._is_customised = False
235: 
236:     def __copy__(self):
237:         obj = self.__new__(self.__class__)
238:         obj.__dict__.update(self.__dict__)
239:         obj.distutils_vars = obj.distutils_vars.clone(obj._environment_hook)
240:         obj.command_vars = obj.command_vars.clone(obj._environment_hook)
241:         obj.flag_vars = obj.flag_vars.clone(obj._environment_hook)
242:         obj.executables = obj.executables.copy()
243:         return obj
244: 
245:     def copy(self):
246:         return self.__copy__()
247: 
248:     # Use properties for the attributes used by CCompiler. Setting them
249:     # as attributes from the self.executables dictionary is error-prone,
250:     # so we get them from there each time.
251:     def _command_property(key):
252:         def fget(self):
253:             assert self._is_customised
254:             return self.executables[key]
255:         return property(fget=fget)
256:     version_cmd = _command_property('version_cmd')
257:     compiler_f77 = _command_property('compiler_f77')
258:     compiler_f90 = _command_property('compiler_f90')
259:     compiler_fix = _command_property('compiler_fix')
260:     linker_so = _command_property('linker_so')
261:     linker_exe = _command_property('linker_exe')
262:     archiver = _command_property('archiver')
263:     ranlib = _command_property('ranlib')
264: 
265:     # Make our terminology consistent.
266:     def set_executable(self, key, value):
267:         self.set_command(key, value)
268: 
269:     def set_commands(self, **kw):
270:         for k, v in kw.items():
271:             self.set_command(k, v)
272: 
273:     def set_command(self, key, value):
274:         if not key in self._executable_keys:
275:             raise ValueError(
276:                 "unknown executable '%s' for class %s" %
277:                 (key, self.__class__.__name__))
278:         if is_string(value):
279:             value = split_quoted(value)
280:         assert value is None or is_sequence_of_strings(value[1:]), (key, value)
281:         self.executables[key] = value
282: 
283:     ######################################################################
284:     ## Methods that subclasses may redefine. But don't call these methods!
285:     ## They are private to FCompiler class and may return unexpected
286:     ## results if used elsewhere. So, you have been warned..
287: 
288:     def find_executables(self):
289:         '''Go through the self.executables dictionary, and attempt to
290:         find and assign appropiate executables.
291: 
292:         Executable names are looked for in the environment (environment
293:         variables, the distutils.cfg, and command line), the 0th-element of
294:         the command list, and the self.possible_executables list.
295: 
296:         Also, if the 0th element is "<F77>" or "<F90>", the Fortran 77
297:         or the Fortran 90 compiler executable is used, unless overridden
298:         by an environment setting.
299: 
300:         Subclasses should call this if overriden.
301:         '''
302:         assert self._is_customised
303:         exe_cache = self._exe_cache
304:         def cached_find_executable(exe):
305:             if exe in exe_cache:
306:                 return exe_cache[exe]
307:             fc_exe = find_executable(exe)
308:             exe_cache[exe] = exe_cache[fc_exe] = fc_exe
309:             return fc_exe
310:         def verify_command_form(name, value):
311:             if value is not None and not is_sequence_of_strings(value):
312:                 raise ValueError(
313:                     "%s value %r is invalid in class %s" %
314:                     (name, value, self.__class__.__name__))
315:         def set_exe(exe_key, f77=None, f90=None):
316:             cmd = self.executables.get(exe_key, None)
317:             if not cmd:
318:                 return None
319:             # Note that we get cmd[0] here if the environment doesn't
320:             # have anything set
321:             exe_from_environ = getattr(self.command_vars, exe_key)
322:             if not exe_from_environ:
323:                 possibles = [f90, f77] + self.possible_executables
324:             else:
325:                 possibles = [exe_from_environ] + self.possible_executables
326: 
327:             seen = set()
328:             unique_possibles = []
329:             for e in possibles:
330:                 if e == '<F77>':
331:                     e = f77
332:                 elif e == '<F90>':
333:                     e = f90
334:                 if not e or e in seen:
335:                     continue
336:                 seen.add(e)
337:                 unique_possibles.append(e)
338: 
339:             for exe in unique_possibles:
340:                 fc_exe = cached_find_executable(exe)
341:                 if fc_exe:
342:                     cmd[0] = fc_exe
343:                     return fc_exe
344:             self.set_command(exe_key, None)
345:             return None
346: 
347:         ctype = self.compiler_type
348:         f90 = set_exe('compiler_f90')
349:         if not f90:
350:             f77 = set_exe('compiler_f77')
351:             if f77:
352:                 log.warn('%s: no Fortran 90 compiler found' % ctype)
353:             else:
354:                 raise CompilerNotFound('%s: f90 nor f77' % ctype)
355:         else:
356:             f77 = set_exe('compiler_f77', f90=f90)
357:             if not f77:
358:                 log.warn('%s: no Fortran 77 compiler found' % ctype)
359:             set_exe('compiler_fix', f90=f90)
360: 
361:         set_exe('linker_so', f77=f77, f90=f90)
362:         set_exe('linker_exe', f77=f77, f90=f90)
363:         set_exe('version_cmd', f77=f77, f90=f90)
364:         set_exe('archiver')
365:         set_exe('ranlib')
366: 
367:     def update_executables(elf):
368:         '''Called at the beginning of customisation. Subclasses should
369:         override this if they need to set up the executables dictionary.
370: 
371:         Note that self.find_executables() is run afterwards, so the
372:         self.executables dictionary values can contain <F77> or <F90> as
373:         the command, which will be replaced by the found F77 or F90
374:         compiler.
375:         '''
376:         pass
377: 
378:     def get_flags(self):
379:         '''List of flags common to all compiler types.'''
380:         return [] + self.pic_flags
381: 
382:     def _get_command_flags(self, key):
383:         cmd = self.executables.get(key, None)
384:         if cmd is None:
385:             return []
386:         return cmd[1:]
387: 
388:     def get_flags_f77(self):
389:         '''List of Fortran 77 specific flags.'''
390:         return self._get_command_flags('compiler_f77')
391:     def get_flags_f90(self):
392:         '''List of Fortran 90 specific flags.'''
393:         return self._get_command_flags('compiler_f90')
394:     def get_flags_free(self):
395:         '''List of Fortran 90 free format specific flags.'''
396:         return []
397:     def get_flags_fix(self):
398:         '''List of Fortran 90 fixed format specific flags.'''
399:         return self._get_command_flags('compiler_fix')
400:     def get_flags_linker_so(self):
401:         '''List of linker flags to build a shared library.'''
402:         return self._get_command_flags('linker_so')
403:     def get_flags_linker_exe(self):
404:         '''List of linker flags to build an executable.'''
405:         return self._get_command_flags('linker_exe')
406:     def get_flags_ar(self):
407:         '''List of archiver flags. '''
408:         return self._get_command_flags('archiver')
409:     def get_flags_opt(self):
410:         '''List of architecture independent compiler flags.'''
411:         return []
412:     def get_flags_arch(self):
413:         '''List of architecture dependent compiler flags.'''
414:         return []
415:     def get_flags_debug(self):
416:         '''List of compiler flags to compile with debugging information.'''
417:         return []
418: 
419:     get_flags_opt_f77 = get_flags_opt_f90 = get_flags_opt
420:     get_flags_arch_f77 = get_flags_arch_f90 = get_flags_arch
421:     get_flags_debug_f77 = get_flags_debug_f90 = get_flags_debug
422: 
423:     def get_libraries(self):
424:         '''List of compiler libraries.'''
425:         return self.libraries[:]
426:     def get_library_dirs(self):
427:         '''List of compiler library directories.'''
428:         return self.library_dirs[:]
429: 
430:     def get_version(self, force=False, ok_status=[0]):
431:         assert self._is_customised
432:         version = CCompiler.get_version(self, force=force, ok_status=ok_status)
433:         if version is None:
434:             raise CompilerNotFound()
435:         return version
436: 
437:     ############################################################
438: 
439:     ## Public methods:
440: 
441:     def customize(self, dist = None):
442:         '''Customize Fortran compiler.
443: 
444:         This method gets Fortran compiler specific information from
445:         (i) class definition, (ii) environment, (iii) distutils config
446:         files, and (iv) command line (later overrides earlier).
447: 
448:         This method should be always called after constructing a
449:         compiler instance. But not in __init__ because Distribution
450:         instance is needed for (iii) and (iv).
451:         '''
452:         log.info('customize %s' % (self.__class__.__name__))
453: 
454:         self._is_customised = True
455: 
456:         self.distutils_vars.use_distribution(dist)
457:         self.command_vars.use_distribution(dist)
458:         self.flag_vars.use_distribution(dist)
459: 
460:         self.update_executables()
461: 
462:         # find_executables takes care of setting the compiler commands,
463:         # version_cmd, linker_so, linker_exe, ar, and ranlib
464:         self.find_executables()
465: 
466:         noopt = self.distutils_vars.get('noopt', False)
467:         noarch = self.distutils_vars.get('noarch', noopt)
468:         debug = self.distutils_vars.get('debug', False)
469: 
470:         f77 = self.command_vars.compiler_f77
471:         f90 = self.command_vars.compiler_f90
472: 
473:         f77flags = []
474:         f90flags = []
475:         freeflags = []
476:         fixflags = []
477: 
478:         if f77:
479:             f77flags = self.flag_vars.f77
480:         if f90:
481:             f90flags = self.flag_vars.f90
482:             freeflags = self.flag_vars.free
483:         # XXX Assuming that free format is default for f90 compiler.
484:         fix = self.command_vars.compiler_fix
485:         if fix:
486:             fixflags = self.flag_vars.fix + f90flags
487: 
488:         oflags, aflags, dflags = [], [], []
489:         # examine get_flags_<tag>_<compiler> for extra flags
490:         # only add them if the method is different from get_flags_<tag>
491:         def get_flags(tag, flags):
492:             # note that self.flag_vars.<tag> calls self.get_flags_<tag>()
493:             flags.extend(getattr(self.flag_vars, tag))
494:             this_get = getattr(self, 'get_flags_' + tag)
495:             for name, c, flagvar in [('f77', f77, f77flags),
496:                                      ('f90', f90, f90flags),
497:                                      ('f90', fix, fixflags)]:
498:                 t = '%s_%s' % (tag, name)
499:                 if c and this_get is not getattr(self, 'get_flags_' + t):
500:                     flagvar.extend(getattr(self.flag_vars, t))
501:         if not noopt:
502:             get_flags('opt', oflags)
503:             if not noarch:
504:                 get_flags('arch', aflags)
505:         if debug:
506:             get_flags('debug', dflags)
507: 
508:         fflags = self.flag_vars.flags + dflags + oflags + aflags
509: 
510:         if f77:
511:             self.set_commands(compiler_f77=[f77]+f77flags+fflags)
512:         if f90:
513:             self.set_commands(compiler_f90=[f90]+freeflags+f90flags+fflags)
514:         if fix:
515:             self.set_commands(compiler_fix=[fix]+fixflags+fflags)
516: 
517: 
518:         #XXX: Do we need LDSHARED->SOSHARED, LDFLAGS->SOFLAGS
519:         linker_so = self.linker_so
520:         if linker_so:
521:             linker_so_flags = self.flag_vars.linker_so
522:             if sys.platform.startswith('aix'):
523:                 python_lib = get_python_lib(standard_lib=1)
524:                 ld_so_aix = os.path.join(python_lib, 'config', 'ld_so_aix')
525:                 python_exp = os.path.join(python_lib, 'config', 'python.exp')
526:                 linker_so = [ld_so_aix] + linker_so + ['-bI:'+python_exp]
527:             self.set_commands(linker_so=linker_so+linker_so_flags)
528: 
529:         linker_exe = self.linker_exe
530:         if linker_exe:
531:             linker_exe_flags = self.flag_vars.linker_exe
532:             self.set_commands(linker_exe=linker_exe+linker_exe_flags)
533: 
534:         ar = self.command_vars.archiver
535:         if ar:
536:             arflags = self.flag_vars.ar
537:             self.set_commands(archiver=[ar]+arflags)
538: 
539:         self.set_library_dirs(self.get_library_dirs())
540:         self.set_libraries(self.get_libraries())
541: 
542:     def dump_properties(self):
543:         '''Print out the attributes of a compiler instance.'''
544:         props = []
545:         for key in list(self.executables.keys()) + \
546:                 ['version', 'libraries', 'library_dirs',
547:                  'object_switch', 'compile_switch']:
548:             if hasattr(self, key):
549:                 v = getattr(self, key)
550:                 props.append((key, None, '= '+repr(v)))
551:         props.sort()
552: 
553:         pretty_printer = FancyGetopt(props)
554:         for l in pretty_printer.generate_help("%s instance properties:" \
555:                                               % (self.__class__.__name__)):
556:             if l[:4]=='  --':
557:                 l = '  ' + l[4:]
558:             print(l)
559: 
560:     ###################
561: 
562:     def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
563:         '''Compile 'src' to product 'obj'.'''
564:         src_flags = {}
565:         if is_f_file(src) and not has_f90_header(src):
566:             flavor = ':f77'
567:             compiler = self.compiler_f77
568:             src_flags = get_f77flags(src)
569:             extra_compile_args = self.extra_f77_compile_args or []
570:         elif is_free_format(src):
571:             flavor = ':f90'
572:             compiler = self.compiler_f90
573:             if compiler is None:
574:                 raise DistutilsExecError('f90 not supported by %s needed for %s'\
575:                       % (self.__class__.__name__, src))
576:             extra_compile_args = self.extra_f90_compile_args or []
577:         else:
578:             flavor = ':fix'
579:             compiler = self.compiler_fix
580:             if compiler is None:
581:                 raise DistutilsExecError('f90 (fixed) not supported by %s needed for %s'\
582:                       % (self.__class__.__name__, src))
583:             extra_compile_args = self.extra_f90_compile_args or []
584:         if self.object_switch[-1]==' ':
585:             o_args = [self.object_switch.strip(), obj]
586:         else:
587:             o_args = [self.object_switch.strip()+obj]
588: 
589:         assert self.compile_switch.strip()
590:         s_args = [self.compile_switch, src]
591: 
592:         if extra_compile_args:
593:             log.info('extra %s options: %r' \
594:                      % (flavor[1:], ' '.join(extra_compile_args)))
595: 
596:         extra_flags = src_flags.get(self.compiler_type, [])
597:         if extra_flags:
598:             log.info('using compile options from source: %r' \
599:                      % ' '.join(extra_flags))
600: 
601:         command = compiler + cc_args + extra_flags + s_args + o_args \
602:                   + extra_postargs + extra_compile_args
603: 
604:         display = '%s: %s' % (os.path.basename(compiler[0]) + flavor,
605:                               src)
606:         try:
607:             self.spawn(command, display=display)
608:         except DistutilsExecError:
609:             msg = str(get_exception())
610:             raise CompileError(msg)
611: 
612:     def module_options(self, module_dirs, module_build_dir):
613:         options = []
614:         if self.module_dir_switch is not None:
615:             if self.module_dir_switch[-1]==' ':
616:                 options.extend([self.module_dir_switch.strip(), module_build_dir])
617:             else:
618:                 options.append(self.module_dir_switch.strip()+module_build_dir)
619:         else:
620:             print('XXX: module_build_dir=%r option ignored' % (module_build_dir))
621:             print('XXX: Fix module_dir_switch for ', self.__class__.__name__)
622:         if self.module_include_switch is not None:
623:             for d in [module_build_dir]+module_dirs:
624:                 options.append('%s%s' % (self.module_include_switch, d))
625:         else:
626:             print('XXX: module_dirs=%r option ignored' % (module_dirs))
627:             print('XXX: Fix module_include_switch for ', self.__class__.__name__)
628:         return options
629: 
630:     def library_option(self, lib):
631:         return "-l" + lib
632:     def library_dir_option(self, dir):
633:         return "-L" + dir
634: 
635:     def link(self, target_desc, objects,
636:              output_filename, output_dir=None, libraries=None,
637:              library_dirs=None, runtime_library_dirs=None,
638:              export_symbols=None, debug=0, extra_preargs=None,
639:              extra_postargs=None, build_temp=None, target_lang=None):
640:         objects, output_dir = self._fix_object_args(objects, output_dir)
641:         libraries, library_dirs, runtime_library_dirs = \
642:             self._fix_lib_args(libraries, library_dirs, runtime_library_dirs)
643: 
644:         lib_opts = gen_lib_options(self, library_dirs, runtime_library_dirs,
645:                                    libraries)
646:         if is_string(output_dir):
647:             output_filename = os.path.join(output_dir, output_filename)
648:         elif output_dir is not None:
649:             raise TypeError("'output_dir' must be a string or None")
650: 
651:         if self._need_link(objects, output_filename):
652:             if self.library_switch[-1]==' ':
653:                 o_args = [self.library_switch.strip(), output_filename]
654:             else:
655:                 o_args = [self.library_switch.strip()+output_filename]
656: 
657:             if is_string(self.objects):
658:                 ld_args = objects + [self.objects]
659:             else:
660:                 ld_args = objects + self.objects
661:             ld_args = ld_args + lib_opts + o_args
662:             if debug:
663:                 ld_args[:0] = ['-g']
664:             if extra_preargs:
665:                 ld_args[:0] = extra_preargs
666:             if extra_postargs:
667:                 ld_args.extend(extra_postargs)
668:             self.mkpath(os.path.dirname(output_filename))
669:             if target_desc == CCompiler.EXECUTABLE:
670:                 linker = self.linker_exe[:]
671:             else:
672:                 linker = self.linker_so[:]
673:             command = linker + ld_args
674:             try:
675:                 self.spawn(command)
676:             except DistutilsExecError:
677:                 msg = str(get_exception())
678:                 raise LinkError(msg)
679:         else:
680:             log.debug("skipping %s (up-to-date)", output_filename)
681: 
682:     def _environment_hook(self, name, hook_name):
683:         if hook_name is None:
684:             return None
685:         if is_string(hook_name):
686:             if hook_name.startswith('self.'):
687:                 hook_name = hook_name[5:]
688:                 hook = getattr(self, hook_name)
689:                 return hook()
690:             elif hook_name.startswith('exe.'):
691:                 hook_name = hook_name[4:]
692:                 var = self.executables[hook_name]
693:                 if var:
694:                     return var[0]
695:                 else:
696:                     return None
697:             elif hook_name.startswith('flags.'):
698:                 hook_name = hook_name[6:]
699:                 hook = getattr(self, 'get_flags_' + hook_name)
700:                 return hook()
701:         else:
702:             return hook_name()
703: 
704:     ## class FCompiler
705: 
706: _default_compilers = (
707:     # sys.platform mappings
708:     ('win32', ('gnu', 'intelv', 'absoft', 'compaqv', 'intelev', 'gnu95', 'g95',
709:                'intelvem', 'intelem')),
710:     ('cygwin.*', ('gnu', 'intelv', 'absoft', 'compaqv', 'intelev', 'gnu95', 'g95')),
711:     ('linux.*', ('gnu95', 'intel', 'lahey', 'pg', 'absoft', 'nag', 'vast', 'compaq',
712:                 'intele', 'intelem', 'gnu', 'g95', 'pathf95')),
713:     ('darwin.*', ('gnu95', 'nag', 'absoft', 'ibm', 'intel', 'gnu', 'g95', 'pg')),
714:     ('sunos.*', ('sun', 'gnu', 'gnu95', 'g95')),
715:     ('irix.*', ('mips', 'gnu', 'gnu95',)),
716:     ('aix.*', ('ibm', 'gnu', 'gnu95',)),
717:     # os.name mappings
718:     ('posix', ('gnu', 'gnu95',)),
719:     ('nt', ('gnu', 'gnu95',)),
720:     ('mac', ('gnu95', 'gnu', 'pg')),
721:     )
722: 
723: fcompiler_class = None
724: fcompiler_aliases = None
725: 
726: def load_all_fcompiler_classes():
727:     '''Cache all the FCompiler classes found in modules in the
728:     numpy.distutils.fcompiler package.
729:     '''
730:     from glob import glob
731:     global fcompiler_class, fcompiler_aliases
732:     if fcompiler_class is not None:
733:         return
734:     pys = os.path.join(os.path.dirname(__file__), '*.py')
735:     fcompiler_class = {}
736:     fcompiler_aliases = {}
737:     for fname in glob(pys):
738:         module_name, ext = os.path.splitext(os.path.basename(fname))
739:         module_name = 'numpy.distutils.fcompiler.' + module_name
740:         __import__ (module_name)
741:         module = sys.modules[module_name]
742:         if hasattr(module, 'compilers'):
743:             for cname in module.compilers:
744:                 klass = getattr(module, cname)
745:                 desc = (klass.compiler_type, klass, klass.description)
746:                 fcompiler_class[klass.compiler_type] = desc
747:                 for alias in klass.compiler_aliases:
748:                     if alias in fcompiler_aliases:
749:                         raise ValueError("alias %r defined for both %s and %s"
750:                                          % (alias, klass.__name__,
751:                                             fcompiler_aliases[alias][1].__name__))
752:                     fcompiler_aliases[alias] = desc
753: 
754: def _find_existing_fcompiler(compiler_types,
755:                              osname=None, platform=None,
756:                              requiref90=False,
757:                              c_compiler=None):
758:     from numpy.distutils.core import get_distribution
759:     dist = get_distribution(always=True)
760:     for compiler_type in compiler_types:
761:         v = None
762:         try:
763:             c = new_fcompiler(plat=platform, compiler=compiler_type,
764:                               c_compiler=c_compiler)
765:             c.customize(dist)
766:             v = c.get_version()
767:             if requiref90 and c.compiler_f90 is None:
768:                 v = None
769:                 new_compiler = c.suggested_f90_compiler
770:                 if new_compiler:
771:                     log.warn('Trying %r compiler as suggested by %r '
772:                              'compiler for f90 support.' % (compiler_type,
773:                                                             new_compiler))
774:                     c = new_fcompiler(plat=platform, compiler=new_compiler,
775:                                       c_compiler=c_compiler)
776:                     c.customize(dist)
777:                     v = c.get_version()
778:                     if v is not None:
779:                         compiler_type = new_compiler
780:             if requiref90 and c.compiler_f90 is None:
781:                 raise ValueError('%s does not support compiling f90 codes, '
782:                                  'skipping.' % (c.__class__.__name__))
783:         except DistutilsModuleError:
784:             log.debug("_find_existing_fcompiler: compiler_type='%s' raised DistutilsModuleError", compiler_type)
785:         except CompilerNotFound:
786:             log.debug("_find_existing_fcompiler: compiler_type='%s' not found", compiler_type)
787:         if v is not None:
788:             return compiler_type
789:     return None
790: 
791: def available_fcompilers_for_platform(osname=None, platform=None):
792:     if osname is None:
793:         osname = os.name
794:     if platform is None:
795:         platform = sys.platform
796:     matching_compiler_types = []
797:     for pattern, compiler_type in _default_compilers:
798:         if re.match(pattern, platform) or re.match(pattern, osname):
799:             for ct in compiler_type:
800:                 if ct not in matching_compiler_types:
801:                     matching_compiler_types.append(ct)
802:     if not matching_compiler_types:
803:         matching_compiler_types.append('gnu')
804:     return matching_compiler_types
805: 
806: def get_default_fcompiler(osname=None, platform=None, requiref90=False,
807:                           c_compiler=None):
808:     '''Determine the default Fortran compiler to use for the given
809:     platform.'''
810:     matching_compiler_types = available_fcompilers_for_platform(osname,
811:                                                                 platform)
812:     compiler_type =  _find_existing_fcompiler(matching_compiler_types,
813:                                               osname=osname,
814:                                               platform=platform,
815:                                               requiref90=requiref90,
816:                                               c_compiler=c_compiler)
817:     return compiler_type
818: 
819: # Flag to avoid rechecking for Fortran compiler every time
820: failed_fcompilers = set()
821: 
822: def new_fcompiler(plat=None,
823:                   compiler=None,
824:                   verbose=0,
825:                   dry_run=0,
826:                   force=0,
827:                   requiref90=False,
828:                   c_compiler = None):
829:     '''Generate an instance of some FCompiler subclass for the supplied
830:     platform/compiler combination.
831:     '''
832:     global failed_fcompilers
833:     fcompiler_key = (plat, compiler)
834:     if fcompiler_key in failed_fcompilers:
835:         return None
836: 
837:     load_all_fcompiler_classes()
838:     if plat is None:
839:         plat = os.name
840:     if compiler is None:
841:         compiler = get_default_fcompiler(plat, requiref90=requiref90,
842:                                          c_compiler=c_compiler)
843:     if compiler in fcompiler_class:
844:         module_name, klass, long_description = fcompiler_class[compiler]
845:     elif compiler in fcompiler_aliases:
846:         module_name, klass, long_description = fcompiler_aliases[compiler]
847:     else:
848:         msg = "don't know how to compile Fortran code on platform '%s'" % plat
849:         if compiler is not None:
850:             msg = msg + " with '%s' compiler." % compiler
851:             msg = msg + " Supported compilers are: %s)" \
852:                   % (','.join(fcompiler_class.keys()))
853:         log.warn(msg)
854:         failed_fcompilers.add(fcompiler_key)
855:         return None
856: 
857:     compiler = klass(verbose=verbose, dry_run=dry_run, force=force)
858:     compiler.c_compiler = c_compiler
859:     return compiler
860: 
861: def show_fcompilers(dist=None):
862:     '''Print list of available compilers (used by the "--help-fcompiler"
863:     option to "config_fc").
864:     '''
865:     if dist is None:
866:         from distutils.dist import Distribution
867:         from numpy.distutils.command.config_compiler import config_fc
868:         dist = Distribution()
869:         dist.script_name = os.path.basename(sys.argv[0])
870:         dist.script_args = ['config_fc'] + sys.argv[1:]
871:         try:
872:             dist.script_args.remove('--help-fcompiler')
873:         except ValueError:
874:             pass
875:         dist.cmdclass['config_fc'] = config_fc
876:         dist.parse_config_files()
877:         dist.parse_command_line()
878:     compilers = []
879:     compilers_na = []
880:     compilers_ni = []
881:     if not fcompiler_class:
882:         load_all_fcompiler_classes()
883:     platform_compilers = available_fcompilers_for_platform()
884:     for compiler in platform_compilers:
885:         v = None
886:         log.set_verbosity(-2)
887:         try:
888:             c = new_fcompiler(compiler=compiler, verbose=dist.verbose)
889:             c.customize(dist)
890:             v = c.get_version()
891:         except (DistutilsModuleError, CompilerNotFound):
892:             e = get_exception()
893:             log.debug("show_fcompilers: %s not found" % (compiler,))
894:             log.debug(repr(e))
895: 
896:         if v is None:
897:             compilers_na.append(("fcompiler="+compiler, None,
898:                               fcompiler_class[compiler][2]))
899:         else:
900:             c.dump_properties()
901:             compilers.append(("fcompiler="+compiler, None,
902:                               fcompiler_class[compiler][2] + ' (%s)' % v))
903: 
904:     compilers_ni = list(set(fcompiler_class.keys()) - set(platform_compilers))
905:     compilers_ni = [("fcompiler="+fc, None, fcompiler_class[fc][2])
906:                     for fc in compilers_ni]
907: 
908:     compilers.sort()
909:     compilers_na.sort()
910:     compilers_ni.sort()
911:     pretty_printer = FancyGetopt(compilers)
912:     pretty_printer.print_help("Fortran compilers found:")
913:     pretty_printer = FancyGetopt(compilers_na)
914:     pretty_printer.print_help("Compilers available for this "
915:                               "platform, but not found:")
916:     if compilers_ni:
917:         pretty_printer = FancyGetopt(compilers_ni)
918:         pretty_printer.print_help("Compilers not available on this platform:")
919:     print("For compiler details, run 'config_fc --verbose' setup command.")
920: 
921: 
922: def dummy_fortran_file():
923:     fo, name = make_temp_file(suffix='.f')
924:     fo.write("      subroutine dummy()\n      end\n")
925:     fo.close()
926:     return name[:-2]
927: 
928: 
929: is_f_file = re.compile(r'.*[.](for|ftn|f77|f)\Z', re.I).match
930: _has_f_header = re.compile(r'-[*]-\s*fortran\s*-[*]-', re.I).search
931: _has_f90_header = re.compile(r'-[*]-\s*f90\s*-[*]-', re.I).search
932: _has_fix_header = re.compile(r'-[*]-\s*fix\s*-[*]-', re.I).search
933: _free_f90_start = re.compile(r'[^c*!]\s*[^\s\d\t]', re.I).match
934: 
935: def is_free_format(file):
936:     '''Check if file is in free format Fortran.'''
937:     # f90 allows both fixed and free format, assuming fixed unless
938:     # signs of free format are detected.
939:     result = 0
940:     f = open_latin1(file, 'r')
941:     line = f.readline()
942:     n = 10000 # the number of non-comment lines to scan for hints
943:     if _has_f_header(line):
944:         n = 0
945:     elif _has_f90_header(line):
946:         n = 0
947:         result = 1
948:     while n>0 and line:
949:         line = line.rstrip()
950:         if line and line[0]!='!':
951:             n -= 1
952:             if (line[0]!='\t' and _free_f90_start(line[:5])) or line[-1:]=='&':
953:                 result = 1
954:                 break
955:         line = f.readline()
956:     f.close()
957:     return result
958: 
959: def has_f90_header(src):
960:     f = open_latin1(src, 'r')
961:     line = f.readline()
962:     f.close()
963:     return _has_f90_header(line) or _has_fix_header(line)
964: 
965: _f77flags_re = re.compile(r'(c|)f77flags\s*\(\s*(?P<fcname>\w+)\s*\)\s*=\s*(?P<fflags>.*)', re.I)
966: def get_f77flags(src):
967:     '''
968:     Search the first 20 lines of fortran 77 code for line pattern
969:       `CF77FLAGS(<fcompiler type>)=<f77 flags>`
970:     Return a dictionary {<fcompiler type>:<f77 flags>}.
971:     '''
972:     flags = {}
973:     f = open_latin1(src, 'r')
974:     i = 0
975:     for line in f:
976:         i += 1
977:         if i>20: break
978:         m = _f77flags_re.match(line)
979:         if not m: continue
980:         fcname = m.group('fcname').strip()
981:         fflags = m.group('fflags').strip()
982:         flags[fcname] = split_quoted(fflags)
983:     f.close()
984:     return flags
985: 
986: # TODO: implement get_f90flags and use it in _compile similarly to get_f77flags
987: 
988: if __name__ == '__main__':
989:     show_fcompilers()
990: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_63503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, (-1)), 'str', "numpy.distutils.fcompiler\n\nContains FCompiler, an abstract base class that defines the interface\nfor the numpy.distutils Fortran compiler abstraction model.\n\nTerminology:\n\nTo be consistent, where the term 'executable' is used, it means the single\nfile, like 'gcc', that is executed, and should be a string. In contrast,\n'command' means the entire command line, like ['gcc', '-c', 'file.c'], and\nshould be a list.\n\nBut note that FCompiler.executables is actually a dictionary of commands.\n\n")

# Assigning a List to a Name (line 18):

# Assigning a List to a Name (line 18):
__all__ = ['FCompiler', 'new_fcompiler', 'show_fcompilers', 'dummy_fortran_file']
module_type_store.set_exportable_members(['FCompiler', 'new_fcompiler', 'show_fcompilers', 'dummy_fortran_file'])

# Obtaining an instance of the builtin type 'list' (line 18)
list_63504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
str_63505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 11), 'str', 'FCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 10), list_63504, str_63505)
# Adding element type (line 18)
str_63506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 24), 'str', 'new_fcompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 10), list_63504, str_63506)
# Adding element type (line 18)
str_63507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 41), 'str', 'show_fcompilers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 10), list_63504, str_63507)
# Adding element type (line 18)
str_63508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 11), 'str', 'dummy_fortran_file')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 10), list_63504, str_63508)

# Assigning a type to the variable '__all__' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), '__all__', list_63504)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import os' statement (line 21)
import os

import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import sys' statement (line 22)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'import re' statement (line 23)
import re

import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'import types' statement (line 24)
import types

import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'types', types, module_type_store)



# SSA begins for try-except statement (line 25)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
# Getting the type of 'set' (line 26)
set_63509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'set')
# SSA branch for the except part of a try statement (line 25)
# SSA branch for the except 'NameError' branch of a try statement (line 25)
module_type_store.open_ssa_branch('except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 4))

# 'from sets import set' statement (line 28)
from sets import Set as set

import_from_module(stypy.reporting.localization.Localization(__file__, 28, 4), 'sets', None, module_type_store, ['Set'], [set])
# Adding an alias
module_type_store.add_alias('set', 'Set')

# SSA join for try-except statement (line 25)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'from numpy.compat import open_latin1' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_63510 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy.compat')

if (type(import_63510) is not StypyTypeError):

    if (import_63510 != 'pyd_module'):
        __import__(import_63510)
        sys_modules_63511 = sys.modules[import_63510]
        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy.compat', sys_modules_63511.module_type_store, module_type_store, ['open_latin1'])
        nest_module(stypy.reporting.localization.Localization(__file__, 30, 0), __file__, sys_modules_63511, sys_modules_63511.module_type_store, module_type_store)
    else:
        from numpy.compat import open_latin1

        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy.compat', None, module_type_store, ['open_latin1'], [open_latin1])

else:
    # Assigning a type to the variable 'numpy.compat' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy.compat', import_63510)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 0))

# 'from distutils.sysconfig import get_python_lib' statement (line 32)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_63512 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'distutils.sysconfig')

if (type(import_63512) is not StypyTypeError):

    if (import_63512 != 'pyd_module'):
        __import__(import_63512)
        sys_modules_63513 = sys.modules[import_63512]
        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'distutils.sysconfig', sys_modules_63513.module_type_store, module_type_store, ['get_python_lib'])
        nest_module(stypy.reporting.localization.Localization(__file__, 32, 0), __file__, sys_modules_63513, sys_modules_63513.module_type_store, module_type_store)
    else:
        from distutils.sysconfig import get_python_lib

        import_from_module(stypy.reporting.localization.Localization(__file__, 32, 0), 'distutils.sysconfig', None, module_type_store, ['get_python_lib'], [get_python_lib])

else:
    # Assigning a type to the variable 'distutils.sysconfig' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'distutils.sysconfig', import_63512)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 0))

# 'from distutils.fancy_getopt import FancyGetopt' statement (line 33)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_63514 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'distutils.fancy_getopt')

if (type(import_63514) is not StypyTypeError):

    if (import_63514 != 'pyd_module'):
        __import__(import_63514)
        sys_modules_63515 = sys.modules[import_63514]
        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'distutils.fancy_getopt', sys_modules_63515.module_type_store, module_type_store, ['FancyGetopt'])
        nest_module(stypy.reporting.localization.Localization(__file__, 33, 0), __file__, sys_modules_63515, sys_modules_63515.module_type_store, module_type_store)
    else:
        from distutils.fancy_getopt import FancyGetopt

        import_from_module(stypy.reporting.localization.Localization(__file__, 33, 0), 'distutils.fancy_getopt', None, module_type_store, ['FancyGetopt'], [FancyGetopt])

else:
    # Assigning a type to the variable 'distutils.fancy_getopt' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'distutils.fancy_getopt', import_63514)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 34, 0))

# 'from distutils.errors import DistutilsModuleError, DistutilsExecError, CompileError, LinkError, DistutilsPlatformError' statement (line 34)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_63516 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'distutils.errors')

if (type(import_63516) is not StypyTypeError):

    if (import_63516 != 'pyd_module'):
        __import__(import_63516)
        sys_modules_63517 = sys.modules[import_63516]
        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'distutils.errors', sys_modules_63517.module_type_store, module_type_store, ['DistutilsModuleError', 'DistutilsExecError', 'CompileError', 'LinkError', 'DistutilsPlatformError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 34, 0), __file__, sys_modules_63517, sys_modules_63517.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsModuleError, DistutilsExecError, CompileError, LinkError, DistutilsPlatformError

        import_from_module(stypy.reporting.localization.Localization(__file__, 34, 0), 'distutils.errors', None, module_type_store, ['DistutilsModuleError', 'DistutilsExecError', 'CompileError', 'LinkError', 'DistutilsPlatformError'], [DistutilsModuleError, DistutilsExecError, CompileError, LinkError, DistutilsPlatformError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'distutils.errors', import_63516)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 0))

# 'from distutils.util import split_quoted, strtobool' statement (line 36)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_63518 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'distutils.util')

if (type(import_63518) is not StypyTypeError):

    if (import_63518 != 'pyd_module'):
        __import__(import_63518)
        sys_modules_63519 = sys.modules[import_63518]
        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'distutils.util', sys_modules_63519.module_type_store, module_type_store, ['split_quoted', 'strtobool'])
        nest_module(stypy.reporting.localization.Localization(__file__, 36, 0), __file__, sys_modules_63519, sys_modules_63519.module_type_store, module_type_store)
    else:
        from distutils.util import split_quoted, strtobool

        import_from_module(stypy.reporting.localization.Localization(__file__, 36, 0), 'distutils.util', None, module_type_store, ['split_quoted', 'strtobool'], [split_quoted, strtobool])

else:
    # Assigning a type to the variable 'distutils.util' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'distutils.util', import_63518)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 38, 0))

# 'from numpy.distutils.ccompiler import CCompiler, gen_lib_options' statement (line 38)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_63520 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'numpy.distutils.ccompiler')

if (type(import_63520) is not StypyTypeError):

    if (import_63520 != 'pyd_module'):
        __import__(import_63520)
        sys_modules_63521 = sys.modules[import_63520]
        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'numpy.distutils.ccompiler', sys_modules_63521.module_type_store, module_type_store, ['CCompiler', 'gen_lib_options'])
        nest_module(stypy.reporting.localization.Localization(__file__, 38, 0), __file__, sys_modules_63521, sys_modules_63521.module_type_store, module_type_store)
    else:
        from numpy.distutils.ccompiler import CCompiler, gen_lib_options

        import_from_module(stypy.reporting.localization.Localization(__file__, 38, 0), 'numpy.distutils.ccompiler', None, module_type_store, ['CCompiler', 'gen_lib_options'], [CCompiler, gen_lib_options])

else:
    # Assigning a type to the variable 'numpy.distutils.ccompiler' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'numpy.distutils.ccompiler', import_63520)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 39, 0))

# 'from numpy.distutils import log' statement (line 39)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_63522 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'numpy.distutils')

if (type(import_63522) is not StypyTypeError):

    if (import_63522 != 'pyd_module'):
        __import__(import_63522)
        sys_modules_63523 = sys.modules[import_63522]
        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'numpy.distutils', sys_modules_63523.module_type_store, module_type_store, ['log'])
        nest_module(stypy.reporting.localization.Localization(__file__, 39, 0), __file__, sys_modules_63523, sys_modules_63523.module_type_store, module_type_store)
    else:
        from numpy.distutils import log

        import_from_module(stypy.reporting.localization.Localization(__file__, 39, 0), 'numpy.distutils', None, module_type_store, ['log'], [log])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'numpy.distutils', import_63522)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 0))

# 'from numpy.distutils.misc_util import is_string, all_strings, is_sequence, make_temp_file, get_shared_lib_extension' statement (line 40)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_63524 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'numpy.distutils.misc_util')

if (type(import_63524) is not StypyTypeError):

    if (import_63524 != 'pyd_module'):
        __import__(import_63524)
        sys_modules_63525 = sys.modules[import_63524]
        import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'numpy.distutils.misc_util', sys_modules_63525.module_type_store, module_type_store, ['is_string', 'all_strings', 'is_sequence', 'make_temp_file', 'get_shared_lib_extension'])
        nest_module(stypy.reporting.localization.Localization(__file__, 40, 0), __file__, sys_modules_63525, sys_modules_63525.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import is_string, all_strings, is_sequence, make_temp_file, get_shared_lib_extension

        import_from_module(stypy.reporting.localization.Localization(__file__, 40, 0), 'numpy.distutils.misc_util', None, module_type_store, ['is_string', 'all_strings', 'is_sequence', 'make_temp_file', 'get_shared_lib_extension'], [is_string, all_strings, is_sequence, make_temp_file, get_shared_lib_extension])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'numpy.distutils.misc_util', import_63524)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 42, 0))

# 'from numpy.distutils.environment import EnvironmentConfig' statement (line 42)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_63526 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'numpy.distutils.environment')

if (type(import_63526) is not StypyTypeError):

    if (import_63526 != 'pyd_module'):
        __import__(import_63526)
        sys_modules_63527 = sys.modules[import_63526]
        import_from_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'numpy.distutils.environment', sys_modules_63527.module_type_store, module_type_store, ['EnvironmentConfig'])
        nest_module(stypy.reporting.localization.Localization(__file__, 42, 0), __file__, sys_modules_63527, sys_modules_63527.module_type_store, module_type_store)
    else:
        from numpy.distutils.environment import EnvironmentConfig

        import_from_module(stypy.reporting.localization.Localization(__file__, 42, 0), 'numpy.distutils.environment', None, module_type_store, ['EnvironmentConfig'], [EnvironmentConfig])

else:
    # Assigning a type to the variable 'numpy.distutils.environment' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'numpy.distutils.environment', import_63526)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 43, 0))

# 'from numpy.distutils.exec_command import find_executable' statement (line 43)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_63528 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'numpy.distutils.exec_command')

if (type(import_63528) is not StypyTypeError):

    if (import_63528 != 'pyd_module'):
        __import__(import_63528)
        sys_modules_63529 = sys.modules[import_63528]
        import_from_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'numpy.distutils.exec_command', sys_modules_63529.module_type_store, module_type_store, ['find_executable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 43, 0), __file__, sys_modules_63529, sys_modules_63529.module_type_store, module_type_store)
    else:
        from numpy.distutils.exec_command import find_executable

        import_from_module(stypy.reporting.localization.Localization(__file__, 43, 0), 'numpy.distutils.exec_command', None, module_type_store, ['find_executable'], [find_executable])

else:
    # Assigning a type to the variable 'numpy.distutils.exec_command' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'numpy.distutils.exec_command', import_63528)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 0))

# 'from numpy.distutils.compat import get_exception' statement (line 44)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_63530 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.distutils.compat')

if (type(import_63530) is not StypyTypeError):

    if (import_63530 != 'pyd_module'):
        __import__(import_63530)
        sys_modules_63531 = sys.modules[import_63530]
        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.distutils.compat', sys_modules_63531.module_type_store, module_type_store, ['get_exception'])
        nest_module(stypy.reporting.localization.Localization(__file__, 44, 0), __file__, sys_modules_63531, sys_modules_63531.module_type_store, module_type_store)
    else:
        from numpy.distutils.compat import get_exception

        import_from_module(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.distutils.compat', None, module_type_store, ['get_exception'], [get_exception])

else:
    # Assigning a type to the variable 'numpy.distutils.compat' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'numpy.distutils.compat', import_63530)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')


# Assigning a Name to a Name (line 46):

# Assigning a Name to a Name (line 46):
# Getting the type of 'type' (line 46)
type_63532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'type')
# Assigning a type to the variable '__metaclass__' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), '__metaclass__', type_63532)
# Declaration of the 'CompilerNotFound' class
# Getting the type of 'Exception' (line 48)
Exception_63533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'Exception')

class CompilerNotFound(Exception_63533, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 48, 0, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CompilerNotFound.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'CompilerNotFound' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'CompilerNotFound', CompilerNotFound)

@norecursion
def flaglist(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'flaglist'
    module_type_store = module_type_store.open_function_context('flaglist', 51, 0, False)
    
    # Passed parameters checking function
    flaglist.stypy_localization = localization
    flaglist.stypy_type_of_self = None
    flaglist.stypy_type_store = module_type_store
    flaglist.stypy_function_name = 'flaglist'
    flaglist.stypy_param_names_list = ['s']
    flaglist.stypy_varargs_param_name = None
    flaglist.stypy_kwargs_param_name = None
    flaglist.stypy_call_defaults = defaults
    flaglist.stypy_call_varargs = varargs
    flaglist.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'flaglist', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'flaglist', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'flaglist(...)' code ##################

    
    
    # Call to is_string(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 's' (line 52)
    s_63535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 17), 's', False)
    # Processing the call keyword arguments (line 52)
    kwargs_63536 = {}
    # Getting the type of 'is_string' (line 52)
    is_string_63534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 7), 'is_string', False)
    # Calling is_string(args, kwargs) (line 52)
    is_string_call_result_63537 = invoke(stypy.reporting.localization.Localization(__file__, 52, 7), is_string_63534, *[s_63535], **kwargs_63536)
    
    # Testing the type of an if condition (line 52)
    if_condition_63538 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 4), is_string_call_result_63537)
    # Assigning a type to the variable 'if_condition_63538' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'if_condition_63538', if_condition_63538)
    # SSA begins for if statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to split_quoted(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 's' (line 53)
    s_63540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 28), 's', False)
    # Processing the call keyword arguments (line 53)
    kwargs_63541 = {}
    # Getting the type of 'split_quoted' (line 53)
    split_quoted_63539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'split_quoted', False)
    # Calling split_quoted(args, kwargs) (line 53)
    split_quoted_call_result_63542 = invoke(stypy.reporting.localization.Localization(__file__, 53, 15), split_quoted_63539, *[s_63540], **kwargs_63541)
    
    # Assigning a type to the variable 'stypy_return_type' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'stypy_return_type', split_quoted_call_result_63542)
    # SSA branch for the else part of an if statement (line 52)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 's' (line 55)
    s_63543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 15), 's')
    # Assigning a type to the variable 'stypy_return_type' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'stypy_return_type', s_63543)
    # SSA join for if statement (line 52)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'flaglist(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'flaglist' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_63544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_63544)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'flaglist'
    return stypy_return_type_63544

# Assigning a type to the variable 'flaglist' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'flaglist', flaglist)

@norecursion
def str2bool(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'str2bool'
    module_type_store = module_type_store.open_function_context('str2bool', 57, 0, False)
    
    # Passed parameters checking function
    str2bool.stypy_localization = localization
    str2bool.stypy_type_of_self = None
    str2bool.stypy_type_store = module_type_store
    str2bool.stypy_function_name = 'str2bool'
    str2bool.stypy_param_names_list = ['s']
    str2bool.stypy_varargs_param_name = None
    str2bool.stypy_kwargs_param_name = None
    str2bool.stypy_call_defaults = defaults
    str2bool.stypy_call_varargs = varargs
    str2bool.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'str2bool', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'str2bool', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'str2bool(...)' code ##################

    
    
    # Call to is_string(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 's' (line 58)
    s_63546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 17), 's', False)
    # Processing the call keyword arguments (line 58)
    kwargs_63547 = {}
    # Getting the type of 'is_string' (line 58)
    is_string_63545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 7), 'is_string', False)
    # Calling is_string(args, kwargs) (line 58)
    is_string_call_result_63548 = invoke(stypy.reporting.localization.Localization(__file__, 58, 7), is_string_63545, *[s_63546], **kwargs_63547)
    
    # Testing the type of an if condition (line 58)
    if_condition_63549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 4), is_string_call_result_63548)
    # Assigning a type to the variable 'if_condition_63549' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'if_condition_63549', if_condition_63549)
    # SSA begins for if statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to strtobool(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 's' (line 59)
    s_63551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 's', False)
    # Processing the call keyword arguments (line 59)
    kwargs_63552 = {}
    # Getting the type of 'strtobool' (line 59)
    strtobool_63550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'strtobool', False)
    # Calling strtobool(args, kwargs) (line 59)
    strtobool_call_result_63553 = invoke(stypy.reporting.localization.Localization(__file__, 59, 15), strtobool_63550, *[s_63551], **kwargs_63552)
    
    # Assigning a type to the variable 'stypy_return_type' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type', strtobool_call_result_63553)
    # SSA join for if statement (line 58)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to bool(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 's' (line 60)
    s_63555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 's', False)
    # Processing the call keyword arguments (line 60)
    kwargs_63556 = {}
    # Getting the type of 'bool' (line 60)
    bool_63554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'bool', False)
    # Calling bool(args, kwargs) (line 60)
    bool_call_result_63557 = invoke(stypy.reporting.localization.Localization(__file__, 60, 11), bool_63554, *[s_63555], **kwargs_63556)
    
    # Assigning a type to the variable 'stypy_return_type' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type', bool_call_result_63557)
    
    # ################# End of 'str2bool(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'str2bool' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_63558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_63558)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'str2bool'
    return stypy_return_type_63558

# Assigning a type to the variable 'str2bool' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'str2bool', str2bool)

@norecursion
def is_sequence_of_strings(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_sequence_of_strings'
    module_type_store = module_type_store.open_function_context('is_sequence_of_strings', 62, 0, False)
    
    # Passed parameters checking function
    is_sequence_of_strings.stypy_localization = localization
    is_sequence_of_strings.stypy_type_of_self = None
    is_sequence_of_strings.stypy_type_store = module_type_store
    is_sequence_of_strings.stypy_function_name = 'is_sequence_of_strings'
    is_sequence_of_strings.stypy_param_names_list = ['seq']
    is_sequence_of_strings.stypy_varargs_param_name = None
    is_sequence_of_strings.stypy_kwargs_param_name = None
    is_sequence_of_strings.stypy_call_defaults = defaults
    is_sequence_of_strings.stypy_call_varargs = varargs
    is_sequence_of_strings.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_sequence_of_strings', ['seq'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_sequence_of_strings', localization, ['seq'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_sequence_of_strings(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to is_sequence(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'seq' (line 63)
    seq_63560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 23), 'seq', False)
    # Processing the call keyword arguments (line 63)
    kwargs_63561 = {}
    # Getting the type of 'is_sequence' (line 63)
    is_sequence_63559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'is_sequence', False)
    # Calling is_sequence(args, kwargs) (line 63)
    is_sequence_call_result_63562 = invoke(stypy.reporting.localization.Localization(__file__, 63, 11), is_sequence_63559, *[seq_63560], **kwargs_63561)
    
    
    # Call to all_strings(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'seq' (line 63)
    seq_63564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 44), 'seq', False)
    # Processing the call keyword arguments (line 63)
    kwargs_63565 = {}
    # Getting the type of 'all_strings' (line 63)
    all_strings_63563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 32), 'all_strings', False)
    # Calling all_strings(args, kwargs) (line 63)
    all_strings_call_result_63566 = invoke(stypy.reporting.localization.Localization(__file__, 63, 32), all_strings_63563, *[seq_63564], **kwargs_63565)
    
    # Applying the binary operator 'and' (line 63)
    result_and_keyword_63567 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 11), 'and', is_sequence_call_result_63562, all_strings_call_result_63566)
    
    # Assigning a type to the variable 'stypy_return_type' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type', result_and_keyword_63567)
    
    # ################# End of 'is_sequence_of_strings(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_sequence_of_strings' in the type store
    # Getting the type of 'stypy_return_type' (line 62)
    stypy_return_type_63568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_63568)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_sequence_of_strings'
    return stypy_return_type_63568

# Assigning a type to the variable 'is_sequence_of_strings' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'is_sequence_of_strings', is_sequence_of_strings)
# Declaration of the 'FCompiler' class
# Getting the type of 'CCompiler' (line 65)
CCompiler_63569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'CCompiler')

class FCompiler(CCompiler_63569, ):
    str_63570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, (-1)), 'str', "Abstract base class to define the interface that must be implemented\n    by real Fortran compiler classes.\n\n    Methods that subclasses may redefine:\n\n        update_executables(), find_executables(), get_version()\n        get_flags(), get_flags_opt(), get_flags_arch(), get_flags_debug()\n        get_flags_f77(), get_flags_opt_f77(), get_flags_arch_f77(),\n        get_flags_debug_f77(), get_flags_f90(), get_flags_opt_f90(),\n        get_flags_arch_f90(), get_flags_debug_f90(),\n        get_flags_fix(), get_flags_linker_so()\n\n    DON'T call these methods (except get_version) after\n    constructing a compiler instance or inside any other method.\n    All methods, except update_executables() and find_executables(),\n    may call the get_version() method.\n\n    After constructing a compiler instance, always call customize(dist=None)\n    method that finalizes compiler construction and makes the following\n    attributes available:\n      compiler_f77\n      compiler_f90\n      compiler_fix\n      linker_so\n      archiver\n      ranlib\n      libraries\n      library_dirs\n    ")
    
    # Assigning a Call to a Name (line 106):
    
    # Assigning a Call to a Name (line 114):
    
    # Assigning a Call to a Name (line 126):
    
    # Assigning a Dict to a Name (line 147):
    
    # Assigning a List to a Name (line 156):
    
    # Assigning a Name to a Name (line 161):
    
    # Assigning a Tuple to a Name (line 162):
    
    # Assigning a Name to a Name (line 163):
    
    # Assigning a List to a Name (line 165):
    
    # Assigning a Dict to a Name (line 166):
    
    # Assigning a Name to a Name (line 180):
    
    # Assigning a Str to a Name (line 182):
    
    # Assigning a Str to a Name (line 183):
    
    # Assigning a Str to a Name (line 187):
    
    # Assigning a Name to a Name (line 192):
    
    # Assigning a Str to a Name (line 195):
    
    # Assigning a List to a Name (line 197):
    
    # Assigning a List to a Name (line 199):
    
    # Assigning a Str to a Name (line 200):
    
    # Assigning a Call to a Name (line 202):
    
    # Assigning a Str to a Name (line 203):
    
    # Assigning a Str to a Name (line 204):
    
    # Assigning a Str to a Name (line 205):
    
    # Assigning a Str to a Name (line 206):
    
    # Assigning a Dict to a Name (line 208):
    
    # Assigning a List to a Name (line 210):
    
    # Assigning a Name to a Name (line 216):
    
    # Assigning a List to a Name (line 219):
    
    # Assigning a List to a Name (line 220):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 222, 4, False)
        # Assigning a type to the variable 'self' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.__init__', [], 'args', 'kw', defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'self' (line 223)
        self_63573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 27), 'self', False)
        # Getting the type of 'args' (line 223)
        args_63574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 34), 'args', False)
        # Processing the call keyword arguments (line 223)
        # Getting the type of 'kw' (line 223)
        kw_63575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 42), 'kw', False)
        kwargs_63576 = {'kw_63575': kw_63575}
        # Getting the type of 'CCompiler' (line 223)
        CCompiler_63571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'CCompiler', False)
        # Obtaining the member '__init__' of a type (line 223)
        init___63572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), CCompiler_63571, '__init__')
        # Calling __init__(args, kwargs) (line 223)
        init___call_result_63577 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), init___63572, *[self_63573, args_63574], **kwargs_63576)
        
        
        # Assigning a Call to a Attribute (line 224):
        
        # Assigning a Call to a Attribute (line 224):
        
        # Call to clone(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'self' (line 224)
        self_63581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 56), 'self', False)
        # Obtaining the member '_environment_hook' of a type (line 224)
        _environment_hook_63582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 56), self_63581, '_environment_hook')
        # Processing the call keyword arguments (line 224)
        kwargs_63583 = {}
        # Getting the type of 'self' (line 224)
        self_63578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 30), 'self', False)
        # Obtaining the member 'distutils_vars' of a type (line 224)
        distutils_vars_63579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 30), self_63578, 'distutils_vars')
        # Obtaining the member 'clone' of a type (line 224)
        clone_63580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 30), distutils_vars_63579, 'clone')
        # Calling clone(args, kwargs) (line 224)
        clone_call_result_63584 = invoke(stypy.reporting.localization.Localization(__file__, 224, 30), clone_63580, *[_environment_hook_63582], **kwargs_63583)
        
        # Getting the type of 'self' (line 224)
        self_63585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'self')
        # Setting the type of the member 'distutils_vars' of a type (line 224)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), self_63585, 'distutils_vars', clone_call_result_63584)
        
        # Assigning a Call to a Attribute (line 225):
        
        # Assigning a Call to a Attribute (line 225):
        
        # Call to clone(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'self' (line 225)
        self_63589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 52), 'self', False)
        # Obtaining the member '_environment_hook' of a type (line 225)
        _environment_hook_63590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 52), self_63589, '_environment_hook')
        # Processing the call keyword arguments (line 225)
        kwargs_63591 = {}
        # Getting the type of 'self' (line 225)
        self_63586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 28), 'self', False)
        # Obtaining the member 'command_vars' of a type (line 225)
        command_vars_63587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 28), self_63586, 'command_vars')
        # Obtaining the member 'clone' of a type (line 225)
        clone_63588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 28), command_vars_63587, 'clone')
        # Calling clone(args, kwargs) (line 225)
        clone_call_result_63592 = invoke(stypy.reporting.localization.Localization(__file__, 225, 28), clone_63588, *[_environment_hook_63590], **kwargs_63591)
        
        # Getting the type of 'self' (line 225)
        self_63593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'self')
        # Setting the type of the member 'command_vars' of a type (line 225)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), self_63593, 'command_vars', clone_call_result_63592)
        
        # Assigning a Call to a Attribute (line 226):
        
        # Assigning a Call to a Attribute (line 226):
        
        # Call to clone(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'self' (line 226)
        self_63597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 46), 'self', False)
        # Obtaining the member '_environment_hook' of a type (line 226)
        _environment_hook_63598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 46), self_63597, '_environment_hook')
        # Processing the call keyword arguments (line 226)
        kwargs_63599 = {}
        # Getting the type of 'self' (line 226)
        self_63594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 25), 'self', False)
        # Obtaining the member 'flag_vars' of a type (line 226)
        flag_vars_63595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 25), self_63594, 'flag_vars')
        # Obtaining the member 'clone' of a type (line 226)
        clone_63596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 25), flag_vars_63595, 'clone')
        # Calling clone(args, kwargs) (line 226)
        clone_call_result_63600 = invoke(stypy.reporting.localization.Localization(__file__, 226, 25), clone_63596, *[_environment_hook_63598], **kwargs_63599)
        
        # Getting the type of 'self' (line 226)
        self_63601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'self')
        # Setting the type of the member 'flag_vars' of a type (line 226)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 8), self_63601, 'flag_vars', clone_call_result_63600)
        
        # Assigning a Call to a Attribute (line 227):
        
        # Assigning a Call to a Attribute (line 227):
        
        # Call to copy(...): (line 227)
        # Processing the call keyword arguments (line 227)
        kwargs_63605 = {}
        # Getting the type of 'self' (line 227)
        self_63602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 27), 'self', False)
        # Obtaining the member 'executables' of a type (line 227)
        executables_63603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 27), self_63602, 'executables')
        # Obtaining the member 'copy' of a type (line 227)
        copy_63604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 27), executables_63603, 'copy')
        # Calling copy(args, kwargs) (line 227)
        copy_call_result_63606 = invoke(stypy.reporting.localization.Localization(__file__, 227, 27), copy_63604, *[], **kwargs_63605)
        
        # Getting the type of 'self' (line 227)
        self_63607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'self')
        # Setting the type of the member 'executables' of a type (line 227)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), self_63607, 'executables', copy_call_result_63606)
        
        # Getting the type of 'self' (line 228)
        self_63608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 17), 'self')
        # Obtaining the member '_executable_keys' of a type (line 228)
        _executable_keys_63609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 17), self_63608, '_executable_keys')
        # Testing the type of a for loop iterable (line 228)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 228, 8), _executable_keys_63609)
        # Getting the type of the for loop variable (line 228)
        for_loop_var_63610 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 228, 8), _executable_keys_63609)
        # Assigning a type to the variable 'e' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'e', for_loop_var_63610)
        # SSA begins for a for statement (line 228)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'e' (line 229)
        e_63611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'e')
        # Getting the type of 'self' (line 229)
        self_63612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 24), 'self')
        # Obtaining the member 'executables' of a type (line 229)
        executables_63613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 24), self_63612, 'executables')
        # Applying the binary operator 'notin' (line 229)
        result_contains_63614 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 15), 'notin', e_63611, executables_63613)
        
        # Testing the type of an if condition (line 229)
        if_condition_63615 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 12), result_contains_63614)
        # Assigning a type to the variable 'if_condition_63615' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'if_condition_63615', if_condition_63615)
        # SSA begins for if statement (line 229)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 230):
        
        # Assigning a Name to a Subscript (line 230):
        # Getting the type of 'None' (line 230)
        None_63616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 38), 'None')
        # Getting the type of 'self' (line 230)
        self_63617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'self')
        # Obtaining the member 'executables' of a type (line 230)
        executables_63618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 16), self_63617, 'executables')
        # Getting the type of 'e' (line 230)
        e_63619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 33), 'e')
        # Storing an element on a container (line 230)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 16), executables_63618, (e_63619, None_63616))
        # SSA join for if statement (line 229)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 234):
        
        # Assigning a Name to a Attribute (line 234):
        # Getting the type of 'False' (line 234)
        False_63620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 30), 'False')
        # Getting the type of 'self' (line 234)
        self_63621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'self')
        # Setting the type of the member '_is_customised' of a type (line 234)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 8), self_63621, '_is_customised', False_63620)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __copy__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__copy__'
        module_type_store = module_type_store.open_function_context('__copy__', 236, 4, False)
        # Assigning a type to the variable 'self' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.__copy__.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.__copy__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.__copy__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.__copy__.__dict__.__setitem__('stypy_function_name', 'FCompiler.__copy__')
        FCompiler.__copy__.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.__copy__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.__copy__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.__copy__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.__copy__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.__copy__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.__copy__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.__copy__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__copy__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__copy__(...)' code ##################

        
        # Assigning a Call to a Name (line 237):
        
        # Assigning a Call to a Name (line 237):
        
        # Call to __new__(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'self' (line 237)
        self_63624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 27), 'self', False)
        # Obtaining the member '__class__' of a type (line 237)
        class___63625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 27), self_63624, '__class__')
        # Processing the call keyword arguments (line 237)
        kwargs_63626 = {}
        # Getting the type of 'self' (line 237)
        self_63622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 14), 'self', False)
        # Obtaining the member '__new__' of a type (line 237)
        new___63623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 14), self_63622, '__new__')
        # Calling __new__(args, kwargs) (line 237)
        new___call_result_63627 = invoke(stypy.reporting.localization.Localization(__file__, 237, 14), new___63623, *[class___63625], **kwargs_63626)
        
        # Assigning a type to the variable 'obj' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'obj', new___call_result_63627)
        
        # Call to update(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'self' (line 238)
        self_63631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 28), 'self', False)
        # Obtaining the member '__dict__' of a type (line 238)
        dict___63632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 28), self_63631, '__dict__')
        # Processing the call keyword arguments (line 238)
        kwargs_63633 = {}
        # Getting the type of 'obj' (line 238)
        obj_63628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'obj', False)
        # Obtaining the member '__dict__' of a type (line 238)
        dict___63629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), obj_63628, '__dict__')
        # Obtaining the member 'update' of a type (line 238)
        update_63630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), dict___63629, 'update')
        # Calling update(args, kwargs) (line 238)
        update_call_result_63634 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), update_63630, *[dict___63632], **kwargs_63633)
        
        
        # Assigning a Call to a Attribute (line 239):
        
        # Assigning a Call to a Attribute (line 239):
        
        # Call to clone(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'obj' (line 239)
        obj_63638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 54), 'obj', False)
        # Obtaining the member '_environment_hook' of a type (line 239)
        _environment_hook_63639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 54), obj_63638, '_environment_hook')
        # Processing the call keyword arguments (line 239)
        kwargs_63640 = {}
        # Getting the type of 'obj' (line 239)
        obj_63635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 29), 'obj', False)
        # Obtaining the member 'distutils_vars' of a type (line 239)
        distutils_vars_63636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 29), obj_63635, 'distutils_vars')
        # Obtaining the member 'clone' of a type (line 239)
        clone_63637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 29), distutils_vars_63636, 'clone')
        # Calling clone(args, kwargs) (line 239)
        clone_call_result_63641 = invoke(stypy.reporting.localization.Localization(__file__, 239, 29), clone_63637, *[_environment_hook_63639], **kwargs_63640)
        
        # Getting the type of 'obj' (line 239)
        obj_63642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'obj')
        # Setting the type of the member 'distutils_vars' of a type (line 239)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), obj_63642, 'distutils_vars', clone_call_result_63641)
        
        # Assigning a Call to a Attribute (line 240):
        
        # Assigning a Call to a Attribute (line 240):
        
        # Call to clone(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'obj' (line 240)
        obj_63646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 50), 'obj', False)
        # Obtaining the member '_environment_hook' of a type (line 240)
        _environment_hook_63647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 50), obj_63646, '_environment_hook')
        # Processing the call keyword arguments (line 240)
        kwargs_63648 = {}
        # Getting the type of 'obj' (line 240)
        obj_63643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 27), 'obj', False)
        # Obtaining the member 'command_vars' of a type (line 240)
        command_vars_63644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 27), obj_63643, 'command_vars')
        # Obtaining the member 'clone' of a type (line 240)
        clone_63645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 27), command_vars_63644, 'clone')
        # Calling clone(args, kwargs) (line 240)
        clone_call_result_63649 = invoke(stypy.reporting.localization.Localization(__file__, 240, 27), clone_63645, *[_environment_hook_63647], **kwargs_63648)
        
        # Getting the type of 'obj' (line 240)
        obj_63650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'obj')
        # Setting the type of the member 'command_vars' of a type (line 240)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), obj_63650, 'command_vars', clone_call_result_63649)
        
        # Assigning a Call to a Attribute (line 241):
        
        # Assigning a Call to a Attribute (line 241):
        
        # Call to clone(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'obj' (line 241)
        obj_63654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 44), 'obj', False)
        # Obtaining the member '_environment_hook' of a type (line 241)
        _environment_hook_63655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 44), obj_63654, '_environment_hook')
        # Processing the call keyword arguments (line 241)
        kwargs_63656 = {}
        # Getting the type of 'obj' (line 241)
        obj_63651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 24), 'obj', False)
        # Obtaining the member 'flag_vars' of a type (line 241)
        flag_vars_63652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 24), obj_63651, 'flag_vars')
        # Obtaining the member 'clone' of a type (line 241)
        clone_63653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 24), flag_vars_63652, 'clone')
        # Calling clone(args, kwargs) (line 241)
        clone_call_result_63657 = invoke(stypy.reporting.localization.Localization(__file__, 241, 24), clone_63653, *[_environment_hook_63655], **kwargs_63656)
        
        # Getting the type of 'obj' (line 241)
        obj_63658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'obj')
        # Setting the type of the member 'flag_vars' of a type (line 241)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), obj_63658, 'flag_vars', clone_call_result_63657)
        
        # Assigning a Call to a Attribute (line 242):
        
        # Assigning a Call to a Attribute (line 242):
        
        # Call to copy(...): (line 242)
        # Processing the call keyword arguments (line 242)
        kwargs_63662 = {}
        # Getting the type of 'obj' (line 242)
        obj_63659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 26), 'obj', False)
        # Obtaining the member 'executables' of a type (line 242)
        executables_63660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 26), obj_63659, 'executables')
        # Obtaining the member 'copy' of a type (line 242)
        copy_63661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 26), executables_63660, 'copy')
        # Calling copy(args, kwargs) (line 242)
        copy_call_result_63663 = invoke(stypy.reporting.localization.Localization(__file__, 242, 26), copy_63661, *[], **kwargs_63662)
        
        # Getting the type of 'obj' (line 242)
        obj_63664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'obj')
        # Setting the type of the member 'executables' of a type (line 242)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), obj_63664, 'executables', copy_call_result_63663)
        # Getting the type of 'obj' (line 243)
        obj_63665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'stypy_return_type', obj_63665)
        
        # ################# End of '__copy__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__copy__' in the type store
        # Getting the type of 'stypy_return_type' (line 236)
        stypy_return_type_63666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63666)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__copy__'
        return stypy_return_type_63666


    @norecursion
    def copy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy'
        module_type_store = module_type_store.open_function_context('copy', 245, 4, False)
        # Assigning a type to the variable 'self' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.copy.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.copy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.copy.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.copy.__dict__.__setitem__('stypy_function_name', 'FCompiler.copy')
        FCompiler.copy.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.copy.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.copy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.copy.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.copy.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.copy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.copy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.copy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy(...)' code ##################

        
        # Call to __copy__(...): (line 246)
        # Processing the call keyword arguments (line 246)
        kwargs_63669 = {}
        # Getting the type of 'self' (line 246)
        self_63667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'self', False)
        # Obtaining the member '__copy__' of a type (line 246)
        copy___63668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 15), self_63667, '__copy__')
        # Calling __copy__(args, kwargs) (line 246)
        copy___call_result_63670 = invoke(stypy.reporting.localization.Localization(__file__, 246, 15), copy___63668, *[], **kwargs_63669)
        
        # Assigning a type to the variable 'stypy_return_type' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'stypy_return_type', copy___call_result_63670)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 245)
        stypy_return_type_63671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63671)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_63671


    @staticmethod
    @norecursion
    def _command_property(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_command_property'
        module_type_store = module_type_store.open_function_context('_command_property', 251, 4, False)
        
        # Passed parameters checking function
        FCompiler._command_property.__dict__.__setitem__('stypy_localization', localization)
        FCompiler._command_property.__dict__.__setitem__('stypy_type_of_self', None)
        FCompiler._command_property.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler._command_property.__dict__.__setitem__('stypy_function_name', '_command_property')
        FCompiler._command_property.__dict__.__setitem__('stypy_param_names_list', ['key'])
        FCompiler._command_property.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler._command_property.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler._command_property.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler._command_property.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler._command_property.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler._command_property.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, '_command_property', ['key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_command_property', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_command_property(...)' code ##################


        @norecursion
        def fget(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'fget'
            module_type_store = module_type_store.open_function_context('fget', 252, 8, False)
            
            # Passed parameters checking function
            fget.stypy_localization = localization
            fget.stypy_type_of_self = None
            fget.stypy_type_store = module_type_store
            fget.stypy_function_name = 'fget'
            fget.stypy_param_names_list = ['self']
            fget.stypy_varargs_param_name = None
            fget.stypy_kwargs_param_name = None
            fget.stypy_call_defaults = defaults
            fget.stypy_call_varargs = varargs
            fget.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'fget', ['self'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'fget', localization, ['self'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'fget(...)' code ##################

            # Evaluating assert statement condition
            # Getting the type of 'self' (line 253)
            self_63672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 19), 'self')
            # Obtaining the member '_is_customised' of a type (line 253)
            _is_customised_63673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 19), self_63672, '_is_customised')
            
            # Obtaining the type of the subscript
            # Getting the type of 'key' (line 254)
            key_63674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 36), 'key')
            # Getting the type of 'self' (line 254)
            self_63675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 19), 'self')
            # Obtaining the member 'executables' of a type (line 254)
            executables_63676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 19), self_63675, 'executables')
            # Obtaining the member '__getitem__' of a type (line 254)
            getitem___63677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 19), executables_63676, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 254)
            subscript_call_result_63678 = invoke(stypy.reporting.localization.Localization(__file__, 254, 19), getitem___63677, key_63674)
            
            # Assigning a type to the variable 'stypy_return_type' (line 254)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'stypy_return_type', subscript_call_result_63678)
            
            # ################# End of 'fget(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'fget' in the type store
            # Getting the type of 'stypy_return_type' (line 252)
            stypy_return_type_63679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_63679)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'fget'
            return stypy_return_type_63679

        # Assigning a type to the variable 'fget' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'fget', fget)
        
        # Call to property(...): (line 255)
        # Processing the call keyword arguments (line 255)
        # Getting the type of 'fget' (line 255)
        fget_63681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 29), 'fget', False)
        keyword_63682 = fget_63681
        kwargs_63683 = {'fget': keyword_63682}
        # Getting the type of 'property' (line 255)
        property_63680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 15), 'property', False)
        # Calling property(args, kwargs) (line 255)
        property_call_result_63684 = invoke(stypy.reporting.localization.Localization(__file__, 255, 15), property_63680, *[], **kwargs_63683)
        
        # Assigning a type to the variable 'stypy_return_type' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'stypy_return_type', property_call_result_63684)
        
        # ################# End of '_command_property(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_command_property' in the type store
        # Getting the type of 'stypy_return_type' (line 251)
        stypy_return_type_63685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63685)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_command_property'
        return stypy_return_type_63685

    
    # Assigning a Call to a Name (line 256):
    
    # Assigning a Call to a Name (line 257):
    
    # Assigning a Call to a Name (line 258):
    
    # Assigning a Call to a Name (line 259):
    
    # Assigning a Call to a Name (line 260):
    
    # Assigning a Call to a Name (line 261):
    
    # Assigning a Call to a Name (line 262):
    
    # Assigning a Call to a Name (line 263):

    @norecursion
    def set_executable(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_executable'
        module_type_store = module_type_store.open_function_context('set_executable', 266, 4, False)
        # Assigning a type to the variable 'self' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.set_executable.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.set_executable.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.set_executable.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.set_executable.__dict__.__setitem__('stypy_function_name', 'FCompiler.set_executable')
        FCompiler.set_executable.__dict__.__setitem__('stypy_param_names_list', ['key', 'value'])
        FCompiler.set_executable.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.set_executable.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.set_executable.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.set_executable.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.set_executable.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.set_executable.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.set_executable', ['key', 'value'], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_command(...): (line 267)
        # Processing the call arguments (line 267)
        # Getting the type of 'key' (line 267)
        key_63688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 25), 'key', False)
        # Getting the type of 'value' (line 267)
        value_63689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 30), 'value', False)
        # Processing the call keyword arguments (line 267)
        kwargs_63690 = {}
        # Getting the type of 'self' (line 267)
        self_63686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'self', False)
        # Obtaining the member 'set_command' of a type (line 267)
        set_command_63687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 8), self_63686, 'set_command')
        # Calling set_command(args, kwargs) (line 267)
        set_command_call_result_63691 = invoke(stypy.reporting.localization.Localization(__file__, 267, 8), set_command_63687, *[key_63688, value_63689], **kwargs_63690)
        
        
        # ################# End of 'set_executable(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_executable' in the type store
        # Getting the type of 'stypy_return_type' (line 266)
        stypy_return_type_63692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63692)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_executable'
        return stypy_return_type_63692


    @norecursion
    def set_commands(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_commands'
        module_type_store = module_type_store.open_function_context('set_commands', 269, 4, False)
        # Assigning a type to the variable 'self' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.set_commands.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.set_commands.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.set_commands.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.set_commands.__dict__.__setitem__('stypy_function_name', 'FCompiler.set_commands')
        FCompiler.set_commands.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.set_commands.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.set_commands.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
        FCompiler.set_commands.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.set_commands.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.set_commands.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.set_commands.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.set_commands', [], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_commands', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_commands(...)' code ##################

        
        
        # Call to items(...): (line 270)
        # Processing the call keyword arguments (line 270)
        kwargs_63695 = {}
        # Getting the type of 'kw' (line 270)
        kw_63693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'kw', False)
        # Obtaining the member 'items' of a type (line 270)
        items_63694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 20), kw_63693, 'items')
        # Calling items(args, kwargs) (line 270)
        items_call_result_63696 = invoke(stypy.reporting.localization.Localization(__file__, 270, 20), items_63694, *[], **kwargs_63695)
        
        # Testing the type of a for loop iterable (line 270)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 270, 8), items_call_result_63696)
        # Getting the type of the for loop variable (line 270)
        for_loop_var_63697 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 270, 8), items_call_result_63696)
        # Assigning a type to the variable 'k' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 8), for_loop_var_63697))
        # Assigning a type to the variable 'v' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 8), for_loop_var_63697))
        # SSA begins for a for statement (line 270)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to set_command(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'k' (line 271)
        k_63700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 29), 'k', False)
        # Getting the type of 'v' (line 271)
        v_63701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 32), 'v', False)
        # Processing the call keyword arguments (line 271)
        kwargs_63702 = {}
        # Getting the type of 'self' (line 271)
        self_63698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'self', False)
        # Obtaining the member 'set_command' of a type (line 271)
        set_command_63699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 12), self_63698, 'set_command')
        # Calling set_command(args, kwargs) (line 271)
        set_command_call_result_63703 = invoke(stypy.reporting.localization.Localization(__file__, 271, 12), set_command_63699, *[k_63700, v_63701], **kwargs_63702)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_commands(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_commands' in the type store
        # Getting the type of 'stypy_return_type' (line 269)
        stypy_return_type_63704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63704)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_commands'
        return stypy_return_type_63704


    @norecursion
    def set_command(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_command'
        module_type_store = module_type_store.open_function_context('set_command', 273, 4, False)
        # Assigning a type to the variable 'self' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.set_command.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.set_command.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.set_command.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.set_command.__dict__.__setitem__('stypy_function_name', 'FCompiler.set_command')
        FCompiler.set_command.__dict__.__setitem__('stypy_param_names_list', ['key', 'value'])
        FCompiler.set_command.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.set_command.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.set_command.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.set_command.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.set_command.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.set_command.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.set_command', ['key', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_command', localization, ['key', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_command(...)' code ##################

        
        
        
        # Getting the type of 'key' (line 274)
        key_63705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 15), 'key')
        # Getting the type of 'self' (line 274)
        self_63706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 22), 'self')
        # Obtaining the member '_executable_keys' of a type (line 274)
        _executable_keys_63707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 22), self_63706, '_executable_keys')
        # Applying the binary operator 'in' (line 274)
        result_contains_63708 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 15), 'in', key_63705, _executable_keys_63707)
        
        # Applying the 'not' unary operator (line 274)
        result_not__63709 = python_operator(stypy.reporting.localization.Localization(__file__, 274, 11), 'not', result_contains_63708)
        
        # Testing the type of an if condition (line 274)
        if_condition_63710 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 274, 8), result_not__63709)
        # Assigning a type to the variable 'if_condition_63710' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'if_condition_63710', if_condition_63710)
        # SSA begins for if statement (line 274)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 275)
        # Processing the call arguments (line 275)
        str_63712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 16), 'str', "unknown executable '%s' for class %s")
        
        # Obtaining an instance of the builtin type 'tuple' (line 277)
        tuple_63713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 277)
        # Adding element type (line 277)
        # Getting the type of 'key' (line 277)
        key_63714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 17), 'key', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 17), tuple_63713, key_63714)
        # Adding element type (line 277)
        # Getting the type of 'self' (line 277)
        self_63715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 22), 'self', False)
        # Obtaining the member '__class__' of a type (line 277)
        class___63716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 22), self_63715, '__class__')
        # Obtaining the member '__name__' of a type (line 277)
        name___63717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 22), class___63716, '__name__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 17), tuple_63713, name___63717)
        
        # Applying the binary operator '%' (line 276)
        result_mod_63718 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 16), '%', str_63712, tuple_63713)
        
        # Processing the call keyword arguments (line 275)
        kwargs_63719 = {}
        # Getting the type of 'ValueError' (line 275)
        ValueError_63711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 275)
        ValueError_call_result_63720 = invoke(stypy.reporting.localization.Localization(__file__, 275, 18), ValueError_63711, *[result_mod_63718], **kwargs_63719)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 275, 12), ValueError_call_result_63720, 'raise parameter', BaseException)
        # SSA join for if statement (line 274)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to is_string(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'value' (line 278)
        value_63722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 21), 'value', False)
        # Processing the call keyword arguments (line 278)
        kwargs_63723 = {}
        # Getting the type of 'is_string' (line 278)
        is_string_63721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 11), 'is_string', False)
        # Calling is_string(args, kwargs) (line 278)
        is_string_call_result_63724 = invoke(stypy.reporting.localization.Localization(__file__, 278, 11), is_string_63721, *[value_63722], **kwargs_63723)
        
        # Testing the type of an if condition (line 278)
        if_condition_63725 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 8), is_string_call_result_63724)
        # Assigning a type to the variable 'if_condition_63725' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'if_condition_63725', if_condition_63725)
        # SSA begins for if statement (line 278)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 279):
        
        # Assigning a Call to a Name (line 279):
        
        # Call to split_quoted(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'value' (line 279)
        value_63727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 33), 'value', False)
        # Processing the call keyword arguments (line 279)
        kwargs_63728 = {}
        # Getting the type of 'split_quoted' (line 279)
        split_quoted_63726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 20), 'split_quoted', False)
        # Calling split_quoted(args, kwargs) (line 279)
        split_quoted_call_result_63729 = invoke(stypy.reporting.localization.Localization(__file__, 279, 20), split_quoted_63726, *[value_63727], **kwargs_63728)
        
        # Assigning a type to the variable 'value' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'value', split_quoted_call_result_63729)
        # SSA join for if statement (line 278)
        module_type_store = module_type_store.join_ssa_context()
        
        # Evaluating assert statement condition
        
        # Evaluating a boolean operation
        
        # Getting the type of 'value' (line 280)
        value_63730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'value')
        # Getting the type of 'None' (line 280)
        None_63731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 24), 'None')
        # Applying the binary operator 'is' (line 280)
        result_is__63732 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 15), 'is', value_63730, None_63731)
        
        
        # Call to is_sequence_of_strings(...): (line 280)
        # Processing the call arguments (line 280)
        
        # Obtaining the type of the subscript
        int_63734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 61), 'int')
        slice_63735 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 280, 55), int_63734, None, None)
        # Getting the type of 'value' (line 280)
        value_63736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 55), 'value', False)
        # Obtaining the member '__getitem__' of a type (line 280)
        getitem___63737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 55), value_63736, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 280)
        subscript_call_result_63738 = invoke(stypy.reporting.localization.Localization(__file__, 280, 55), getitem___63737, slice_63735)
        
        # Processing the call keyword arguments (line 280)
        kwargs_63739 = {}
        # Getting the type of 'is_sequence_of_strings' (line 280)
        is_sequence_of_strings_63733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 32), 'is_sequence_of_strings', False)
        # Calling is_sequence_of_strings(args, kwargs) (line 280)
        is_sequence_of_strings_call_result_63740 = invoke(stypy.reporting.localization.Localization(__file__, 280, 32), is_sequence_of_strings_63733, *[subscript_call_result_63738], **kwargs_63739)
        
        # Applying the binary operator 'or' (line 280)
        result_or_keyword_63741 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 15), 'or', result_is__63732, is_sequence_of_strings_call_result_63740)
        
        
        # Assigning a Name to a Subscript (line 281):
        
        # Assigning a Name to a Subscript (line 281):
        # Getting the type of 'value' (line 281)
        value_63742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 32), 'value')
        # Getting the type of 'self' (line 281)
        self_63743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'self')
        # Obtaining the member 'executables' of a type (line 281)
        executables_63744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), self_63743, 'executables')
        # Getting the type of 'key' (line 281)
        key_63745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 25), 'key')
        # Storing an element on a container (line 281)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 8), executables_63744, (key_63745, value_63742))
        
        # ################# End of 'set_command(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_command' in the type store
        # Getting the type of 'stypy_return_type' (line 273)
        stypy_return_type_63746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63746)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_command'
        return stypy_return_type_63746


    @norecursion
    def find_executables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find_executables'
        module_type_store = module_type_store.open_function_context('find_executables', 288, 4, False)
        # Assigning a type to the variable 'self' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.find_executables.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.find_executables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.find_executables.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.find_executables.__dict__.__setitem__('stypy_function_name', 'FCompiler.find_executables')
        FCompiler.find_executables.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.find_executables.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.find_executables.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.find_executables.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.find_executables.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.find_executables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.find_executables.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.find_executables', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_executables', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_executables(...)' code ##################

        str_63747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, (-1)), 'str', 'Go through the self.executables dictionary, and attempt to\n        find and assign appropiate executables.\n\n        Executable names are looked for in the environment (environment\n        variables, the distutils.cfg, and command line), the 0th-element of\n        the command list, and the self.possible_executables list.\n\n        Also, if the 0th element is "<F77>" or "<F90>", the Fortran 77\n        or the Fortran 90 compiler executable is used, unless overridden\n        by an environment setting.\n\n        Subclasses should call this if overriden.\n        ')
        # Evaluating assert statement condition
        # Getting the type of 'self' (line 302)
        self_63748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 15), 'self')
        # Obtaining the member '_is_customised' of a type (line 302)
        _is_customised_63749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 15), self_63748, '_is_customised')
        
        # Assigning a Attribute to a Name (line 303):
        
        # Assigning a Attribute to a Name (line 303):
        # Getting the type of 'self' (line 303)
        self_63750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'self')
        # Obtaining the member '_exe_cache' of a type (line 303)
        _exe_cache_63751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 20), self_63750, '_exe_cache')
        # Assigning a type to the variable 'exe_cache' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'exe_cache', _exe_cache_63751)

        @norecursion
        def cached_find_executable(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cached_find_executable'
            module_type_store = module_type_store.open_function_context('cached_find_executable', 304, 8, False)
            
            # Passed parameters checking function
            cached_find_executable.stypy_localization = localization
            cached_find_executable.stypy_type_of_self = None
            cached_find_executable.stypy_type_store = module_type_store
            cached_find_executable.stypy_function_name = 'cached_find_executable'
            cached_find_executable.stypy_param_names_list = ['exe']
            cached_find_executable.stypy_varargs_param_name = None
            cached_find_executable.stypy_kwargs_param_name = None
            cached_find_executable.stypy_call_defaults = defaults
            cached_find_executable.stypy_call_varargs = varargs
            cached_find_executable.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'cached_find_executable', ['exe'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'cached_find_executable', localization, ['exe'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'cached_find_executable(...)' code ##################

            
            
            # Getting the type of 'exe' (line 305)
            exe_63752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 15), 'exe')
            # Getting the type of 'exe_cache' (line 305)
            exe_cache_63753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 22), 'exe_cache')
            # Applying the binary operator 'in' (line 305)
            result_contains_63754 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 15), 'in', exe_63752, exe_cache_63753)
            
            # Testing the type of an if condition (line 305)
            if_condition_63755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 12), result_contains_63754)
            # Assigning a type to the variable 'if_condition_63755' (line 305)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'if_condition_63755', if_condition_63755)
            # SSA begins for if statement (line 305)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            # Getting the type of 'exe' (line 306)
            exe_63756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 33), 'exe')
            # Getting the type of 'exe_cache' (line 306)
            exe_cache_63757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 23), 'exe_cache')
            # Obtaining the member '__getitem__' of a type (line 306)
            getitem___63758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 23), exe_cache_63757, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 306)
            subscript_call_result_63759 = invoke(stypy.reporting.localization.Localization(__file__, 306, 23), getitem___63758, exe_63756)
            
            # Assigning a type to the variable 'stypy_return_type' (line 306)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 16), 'stypy_return_type', subscript_call_result_63759)
            # SSA join for if statement (line 305)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 307):
            
            # Assigning a Call to a Name (line 307):
            
            # Call to find_executable(...): (line 307)
            # Processing the call arguments (line 307)
            # Getting the type of 'exe' (line 307)
            exe_63761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 37), 'exe', False)
            # Processing the call keyword arguments (line 307)
            kwargs_63762 = {}
            # Getting the type of 'find_executable' (line 307)
            find_executable_63760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 21), 'find_executable', False)
            # Calling find_executable(args, kwargs) (line 307)
            find_executable_call_result_63763 = invoke(stypy.reporting.localization.Localization(__file__, 307, 21), find_executable_63760, *[exe_63761], **kwargs_63762)
            
            # Assigning a type to the variable 'fc_exe' (line 307)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 12), 'fc_exe', find_executable_call_result_63763)
            
            # Multiple assignment of 2 elements.
            
            # Assigning a Name to a Subscript (line 308):
            # Getting the type of 'fc_exe' (line 308)
            fc_exe_63764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 49), 'fc_exe')
            # Getting the type of 'exe_cache' (line 308)
            exe_cache_63765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 29), 'exe_cache')
            # Getting the type of 'fc_exe' (line 308)
            fc_exe_63766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 39), 'fc_exe')
            # Storing an element on a container (line 308)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 29), exe_cache_63765, (fc_exe_63766, fc_exe_63764))
            
            # Assigning a Subscript to a Subscript (line 308):
            
            # Obtaining the type of the subscript
            # Getting the type of 'fc_exe' (line 308)
            fc_exe_63767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 39), 'fc_exe')
            # Getting the type of 'exe_cache' (line 308)
            exe_cache_63768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 29), 'exe_cache')
            # Obtaining the member '__getitem__' of a type (line 308)
            getitem___63769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 29), exe_cache_63768, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 308)
            subscript_call_result_63770 = invoke(stypy.reporting.localization.Localization(__file__, 308, 29), getitem___63769, fc_exe_63767)
            
            # Getting the type of 'exe_cache' (line 308)
            exe_cache_63771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 12), 'exe_cache')
            # Getting the type of 'exe' (line 308)
            exe_63772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 22), 'exe')
            # Storing an element on a container (line 308)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 12), exe_cache_63771, (exe_63772, subscript_call_result_63770))
            # Getting the type of 'fc_exe' (line 309)
            fc_exe_63773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 'fc_exe')
            # Assigning a type to the variable 'stypy_return_type' (line 309)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'stypy_return_type', fc_exe_63773)
            
            # ################# End of 'cached_find_executable(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cached_find_executable' in the type store
            # Getting the type of 'stypy_return_type' (line 304)
            stypy_return_type_63774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_63774)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cached_find_executable'
            return stypy_return_type_63774

        # Assigning a type to the variable 'cached_find_executable' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'cached_find_executable', cached_find_executable)

        @norecursion
        def verify_command_form(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'verify_command_form'
            module_type_store = module_type_store.open_function_context('verify_command_form', 310, 8, False)
            
            # Passed parameters checking function
            verify_command_form.stypy_localization = localization
            verify_command_form.stypy_type_of_self = None
            verify_command_form.stypy_type_store = module_type_store
            verify_command_form.stypy_function_name = 'verify_command_form'
            verify_command_form.stypy_param_names_list = ['name', 'value']
            verify_command_form.stypy_varargs_param_name = None
            verify_command_form.stypy_kwargs_param_name = None
            verify_command_form.stypy_call_defaults = defaults
            verify_command_form.stypy_call_varargs = varargs
            verify_command_form.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'verify_command_form', ['name', 'value'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'verify_command_form', localization, ['name', 'value'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'verify_command_form(...)' code ##################

            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'value' (line 311)
            value_63775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 15), 'value')
            # Getting the type of 'None' (line 311)
            None_63776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 28), 'None')
            # Applying the binary operator 'isnot' (line 311)
            result_is_not_63777 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 15), 'isnot', value_63775, None_63776)
            
            
            
            # Call to is_sequence_of_strings(...): (line 311)
            # Processing the call arguments (line 311)
            # Getting the type of 'value' (line 311)
            value_63779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 64), 'value', False)
            # Processing the call keyword arguments (line 311)
            kwargs_63780 = {}
            # Getting the type of 'is_sequence_of_strings' (line 311)
            is_sequence_of_strings_63778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 41), 'is_sequence_of_strings', False)
            # Calling is_sequence_of_strings(args, kwargs) (line 311)
            is_sequence_of_strings_call_result_63781 = invoke(stypy.reporting.localization.Localization(__file__, 311, 41), is_sequence_of_strings_63778, *[value_63779], **kwargs_63780)
            
            # Applying the 'not' unary operator (line 311)
            result_not__63782 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 37), 'not', is_sequence_of_strings_call_result_63781)
            
            # Applying the binary operator 'and' (line 311)
            result_and_keyword_63783 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 15), 'and', result_is_not_63777, result_not__63782)
            
            # Testing the type of an if condition (line 311)
            if_condition_63784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 12), result_and_keyword_63783)
            # Assigning a type to the variable 'if_condition_63784' (line 311)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'if_condition_63784', if_condition_63784)
            # SSA begins for if statement (line 311)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 312)
            # Processing the call arguments (line 312)
            str_63786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 20), 'str', '%s value %r is invalid in class %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 314)
            tuple_63787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 21), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 314)
            # Adding element type (line 314)
            # Getting the type of 'name' (line 314)
            name_63788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 21), 'name', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 21), tuple_63787, name_63788)
            # Adding element type (line 314)
            # Getting the type of 'value' (line 314)
            value_63789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 27), 'value', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 21), tuple_63787, value_63789)
            # Adding element type (line 314)
            # Getting the type of 'self' (line 314)
            self_63790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 34), 'self', False)
            # Obtaining the member '__class__' of a type (line 314)
            class___63791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 34), self_63790, '__class__')
            # Obtaining the member '__name__' of a type (line 314)
            name___63792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 34), class___63791, '__name__')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 314, 21), tuple_63787, name___63792)
            
            # Applying the binary operator '%' (line 313)
            result_mod_63793 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 20), '%', str_63786, tuple_63787)
            
            # Processing the call keyword arguments (line 312)
            kwargs_63794 = {}
            # Getting the type of 'ValueError' (line 312)
            ValueError_63785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 22), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 312)
            ValueError_call_result_63795 = invoke(stypy.reporting.localization.Localization(__file__, 312, 22), ValueError_63785, *[result_mod_63793], **kwargs_63794)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 312, 16), ValueError_call_result_63795, 'raise parameter', BaseException)
            # SSA join for if statement (line 311)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'verify_command_form(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'verify_command_form' in the type store
            # Getting the type of 'stypy_return_type' (line 310)
            stypy_return_type_63796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_63796)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'verify_command_form'
            return stypy_return_type_63796

        # Assigning a type to the variable 'verify_command_form' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'verify_command_form', verify_command_form)

        @norecursion
        def set_exe(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            # Getting the type of 'None' (line 315)
            None_63797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 33), 'None')
            # Getting the type of 'None' (line 315)
            None_63798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 43), 'None')
            defaults = [None_63797, None_63798]
            # Create a new context for function 'set_exe'
            module_type_store = module_type_store.open_function_context('set_exe', 315, 8, False)
            
            # Passed parameters checking function
            set_exe.stypy_localization = localization
            set_exe.stypy_type_of_self = None
            set_exe.stypy_type_store = module_type_store
            set_exe.stypy_function_name = 'set_exe'
            set_exe.stypy_param_names_list = ['exe_key', 'f77', 'f90']
            set_exe.stypy_varargs_param_name = None
            set_exe.stypy_kwargs_param_name = None
            set_exe.stypy_call_defaults = defaults
            set_exe.stypy_call_varargs = varargs
            set_exe.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'set_exe', ['exe_key', 'f77', 'f90'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'set_exe', localization, ['exe_key', 'f77', 'f90'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'set_exe(...)' code ##################

            
            # Assigning a Call to a Name (line 316):
            
            # Assigning a Call to a Name (line 316):
            
            # Call to get(...): (line 316)
            # Processing the call arguments (line 316)
            # Getting the type of 'exe_key' (line 316)
            exe_key_63802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 39), 'exe_key', False)
            # Getting the type of 'None' (line 316)
            None_63803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 48), 'None', False)
            # Processing the call keyword arguments (line 316)
            kwargs_63804 = {}
            # Getting the type of 'self' (line 316)
            self_63799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 18), 'self', False)
            # Obtaining the member 'executables' of a type (line 316)
            executables_63800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 18), self_63799, 'executables')
            # Obtaining the member 'get' of a type (line 316)
            get_63801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 18), executables_63800, 'get')
            # Calling get(args, kwargs) (line 316)
            get_call_result_63805 = invoke(stypy.reporting.localization.Localization(__file__, 316, 18), get_63801, *[exe_key_63802, None_63803], **kwargs_63804)
            
            # Assigning a type to the variable 'cmd' (line 316)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'cmd', get_call_result_63805)
            
            
            # Getting the type of 'cmd' (line 317)
            cmd_63806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 19), 'cmd')
            # Applying the 'not' unary operator (line 317)
            result_not__63807 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 15), 'not', cmd_63806)
            
            # Testing the type of an if condition (line 317)
            if_condition_63808 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 12), result_not__63807)
            # Assigning a type to the variable 'if_condition_63808' (line 317)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'if_condition_63808', if_condition_63808)
            # SSA begins for if statement (line 317)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'None' (line 318)
            None_63809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 23), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 318)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 16), 'stypy_return_type', None_63809)
            # SSA join for if statement (line 317)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 321):
            
            # Assigning a Call to a Name (line 321):
            
            # Call to getattr(...): (line 321)
            # Processing the call arguments (line 321)
            # Getting the type of 'self' (line 321)
            self_63811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 39), 'self', False)
            # Obtaining the member 'command_vars' of a type (line 321)
            command_vars_63812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 39), self_63811, 'command_vars')
            # Getting the type of 'exe_key' (line 321)
            exe_key_63813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 58), 'exe_key', False)
            # Processing the call keyword arguments (line 321)
            kwargs_63814 = {}
            # Getting the type of 'getattr' (line 321)
            getattr_63810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 31), 'getattr', False)
            # Calling getattr(args, kwargs) (line 321)
            getattr_call_result_63815 = invoke(stypy.reporting.localization.Localization(__file__, 321, 31), getattr_63810, *[command_vars_63812, exe_key_63813], **kwargs_63814)
            
            # Assigning a type to the variable 'exe_from_environ' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'exe_from_environ', getattr_call_result_63815)
            
            
            # Getting the type of 'exe_from_environ' (line 322)
            exe_from_environ_63816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 19), 'exe_from_environ')
            # Applying the 'not' unary operator (line 322)
            result_not__63817 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 15), 'not', exe_from_environ_63816)
            
            # Testing the type of an if condition (line 322)
            if_condition_63818 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 12), result_not__63817)
            # Assigning a type to the variable 'if_condition_63818' (line 322)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'if_condition_63818', if_condition_63818)
            # SSA begins for if statement (line 322)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 323):
            
            # Assigning a BinOp to a Name (line 323):
            
            # Obtaining an instance of the builtin type 'list' (line 323)
            list_63819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 323)
            # Adding element type (line 323)
            # Getting the type of 'f90' (line 323)
            f90_63820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 29), 'f90')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 28), list_63819, f90_63820)
            # Adding element type (line 323)
            # Getting the type of 'f77' (line 323)
            f77_63821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 34), 'f77')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 323, 28), list_63819, f77_63821)
            
            # Getting the type of 'self' (line 323)
            self_63822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 41), 'self')
            # Obtaining the member 'possible_executables' of a type (line 323)
            possible_executables_63823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 41), self_63822, 'possible_executables')
            # Applying the binary operator '+' (line 323)
            result_add_63824 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 28), '+', list_63819, possible_executables_63823)
            
            # Assigning a type to the variable 'possibles' (line 323)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 16), 'possibles', result_add_63824)
            # SSA branch for the else part of an if statement (line 322)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 325):
            
            # Assigning a BinOp to a Name (line 325):
            
            # Obtaining an instance of the builtin type 'list' (line 325)
            list_63825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 325)
            # Adding element type (line 325)
            # Getting the type of 'exe_from_environ' (line 325)
            exe_from_environ_63826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 29), 'exe_from_environ')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 325, 28), list_63825, exe_from_environ_63826)
            
            # Getting the type of 'self' (line 325)
            self_63827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 49), 'self')
            # Obtaining the member 'possible_executables' of a type (line 325)
            possible_executables_63828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 49), self_63827, 'possible_executables')
            # Applying the binary operator '+' (line 325)
            result_add_63829 = python_operator(stypy.reporting.localization.Localization(__file__, 325, 28), '+', list_63825, possible_executables_63828)
            
            # Assigning a type to the variable 'possibles' (line 325)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 16), 'possibles', result_add_63829)
            # SSA join for if statement (line 322)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 327):
            
            # Assigning a Call to a Name (line 327):
            
            # Call to set(...): (line 327)
            # Processing the call keyword arguments (line 327)
            kwargs_63831 = {}
            # Getting the type of 'set' (line 327)
            set_63830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 19), 'set', False)
            # Calling set(args, kwargs) (line 327)
            set_call_result_63832 = invoke(stypy.reporting.localization.Localization(__file__, 327, 19), set_63830, *[], **kwargs_63831)
            
            # Assigning a type to the variable 'seen' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 12), 'seen', set_call_result_63832)
            
            # Assigning a List to a Name (line 328):
            
            # Assigning a List to a Name (line 328):
            
            # Obtaining an instance of the builtin type 'list' (line 328)
            list_63833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 31), 'list')
            # Adding type elements to the builtin type 'list' instance (line 328)
            
            # Assigning a type to the variable 'unique_possibles' (line 328)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 'unique_possibles', list_63833)
            
            # Getting the type of 'possibles' (line 329)
            possibles_63834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 21), 'possibles')
            # Testing the type of a for loop iterable (line 329)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 329, 12), possibles_63834)
            # Getting the type of the for loop variable (line 329)
            for_loop_var_63835 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 329, 12), possibles_63834)
            # Assigning a type to the variable 'e' (line 329)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'e', for_loop_var_63835)
            # SSA begins for a for statement (line 329)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 'e' (line 330)
            e_63836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 19), 'e')
            str_63837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 24), 'str', '<F77>')
            # Applying the binary operator '==' (line 330)
            result_eq_63838 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 19), '==', e_63836, str_63837)
            
            # Testing the type of an if condition (line 330)
            if_condition_63839 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 16), result_eq_63838)
            # Assigning a type to the variable 'if_condition_63839' (line 330)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 16), 'if_condition_63839', if_condition_63839)
            # SSA begins for if statement (line 330)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 331):
            
            # Assigning a Name to a Name (line 331):
            # Getting the type of 'f77' (line 331)
            f77_63840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 24), 'f77')
            # Assigning a type to the variable 'e' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 20), 'e', f77_63840)
            # SSA branch for the else part of an if statement (line 330)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'e' (line 332)
            e_63841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 21), 'e')
            str_63842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 26), 'str', '<F90>')
            # Applying the binary operator '==' (line 332)
            result_eq_63843 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 21), '==', e_63841, str_63842)
            
            # Testing the type of an if condition (line 332)
            if_condition_63844 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 21), result_eq_63843)
            # Assigning a type to the variable 'if_condition_63844' (line 332)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 21), 'if_condition_63844', if_condition_63844)
            # SSA begins for if statement (line 332)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 333):
            
            # Assigning a Name to a Name (line 333):
            # Getting the type of 'f90' (line 333)
            f90_63845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 24), 'f90')
            # Assigning a type to the variable 'e' (line 333)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 20), 'e', f90_63845)
            # SSA join for if statement (line 332)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 330)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'e' (line 334)
            e_63846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 23), 'e')
            # Applying the 'not' unary operator (line 334)
            result_not__63847 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 19), 'not', e_63846)
            
            
            # Getting the type of 'e' (line 334)
            e_63848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 28), 'e')
            # Getting the type of 'seen' (line 334)
            seen_63849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 33), 'seen')
            # Applying the binary operator 'in' (line 334)
            result_contains_63850 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 28), 'in', e_63848, seen_63849)
            
            # Applying the binary operator 'or' (line 334)
            result_or_keyword_63851 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 19), 'or', result_not__63847, result_contains_63850)
            
            # Testing the type of an if condition (line 334)
            if_condition_63852 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 16), result_or_keyword_63851)
            # Assigning a type to the variable 'if_condition_63852' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 16), 'if_condition_63852', if_condition_63852)
            # SSA begins for if statement (line 334)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 334)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to add(...): (line 336)
            # Processing the call arguments (line 336)
            # Getting the type of 'e' (line 336)
            e_63855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 25), 'e', False)
            # Processing the call keyword arguments (line 336)
            kwargs_63856 = {}
            # Getting the type of 'seen' (line 336)
            seen_63853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 16), 'seen', False)
            # Obtaining the member 'add' of a type (line 336)
            add_63854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 16), seen_63853, 'add')
            # Calling add(args, kwargs) (line 336)
            add_call_result_63857 = invoke(stypy.reporting.localization.Localization(__file__, 336, 16), add_63854, *[e_63855], **kwargs_63856)
            
            
            # Call to append(...): (line 337)
            # Processing the call arguments (line 337)
            # Getting the type of 'e' (line 337)
            e_63860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 40), 'e', False)
            # Processing the call keyword arguments (line 337)
            kwargs_63861 = {}
            # Getting the type of 'unique_possibles' (line 337)
            unique_possibles_63858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 16), 'unique_possibles', False)
            # Obtaining the member 'append' of a type (line 337)
            append_63859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 16), unique_possibles_63858, 'append')
            # Calling append(args, kwargs) (line 337)
            append_call_result_63862 = invoke(stypy.reporting.localization.Localization(__file__, 337, 16), append_63859, *[e_63860], **kwargs_63861)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Getting the type of 'unique_possibles' (line 339)
            unique_possibles_63863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 23), 'unique_possibles')
            # Testing the type of a for loop iterable (line 339)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 339, 12), unique_possibles_63863)
            # Getting the type of the for loop variable (line 339)
            for_loop_var_63864 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 339, 12), unique_possibles_63863)
            # Assigning a type to the variable 'exe' (line 339)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'exe', for_loop_var_63864)
            # SSA begins for a for statement (line 339)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 340):
            
            # Assigning a Call to a Name (line 340):
            
            # Call to cached_find_executable(...): (line 340)
            # Processing the call arguments (line 340)
            # Getting the type of 'exe' (line 340)
            exe_63866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 48), 'exe', False)
            # Processing the call keyword arguments (line 340)
            kwargs_63867 = {}
            # Getting the type of 'cached_find_executable' (line 340)
            cached_find_executable_63865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 25), 'cached_find_executable', False)
            # Calling cached_find_executable(args, kwargs) (line 340)
            cached_find_executable_call_result_63868 = invoke(stypy.reporting.localization.Localization(__file__, 340, 25), cached_find_executable_63865, *[exe_63866], **kwargs_63867)
            
            # Assigning a type to the variable 'fc_exe' (line 340)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 16), 'fc_exe', cached_find_executable_call_result_63868)
            
            # Getting the type of 'fc_exe' (line 341)
            fc_exe_63869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 19), 'fc_exe')
            # Testing the type of an if condition (line 341)
            if_condition_63870 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 16), fc_exe_63869)
            # Assigning a type to the variable 'if_condition_63870' (line 341)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 16), 'if_condition_63870', if_condition_63870)
            # SSA begins for if statement (line 341)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Subscript (line 342):
            
            # Assigning a Name to a Subscript (line 342):
            # Getting the type of 'fc_exe' (line 342)
            fc_exe_63871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 29), 'fc_exe')
            # Getting the type of 'cmd' (line 342)
            cmd_63872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 20), 'cmd')
            int_63873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 24), 'int')
            # Storing an element on a container (line 342)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 342, 20), cmd_63872, (int_63873, fc_exe_63871))
            # Getting the type of 'fc_exe' (line 343)
            fc_exe_63874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 27), 'fc_exe')
            # Assigning a type to the variable 'stypy_return_type' (line 343)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 20), 'stypy_return_type', fc_exe_63874)
            # SSA join for if statement (line 341)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to set_command(...): (line 344)
            # Processing the call arguments (line 344)
            # Getting the type of 'exe_key' (line 344)
            exe_key_63877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 29), 'exe_key', False)
            # Getting the type of 'None' (line 344)
            None_63878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 38), 'None', False)
            # Processing the call keyword arguments (line 344)
            kwargs_63879 = {}
            # Getting the type of 'self' (line 344)
            self_63875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'self', False)
            # Obtaining the member 'set_command' of a type (line 344)
            set_command_63876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 12), self_63875, 'set_command')
            # Calling set_command(args, kwargs) (line 344)
            set_command_call_result_63880 = invoke(stypy.reporting.localization.Localization(__file__, 344, 12), set_command_63876, *[exe_key_63877, None_63878], **kwargs_63879)
            
            # Getting the type of 'None' (line 345)
            None_63881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 345)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'stypy_return_type', None_63881)
            
            # ################# End of 'set_exe(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'set_exe' in the type store
            # Getting the type of 'stypy_return_type' (line 315)
            stypy_return_type_63882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_63882)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'set_exe'
            return stypy_return_type_63882

        # Assigning a type to the variable 'set_exe' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'set_exe', set_exe)
        
        # Assigning a Attribute to a Name (line 347):
        
        # Assigning a Attribute to a Name (line 347):
        # Getting the type of 'self' (line 347)
        self_63883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 16), 'self')
        # Obtaining the member 'compiler_type' of a type (line 347)
        compiler_type_63884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 16), self_63883, 'compiler_type')
        # Assigning a type to the variable 'ctype' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'ctype', compiler_type_63884)
        
        # Assigning a Call to a Name (line 348):
        
        # Assigning a Call to a Name (line 348):
        
        # Call to set_exe(...): (line 348)
        # Processing the call arguments (line 348)
        str_63886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 22), 'str', 'compiler_f90')
        # Processing the call keyword arguments (line 348)
        kwargs_63887 = {}
        # Getting the type of 'set_exe' (line 348)
        set_exe_63885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 14), 'set_exe', False)
        # Calling set_exe(args, kwargs) (line 348)
        set_exe_call_result_63888 = invoke(stypy.reporting.localization.Localization(__file__, 348, 14), set_exe_63885, *[str_63886], **kwargs_63887)
        
        # Assigning a type to the variable 'f90' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'f90', set_exe_call_result_63888)
        
        
        # Getting the type of 'f90' (line 349)
        f90_63889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 15), 'f90')
        # Applying the 'not' unary operator (line 349)
        result_not__63890 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 11), 'not', f90_63889)
        
        # Testing the type of an if condition (line 349)
        if_condition_63891 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 349, 8), result_not__63890)
        # Assigning a type to the variable 'if_condition_63891' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'if_condition_63891', if_condition_63891)
        # SSA begins for if statement (line 349)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 350):
        
        # Assigning a Call to a Name (line 350):
        
        # Call to set_exe(...): (line 350)
        # Processing the call arguments (line 350)
        str_63893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 26), 'str', 'compiler_f77')
        # Processing the call keyword arguments (line 350)
        kwargs_63894 = {}
        # Getting the type of 'set_exe' (line 350)
        set_exe_63892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 18), 'set_exe', False)
        # Calling set_exe(args, kwargs) (line 350)
        set_exe_call_result_63895 = invoke(stypy.reporting.localization.Localization(__file__, 350, 18), set_exe_63892, *[str_63893], **kwargs_63894)
        
        # Assigning a type to the variable 'f77' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'f77', set_exe_call_result_63895)
        
        # Getting the type of 'f77' (line 351)
        f77_63896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 15), 'f77')
        # Testing the type of an if condition (line 351)
        if_condition_63897 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 351, 12), f77_63896)
        # Assigning a type to the variable 'if_condition_63897' (line 351)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'if_condition_63897', if_condition_63897)
        # SSA begins for if statement (line 351)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 352)
        # Processing the call arguments (line 352)
        str_63900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 25), 'str', '%s: no Fortran 90 compiler found')
        # Getting the type of 'ctype' (line 352)
        ctype_63901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 62), 'ctype', False)
        # Applying the binary operator '%' (line 352)
        result_mod_63902 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 25), '%', str_63900, ctype_63901)
        
        # Processing the call keyword arguments (line 352)
        kwargs_63903 = {}
        # Getting the type of 'log' (line 352)
        log_63898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 16), 'log', False)
        # Obtaining the member 'warn' of a type (line 352)
        warn_63899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 16), log_63898, 'warn')
        # Calling warn(args, kwargs) (line 352)
        warn_call_result_63904 = invoke(stypy.reporting.localization.Localization(__file__, 352, 16), warn_63899, *[result_mod_63902], **kwargs_63903)
        
        # SSA branch for the else part of an if statement (line 351)
        module_type_store.open_ssa_branch('else')
        
        # Call to CompilerNotFound(...): (line 354)
        # Processing the call arguments (line 354)
        str_63906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 39), 'str', '%s: f90 nor f77')
        # Getting the type of 'ctype' (line 354)
        ctype_63907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 59), 'ctype', False)
        # Applying the binary operator '%' (line 354)
        result_mod_63908 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 39), '%', str_63906, ctype_63907)
        
        # Processing the call keyword arguments (line 354)
        kwargs_63909 = {}
        # Getting the type of 'CompilerNotFound' (line 354)
        CompilerNotFound_63905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 22), 'CompilerNotFound', False)
        # Calling CompilerNotFound(args, kwargs) (line 354)
        CompilerNotFound_call_result_63910 = invoke(stypy.reporting.localization.Localization(__file__, 354, 22), CompilerNotFound_63905, *[result_mod_63908], **kwargs_63909)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 354, 16), CompilerNotFound_call_result_63910, 'raise parameter', BaseException)
        # SSA join for if statement (line 351)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 349)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 356):
        
        # Assigning a Call to a Name (line 356):
        
        # Call to set_exe(...): (line 356)
        # Processing the call arguments (line 356)
        str_63912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 26), 'str', 'compiler_f77')
        # Processing the call keyword arguments (line 356)
        # Getting the type of 'f90' (line 356)
        f90_63913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 46), 'f90', False)
        keyword_63914 = f90_63913
        kwargs_63915 = {'f90': keyword_63914}
        # Getting the type of 'set_exe' (line 356)
        set_exe_63911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 18), 'set_exe', False)
        # Calling set_exe(args, kwargs) (line 356)
        set_exe_call_result_63916 = invoke(stypy.reporting.localization.Localization(__file__, 356, 18), set_exe_63911, *[str_63912], **kwargs_63915)
        
        # Assigning a type to the variable 'f77' (line 356)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'f77', set_exe_call_result_63916)
        
        
        # Getting the type of 'f77' (line 357)
        f77_63917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 19), 'f77')
        # Applying the 'not' unary operator (line 357)
        result_not__63918 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 15), 'not', f77_63917)
        
        # Testing the type of an if condition (line 357)
        if_condition_63919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 12), result_not__63918)
        # Assigning a type to the variable 'if_condition_63919' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'if_condition_63919', if_condition_63919)
        # SSA begins for if statement (line 357)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 358)
        # Processing the call arguments (line 358)
        str_63922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 25), 'str', '%s: no Fortran 77 compiler found')
        # Getting the type of 'ctype' (line 358)
        ctype_63923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 62), 'ctype', False)
        # Applying the binary operator '%' (line 358)
        result_mod_63924 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 25), '%', str_63922, ctype_63923)
        
        # Processing the call keyword arguments (line 358)
        kwargs_63925 = {}
        # Getting the type of 'log' (line 358)
        log_63920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'log', False)
        # Obtaining the member 'warn' of a type (line 358)
        warn_63921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 16), log_63920, 'warn')
        # Calling warn(args, kwargs) (line 358)
        warn_call_result_63926 = invoke(stypy.reporting.localization.Localization(__file__, 358, 16), warn_63921, *[result_mod_63924], **kwargs_63925)
        
        # SSA join for if statement (line 357)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_exe(...): (line 359)
        # Processing the call arguments (line 359)
        str_63928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 20), 'str', 'compiler_fix')
        # Processing the call keyword arguments (line 359)
        # Getting the type of 'f90' (line 359)
        f90_63929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 40), 'f90', False)
        keyword_63930 = f90_63929
        kwargs_63931 = {'f90': keyword_63930}
        # Getting the type of 'set_exe' (line 359)
        set_exe_63927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'set_exe', False)
        # Calling set_exe(args, kwargs) (line 359)
        set_exe_call_result_63932 = invoke(stypy.reporting.localization.Localization(__file__, 359, 12), set_exe_63927, *[str_63928], **kwargs_63931)
        
        # SSA join for if statement (line 349)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_exe(...): (line 361)
        # Processing the call arguments (line 361)
        str_63934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 16), 'str', 'linker_so')
        # Processing the call keyword arguments (line 361)
        # Getting the type of 'f77' (line 361)
        f77_63935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 33), 'f77', False)
        keyword_63936 = f77_63935
        # Getting the type of 'f90' (line 361)
        f90_63937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 42), 'f90', False)
        keyword_63938 = f90_63937
        kwargs_63939 = {'f90': keyword_63938, 'f77': keyword_63936}
        # Getting the type of 'set_exe' (line 361)
        set_exe_63933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'set_exe', False)
        # Calling set_exe(args, kwargs) (line 361)
        set_exe_call_result_63940 = invoke(stypy.reporting.localization.Localization(__file__, 361, 8), set_exe_63933, *[str_63934], **kwargs_63939)
        
        
        # Call to set_exe(...): (line 362)
        # Processing the call arguments (line 362)
        str_63942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 16), 'str', 'linker_exe')
        # Processing the call keyword arguments (line 362)
        # Getting the type of 'f77' (line 362)
        f77_63943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 34), 'f77', False)
        keyword_63944 = f77_63943
        # Getting the type of 'f90' (line 362)
        f90_63945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 43), 'f90', False)
        keyword_63946 = f90_63945
        kwargs_63947 = {'f90': keyword_63946, 'f77': keyword_63944}
        # Getting the type of 'set_exe' (line 362)
        set_exe_63941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'set_exe', False)
        # Calling set_exe(args, kwargs) (line 362)
        set_exe_call_result_63948 = invoke(stypy.reporting.localization.Localization(__file__, 362, 8), set_exe_63941, *[str_63942], **kwargs_63947)
        
        
        # Call to set_exe(...): (line 363)
        # Processing the call arguments (line 363)
        str_63950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 16), 'str', 'version_cmd')
        # Processing the call keyword arguments (line 363)
        # Getting the type of 'f77' (line 363)
        f77_63951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 35), 'f77', False)
        keyword_63952 = f77_63951
        # Getting the type of 'f90' (line 363)
        f90_63953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 44), 'f90', False)
        keyword_63954 = f90_63953
        kwargs_63955 = {'f90': keyword_63954, 'f77': keyword_63952}
        # Getting the type of 'set_exe' (line 363)
        set_exe_63949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'set_exe', False)
        # Calling set_exe(args, kwargs) (line 363)
        set_exe_call_result_63956 = invoke(stypy.reporting.localization.Localization(__file__, 363, 8), set_exe_63949, *[str_63950], **kwargs_63955)
        
        
        # Call to set_exe(...): (line 364)
        # Processing the call arguments (line 364)
        str_63958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 16), 'str', 'archiver')
        # Processing the call keyword arguments (line 364)
        kwargs_63959 = {}
        # Getting the type of 'set_exe' (line 364)
        set_exe_63957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 8), 'set_exe', False)
        # Calling set_exe(args, kwargs) (line 364)
        set_exe_call_result_63960 = invoke(stypy.reporting.localization.Localization(__file__, 364, 8), set_exe_63957, *[str_63958], **kwargs_63959)
        
        
        # Call to set_exe(...): (line 365)
        # Processing the call arguments (line 365)
        str_63962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, 16), 'str', 'ranlib')
        # Processing the call keyword arguments (line 365)
        kwargs_63963 = {}
        # Getting the type of 'set_exe' (line 365)
        set_exe_63961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'set_exe', False)
        # Calling set_exe(args, kwargs) (line 365)
        set_exe_call_result_63964 = invoke(stypy.reporting.localization.Localization(__file__, 365, 8), set_exe_63961, *[str_63962], **kwargs_63963)
        
        
        # ################# End of 'find_executables(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_executables' in the type store
        # Getting the type of 'stypy_return_type' (line 288)
        stypy_return_type_63965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63965)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_executables'
        return stypy_return_type_63965


    @norecursion
    def update_executables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'update_executables'
        module_type_store = module_type_store.open_function_context('update_executables', 367, 4, False)
        # Assigning a type to the variable 'self' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.update_executables.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.update_executables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.update_executables.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.update_executables.__dict__.__setitem__('stypy_function_name', 'FCompiler.update_executables')
        FCompiler.update_executables.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.update_executables.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.update_executables.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.update_executables.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.update_executables.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.update_executables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.update_executables.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.update_executables', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'update_executables', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'update_executables(...)' code ##################

        str_63966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, (-1)), 'str', 'Called at the beginning of customisation. Subclasses should\n        override this if they need to set up the executables dictionary.\n\n        Note that self.find_executables() is run afterwards, so the\n        self.executables dictionary values can contain <F77> or <F90> as\n        the command, which will be replaced by the found F77 or F90\n        compiler.\n        ')
        pass
        
        # ################# End of 'update_executables(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'update_executables' in the type store
        # Getting the type of 'stypy_return_type' (line 367)
        stypy_return_type_63967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63967)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'update_executables'
        return stypy_return_type_63967


    @norecursion
    def get_flags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags'
        module_type_store = module_type_store.open_function_context('get_flags', 378, 4, False)
        # Assigning a type to the variable 'self' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.get_flags.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.get_flags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.get_flags.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.get_flags.__dict__.__setitem__('stypy_function_name', 'FCompiler.get_flags')
        FCompiler.get_flags.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.get_flags.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.get_flags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.get_flags.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.get_flags.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.get_flags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.get_flags.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.get_flags', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags(...)' code ##################

        str_63968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 8), 'str', 'List of flags common to all compiler types.')
        
        # Obtaining an instance of the builtin type 'list' (line 380)
        list_63969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 380)
        
        # Getting the type of 'self' (line 380)
        self_63970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 20), 'self')
        # Obtaining the member 'pic_flags' of a type (line 380)
        pic_flags_63971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 20), self_63970, 'pic_flags')
        # Applying the binary operator '+' (line 380)
        result_add_63972 = python_operator(stypy.reporting.localization.Localization(__file__, 380, 15), '+', list_63969, pic_flags_63971)
        
        # Assigning a type to the variable 'stypy_return_type' (line 380)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'stypy_return_type', result_add_63972)
        
        # ################# End of 'get_flags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags' in the type store
        # Getting the type of 'stypy_return_type' (line 378)
        stypy_return_type_63973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63973)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags'
        return stypy_return_type_63973


    @norecursion
    def _get_command_flags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_get_command_flags'
        module_type_store = module_type_store.open_function_context('_get_command_flags', 382, 4, False)
        # Assigning a type to the variable 'self' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler._get_command_flags.__dict__.__setitem__('stypy_localization', localization)
        FCompiler._get_command_flags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler._get_command_flags.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler._get_command_flags.__dict__.__setitem__('stypy_function_name', 'FCompiler._get_command_flags')
        FCompiler._get_command_flags.__dict__.__setitem__('stypy_param_names_list', ['key'])
        FCompiler._get_command_flags.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler._get_command_flags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler._get_command_flags.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler._get_command_flags.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler._get_command_flags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler._get_command_flags.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler._get_command_flags', ['key'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_get_command_flags', localization, ['key'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_get_command_flags(...)' code ##################

        
        # Assigning a Call to a Name (line 383):
        
        # Assigning a Call to a Name (line 383):
        
        # Call to get(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 'key' (line 383)
        key_63977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 35), 'key', False)
        # Getting the type of 'None' (line 383)
        None_63978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 40), 'None', False)
        # Processing the call keyword arguments (line 383)
        kwargs_63979 = {}
        # Getting the type of 'self' (line 383)
        self_63974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 14), 'self', False)
        # Obtaining the member 'executables' of a type (line 383)
        executables_63975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 14), self_63974, 'executables')
        # Obtaining the member 'get' of a type (line 383)
        get_63976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 14), executables_63975, 'get')
        # Calling get(args, kwargs) (line 383)
        get_call_result_63980 = invoke(stypy.reporting.localization.Localization(__file__, 383, 14), get_63976, *[key_63977, None_63978], **kwargs_63979)
        
        # Assigning a type to the variable 'cmd' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'cmd', get_call_result_63980)
        
        # Type idiom detected: calculating its left and rigth part (line 384)
        # Getting the type of 'cmd' (line 384)
        cmd_63981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 11), 'cmd')
        # Getting the type of 'None' (line 384)
        None_63982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 18), 'None')
        
        (may_be_63983, more_types_in_union_63984) = may_be_none(cmd_63981, None_63982)

        if may_be_63983:

            if more_types_in_union_63984:
                # Runtime conditional SSA (line 384)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Obtaining an instance of the builtin type 'list' (line 385)
            list_63985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 19), 'list')
            # Adding type elements to the builtin type 'list' instance (line 385)
            
            # Assigning a type to the variable 'stypy_return_type' (line 385)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 12), 'stypy_return_type', list_63985)

            if more_types_in_union_63984:
                # SSA join for if statement (line 384)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Obtaining the type of the subscript
        int_63986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 19), 'int')
        slice_63987 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 386, 15), int_63986, None, None)
        # Getting the type of 'cmd' (line 386)
        cmd_63988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 15), 'cmd')
        # Obtaining the member '__getitem__' of a type (line 386)
        getitem___63989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 15), cmd_63988, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 386)
        subscript_call_result_63990 = invoke(stypy.reporting.localization.Localization(__file__, 386, 15), getitem___63989, slice_63987)
        
        # Assigning a type to the variable 'stypy_return_type' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 8), 'stypy_return_type', subscript_call_result_63990)
        
        # ################# End of '_get_command_flags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_get_command_flags' in the type store
        # Getting the type of 'stypy_return_type' (line 382)
        stypy_return_type_63991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63991)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_get_command_flags'
        return stypy_return_type_63991


    @norecursion
    def get_flags_f77(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_f77'
        module_type_store = module_type_store.open_function_context('get_flags_f77', 388, 4, False)
        # Assigning a type to the variable 'self' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.get_flags_f77.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.get_flags_f77.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.get_flags_f77.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.get_flags_f77.__dict__.__setitem__('stypy_function_name', 'FCompiler.get_flags_f77')
        FCompiler.get_flags_f77.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.get_flags_f77.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.get_flags_f77.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.get_flags_f77.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.get_flags_f77.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.get_flags_f77.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.get_flags_f77.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.get_flags_f77', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_f77', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_f77(...)' code ##################

        str_63992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 8), 'str', 'List of Fortran 77 specific flags.')
        
        # Call to _get_command_flags(...): (line 390)
        # Processing the call arguments (line 390)
        str_63995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 39), 'str', 'compiler_f77')
        # Processing the call keyword arguments (line 390)
        kwargs_63996 = {}
        # Getting the type of 'self' (line 390)
        self_63993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 15), 'self', False)
        # Obtaining the member '_get_command_flags' of a type (line 390)
        _get_command_flags_63994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 15), self_63993, '_get_command_flags')
        # Calling _get_command_flags(args, kwargs) (line 390)
        _get_command_flags_call_result_63997 = invoke(stypy.reporting.localization.Localization(__file__, 390, 15), _get_command_flags_63994, *[str_63995], **kwargs_63996)
        
        # Assigning a type to the variable 'stypy_return_type' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'stypy_return_type', _get_command_flags_call_result_63997)
        
        # ################# End of 'get_flags_f77(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_f77' in the type store
        # Getting the type of 'stypy_return_type' (line 388)
        stypy_return_type_63998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63998)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_f77'
        return stypy_return_type_63998


    @norecursion
    def get_flags_f90(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_f90'
        module_type_store = module_type_store.open_function_context('get_flags_f90', 391, 4, False)
        # Assigning a type to the variable 'self' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.get_flags_f90.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.get_flags_f90.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.get_flags_f90.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.get_flags_f90.__dict__.__setitem__('stypy_function_name', 'FCompiler.get_flags_f90')
        FCompiler.get_flags_f90.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.get_flags_f90.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.get_flags_f90.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.get_flags_f90.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.get_flags_f90.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.get_flags_f90.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.get_flags_f90.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.get_flags_f90', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_f90', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_f90(...)' code ##################

        str_63999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 8), 'str', 'List of Fortran 90 specific flags.')
        
        # Call to _get_command_flags(...): (line 393)
        # Processing the call arguments (line 393)
        str_64002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 39), 'str', 'compiler_f90')
        # Processing the call keyword arguments (line 393)
        kwargs_64003 = {}
        # Getting the type of 'self' (line 393)
        self_64000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 15), 'self', False)
        # Obtaining the member '_get_command_flags' of a type (line 393)
        _get_command_flags_64001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 15), self_64000, '_get_command_flags')
        # Calling _get_command_flags(args, kwargs) (line 393)
        _get_command_flags_call_result_64004 = invoke(stypy.reporting.localization.Localization(__file__, 393, 15), _get_command_flags_64001, *[str_64002], **kwargs_64003)
        
        # Assigning a type to the variable 'stypy_return_type' (line 393)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'stypy_return_type', _get_command_flags_call_result_64004)
        
        # ################# End of 'get_flags_f90(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_f90' in the type store
        # Getting the type of 'stypy_return_type' (line 391)
        stypy_return_type_64005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64005)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_f90'
        return stypy_return_type_64005


    @norecursion
    def get_flags_free(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_free'
        module_type_store = module_type_store.open_function_context('get_flags_free', 394, 4, False)
        # Assigning a type to the variable 'self' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.get_flags_free.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.get_flags_free.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.get_flags_free.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.get_flags_free.__dict__.__setitem__('stypy_function_name', 'FCompiler.get_flags_free')
        FCompiler.get_flags_free.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.get_flags_free.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.get_flags_free.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.get_flags_free.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.get_flags_free.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.get_flags_free.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.get_flags_free.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.get_flags_free', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_free', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_free(...)' code ##################

        str_64006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 8), 'str', 'List of Fortran 90 free format specific flags.')
        
        # Obtaining an instance of the builtin type 'list' (line 396)
        list_64007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 396)
        
        # Assigning a type to the variable 'stypy_return_type' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'stypy_return_type', list_64007)
        
        # ################# End of 'get_flags_free(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_free' in the type store
        # Getting the type of 'stypy_return_type' (line 394)
        stypy_return_type_64008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64008)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_free'
        return stypy_return_type_64008


    @norecursion
    def get_flags_fix(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_fix'
        module_type_store = module_type_store.open_function_context('get_flags_fix', 397, 4, False)
        # Assigning a type to the variable 'self' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.get_flags_fix.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.get_flags_fix.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.get_flags_fix.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.get_flags_fix.__dict__.__setitem__('stypy_function_name', 'FCompiler.get_flags_fix')
        FCompiler.get_flags_fix.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.get_flags_fix.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.get_flags_fix.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.get_flags_fix.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.get_flags_fix.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.get_flags_fix.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.get_flags_fix.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.get_flags_fix', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_fix', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_fix(...)' code ##################

        str_64009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 8), 'str', 'List of Fortran 90 fixed format specific flags.')
        
        # Call to _get_command_flags(...): (line 399)
        # Processing the call arguments (line 399)
        str_64012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 39), 'str', 'compiler_fix')
        # Processing the call keyword arguments (line 399)
        kwargs_64013 = {}
        # Getting the type of 'self' (line 399)
        self_64010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 15), 'self', False)
        # Obtaining the member '_get_command_flags' of a type (line 399)
        _get_command_flags_64011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 15), self_64010, '_get_command_flags')
        # Calling _get_command_flags(args, kwargs) (line 399)
        _get_command_flags_call_result_64014 = invoke(stypy.reporting.localization.Localization(__file__, 399, 15), _get_command_flags_64011, *[str_64012], **kwargs_64013)
        
        # Assigning a type to the variable 'stypy_return_type' (line 399)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'stypy_return_type', _get_command_flags_call_result_64014)
        
        # ################# End of 'get_flags_fix(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_fix' in the type store
        # Getting the type of 'stypy_return_type' (line 397)
        stypy_return_type_64015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64015)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_fix'
        return stypy_return_type_64015


    @norecursion
    def get_flags_linker_so(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_linker_so'
        module_type_store = module_type_store.open_function_context('get_flags_linker_so', 400, 4, False)
        # Assigning a type to the variable 'self' (line 401)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_function_name', 'FCompiler.get_flags_linker_so')
        FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.get_flags_linker_so', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_linker_so', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_linker_so(...)' code ##################

        str_64016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 8), 'str', 'List of linker flags to build a shared library.')
        
        # Call to _get_command_flags(...): (line 402)
        # Processing the call arguments (line 402)
        str_64019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 39), 'str', 'linker_so')
        # Processing the call keyword arguments (line 402)
        kwargs_64020 = {}
        # Getting the type of 'self' (line 402)
        self_64017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 15), 'self', False)
        # Obtaining the member '_get_command_flags' of a type (line 402)
        _get_command_flags_64018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 15), self_64017, '_get_command_flags')
        # Calling _get_command_flags(args, kwargs) (line 402)
        _get_command_flags_call_result_64021 = invoke(stypy.reporting.localization.Localization(__file__, 402, 15), _get_command_flags_64018, *[str_64019], **kwargs_64020)
        
        # Assigning a type to the variable 'stypy_return_type' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'stypy_return_type', _get_command_flags_call_result_64021)
        
        # ################# End of 'get_flags_linker_so(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_linker_so' in the type store
        # Getting the type of 'stypy_return_type' (line 400)
        stypy_return_type_64022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64022)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_linker_so'
        return stypy_return_type_64022


    @norecursion
    def get_flags_linker_exe(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_linker_exe'
        module_type_store = module_type_store.open_function_context('get_flags_linker_exe', 403, 4, False)
        # Assigning a type to the variable 'self' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.get_flags_linker_exe.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.get_flags_linker_exe.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.get_flags_linker_exe.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.get_flags_linker_exe.__dict__.__setitem__('stypy_function_name', 'FCompiler.get_flags_linker_exe')
        FCompiler.get_flags_linker_exe.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.get_flags_linker_exe.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.get_flags_linker_exe.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.get_flags_linker_exe.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.get_flags_linker_exe.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.get_flags_linker_exe.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.get_flags_linker_exe.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.get_flags_linker_exe', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_linker_exe', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_linker_exe(...)' code ##################

        str_64023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 8), 'str', 'List of linker flags to build an executable.')
        
        # Call to _get_command_flags(...): (line 405)
        # Processing the call arguments (line 405)
        str_64026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 39), 'str', 'linker_exe')
        # Processing the call keyword arguments (line 405)
        kwargs_64027 = {}
        # Getting the type of 'self' (line 405)
        self_64024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 15), 'self', False)
        # Obtaining the member '_get_command_flags' of a type (line 405)
        _get_command_flags_64025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 15), self_64024, '_get_command_flags')
        # Calling _get_command_flags(args, kwargs) (line 405)
        _get_command_flags_call_result_64028 = invoke(stypy.reporting.localization.Localization(__file__, 405, 15), _get_command_flags_64025, *[str_64026], **kwargs_64027)
        
        # Assigning a type to the variable 'stypy_return_type' (line 405)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'stypy_return_type', _get_command_flags_call_result_64028)
        
        # ################# End of 'get_flags_linker_exe(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_linker_exe' in the type store
        # Getting the type of 'stypy_return_type' (line 403)
        stypy_return_type_64029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64029)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_linker_exe'
        return stypy_return_type_64029


    @norecursion
    def get_flags_ar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_ar'
        module_type_store = module_type_store.open_function_context('get_flags_ar', 406, 4, False)
        # Assigning a type to the variable 'self' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.get_flags_ar.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.get_flags_ar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.get_flags_ar.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.get_flags_ar.__dict__.__setitem__('stypy_function_name', 'FCompiler.get_flags_ar')
        FCompiler.get_flags_ar.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.get_flags_ar.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.get_flags_ar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.get_flags_ar.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.get_flags_ar.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.get_flags_ar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.get_flags_ar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.get_flags_ar', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_ar', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_ar(...)' code ##################

        str_64030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 8), 'str', 'List of archiver flags. ')
        
        # Call to _get_command_flags(...): (line 408)
        # Processing the call arguments (line 408)
        str_64033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 39), 'str', 'archiver')
        # Processing the call keyword arguments (line 408)
        kwargs_64034 = {}
        # Getting the type of 'self' (line 408)
        self_64031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 15), 'self', False)
        # Obtaining the member '_get_command_flags' of a type (line 408)
        _get_command_flags_64032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 408, 15), self_64031, '_get_command_flags')
        # Calling _get_command_flags(args, kwargs) (line 408)
        _get_command_flags_call_result_64035 = invoke(stypy.reporting.localization.Localization(__file__, 408, 15), _get_command_flags_64032, *[str_64033], **kwargs_64034)
        
        # Assigning a type to the variable 'stypy_return_type' (line 408)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 8), 'stypy_return_type', _get_command_flags_call_result_64035)
        
        # ################# End of 'get_flags_ar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_ar' in the type store
        # Getting the type of 'stypy_return_type' (line 406)
        stypy_return_type_64036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64036)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_ar'
        return stypy_return_type_64036


    @norecursion
    def get_flags_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_opt'
        module_type_store = module_type_store.open_function_context('get_flags_opt', 409, 4, False)
        # Assigning a type to the variable 'self' (line 410)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'FCompiler.get_flags_opt')
        FCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_opt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_opt(...)' code ##################

        str_64037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 8), 'str', 'List of architecture independent compiler flags.')
        
        # Obtaining an instance of the builtin type 'list' (line 411)
        list_64038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 411)
        
        # Assigning a type to the variable 'stypy_return_type' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'stypy_return_type', list_64038)
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 409)
        stypy_return_type_64039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64039)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_64039


    @norecursion
    def get_flags_arch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_arch'
        module_type_store = module_type_store.open_function_context('get_flags_arch', 412, 4, False)
        # Assigning a type to the variable 'self' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.get_flags_arch.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.get_flags_arch.__dict__.__setitem__('stypy_function_name', 'FCompiler.get_flags_arch')
        FCompiler.get_flags_arch.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.get_flags_arch.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.get_flags_arch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.get_flags_arch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.get_flags_arch', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_arch', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_arch(...)' code ##################

        str_64040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 8), 'str', 'List of architecture dependent compiler flags.')
        
        # Obtaining an instance of the builtin type 'list' (line 414)
        list_64041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 414)
        
        # Assigning a type to the variable 'stypy_return_type' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 8), 'stypy_return_type', list_64041)
        
        # ################# End of 'get_flags_arch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_arch' in the type store
        # Getting the type of 'stypy_return_type' (line 412)
        stypy_return_type_64042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64042)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_arch'
        return stypy_return_type_64042


    @norecursion
    def get_flags_debug(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_debug'
        module_type_store = module_type_store.open_function_context('get_flags_debug', 415, 4, False)
        # Assigning a type to the variable 'self' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.get_flags_debug.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.get_flags_debug.__dict__.__setitem__('stypy_function_name', 'FCompiler.get_flags_debug')
        FCompiler.get_flags_debug.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.get_flags_debug.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.get_flags_debug.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.get_flags_debug.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.get_flags_debug', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_debug', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_debug(...)' code ##################

        str_64043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 8), 'str', 'List of compiler flags to compile with debugging information.')
        
        # Obtaining an instance of the builtin type 'list' (line 417)
        list_64044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 417)
        
        # Assigning a type to the variable 'stypy_return_type' (line 417)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'stypy_return_type', list_64044)
        
        # ################# End of 'get_flags_debug(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_debug' in the type store
        # Getting the type of 'stypy_return_type' (line 415)
        stypy_return_type_64045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64045)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_debug'
        return stypy_return_type_64045

    
    # Multiple assignment of 2 elements.
    
    # Multiple assignment of 2 elements.
    
    # Multiple assignment of 2 elements.

    @norecursion
    def get_libraries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_libraries'
        module_type_store = module_type_store.open_function_context('get_libraries', 423, 4, False)
        # Assigning a type to the variable 'self' (line 424)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.get_libraries.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.get_libraries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.get_libraries.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.get_libraries.__dict__.__setitem__('stypy_function_name', 'FCompiler.get_libraries')
        FCompiler.get_libraries.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.get_libraries.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.get_libraries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.get_libraries.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.get_libraries.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.get_libraries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.get_libraries.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.get_libraries', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_libraries', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_libraries(...)' code ##################

        str_64046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 8), 'str', 'List of compiler libraries.')
        
        # Obtaining the type of the subscript
        slice_64047 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 425, 15), None, None, None)
        # Getting the type of 'self' (line 425)
        self_64048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 15), 'self')
        # Obtaining the member 'libraries' of a type (line 425)
        libraries_64049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 15), self_64048, 'libraries')
        # Obtaining the member '__getitem__' of a type (line 425)
        getitem___64050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 15), libraries_64049, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 425)
        subscript_call_result_64051 = invoke(stypy.reporting.localization.Localization(__file__, 425, 15), getitem___64050, slice_64047)
        
        # Assigning a type to the variable 'stypy_return_type' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'stypy_return_type', subscript_call_result_64051)
        
        # ################# End of 'get_libraries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_libraries' in the type store
        # Getting the type of 'stypy_return_type' (line 423)
        stypy_return_type_64052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64052)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_libraries'
        return stypy_return_type_64052


    @norecursion
    def get_library_dirs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_library_dirs'
        module_type_store = module_type_store.open_function_context('get_library_dirs', 426, 4, False)
        # Assigning a type to the variable 'self' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.get_library_dirs.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.get_library_dirs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.get_library_dirs.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.get_library_dirs.__dict__.__setitem__('stypy_function_name', 'FCompiler.get_library_dirs')
        FCompiler.get_library_dirs.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.get_library_dirs.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.get_library_dirs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.get_library_dirs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.get_library_dirs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_library_dirs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_library_dirs(...)' code ##################

        str_64053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 8), 'str', 'List of compiler library directories.')
        
        # Obtaining the type of the subscript
        slice_64054 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 428, 15), None, None, None)
        # Getting the type of 'self' (line 428)
        self_64055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 15), 'self')
        # Obtaining the member 'library_dirs' of a type (line 428)
        library_dirs_64056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 15), self_64055, 'library_dirs')
        # Obtaining the member '__getitem__' of a type (line 428)
        getitem___64057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 15), library_dirs_64056, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 428)
        subscript_call_result_64058 = invoke(stypy.reporting.localization.Localization(__file__, 428, 15), getitem___64057, slice_64054)
        
        # Assigning a type to the variable 'stypy_return_type' (line 428)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'stypy_return_type', subscript_call_result_64058)
        
        # ################# End of 'get_library_dirs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_library_dirs' in the type store
        # Getting the type of 'stypy_return_type' (line 426)
        stypy_return_type_64059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64059)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_library_dirs'
        return stypy_return_type_64059


    @norecursion
    def get_version(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 430)
        False_64060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 32), 'False')
        
        # Obtaining an instance of the builtin type 'list' (line 430)
        list_64061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 430)
        # Adding element type (line 430)
        int_64062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 49), list_64061, int_64062)
        
        defaults = [False_64060, list_64061]
        # Create a new context for function 'get_version'
        module_type_store = module_type_store.open_function_context('get_version', 430, 4, False)
        # Assigning a type to the variable 'self' (line 431)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.get_version.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.get_version.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.get_version.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.get_version.__dict__.__setitem__('stypy_function_name', 'FCompiler.get_version')
        FCompiler.get_version.__dict__.__setitem__('stypy_param_names_list', ['force', 'ok_status'])
        FCompiler.get_version.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.get_version.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.get_version.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.get_version.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.get_version.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.get_version.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.get_version', ['force', 'ok_status'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_version', localization, ['force', 'ok_status'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_version(...)' code ##################

        # Evaluating assert statement condition
        # Getting the type of 'self' (line 431)
        self_64063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 15), 'self')
        # Obtaining the member '_is_customised' of a type (line 431)
        _is_customised_64064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 15), self_64063, '_is_customised')
        
        # Assigning a Call to a Name (line 432):
        
        # Assigning a Call to a Name (line 432):
        
        # Call to get_version(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 'self' (line 432)
        self_64067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 40), 'self', False)
        # Processing the call keyword arguments (line 432)
        # Getting the type of 'force' (line 432)
        force_64068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 52), 'force', False)
        keyword_64069 = force_64068
        # Getting the type of 'ok_status' (line 432)
        ok_status_64070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 69), 'ok_status', False)
        keyword_64071 = ok_status_64070
        kwargs_64072 = {'ok_status': keyword_64071, 'force': keyword_64069}
        # Getting the type of 'CCompiler' (line 432)
        CCompiler_64065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 18), 'CCompiler', False)
        # Obtaining the member 'get_version' of a type (line 432)
        get_version_64066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 18), CCompiler_64065, 'get_version')
        # Calling get_version(args, kwargs) (line 432)
        get_version_call_result_64073 = invoke(stypy.reporting.localization.Localization(__file__, 432, 18), get_version_64066, *[self_64067], **kwargs_64072)
        
        # Assigning a type to the variable 'version' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'version', get_version_call_result_64073)
        
        # Type idiom detected: calculating its left and rigth part (line 433)
        # Getting the type of 'version' (line 433)
        version_64074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 11), 'version')
        # Getting the type of 'None' (line 433)
        None_64075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 22), 'None')
        
        (may_be_64076, more_types_in_union_64077) = may_be_none(version_64074, None_64075)

        if may_be_64076:

            if more_types_in_union_64077:
                # Runtime conditional SSA (line 433)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to CompilerNotFound(...): (line 434)
            # Processing the call keyword arguments (line 434)
            kwargs_64079 = {}
            # Getting the type of 'CompilerNotFound' (line 434)
            CompilerNotFound_64078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 18), 'CompilerNotFound', False)
            # Calling CompilerNotFound(args, kwargs) (line 434)
            CompilerNotFound_call_result_64080 = invoke(stypy.reporting.localization.Localization(__file__, 434, 18), CompilerNotFound_64078, *[], **kwargs_64079)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 434, 12), CompilerNotFound_call_result_64080, 'raise parameter', BaseException)

            if more_types_in_union_64077:
                # SSA join for if statement (line 433)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'version' (line 435)
        version_64081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 15), 'version')
        # Assigning a type to the variable 'stypy_return_type' (line 435)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'stypy_return_type', version_64081)
        
        # ################# End of 'get_version(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_version' in the type store
        # Getting the type of 'stypy_return_type' (line 430)
        stypy_return_type_64082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64082)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_version'
        return stypy_return_type_64082


    @norecursion
    def customize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 441)
        None_64083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 31), 'None')
        defaults = [None_64083]
        # Create a new context for function 'customize'
        module_type_store = module_type_store.open_function_context('customize', 441, 4, False)
        # Assigning a type to the variable 'self' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.customize.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.customize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.customize.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.customize.__dict__.__setitem__('stypy_function_name', 'FCompiler.customize')
        FCompiler.customize.__dict__.__setitem__('stypy_param_names_list', ['dist'])
        FCompiler.customize.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.customize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.customize.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.customize.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.customize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.customize.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.customize', ['dist'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'customize', localization, ['dist'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'customize(...)' code ##################

        str_64084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, (-1)), 'str', 'Customize Fortran compiler.\n\n        This method gets Fortran compiler specific information from\n        (i) class definition, (ii) environment, (iii) distutils config\n        files, and (iv) command line (later overrides earlier).\n\n        This method should be always called after constructing a\n        compiler instance. But not in __init__ because Distribution\n        instance is needed for (iii) and (iv).\n        ')
        
        # Call to info(...): (line 452)
        # Processing the call arguments (line 452)
        str_64087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 17), 'str', 'customize %s')
        # Getting the type of 'self' (line 452)
        self_64088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 35), 'self', False)
        # Obtaining the member '__class__' of a type (line 452)
        class___64089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 35), self_64088, '__class__')
        # Obtaining the member '__name__' of a type (line 452)
        name___64090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 35), class___64089, '__name__')
        # Applying the binary operator '%' (line 452)
        result_mod_64091 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 17), '%', str_64087, name___64090)
        
        # Processing the call keyword arguments (line 452)
        kwargs_64092 = {}
        # Getting the type of 'log' (line 452)
        log_64085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 452)
        info_64086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 8), log_64085, 'info')
        # Calling info(args, kwargs) (line 452)
        info_call_result_64093 = invoke(stypy.reporting.localization.Localization(__file__, 452, 8), info_64086, *[result_mod_64091], **kwargs_64092)
        
        
        # Assigning a Name to a Attribute (line 454):
        
        # Assigning a Name to a Attribute (line 454):
        # Getting the type of 'True' (line 454)
        True_64094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 30), 'True')
        # Getting the type of 'self' (line 454)
        self_64095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'self')
        # Setting the type of the member '_is_customised' of a type (line 454)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), self_64095, '_is_customised', True_64094)
        
        # Call to use_distribution(...): (line 456)
        # Processing the call arguments (line 456)
        # Getting the type of 'dist' (line 456)
        dist_64099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 45), 'dist', False)
        # Processing the call keyword arguments (line 456)
        kwargs_64100 = {}
        # Getting the type of 'self' (line 456)
        self_64096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'self', False)
        # Obtaining the member 'distutils_vars' of a type (line 456)
        distutils_vars_64097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 8), self_64096, 'distutils_vars')
        # Obtaining the member 'use_distribution' of a type (line 456)
        use_distribution_64098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 8), distutils_vars_64097, 'use_distribution')
        # Calling use_distribution(args, kwargs) (line 456)
        use_distribution_call_result_64101 = invoke(stypy.reporting.localization.Localization(__file__, 456, 8), use_distribution_64098, *[dist_64099], **kwargs_64100)
        
        
        # Call to use_distribution(...): (line 457)
        # Processing the call arguments (line 457)
        # Getting the type of 'dist' (line 457)
        dist_64105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 43), 'dist', False)
        # Processing the call keyword arguments (line 457)
        kwargs_64106 = {}
        # Getting the type of 'self' (line 457)
        self_64102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'self', False)
        # Obtaining the member 'command_vars' of a type (line 457)
        command_vars_64103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 8), self_64102, 'command_vars')
        # Obtaining the member 'use_distribution' of a type (line 457)
        use_distribution_64104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 8), command_vars_64103, 'use_distribution')
        # Calling use_distribution(args, kwargs) (line 457)
        use_distribution_call_result_64107 = invoke(stypy.reporting.localization.Localization(__file__, 457, 8), use_distribution_64104, *[dist_64105], **kwargs_64106)
        
        
        # Call to use_distribution(...): (line 458)
        # Processing the call arguments (line 458)
        # Getting the type of 'dist' (line 458)
        dist_64111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 40), 'dist', False)
        # Processing the call keyword arguments (line 458)
        kwargs_64112 = {}
        # Getting the type of 'self' (line 458)
        self_64108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'self', False)
        # Obtaining the member 'flag_vars' of a type (line 458)
        flag_vars_64109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), self_64108, 'flag_vars')
        # Obtaining the member 'use_distribution' of a type (line 458)
        use_distribution_64110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 8), flag_vars_64109, 'use_distribution')
        # Calling use_distribution(args, kwargs) (line 458)
        use_distribution_call_result_64113 = invoke(stypy.reporting.localization.Localization(__file__, 458, 8), use_distribution_64110, *[dist_64111], **kwargs_64112)
        
        
        # Call to update_executables(...): (line 460)
        # Processing the call keyword arguments (line 460)
        kwargs_64116 = {}
        # Getting the type of 'self' (line 460)
        self_64114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 8), 'self', False)
        # Obtaining the member 'update_executables' of a type (line 460)
        update_executables_64115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 8), self_64114, 'update_executables')
        # Calling update_executables(args, kwargs) (line 460)
        update_executables_call_result_64117 = invoke(stypy.reporting.localization.Localization(__file__, 460, 8), update_executables_64115, *[], **kwargs_64116)
        
        
        # Call to find_executables(...): (line 464)
        # Processing the call keyword arguments (line 464)
        kwargs_64120 = {}
        # Getting the type of 'self' (line 464)
        self_64118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'self', False)
        # Obtaining the member 'find_executables' of a type (line 464)
        find_executables_64119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 8), self_64118, 'find_executables')
        # Calling find_executables(args, kwargs) (line 464)
        find_executables_call_result_64121 = invoke(stypy.reporting.localization.Localization(__file__, 464, 8), find_executables_64119, *[], **kwargs_64120)
        
        
        # Assigning a Call to a Name (line 466):
        
        # Assigning a Call to a Name (line 466):
        
        # Call to get(...): (line 466)
        # Processing the call arguments (line 466)
        str_64125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 40), 'str', 'noopt')
        # Getting the type of 'False' (line 466)
        False_64126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 49), 'False', False)
        # Processing the call keyword arguments (line 466)
        kwargs_64127 = {}
        # Getting the type of 'self' (line 466)
        self_64122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 16), 'self', False)
        # Obtaining the member 'distutils_vars' of a type (line 466)
        distutils_vars_64123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 16), self_64122, 'distutils_vars')
        # Obtaining the member 'get' of a type (line 466)
        get_64124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 16), distutils_vars_64123, 'get')
        # Calling get(args, kwargs) (line 466)
        get_call_result_64128 = invoke(stypy.reporting.localization.Localization(__file__, 466, 16), get_64124, *[str_64125, False_64126], **kwargs_64127)
        
        # Assigning a type to the variable 'noopt' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'noopt', get_call_result_64128)
        
        # Assigning a Call to a Name (line 467):
        
        # Assigning a Call to a Name (line 467):
        
        # Call to get(...): (line 467)
        # Processing the call arguments (line 467)
        str_64132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 41), 'str', 'noarch')
        # Getting the type of 'noopt' (line 467)
        noopt_64133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 51), 'noopt', False)
        # Processing the call keyword arguments (line 467)
        kwargs_64134 = {}
        # Getting the type of 'self' (line 467)
        self_64129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 17), 'self', False)
        # Obtaining the member 'distutils_vars' of a type (line 467)
        distutils_vars_64130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 17), self_64129, 'distutils_vars')
        # Obtaining the member 'get' of a type (line 467)
        get_64131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 17), distutils_vars_64130, 'get')
        # Calling get(args, kwargs) (line 467)
        get_call_result_64135 = invoke(stypy.reporting.localization.Localization(__file__, 467, 17), get_64131, *[str_64132, noopt_64133], **kwargs_64134)
        
        # Assigning a type to the variable 'noarch' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'noarch', get_call_result_64135)
        
        # Assigning a Call to a Name (line 468):
        
        # Assigning a Call to a Name (line 468):
        
        # Call to get(...): (line 468)
        # Processing the call arguments (line 468)
        str_64139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 40), 'str', 'debug')
        # Getting the type of 'False' (line 468)
        False_64140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 49), 'False', False)
        # Processing the call keyword arguments (line 468)
        kwargs_64141 = {}
        # Getting the type of 'self' (line 468)
        self_64136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 16), 'self', False)
        # Obtaining the member 'distutils_vars' of a type (line 468)
        distutils_vars_64137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 16), self_64136, 'distutils_vars')
        # Obtaining the member 'get' of a type (line 468)
        get_64138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 16), distutils_vars_64137, 'get')
        # Calling get(args, kwargs) (line 468)
        get_call_result_64142 = invoke(stypy.reporting.localization.Localization(__file__, 468, 16), get_64138, *[str_64139, False_64140], **kwargs_64141)
        
        # Assigning a type to the variable 'debug' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'debug', get_call_result_64142)
        
        # Assigning a Attribute to a Name (line 470):
        
        # Assigning a Attribute to a Name (line 470):
        # Getting the type of 'self' (line 470)
        self_64143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 14), 'self')
        # Obtaining the member 'command_vars' of a type (line 470)
        command_vars_64144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 14), self_64143, 'command_vars')
        # Obtaining the member 'compiler_f77' of a type (line 470)
        compiler_f77_64145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 14), command_vars_64144, 'compiler_f77')
        # Assigning a type to the variable 'f77' (line 470)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 8), 'f77', compiler_f77_64145)
        
        # Assigning a Attribute to a Name (line 471):
        
        # Assigning a Attribute to a Name (line 471):
        # Getting the type of 'self' (line 471)
        self_64146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 14), 'self')
        # Obtaining the member 'command_vars' of a type (line 471)
        command_vars_64147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 14), self_64146, 'command_vars')
        # Obtaining the member 'compiler_f90' of a type (line 471)
        compiler_f90_64148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 14), command_vars_64147, 'compiler_f90')
        # Assigning a type to the variable 'f90' (line 471)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 471, 8), 'f90', compiler_f90_64148)
        
        # Assigning a List to a Name (line 473):
        
        # Assigning a List to a Name (line 473):
        
        # Obtaining an instance of the builtin type 'list' (line 473)
        list_64149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 473)
        
        # Assigning a type to the variable 'f77flags' (line 473)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 473, 8), 'f77flags', list_64149)
        
        # Assigning a List to a Name (line 474):
        
        # Assigning a List to a Name (line 474):
        
        # Obtaining an instance of the builtin type 'list' (line 474)
        list_64150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 474)
        
        # Assigning a type to the variable 'f90flags' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'f90flags', list_64150)
        
        # Assigning a List to a Name (line 475):
        
        # Assigning a List to a Name (line 475):
        
        # Obtaining an instance of the builtin type 'list' (line 475)
        list_64151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 475)
        
        # Assigning a type to the variable 'freeflags' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'freeflags', list_64151)
        
        # Assigning a List to a Name (line 476):
        
        # Assigning a List to a Name (line 476):
        
        # Obtaining an instance of the builtin type 'list' (line 476)
        list_64152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 476)
        
        # Assigning a type to the variable 'fixflags' (line 476)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 8), 'fixflags', list_64152)
        
        # Getting the type of 'f77' (line 478)
        f77_64153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 11), 'f77')
        # Testing the type of an if condition (line 478)
        if_condition_64154 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 478, 8), f77_64153)
        # Assigning a type to the variable 'if_condition_64154' (line 478)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'if_condition_64154', if_condition_64154)
        # SSA begins for if statement (line 478)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 479):
        
        # Assigning a Attribute to a Name (line 479):
        # Getting the type of 'self' (line 479)
        self_64155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 23), 'self')
        # Obtaining the member 'flag_vars' of a type (line 479)
        flag_vars_64156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 23), self_64155, 'flag_vars')
        # Obtaining the member 'f77' of a type (line 479)
        f77_64157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 23), flag_vars_64156, 'f77')
        # Assigning a type to the variable 'f77flags' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'f77flags', f77_64157)
        # SSA join for if statement (line 478)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'f90' (line 480)
        f90_64158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 11), 'f90')
        # Testing the type of an if condition (line 480)
        if_condition_64159 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 480, 8), f90_64158)
        # Assigning a type to the variable 'if_condition_64159' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'if_condition_64159', if_condition_64159)
        # SSA begins for if statement (line 480)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 481):
        
        # Assigning a Attribute to a Name (line 481):
        # Getting the type of 'self' (line 481)
        self_64160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 23), 'self')
        # Obtaining the member 'flag_vars' of a type (line 481)
        flag_vars_64161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 23), self_64160, 'flag_vars')
        # Obtaining the member 'f90' of a type (line 481)
        f90_64162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 23), flag_vars_64161, 'f90')
        # Assigning a type to the variable 'f90flags' (line 481)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'f90flags', f90_64162)
        
        # Assigning a Attribute to a Name (line 482):
        
        # Assigning a Attribute to a Name (line 482):
        # Getting the type of 'self' (line 482)
        self_64163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 24), 'self')
        # Obtaining the member 'flag_vars' of a type (line 482)
        flag_vars_64164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 24), self_64163, 'flag_vars')
        # Obtaining the member 'free' of a type (line 482)
        free_64165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 24), flag_vars_64164, 'free')
        # Assigning a type to the variable 'freeflags' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'freeflags', free_64165)
        # SSA join for if statement (line 480)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 484):
        
        # Assigning a Attribute to a Name (line 484):
        # Getting the type of 'self' (line 484)
        self_64166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 14), 'self')
        # Obtaining the member 'command_vars' of a type (line 484)
        command_vars_64167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 14), self_64166, 'command_vars')
        # Obtaining the member 'compiler_fix' of a type (line 484)
        compiler_fix_64168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 484, 14), command_vars_64167, 'compiler_fix')
        # Assigning a type to the variable 'fix' (line 484)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'fix', compiler_fix_64168)
        
        # Getting the type of 'fix' (line 485)
        fix_64169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 11), 'fix')
        # Testing the type of an if condition (line 485)
        if_condition_64170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 485, 8), fix_64169)
        # Assigning a type to the variable 'if_condition_64170' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'if_condition_64170', if_condition_64170)
        # SSA begins for if statement (line 485)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 486):
        
        # Assigning a BinOp to a Name (line 486):
        # Getting the type of 'self' (line 486)
        self_64171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 23), 'self')
        # Obtaining the member 'flag_vars' of a type (line 486)
        flag_vars_64172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 23), self_64171, 'flag_vars')
        # Obtaining the member 'fix' of a type (line 486)
        fix_64173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 23), flag_vars_64172, 'fix')
        # Getting the type of 'f90flags' (line 486)
        f90flags_64174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 44), 'f90flags')
        # Applying the binary operator '+' (line 486)
        result_add_64175 = python_operator(stypy.reporting.localization.Localization(__file__, 486, 23), '+', fix_64173, f90flags_64174)
        
        # Assigning a type to the variable 'fixflags' (line 486)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 486, 12), 'fixflags', result_add_64175)
        # SSA join for if statement (line 485)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Tuple (line 488):
        
        # Assigning a List to a Name (line 488):
        
        # Obtaining an instance of the builtin type 'list' (line 488)
        list_64176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 488)
        
        # Assigning a type to the variable 'tuple_assignment_63481' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'tuple_assignment_63481', list_64176)
        
        # Assigning a List to a Name (line 488):
        
        # Obtaining an instance of the builtin type 'list' (line 488)
        list_64177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 488)
        
        # Assigning a type to the variable 'tuple_assignment_63482' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'tuple_assignment_63482', list_64177)
        
        # Assigning a List to a Name (line 488):
        
        # Obtaining an instance of the builtin type 'list' (line 488)
        list_64178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 488, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 488)
        
        # Assigning a type to the variable 'tuple_assignment_63483' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'tuple_assignment_63483', list_64178)
        
        # Assigning a Name to a Name (line 488):
        # Getting the type of 'tuple_assignment_63481' (line 488)
        tuple_assignment_63481_64179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'tuple_assignment_63481')
        # Assigning a type to the variable 'oflags' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'oflags', tuple_assignment_63481_64179)
        
        # Assigning a Name to a Name (line 488):
        # Getting the type of 'tuple_assignment_63482' (line 488)
        tuple_assignment_63482_64180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'tuple_assignment_63482')
        # Assigning a type to the variable 'aflags' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 16), 'aflags', tuple_assignment_63482_64180)
        
        # Assigning a Name to a Name (line 488):
        # Getting the type of 'tuple_assignment_63483' (line 488)
        tuple_assignment_63483_64181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'tuple_assignment_63483')
        # Assigning a type to the variable 'dflags' (line 488)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 24), 'dflags', tuple_assignment_63483_64181)

        @norecursion
        def get_flags(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'get_flags'
            module_type_store = module_type_store.open_function_context('get_flags', 491, 8, False)
            
            # Passed parameters checking function
            get_flags.stypy_localization = localization
            get_flags.stypy_type_of_self = None
            get_flags.stypy_type_store = module_type_store
            get_flags.stypy_function_name = 'get_flags'
            get_flags.stypy_param_names_list = ['tag', 'flags']
            get_flags.stypy_varargs_param_name = None
            get_flags.stypy_kwargs_param_name = None
            get_flags.stypy_call_defaults = defaults
            get_flags.stypy_call_varargs = varargs
            get_flags.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'get_flags', ['tag', 'flags'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'get_flags', localization, ['tag', 'flags'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'get_flags(...)' code ##################

            
            # Call to extend(...): (line 493)
            # Processing the call arguments (line 493)
            
            # Call to getattr(...): (line 493)
            # Processing the call arguments (line 493)
            # Getting the type of 'self' (line 493)
            self_64185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 33), 'self', False)
            # Obtaining the member 'flag_vars' of a type (line 493)
            flag_vars_64186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 33), self_64185, 'flag_vars')
            # Getting the type of 'tag' (line 493)
            tag_64187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 49), 'tag', False)
            # Processing the call keyword arguments (line 493)
            kwargs_64188 = {}
            # Getting the type of 'getattr' (line 493)
            getattr_64184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 25), 'getattr', False)
            # Calling getattr(args, kwargs) (line 493)
            getattr_call_result_64189 = invoke(stypy.reporting.localization.Localization(__file__, 493, 25), getattr_64184, *[flag_vars_64186, tag_64187], **kwargs_64188)
            
            # Processing the call keyword arguments (line 493)
            kwargs_64190 = {}
            # Getting the type of 'flags' (line 493)
            flags_64182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), 'flags', False)
            # Obtaining the member 'extend' of a type (line 493)
            extend_64183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 12), flags_64182, 'extend')
            # Calling extend(args, kwargs) (line 493)
            extend_call_result_64191 = invoke(stypy.reporting.localization.Localization(__file__, 493, 12), extend_64183, *[getattr_call_result_64189], **kwargs_64190)
            
            
            # Assigning a Call to a Name (line 494):
            
            # Assigning a Call to a Name (line 494):
            
            # Call to getattr(...): (line 494)
            # Processing the call arguments (line 494)
            # Getting the type of 'self' (line 494)
            self_64193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 31), 'self', False)
            str_64194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 37), 'str', 'get_flags_')
            # Getting the type of 'tag' (line 494)
            tag_64195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 52), 'tag', False)
            # Applying the binary operator '+' (line 494)
            result_add_64196 = python_operator(stypy.reporting.localization.Localization(__file__, 494, 37), '+', str_64194, tag_64195)
            
            # Processing the call keyword arguments (line 494)
            kwargs_64197 = {}
            # Getting the type of 'getattr' (line 494)
            getattr_64192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 23), 'getattr', False)
            # Calling getattr(args, kwargs) (line 494)
            getattr_call_result_64198 = invoke(stypy.reporting.localization.Localization(__file__, 494, 23), getattr_64192, *[self_64193, result_add_64196], **kwargs_64197)
            
            # Assigning a type to the variable 'this_get' (line 494)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'this_get', getattr_call_result_64198)
            
            
            # Obtaining an instance of the builtin type 'list' (line 495)
            list_64199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 36), 'list')
            # Adding type elements to the builtin type 'list' instance (line 495)
            # Adding element type (line 495)
            
            # Obtaining an instance of the builtin type 'tuple' (line 495)
            tuple_64200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 38), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 495)
            # Adding element type (line 495)
            str_64201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 38), 'str', 'f77')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 38), tuple_64200, str_64201)
            # Adding element type (line 495)
            # Getting the type of 'f77' (line 495)
            f77_64202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 45), 'f77')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 38), tuple_64200, f77_64202)
            # Adding element type (line 495)
            # Getting the type of 'f77flags' (line 495)
            f77flags_64203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 50), 'f77flags')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 38), tuple_64200, f77flags_64203)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 36), list_64199, tuple_64200)
            # Adding element type (line 495)
            
            # Obtaining an instance of the builtin type 'tuple' (line 496)
            tuple_64204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 38), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 496)
            # Adding element type (line 496)
            str_64205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 38), 'str', 'f90')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 38), tuple_64204, str_64205)
            # Adding element type (line 496)
            # Getting the type of 'f90' (line 496)
            f90_64206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 45), 'f90')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 38), tuple_64204, f90_64206)
            # Adding element type (line 496)
            # Getting the type of 'f90flags' (line 496)
            f90flags_64207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 50), 'f90flags')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 38), tuple_64204, f90flags_64207)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 36), list_64199, tuple_64204)
            # Adding element type (line 495)
            
            # Obtaining an instance of the builtin type 'tuple' (line 497)
            tuple_64208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 38), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 497)
            # Adding element type (line 497)
            str_64209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 38), 'str', 'f90')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 38), tuple_64208, str_64209)
            # Adding element type (line 497)
            # Getting the type of 'fix' (line 497)
            fix_64210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 45), 'fix')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 38), tuple_64208, fix_64210)
            # Adding element type (line 497)
            # Getting the type of 'fixflags' (line 497)
            fixflags_64211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 50), 'fixflags')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 497, 38), tuple_64208, fixflags_64211)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 36), list_64199, tuple_64208)
            
            # Testing the type of a for loop iterable (line 495)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 495, 12), list_64199)
            # Getting the type of the for loop variable (line 495)
            for_loop_var_64212 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 495, 12), list_64199)
            # Assigning a type to the variable 'name' (line 495)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 12), for_loop_var_64212))
            # Assigning a type to the variable 'c' (line 495)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 12), for_loop_var_64212))
            # Assigning a type to the variable 'flagvar' (line 495)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'flagvar', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 12), for_loop_var_64212))
            # SSA begins for a for statement (line 495)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 498):
            
            # Assigning a BinOp to a Name (line 498):
            str_64213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 20), 'str', '%s_%s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 498)
            tuple_64214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 498)
            # Adding element type (line 498)
            # Getting the type of 'tag' (line 498)
            tag_64215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 31), 'tag')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 31), tuple_64214, tag_64215)
            # Adding element type (line 498)
            # Getting the type of 'name' (line 498)
            name_64216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 36), 'name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 498, 31), tuple_64214, name_64216)
            
            # Applying the binary operator '%' (line 498)
            result_mod_64217 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 20), '%', str_64213, tuple_64214)
            
            # Assigning a type to the variable 't' (line 498)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 16), 't', result_mod_64217)
            
            
            # Evaluating a boolean operation
            # Getting the type of 'c' (line 499)
            c_64218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 19), 'c')
            
            # Getting the type of 'this_get' (line 499)
            this_get_64219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 25), 'this_get')
            
            # Call to getattr(...): (line 499)
            # Processing the call arguments (line 499)
            # Getting the type of 'self' (line 499)
            self_64221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 49), 'self', False)
            str_64222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 55), 'str', 'get_flags_')
            # Getting the type of 't' (line 499)
            t_64223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 70), 't', False)
            # Applying the binary operator '+' (line 499)
            result_add_64224 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 55), '+', str_64222, t_64223)
            
            # Processing the call keyword arguments (line 499)
            kwargs_64225 = {}
            # Getting the type of 'getattr' (line 499)
            getattr_64220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 41), 'getattr', False)
            # Calling getattr(args, kwargs) (line 499)
            getattr_call_result_64226 = invoke(stypy.reporting.localization.Localization(__file__, 499, 41), getattr_64220, *[self_64221, result_add_64224], **kwargs_64225)
            
            # Applying the binary operator 'isnot' (line 499)
            result_is_not_64227 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 25), 'isnot', this_get_64219, getattr_call_result_64226)
            
            # Applying the binary operator 'and' (line 499)
            result_and_keyword_64228 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 19), 'and', c_64218, result_is_not_64227)
            
            # Testing the type of an if condition (line 499)
            if_condition_64229 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 499, 16), result_and_keyword_64228)
            # Assigning a type to the variable 'if_condition_64229' (line 499)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 16), 'if_condition_64229', if_condition_64229)
            # SSA begins for if statement (line 499)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to extend(...): (line 500)
            # Processing the call arguments (line 500)
            
            # Call to getattr(...): (line 500)
            # Processing the call arguments (line 500)
            # Getting the type of 'self' (line 500)
            self_64233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 43), 'self', False)
            # Obtaining the member 'flag_vars' of a type (line 500)
            flag_vars_64234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 43), self_64233, 'flag_vars')
            # Getting the type of 't' (line 500)
            t_64235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 59), 't', False)
            # Processing the call keyword arguments (line 500)
            kwargs_64236 = {}
            # Getting the type of 'getattr' (line 500)
            getattr_64232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 35), 'getattr', False)
            # Calling getattr(args, kwargs) (line 500)
            getattr_call_result_64237 = invoke(stypy.reporting.localization.Localization(__file__, 500, 35), getattr_64232, *[flag_vars_64234, t_64235], **kwargs_64236)
            
            # Processing the call keyword arguments (line 500)
            kwargs_64238 = {}
            # Getting the type of 'flagvar' (line 500)
            flagvar_64230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 20), 'flagvar', False)
            # Obtaining the member 'extend' of a type (line 500)
            extend_64231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 20), flagvar_64230, 'extend')
            # Calling extend(args, kwargs) (line 500)
            extend_call_result_64239 = invoke(stypy.reporting.localization.Localization(__file__, 500, 20), extend_64231, *[getattr_call_result_64237], **kwargs_64238)
            
            # SSA join for if statement (line 499)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'get_flags(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'get_flags' in the type store
            # Getting the type of 'stypy_return_type' (line 491)
            stypy_return_type_64240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_64240)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'get_flags'
            return stypy_return_type_64240

        # Assigning a type to the variable 'get_flags' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'get_flags', get_flags)
        
        
        # Getting the type of 'noopt' (line 501)
        noopt_64241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 15), 'noopt')
        # Applying the 'not' unary operator (line 501)
        result_not__64242 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 11), 'not', noopt_64241)
        
        # Testing the type of an if condition (line 501)
        if_condition_64243 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 501, 8), result_not__64242)
        # Assigning a type to the variable 'if_condition_64243' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'if_condition_64243', if_condition_64243)
        # SSA begins for if statement (line 501)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to get_flags(...): (line 502)
        # Processing the call arguments (line 502)
        str_64245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 22), 'str', 'opt')
        # Getting the type of 'oflags' (line 502)
        oflags_64246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 29), 'oflags', False)
        # Processing the call keyword arguments (line 502)
        kwargs_64247 = {}
        # Getting the type of 'get_flags' (line 502)
        get_flags_64244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 12), 'get_flags', False)
        # Calling get_flags(args, kwargs) (line 502)
        get_flags_call_result_64248 = invoke(stypy.reporting.localization.Localization(__file__, 502, 12), get_flags_64244, *[str_64245, oflags_64246], **kwargs_64247)
        
        
        
        # Getting the type of 'noarch' (line 503)
        noarch_64249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 19), 'noarch')
        # Applying the 'not' unary operator (line 503)
        result_not__64250 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 15), 'not', noarch_64249)
        
        # Testing the type of an if condition (line 503)
        if_condition_64251 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 503, 12), result_not__64250)
        # Assigning a type to the variable 'if_condition_64251' (line 503)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 503, 12), 'if_condition_64251', if_condition_64251)
        # SSA begins for if statement (line 503)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to get_flags(...): (line 504)
        # Processing the call arguments (line 504)
        str_64253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 26), 'str', 'arch')
        # Getting the type of 'aflags' (line 504)
        aflags_64254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 34), 'aflags', False)
        # Processing the call keyword arguments (line 504)
        kwargs_64255 = {}
        # Getting the type of 'get_flags' (line 504)
        get_flags_64252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 16), 'get_flags', False)
        # Calling get_flags(args, kwargs) (line 504)
        get_flags_call_result_64256 = invoke(stypy.reporting.localization.Localization(__file__, 504, 16), get_flags_64252, *[str_64253, aflags_64254], **kwargs_64255)
        
        # SSA join for if statement (line 503)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 501)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'debug' (line 505)
        debug_64257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 11), 'debug')
        # Testing the type of an if condition (line 505)
        if_condition_64258 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 505, 8), debug_64257)
        # Assigning a type to the variable 'if_condition_64258' (line 505)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'if_condition_64258', if_condition_64258)
        # SSA begins for if statement (line 505)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to get_flags(...): (line 506)
        # Processing the call arguments (line 506)
        str_64260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 22), 'str', 'debug')
        # Getting the type of 'dflags' (line 506)
        dflags_64261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 31), 'dflags', False)
        # Processing the call keyword arguments (line 506)
        kwargs_64262 = {}
        # Getting the type of 'get_flags' (line 506)
        get_flags_64259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 12), 'get_flags', False)
        # Calling get_flags(args, kwargs) (line 506)
        get_flags_call_result_64263 = invoke(stypy.reporting.localization.Localization(__file__, 506, 12), get_flags_64259, *[str_64260, dflags_64261], **kwargs_64262)
        
        # SSA join for if statement (line 505)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 508):
        
        # Assigning a BinOp to a Name (line 508):
        # Getting the type of 'self' (line 508)
        self_64264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 17), 'self')
        # Obtaining the member 'flag_vars' of a type (line 508)
        flag_vars_64265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 17), self_64264, 'flag_vars')
        # Obtaining the member 'flags' of a type (line 508)
        flags_64266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 17), flag_vars_64265, 'flags')
        # Getting the type of 'dflags' (line 508)
        dflags_64267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 40), 'dflags')
        # Applying the binary operator '+' (line 508)
        result_add_64268 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 17), '+', flags_64266, dflags_64267)
        
        # Getting the type of 'oflags' (line 508)
        oflags_64269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 49), 'oflags')
        # Applying the binary operator '+' (line 508)
        result_add_64270 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 47), '+', result_add_64268, oflags_64269)
        
        # Getting the type of 'aflags' (line 508)
        aflags_64271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 58), 'aflags')
        # Applying the binary operator '+' (line 508)
        result_add_64272 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 56), '+', result_add_64270, aflags_64271)
        
        # Assigning a type to the variable 'fflags' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 'fflags', result_add_64272)
        
        # Getting the type of 'f77' (line 510)
        f77_64273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 11), 'f77')
        # Testing the type of an if condition (line 510)
        if_condition_64274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 510, 8), f77_64273)
        # Assigning a type to the variable 'if_condition_64274' (line 510)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'if_condition_64274', if_condition_64274)
        # SSA begins for if statement (line 510)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_commands(...): (line 511)
        # Processing the call keyword arguments (line 511)
        
        # Obtaining an instance of the builtin type 'list' (line 511)
        list_64277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 511)
        # Adding element type (line 511)
        # Getting the type of 'f77' (line 511)
        f77_64278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 44), 'f77', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 511, 43), list_64277, f77_64278)
        
        # Getting the type of 'f77flags' (line 511)
        f77flags_64279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 49), 'f77flags', False)
        # Applying the binary operator '+' (line 511)
        result_add_64280 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 43), '+', list_64277, f77flags_64279)
        
        # Getting the type of 'fflags' (line 511)
        fflags_64281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 58), 'fflags', False)
        # Applying the binary operator '+' (line 511)
        result_add_64282 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 57), '+', result_add_64280, fflags_64281)
        
        keyword_64283 = result_add_64282
        kwargs_64284 = {'compiler_f77': keyword_64283}
        # Getting the type of 'self' (line 511)
        self_64275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 12), 'self', False)
        # Obtaining the member 'set_commands' of a type (line 511)
        set_commands_64276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 12), self_64275, 'set_commands')
        # Calling set_commands(args, kwargs) (line 511)
        set_commands_call_result_64285 = invoke(stypy.reporting.localization.Localization(__file__, 511, 12), set_commands_64276, *[], **kwargs_64284)
        
        # SSA join for if statement (line 510)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'f90' (line 512)
        f90_64286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 11), 'f90')
        # Testing the type of an if condition (line 512)
        if_condition_64287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 512, 8), f90_64286)
        # Assigning a type to the variable 'if_condition_64287' (line 512)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'if_condition_64287', if_condition_64287)
        # SSA begins for if statement (line 512)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_commands(...): (line 513)
        # Processing the call keyword arguments (line 513)
        
        # Obtaining an instance of the builtin type 'list' (line 513)
        list_64290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 513, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 513)
        # Adding element type (line 513)
        # Getting the type of 'f90' (line 513)
        f90_64291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 44), 'f90', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 43), list_64290, f90_64291)
        
        # Getting the type of 'freeflags' (line 513)
        freeflags_64292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 49), 'freeflags', False)
        # Applying the binary operator '+' (line 513)
        result_add_64293 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 43), '+', list_64290, freeflags_64292)
        
        # Getting the type of 'f90flags' (line 513)
        f90flags_64294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 59), 'f90flags', False)
        # Applying the binary operator '+' (line 513)
        result_add_64295 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 58), '+', result_add_64293, f90flags_64294)
        
        # Getting the type of 'fflags' (line 513)
        fflags_64296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 68), 'fflags', False)
        # Applying the binary operator '+' (line 513)
        result_add_64297 = python_operator(stypy.reporting.localization.Localization(__file__, 513, 67), '+', result_add_64295, fflags_64296)
        
        keyword_64298 = result_add_64297
        kwargs_64299 = {'compiler_f90': keyword_64298}
        # Getting the type of 'self' (line 513)
        self_64288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 12), 'self', False)
        # Obtaining the member 'set_commands' of a type (line 513)
        set_commands_64289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 12), self_64288, 'set_commands')
        # Calling set_commands(args, kwargs) (line 513)
        set_commands_call_result_64300 = invoke(stypy.reporting.localization.Localization(__file__, 513, 12), set_commands_64289, *[], **kwargs_64299)
        
        # SSA join for if statement (line 512)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'fix' (line 514)
        fix_64301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 11), 'fix')
        # Testing the type of an if condition (line 514)
        if_condition_64302 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 514, 8), fix_64301)
        # Assigning a type to the variable 'if_condition_64302' (line 514)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 8), 'if_condition_64302', if_condition_64302)
        # SSA begins for if statement (line 514)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_commands(...): (line 515)
        # Processing the call keyword arguments (line 515)
        
        # Obtaining an instance of the builtin type 'list' (line 515)
        list_64305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 515)
        # Adding element type (line 515)
        # Getting the type of 'fix' (line 515)
        fix_64306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 44), 'fix', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 515, 43), list_64305, fix_64306)
        
        # Getting the type of 'fixflags' (line 515)
        fixflags_64307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 49), 'fixflags', False)
        # Applying the binary operator '+' (line 515)
        result_add_64308 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 43), '+', list_64305, fixflags_64307)
        
        # Getting the type of 'fflags' (line 515)
        fflags_64309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 58), 'fflags', False)
        # Applying the binary operator '+' (line 515)
        result_add_64310 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 57), '+', result_add_64308, fflags_64309)
        
        keyword_64311 = result_add_64310
        kwargs_64312 = {'compiler_fix': keyword_64311}
        # Getting the type of 'self' (line 515)
        self_64303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'self', False)
        # Obtaining the member 'set_commands' of a type (line 515)
        set_commands_64304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 12), self_64303, 'set_commands')
        # Calling set_commands(args, kwargs) (line 515)
        set_commands_call_result_64313 = invoke(stypy.reporting.localization.Localization(__file__, 515, 12), set_commands_64304, *[], **kwargs_64312)
        
        # SSA join for if statement (line 514)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 519):
        
        # Assigning a Attribute to a Name (line 519):
        # Getting the type of 'self' (line 519)
        self_64314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 20), 'self')
        # Obtaining the member 'linker_so' of a type (line 519)
        linker_so_64315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 20), self_64314, 'linker_so')
        # Assigning a type to the variable 'linker_so' (line 519)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'linker_so', linker_so_64315)
        
        # Getting the type of 'linker_so' (line 520)
        linker_so_64316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 11), 'linker_so')
        # Testing the type of an if condition (line 520)
        if_condition_64317 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 520, 8), linker_so_64316)
        # Assigning a type to the variable 'if_condition_64317' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'if_condition_64317', if_condition_64317)
        # SSA begins for if statement (line 520)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 521):
        
        # Assigning a Attribute to a Name (line 521):
        # Getting the type of 'self' (line 521)
        self_64318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 30), 'self')
        # Obtaining the member 'flag_vars' of a type (line 521)
        flag_vars_64319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 30), self_64318, 'flag_vars')
        # Obtaining the member 'linker_so' of a type (line 521)
        linker_so_64320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 30), flag_vars_64319, 'linker_so')
        # Assigning a type to the variable 'linker_so_flags' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 12), 'linker_so_flags', linker_so_64320)
        
        
        # Call to startswith(...): (line 522)
        # Processing the call arguments (line 522)
        str_64324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 39), 'str', 'aix')
        # Processing the call keyword arguments (line 522)
        kwargs_64325 = {}
        # Getting the type of 'sys' (line 522)
        sys_64321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 15), 'sys', False)
        # Obtaining the member 'platform' of a type (line 522)
        platform_64322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 15), sys_64321, 'platform')
        # Obtaining the member 'startswith' of a type (line 522)
        startswith_64323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 15), platform_64322, 'startswith')
        # Calling startswith(args, kwargs) (line 522)
        startswith_call_result_64326 = invoke(stypy.reporting.localization.Localization(__file__, 522, 15), startswith_64323, *[str_64324], **kwargs_64325)
        
        # Testing the type of an if condition (line 522)
        if_condition_64327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 522, 12), startswith_call_result_64326)
        # Assigning a type to the variable 'if_condition_64327' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 12), 'if_condition_64327', if_condition_64327)
        # SSA begins for if statement (line 522)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 523):
        
        # Assigning a Call to a Name (line 523):
        
        # Call to get_python_lib(...): (line 523)
        # Processing the call keyword arguments (line 523)
        int_64329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 57), 'int')
        keyword_64330 = int_64329
        kwargs_64331 = {'standard_lib': keyword_64330}
        # Getting the type of 'get_python_lib' (line 523)
        get_python_lib_64328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 29), 'get_python_lib', False)
        # Calling get_python_lib(args, kwargs) (line 523)
        get_python_lib_call_result_64332 = invoke(stypy.reporting.localization.Localization(__file__, 523, 29), get_python_lib_64328, *[], **kwargs_64331)
        
        # Assigning a type to the variable 'python_lib' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 16), 'python_lib', get_python_lib_call_result_64332)
        
        # Assigning a Call to a Name (line 524):
        
        # Assigning a Call to a Name (line 524):
        
        # Call to join(...): (line 524)
        # Processing the call arguments (line 524)
        # Getting the type of 'python_lib' (line 524)
        python_lib_64336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 41), 'python_lib', False)
        str_64337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 53), 'str', 'config')
        str_64338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 63), 'str', 'ld_so_aix')
        # Processing the call keyword arguments (line 524)
        kwargs_64339 = {}
        # Getting the type of 'os' (line 524)
        os_64333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 524)
        path_64334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 28), os_64333, 'path')
        # Obtaining the member 'join' of a type (line 524)
        join_64335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 28), path_64334, 'join')
        # Calling join(args, kwargs) (line 524)
        join_call_result_64340 = invoke(stypy.reporting.localization.Localization(__file__, 524, 28), join_64335, *[python_lib_64336, str_64337, str_64338], **kwargs_64339)
        
        # Assigning a type to the variable 'ld_so_aix' (line 524)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 16), 'ld_so_aix', join_call_result_64340)
        
        # Assigning a Call to a Name (line 525):
        
        # Assigning a Call to a Name (line 525):
        
        # Call to join(...): (line 525)
        # Processing the call arguments (line 525)
        # Getting the type of 'python_lib' (line 525)
        python_lib_64344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 42), 'python_lib', False)
        str_64345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 54), 'str', 'config')
        str_64346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 525, 64), 'str', 'python.exp')
        # Processing the call keyword arguments (line 525)
        kwargs_64347 = {}
        # Getting the type of 'os' (line 525)
        os_64341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 29), 'os', False)
        # Obtaining the member 'path' of a type (line 525)
        path_64342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 29), os_64341, 'path')
        # Obtaining the member 'join' of a type (line 525)
        join_64343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 29), path_64342, 'join')
        # Calling join(args, kwargs) (line 525)
        join_call_result_64348 = invoke(stypy.reporting.localization.Localization(__file__, 525, 29), join_64343, *[python_lib_64344, str_64345, str_64346], **kwargs_64347)
        
        # Assigning a type to the variable 'python_exp' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 16), 'python_exp', join_call_result_64348)
        
        # Assigning a BinOp to a Name (line 526):
        
        # Assigning a BinOp to a Name (line 526):
        
        # Obtaining an instance of the builtin type 'list' (line 526)
        list_64349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 526)
        # Adding element type (line 526)
        # Getting the type of 'ld_so_aix' (line 526)
        ld_so_aix_64350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 29), 'ld_so_aix')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 28), list_64349, ld_so_aix_64350)
        
        # Getting the type of 'linker_so' (line 526)
        linker_so_64351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 42), 'linker_so')
        # Applying the binary operator '+' (line 526)
        result_add_64352 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 28), '+', list_64349, linker_so_64351)
        
        
        # Obtaining an instance of the builtin type 'list' (line 526)
        list_64353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 526)
        # Adding element type (line 526)
        str_64354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 55), 'str', '-bI:')
        # Getting the type of 'python_exp' (line 526)
        python_exp_64355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 62), 'python_exp')
        # Applying the binary operator '+' (line 526)
        result_add_64356 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 55), '+', str_64354, python_exp_64355)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 526, 54), list_64353, result_add_64356)
        
        # Applying the binary operator '+' (line 526)
        result_add_64357 = python_operator(stypy.reporting.localization.Localization(__file__, 526, 52), '+', result_add_64352, list_64353)
        
        # Assigning a type to the variable 'linker_so' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 16), 'linker_so', result_add_64357)
        # SSA join for if statement (line 522)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_commands(...): (line 527)
        # Processing the call keyword arguments (line 527)
        # Getting the type of 'linker_so' (line 527)
        linker_so_64360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 40), 'linker_so', False)
        # Getting the type of 'linker_so_flags' (line 527)
        linker_so_flags_64361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 50), 'linker_so_flags', False)
        # Applying the binary operator '+' (line 527)
        result_add_64362 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 40), '+', linker_so_64360, linker_so_flags_64361)
        
        keyword_64363 = result_add_64362
        kwargs_64364 = {'linker_so': keyword_64363}
        # Getting the type of 'self' (line 527)
        self_64358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'self', False)
        # Obtaining the member 'set_commands' of a type (line 527)
        set_commands_64359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 12), self_64358, 'set_commands')
        # Calling set_commands(args, kwargs) (line 527)
        set_commands_call_result_64365 = invoke(stypy.reporting.localization.Localization(__file__, 527, 12), set_commands_64359, *[], **kwargs_64364)
        
        # SSA join for if statement (line 520)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 529):
        
        # Assigning a Attribute to a Name (line 529):
        # Getting the type of 'self' (line 529)
        self_64366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 21), 'self')
        # Obtaining the member 'linker_exe' of a type (line 529)
        linker_exe_64367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 21), self_64366, 'linker_exe')
        # Assigning a type to the variable 'linker_exe' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 8), 'linker_exe', linker_exe_64367)
        
        # Getting the type of 'linker_exe' (line 530)
        linker_exe_64368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 11), 'linker_exe')
        # Testing the type of an if condition (line 530)
        if_condition_64369 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 530, 8), linker_exe_64368)
        # Assigning a type to the variable 'if_condition_64369' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'if_condition_64369', if_condition_64369)
        # SSA begins for if statement (line 530)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 531):
        
        # Assigning a Attribute to a Name (line 531):
        # Getting the type of 'self' (line 531)
        self_64370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 31), 'self')
        # Obtaining the member 'flag_vars' of a type (line 531)
        flag_vars_64371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 31), self_64370, 'flag_vars')
        # Obtaining the member 'linker_exe' of a type (line 531)
        linker_exe_64372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 31), flag_vars_64371, 'linker_exe')
        # Assigning a type to the variable 'linker_exe_flags' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'linker_exe_flags', linker_exe_64372)
        
        # Call to set_commands(...): (line 532)
        # Processing the call keyword arguments (line 532)
        # Getting the type of 'linker_exe' (line 532)
        linker_exe_64375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 41), 'linker_exe', False)
        # Getting the type of 'linker_exe_flags' (line 532)
        linker_exe_flags_64376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 52), 'linker_exe_flags', False)
        # Applying the binary operator '+' (line 532)
        result_add_64377 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 41), '+', linker_exe_64375, linker_exe_flags_64376)
        
        keyword_64378 = result_add_64377
        kwargs_64379 = {'linker_exe': keyword_64378}
        # Getting the type of 'self' (line 532)
        self_64373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'self', False)
        # Obtaining the member 'set_commands' of a type (line 532)
        set_commands_64374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 12), self_64373, 'set_commands')
        # Calling set_commands(args, kwargs) (line 532)
        set_commands_call_result_64380 = invoke(stypy.reporting.localization.Localization(__file__, 532, 12), set_commands_64374, *[], **kwargs_64379)
        
        # SSA join for if statement (line 530)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 534):
        
        # Assigning a Attribute to a Name (line 534):
        # Getting the type of 'self' (line 534)
        self_64381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 13), 'self')
        # Obtaining the member 'command_vars' of a type (line 534)
        command_vars_64382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 13), self_64381, 'command_vars')
        # Obtaining the member 'archiver' of a type (line 534)
        archiver_64383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 13), command_vars_64382, 'archiver')
        # Assigning a type to the variable 'ar' (line 534)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'ar', archiver_64383)
        
        # Getting the type of 'ar' (line 535)
        ar_64384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 11), 'ar')
        # Testing the type of an if condition (line 535)
        if_condition_64385 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 535, 8), ar_64384)
        # Assigning a type to the variable 'if_condition_64385' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 8), 'if_condition_64385', if_condition_64385)
        # SSA begins for if statement (line 535)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 536):
        
        # Assigning a Attribute to a Name (line 536):
        # Getting the type of 'self' (line 536)
        self_64386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 22), 'self')
        # Obtaining the member 'flag_vars' of a type (line 536)
        flag_vars_64387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 22), self_64386, 'flag_vars')
        # Obtaining the member 'ar' of a type (line 536)
        ar_64388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 22), flag_vars_64387, 'ar')
        # Assigning a type to the variable 'arflags' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 12), 'arflags', ar_64388)
        
        # Call to set_commands(...): (line 537)
        # Processing the call keyword arguments (line 537)
        
        # Obtaining an instance of the builtin type 'list' (line 537)
        list_64391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 537)
        # Adding element type (line 537)
        # Getting the type of 'ar' (line 537)
        ar_64392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 40), 'ar', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 39), list_64391, ar_64392)
        
        # Getting the type of 'arflags' (line 537)
        arflags_64393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 44), 'arflags', False)
        # Applying the binary operator '+' (line 537)
        result_add_64394 = python_operator(stypy.reporting.localization.Localization(__file__, 537, 39), '+', list_64391, arflags_64393)
        
        keyword_64395 = result_add_64394
        kwargs_64396 = {'archiver': keyword_64395}
        # Getting the type of 'self' (line 537)
        self_64389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'self', False)
        # Obtaining the member 'set_commands' of a type (line 537)
        set_commands_64390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 12), self_64389, 'set_commands')
        # Calling set_commands(args, kwargs) (line 537)
        set_commands_call_result_64397 = invoke(stypy.reporting.localization.Localization(__file__, 537, 12), set_commands_64390, *[], **kwargs_64396)
        
        # SSA join for if statement (line 535)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_library_dirs(...): (line 539)
        # Processing the call arguments (line 539)
        
        # Call to get_library_dirs(...): (line 539)
        # Processing the call keyword arguments (line 539)
        kwargs_64402 = {}
        # Getting the type of 'self' (line 539)
        self_64400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 30), 'self', False)
        # Obtaining the member 'get_library_dirs' of a type (line 539)
        get_library_dirs_64401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 30), self_64400, 'get_library_dirs')
        # Calling get_library_dirs(args, kwargs) (line 539)
        get_library_dirs_call_result_64403 = invoke(stypy.reporting.localization.Localization(__file__, 539, 30), get_library_dirs_64401, *[], **kwargs_64402)
        
        # Processing the call keyword arguments (line 539)
        kwargs_64404 = {}
        # Getting the type of 'self' (line 539)
        self_64398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'self', False)
        # Obtaining the member 'set_library_dirs' of a type (line 539)
        set_library_dirs_64399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 8), self_64398, 'set_library_dirs')
        # Calling set_library_dirs(args, kwargs) (line 539)
        set_library_dirs_call_result_64405 = invoke(stypy.reporting.localization.Localization(__file__, 539, 8), set_library_dirs_64399, *[get_library_dirs_call_result_64403], **kwargs_64404)
        
        
        # Call to set_libraries(...): (line 540)
        # Processing the call arguments (line 540)
        
        # Call to get_libraries(...): (line 540)
        # Processing the call keyword arguments (line 540)
        kwargs_64410 = {}
        # Getting the type of 'self' (line 540)
        self_64408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 27), 'self', False)
        # Obtaining the member 'get_libraries' of a type (line 540)
        get_libraries_64409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 27), self_64408, 'get_libraries')
        # Calling get_libraries(args, kwargs) (line 540)
        get_libraries_call_result_64411 = invoke(stypy.reporting.localization.Localization(__file__, 540, 27), get_libraries_64409, *[], **kwargs_64410)
        
        # Processing the call keyword arguments (line 540)
        kwargs_64412 = {}
        # Getting the type of 'self' (line 540)
        self_64406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'self', False)
        # Obtaining the member 'set_libraries' of a type (line 540)
        set_libraries_64407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 8), self_64406, 'set_libraries')
        # Calling set_libraries(args, kwargs) (line 540)
        set_libraries_call_result_64413 = invoke(stypy.reporting.localization.Localization(__file__, 540, 8), set_libraries_64407, *[get_libraries_call_result_64411], **kwargs_64412)
        
        
        # ################# End of 'customize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'customize' in the type store
        # Getting the type of 'stypy_return_type' (line 441)
        stypy_return_type_64414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64414)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'customize'
        return stypy_return_type_64414


    @norecursion
    def dump_properties(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dump_properties'
        module_type_store = module_type_store.open_function_context('dump_properties', 542, 4, False)
        # Assigning a type to the variable 'self' (line 543)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.dump_properties.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.dump_properties.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.dump_properties.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.dump_properties.__dict__.__setitem__('stypy_function_name', 'FCompiler.dump_properties')
        FCompiler.dump_properties.__dict__.__setitem__('stypy_param_names_list', [])
        FCompiler.dump_properties.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.dump_properties.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.dump_properties.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.dump_properties.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.dump_properties.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.dump_properties.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.dump_properties', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dump_properties', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dump_properties(...)' code ##################

        str_64415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 8), 'str', 'Print out the attributes of a compiler instance.')
        
        # Assigning a List to a Name (line 544):
        
        # Assigning a List to a Name (line 544):
        
        # Obtaining an instance of the builtin type 'list' (line 544)
        list_64416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 544)
        
        # Assigning a type to the variable 'props' (line 544)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'props', list_64416)
        
        
        # Call to list(...): (line 545)
        # Processing the call arguments (line 545)
        
        # Call to keys(...): (line 545)
        # Processing the call keyword arguments (line 545)
        kwargs_64421 = {}
        # Getting the type of 'self' (line 545)
        self_64418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 24), 'self', False)
        # Obtaining the member 'executables' of a type (line 545)
        executables_64419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 24), self_64418, 'executables')
        # Obtaining the member 'keys' of a type (line 545)
        keys_64420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 24), executables_64419, 'keys')
        # Calling keys(args, kwargs) (line 545)
        keys_call_result_64422 = invoke(stypy.reporting.localization.Localization(__file__, 545, 24), keys_64420, *[], **kwargs_64421)
        
        # Processing the call keyword arguments (line 545)
        kwargs_64423 = {}
        # Getting the type of 'list' (line 545)
        list_64417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 19), 'list', False)
        # Calling list(args, kwargs) (line 545)
        list_call_result_64424 = invoke(stypy.reporting.localization.Localization(__file__, 545, 19), list_64417, *[keys_call_result_64422], **kwargs_64423)
        
        
        # Obtaining an instance of the builtin type 'list' (line 546)
        list_64425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 546)
        # Adding element type (line 546)
        str_64426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 17), 'str', 'version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 16), list_64425, str_64426)
        # Adding element type (line 546)
        str_64427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 28), 'str', 'libraries')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 16), list_64425, str_64427)
        # Adding element type (line 546)
        str_64428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 546, 41), 'str', 'library_dirs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 16), list_64425, str_64428)
        # Adding element type (line 546)
        str_64429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 17), 'str', 'object_switch')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 16), list_64425, str_64429)
        # Adding element type (line 546)
        str_64430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 34), 'str', 'compile_switch')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 546, 16), list_64425, str_64430)
        
        # Applying the binary operator '+' (line 545)
        result_add_64431 = python_operator(stypy.reporting.localization.Localization(__file__, 545, 19), '+', list_call_result_64424, list_64425)
        
        # Testing the type of a for loop iterable (line 545)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 545, 8), result_add_64431)
        # Getting the type of the for loop variable (line 545)
        for_loop_var_64432 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 545, 8), result_add_64431)
        # Assigning a type to the variable 'key' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'key', for_loop_var_64432)
        # SSA begins for a for statement (line 545)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to hasattr(...): (line 548)
        # Processing the call arguments (line 548)
        # Getting the type of 'self' (line 548)
        self_64434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 23), 'self', False)
        # Getting the type of 'key' (line 548)
        key_64435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 29), 'key', False)
        # Processing the call keyword arguments (line 548)
        kwargs_64436 = {}
        # Getting the type of 'hasattr' (line 548)
        hasattr_64433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 548)
        hasattr_call_result_64437 = invoke(stypy.reporting.localization.Localization(__file__, 548, 15), hasattr_64433, *[self_64434, key_64435], **kwargs_64436)
        
        # Testing the type of an if condition (line 548)
        if_condition_64438 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 548, 12), hasattr_call_result_64437)
        # Assigning a type to the variable 'if_condition_64438' (line 548)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 12), 'if_condition_64438', if_condition_64438)
        # SSA begins for if statement (line 548)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 549):
        
        # Assigning a Call to a Name (line 549):
        
        # Call to getattr(...): (line 549)
        # Processing the call arguments (line 549)
        # Getting the type of 'self' (line 549)
        self_64440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 28), 'self', False)
        # Getting the type of 'key' (line 549)
        key_64441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 34), 'key', False)
        # Processing the call keyword arguments (line 549)
        kwargs_64442 = {}
        # Getting the type of 'getattr' (line 549)
        getattr_64439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 549)
        getattr_call_result_64443 = invoke(stypy.reporting.localization.Localization(__file__, 549, 20), getattr_64439, *[self_64440, key_64441], **kwargs_64442)
        
        # Assigning a type to the variable 'v' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 16), 'v', getattr_call_result_64443)
        
        # Call to append(...): (line 550)
        # Processing the call arguments (line 550)
        
        # Obtaining an instance of the builtin type 'tuple' (line 550)
        tuple_64446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 550)
        # Adding element type (line 550)
        # Getting the type of 'key' (line 550)
        key_64447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 30), 'key', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 30), tuple_64446, key_64447)
        # Adding element type (line 550)
        # Getting the type of 'None' (line 550)
        None_64448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 35), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 30), tuple_64446, None_64448)
        # Adding element type (line 550)
        str_64449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 41), 'str', '= ')
        
        # Call to repr(...): (line 550)
        # Processing the call arguments (line 550)
        # Getting the type of 'v' (line 550)
        v_64451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 51), 'v', False)
        # Processing the call keyword arguments (line 550)
        kwargs_64452 = {}
        # Getting the type of 'repr' (line 550)
        repr_64450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 46), 'repr', False)
        # Calling repr(args, kwargs) (line 550)
        repr_call_result_64453 = invoke(stypy.reporting.localization.Localization(__file__, 550, 46), repr_64450, *[v_64451], **kwargs_64452)
        
        # Applying the binary operator '+' (line 550)
        result_add_64454 = python_operator(stypy.reporting.localization.Localization(__file__, 550, 41), '+', str_64449, repr_call_result_64453)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 550, 30), tuple_64446, result_add_64454)
        
        # Processing the call keyword arguments (line 550)
        kwargs_64455 = {}
        # Getting the type of 'props' (line 550)
        props_64444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 16), 'props', False)
        # Obtaining the member 'append' of a type (line 550)
        append_64445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 16), props_64444, 'append')
        # Calling append(args, kwargs) (line 550)
        append_call_result_64456 = invoke(stypy.reporting.localization.Localization(__file__, 550, 16), append_64445, *[tuple_64446], **kwargs_64455)
        
        # SSA join for if statement (line 548)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to sort(...): (line 551)
        # Processing the call keyword arguments (line 551)
        kwargs_64459 = {}
        # Getting the type of 'props' (line 551)
        props_64457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 8), 'props', False)
        # Obtaining the member 'sort' of a type (line 551)
        sort_64458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 8), props_64457, 'sort')
        # Calling sort(args, kwargs) (line 551)
        sort_call_result_64460 = invoke(stypy.reporting.localization.Localization(__file__, 551, 8), sort_64458, *[], **kwargs_64459)
        
        
        # Assigning a Call to a Name (line 553):
        
        # Assigning a Call to a Name (line 553):
        
        # Call to FancyGetopt(...): (line 553)
        # Processing the call arguments (line 553)
        # Getting the type of 'props' (line 553)
        props_64462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 37), 'props', False)
        # Processing the call keyword arguments (line 553)
        kwargs_64463 = {}
        # Getting the type of 'FancyGetopt' (line 553)
        FancyGetopt_64461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 25), 'FancyGetopt', False)
        # Calling FancyGetopt(args, kwargs) (line 553)
        FancyGetopt_call_result_64464 = invoke(stypy.reporting.localization.Localization(__file__, 553, 25), FancyGetopt_64461, *[props_64462], **kwargs_64463)
        
        # Assigning a type to the variable 'pretty_printer' (line 553)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 8), 'pretty_printer', FancyGetopt_call_result_64464)
        
        
        # Call to generate_help(...): (line 554)
        # Processing the call arguments (line 554)
        str_64467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 46), 'str', '%s instance properties:')
        # Getting the type of 'self' (line 555)
        self_64468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 49), 'self', False)
        # Obtaining the member '__class__' of a type (line 555)
        class___64469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 49), self_64468, '__class__')
        # Obtaining the member '__name__' of a type (line 555)
        name___64470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 49), class___64469, '__name__')
        # Applying the binary operator '%' (line 554)
        result_mod_64471 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 46), '%', str_64467, name___64470)
        
        # Processing the call keyword arguments (line 554)
        kwargs_64472 = {}
        # Getting the type of 'pretty_printer' (line 554)
        pretty_printer_64465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 17), 'pretty_printer', False)
        # Obtaining the member 'generate_help' of a type (line 554)
        generate_help_64466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 17), pretty_printer_64465, 'generate_help')
        # Calling generate_help(args, kwargs) (line 554)
        generate_help_call_result_64473 = invoke(stypy.reporting.localization.Localization(__file__, 554, 17), generate_help_64466, *[result_mod_64471], **kwargs_64472)
        
        # Testing the type of a for loop iterable (line 554)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 554, 8), generate_help_call_result_64473)
        # Getting the type of the for loop variable (line 554)
        for_loop_var_64474 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 554, 8), generate_help_call_result_64473)
        # Assigning a type to the variable 'l' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'l', for_loop_var_64474)
        # SSA begins for a for statement (line 554)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Obtaining the type of the subscript
        int_64475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 18), 'int')
        slice_64476 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 556, 15), None, int_64475, None)
        # Getting the type of 'l' (line 556)
        l_64477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 15), 'l')
        # Obtaining the member '__getitem__' of a type (line 556)
        getitem___64478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 15), l_64477, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 556)
        subscript_call_result_64479 = invoke(stypy.reporting.localization.Localization(__file__, 556, 15), getitem___64478, slice_64476)
        
        str_64480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 22), 'str', '  --')
        # Applying the binary operator '==' (line 556)
        result_eq_64481 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 15), '==', subscript_call_result_64479, str_64480)
        
        # Testing the type of an if condition (line 556)
        if_condition_64482 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 556, 12), result_eq_64481)
        # Assigning a type to the variable 'if_condition_64482' (line 556)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 12), 'if_condition_64482', if_condition_64482)
        # SSA begins for if statement (line 556)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 557):
        
        # Assigning a BinOp to a Name (line 557):
        str_64483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 20), 'str', '  ')
        
        # Obtaining the type of the subscript
        int_64484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 29), 'int')
        slice_64485 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 557, 27), int_64484, None, None)
        # Getting the type of 'l' (line 557)
        l_64486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 27), 'l')
        # Obtaining the member '__getitem__' of a type (line 557)
        getitem___64487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 27), l_64486, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 557)
        subscript_call_result_64488 = invoke(stypy.reporting.localization.Localization(__file__, 557, 27), getitem___64487, slice_64485)
        
        # Applying the binary operator '+' (line 557)
        result_add_64489 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 20), '+', str_64483, subscript_call_result_64488)
        
        # Assigning a type to the variable 'l' (line 557)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 16), 'l', result_add_64489)
        # SSA join for if statement (line 556)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to print(...): (line 558)
        # Processing the call arguments (line 558)
        # Getting the type of 'l' (line 558)
        l_64491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 18), 'l', False)
        # Processing the call keyword arguments (line 558)
        kwargs_64492 = {}
        # Getting the type of 'print' (line 558)
        print_64490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'print', False)
        # Calling print(args, kwargs) (line 558)
        print_call_result_64493 = invoke(stypy.reporting.localization.Localization(__file__, 558, 12), print_64490, *[l_64491], **kwargs_64492)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'dump_properties(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dump_properties' in the type store
        # Getting the type of 'stypy_return_type' (line 542)
        stypy_return_type_64494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64494)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dump_properties'
        return stypy_return_type_64494


    @norecursion
    def _compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_compile'
        module_type_store = module_type_store.open_function_context('_compile', 562, 4, False)
        # Assigning a type to the variable 'self' (line 563)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler._compile.__dict__.__setitem__('stypy_localization', localization)
        FCompiler._compile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler._compile.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler._compile.__dict__.__setitem__('stypy_function_name', 'FCompiler._compile')
        FCompiler._compile.__dict__.__setitem__('stypy_param_names_list', ['obj', 'src', 'ext', 'cc_args', 'extra_postargs', 'pp_opts'])
        FCompiler._compile.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler._compile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler._compile.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler._compile.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler._compile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler._compile.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler._compile', ['obj', 'src', 'ext', 'cc_args', 'extra_postargs', 'pp_opts'], None, None, defaults, varargs, kwargs)

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

        str_64495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 8), 'str', "Compile 'src' to product 'obj'.")
        
        # Assigning a Dict to a Name (line 564):
        
        # Assigning a Dict to a Name (line 564):
        
        # Obtaining an instance of the builtin type 'dict' (line 564)
        dict_64496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 20), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 564)
        
        # Assigning a type to the variable 'src_flags' (line 564)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'src_flags', dict_64496)
        
        
        # Evaluating a boolean operation
        
        # Call to is_f_file(...): (line 565)
        # Processing the call arguments (line 565)
        # Getting the type of 'src' (line 565)
        src_64498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 21), 'src', False)
        # Processing the call keyword arguments (line 565)
        kwargs_64499 = {}
        # Getting the type of 'is_f_file' (line 565)
        is_f_file_64497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 11), 'is_f_file', False)
        # Calling is_f_file(args, kwargs) (line 565)
        is_f_file_call_result_64500 = invoke(stypy.reporting.localization.Localization(__file__, 565, 11), is_f_file_64497, *[src_64498], **kwargs_64499)
        
        
        
        # Call to has_f90_header(...): (line 565)
        # Processing the call arguments (line 565)
        # Getting the type of 'src' (line 565)
        src_64502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 49), 'src', False)
        # Processing the call keyword arguments (line 565)
        kwargs_64503 = {}
        # Getting the type of 'has_f90_header' (line 565)
        has_f90_header_64501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 34), 'has_f90_header', False)
        # Calling has_f90_header(args, kwargs) (line 565)
        has_f90_header_call_result_64504 = invoke(stypy.reporting.localization.Localization(__file__, 565, 34), has_f90_header_64501, *[src_64502], **kwargs_64503)
        
        # Applying the 'not' unary operator (line 565)
        result_not__64505 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 30), 'not', has_f90_header_call_result_64504)
        
        # Applying the binary operator 'and' (line 565)
        result_and_keyword_64506 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 11), 'and', is_f_file_call_result_64500, result_not__64505)
        
        # Testing the type of an if condition (line 565)
        if_condition_64507 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 565, 8), result_and_keyword_64506)
        # Assigning a type to the variable 'if_condition_64507' (line 565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'if_condition_64507', if_condition_64507)
        # SSA begins for if statement (line 565)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 566):
        
        # Assigning a Str to a Name (line 566):
        str_64508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 21), 'str', ':f77')
        # Assigning a type to the variable 'flavor' (line 566)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 12), 'flavor', str_64508)
        
        # Assigning a Attribute to a Name (line 567):
        
        # Assigning a Attribute to a Name (line 567):
        # Getting the type of 'self' (line 567)
        self_64509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 23), 'self')
        # Obtaining the member 'compiler_f77' of a type (line 567)
        compiler_f77_64510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 567, 23), self_64509, 'compiler_f77')
        # Assigning a type to the variable 'compiler' (line 567)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 12), 'compiler', compiler_f77_64510)
        
        # Assigning a Call to a Name (line 568):
        
        # Assigning a Call to a Name (line 568):
        
        # Call to get_f77flags(...): (line 568)
        # Processing the call arguments (line 568)
        # Getting the type of 'src' (line 568)
        src_64512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 37), 'src', False)
        # Processing the call keyword arguments (line 568)
        kwargs_64513 = {}
        # Getting the type of 'get_f77flags' (line 568)
        get_f77flags_64511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 24), 'get_f77flags', False)
        # Calling get_f77flags(args, kwargs) (line 568)
        get_f77flags_call_result_64514 = invoke(stypy.reporting.localization.Localization(__file__, 568, 24), get_f77flags_64511, *[src_64512], **kwargs_64513)
        
        # Assigning a type to the variable 'src_flags' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 12), 'src_flags', get_f77flags_call_result_64514)
        
        # Assigning a BoolOp to a Name (line 569):
        
        # Assigning a BoolOp to a Name (line 569):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 569)
        self_64515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 33), 'self')
        # Obtaining the member 'extra_f77_compile_args' of a type (line 569)
        extra_f77_compile_args_64516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 33), self_64515, 'extra_f77_compile_args')
        
        # Obtaining an instance of the builtin type 'list' (line 569)
        list_64517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 569)
        
        # Applying the binary operator 'or' (line 569)
        result_or_keyword_64518 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 33), 'or', extra_f77_compile_args_64516, list_64517)
        
        # Assigning a type to the variable 'extra_compile_args' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'extra_compile_args', result_or_keyword_64518)
        # SSA branch for the else part of an if statement (line 565)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to is_free_format(...): (line 570)
        # Processing the call arguments (line 570)
        # Getting the type of 'src' (line 570)
        src_64520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 28), 'src', False)
        # Processing the call keyword arguments (line 570)
        kwargs_64521 = {}
        # Getting the type of 'is_free_format' (line 570)
        is_free_format_64519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 13), 'is_free_format', False)
        # Calling is_free_format(args, kwargs) (line 570)
        is_free_format_call_result_64522 = invoke(stypy.reporting.localization.Localization(__file__, 570, 13), is_free_format_64519, *[src_64520], **kwargs_64521)
        
        # Testing the type of an if condition (line 570)
        if_condition_64523 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 570, 13), is_free_format_call_result_64522)
        # Assigning a type to the variable 'if_condition_64523' (line 570)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 13), 'if_condition_64523', if_condition_64523)
        # SSA begins for if statement (line 570)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 571):
        
        # Assigning a Str to a Name (line 571):
        str_64524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 21), 'str', ':f90')
        # Assigning a type to the variable 'flavor' (line 571)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 12), 'flavor', str_64524)
        
        # Assigning a Attribute to a Name (line 572):
        
        # Assigning a Attribute to a Name (line 572):
        # Getting the type of 'self' (line 572)
        self_64525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 23), 'self')
        # Obtaining the member 'compiler_f90' of a type (line 572)
        compiler_f90_64526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 23), self_64525, 'compiler_f90')
        # Assigning a type to the variable 'compiler' (line 572)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'compiler', compiler_f90_64526)
        
        # Type idiom detected: calculating its left and rigth part (line 573)
        # Getting the type of 'compiler' (line 573)
        compiler_64527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 15), 'compiler')
        # Getting the type of 'None' (line 573)
        None_64528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 27), 'None')
        
        (may_be_64529, more_types_in_union_64530) = may_be_none(compiler_64527, None_64528)

        if may_be_64529:

            if more_types_in_union_64530:
                # Runtime conditional SSA (line 573)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to DistutilsExecError(...): (line 574)
            # Processing the call arguments (line 574)
            str_64532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 41), 'str', 'f90 not supported by %s needed for %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 575)
            tuple_64533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 25), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 575)
            # Adding element type (line 575)
            # Getting the type of 'self' (line 575)
            self_64534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 25), 'self', False)
            # Obtaining the member '__class__' of a type (line 575)
            class___64535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 25), self_64534, '__class__')
            # Obtaining the member '__name__' of a type (line 575)
            name___64536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 25), class___64535, '__name__')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 25), tuple_64533, name___64536)
            # Adding element type (line 575)
            # Getting the type of 'src' (line 575)
            src_64537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 50), 'src', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 575, 25), tuple_64533, src_64537)
            
            # Applying the binary operator '%' (line 574)
            result_mod_64538 = python_operator(stypy.reporting.localization.Localization(__file__, 574, 41), '%', str_64532, tuple_64533)
            
            # Processing the call keyword arguments (line 574)
            kwargs_64539 = {}
            # Getting the type of 'DistutilsExecError' (line 574)
            DistutilsExecError_64531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 22), 'DistutilsExecError', False)
            # Calling DistutilsExecError(args, kwargs) (line 574)
            DistutilsExecError_call_result_64540 = invoke(stypy.reporting.localization.Localization(__file__, 574, 22), DistutilsExecError_64531, *[result_mod_64538], **kwargs_64539)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 574, 16), DistutilsExecError_call_result_64540, 'raise parameter', BaseException)

            if more_types_in_union_64530:
                # SSA join for if statement (line 573)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BoolOp to a Name (line 576):
        
        # Assigning a BoolOp to a Name (line 576):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 576)
        self_64541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 33), 'self')
        # Obtaining the member 'extra_f90_compile_args' of a type (line 576)
        extra_f90_compile_args_64542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 33), self_64541, 'extra_f90_compile_args')
        
        # Obtaining an instance of the builtin type 'list' (line 576)
        list_64543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 576)
        
        # Applying the binary operator 'or' (line 576)
        result_or_keyword_64544 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 33), 'or', extra_f90_compile_args_64542, list_64543)
        
        # Assigning a type to the variable 'extra_compile_args' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 12), 'extra_compile_args', result_or_keyword_64544)
        # SSA branch for the else part of an if statement (line 570)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 578):
        
        # Assigning a Str to a Name (line 578):
        str_64545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 21), 'str', ':fix')
        # Assigning a type to the variable 'flavor' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'flavor', str_64545)
        
        # Assigning a Attribute to a Name (line 579):
        
        # Assigning a Attribute to a Name (line 579):
        # Getting the type of 'self' (line 579)
        self_64546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 23), 'self')
        # Obtaining the member 'compiler_fix' of a type (line 579)
        compiler_fix_64547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 23), self_64546, 'compiler_fix')
        # Assigning a type to the variable 'compiler' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 12), 'compiler', compiler_fix_64547)
        
        # Type idiom detected: calculating its left and rigth part (line 580)
        # Getting the type of 'compiler' (line 580)
        compiler_64548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 15), 'compiler')
        # Getting the type of 'None' (line 580)
        None_64549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 27), 'None')
        
        (may_be_64550, more_types_in_union_64551) = may_be_none(compiler_64548, None_64549)

        if may_be_64550:

            if more_types_in_union_64551:
                # Runtime conditional SSA (line 580)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to DistutilsExecError(...): (line 581)
            # Processing the call arguments (line 581)
            str_64553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 41), 'str', 'f90 (fixed) not supported by %s needed for %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 582)
            tuple_64554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 25), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 582)
            # Adding element type (line 582)
            # Getting the type of 'self' (line 582)
            self_64555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 25), 'self', False)
            # Obtaining the member '__class__' of a type (line 582)
            class___64556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 25), self_64555, '__class__')
            # Obtaining the member '__name__' of a type (line 582)
            name___64557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 25), class___64556, '__name__')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 25), tuple_64554, name___64557)
            # Adding element type (line 582)
            # Getting the type of 'src' (line 582)
            src_64558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 50), 'src', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 582, 25), tuple_64554, src_64558)
            
            # Applying the binary operator '%' (line 581)
            result_mod_64559 = python_operator(stypy.reporting.localization.Localization(__file__, 581, 41), '%', str_64553, tuple_64554)
            
            # Processing the call keyword arguments (line 581)
            kwargs_64560 = {}
            # Getting the type of 'DistutilsExecError' (line 581)
            DistutilsExecError_64552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 22), 'DistutilsExecError', False)
            # Calling DistutilsExecError(args, kwargs) (line 581)
            DistutilsExecError_call_result_64561 = invoke(stypy.reporting.localization.Localization(__file__, 581, 22), DistutilsExecError_64552, *[result_mod_64559], **kwargs_64560)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 581, 16), DistutilsExecError_call_result_64561, 'raise parameter', BaseException)

            if more_types_in_union_64551:
                # SSA join for if statement (line 580)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BoolOp to a Name (line 583):
        
        # Assigning a BoolOp to a Name (line 583):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 583)
        self_64562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 33), 'self')
        # Obtaining the member 'extra_f90_compile_args' of a type (line 583)
        extra_f90_compile_args_64563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 33), self_64562, 'extra_f90_compile_args')
        
        # Obtaining an instance of the builtin type 'list' (line 583)
        list_64564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 64), 'list')
        # Adding type elements to the builtin type 'list' instance (line 583)
        
        # Applying the binary operator 'or' (line 583)
        result_or_keyword_64565 = python_operator(stypy.reporting.localization.Localization(__file__, 583, 33), 'or', extra_f90_compile_args_64563, list_64564)
        
        # Assigning a type to the variable 'extra_compile_args' (line 583)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'extra_compile_args', result_or_keyword_64565)
        # SSA join for if statement (line 570)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 565)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_64566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 30), 'int')
        # Getting the type of 'self' (line 584)
        self_64567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 11), 'self')
        # Obtaining the member 'object_switch' of a type (line 584)
        object_switch_64568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 11), self_64567, 'object_switch')
        # Obtaining the member '__getitem__' of a type (line 584)
        getitem___64569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 11), object_switch_64568, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 584)
        subscript_call_result_64570 = invoke(stypy.reporting.localization.Localization(__file__, 584, 11), getitem___64569, int_64566)
        
        str_64571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 35), 'str', ' ')
        # Applying the binary operator '==' (line 584)
        result_eq_64572 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 11), '==', subscript_call_result_64570, str_64571)
        
        # Testing the type of an if condition (line 584)
        if_condition_64573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 584, 8), result_eq_64572)
        # Assigning a type to the variable 'if_condition_64573' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'if_condition_64573', if_condition_64573)
        # SSA begins for if statement (line 584)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 585):
        
        # Assigning a List to a Name (line 585):
        
        # Obtaining an instance of the builtin type 'list' (line 585)
        list_64574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 585, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 585)
        # Adding element type (line 585)
        
        # Call to strip(...): (line 585)
        # Processing the call keyword arguments (line 585)
        kwargs_64578 = {}
        # Getting the type of 'self' (line 585)
        self_64575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 22), 'self', False)
        # Obtaining the member 'object_switch' of a type (line 585)
        object_switch_64576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 22), self_64575, 'object_switch')
        # Obtaining the member 'strip' of a type (line 585)
        strip_64577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 22), object_switch_64576, 'strip')
        # Calling strip(args, kwargs) (line 585)
        strip_call_result_64579 = invoke(stypy.reporting.localization.Localization(__file__, 585, 22), strip_64577, *[], **kwargs_64578)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 21), list_64574, strip_call_result_64579)
        # Adding element type (line 585)
        # Getting the type of 'obj' (line 585)
        obj_64580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 50), 'obj')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 21), list_64574, obj_64580)
        
        # Assigning a type to the variable 'o_args' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 12), 'o_args', list_64574)
        # SSA branch for the else part of an if statement (line 584)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 587):
        
        # Assigning a List to a Name (line 587):
        
        # Obtaining an instance of the builtin type 'list' (line 587)
        list_64581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 587)
        # Adding element type (line 587)
        
        # Call to strip(...): (line 587)
        # Processing the call keyword arguments (line 587)
        kwargs_64585 = {}
        # Getting the type of 'self' (line 587)
        self_64582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 22), 'self', False)
        # Obtaining the member 'object_switch' of a type (line 587)
        object_switch_64583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 22), self_64582, 'object_switch')
        # Obtaining the member 'strip' of a type (line 587)
        strip_64584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 22), object_switch_64583, 'strip')
        # Calling strip(args, kwargs) (line 587)
        strip_call_result_64586 = invoke(stypy.reporting.localization.Localization(__file__, 587, 22), strip_64584, *[], **kwargs_64585)
        
        # Getting the type of 'obj' (line 587)
        obj_64587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 49), 'obj')
        # Applying the binary operator '+' (line 587)
        result_add_64588 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 22), '+', strip_call_result_64586, obj_64587)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 587, 21), list_64581, result_add_64588)
        
        # Assigning a type to the variable 'o_args' (line 587)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 12), 'o_args', list_64581)
        # SSA join for if statement (line 584)
        module_type_store = module_type_store.join_ssa_context()
        
        # Evaluating assert statement condition
        
        # Call to strip(...): (line 589)
        # Processing the call keyword arguments (line 589)
        kwargs_64592 = {}
        # Getting the type of 'self' (line 589)
        self_64589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 15), 'self', False)
        # Obtaining the member 'compile_switch' of a type (line 589)
        compile_switch_64590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 15), self_64589, 'compile_switch')
        # Obtaining the member 'strip' of a type (line 589)
        strip_64591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 15), compile_switch_64590, 'strip')
        # Calling strip(args, kwargs) (line 589)
        strip_call_result_64593 = invoke(stypy.reporting.localization.Localization(__file__, 589, 15), strip_64591, *[], **kwargs_64592)
        
        
        # Assigning a List to a Name (line 590):
        
        # Assigning a List to a Name (line 590):
        
        # Obtaining an instance of the builtin type 'list' (line 590)
        list_64594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 590)
        # Adding element type (line 590)
        # Getting the type of 'self' (line 590)
        self_64595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 18), 'self')
        # Obtaining the member 'compile_switch' of a type (line 590)
        compile_switch_64596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 18), self_64595, 'compile_switch')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 17), list_64594, compile_switch_64596)
        # Adding element type (line 590)
        # Getting the type of 'src' (line 590)
        src_64597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 39), 'src')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 17), list_64594, src_64597)
        
        # Assigning a type to the variable 's_args' (line 590)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 590, 8), 's_args', list_64594)
        
        # Getting the type of 'extra_compile_args' (line 592)
        extra_compile_args_64598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 11), 'extra_compile_args')
        # Testing the type of an if condition (line 592)
        if_condition_64599 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 592, 8), extra_compile_args_64598)
        # Assigning a type to the variable 'if_condition_64599' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'if_condition_64599', if_condition_64599)
        # SSA begins for if statement (line 592)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 593)
        # Processing the call arguments (line 593)
        str_64602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 593, 21), 'str', 'extra %s options: %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 594)
        tuple_64603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 594)
        # Adding element type (line 594)
        
        # Obtaining the type of the subscript
        int_64604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 31), 'int')
        slice_64605 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 594, 24), int_64604, None, None)
        # Getting the type of 'flavor' (line 594)
        flavor_64606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 24), 'flavor', False)
        # Obtaining the member '__getitem__' of a type (line 594)
        getitem___64607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 24), flavor_64606, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 594)
        subscript_call_result_64608 = invoke(stypy.reporting.localization.Localization(__file__, 594, 24), getitem___64607, slice_64605)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 24), tuple_64603, subscript_call_result_64608)
        # Adding element type (line 594)
        
        # Call to join(...): (line 594)
        # Processing the call arguments (line 594)
        # Getting the type of 'extra_compile_args' (line 594)
        extra_compile_args_64611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 45), 'extra_compile_args', False)
        # Processing the call keyword arguments (line 594)
        kwargs_64612 = {}
        str_64609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 36), 'str', ' ')
        # Obtaining the member 'join' of a type (line 594)
        join_64610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 36), str_64609, 'join')
        # Calling join(args, kwargs) (line 594)
        join_call_result_64613 = invoke(stypy.reporting.localization.Localization(__file__, 594, 36), join_64610, *[extra_compile_args_64611], **kwargs_64612)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 594, 24), tuple_64603, join_call_result_64613)
        
        # Applying the binary operator '%' (line 593)
        result_mod_64614 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 21), '%', str_64602, tuple_64603)
        
        # Processing the call keyword arguments (line 593)
        kwargs_64615 = {}
        # Getting the type of 'log' (line 593)
        log_64600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 593)
        info_64601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 12), log_64600, 'info')
        # Calling info(args, kwargs) (line 593)
        info_call_result_64616 = invoke(stypy.reporting.localization.Localization(__file__, 593, 12), info_64601, *[result_mod_64614], **kwargs_64615)
        
        # SSA join for if statement (line 592)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 596):
        
        # Assigning a Call to a Name (line 596):
        
        # Call to get(...): (line 596)
        # Processing the call arguments (line 596)
        # Getting the type of 'self' (line 596)
        self_64619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 36), 'self', False)
        # Obtaining the member 'compiler_type' of a type (line 596)
        compiler_type_64620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 36), self_64619, 'compiler_type')
        
        # Obtaining an instance of the builtin type 'list' (line 596)
        list_64621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 596, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 596)
        
        # Processing the call keyword arguments (line 596)
        kwargs_64622 = {}
        # Getting the type of 'src_flags' (line 596)
        src_flags_64617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 22), 'src_flags', False)
        # Obtaining the member 'get' of a type (line 596)
        get_64618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 22), src_flags_64617, 'get')
        # Calling get(args, kwargs) (line 596)
        get_call_result_64623 = invoke(stypy.reporting.localization.Localization(__file__, 596, 22), get_64618, *[compiler_type_64620, list_64621], **kwargs_64622)
        
        # Assigning a type to the variable 'extra_flags' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'extra_flags', get_call_result_64623)
        
        # Getting the type of 'extra_flags' (line 597)
        extra_flags_64624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 11), 'extra_flags')
        # Testing the type of an if condition (line 597)
        if_condition_64625 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 597, 8), extra_flags_64624)
        # Assigning a type to the variable 'if_condition_64625' (line 597)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'if_condition_64625', if_condition_64625)
        # SSA begins for if statement (line 597)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 598)
        # Processing the call arguments (line 598)
        str_64628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 21), 'str', 'using compile options from source: %r')
        
        # Call to join(...): (line 599)
        # Processing the call arguments (line 599)
        # Getting the type of 'extra_flags' (line 599)
        extra_flags_64631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 32), 'extra_flags', False)
        # Processing the call keyword arguments (line 599)
        kwargs_64632 = {}
        str_64629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 23), 'str', ' ')
        # Obtaining the member 'join' of a type (line 599)
        join_64630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 23), str_64629, 'join')
        # Calling join(args, kwargs) (line 599)
        join_call_result_64633 = invoke(stypy.reporting.localization.Localization(__file__, 599, 23), join_64630, *[extra_flags_64631], **kwargs_64632)
        
        # Applying the binary operator '%' (line 598)
        result_mod_64634 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 21), '%', str_64628, join_call_result_64633)
        
        # Processing the call keyword arguments (line 598)
        kwargs_64635 = {}
        # Getting the type of 'log' (line 598)
        log_64626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 598)
        info_64627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 12), log_64626, 'info')
        # Calling info(args, kwargs) (line 598)
        info_call_result_64636 = invoke(stypy.reporting.localization.Localization(__file__, 598, 12), info_64627, *[result_mod_64634], **kwargs_64635)
        
        # SSA join for if statement (line 597)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 601):
        
        # Assigning a BinOp to a Name (line 601):
        # Getting the type of 'compiler' (line 601)
        compiler_64637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 18), 'compiler')
        # Getting the type of 'cc_args' (line 601)
        cc_args_64638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 29), 'cc_args')
        # Applying the binary operator '+' (line 601)
        result_add_64639 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 18), '+', compiler_64637, cc_args_64638)
        
        # Getting the type of 'extra_flags' (line 601)
        extra_flags_64640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 39), 'extra_flags')
        # Applying the binary operator '+' (line 601)
        result_add_64641 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 37), '+', result_add_64639, extra_flags_64640)
        
        # Getting the type of 's_args' (line 601)
        s_args_64642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 53), 's_args')
        # Applying the binary operator '+' (line 601)
        result_add_64643 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 51), '+', result_add_64641, s_args_64642)
        
        # Getting the type of 'o_args' (line 601)
        o_args_64644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 62), 'o_args')
        # Applying the binary operator '+' (line 601)
        result_add_64645 = python_operator(stypy.reporting.localization.Localization(__file__, 601, 60), '+', result_add_64643, o_args_64644)
        
        # Getting the type of 'extra_postargs' (line 602)
        extra_postargs_64646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 20), 'extra_postargs')
        # Applying the binary operator '+' (line 602)
        result_add_64647 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 18), '+', result_add_64645, extra_postargs_64646)
        
        # Getting the type of 'extra_compile_args' (line 602)
        extra_compile_args_64648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 37), 'extra_compile_args')
        # Applying the binary operator '+' (line 602)
        result_add_64649 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 35), '+', result_add_64647, extra_compile_args_64648)
        
        # Assigning a type to the variable 'command' (line 601)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 8), 'command', result_add_64649)
        
        # Assigning a BinOp to a Name (line 604):
        
        # Assigning a BinOp to a Name (line 604):
        str_64650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 18), 'str', '%s: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 604)
        tuple_64651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 604)
        # Adding element type (line 604)
        
        # Call to basename(...): (line 604)
        # Processing the call arguments (line 604)
        
        # Obtaining the type of the subscript
        int_64655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 56), 'int')
        # Getting the type of 'compiler' (line 604)
        compiler_64656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 47), 'compiler', False)
        # Obtaining the member '__getitem__' of a type (line 604)
        getitem___64657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 47), compiler_64656, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 604)
        subscript_call_result_64658 = invoke(stypy.reporting.localization.Localization(__file__, 604, 47), getitem___64657, int_64655)
        
        # Processing the call keyword arguments (line 604)
        kwargs_64659 = {}
        # Getting the type of 'os' (line 604)
        os_64652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 604)
        path_64653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 30), os_64652, 'path')
        # Obtaining the member 'basename' of a type (line 604)
        basename_64654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 30), path_64653, 'basename')
        # Calling basename(args, kwargs) (line 604)
        basename_call_result_64660 = invoke(stypy.reporting.localization.Localization(__file__, 604, 30), basename_64654, *[subscript_call_result_64658], **kwargs_64659)
        
        # Getting the type of 'flavor' (line 604)
        flavor_64661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 62), 'flavor')
        # Applying the binary operator '+' (line 604)
        result_add_64662 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 30), '+', basename_call_result_64660, flavor_64661)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 30), tuple_64651, result_add_64662)
        # Adding element type (line 604)
        # Getting the type of 'src' (line 605)
        src_64663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 30), 'src')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 30), tuple_64651, src_64663)
        
        # Applying the binary operator '%' (line 604)
        result_mod_64664 = python_operator(stypy.reporting.localization.Localization(__file__, 604, 18), '%', str_64650, tuple_64651)
        
        # Assigning a type to the variable 'display' (line 604)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 8), 'display', result_mod_64664)
        
        
        # SSA begins for try-except statement (line 606)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 607)
        # Processing the call arguments (line 607)
        # Getting the type of 'command' (line 607)
        command_64667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 23), 'command', False)
        # Processing the call keyword arguments (line 607)
        # Getting the type of 'display' (line 607)
        display_64668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 40), 'display', False)
        keyword_64669 = display_64668
        kwargs_64670 = {'display': keyword_64669}
        # Getting the type of 'self' (line 607)
        self_64665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 12), 'self', False)
        # Obtaining the member 'spawn' of a type (line 607)
        spawn_64666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 12), self_64665, 'spawn')
        # Calling spawn(args, kwargs) (line 607)
        spawn_call_result_64671 = invoke(stypy.reporting.localization.Localization(__file__, 607, 12), spawn_64666, *[command_64667], **kwargs_64670)
        
        # SSA branch for the except part of a try statement (line 606)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 606)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 609):
        
        # Assigning a Call to a Name (line 609):
        
        # Call to str(...): (line 609)
        # Processing the call arguments (line 609)
        
        # Call to get_exception(...): (line 609)
        # Processing the call keyword arguments (line 609)
        kwargs_64674 = {}
        # Getting the type of 'get_exception' (line 609)
        get_exception_64673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 22), 'get_exception', False)
        # Calling get_exception(args, kwargs) (line 609)
        get_exception_call_result_64675 = invoke(stypy.reporting.localization.Localization(__file__, 609, 22), get_exception_64673, *[], **kwargs_64674)
        
        # Processing the call keyword arguments (line 609)
        kwargs_64676 = {}
        # Getting the type of 'str' (line 609)
        str_64672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 18), 'str', False)
        # Calling str(args, kwargs) (line 609)
        str_call_result_64677 = invoke(stypy.reporting.localization.Localization(__file__, 609, 18), str_64672, *[get_exception_call_result_64675], **kwargs_64676)
        
        # Assigning a type to the variable 'msg' (line 609)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 12), 'msg', str_call_result_64677)
        
        # Call to CompileError(...): (line 610)
        # Processing the call arguments (line 610)
        # Getting the type of 'msg' (line 610)
        msg_64679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 31), 'msg', False)
        # Processing the call keyword arguments (line 610)
        kwargs_64680 = {}
        # Getting the type of 'CompileError' (line 610)
        CompileError_64678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 18), 'CompileError', False)
        # Calling CompileError(args, kwargs) (line 610)
        CompileError_call_result_64681 = invoke(stypy.reporting.localization.Localization(__file__, 610, 18), CompileError_64678, *[msg_64679], **kwargs_64680)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 610, 12), CompileError_call_result_64681, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 606)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_compile' in the type store
        # Getting the type of 'stypy_return_type' (line 562)
        stypy_return_type_64682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64682)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_compile'
        return stypy_return_type_64682


    @norecursion
    def module_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'module_options'
        module_type_store = module_type_store.open_function_context('module_options', 612, 4, False)
        # Assigning a type to the variable 'self' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.module_options.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.module_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.module_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.module_options.__dict__.__setitem__('stypy_function_name', 'FCompiler.module_options')
        FCompiler.module_options.__dict__.__setitem__('stypy_param_names_list', ['module_dirs', 'module_build_dir'])
        FCompiler.module_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.module_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.module_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.module_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.module_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.module_options.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.module_options', ['module_dirs', 'module_build_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'module_options', localization, ['module_dirs', 'module_build_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'module_options(...)' code ##################

        
        # Assigning a List to a Name (line 613):
        
        # Assigning a List to a Name (line 613):
        
        # Obtaining an instance of the builtin type 'list' (line 613)
        list_64683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 613)
        
        # Assigning a type to the variable 'options' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'options', list_64683)
        
        
        # Getting the type of 'self' (line 614)
        self_64684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 11), 'self')
        # Obtaining the member 'module_dir_switch' of a type (line 614)
        module_dir_switch_64685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 11), self_64684, 'module_dir_switch')
        # Getting the type of 'None' (line 614)
        None_64686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 41), 'None')
        # Applying the binary operator 'isnot' (line 614)
        result_is_not_64687 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 11), 'isnot', module_dir_switch_64685, None_64686)
        
        # Testing the type of an if condition (line 614)
        if_condition_64688 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 614, 8), result_is_not_64687)
        # Assigning a type to the variable 'if_condition_64688' (line 614)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 8), 'if_condition_64688', if_condition_64688)
        # SSA begins for if statement (line 614)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Obtaining the type of the subscript
        int_64689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 38), 'int')
        # Getting the type of 'self' (line 615)
        self_64690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 15), 'self')
        # Obtaining the member 'module_dir_switch' of a type (line 615)
        module_dir_switch_64691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 15), self_64690, 'module_dir_switch')
        # Obtaining the member '__getitem__' of a type (line 615)
        getitem___64692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 15), module_dir_switch_64691, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 615)
        subscript_call_result_64693 = invoke(stypy.reporting.localization.Localization(__file__, 615, 15), getitem___64692, int_64689)
        
        str_64694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 43), 'str', ' ')
        # Applying the binary operator '==' (line 615)
        result_eq_64695 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 15), '==', subscript_call_result_64693, str_64694)
        
        # Testing the type of an if condition (line 615)
        if_condition_64696 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 615, 12), result_eq_64695)
        # Assigning a type to the variable 'if_condition_64696' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 12), 'if_condition_64696', if_condition_64696)
        # SSA begins for if statement (line 615)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 616)
        # Processing the call arguments (line 616)
        
        # Obtaining an instance of the builtin type 'list' (line 616)
        list_64699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 616, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 616)
        # Adding element type (line 616)
        
        # Call to strip(...): (line 616)
        # Processing the call keyword arguments (line 616)
        kwargs_64703 = {}
        # Getting the type of 'self' (line 616)
        self_64700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 32), 'self', False)
        # Obtaining the member 'module_dir_switch' of a type (line 616)
        module_dir_switch_64701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 32), self_64700, 'module_dir_switch')
        # Obtaining the member 'strip' of a type (line 616)
        strip_64702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 32), module_dir_switch_64701, 'strip')
        # Calling strip(args, kwargs) (line 616)
        strip_call_result_64704 = invoke(stypy.reporting.localization.Localization(__file__, 616, 32), strip_64702, *[], **kwargs_64703)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 31), list_64699, strip_call_result_64704)
        # Adding element type (line 616)
        # Getting the type of 'module_build_dir' (line 616)
        module_build_dir_64705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 64), 'module_build_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 616, 31), list_64699, module_build_dir_64705)
        
        # Processing the call keyword arguments (line 616)
        kwargs_64706 = {}
        # Getting the type of 'options' (line 616)
        options_64697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 16), 'options', False)
        # Obtaining the member 'extend' of a type (line 616)
        extend_64698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 16), options_64697, 'extend')
        # Calling extend(args, kwargs) (line 616)
        extend_call_result_64707 = invoke(stypy.reporting.localization.Localization(__file__, 616, 16), extend_64698, *[list_64699], **kwargs_64706)
        
        # SSA branch for the else part of an if statement (line 615)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 618)
        # Processing the call arguments (line 618)
        
        # Call to strip(...): (line 618)
        # Processing the call keyword arguments (line 618)
        kwargs_64713 = {}
        # Getting the type of 'self' (line 618)
        self_64710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 31), 'self', False)
        # Obtaining the member 'module_dir_switch' of a type (line 618)
        module_dir_switch_64711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 31), self_64710, 'module_dir_switch')
        # Obtaining the member 'strip' of a type (line 618)
        strip_64712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 31), module_dir_switch_64711, 'strip')
        # Calling strip(args, kwargs) (line 618)
        strip_call_result_64714 = invoke(stypy.reporting.localization.Localization(__file__, 618, 31), strip_64712, *[], **kwargs_64713)
        
        # Getting the type of 'module_build_dir' (line 618)
        module_build_dir_64715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 62), 'module_build_dir', False)
        # Applying the binary operator '+' (line 618)
        result_add_64716 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 31), '+', strip_call_result_64714, module_build_dir_64715)
        
        # Processing the call keyword arguments (line 618)
        kwargs_64717 = {}
        # Getting the type of 'options' (line 618)
        options_64708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 16), 'options', False)
        # Obtaining the member 'append' of a type (line 618)
        append_64709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 16), options_64708, 'append')
        # Calling append(args, kwargs) (line 618)
        append_call_result_64718 = invoke(stypy.reporting.localization.Localization(__file__, 618, 16), append_64709, *[result_add_64716], **kwargs_64717)
        
        # SSA join for if statement (line 615)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 614)
        module_type_store.open_ssa_branch('else')
        
        # Call to print(...): (line 620)
        # Processing the call arguments (line 620)
        str_64720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 620, 18), 'str', 'XXX: module_build_dir=%r option ignored')
        # Getting the type of 'module_build_dir' (line 620)
        module_build_dir_64721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 63), 'module_build_dir', False)
        # Applying the binary operator '%' (line 620)
        result_mod_64722 = python_operator(stypy.reporting.localization.Localization(__file__, 620, 18), '%', str_64720, module_build_dir_64721)
        
        # Processing the call keyword arguments (line 620)
        kwargs_64723 = {}
        # Getting the type of 'print' (line 620)
        print_64719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 12), 'print', False)
        # Calling print(args, kwargs) (line 620)
        print_call_result_64724 = invoke(stypy.reporting.localization.Localization(__file__, 620, 12), print_64719, *[result_mod_64722], **kwargs_64723)
        
        
        # Call to print(...): (line 621)
        # Processing the call arguments (line 621)
        str_64726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 18), 'str', 'XXX: Fix module_dir_switch for ')
        # Getting the type of 'self' (line 621)
        self_64727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 53), 'self', False)
        # Obtaining the member '__class__' of a type (line 621)
        class___64728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 53), self_64727, '__class__')
        # Obtaining the member '__name__' of a type (line 621)
        name___64729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 53), class___64728, '__name__')
        # Processing the call keyword arguments (line 621)
        kwargs_64730 = {}
        # Getting the type of 'print' (line 621)
        print_64725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 12), 'print', False)
        # Calling print(args, kwargs) (line 621)
        print_call_result_64731 = invoke(stypy.reporting.localization.Localization(__file__, 621, 12), print_64725, *[str_64726, name___64729], **kwargs_64730)
        
        # SSA join for if statement (line 614)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 622)
        self_64732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 11), 'self')
        # Obtaining the member 'module_include_switch' of a type (line 622)
        module_include_switch_64733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 11), self_64732, 'module_include_switch')
        # Getting the type of 'None' (line 622)
        None_64734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 45), 'None')
        # Applying the binary operator 'isnot' (line 622)
        result_is_not_64735 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 11), 'isnot', module_include_switch_64733, None_64734)
        
        # Testing the type of an if condition (line 622)
        if_condition_64736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 622, 8), result_is_not_64735)
        # Assigning a type to the variable 'if_condition_64736' (line 622)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 622, 8), 'if_condition_64736', if_condition_64736)
        # SSA begins for if statement (line 622)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Obtaining an instance of the builtin type 'list' (line 623)
        list_64737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 623)
        # Adding element type (line 623)
        # Getting the type of 'module_build_dir' (line 623)
        module_build_dir_64738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 22), 'module_build_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 623, 21), list_64737, module_build_dir_64738)
        
        # Getting the type of 'module_dirs' (line 623)
        module_dirs_64739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 40), 'module_dirs')
        # Applying the binary operator '+' (line 623)
        result_add_64740 = python_operator(stypy.reporting.localization.Localization(__file__, 623, 21), '+', list_64737, module_dirs_64739)
        
        # Testing the type of a for loop iterable (line 623)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 623, 12), result_add_64740)
        # Getting the type of the for loop variable (line 623)
        for_loop_var_64741 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 623, 12), result_add_64740)
        # Assigning a type to the variable 'd' (line 623)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 12), 'd', for_loop_var_64741)
        # SSA begins for a for statement (line 623)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 624)
        # Processing the call arguments (line 624)
        str_64744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 31), 'str', '%s%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 624)
        tuple_64745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 624)
        # Adding element type (line 624)
        # Getting the type of 'self' (line 624)
        self_64746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 41), 'self', False)
        # Obtaining the member 'module_include_switch' of a type (line 624)
        module_include_switch_64747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 41), self_64746, 'module_include_switch')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 41), tuple_64745, module_include_switch_64747)
        # Adding element type (line 624)
        # Getting the type of 'd' (line 624)
        d_64748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 69), 'd', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 624, 41), tuple_64745, d_64748)
        
        # Applying the binary operator '%' (line 624)
        result_mod_64749 = python_operator(stypy.reporting.localization.Localization(__file__, 624, 31), '%', str_64744, tuple_64745)
        
        # Processing the call keyword arguments (line 624)
        kwargs_64750 = {}
        # Getting the type of 'options' (line 624)
        options_64742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 16), 'options', False)
        # Obtaining the member 'append' of a type (line 624)
        append_64743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 16), options_64742, 'append')
        # Calling append(args, kwargs) (line 624)
        append_call_result_64751 = invoke(stypy.reporting.localization.Localization(__file__, 624, 16), append_64743, *[result_mod_64749], **kwargs_64750)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 622)
        module_type_store.open_ssa_branch('else')
        
        # Call to print(...): (line 626)
        # Processing the call arguments (line 626)
        str_64753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 626, 18), 'str', 'XXX: module_dirs=%r option ignored')
        # Getting the type of 'module_dirs' (line 626)
        module_dirs_64754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 58), 'module_dirs', False)
        # Applying the binary operator '%' (line 626)
        result_mod_64755 = python_operator(stypy.reporting.localization.Localization(__file__, 626, 18), '%', str_64753, module_dirs_64754)
        
        # Processing the call keyword arguments (line 626)
        kwargs_64756 = {}
        # Getting the type of 'print' (line 626)
        print_64752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 12), 'print', False)
        # Calling print(args, kwargs) (line 626)
        print_call_result_64757 = invoke(stypy.reporting.localization.Localization(__file__, 626, 12), print_64752, *[result_mod_64755], **kwargs_64756)
        
        
        # Call to print(...): (line 627)
        # Processing the call arguments (line 627)
        str_64759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 18), 'str', 'XXX: Fix module_include_switch for ')
        # Getting the type of 'self' (line 627)
        self_64760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 57), 'self', False)
        # Obtaining the member '__class__' of a type (line 627)
        class___64761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 57), self_64760, '__class__')
        # Obtaining the member '__name__' of a type (line 627)
        name___64762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 57), class___64761, '__name__')
        # Processing the call keyword arguments (line 627)
        kwargs_64763 = {}
        # Getting the type of 'print' (line 627)
        print_64758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 12), 'print', False)
        # Calling print(args, kwargs) (line 627)
        print_call_result_64764 = invoke(stypy.reporting.localization.Localization(__file__, 627, 12), print_64758, *[str_64759, name___64762], **kwargs_64763)
        
        # SSA join for if statement (line 622)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'options' (line 628)
        options_64765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 15), 'options')
        # Assigning a type to the variable 'stypy_return_type' (line 628)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 8), 'stypy_return_type', options_64765)
        
        # ################# End of 'module_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'module_options' in the type store
        # Getting the type of 'stypy_return_type' (line 612)
        stypy_return_type_64766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64766)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'module_options'
        return stypy_return_type_64766


    @norecursion
    def library_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'library_option'
        module_type_store = module_type_store.open_function_context('library_option', 630, 4, False)
        # Assigning a type to the variable 'self' (line 631)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.library_option.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.library_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.library_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.library_option.__dict__.__setitem__('stypy_function_name', 'FCompiler.library_option')
        FCompiler.library_option.__dict__.__setitem__('stypy_param_names_list', ['lib'])
        FCompiler.library_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.library_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.library_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.library_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.library_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.library_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.library_option', ['lib'], None, None, defaults, varargs, kwargs)

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

        str_64767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 15), 'str', '-l')
        # Getting the type of 'lib' (line 631)
        lib_64768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 22), 'lib')
        # Applying the binary operator '+' (line 631)
        result_add_64769 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 15), '+', str_64767, lib_64768)
        
        # Assigning a type to the variable 'stypy_return_type' (line 631)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'stypy_return_type', result_add_64769)
        
        # ################# End of 'library_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'library_option' in the type store
        # Getting the type of 'stypy_return_type' (line 630)
        stypy_return_type_64770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64770)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'library_option'
        return stypy_return_type_64770


    @norecursion
    def library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'library_dir_option'
        module_type_store = module_type_store.open_function_context('library_dir_option', 632, 4, False)
        # Assigning a type to the variable 'self' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.library_dir_option.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.library_dir_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.library_dir_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.library_dir_option.__dict__.__setitem__('stypy_function_name', 'FCompiler.library_dir_option')
        FCompiler.library_dir_option.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        FCompiler.library_dir_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.library_dir_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.library_dir_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.library_dir_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.library_dir_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.library_dir_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.library_dir_option', ['dir'], None, None, defaults, varargs, kwargs)

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

        str_64771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 15), 'str', '-L')
        # Getting the type of 'dir' (line 633)
        dir_64772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 22), 'dir')
        # Applying the binary operator '+' (line 633)
        result_add_64773 = python_operator(stypy.reporting.localization.Localization(__file__, 633, 15), '+', str_64771, dir_64772)
        
        # Assigning a type to the variable 'stypy_return_type' (line 633)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'stypy_return_type', result_add_64773)
        
        # ################# End of 'library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 632)
        stypy_return_type_64774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64774)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'library_dir_option'
        return stypy_return_type_64774


    @norecursion
    def link(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 636)
        None_64775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 41), 'None')
        # Getting the type of 'None' (line 636)
        None_64776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 57), 'None')
        # Getting the type of 'None' (line 637)
        None_64777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 26), 'None')
        # Getting the type of 'None' (line 637)
        None_64778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 53), 'None')
        # Getting the type of 'None' (line 638)
        None_64779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 28), 'None')
        int_64780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 638, 40), 'int')
        # Getting the type of 'None' (line 638)
        None_64781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 57), 'None')
        # Getting the type of 'None' (line 639)
        None_64782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 28), 'None')
        # Getting the type of 'None' (line 639)
        None_64783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 45), 'None')
        # Getting the type of 'None' (line 639)
        None_64784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 63), 'None')
        defaults = [None_64775, None_64776, None_64777, None_64778, None_64779, int_64780, None_64781, None_64782, None_64783, None_64784]
        # Create a new context for function 'link'
        module_type_store = module_type_store.open_function_context('link', 635, 4, False)
        # Assigning a type to the variable 'self' (line 636)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler.link.__dict__.__setitem__('stypy_localization', localization)
        FCompiler.link.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler.link.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler.link.__dict__.__setitem__('stypy_function_name', 'FCompiler.link')
        FCompiler.link.__dict__.__setitem__('stypy_param_names_list', ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'])
        FCompiler.link.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler.link.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler.link.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler.link.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler.link.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler.link.__dict__.__setitem__('stypy_declared_arg_number', 14)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler.link', ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Tuple (line 640):
        
        # Assigning a Call to a Name:
        
        # Call to _fix_object_args(...): (line 640)
        # Processing the call arguments (line 640)
        # Getting the type of 'objects' (line 640)
        objects_64787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 52), 'objects', False)
        # Getting the type of 'output_dir' (line 640)
        output_dir_64788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 61), 'output_dir', False)
        # Processing the call keyword arguments (line 640)
        kwargs_64789 = {}
        # Getting the type of 'self' (line 640)
        self_64785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 30), 'self', False)
        # Obtaining the member '_fix_object_args' of a type (line 640)
        _fix_object_args_64786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 30), self_64785, '_fix_object_args')
        # Calling _fix_object_args(args, kwargs) (line 640)
        _fix_object_args_call_result_64790 = invoke(stypy.reporting.localization.Localization(__file__, 640, 30), _fix_object_args_64786, *[objects_64787, output_dir_64788], **kwargs_64789)
        
        # Assigning a type to the variable 'call_assignment_63484' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'call_assignment_63484', _fix_object_args_call_result_64790)
        
        # Assigning a Call to a Name (line 640):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_64793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 8), 'int')
        # Processing the call keyword arguments
        kwargs_64794 = {}
        # Getting the type of 'call_assignment_63484' (line 640)
        call_assignment_63484_64791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'call_assignment_63484', False)
        # Obtaining the member '__getitem__' of a type (line 640)
        getitem___64792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 8), call_assignment_63484_64791, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_64795 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___64792, *[int_64793], **kwargs_64794)
        
        # Assigning a type to the variable 'call_assignment_63485' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'call_assignment_63485', getitem___call_result_64795)
        
        # Assigning a Name to a Name (line 640):
        # Getting the type of 'call_assignment_63485' (line 640)
        call_assignment_63485_64796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'call_assignment_63485')
        # Assigning a type to the variable 'objects' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'objects', call_assignment_63485_64796)
        
        # Assigning a Call to a Name (line 640):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_64799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 8), 'int')
        # Processing the call keyword arguments
        kwargs_64800 = {}
        # Getting the type of 'call_assignment_63484' (line 640)
        call_assignment_63484_64797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'call_assignment_63484', False)
        # Obtaining the member '__getitem__' of a type (line 640)
        getitem___64798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 8), call_assignment_63484_64797, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_64801 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___64798, *[int_64799], **kwargs_64800)
        
        # Assigning a type to the variable 'call_assignment_63486' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'call_assignment_63486', getitem___call_result_64801)
        
        # Assigning a Name to a Name (line 640):
        # Getting the type of 'call_assignment_63486' (line 640)
        call_assignment_63486_64802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'call_assignment_63486')
        # Assigning a type to the variable 'output_dir' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 17), 'output_dir', call_assignment_63486_64802)
        
        # Assigning a Call to a Tuple (line 641):
        
        # Assigning a Call to a Name:
        
        # Call to _fix_lib_args(...): (line 642)
        # Processing the call arguments (line 642)
        # Getting the type of 'libraries' (line 642)
        libraries_64805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 31), 'libraries', False)
        # Getting the type of 'library_dirs' (line 642)
        library_dirs_64806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 42), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 642)
        runtime_library_dirs_64807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 56), 'runtime_library_dirs', False)
        # Processing the call keyword arguments (line 642)
        kwargs_64808 = {}
        # Getting the type of 'self' (line 642)
        self_64803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 12), 'self', False)
        # Obtaining the member '_fix_lib_args' of a type (line 642)
        _fix_lib_args_64804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 12), self_64803, '_fix_lib_args')
        # Calling _fix_lib_args(args, kwargs) (line 642)
        _fix_lib_args_call_result_64809 = invoke(stypy.reporting.localization.Localization(__file__, 642, 12), _fix_lib_args_64804, *[libraries_64805, library_dirs_64806, runtime_library_dirs_64807], **kwargs_64808)
        
        # Assigning a type to the variable 'call_assignment_63487' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'call_assignment_63487', _fix_lib_args_call_result_64809)
        
        # Assigning a Call to a Name (line 641):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_64812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 8), 'int')
        # Processing the call keyword arguments
        kwargs_64813 = {}
        # Getting the type of 'call_assignment_63487' (line 641)
        call_assignment_63487_64810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'call_assignment_63487', False)
        # Obtaining the member '__getitem__' of a type (line 641)
        getitem___64811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 8), call_assignment_63487_64810, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_64814 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___64811, *[int_64812], **kwargs_64813)
        
        # Assigning a type to the variable 'call_assignment_63488' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'call_assignment_63488', getitem___call_result_64814)
        
        # Assigning a Name to a Name (line 641):
        # Getting the type of 'call_assignment_63488' (line 641)
        call_assignment_63488_64815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'call_assignment_63488')
        # Assigning a type to the variable 'libraries' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'libraries', call_assignment_63488_64815)
        
        # Assigning a Call to a Name (line 641):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_64818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 8), 'int')
        # Processing the call keyword arguments
        kwargs_64819 = {}
        # Getting the type of 'call_assignment_63487' (line 641)
        call_assignment_63487_64816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'call_assignment_63487', False)
        # Obtaining the member '__getitem__' of a type (line 641)
        getitem___64817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 8), call_assignment_63487_64816, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_64820 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___64817, *[int_64818], **kwargs_64819)
        
        # Assigning a type to the variable 'call_assignment_63489' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'call_assignment_63489', getitem___call_result_64820)
        
        # Assigning a Name to a Name (line 641):
        # Getting the type of 'call_assignment_63489' (line 641)
        call_assignment_63489_64821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'call_assignment_63489')
        # Assigning a type to the variable 'library_dirs' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 19), 'library_dirs', call_assignment_63489_64821)
        
        # Assigning a Call to a Name (line 641):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_64824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 8), 'int')
        # Processing the call keyword arguments
        kwargs_64825 = {}
        # Getting the type of 'call_assignment_63487' (line 641)
        call_assignment_63487_64822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'call_assignment_63487', False)
        # Obtaining the member '__getitem__' of a type (line 641)
        getitem___64823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 8), call_assignment_63487_64822, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_64826 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___64823, *[int_64824], **kwargs_64825)
        
        # Assigning a type to the variable 'call_assignment_63490' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'call_assignment_63490', getitem___call_result_64826)
        
        # Assigning a Name to a Name (line 641):
        # Getting the type of 'call_assignment_63490' (line 641)
        call_assignment_63490_64827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'call_assignment_63490')
        # Assigning a type to the variable 'runtime_library_dirs' (line 641)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 33), 'runtime_library_dirs', call_assignment_63490_64827)
        
        # Assigning a Call to a Name (line 644):
        
        # Assigning a Call to a Name (line 644):
        
        # Call to gen_lib_options(...): (line 644)
        # Processing the call arguments (line 644)
        # Getting the type of 'self' (line 644)
        self_64829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 35), 'self', False)
        # Getting the type of 'library_dirs' (line 644)
        library_dirs_64830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 41), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 644)
        runtime_library_dirs_64831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 55), 'runtime_library_dirs', False)
        # Getting the type of 'libraries' (line 645)
        libraries_64832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 35), 'libraries', False)
        # Processing the call keyword arguments (line 644)
        kwargs_64833 = {}
        # Getting the type of 'gen_lib_options' (line 644)
        gen_lib_options_64828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 19), 'gen_lib_options', False)
        # Calling gen_lib_options(args, kwargs) (line 644)
        gen_lib_options_call_result_64834 = invoke(stypy.reporting.localization.Localization(__file__, 644, 19), gen_lib_options_64828, *[self_64829, library_dirs_64830, runtime_library_dirs_64831, libraries_64832], **kwargs_64833)
        
        # Assigning a type to the variable 'lib_opts' (line 644)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'lib_opts', gen_lib_options_call_result_64834)
        
        
        # Call to is_string(...): (line 646)
        # Processing the call arguments (line 646)
        # Getting the type of 'output_dir' (line 646)
        output_dir_64836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 21), 'output_dir', False)
        # Processing the call keyword arguments (line 646)
        kwargs_64837 = {}
        # Getting the type of 'is_string' (line 646)
        is_string_64835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 11), 'is_string', False)
        # Calling is_string(args, kwargs) (line 646)
        is_string_call_result_64838 = invoke(stypy.reporting.localization.Localization(__file__, 646, 11), is_string_64835, *[output_dir_64836], **kwargs_64837)
        
        # Testing the type of an if condition (line 646)
        if_condition_64839 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 646, 8), is_string_call_result_64838)
        # Assigning a type to the variable 'if_condition_64839' (line 646)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'if_condition_64839', if_condition_64839)
        # SSA begins for if statement (line 646)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 647):
        
        # Assigning a Call to a Name (line 647):
        
        # Call to join(...): (line 647)
        # Processing the call arguments (line 647)
        # Getting the type of 'output_dir' (line 647)
        output_dir_64843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 43), 'output_dir', False)
        # Getting the type of 'output_filename' (line 647)
        output_filename_64844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 55), 'output_filename', False)
        # Processing the call keyword arguments (line 647)
        kwargs_64845 = {}
        # Getting the type of 'os' (line 647)
        os_64840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 647)
        path_64841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 30), os_64840, 'path')
        # Obtaining the member 'join' of a type (line 647)
        join_64842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 30), path_64841, 'join')
        # Calling join(args, kwargs) (line 647)
        join_call_result_64846 = invoke(stypy.reporting.localization.Localization(__file__, 647, 30), join_64842, *[output_dir_64843, output_filename_64844], **kwargs_64845)
        
        # Assigning a type to the variable 'output_filename' (line 647)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 12), 'output_filename', join_call_result_64846)
        # SSA branch for the else part of an if statement (line 646)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 648)
        # Getting the type of 'output_dir' (line 648)
        output_dir_64847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 13), 'output_dir')
        # Getting the type of 'None' (line 648)
        None_64848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 31), 'None')
        
        (may_be_64849, more_types_in_union_64850) = may_not_be_none(output_dir_64847, None_64848)

        if may_be_64849:

            if more_types_in_union_64850:
                # Runtime conditional SSA (line 648)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to TypeError(...): (line 649)
            # Processing the call arguments (line 649)
            str_64852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 28), 'str', "'output_dir' must be a string or None")
            # Processing the call keyword arguments (line 649)
            kwargs_64853 = {}
            # Getting the type of 'TypeError' (line 649)
            TypeError_64851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 649)
            TypeError_call_result_64854 = invoke(stypy.reporting.localization.Localization(__file__, 649, 18), TypeError_64851, *[str_64852], **kwargs_64853)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 649, 12), TypeError_call_result_64854, 'raise parameter', BaseException)

            if more_types_in_union_64850:
                # SSA join for if statement (line 648)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 646)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to _need_link(...): (line 651)
        # Processing the call arguments (line 651)
        # Getting the type of 'objects' (line 651)
        objects_64857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 27), 'objects', False)
        # Getting the type of 'output_filename' (line 651)
        output_filename_64858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 36), 'output_filename', False)
        # Processing the call keyword arguments (line 651)
        kwargs_64859 = {}
        # Getting the type of 'self' (line 651)
        self_64855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 11), 'self', False)
        # Obtaining the member '_need_link' of a type (line 651)
        _need_link_64856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 11), self_64855, '_need_link')
        # Calling _need_link(args, kwargs) (line 651)
        _need_link_call_result_64860 = invoke(stypy.reporting.localization.Localization(__file__, 651, 11), _need_link_64856, *[objects_64857, output_filename_64858], **kwargs_64859)
        
        # Testing the type of an if condition (line 651)
        if_condition_64861 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 651, 8), _need_link_call_result_64860)
        # Assigning a type to the variable 'if_condition_64861' (line 651)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 651, 8), 'if_condition_64861', if_condition_64861)
        # SSA begins for if statement (line 651)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Obtaining the type of the subscript
        int_64862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 35), 'int')
        # Getting the type of 'self' (line 652)
        self_64863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 15), 'self')
        # Obtaining the member 'library_switch' of a type (line 652)
        library_switch_64864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 15), self_64863, 'library_switch')
        # Obtaining the member '__getitem__' of a type (line 652)
        getitem___64865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 15), library_switch_64864, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 652)
        subscript_call_result_64866 = invoke(stypy.reporting.localization.Localization(__file__, 652, 15), getitem___64865, int_64862)
        
        str_64867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 40), 'str', ' ')
        # Applying the binary operator '==' (line 652)
        result_eq_64868 = python_operator(stypy.reporting.localization.Localization(__file__, 652, 15), '==', subscript_call_result_64866, str_64867)
        
        # Testing the type of an if condition (line 652)
        if_condition_64869 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 652, 12), result_eq_64868)
        # Assigning a type to the variable 'if_condition_64869' (line 652)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 12), 'if_condition_64869', if_condition_64869)
        # SSA begins for if statement (line 652)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 653):
        
        # Assigning a List to a Name (line 653):
        
        # Obtaining an instance of the builtin type 'list' (line 653)
        list_64870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 653)
        # Adding element type (line 653)
        
        # Call to strip(...): (line 653)
        # Processing the call keyword arguments (line 653)
        kwargs_64874 = {}
        # Getting the type of 'self' (line 653)
        self_64871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 26), 'self', False)
        # Obtaining the member 'library_switch' of a type (line 653)
        library_switch_64872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 26), self_64871, 'library_switch')
        # Obtaining the member 'strip' of a type (line 653)
        strip_64873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 26), library_switch_64872, 'strip')
        # Calling strip(args, kwargs) (line 653)
        strip_call_result_64875 = invoke(stypy.reporting.localization.Localization(__file__, 653, 26), strip_64873, *[], **kwargs_64874)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 653, 25), list_64870, strip_call_result_64875)
        # Adding element type (line 653)
        # Getting the type of 'output_filename' (line 653)
        output_filename_64876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 55), 'output_filename')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 653, 25), list_64870, output_filename_64876)
        
        # Assigning a type to the variable 'o_args' (line 653)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 16), 'o_args', list_64870)
        # SSA branch for the else part of an if statement (line 652)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 655):
        
        # Assigning a List to a Name (line 655):
        
        # Obtaining an instance of the builtin type 'list' (line 655)
        list_64877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 655, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 655)
        # Adding element type (line 655)
        
        # Call to strip(...): (line 655)
        # Processing the call keyword arguments (line 655)
        kwargs_64881 = {}
        # Getting the type of 'self' (line 655)
        self_64878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 26), 'self', False)
        # Obtaining the member 'library_switch' of a type (line 655)
        library_switch_64879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 26), self_64878, 'library_switch')
        # Obtaining the member 'strip' of a type (line 655)
        strip_64880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 655, 26), library_switch_64879, 'strip')
        # Calling strip(args, kwargs) (line 655)
        strip_call_result_64882 = invoke(stypy.reporting.localization.Localization(__file__, 655, 26), strip_64880, *[], **kwargs_64881)
        
        # Getting the type of 'output_filename' (line 655)
        output_filename_64883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 655, 54), 'output_filename')
        # Applying the binary operator '+' (line 655)
        result_add_64884 = python_operator(stypy.reporting.localization.Localization(__file__, 655, 26), '+', strip_call_result_64882, output_filename_64883)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 655, 25), list_64877, result_add_64884)
        
        # Assigning a type to the variable 'o_args' (line 655)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 655, 16), 'o_args', list_64877)
        # SSA join for if statement (line 652)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to is_string(...): (line 657)
        # Processing the call arguments (line 657)
        # Getting the type of 'self' (line 657)
        self_64886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 25), 'self', False)
        # Obtaining the member 'objects' of a type (line 657)
        objects_64887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 25), self_64886, 'objects')
        # Processing the call keyword arguments (line 657)
        kwargs_64888 = {}
        # Getting the type of 'is_string' (line 657)
        is_string_64885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 15), 'is_string', False)
        # Calling is_string(args, kwargs) (line 657)
        is_string_call_result_64889 = invoke(stypy.reporting.localization.Localization(__file__, 657, 15), is_string_64885, *[objects_64887], **kwargs_64888)
        
        # Testing the type of an if condition (line 657)
        if_condition_64890 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 657, 12), is_string_call_result_64889)
        # Assigning a type to the variable 'if_condition_64890' (line 657)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 12), 'if_condition_64890', if_condition_64890)
        # SSA begins for if statement (line 657)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 658):
        
        # Assigning a BinOp to a Name (line 658):
        # Getting the type of 'objects' (line 658)
        objects_64891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 26), 'objects')
        
        # Obtaining an instance of the builtin type 'list' (line 658)
        list_64892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 658, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 658)
        # Adding element type (line 658)
        # Getting the type of 'self' (line 658)
        self_64893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 37), 'self')
        # Obtaining the member 'objects' of a type (line 658)
        objects_64894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 658, 37), self_64893, 'objects')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 658, 36), list_64892, objects_64894)
        
        # Applying the binary operator '+' (line 658)
        result_add_64895 = python_operator(stypy.reporting.localization.Localization(__file__, 658, 26), '+', objects_64891, list_64892)
        
        # Assigning a type to the variable 'ld_args' (line 658)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 16), 'ld_args', result_add_64895)
        # SSA branch for the else part of an if statement (line 657)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 660):
        
        # Assigning a BinOp to a Name (line 660):
        # Getting the type of 'objects' (line 660)
        objects_64896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 26), 'objects')
        # Getting the type of 'self' (line 660)
        self_64897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 36), 'self')
        # Obtaining the member 'objects' of a type (line 660)
        objects_64898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 36), self_64897, 'objects')
        # Applying the binary operator '+' (line 660)
        result_add_64899 = python_operator(stypy.reporting.localization.Localization(__file__, 660, 26), '+', objects_64896, objects_64898)
        
        # Assigning a type to the variable 'ld_args' (line 660)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 16), 'ld_args', result_add_64899)
        # SSA join for if statement (line 657)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 661):
        
        # Assigning a BinOp to a Name (line 661):
        # Getting the type of 'ld_args' (line 661)
        ld_args_64900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 22), 'ld_args')
        # Getting the type of 'lib_opts' (line 661)
        lib_opts_64901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 32), 'lib_opts')
        # Applying the binary operator '+' (line 661)
        result_add_64902 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 22), '+', ld_args_64900, lib_opts_64901)
        
        # Getting the type of 'o_args' (line 661)
        o_args_64903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 43), 'o_args')
        # Applying the binary operator '+' (line 661)
        result_add_64904 = python_operator(stypy.reporting.localization.Localization(__file__, 661, 41), '+', result_add_64902, o_args_64903)
        
        # Assigning a type to the variable 'ld_args' (line 661)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'ld_args', result_add_64904)
        
        # Getting the type of 'debug' (line 662)
        debug_64905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 15), 'debug')
        # Testing the type of an if condition (line 662)
        if_condition_64906 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 662, 12), debug_64905)
        # Assigning a type to the variable 'if_condition_64906' (line 662)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 12), 'if_condition_64906', if_condition_64906)
        # SSA begins for if statement (line 662)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Subscript (line 663):
        
        # Assigning a List to a Subscript (line 663):
        
        # Obtaining an instance of the builtin type 'list' (line 663)
        list_64907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 663)
        # Adding element type (line 663)
        str_64908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 31), 'str', '-g')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 30), list_64907, str_64908)
        
        # Getting the type of 'ld_args' (line 663)
        ld_args_64909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 16), 'ld_args')
        int_64910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 663, 25), 'int')
        slice_64911 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 663, 16), None, int_64910, None)
        # Storing an element on a container (line 663)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 663, 16), ld_args_64909, (slice_64911, list_64907))
        # SSA join for if statement (line 662)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'extra_preargs' (line 664)
        extra_preargs_64912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 15), 'extra_preargs')
        # Testing the type of an if condition (line 664)
        if_condition_64913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 664, 12), extra_preargs_64912)
        # Assigning a type to the variable 'if_condition_64913' (line 664)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 12), 'if_condition_64913', if_condition_64913)
        # SSA begins for if statement (line 664)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 665):
        
        # Assigning a Name to a Subscript (line 665):
        # Getting the type of 'extra_preargs' (line 665)
        extra_preargs_64914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 30), 'extra_preargs')
        # Getting the type of 'ld_args' (line 665)
        ld_args_64915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 16), 'ld_args')
        int_64916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 25), 'int')
        slice_64917 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 665, 16), None, int_64916, None)
        # Storing an element on a container (line 665)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 665, 16), ld_args_64915, (slice_64917, extra_preargs_64914))
        # SSA join for if statement (line 664)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'extra_postargs' (line 666)
        extra_postargs_64918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 15), 'extra_postargs')
        # Testing the type of an if condition (line 666)
        if_condition_64919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 666, 12), extra_postargs_64918)
        # Assigning a type to the variable 'if_condition_64919' (line 666)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 12), 'if_condition_64919', if_condition_64919)
        # SSA begins for if statement (line 666)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 667)
        # Processing the call arguments (line 667)
        # Getting the type of 'extra_postargs' (line 667)
        extra_postargs_64922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 31), 'extra_postargs', False)
        # Processing the call keyword arguments (line 667)
        kwargs_64923 = {}
        # Getting the type of 'ld_args' (line 667)
        ld_args_64920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 16), 'ld_args', False)
        # Obtaining the member 'extend' of a type (line 667)
        extend_64921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 16), ld_args_64920, 'extend')
        # Calling extend(args, kwargs) (line 667)
        extend_call_result_64924 = invoke(stypy.reporting.localization.Localization(__file__, 667, 16), extend_64921, *[extra_postargs_64922], **kwargs_64923)
        
        # SSA join for if statement (line 666)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to mkpath(...): (line 668)
        # Processing the call arguments (line 668)
        
        # Call to dirname(...): (line 668)
        # Processing the call arguments (line 668)
        # Getting the type of 'output_filename' (line 668)
        output_filename_64930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 40), 'output_filename', False)
        # Processing the call keyword arguments (line 668)
        kwargs_64931 = {}
        # Getting the type of 'os' (line 668)
        os_64927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 668)
        path_64928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 24), os_64927, 'path')
        # Obtaining the member 'dirname' of a type (line 668)
        dirname_64929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 24), path_64928, 'dirname')
        # Calling dirname(args, kwargs) (line 668)
        dirname_call_result_64932 = invoke(stypy.reporting.localization.Localization(__file__, 668, 24), dirname_64929, *[output_filename_64930], **kwargs_64931)
        
        # Processing the call keyword arguments (line 668)
        kwargs_64933 = {}
        # Getting the type of 'self' (line 668)
        self_64925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 12), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 668)
        mkpath_64926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 12), self_64925, 'mkpath')
        # Calling mkpath(args, kwargs) (line 668)
        mkpath_call_result_64934 = invoke(stypy.reporting.localization.Localization(__file__, 668, 12), mkpath_64926, *[dirname_call_result_64932], **kwargs_64933)
        
        
        
        # Getting the type of 'target_desc' (line 669)
        target_desc_64935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 15), 'target_desc')
        # Getting the type of 'CCompiler' (line 669)
        CCompiler_64936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 30), 'CCompiler')
        # Obtaining the member 'EXECUTABLE' of a type (line 669)
        EXECUTABLE_64937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 669, 30), CCompiler_64936, 'EXECUTABLE')
        # Applying the binary operator '==' (line 669)
        result_eq_64938 = python_operator(stypy.reporting.localization.Localization(__file__, 669, 15), '==', target_desc_64935, EXECUTABLE_64937)
        
        # Testing the type of an if condition (line 669)
        if_condition_64939 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 669, 12), result_eq_64938)
        # Assigning a type to the variable 'if_condition_64939' (line 669)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 12), 'if_condition_64939', if_condition_64939)
        # SSA begins for if statement (line 669)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 670):
        
        # Assigning a Subscript to a Name (line 670):
        
        # Obtaining the type of the subscript
        slice_64940 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 670, 25), None, None, None)
        # Getting the type of 'self' (line 670)
        self_64941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 25), 'self')
        # Obtaining the member 'linker_exe' of a type (line 670)
        linker_exe_64942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 25), self_64941, 'linker_exe')
        # Obtaining the member '__getitem__' of a type (line 670)
        getitem___64943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 25), linker_exe_64942, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 670)
        subscript_call_result_64944 = invoke(stypy.reporting.localization.Localization(__file__, 670, 25), getitem___64943, slice_64940)
        
        # Assigning a type to the variable 'linker' (line 670)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 16), 'linker', subscript_call_result_64944)
        # SSA branch for the else part of an if statement (line 669)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 672):
        
        # Assigning a Subscript to a Name (line 672):
        
        # Obtaining the type of the subscript
        slice_64945 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 672, 25), None, None, None)
        # Getting the type of 'self' (line 672)
        self_64946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 25), 'self')
        # Obtaining the member 'linker_so' of a type (line 672)
        linker_so_64947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 25), self_64946, 'linker_so')
        # Obtaining the member '__getitem__' of a type (line 672)
        getitem___64948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 672, 25), linker_so_64947, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 672)
        subscript_call_result_64949 = invoke(stypy.reporting.localization.Localization(__file__, 672, 25), getitem___64948, slice_64945)
        
        # Assigning a type to the variable 'linker' (line 672)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 16), 'linker', subscript_call_result_64949)
        # SSA join for if statement (line 669)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 673):
        
        # Assigning a BinOp to a Name (line 673):
        # Getting the type of 'linker' (line 673)
        linker_64950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 22), 'linker')
        # Getting the type of 'ld_args' (line 673)
        ld_args_64951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 31), 'ld_args')
        # Applying the binary operator '+' (line 673)
        result_add_64952 = python_operator(stypy.reporting.localization.Localization(__file__, 673, 22), '+', linker_64950, ld_args_64951)
        
        # Assigning a type to the variable 'command' (line 673)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 12), 'command', result_add_64952)
        
        
        # SSA begins for try-except statement (line 674)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 675)
        # Processing the call arguments (line 675)
        # Getting the type of 'command' (line 675)
        command_64955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 27), 'command', False)
        # Processing the call keyword arguments (line 675)
        kwargs_64956 = {}
        # Getting the type of 'self' (line 675)
        self_64953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 675)
        spawn_64954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 675, 16), self_64953, 'spawn')
        # Calling spawn(args, kwargs) (line 675)
        spawn_call_result_64957 = invoke(stypy.reporting.localization.Localization(__file__, 675, 16), spawn_64954, *[command_64955], **kwargs_64956)
        
        # SSA branch for the except part of a try statement (line 674)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 674)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 677):
        
        # Assigning a Call to a Name (line 677):
        
        # Call to str(...): (line 677)
        # Processing the call arguments (line 677)
        
        # Call to get_exception(...): (line 677)
        # Processing the call keyword arguments (line 677)
        kwargs_64960 = {}
        # Getting the type of 'get_exception' (line 677)
        get_exception_64959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 26), 'get_exception', False)
        # Calling get_exception(args, kwargs) (line 677)
        get_exception_call_result_64961 = invoke(stypy.reporting.localization.Localization(__file__, 677, 26), get_exception_64959, *[], **kwargs_64960)
        
        # Processing the call keyword arguments (line 677)
        kwargs_64962 = {}
        # Getting the type of 'str' (line 677)
        str_64958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 22), 'str', False)
        # Calling str(args, kwargs) (line 677)
        str_call_result_64963 = invoke(stypy.reporting.localization.Localization(__file__, 677, 22), str_64958, *[get_exception_call_result_64961], **kwargs_64962)
        
        # Assigning a type to the variable 'msg' (line 677)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 16), 'msg', str_call_result_64963)
        
        # Call to LinkError(...): (line 678)
        # Processing the call arguments (line 678)
        # Getting the type of 'msg' (line 678)
        msg_64965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 32), 'msg', False)
        # Processing the call keyword arguments (line 678)
        kwargs_64966 = {}
        # Getting the type of 'LinkError' (line 678)
        LinkError_64964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 22), 'LinkError', False)
        # Calling LinkError(args, kwargs) (line 678)
        LinkError_call_result_64967 = invoke(stypy.reporting.localization.Localization(__file__, 678, 22), LinkError_64964, *[msg_64965], **kwargs_64966)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 678, 16), LinkError_call_result_64967, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 674)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 651)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug(...): (line 680)
        # Processing the call arguments (line 680)
        str_64970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 680, 22), 'str', 'skipping %s (up-to-date)')
        # Getting the type of 'output_filename' (line 680)
        output_filename_64971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 50), 'output_filename', False)
        # Processing the call keyword arguments (line 680)
        kwargs_64972 = {}
        # Getting the type of 'log' (line 680)
        log_64968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 680)
        debug_64969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 680, 12), log_64968, 'debug')
        # Calling debug(args, kwargs) (line 680)
        debug_call_result_64973 = invoke(stypy.reporting.localization.Localization(__file__, 680, 12), debug_64969, *[str_64970, output_filename_64971], **kwargs_64972)
        
        # SSA join for if statement (line 651)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'link(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'link' in the type store
        # Getting the type of 'stypy_return_type' (line 635)
        stypy_return_type_64974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64974)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'link'
        return stypy_return_type_64974


    @norecursion
    def _environment_hook(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_environment_hook'
        module_type_store = module_type_store.open_function_context('_environment_hook', 682, 4, False)
        # Assigning a type to the variable 'self' (line 683)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FCompiler._environment_hook.__dict__.__setitem__('stypy_localization', localization)
        FCompiler._environment_hook.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FCompiler._environment_hook.__dict__.__setitem__('stypy_type_store', module_type_store)
        FCompiler._environment_hook.__dict__.__setitem__('stypy_function_name', 'FCompiler._environment_hook')
        FCompiler._environment_hook.__dict__.__setitem__('stypy_param_names_list', ['name', 'hook_name'])
        FCompiler._environment_hook.__dict__.__setitem__('stypy_varargs_param_name', None)
        FCompiler._environment_hook.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FCompiler._environment_hook.__dict__.__setitem__('stypy_call_defaults', defaults)
        FCompiler._environment_hook.__dict__.__setitem__('stypy_call_varargs', varargs)
        FCompiler._environment_hook.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FCompiler._environment_hook.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FCompiler._environment_hook', ['name', 'hook_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_environment_hook', localization, ['name', 'hook_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_environment_hook(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 683)
        # Getting the type of 'hook_name' (line 683)
        hook_name_64975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 11), 'hook_name')
        # Getting the type of 'None' (line 683)
        None_64976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 24), 'None')
        
        (may_be_64977, more_types_in_union_64978) = may_be_none(hook_name_64975, None_64976)

        if may_be_64977:

            if more_types_in_union_64978:
                # Runtime conditional SSA (line 683)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'None' (line 684)
            None_64979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 684)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 12), 'stypy_return_type', None_64979)

            if more_types_in_union_64978:
                # SSA join for if statement (line 683)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to is_string(...): (line 685)
        # Processing the call arguments (line 685)
        # Getting the type of 'hook_name' (line 685)
        hook_name_64981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 21), 'hook_name', False)
        # Processing the call keyword arguments (line 685)
        kwargs_64982 = {}
        # Getting the type of 'is_string' (line 685)
        is_string_64980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 11), 'is_string', False)
        # Calling is_string(args, kwargs) (line 685)
        is_string_call_result_64983 = invoke(stypy.reporting.localization.Localization(__file__, 685, 11), is_string_64980, *[hook_name_64981], **kwargs_64982)
        
        # Testing the type of an if condition (line 685)
        if_condition_64984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 685, 8), is_string_call_result_64983)
        # Assigning a type to the variable 'if_condition_64984' (line 685)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 8), 'if_condition_64984', if_condition_64984)
        # SSA begins for if statement (line 685)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to startswith(...): (line 686)
        # Processing the call arguments (line 686)
        str_64987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 36), 'str', 'self.')
        # Processing the call keyword arguments (line 686)
        kwargs_64988 = {}
        # Getting the type of 'hook_name' (line 686)
        hook_name_64985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 15), 'hook_name', False)
        # Obtaining the member 'startswith' of a type (line 686)
        startswith_64986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 686, 15), hook_name_64985, 'startswith')
        # Calling startswith(args, kwargs) (line 686)
        startswith_call_result_64989 = invoke(stypy.reporting.localization.Localization(__file__, 686, 15), startswith_64986, *[str_64987], **kwargs_64988)
        
        # Testing the type of an if condition (line 686)
        if_condition_64990 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 686, 12), startswith_call_result_64989)
        # Assigning a type to the variable 'if_condition_64990' (line 686)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 12), 'if_condition_64990', if_condition_64990)
        # SSA begins for if statement (line 686)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 687):
        
        # Assigning a Subscript to a Name (line 687):
        
        # Obtaining the type of the subscript
        int_64991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 38), 'int')
        slice_64992 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 687, 28), int_64991, None, None)
        # Getting the type of 'hook_name' (line 687)
        hook_name_64993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 28), 'hook_name')
        # Obtaining the member '__getitem__' of a type (line 687)
        getitem___64994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 687, 28), hook_name_64993, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 687)
        subscript_call_result_64995 = invoke(stypy.reporting.localization.Localization(__file__, 687, 28), getitem___64994, slice_64992)
        
        # Assigning a type to the variable 'hook_name' (line 687)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 16), 'hook_name', subscript_call_result_64995)
        
        # Assigning a Call to a Name (line 688):
        
        # Assigning a Call to a Name (line 688):
        
        # Call to getattr(...): (line 688)
        # Processing the call arguments (line 688)
        # Getting the type of 'self' (line 688)
        self_64997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 31), 'self', False)
        # Getting the type of 'hook_name' (line 688)
        hook_name_64998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 37), 'hook_name', False)
        # Processing the call keyword arguments (line 688)
        kwargs_64999 = {}
        # Getting the type of 'getattr' (line 688)
        getattr_64996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 23), 'getattr', False)
        # Calling getattr(args, kwargs) (line 688)
        getattr_call_result_65000 = invoke(stypy.reporting.localization.Localization(__file__, 688, 23), getattr_64996, *[self_64997, hook_name_64998], **kwargs_64999)
        
        # Assigning a type to the variable 'hook' (line 688)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 16), 'hook', getattr_call_result_65000)
        
        # Call to hook(...): (line 689)
        # Processing the call keyword arguments (line 689)
        kwargs_65002 = {}
        # Getting the type of 'hook' (line 689)
        hook_65001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 23), 'hook', False)
        # Calling hook(args, kwargs) (line 689)
        hook_call_result_65003 = invoke(stypy.reporting.localization.Localization(__file__, 689, 23), hook_65001, *[], **kwargs_65002)
        
        # Assigning a type to the variable 'stypy_return_type' (line 689)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 16), 'stypy_return_type', hook_call_result_65003)
        # SSA branch for the else part of an if statement (line 686)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to startswith(...): (line 690)
        # Processing the call arguments (line 690)
        str_65006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 38), 'str', 'exe.')
        # Processing the call keyword arguments (line 690)
        kwargs_65007 = {}
        # Getting the type of 'hook_name' (line 690)
        hook_name_65004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 17), 'hook_name', False)
        # Obtaining the member 'startswith' of a type (line 690)
        startswith_65005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 17), hook_name_65004, 'startswith')
        # Calling startswith(args, kwargs) (line 690)
        startswith_call_result_65008 = invoke(stypy.reporting.localization.Localization(__file__, 690, 17), startswith_65005, *[str_65006], **kwargs_65007)
        
        # Testing the type of an if condition (line 690)
        if_condition_65009 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 690, 17), startswith_call_result_65008)
        # Assigning a type to the variable 'if_condition_65009' (line 690)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 17), 'if_condition_65009', if_condition_65009)
        # SSA begins for if statement (line 690)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 691):
        
        # Assigning a Subscript to a Name (line 691):
        
        # Obtaining the type of the subscript
        int_65010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 38), 'int')
        slice_65011 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 691, 28), int_65010, None, None)
        # Getting the type of 'hook_name' (line 691)
        hook_name_65012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 28), 'hook_name')
        # Obtaining the member '__getitem__' of a type (line 691)
        getitem___65013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 28), hook_name_65012, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 691)
        subscript_call_result_65014 = invoke(stypy.reporting.localization.Localization(__file__, 691, 28), getitem___65013, slice_65011)
        
        # Assigning a type to the variable 'hook_name' (line 691)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 16), 'hook_name', subscript_call_result_65014)
        
        # Assigning a Subscript to a Name (line 692):
        
        # Assigning a Subscript to a Name (line 692):
        
        # Obtaining the type of the subscript
        # Getting the type of 'hook_name' (line 692)
        hook_name_65015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 39), 'hook_name')
        # Getting the type of 'self' (line 692)
        self_65016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 22), 'self')
        # Obtaining the member 'executables' of a type (line 692)
        executables_65017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 22), self_65016, 'executables')
        # Obtaining the member '__getitem__' of a type (line 692)
        getitem___65018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 692, 22), executables_65017, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 692)
        subscript_call_result_65019 = invoke(stypy.reporting.localization.Localization(__file__, 692, 22), getitem___65018, hook_name_65015)
        
        # Assigning a type to the variable 'var' (line 692)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 16), 'var', subscript_call_result_65019)
        
        # Getting the type of 'var' (line 693)
        var_65020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 19), 'var')
        # Testing the type of an if condition (line 693)
        if_condition_65021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 693, 16), var_65020)
        # Assigning a type to the variable 'if_condition_65021' (line 693)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 16), 'if_condition_65021', if_condition_65021)
        # SSA begins for if statement (line 693)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        int_65022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 31), 'int')
        # Getting the type of 'var' (line 694)
        var_65023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 27), 'var')
        # Obtaining the member '__getitem__' of a type (line 694)
        getitem___65024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 27), var_65023, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 694)
        subscript_call_result_65025 = invoke(stypy.reporting.localization.Localization(__file__, 694, 27), getitem___65024, int_65022)
        
        # Assigning a type to the variable 'stypy_return_type' (line 694)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 20), 'stypy_return_type', subscript_call_result_65025)
        # SSA branch for the else part of an if statement (line 693)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'None' (line 696)
        None_65026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 27), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 696)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 20), 'stypy_return_type', None_65026)
        # SSA join for if statement (line 693)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 690)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to startswith(...): (line 697)
        # Processing the call arguments (line 697)
        str_65029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 697, 38), 'str', 'flags.')
        # Processing the call keyword arguments (line 697)
        kwargs_65030 = {}
        # Getting the type of 'hook_name' (line 697)
        hook_name_65027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 17), 'hook_name', False)
        # Obtaining the member 'startswith' of a type (line 697)
        startswith_65028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 697, 17), hook_name_65027, 'startswith')
        # Calling startswith(args, kwargs) (line 697)
        startswith_call_result_65031 = invoke(stypy.reporting.localization.Localization(__file__, 697, 17), startswith_65028, *[str_65029], **kwargs_65030)
        
        # Testing the type of an if condition (line 697)
        if_condition_65032 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 697, 17), startswith_call_result_65031)
        # Assigning a type to the variable 'if_condition_65032' (line 697)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 17), 'if_condition_65032', if_condition_65032)
        # SSA begins for if statement (line 697)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 698):
        
        # Assigning a Subscript to a Name (line 698):
        
        # Obtaining the type of the subscript
        int_65033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 38), 'int')
        slice_65034 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 698, 28), int_65033, None, None)
        # Getting the type of 'hook_name' (line 698)
        hook_name_65035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 28), 'hook_name')
        # Obtaining the member '__getitem__' of a type (line 698)
        getitem___65036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 28), hook_name_65035, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 698)
        subscript_call_result_65037 = invoke(stypy.reporting.localization.Localization(__file__, 698, 28), getitem___65036, slice_65034)
        
        # Assigning a type to the variable 'hook_name' (line 698)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 16), 'hook_name', subscript_call_result_65037)
        
        # Assigning a Call to a Name (line 699):
        
        # Assigning a Call to a Name (line 699):
        
        # Call to getattr(...): (line 699)
        # Processing the call arguments (line 699)
        # Getting the type of 'self' (line 699)
        self_65039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 31), 'self', False)
        str_65040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 37), 'str', 'get_flags_')
        # Getting the type of 'hook_name' (line 699)
        hook_name_65041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 52), 'hook_name', False)
        # Applying the binary operator '+' (line 699)
        result_add_65042 = python_operator(stypy.reporting.localization.Localization(__file__, 699, 37), '+', str_65040, hook_name_65041)
        
        # Processing the call keyword arguments (line 699)
        kwargs_65043 = {}
        # Getting the type of 'getattr' (line 699)
        getattr_65038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 23), 'getattr', False)
        # Calling getattr(args, kwargs) (line 699)
        getattr_call_result_65044 = invoke(stypy.reporting.localization.Localization(__file__, 699, 23), getattr_65038, *[self_65039, result_add_65042], **kwargs_65043)
        
        # Assigning a type to the variable 'hook' (line 699)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 16), 'hook', getattr_call_result_65044)
        
        # Call to hook(...): (line 700)
        # Processing the call keyword arguments (line 700)
        kwargs_65046 = {}
        # Getting the type of 'hook' (line 700)
        hook_65045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 23), 'hook', False)
        # Calling hook(args, kwargs) (line 700)
        hook_call_result_65047 = invoke(stypy.reporting.localization.Localization(__file__, 700, 23), hook_65045, *[], **kwargs_65046)
        
        # Assigning a type to the variable 'stypy_return_type' (line 700)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 16), 'stypy_return_type', hook_call_result_65047)
        # SSA join for if statement (line 697)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 690)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 686)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 685)
        module_type_store.open_ssa_branch('else')
        
        # Call to hook_name(...): (line 702)
        # Processing the call keyword arguments (line 702)
        kwargs_65049 = {}
        # Getting the type of 'hook_name' (line 702)
        hook_name_65048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 19), 'hook_name', False)
        # Calling hook_name(args, kwargs) (line 702)
        hook_name_call_result_65050 = invoke(stypy.reporting.localization.Localization(__file__, 702, 19), hook_name_65048, *[], **kwargs_65049)
        
        # Assigning a type to the variable 'stypy_return_type' (line 702)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 12), 'stypy_return_type', hook_name_call_result_65050)
        # SSA join for if statement (line 685)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_environment_hook(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_environment_hook' in the type store
        # Getting the type of 'stypy_return_type' (line 682)
        stypy_return_type_65051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_65051)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_environment_hook'
        return stypy_return_type_65051


# Assigning a type to the variable 'FCompiler' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'FCompiler', FCompiler)

# Assigning a Call to a Name (line 106):

# Call to EnvironmentConfig(...): (line 106)
# Processing the call keyword arguments (line 106)
str_65053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 26), 'str', 'config_fc')
keyword_65054 = str_65053

# Obtaining an instance of the builtin type 'tuple' (line 108)
tuple_65055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 108)
# Adding element type (line 108)
# Getting the type of 'None' (line 108)
None_65056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 17), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 17), tuple_65055, None_65056)
# Adding element type (line 108)
# Getting the type of 'None' (line 108)
None_65057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 17), tuple_65055, None_65057)
# Adding element type (line 108)
str_65058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 29), 'str', 'noopt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 17), tuple_65055, str_65058)
# Adding element type (line 108)
# Getting the type of 'str2bool' (line 108)
str2bool_65059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 38), 'str2bool', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 17), tuple_65055, str2bool_65059)

keyword_65060 = tuple_65055

# Obtaining an instance of the builtin type 'tuple' (line 109)
tuple_65061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 109)
# Adding element type (line 109)
# Getting the type of 'None' (line 109)
None_65062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 18), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 18), tuple_65061, None_65062)
# Adding element type (line 109)
# Getting the type of 'None' (line 109)
None_65063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 24), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 18), tuple_65061, None_65063)
# Adding element type (line 109)
str_65064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 30), 'str', 'noarch')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 18), tuple_65061, str_65064)
# Adding element type (line 109)
# Getting the type of 'str2bool' (line 109)
str2bool_65065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 40), 'str2bool', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 18), tuple_65061, str2bool_65065)

keyword_65066 = tuple_65061

# Obtaining an instance of the builtin type 'tuple' (line 110)
tuple_65067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 110)
# Adding element type (line 110)
# Getting the type of 'None' (line 110)
None_65068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 17), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 17), tuple_65067, None_65068)
# Adding element type (line 110)
# Getting the type of 'None' (line 110)
None_65069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 17), tuple_65067, None_65069)
# Adding element type (line 110)
str_65070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 29), 'str', 'debug')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 17), tuple_65067, str_65070)
# Adding element type (line 110)
# Getting the type of 'str2bool' (line 110)
str2bool_65071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 38), 'str2bool', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 17), tuple_65067, str2bool_65071)

keyword_65072 = tuple_65067

# Obtaining an instance of the builtin type 'tuple' (line 111)
tuple_65073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 111)
# Adding element type (line 111)
# Getting the type of 'None' (line 111)
None_65074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 19), tuple_65073, None_65074)
# Adding element type (line 111)
# Getting the type of 'None' (line 111)
None_65075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 19), tuple_65073, None_65075)
# Adding element type (line 111)
str_65076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 31), 'str', 'verbose')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 19), tuple_65073, str_65076)
# Adding element type (line 111)
# Getting the type of 'str2bool' (line 111)
str2bool_65077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 42), 'str2bool', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 19), tuple_65073, str2bool_65077)

keyword_65078 = tuple_65073
kwargs_65079 = {'debug': keyword_65072, 'noopt': keyword_65060, 'distutils_section': keyword_65054, 'verbose': keyword_65078, 'noarch': keyword_65066}
# Getting the type of 'EnvironmentConfig' (line 106)
EnvironmentConfig_65052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 21), 'EnvironmentConfig', False)
# Calling EnvironmentConfig(args, kwargs) (line 106)
EnvironmentConfig_call_result_65080 = invoke(stypy.reporting.localization.Localization(__file__, 106, 21), EnvironmentConfig_65052, *[], **kwargs_65079)

# Getting the type of 'FCompiler'
FCompiler_65081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'distutils_vars' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65081, 'distutils_vars', EnvironmentConfig_call_result_65080)

# Assigning a Call to a Name (line 114):

# Call to EnvironmentConfig(...): (line 114)
# Processing the call keyword arguments (line 114)
str_65083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 26), 'str', 'config_fc')
keyword_65084 = str_65083

# Obtaining an instance of the builtin type 'tuple' (line 116)
tuple_65085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 116)
# Adding element type (line 116)
str_65086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 24), 'str', 'exe.compiler_f77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 24), tuple_65085, str_65086)
# Adding element type (line 116)
str_65087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 44), 'str', 'F77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 24), tuple_65085, str_65087)
# Adding element type (line 116)
str_65088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 51), 'str', 'f77exec')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 24), tuple_65085, str_65088)
# Adding element type (line 116)
# Getting the type of 'None' (line 116)
None_65089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 62), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 24), tuple_65085, None_65089)

keyword_65090 = tuple_65085

# Obtaining an instance of the builtin type 'tuple' (line 117)
tuple_65091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 117)
# Adding element type (line 117)
str_65092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 24), 'str', 'exe.compiler_f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 24), tuple_65091, str_65092)
# Adding element type (line 117)
str_65093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 44), 'str', 'F90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 24), tuple_65091, str_65093)
# Adding element type (line 117)
str_65094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 51), 'str', 'f90exec')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 24), tuple_65091, str_65094)
# Adding element type (line 117)
# Getting the type of 'None' (line 117)
None_65095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 62), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 24), tuple_65091, None_65095)

keyword_65096 = tuple_65091

# Obtaining an instance of the builtin type 'tuple' (line 118)
tuple_65097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 118)
# Adding element type (line 118)
str_65098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 24), 'str', 'exe.compiler_fix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 24), tuple_65097, str_65098)
# Adding element type (line 118)
str_65099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 44), 'str', 'F90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 24), tuple_65097, str_65099)
# Adding element type (line 118)
str_65100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 51), 'str', 'f90exec')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 24), tuple_65097, str_65100)
# Adding element type (line 118)
# Getting the type of 'None' (line 118)
None_65101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 62), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 24), tuple_65097, None_65101)

keyword_65102 = tuple_65097

# Obtaining an instance of the builtin type 'tuple' (line 119)
tuple_65103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 119)
# Adding element type (line 119)
str_65104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 23), 'str', 'exe.version_cmd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 23), tuple_65103, str_65104)
# Adding element type (line 119)
# Getting the type of 'None' (line 119)
None_65105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 42), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 23), tuple_65103, None_65105)
# Adding element type (line 119)
# Getting the type of 'None' (line 119)
None_65106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 48), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 23), tuple_65103, None_65106)
# Adding element type (line 119)
# Getting the type of 'None' (line 119)
None_65107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 54), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 23), tuple_65103, None_65107)

keyword_65108 = tuple_65103

# Obtaining an instance of the builtin type 'tuple' (line 120)
tuple_65109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 120)
# Adding element type (line 120)
str_65110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 21), 'str', 'exe.linker_so')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 21), tuple_65109, str_65110)
# Adding element type (line 120)
str_65111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 38), 'str', 'LDSHARED')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 21), tuple_65109, str_65111)
# Adding element type (line 120)
str_65112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 50), 'str', 'ldshared')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 21), tuple_65109, str_65112)
# Adding element type (line 120)
# Getting the type of 'None' (line 120)
None_65113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 62), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 21), tuple_65109, None_65113)

keyword_65114 = tuple_65109

# Obtaining an instance of the builtin type 'tuple' (line 121)
tuple_65115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 121)
# Adding element type (line 121)
str_65116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 22), 'str', 'exe.linker_exe')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 22), tuple_65115, str_65116)
# Adding element type (line 121)
str_65117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 40), 'str', 'LD')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 22), tuple_65115, str_65117)
# Adding element type (line 121)
str_65118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 46), 'str', 'ld')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 22), tuple_65115, str_65118)
# Adding element type (line 121)
# Getting the type of 'None' (line 121)
None_65119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 52), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 22), tuple_65115, None_65119)

keyword_65120 = tuple_65115

# Obtaining an instance of the builtin type 'tuple' (line 122)
tuple_65121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 122)
# Adding element type (line 122)
# Getting the type of 'None' (line 122)
None_65122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 20), tuple_65121, None_65122)
# Adding element type (line 122)
str_65123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 26), 'str', 'AR')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 20), tuple_65121, str_65123)
# Adding element type (line 122)
str_65124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 32), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 20), tuple_65121, str_65124)
# Adding element type (line 122)
# Getting the type of 'None' (line 122)
None_65125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 38), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 20), tuple_65121, None_65125)

keyword_65126 = tuple_65121

# Obtaining an instance of the builtin type 'tuple' (line 123)
tuple_65127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 123)
# Adding element type (line 123)
# Getting the type of 'None' (line 123)
None_65128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 18), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 18), tuple_65127, None_65128)
# Adding element type (line 123)
str_65129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 24), 'str', 'RANLIB')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 18), tuple_65127, str_65129)
# Adding element type (line 123)
str_65130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 34), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 18), tuple_65127, str_65130)
# Adding element type (line 123)
# Getting the type of 'None' (line 123)
None_65131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 44), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 18), tuple_65127, None_65131)

keyword_65132 = tuple_65127
kwargs_65133 = {'compiler_f77': keyword_65090, 'compiler_fix': keyword_65102, 'linker_exe': keyword_65120, 'ranlib': keyword_65132, 'distutils_section': keyword_65084, 'archiver': keyword_65126, 'version_cmd': keyword_65108, 'linker_so': keyword_65114, 'compiler_f90': keyword_65096}
# Getting the type of 'EnvironmentConfig' (line 114)
EnvironmentConfig_65082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 19), 'EnvironmentConfig', False)
# Calling EnvironmentConfig(args, kwargs) (line 114)
EnvironmentConfig_call_result_65134 = invoke(stypy.reporting.localization.Localization(__file__, 114, 19), EnvironmentConfig_65082, *[], **kwargs_65133)

# Getting the type of 'FCompiler'
FCompiler_65135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'command_vars' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65135, 'command_vars', EnvironmentConfig_call_result_65134)

# Assigning a Call to a Name (line 126):

# Call to EnvironmentConfig(...): (line 126)
# Processing the call keyword arguments (line 126)
str_65137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 26), 'str', 'config_fc')
keyword_65138 = str_65137

# Obtaining an instance of the builtin type 'tuple' (line 128)
tuple_65139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 128)
# Adding element type (line 128)
str_65140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 15), 'str', 'flags.f77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 15), tuple_65139, str_65140)
# Adding element type (line 128)
str_65141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 28), 'str', 'F77FLAGS')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 15), tuple_65139, str_65141)
# Adding element type (line 128)
str_65142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 40), 'str', 'f77flags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 15), tuple_65139, str_65142)
# Adding element type (line 128)
# Getting the type of 'flaglist' (line 128)
flaglist_65143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 52), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 15), tuple_65139, flaglist_65143)

keyword_65144 = tuple_65139

# Obtaining an instance of the builtin type 'tuple' (line 129)
tuple_65145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 129)
# Adding element type (line 129)
str_65146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 15), 'str', 'flags.f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 15), tuple_65145, str_65146)
# Adding element type (line 129)
str_65147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 28), 'str', 'F90FLAGS')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 15), tuple_65145, str_65147)
# Adding element type (line 129)
str_65148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 40), 'str', 'f90flags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 15), tuple_65145, str_65148)
# Adding element type (line 129)
# Getting the type of 'flaglist' (line 129)
flaglist_65149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 52), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 15), tuple_65145, flaglist_65149)

keyword_65150 = tuple_65145

# Obtaining an instance of the builtin type 'tuple' (line 130)
tuple_65151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 130)
# Adding element type (line 130)
str_65152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 16), 'str', 'flags.free')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 16), tuple_65151, str_65152)
# Adding element type (line 130)
str_65153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 30), 'str', 'FREEFLAGS')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 16), tuple_65151, str_65153)
# Adding element type (line 130)
str_65154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 43), 'str', 'freeflags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 16), tuple_65151, str_65154)
# Adding element type (line 130)
# Getting the type of 'flaglist' (line 130)
flaglist_65155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 56), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 16), tuple_65151, flaglist_65155)

keyword_65156 = tuple_65151

# Obtaining an instance of the builtin type 'tuple' (line 131)
tuple_65157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 131)
# Adding element type (line 131)
str_65158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 15), 'str', 'flags.fix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 15), tuple_65157, str_65158)
# Adding element type (line 131)
# Getting the type of 'None' (line 131)
None_65159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 28), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 15), tuple_65157, None_65159)
# Adding element type (line 131)
# Getting the type of 'None' (line 131)
None_65160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 34), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 15), tuple_65157, None_65160)
# Adding element type (line 131)
# Getting the type of 'flaglist' (line 131)
flaglist_65161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 40), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 15), tuple_65157, flaglist_65161)

keyword_65162 = tuple_65157

# Obtaining an instance of the builtin type 'tuple' (line 132)
tuple_65163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 132)
# Adding element type (line 132)
str_65164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 15), 'str', 'flags.opt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 15), tuple_65163, str_65164)
# Adding element type (line 132)
str_65165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 28), 'str', 'FOPT')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 15), tuple_65163, str_65165)
# Adding element type (line 132)
str_65166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 36), 'str', 'opt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 15), tuple_65163, str_65166)
# Adding element type (line 132)
# Getting the type of 'flaglist' (line 132)
flaglist_65167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 43), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 15), tuple_65163, flaglist_65167)

keyword_65168 = tuple_65163

# Obtaining an instance of the builtin type 'tuple' (line 133)
tuple_65169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 133)
# Adding element type (line 133)
str_65170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 19), 'str', 'flags.opt_f77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 19), tuple_65169, str_65170)
# Adding element type (line 133)
# Getting the type of 'None' (line 133)
None_65171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 36), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 19), tuple_65169, None_65171)
# Adding element type (line 133)
# Getting the type of 'None' (line 133)
None_65172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 42), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 19), tuple_65169, None_65172)
# Adding element type (line 133)
# Getting the type of 'flaglist' (line 133)
flaglist_65173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 48), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 19), tuple_65169, flaglist_65173)

keyword_65174 = tuple_65169

# Obtaining an instance of the builtin type 'tuple' (line 134)
tuple_65175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 134)
# Adding element type (line 134)
str_65176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 19), 'str', 'flags.opt_f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 19), tuple_65175, str_65176)
# Adding element type (line 134)
# Getting the type of 'None' (line 134)
None_65177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 36), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 19), tuple_65175, None_65177)
# Adding element type (line 134)
# Getting the type of 'None' (line 134)
None_65178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 42), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 19), tuple_65175, None_65178)
# Adding element type (line 134)
# Getting the type of 'flaglist' (line 134)
flaglist_65179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 48), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 19), tuple_65175, flaglist_65179)

keyword_65180 = tuple_65175

# Obtaining an instance of the builtin type 'tuple' (line 135)
tuple_65181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 135)
# Adding element type (line 135)
str_65182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 16), 'str', 'flags.arch')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 16), tuple_65181, str_65182)
# Adding element type (line 135)
str_65183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 30), 'str', 'FARCH')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 16), tuple_65181, str_65183)
# Adding element type (line 135)
str_65184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 39), 'str', 'arch')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 16), tuple_65181, str_65184)
# Adding element type (line 135)
# Getting the type of 'flaglist' (line 135)
flaglist_65185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 47), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 16), tuple_65181, flaglist_65185)

keyword_65186 = tuple_65181

# Obtaining an instance of the builtin type 'tuple' (line 136)
tuple_65187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 136)
# Adding element type (line 136)
str_65188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 20), 'str', 'flags.arch_f77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 20), tuple_65187, str_65188)
# Adding element type (line 136)
# Getting the type of 'None' (line 136)
None_65189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 38), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 20), tuple_65187, None_65189)
# Adding element type (line 136)
# Getting the type of 'None' (line 136)
None_65190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 44), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 20), tuple_65187, None_65190)
# Adding element type (line 136)
# Getting the type of 'flaglist' (line 136)
flaglist_65191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 50), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 20), tuple_65187, flaglist_65191)

keyword_65192 = tuple_65187

# Obtaining an instance of the builtin type 'tuple' (line 137)
tuple_65193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 137)
# Adding element type (line 137)
str_65194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 20), 'str', 'flags.arch_f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 20), tuple_65193, str_65194)
# Adding element type (line 137)
# Getting the type of 'None' (line 137)
None_65195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 38), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 20), tuple_65193, None_65195)
# Adding element type (line 137)
# Getting the type of 'None' (line 137)
None_65196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 44), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 20), tuple_65193, None_65196)
# Adding element type (line 137)
# Getting the type of 'flaglist' (line 137)
flaglist_65197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 50), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 20), tuple_65193, flaglist_65197)

keyword_65198 = tuple_65193

# Obtaining an instance of the builtin type 'tuple' (line 138)
tuple_65199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 138)
# Adding element type (line 138)
str_65200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 17), 'str', 'flags.debug')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 17), tuple_65199, str_65200)
# Adding element type (line 138)
str_65201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 32), 'str', 'FDEBUG')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 17), tuple_65199, str_65201)
# Adding element type (line 138)
str_65202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 42), 'str', 'fdebug')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 17), tuple_65199, str_65202)
# Adding element type (line 138)
# Getting the type of 'flaglist' (line 138)
flaglist_65203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 52), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 17), tuple_65199, flaglist_65203)

keyword_65204 = tuple_65199

# Obtaining an instance of the builtin type 'tuple' (line 139)
tuple_65205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 139)
# Adding element type (line 139)
str_65206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 21), 'str', 'flags.debug_f77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 21), tuple_65205, str_65206)
# Adding element type (line 139)
# Getting the type of 'None' (line 139)
None_65207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 40), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 21), tuple_65205, None_65207)
# Adding element type (line 139)
# Getting the type of 'None' (line 139)
None_65208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 46), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 21), tuple_65205, None_65208)
# Adding element type (line 139)
# Getting the type of 'flaglist' (line 139)
flaglist_65209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 52), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 21), tuple_65205, flaglist_65209)

keyword_65210 = tuple_65205

# Obtaining an instance of the builtin type 'tuple' (line 140)
tuple_65211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 140)
# Adding element type (line 140)
str_65212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 21), 'str', 'flags.debug_f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 21), tuple_65211, str_65212)
# Adding element type (line 140)
# Getting the type of 'None' (line 140)
None_65213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 40), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 21), tuple_65211, None_65213)
# Adding element type (line 140)
# Getting the type of 'None' (line 140)
None_65214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 46), 'None', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 21), tuple_65211, None_65214)
# Adding element type (line 140)
# Getting the type of 'flaglist' (line 140)
flaglist_65215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 52), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 21), tuple_65211, flaglist_65215)

keyword_65216 = tuple_65211

# Obtaining an instance of the builtin type 'tuple' (line 141)
tuple_65217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 141)
# Adding element type (line 141)
str_65218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 17), 'str', 'self.get_flags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 17), tuple_65217, str_65218)
# Adding element type (line 141)
str_65219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 35), 'str', 'FFLAGS')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 17), tuple_65217, str_65219)
# Adding element type (line 141)
str_65220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 45), 'str', 'fflags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 17), tuple_65217, str_65220)
# Adding element type (line 141)
# Getting the type of 'flaglist' (line 141)
flaglist_65221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 55), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 17), tuple_65217, flaglist_65221)

keyword_65222 = tuple_65217

# Obtaining an instance of the builtin type 'tuple' (line 142)
tuple_65223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 142)
# Adding element type (line 142)
str_65224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 21), 'str', 'flags.linker_so')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 21), tuple_65223, str_65224)
# Adding element type (line 142)
str_65225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 40), 'str', 'LDFLAGS')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 21), tuple_65223, str_65225)
# Adding element type (line 142)
str_65226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 51), 'str', 'ldflags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 21), tuple_65223, str_65226)
# Adding element type (line 142)
# Getting the type of 'flaglist' (line 142)
flaglist_65227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 62), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 21), tuple_65223, flaglist_65227)

keyword_65228 = tuple_65223

# Obtaining an instance of the builtin type 'tuple' (line 143)
tuple_65229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 143)
# Adding element type (line 143)
str_65230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 22), 'str', 'flags.linker_exe')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 22), tuple_65229, str_65230)
# Adding element type (line 143)
str_65231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 42), 'str', 'LDFLAGS')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 22), tuple_65229, str_65231)
# Adding element type (line 143)
str_65232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 53), 'str', 'ldflags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 22), tuple_65229, str_65232)
# Adding element type (line 143)
# Getting the type of 'flaglist' (line 143)
flaglist_65233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 64), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 22), tuple_65229, flaglist_65233)

keyword_65234 = tuple_65229

# Obtaining an instance of the builtin type 'tuple' (line 144)
tuple_65235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 144)
# Adding element type (line 144)
str_65236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 14), 'str', 'flags.ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 14), tuple_65235, str_65236)
# Adding element type (line 144)
str_65237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 26), 'str', 'ARFLAGS')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 14), tuple_65235, str_65237)
# Adding element type (line 144)
str_65238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 37), 'str', 'arflags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 14), tuple_65235, str_65238)
# Adding element type (line 144)
# Getting the type of 'flaglist' (line 144)
flaglist_65239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 48), 'flaglist', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 14), tuple_65235, flaglist_65239)

keyword_65240 = tuple_65235
kwargs_65241 = {'opt': keyword_65168, 'opt_f77': keyword_65174, 'arch_f77': keyword_65192, 'opt_f90': keyword_65180, 'fix': keyword_65162, 'f90': keyword_65150, 'free': keyword_65156, 'linker_exe': keyword_65234, 'debug_f90': keyword_65216, 'arch_f90': keyword_65198, 'distutils_section': keyword_65138, 'ar': keyword_65240, 'flags': keyword_65222, 'linker_so': keyword_65228, 'debug': keyword_65204, 'debug_f77': keyword_65210, 'f77': keyword_65144, 'arch': keyword_65186}
# Getting the type of 'EnvironmentConfig' (line 126)
EnvironmentConfig_65136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'EnvironmentConfig', False)
# Calling EnvironmentConfig(args, kwargs) (line 126)
EnvironmentConfig_call_result_65242 = invoke(stypy.reporting.localization.Localization(__file__, 126, 16), EnvironmentConfig_65136, *[], **kwargs_65241)

# Getting the type of 'FCompiler'
FCompiler_65243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'flag_vars' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65243, 'flag_vars', EnvironmentConfig_call_result_65242)

# Assigning a Dict to a Name (line 147):

# Obtaining an instance of the builtin type 'dict' (line 147)
dict_65244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 147)
# Adding element type (key, value) (line 147)
str_65245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 20), 'str', '.f')
str_65246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 26), 'str', 'f77')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 19), dict_65244, (str_65245, str_65246))
# Adding element type (key, value) (line 147)
str_65247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 20), 'str', '.for')
str_65248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 28), 'str', 'f77')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 19), dict_65244, (str_65247, str_65248))
# Adding element type (key, value) (line 147)
str_65249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 20), 'str', '.F')
str_65250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 26), 'str', 'f77')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 19), dict_65244, (str_65249, str_65250))
# Adding element type (key, value) (line 147)
str_65251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 20), 'str', '.ftn')
str_65252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 28), 'str', 'f77')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 19), dict_65244, (str_65251, str_65252))
# Adding element type (key, value) (line 147)
str_65253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 20), 'str', '.f77')
str_65254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 28), 'str', 'f77')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 19), dict_65244, (str_65253, str_65254))
# Adding element type (key, value) (line 147)
str_65255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 20), 'str', '.f90')
str_65256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 28), 'str', 'f90')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 19), dict_65244, (str_65255, str_65256))
# Adding element type (key, value) (line 147)
str_65257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 20), 'str', '.F90')
str_65258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 28), 'str', 'f90')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 19), dict_65244, (str_65257, str_65258))
# Adding element type (key, value) (line 147)
str_65259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 20), 'str', '.f95')
str_65260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 28), 'str', 'f90')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 19), dict_65244, (str_65259, str_65260))

# Getting the type of 'FCompiler'
FCompiler_65261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'language_map' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65261, 'language_map', dict_65244)

# Assigning a List to a Name (line 156):

# Obtaining an instance of the builtin type 'list' (line 156)
list_65262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 156)
# Adding element type (line 156)
str_65263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 22), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 21), list_65262, str_65263)
# Adding element type (line 156)
str_65264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 29), 'str', 'f77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 21), list_65262, str_65264)

# Getting the type of 'FCompiler'
FCompiler_65265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'language_order' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65265, 'language_order', list_65262)

# Assigning a Name to a Name (line 161):
# Getting the type of 'None' (line 161)
None_65266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'None')
# Getting the type of 'FCompiler'
FCompiler_65267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65267, 'compiler_type', None_65266)

# Assigning a Tuple to a Name (line 162):

# Obtaining an instance of the builtin type 'tuple' (line 162)
tuple_65268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 162)

# Getting the type of 'FCompiler'
FCompiler_65269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'compiler_aliases' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65269, 'compiler_aliases', tuple_65268)

# Assigning a Name to a Name (line 163):
# Getting the type of 'None' (line 163)
None_65270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 22), 'None')
# Getting the type of 'FCompiler'
FCompiler_65271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'version_pattern' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65271, 'version_pattern', None_65270)

# Assigning a List to a Name (line 165):

# Obtaining an instance of the builtin type 'list' (line 165)
list_65272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 165)

# Getting the type of 'FCompiler'
FCompiler_65273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'possible_executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65273, 'possible_executables', list_65272)

# Assigning a Dict to a Name (line 166):

# Obtaining an instance of the builtin type 'dict' (line 166)
dict_65274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 166)
# Adding element type (key, value) (line 166)
str_65275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 8), 'str', 'version_cmd')

# Obtaining an instance of the builtin type 'list' (line 167)
list_65276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 167)
# Adding element type (line 167)
str_65277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 24), 'str', 'f77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 23), list_65276, str_65277)
# Adding element type (line 167)
str_65278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 31), 'str', '-v')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 23), list_65276, str_65278)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 18), dict_65274, (str_65275, list_65276))
# Adding element type (key, value) (line 166)
str_65279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 168)
list_65280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 24), 'list')
# Adding type elements to the builtin type 'list' instance (line 168)
# Adding element type (line 168)
str_65281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 25), 'str', 'f77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 24), list_65280, str_65281)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 18), dict_65274, (str_65279, list_65280))
# Adding element type (key, value) (line 166)
str_65282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 169)
list_65283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 24), 'list')
# Adding type elements to the builtin type 'list' instance (line 169)
# Adding element type (line 169)
str_65284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 25), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 169, 24), list_65283, str_65284)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 18), dict_65274, (str_65282, list_65283))
# Adding element type (key, value) (line 166)
str_65285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 170)
list_65286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 24), 'list')
# Adding type elements to the builtin type 'list' instance (line 170)
# Adding element type (line 170)
str_65287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 25), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 24), list_65286, str_65287)
# Adding element type (line 170)
str_65288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 32), 'str', '-fixed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 24), list_65286, str_65288)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 18), dict_65274, (str_65285, list_65286))
# Adding element type (key, value) (line 166)
str_65289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 171)
list_65290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 171)
# Adding element type (line 171)
str_65291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 22), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 21), list_65290, str_65291)
# Adding element type (line 171)
str_65292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 29), 'str', '-shared')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 21), list_65290, str_65292)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 18), dict_65274, (str_65289, list_65290))
# Adding element type (key, value) (line 166)
str_65293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 8), 'str', 'linker_exe')

# Obtaining an instance of the builtin type 'list' (line 172)
list_65294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 172)
# Adding element type (line 172)
str_65295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 23), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 22), list_65294, str_65295)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 18), dict_65274, (str_65293, list_65294))
# Adding element type (key, value) (line 166)
str_65296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 173)
list_65297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 173)
# Adding element type (line 173)
str_65298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 21), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 20), list_65297, str_65298)
# Adding element type (line 173)
str_65299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 27), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 20), list_65297, str_65299)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 18), dict_65274, (str_65296, list_65297))
# Adding element type (key, value) (line 166)
str_65300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 8), 'str', 'ranlib')
# Getting the type of 'None' (line 174)
None_65301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 18), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 18), dict_65274, (str_65300, None_65301))

# Getting the type of 'FCompiler'
FCompiler_65302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65302, 'executables', dict_65274)

# Assigning a Name to a Name (line 180):
# Getting the type of 'None' (line 180)
None_65303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 29), 'None')
# Getting the type of 'FCompiler'
FCompiler_65304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'suggested_f90_compiler' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65304, 'suggested_f90_compiler', None_65303)

# Assigning a Str to a Name (line 182):
str_65305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 21), 'str', '-c')
# Getting the type of 'FCompiler'
FCompiler_65306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'compile_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65306, 'compile_switch', str_65305)

# Assigning a Str to a Name (line 183):
str_65307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 20), 'str', '-o ')
# Getting the type of 'FCompiler'
FCompiler_65308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'object_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65308, 'object_switch', str_65307)

# Assigning a Str to a Name (line 187):
str_65309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 21), 'str', '-o ')
# Getting the type of 'FCompiler'
FCompiler_65310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'library_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65310, 'library_switch', str_65309)

# Assigning a Name to a Name (line 192):
# Getting the type of 'None' (line 192)
None_65311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 'None')
# Getting the type of 'FCompiler'
FCompiler_65312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'module_dir_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65312, 'module_dir_switch', None_65311)

# Assigning a Str to a Name (line 195):
str_65313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 28), 'str', '-I')
# Getting the type of 'FCompiler'
FCompiler_65314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'module_include_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65314, 'module_include_switch', str_65313)

# Assigning a List to a Name (line 197):

# Obtaining an instance of the builtin type 'list' (line 197)
list_65315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 197)

# Getting the type of 'FCompiler'
FCompiler_65316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'pic_flags' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65316, 'pic_flags', list_65315)

# Assigning a List to a Name (line 199):

# Obtaining an instance of the builtin type 'list' (line 199)
list_65317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 199)
# Adding element type (line 199)
str_65318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 22), 'str', '.for')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), list_65317, str_65318)
# Adding element type (line 199)
str_65319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 30), 'str', '.ftn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), list_65317, str_65319)
# Adding element type (line 199)
str_65320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 38), 'str', '.f77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), list_65317, str_65320)
# Adding element type (line 199)
str_65321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 46), 'str', '.f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), list_65317, str_65321)
# Adding element type (line 199)
str_65322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 52), 'str', '.f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), list_65317, str_65322)
# Adding element type (line 199)
str_65323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 60), 'str', '.f95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), list_65317, str_65323)
# Adding element type (line 199)
str_65324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 68), 'str', '.F')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), list_65317, str_65324)
# Adding element type (line 199)
str_65325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 74), 'str', '.F90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), list_65317, str_65325)
# Adding element type (line 199)
str_65326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 82), 'str', '.FOR')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 21), list_65317, str_65326)

# Getting the type of 'FCompiler'
FCompiler_65327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'src_extensions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65327, 'src_extensions', list_65317)

# Assigning a Str to a Name (line 200):
str_65328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 20), 'str', '.o')
# Getting the type of 'FCompiler'
FCompiler_65329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'obj_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65329, 'obj_extension', str_65328)

# Assigning a Call to a Name (line 202):

# Call to get_shared_lib_extension(...): (line 202)
# Processing the call keyword arguments (line 202)
kwargs_65331 = {}
# Getting the type of 'get_shared_lib_extension' (line 202)
get_shared_lib_extension_65330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 27), 'get_shared_lib_extension', False)
# Calling get_shared_lib_extension(args, kwargs) (line 202)
get_shared_lib_extension_call_result_65332 = invoke(stypy.reporting.localization.Localization(__file__, 202, 27), get_shared_lib_extension_65330, *[], **kwargs_65331)

# Getting the type of 'FCompiler'
FCompiler_65333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'shared_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65333, 'shared_lib_extension', get_shared_lib_extension_call_result_65332)

# Assigning a Str to a Name (line 203):
str_65334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 27), 'str', '.a')
# Getting the type of 'FCompiler'
FCompiler_65335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'static_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65335, 'static_lib_extension', str_65334)

# Assigning a Str to a Name (line 204):
str_65336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 24), 'str', 'lib%s%s')
# Getting the type of 'FCompiler'
FCompiler_65337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'static_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65337, 'static_lib_format', str_65336)

# Assigning a Str to a Name (line 205):
str_65338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 24), 'str', '%s%s')
# Getting the type of 'FCompiler'
FCompiler_65339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'shared_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65339, 'shared_lib_format', str_65338)

# Assigning a Str to a Name (line 206):
str_65340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 20), 'str', '')
# Getting the type of 'FCompiler'
FCompiler_65341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'exe_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65341, 'exe_extension', str_65340)

# Assigning a Dict to a Name (line 208):

# Obtaining an instance of the builtin type 'dict' (line 208)
dict_65342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 208)

# Getting the type of 'FCompiler'
FCompiler_65343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member '_exe_cache' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65343, '_exe_cache', dict_65342)

# Assigning a List to a Name (line 210):

# Obtaining an instance of the builtin type 'list' (line 210)
list_65344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 210)
# Adding element type (line 210)
str_65345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 24), 'str', 'version_cmd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 23), list_65344, str_65345)
# Adding element type (line 210)
str_65346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 39), 'str', 'compiler_f77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 23), list_65344, str_65346)
# Adding element type (line 210)
str_65347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 55), 'str', 'compiler_f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 23), list_65344, str_65347)
# Adding element type (line 210)
str_65348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 24), 'str', 'compiler_fix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 23), list_65344, str_65348)
# Adding element type (line 210)
str_65349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 40), 'str', 'linker_so')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 23), list_65344, str_65349)
# Adding element type (line 210)
str_65350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 53), 'str', 'linker_exe')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 23), list_65344, str_65350)
# Adding element type (line 210)
str_65351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 67), 'str', 'archiver')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 23), list_65344, str_65351)
# Adding element type (line 210)
str_65352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 24), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 23), list_65344, str_65352)

# Getting the type of 'FCompiler'
FCompiler_65353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member '_executable_keys' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65353, '_executable_keys', list_65344)

# Assigning a Name to a Name (line 216):
# Getting the type of 'None' (line 216)
None_65354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 17), 'None')
# Getting the type of 'FCompiler'
FCompiler_65355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'c_compiler' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65355, 'c_compiler', None_65354)

# Assigning a List to a Name (line 219):

# Obtaining an instance of the builtin type 'list' (line 219)
list_65356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 219)

# Getting the type of 'FCompiler'
FCompiler_65357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'extra_f77_compile_args' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65357, 'extra_f77_compile_args', list_65356)

# Assigning a List to a Name (line 220):

# Obtaining an instance of the builtin type 'list' (line 220)
list_65358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 29), 'list')
# Adding type elements to the builtin type 'list' instance (line 220)

# Getting the type of 'FCompiler'
FCompiler_65359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'extra_f90_compile_args' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65359, 'extra_f90_compile_args', list_65358)

# Assigning a Call to a Name (line 256):

# Call to _command_property(...): (line 256)
# Processing the call arguments (line 256)
str_65362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 36), 'str', 'version_cmd')
# Processing the call keyword arguments (line 256)
kwargs_65363 = {}
# Getting the type of 'FCompiler'
FCompiler_65360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler', False)
# Obtaining the member '_command_property' of a type
_command_property_65361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65360, '_command_property')
# Calling _command_property(args, kwargs) (line 256)
_command_property_call_result_65364 = invoke(stypy.reporting.localization.Localization(__file__, 256, 18), _command_property_65361, *[str_65362], **kwargs_65363)

# Getting the type of 'FCompiler'
FCompiler_65365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'version_cmd' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65365, 'version_cmd', _command_property_call_result_65364)

# Assigning a Call to a Name (line 257):

# Call to _command_property(...): (line 257)
# Processing the call arguments (line 257)
str_65368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 37), 'str', 'compiler_f77')
# Processing the call keyword arguments (line 257)
kwargs_65369 = {}
# Getting the type of 'FCompiler'
FCompiler_65366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler', False)
# Obtaining the member '_command_property' of a type
_command_property_65367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65366, '_command_property')
# Calling _command_property(args, kwargs) (line 257)
_command_property_call_result_65370 = invoke(stypy.reporting.localization.Localization(__file__, 257, 19), _command_property_65367, *[str_65368], **kwargs_65369)

# Getting the type of 'FCompiler'
FCompiler_65371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'compiler_f77' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65371, 'compiler_f77', _command_property_call_result_65370)

# Assigning a Call to a Name (line 258):

# Call to _command_property(...): (line 258)
# Processing the call arguments (line 258)
str_65374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 37), 'str', 'compiler_f90')
# Processing the call keyword arguments (line 258)
kwargs_65375 = {}
# Getting the type of 'FCompiler'
FCompiler_65372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler', False)
# Obtaining the member '_command_property' of a type
_command_property_65373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65372, '_command_property')
# Calling _command_property(args, kwargs) (line 258)
_command_property_call_result_65376 = invoke(stypy.reporting.localization.Localization(__file__, 258, 19), _command_property_65373, *[str_65374], **kwargs_65375)

# Getting the type of 'FCompiler'
FCompiler_65377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'compiler_f90' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65377, 'compiler_f90', _command_property_call_result_65376)

# Assigning a Call to a Name (line 259):

# Call to _command_property(...): (line 259)
# Processing the call arguments (line 259)
str_65380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 37), 'str', 'compiler_fix')
# Processing the call keyword arguments (line 259)
kwargs_65381 = {}
# Getting the type of 'FCompiler'
FCompiler_65378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler', False)
# Obtaining the member '_command_property' of a type
_command_property_65379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65378, '_command_property')
# Calling _command_property(args, kwargs) (line 259)
_command_property_call_result_65382 = invoke(stypy.reporting.localization.Localization(__file__, 259, 19), _command_property_65379, *[str_65380], **kwargs_65381)

# Getting the type of 'FCompiler'
FCompiler_65383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'compiler_fix' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65383, 'compiler_fix', _command_property_call_result_65382)

# Assigning a Call to a Name (line 260):

# Call to _command_property(...): (line 260)
# Processing the call arguments (line 260)
str_65386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 34), 'str', 'linker_so')
# Processing the call keyword arguments (line 260)
kwargs_65387 = {}
# Getting the type of 'FCompiler'
FCompiler_65384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler', False)
# Obtaining the member '_command_property' of a type
_command_property_65385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65384, '_command_property')
# Calling _command_property(args, kwargs) (line 260)
_command_property_call_result_65388 = invoke(stypy.reporting.localization.Localization(__file__, 260, 16), _command_property_65385, *[str_65386], **kwargs_65387)

# Getting the type of 'FCompiler'
FCompiler_65389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'linker_so' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65389, 'linker_so', _command_property_call_result_65388)

# Assigning a Call to a Name (line 261):

# Call to _command_property(...): (line 261)
# Processing the call arguments (line 261)
str_65392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 35), 'str', 'linker_exe')
# Processing the call keyword arguments (line 261)
kwargs_65393 = {}
# Getting the type of 'FCompiler'
FCompiler_65390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler', False)
# Obtaining the member '_command_property' of a type
_command_property_65391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65390, '_command_property')
# Calling _command_property(args, kwargs) (line 261)
_command_property_call_result_65394 = invoke(stypy.reporting.localization.Localization(__file__, 261, 17), _command_property_65391, *[str_65392], **kwargs_65393)

# Getting the type of 'FCompiler'
FCompiler_65395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'linker_exe' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65395, 'linker_exe', _command_property_call_result_65394)

# Assigning a Call to a Name (line 262):

# Call to _command_property(...): (line 262)
# Processing the call arguments (line 262)
str_65398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 33), 'str', 'archiver')
# Processing the call keyword arguments (line 262)
kwargs_65399 = {}
# Getting the type of 'FCompiler'
FCompiler_65396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler', False)
# Obtaining the member '_command_property' of a type
_command_property_65397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65396, '_command_property')
# Calling _command_property(args, kwargs) (line 262)
_command_property_call_result_65400 = invoke(stypy.reporting.localization.Localization(__file__, 262, 15), _command_property_65397, *[str_65398], **kwargs_65399)

# Getting the type of 'FCompiler'
FCompiler_65401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'archiver' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65401, 'archiver', _command_property_call_result_65400)

# Assigning a Call to a Name (line 263):

# Call to _command_property(...): (line 263)
# Processing the call arguments (line 263)
str_65404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 31), 'str', 'ranlib')
# Processing the call keyword arguments (line 263)
kwargs_65405 = {}
# Getting the type of 'FCompiler'
FCompiler_65402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler', False)
# Obtaining the member '_command_property' of a type
_command_property_65403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65402, '_command_property')
# Calling _command_property(args, kwargs) (line 263)
_command_property_call_result_65406 = invoke(stypy.reporting.localization.Localization(__file__, 263, 13), _command_property_65403, *[str_65404], **kwargs_65405)

# Getting the type of 'FCompiler'
FCompiler_65407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'ranlib' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65407, 'ranlib', _command_property_call_result_65406)

# Assigning a Name to a Name (line 419):
# Getting the type of 'FCompiler'
FCompiler_65408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Obtaining the member 'get_flags_opt' of a type
get_flags_opt_65409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65408, 'get_flags_opt')
# Getting the type of 'FCompiler'
FCompiler_65410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'get_flags_opt_f90' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65410, 'get_flags_opt_f90', get_flags_opt_65409)

# Assigning a Name to a Name (line 419):
# Getting the type of 'FCompiler'
FCompiler_65411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Obtaining the member 'get_flags_opt_f90' of a type
get_flags_opt_f90_65412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65411, 'get_flags_opt_f90')
# Getting the type of 'FCompiler'
FCompiler_65413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'get_flags_opt_f77' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65413, 'get_flags_opt_f77', get_flags_opt_f90_65412)

# Assigning a Name to a Name (line 420):
# Getting the type of 'FCompiler'
FCompiler_65414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Obtaining the member 'get_flags_arch' of a type
get_flags_arch_65415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65414, 'get_flags_arch')
# Getting the type of 'FCompiler'
FCompiler_65416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'get_flags_arch_f90' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65416, 'get_flags_arch_f90', get_flags_arch_65415)

# Assigning a Name to a Name (line 420):
# Getting the type of 'FCompiler'
FCompiler_65417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Obtaining the member 'get_flags_arch_f90' of a type
get_flags_arch_f90_65418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65417, 'get_flags_arch_f90')
# Getting the type of 'FCompiler'
FCompiler_65419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'get_flags_arch_f77' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65419, 'get_flags_arch_f77', get_flags_arch_f90_65418)

# Assigning a Name to a Name (line 421):
# Getting the type of 'FCompiler'
FCompiler_65420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Obtaining the member 'get_flags_debug' of a type
get_flags_debug_65421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65420, 'get_flags_debug')
# Getting the type of 'FCompiler'
FCompiler_65422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'get_flags_debug_f90' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65422, 'get_flags_debug_f90', get_flags_debug_65421)

# Assigning a Name to a Name (line 421):
# Getting the type of 'FCompiler'
FCompiler_65423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Obtaining the member 'get_flags_debug_f90' of a type
get_flags_debug_f90_65424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65423, 'get_flags_debug_f90')
# Getting the type of 'FCompiler'
FCompiler_65425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FCompiler')
# Setting the type of the member 'get_flags_debug_f77' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FCompiler_65425, 'get_flags_debug_f77', get_flags_debug_f90_65424)

# Assigning a Tuple to a Name (line 706):

# Assigning a Tuple to a Name (line 706):

# Obtaining an instance of the builtin type 'tuple' (line 708)
tuple_65426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 4), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 708)
# Adding element type (line 708)

# Obtaining an instance of the builtin type 'tuple' (line 708)
tuple_65427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 708)
# Adding element type (line 708)
str_65428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 5), 'str', 'win32')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 5), tuple_65427, str_65428)
# Adding element type (line 708)

# Obtaining an instance of the builtin type 'tuple' (line 708)
tuple_65429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 708)
# Adding element type (line 708)
str_65430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 15), 'str', 'gnu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 15), tuple_65429, str_65430)
# Adding element type (line 708)
str_65431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 22), 'str', 'intelv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 15), tuple_65429, str_65431)
# Adding element type (line 708)
str_65432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 32), 'str', 'absoft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 15), tuple_65429, str_65432)
# Adding element type (line 708)
str_65433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 42), 'str', 'compaqv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 15), tuple_65429, str_65433)
# Adding element type (line 708)
str_65434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 53), 'str', 'intelev')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 15), tuple_65429, str_65434)
# Adding element type (line 708)
str_65435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 64), 'str', 'gnu95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 15), tuple_65429, str_65435)
# Adding element type (line 708)
str_65436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 708, 73), 'str', 'g95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 15), tuple_65429, str_65436)
# Adding element type (line 708)
str_65437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 15), 'str', 'intelvem')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 15), tuple_65429, str_65437)
# Adding element type (line 708)
str_65438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 27), 'str', 'intelem')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 15), tuple_65429, str_65438)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 5), tuple_65427, tuple_65429)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 4), tuple_65426, tuple_65427)
# Adding element type (line 708)

# Obtaining an instance of the builtin type 'tuple' (line 710)
tuple_65439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 710)
# Adding element type (line 710)
str_65440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 5), 'str', 'cygwin.*')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 5), tuple_65439, str_65440)
# Adding element type (line 710)

# Obtaining an instance of the builtin type 'tuple' (line 710)
tuple_65441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 710)
# Adding element type (line 710)
str_65442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 18), 'str', 'gnu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 18), tuple_65441, str_65442)
# Adding element type (line 710)
str_65443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 25), 'str', 'intelv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 18), tuple_65441, str_65443)
# Adding element type (line 710)
str_65444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 35), 'str', 'absoft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 18), tuple_65441, str_65444)
# Adding element type (line 710)
str_65445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 45), 'str', 'compaqv')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 18), tuple_65441, str_65445)
# Adding element type (line 710)
str_65446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 56), 'str', 'intelev')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 18), tuple_65441, str_65446)
# Adding element type (line 710)
str_65447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 67), 'str', 'gnu95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 18), tuple_65441, str_65447)
# Adding element type (line 710)
str_65448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 76), 'str', 'g95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 18), tuple_65441, str_65448)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 710, 5), tuple_65439, tuple_65441)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 4), tuple_65426, tuple_65439)
# Adding element type (line 708)

# Obtaining an instance of the builtin type 'tuple' (line 711)
tuple_65449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 711)
# Adding element type (line 711)
str_65450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 5), 'str', 'linux.*')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 5), tuple_65449, str_65450)
# Adding element type (line 711)

# Obtaining an instance of the builtin type 'tuple' (line 711)
tuple_65451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 711)
# Adding element type (line 711)
str_65452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 17), 'str', 'gnu95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 17), tuple_65451, str_65452)
# Adding element type (line 711)
str_65453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 26), 'str', 'intel')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 17), tuple_65451, str_65453)
# Adding element type (line 711)
str_65454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 35), 'str', 'lahey')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 17), tuple_65451, str_65454)
# Adding element type (line 711)
str_65455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 44), 'str', 'pg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 17), tuple_65451, str_65455)
# Adding element type (line 711)
str_65456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 50), 'str', 'absoft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 17), tuple_65451, str_65456)
# Adding element type (line 711)
str_65457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 60), 'str', 'nag')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 17), tuple_65451, str_65457)
# Adding element type (line 711)
str_65458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 67), 'str', 'vast')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 17), tuple_65451, str_65458)
# Adding element type (line 711)
str_65459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 711, 75), 'str', 'compaq')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 17), tuple_65451, str_65459)
# Adding element type (line 711)
str_65460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 16), 'str', 'intele')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 17), tuple_65451, str_65460)
# Adding element type (line 711)
str_65461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 26), 'str', 'intelem')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 17), tuple_65451, str_65461)
# Adding element type (line 711)
str_65462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 37), 'str', 'gnu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 17), tuple_65451, str_65462)
# Adding element type (line 711)
str_65463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 44), 'str', 'g95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 17), tuple_65451, str_65463)
# Adding element type (line 711)
str_65464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 712, 51), 'str', 'pathf95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 17), tuple_65451, str_65464)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 711, 5), tuple_65449, tuple_65451)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 4), tuple_65426, tuple_65449)
# Adding element type (line 708)

# Obtaining an instance of the builtin type 'tuple' (line 713)
tuple_65465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 713)
# Adding element type (line 713)
str_65466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 5), 'str', 'darwin.*')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 5), tuple_65465, str_65466)
# Adding element type (line 713)

# Obtaining an instance of the builtin type 'tuple' (line 713)
tuple_65467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 713)
# Adding element type (line 713)
str_65468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 18), 'str', 'gnu95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 18), tuple_65467, str_65468)
# Adding element type (line 713)
str_65469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 27), 'str', 'nag')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 18), tuple_65467, str_65469)
# Adding element type (line 713)
str_65470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 34), 'str', 'absoft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 18), tuple_65467, str_65470)
# Adding element type (line 713)
str_65471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 44), 'str', 'ibm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 18), tuple_65467, str_65471)
# Adding element type (line 713)
str_65472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 51), 'str', 'intel')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 18), tuple_65467, str_65472)
# Adding element type (line 713)
str_65473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 60), 'str', 'gnu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 18), tuple_65467, str_65473)
# Adding element type (line 713)
str_65474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 67), 'str', 'g95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 18), tuple_65467, str_65474)
# Adding element type (line 713)
str_65475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 713, 74), 'str', 'pg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 18), tuple_65467, str_65475)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 713, 5), tuple_65465, tuple_65467)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 4), tuple_65426, tuple_65465)
# Adding element type (line 708)

# Obtaining an instance of the builtin type 'tuple' (line 714)
tuple_65476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 714)
# Adding element type (line 714)
str_65477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 5), 'str', 'sunos.*')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 714, 5), tuple_65476, str_65477)
# Adding element type (line 714)

# Obtaining an instance of the builtin type 'tuple' (line 714)
tuple_65478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 714)
# Adding element type (line 714)
str_65479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 17), 'str', 'sun')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 714, 17), tuple_65478, str_65479)
# Adding element type (line 714)
str_65480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 24), 'str', 'gnu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 714, 17), tuple_65478, str_65480)
# Adding element type (line 714)
str_65481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 31), 'str', 'gnu95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 714, 17), tuple_65478, str_65481)
# Adding element type (line 714)
str_65482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 714, 40), 'str', 'g95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 714, 17), tuple_65478, str_65482)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 714, 5), tuple_65476, tuple_65478)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 4), tuple_65426, tuple_65476)
# Adding element type (line 708)

# Obtaining an instance of the builtin type 'tuple' (line 715)
tuple_65483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 715)
# Adding element type (line 715)
str_65484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 5), 'str', 'irix.*')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 5), tuple_65483, str_65484)
# Adding element type (line 715)

# Obtaining an instance of the builtin type 'tuple' (line 715)
tuple_65485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 715)
# Adding element type (line 715)
str_65486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 16), 'str', 'mips')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 16), tuple_65485, str_65486)
# Adding element type (line 715)
str_65487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 24), 'str', 'gnu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 16), tuple_65485, str_65487)
# Adding element type (line 715)
str_65488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 715, 31), 'str', 'gnu95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 16), tuple_65485, str_65488)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 715, 5), tuple_65483, tuple_65485)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 4), tuple_65426, tuple_65483)
# Adding element type (line 708)

# Obtaining an instance of the builtin type 'tuple' (line 716)
tuple_65489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 716)
# Adding element type (line 716)
str_65490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 5), 'str', 'aix.*')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 716, 5), tuple_65489, str_65490)
# Adding element type (line 716)

# Obtaining an instance of the builtin type 'tuple' (line 716)
tuple_65491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 716)
# Adding element type (line 716)
str_65492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 15), 'str', 'ibm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 716, 15), tuple_65491, str_65492)
# Adding element type (line 716)
str_65493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 22), 'str', 'gnu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 716, 15), tuple_65491, str_65493)
# Adding element type (line 716)
str_65494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 716, 29), 'str', 'gnu95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 716, 15), tuple_65491, str_65494)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 716, 5), tuple_65489, tuple_65491)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 4), tuple_65426, tuple_65489)
# Adding element type (line 708)

# Obtaining an instance of the builtin type 'tuple' (line 718)
tuple_65495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 718)
# Adding element type (line 718)
str_65496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 5), 'str', 'posix')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 5), tuple_65495, str_65496)
# Adding element type (line 718)

# Obtaining an instance of the builtin type 'tuple' (line 718)
tuple_65497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 718)
# Adding element type (line 718)
str_65498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 15), 'str', 'gnu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 15), tuple_65497, str_65498)
# Adding element type (line 718)
str_65499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 22), 'str', 'gnu95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 15), tuple_65497, str_65499)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 5), tuple_65495, tuple_65497)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 4), tuple_65426, tuple_65495)
# Adding element type (line 708)

# Obtaining an instance of the builtin type 'tuple' (line 719)
tuple_65500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 719)
# Adding element type (line 719)
str_65501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 5), 'str', 'nt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 719, 5), tuple_65500, str_65501)
# Adding element type (line 719)

# Obtaining an instance of the builtin type 'tuple' (line 719)
tuple_65502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 719)
# Adding element type (line 719)
str_65503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 12), 'str', 'gnu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 719, 12), tuple_65502, str_65503)
# Adding element type (line 719)
str_65504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 719, 19), 'str', 'gnu95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 719, 12), tuple_65502, str_65504)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 719, 5), tuple_65500, tuple_65502)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 4), tuple_65426, tuple_65500)
# Adding element type (line 708)

# Obtaining an instance of the builtin type 'tuple' (line 720)
tuple_65505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 720)
# Adding element type (line 720)
str_65506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 5), 'str', 'mac')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 720, 5), tuple_65505, str_65506)
# Adding element type (line 720)

# Obtaining an instance of the builtin type 'tuple' (line 720)
tuple_65507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 720)
# Adding element type (line 720)
str_65508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 13), 'str', 'gnu95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 720, 13), tuple_65507, str_65508)
# Adding element type (line 720)
str_65509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 22), 'str', 'gnu')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 720, 13), tuple_65507, str_65509)
# Adding element type (line 720)
str_65510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 720, 29), 'str', 'pg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 720, 13), tuple_65507, str_65510)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 720, 5), tuple_65505, tuple_65507)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 708, 4), tuple_65426, tuple_65505)

# Assigning a type to the variable '_default_compilers' (line 706)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 0), '_default_compilers', tuple_65426)

# Assigning a Name to a Name (line 723):

# Assigning a Name to a Name (line 723):
# Getting the type of 'None' (line 723)
None_65511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 18), 'None')
# Assigning a type to the variable 'fcompiler_class' (line 723)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 0), 'fcompiler_class', None_65511)

# Assigning a Name to a Name (line 724):

# Assigning a Name to a Name (line 724):
# Getting the type of 'None' (line 724)
None_65512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 20), 'None')
# Assigning a type to the variable 'fcompiler_aliases' (line 724)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 0), 'fcompiler_aliases', None_65512)

@norecursion
def load_all_fcompiler_classes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'load_all_fcompiler_classes'
    module_type_store = module_type_store.open_function_context('load_all_fcompiler_classes', 726, 0, False)
    
    # Passed parameters checking function
    load_all_fcompiler_classes.stypy_localization = localization
    load_all_fcompiler_classes.stypy_type_of_self = None
    load_all_fcompiler_classes.stypy_type_store = module_type_store
    load_all_fcompiler_classes.stypy_function_name = 'load_all_fcompiler_classes'
    load_all_fcompiler_classes.stypy_param_names_list = []
    load_all_fcompiler_classes.stypy_varargs_param_name = None
    load_all_fcompiler_classes.stypy_kwargs_param_name = None
    load_all_fcompiler_classes.stypy_call_defaults = defaults
    load_all_fcompiler_classes.stypy_call_varargs = varargs
    load_all_fcompiler_classes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'load_all_fcompiler_classes', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'load_all_fcompiler_classes', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'load_all_fcompiler_classes(...)' code ##################

    str_65513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 729, (-1)), 'str', 'Cache all the FCompiler classes found in modules in the\n    numpy.distutils.fcompiler package.\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 730, 4))
    
    # 'from glob import glob' statement (line 730)
    from glob import glob

    import_from_module(stypy.reporting.localization.Localization(__file__, 730, 4), 'glob', None, module_type_store, ['glob'], [glob])
    
    # Marking variables as global (line 731)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 731, 4), 'fcompiler_class')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 731, 4), 'fcompiler_aliases')
    
    # Type idiom detected: calculating its left and rigth part (line 732)
    # Getting the type of 'fcompiler_class' (line 732)
    fcompiler_class_65514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 4), 'fcompiler_class')
    # Getting the type of 'None' (line 732)
    None_65515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 30), 'None')
    
    (may_be_65516, more_types_in_union_65517) = may_not_be_none(fcompiler_class_65514, None_65515)

    if may_be_65516:

        if more_types_in_union_65517:
            # Runtime conditional SSA (line 732)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'stypy_return_type' (line 733)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 733, 8), 'stypy_return_type', types.NoneType)

        if more_types_in_union_65517:
            # SSA join for if statement (line 732)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 734):
    
    # Assigning a Call to a Name (line 734):
    
    # Call to join(...): (line 734)
    # Processing the call arguments (line 734)
    
    # Call to dirname(...): (line 734)
    # Processing the call arguments (line 734)
    # Getting the type of '__file__' (line 734)
    file___65524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 39), '__file__', False)
    # Processing the call keyword arguments (line 734)
    kwargs_65525 = {}
    # Getting the type of 'os' (line 734)
    os_65521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 23), 'os', False)
    # Obtaining the member 'path' of a type (line 734)
    path_65522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 23), os_65521, 'path')
    # Obtaining the member 'dirname' of a type (line 734)
    dirname_65523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 23), path_65522, 'dirname')
    # Calling dirname(args, kwargs) (line 734)
    dirname_call_result_65526 = invoke(stypy.reporting.localization.Localization(__file__, 734, 23), dirname_65523, *[file___65524], **kwargs_65525)
    
    str_65527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 734, 50), 'str', '*.py')
    # Processing the call keyword arguments (line 734)
    kwargs_65528 = {}
    # Getting the type of 'os' (line 734)
    os_65518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 734, 10), 'os', False)
    # Obtaining the member 'path' of a type (line 734)
    path_65519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 10), os_65518, 'path')
    # Obtaining the member 'join' of a type (line 734)
    join_65520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 734, 10), path_65519, 'join')
    # Calling join(args, kwargs) (line 734)
    join_call_result_65529 = invoke(stypy.reporting.localization.Localization(__file__, 734, 10), join_65520, *[dirname_call_result_65526, str_65527], **kwargs_65528)
    
    # Assigning a type to the variable 'pys' (line 734)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 734, 4), 'pys', join_call_result_65529)
    
    # Assigning a Dict to a Name (line 735):
    
    # Assigning a Dict to a Name (line 735):
    
    # Obtaining an instance of the builtin type 'dict' (line 735)
    dict_65530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 735, 22), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 735)
    
    # Assigning a type to the variable 'fcompiler_class' (line 735)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 4), 'fcompiler_class', dict_65530)
    
    # Assigning a Dict to a Name (line 736):
    
    # Assigning a Dict to a Name (line 736):
    
    # Obtaining an instance of the builtin type 'dict' (line 736)
    dict_65531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 736, 24), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 736)
    
    # Assigning a type to the variable 'fcompiler_aliases' (line 736)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 4), 'fcompiler_aliases', dict_65531)
    
    
    # Call to glob(...): (line 737)
    # Processing the call arguments (line 737)
    # Getting the type of 'pys' (line 737)
    pys_65533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 22), 'pys', False)
    # Processing the call keyword arguments (line 737)
    kwargs_65534 = {}
    # Getting the type of 'glob' (line 737)
    glob_65532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 17), 'glob', False)
    # Calling glob(args, kwargs) (line 737)
    glob_call_result_65535 = invoke(stypy.reporting.localization.Localization(__file__, 737, 17), glob_65532, *[pys_65533], **kwargs_65534)
    
    # Testing the type of a for loop iterable (line 737)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 737, 4), glob_call_result_65535)
    # Getting the type of the for loop variable (line 737)
    for_loop_var_65536 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 737, 4), glob_call_result_65535)
    # Assigning a type to the variable 'fname' (line 737)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 4), 'fname', for_loop_var_65536)
    # SSA begins for a for statement (line 737)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Tuple (line 738):
    
    # Assigning a Call to a Name:
    
    # Call to splitext(...): (line 738)
    # Processing the call arguments (line 738)
    
    # Call to basename(...): (line 738)
    # Processing the call arguments (line 738)
    # Getting the type of 'fname' (line 738)
    fname_65543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 61), 'fname', False)
    # Processing the call keyword arguments (line 738)
    kwargs_65544 = {}
    # Getting the type of 'os' (line 738)
    os_65540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 44), 'os', False)
    # Obtaining the member 'path' of a type (line 738)
    path_65541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 44), os_65540, 'path')
    # Obtaining the member 'basename' of a type (line 738)
    basename_65542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 44), path_65541, 'basename')
    # Calling basename(args, kwargs) (line 738)
    basename_call_result_65545 = invoke(stypy.reporting.localization.Localization(__file__, 738, 44), basename_65542, *[fname_65543], **kwargs_65544)
    
    # Processing the call keyword arguments (line 738)
    kwargs_65546 = {}
    # Getting the type of 'os' (line 738)
    os_65537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 27), 'os', False)
    # Obtaining the member 'path' of a type (line 738)
    path_65538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 27), os_65537, 'path')
    # Obtaining the member 'splitext' of a type (line 738)
    splitext_65539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 27), path_65538, 'splitext')
    # Calling splitext(args, kwargs) (line 738)
    splitext_call_result_65547 = invoke(stypy.reporting.localization.Localization(__file__, 738, 27), splitext_65539, *[basename_call_result_65545], **kwargs_65546)
    
    # Assigning a type to the variable 'call_assignment_63491' (line 738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'call_assignment_63491', splitext_call_result_65547)
    
    # Assigning a Call to a Name (line 738):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_65550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 8), 'int')
    # Processing the call keyword arguments
    kwargs_65551 = {}
    # Getting the type of 'call_assignment_63491' (line 738)
    call_assignment_63491_65548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'call_assignment_63491', False)
    # Obtaining the member '__getitem__' of a type (line 738)
    getitem___65549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 8), call_assignment_63491_65548, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_65552 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___65549, *[int_65550], **kwargs_65551)
    
    # Assigning a type to the variable 'call_assignment_63492' (line 738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'call_assignment_63492', getitem___call_result_65552)
    
    # Assigning a Name to a Name (line 738):
    # Getting the type of 'call_assignment_63492' (line 738)
    call_assignment_63492_65553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'call_assignment_63492')
    # Assigning a type to the variable 'module_name' (line 738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'module_name', call_assignment_63492_65553)
    
    # Assigning a Call to a Name (line 738):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_65556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 8), 'int')
    # Processing the call keyword arguments
    kwargs_65557 = {}
    # Getting the type of 'call_assignment_63491' (line 738)
    call_assignment_63491_65554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'call_assignment_63491', False)
    # Obtaining the member '__getitem__' of a type (line 738)
    getitem___65555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 738, 8), call_assignment_63491_65554, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_65558 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___65555, *[int_65556], **kwargs_65557)
    
    # Assigning a type to the variable 'call_assignment_63493' (line 738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'call_assignment_63493', getitem___call_result_65558)
    
    # Assigning a Name to a Name (line 738):
    # Getting the type of 'call_assignment_63493' (line 738)
    call_assignment_63493_65559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 8), 'call_assignment_63493')
    # Assigning a type to the variable 'ext' (line 738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 21), 'ext', call_assignment_63493_65559)
    
    # Assigning a BinOp to a Name (line 739):
    
    # Assigning a BinOp to a Name (line 739):
    str_65560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 739, 22), 'str', 'numpy.distutils.fcompiler.')
    # Getting the type of 'module_name' (line 739)
    module_name_65561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 739, 53), 'module_name')
    # Applying the binary operator '+' (line 739)
    result_add_65562 = python_operator(stypy.reporting.localization.Localization(__file__, 739, 22), '+', str_65560, module_name_65561)
    
    # Assigning a type to the variable 'module_name' (line 739)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 739, 8), 'module_name', result_add_65562)
    
    # Call to __import__(...): (line 740)
    # Processing the call arguments (line 740)
    # Getting the type of 'module_name' (line 740)
    module_name_65564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 20), 'module_name', False)
    # Processing the call keyword arguments (line 740)
    kwargs_65565 = {}
    # Getting the type of '__import__' (line 740)
    import___65563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 740, 8), '__import__', False)
    # Calling __import__(args, kwargs) (line 740)
    import___call_result_65566 = invoke(stypy.reporting.localization.Localization(__file__, 740, 8), import___65563, *[module_name_65564], **kwargs_65565)
    
    
    # Assigning a Subscript to a Name (line 741):
    
    # Assigning a Subscript to a Name (line 741):
    
    # Obtaining the type of the subscript
    # Getting the type of 'module_name' (line 741)
    module_name_65567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 29), 'module_name')
    # Getting the type of 'sys' (line 741)
    sys_65568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 17), 'sys')
    # Obtaining the member 'modules' of a type (line 741)
    modules_65569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 17), sys_65568, 'modules')
    # Obtaining the member '__getitem__' of a type (line 741)
    getitem___65570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 741, 17), modules_65569, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 741)
    subscript_call_result_65571 = invoke(stypy.reporting.localization.Localization(__file__, 741, 17), getitem___65570, module_name_65567)
    
    # Assigning a type to the variable 'module' (line 741)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 8), 'module', subscript_call_result_65571)
    
    # Type idiom detected: calculating its left and rigth part (line 742)
    str_65572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 27), 'str', 'compilers')
    # Getting the type of 'module' (line 742)
    module_65573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 19), 'module')
    
    (may_be_65574, more_types_in_union_65575) = may_provide_member(str_65572, module_65573)

    if may_be_65574:

        if more_types_in_union_65575:
            # Runtime conditional SSA (line 742)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'module' (line 742)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 8), 'module', remove_not_member_provider_from_union(module_65573, 'compilers'))
        
        # Getting the type of 'module' (line 743)
        module_65576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 25), 'module')
        # Obtaining the member 'compilers' of a type (line 743)
        compilers_65577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 25), module_65576, 'compilers')
        # Testing the type of a for loop iterable (line 743)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 743, 12), compilers_65577)
        # Getting the type of the for loop variable (line 743)
        for_loop_var_65578 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 743, 12), compilers_65577)
        # Assigning a type to the variable 'cname' (line 743)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 12), 'cname', for_loop_var_65578)
        # SSA begins for a for statement (line 743)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 744):
        
        # Assigning a Call to a Name (line 744):
        
        # Call to getattr(...): (line 744)
        # Processing the call arguments (line 744)
        # Getting the type of 'module' (line 744)
        module_65580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 32), 'module', False)
        # Getting the type of 'cname' (line 744)
        cname_65581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 40), 'cname', False)
        # Processing the call keyword arguments (line 744)
        kwargs_65582 = {}
        # Getting the type of 'getattr' (line 744)
        getattr_65579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 24), 'getattr', False)
        # Calling getattr(args, kwargs) (line 744)
        getattr_call_result_65583 = invoke(stypy.reporting.localization.Localization(__file__, 744, 24), getattr_65579, *[module_65580, cname_65581], **kwargs_65582)
        
        # Assigning a type to the variable 'klass' (line 744)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 16), 'klass', getattr_call_result_65583)
        
        # Assigning a Tuple to a Name (line 745):
        
        # Assigning a Tuple to a Name (line 745):
        
        # Obtaining an instance of the builtin type 'tuple' (line 745)
        tuple_65584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 745, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 745)
        # Adding element type (line 745)
        # Getting the type of 'klass' (line 745)
        klass_65585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 24), 'klass')
        # Obtaining the member 'compiler_type' of a type (line 745)
        compiler_type_65586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 24), klass_65585, 'compiler_type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 24), tuple_65584, compiler_type_65586)
        # Adding element type (line 745)
        # Getting the type of 'klass' (line 745)
        klass_65587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 45), 'klass')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 24), tuple_65584, klass_65587)
        # Adding element type (line 745)
        # Getting the type of 'klass' (line 745)
        klass_65588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 745, 52), 'klass')
        # Obtaining the member 'description' of a type (line 745)
        description_65589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 745, 52), klass_65588, 'description')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 745, 24), tuple_65584, description_65589)
        
        # Assigning a type to the variable 'desc' (line 745)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 745, 16), 'desc', tuple_65584)
        
        # Assigning a Name to a Subscript (line 746):
        
        # Assigning a Name to a Subscript (line 746):
        # Getting the type of 'desc' (line 746)
        desc_65590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 55), 'desc')
        # Getting the type of 'fcompiler_class' (line 746)
        fcompiler_class_65591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 16), 'fcompiler_class')
        # Getting the type of 'klass' (line 746)
        klass_65592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 746, 32), 'klass')
        # Obtaining the member 'compiler_type' of a type (line 746)
        compiler_type_65593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 746, 32), klass_65592, 'compiler_type')
        # Storing an element on a container (line 746)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 746, 16), fcompiler_class_65591, (compiler_type_65593, desc_65590))
        
        # Getting the type of 'klass' (line 747)
        klass_65594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 29), 'klass')
        # Obtaining the member 'compiler_aliases' of a type (line 747)
        compiler_aliases_65595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 747, 29), klass_65594, 'compiler_aliases')
        # Testing the type of a for loop iterable (line 747)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 747, 16), compiler_aliases_65595)
        # Getting the type of the for loop variable (line 747)
        for_loop_var_65596 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 747, 16), compiler_aliases_65595)
        # Assigning a type to the variable 'alias' (line 747)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 16), 'alias', for_loop_var_65596)
        # SSA begins for a for statement (line 747)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'alias' (line 748)
        alias_65597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 23), 'alias')
        # Getting the type of 'fcompiler_aliases' (line 748)
        fcompiler_aliases_65598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 32), 'fcompiler_aliases')
        # Applying the binary operator 'in' (line 748)
        result_contains_65599 = python_operator(stypy.reporting.localization.Localization(__file__, 748, 23), 'in', alias_65597, fcompiler_aliases_65598)
        
        # Testing the type of an if condition (line 748)
        if_condition_65600 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 748, 20), result_contains_65599)
        # Assigning a type to the variable 'if_condition_65600' (line 748)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 20), 'if_condition_65600', if_condition_65600)
        # SSA begins for if statement (line 748)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 749)
        # Processing the call arguments (line 749)
        str_65602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 41), 'str', 'alias %r defined for both %s and %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 750)
        tuple_65603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 750, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 750)
        # Adding element type (line 750)
        # Getting the type of 'alias' (line 750)
        alias_65604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 44), 'alias', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 44), tuple_65603, alias_65604)
        # Adding element type (line 750)
        # Getting the type of 'klass' (line 750)
        klass_65605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 51), 'klass', False)
        # Obtaining the member '__name__' of a type (line 750)
        name___65606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 750, 51), klass_65605, '__name__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 44), tuple_65603, name___65606)
        # Adding element type (line 750)
        
        # Obtaining the type of the subscript
        int_65607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 69), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'alias' (line 751)
        alias_65608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 62), 'alias', False)
        # Getting the type of 'fcompiler_aliases' (line 751)
        fcompiler_aliases_65609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 44), 'fcompiler_aliases', False)
        # Obtaining the member '__getitem__' of a type (line 751)
        getitem___65610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 44), fcompiler_aliases_65609, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 751)
        subscript_call_result_65611 = invoke(stypy.reporting.localization.Localization(__file__, 751, 44), getitem___65610, alias_65608)
        
        # Obtaining the member '__getitem__' of a type (line 751)
        getitem___65612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 44), subscript_call_result_65611, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 751)
        subscript_call_result_65613 = invoke(stypy.reporting.localization.Localization(__file__, 751, 44), getitem___65612, int_65607)
        
        # Obtaining the member '__name__' of a type (line 751)
        name___65614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 751, 44), subscript_call_result_65613, '__name__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 750, 44), tuple_65603, name___65614)
        
        # Applying the binary operator '%' (line 749)
        result_mod_65615 = python_operator(stypy.reporting.localization.Localization(__file__, 749, 41), '%', str_65602, tuple_65603)
        
        # Processing the call keyword arguments (line 749)
        kwargs_65616 = {}
        # Getting the type of 'ValueError' (line 749)
        ValueError_65601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 30), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 749)
        ValueError_call_result_65617 = invoke(stypy.reporting.localization.Localization(__file__, 749, 30), ValueError_65601, *[result_mod_65615], **kwargs_65616)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 749, 24), ValueError_call_result_65617, 'raise parameter', BaseException)
        # SSA join for if statement (line 748)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 752):
        
        # Assigning a Name to a Subscript (line 752):
        # Getting the type of 'desc' (line 752)
        desc_65618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 47), 'desc')
        # Getting the type of 'fcompiler_aliases' (line 752)
        fcompiler_aliases_65619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 20), 'fcompiler_aliases')
        # Getting the type of 'alias' (line 752)
        alias_65620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 38), 'alias')
        # Storing an element on a container (line 752)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 752, 20), fcompiler_aliases_65619, (alias_65620, desc_65618))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_65575:
            # SSA join for if statement (line 742)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'load_all_fcompiler_classes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'load_all_fcompiler_classes' in the type store
    # Getting the type of 'stypy_return_type' (line 726)
    stypy_return_type_65621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_65621)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'load_all_fcompiler_classes'
    return stypy_return_type_65621

# Assigning a type to the variable 'load_all_fcompiler_classes' (line 726)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 0), 'load_all_fcompiler_classes', load_all_fcompiler_classes)

@norecursion
def _find_existing_fcompiler(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 755)
    None_65622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 36), 'None')
    # Getting the type of 'None' (line 755)
    None_65623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 51), 'None')
    # Getting the type of 'False' (line 756)
    False_65624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 40), 'False')
    # Getting the type of 'None' (line 757)
    None_65625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 757, 40), 'None')
    defaults = [None_65622, None_65623, False_65624, None_65625]
    # Create a new context for function '_find_existing_fcompiler'
    module_type_store = module_type_store.open_function_context('_find_existing_fcompiler', 754, 0, False)
    
    # Passed parameters checking function
    _find_existing_fcompiler.stypy_localization = localization
    _find_existing_fcompiler.stypy_type_of_self = None
    _find_existing_fcompiler.stypy_type_store = module_type_store
    _find_existing_fcompiler.stypy_function_name = '_find_existing_fcompiler'
    _find_existing_fcompiler.stypy_param_names_list = ['compiler_types', 'osname', 'platform', 'requiref90', 'c_compiler']
    _find_existing_fcompiler.stypy_varargs_param_name = None
    _find_existing_fcompiler.stypy_kwargs_param_name = None
    _find_existing_fcompiler.stypy_call_defaults = defaults
    _find_existing_fcompiler.stypy_call_varargs = varargs
    _find_existing_fcompiler.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_find_existing_fcompiler', ['compiler_types', 'osname', 'platform', 'requiref90', 'c_compiler'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_find_existing_fcompiler', localization, ['compiler_types', 'osname', 'platform', 'requiref90', 'c_compiler'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_find_existing_fcompiler(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 758, 4))
    
    # 'from numpy.distutils.core import get_distribution' statement (line 758)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    import_65626 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 758, 4), 'numpy.distutils.core')

    if (type(import_65626) is not StypyTypeError):

        if (import_65626 != 'pyd_module'):
            __import__(import_65626)
            sys_modules_65627 = sys.modules[import_65626]
            import_from_module(stypy.reporting.localization.Localization(__file__, 758, 4), 'numpy.distutils.core', sys_modules_65627.module_type_store, module_type_store, ['get_distribution'])
            nest_module(stypy.reporting.localization.Localization(__file__, 758, 4), __file__, sys_modules_65627, sys_modules_65627.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import get_distribution

            import_from_module(stypy.reporting.localization.Localization(__file__, 758, 4), 'numpy.distutils.core', None, module_type_store, ['get_distribution'], [get_distribution])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 758)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 4), 'numpy.distutils.core', import_65626)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    
    
    # Assigning a Call to a Name (line 759):
    
    # Assigning a Call to a Name (line 759):
    
    # Call to get_distribution(...): (line 759)
    # Processing the call keyword arguments (line 759)
    # Getting the type of 'True' (line 759)
    True_65629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 35), 'True', False)
    keyword_65630 = True_65629
    kwargs_65631 = {'always': keyword_65630}
    # Getting the type of 'get_distribution' (line 759)
    get_distribution_65628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 11), 'get_distribution', False)
    # Calling get_distribution(args, kwargs) (line 759)
    get_distribution_call_result_65632 = invoke(stypy.reporting.localization.Localization(__file__, 759, 11), get_distribution_65628, *[], **kwargs_65631)
    
    # Assigning a type to the variable 'dist' (line 759)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 4), 'dist', get_distribution_call_result_65632)
    
    # Getting the type of 'compiler_types' (line 760)
    compiler_types_65633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 25), 'compiler_types')
    # Testing the type of a for loop iterable (line 760)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 760, 4), compiler_types_65633)
    # Getting the type of the for loop variable (line 760)
    for_loop_var_65634 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 760, 4), compiler_types_65633)
    # Assigning a type to the variable 'compiler_type' (line 760)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 4), 'compiler_type', for_loop_var_65634)
    # SSA begins for a for statement (line 760)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 761):
    
    # Assigning a Name to a Name (line 761):
    # Getting the type of 'None' (line 761)
    None_65635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 761, 12), 'None')
    # Assigning a type to the variable 'v' (line 761)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 761, 8), 'v', None_65635)
    
    
    # SSA begins for try-except statement (line 762)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 763):
    
    # Assigning a Call to a Name (line 763):
    
    # Call to new_fcompiler(...): (line 763)
    # Processing the call keyword arguments (line 763)
    # Getting the type of 'platform' (line 763)
    platform_65637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 35), 'platform', False)
    keyword_65638 = platform_65637
    # Getting the type of 'compiler_type' (line 763)
    compiler_type_65639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 54), 'compiler_type', False)
    keyword_65640 = compiler_type_65639
    # Getting the type of 'c_compiler' (line 764)
    c_compiler_65641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 764, 41), 'c_compiler', False)
    keyword_65642 = c_compiler_65641
    kwargs_65643 = {'compiler': keyword_65640, 'c_compiler': keyword_65642, 'plat': keyword_65638}
    # Getting the type of 'new_fcompiler' (line 763)
    new_fcompiler_65636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 16), 'new_fcompiler', False)
    # Calling new_fcompiler(args, kwargs) (line 763)
    new_fcompiler_call_result_65644 = invoke(stypy.reporting.localization.Localization(__file__, 763, 16), new_fcompiler_65636, *[], **kwargs_65643)
    
    # Assigning a type to the variable 'c' (line 763)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 12), 'c', new_fcompiler_call_result_65644)
    
    # Call to customize(...): (line 765)
    # Processing the call arguments (line 765)
    # Getting the type of 'dist' (line 765)
    dist_65647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 24), 'dist', False)
    # Processing the call keyword arguments (line 765)
    kwargs_65648 = {}
    # Getting the type of 'c' (line 765)
    c_65645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 765, 12), 'c', False)
    # Obtaining the member 'customize' of a type (line 765)
    customize_65646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 765, 12), c_65645, 'customize')
    # Calling customize(args, kwargs) (line 765)
    customize_call_result_65649 = invoke(stypy.reporting.localization.Localization(__file__, 765, 12), customize_65646, *[dist_65647], **kwargs_65648)
    
    
    # Assigning a Call to a Name (line 766):
    
    # Assigning a Call to a Name (line 766):
    
    # Call to get_version(...): (line 766)
    # Processing the call keyword arguments (line 766)
    kwargs_65652 = {}
    # Getting the type of 'c' (line 766)
    c_65650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 16), 'c', False)
    # Obtaining the member 'get_version' of a type (line 766)
    get_version_65651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 766, 16), c_65650, 'get_version')
    # Calling get_version(args, kwargs) (line 766)
    get_version_call_result_65653 = invoke(stypy.reporting.localization.Localization(__file__, 766, 16), get_version_65651, *[], **kwargs_65652)
    
    # Assigning a type to the variable 'v' (line 766)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 12), 'v', get_version_call_result_65653)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'requiref90' (line 767)
    requiref90_65654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 15), 'requiref90')
    
    # Getting the type of 'c' (line 767)
    c_65655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 30), 'c')
    # Obtaining the member 'compiler_f90' of a type (line 767)
    compiler_f90_65656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 30), c_65655, 'compiler_f90')
    # Getting the type of 'None' (line 767)
    None_65657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 48), 'None')
    # Applying the binary operator 'is' (line 767)
    result_is__65658 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 30), 'is', compiler_f90_65656, None_65657)
    
    # Applying the binary operator 'and' (line 767)
    result_and_keyword_65659 = python_operator(stypy.reporting.localization.Localization(__file__, 767, 15), 'and', requiref90_65654, result_is__65658)
    
    # Testing the type of an if condition (line 767)
    if_condition_65660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 767, 12), result_and_keyword_65659)
    # Assigning a type to the variable 'if_condition_65660' (line 767)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 12), 'if_condition_65660', if_condition_65660)
    # SSA begins for if statement (line 767)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 768):
    
    # Assigning a Name to a Name (line 768):
    # Getting the type of 'None' (line 768)
    None_65661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 20), 'None')
    # Assigning a type to the variable 'v' (line 768)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 16), 'v', None_65661)
    
    # Assigning a Attribute to a Name (line 769):
    
    # Assigning a Attribute to a Name (line 769):
    # Getting the type of 'c' (line 769)
    c_65662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 31), 'c')
    # Obtaining the member 'suggested_f90_compiler' of a type (line 769)
    suggested_f90_compiler_65663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 769, 31), c_65662, 'suggested_f90_compiler')
    # Assigning a type to the variable 'new_compiler' (line 769)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 16), 'new_compiler', suggested_f90_compiler_65663)
    
    # Getting the type of 'new_compiler' (line 770)
    new_compiler_65664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 19), 'new_compiler')
    # Testing the type of an if condition (line 770)
    if_condition_65665 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 770, 16), new_compiler_65664)
    # Assigning a type to the variable 'if_condition_65665' (line 770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 16), 'if_condition_65665', if_condition_65665)
    # SSA begins for if statement (line 770)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 771)
    # Processing the call arguments (line 771)
    str_65668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 771, 29), 'str', 'Trying %r compiler as suggested by %r compiler for f90 support.')
    
    # Obtaining an instance of the builtin type 'tuple' (line 772)
    tuple_65669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 60), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 772)
    # Adding element type (line 772)
    # Getting the type of 'compiler_type' (line 772)
    compiler_type_65670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 60), 'compiler_type', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 772, 60), tuple_65669, compiler_type_65670)
    # Adding element type (line 772)
    # Getting the type of 'new_compiler' (line 773)
    new_compiler_65671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 60), 'new_compiler', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 772, 60), tuple_65669, new_compiler_65671)
    
    # Applying the binary operator '%' (line 771)
    result_mod_65672 = python_operator(stypy.reporting.localization.Localization(__file__, 771, 29), '%', str_65668, tuple_65669)
    
    # Processing the call keyword arguments (line 771)
    kwargs_65673 = {}
    # Getting the type of 'log' (line 771)
    log_65666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 20), 'log', False)
    # Obtaining the member 'warn' of a type (line 771)
    warn_65667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 20), log_65666, 'warn')
    # Calling warn(args, kwargs) (line 771)
    warn_call_result_65674 = invoke(stypy.reporting.localization.Localization(__file__, 771, 20), warn_65667, *[result_mod_65672], **kwargs_65673)
    
    
    # Assigning a Call to a Name (line 774):
    
    # Assigning a Call to a Name (line 774):
    
    # Call to new_fcompiler(...): (line 774)
    # Processing the call keyword arguments (line 774)
    # Getting the type of 'platform' (line 774)
    platform_65676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 43), 'platform', False)
    keyword_65677 = platform_65676
    # Getting the type of 'new_compiler' (line 774)
    new_compiler_65678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 62), 'new_compiler', False)
    keyword_65679 = new_compiler_65678
    # Getting the type of 'c_compiler' (line 775)
    c_compiler_65680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 49), 'c_compiler', False)
    keyword_65681 = c_compiler_65680
    kwargs_65682 = {'compiler': keyword_65679, 'c_compiler': keyword_65681, 'plat': keyword_65677}
    # Getting the type of 'new_fcompiler' (line 774)
    new_fcompiler_65675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 24), 'new_fcompiler', False)
    # Calling new_fcompiler(args, kwargs) (line 774)
    new_fcompiler_call_result_65683 = invoke(stypy.reporting.localization.Localization(__file__, 774, 24), new_fcompiler_65675, *[], **kwargs_65682)
    
    # Assigning a type to the variable 'c' (line 774)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 20), 'c', new_fcompiler_call_result_65683)
    
    # Call to customize(...): (line 776)
    # Processing the call arguments (line 776)
    # Getting the type of 'dist' (line 776)
    dist_65686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 32), 'dist', False)
    # Processing the call keyword arguments (line 776)
    kwargs_65687 = {}
    # Getting the type of 'c' (line 776)
    c_65684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 20), 'c', False)
    # Obtaining the member 'customize' of a type (line 776)
    customize_65685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 20), c_65684, 'customize')
    # Calling customize(args, kwargs) (line 776)
    customize_call_result_65688 = invoke(stypy.reporting.localization.Localization(__file__, 776, 20), customize_65685, *[dist_65686], **kwargs_65687)
    
    
    # Assigning a Call to a Name (line 777):
    
    # Assigning a Call to a Name (line 777):
    
    # Call to get_version(...): (line 777)
    # Processing the call keyword arguments (line 777)
    kwargs_65691 = {}
    # Getting the type of 'c' (line 777)
    c_65689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 24), 'c', False)
    # Obtaining the member 'get_version' of a type (line 777)
    get_version_65690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 24), c_65689, 'get_version')
    # Calling get_version(args, kwargs) (line 777)
    get_version_call_result_65692 = invoke(stypy.reporting.localization.Localization(__file__, 777, 24), get_version_65690, *[], **kwargs_65691)
    
    # Assigning a type to the variable 'v' (line 777)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 777, 20), 'v', get_version_call_result_65692)
    
    # Type idiom detected: calculating its left and rigth part (line 778)
    # Getting the type of 'v' (line 778)
    v_65693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 20), 'v')
    # Getting the type of 'None' (line 778)
    None_65694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 32), 'None')
    
    (may_be_65695, more_types_in_union_65696) = may_not_be_none(v_65693, None_65694)

    if may_be_65695:

        if more_types_in_union_65696:
            # Runtime conditional SSA (line 778)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 779):
        
        # Assigning a Name to a Name (line 779):
        # Getting the type of 'new_compiler' (line 779)
        new_compiler_65697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 40), 'new_compiler')
        # Assigning a type to the variable 'compiler_type' (line 779)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 24), 'compiler_type', new_compiler_65697)

        if more_types_in_union_65696:
            # SSA join for if statement (line 778)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 770)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 767)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'requiref90' (line 780)
    requiref90_65698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 15), 'requiref90')
    
    # Getting the type of 'c' (line 780)
    c_65699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 30), 'c')
    # Obtaining the member 'compiler_f90' of a type (line 780)
    compiler_f90_65700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 780, 30), c_65699, 'compiler_f90')
    # Getting the type of 'None' (line 780)
    None_65701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 780, 48), 'None')
    # Applying the binary operator 'is' (line 780)
    result_is__65702 = python_operator(stypy.reporting.localization.Localization(__file__, 780, 30), 'is', compiler_f90_65700, None_65701)
    
    # Applying the binary operator 'and' (line 780)
    result_and_keyword_65703 = python_operator(stypy.reporting.localization.Localization(__file__, 780, 15), 'and', requiref90_65698, result_is__65702)
    
    # Testing the type of an if condition (line 780)
    if_condition_65704 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 780, 12), result_and_keyword_65703)
    # Assigning a type to the variable 'if_condition_65704' (line 780)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 780, 12), 'if_condition_65704', if_condition_65704)
    # SSA begins for if statement (line 780)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 781)
    # Processing the call arguments (line 781)
    str_65706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 781, 33), 'str', '%s does not support compiling f90 codes, skipping.')
    # Getting the type of 'c' (line 782)
    c_65707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 48), 'c', False)
    # Obtaining the member '__class__' of a type (line 782)
    class___65708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 48), c_65707, '__class__')
    # Obtaining the member '__name__' of a type (line 782)
    name___65709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 48), class___65708, '__name__')
    # Applying the binary operator '%' (line 781)
    result_mod_65710 = python_operator(stypy.reporting.localization.Localization(__file__, 781, 33), '%', str_65706, name___65709)
    
    # Processing the call keyword arguments (line 781)
    kwargs_65711 = {}
    # Getting the type of 'ValueError' (line 781)
    ValueError_65705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 781)
    ValueError_call_result_65712 = invoke(stypy.reporting.localization.Localization(__file__, 781, 22), ValueError_65705, *[result_mod_65710], **kwargs_65711)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 781, 16), ValueError_call_result_65712, 'raise parameter', BaseException)
    # SSA join for if statement (line 780)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 762)
    # SSA branch for the except 'DistutilsModuleError' branch of a try statement (line 762)
    module_type_store.open_ssa_branch('except')
    
    # Call to debug(...): (line 784)
    # Processing the call arguments (line 784)
    str_65715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 22), 'str', "_find_existing_fcompiler: compiler_type='%s' raised DistutilsModuleError")
    # Getting the type of 'compiler_type' (line 784)
    compiler_type_65716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 98), 'compiler_type', False)
    # Processing the call keyword arguments (line 784)
    kwargs_65717 = {}
    # Getting the type of 'log' (line 784)
    log_65713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 12), 'log', False)
    # Obtaining the member 'debug' of a type (line 784)
    debug_65714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 784, 12), log_65713, 'debug')
    # Calling debug(args, kwargs) (line 784)
    debug_call_result_65718 = invoke(stypy.reporting.localization.Localization(__file__, 784, 12), debug_65714, *[str_65715, compiler_type_65716], **kwargs_65717)
    
    # SSA branch for the except 'CompilerNotFound' branch of a try statement (line 762)
    module_type_store.open_ssa_branch('except')
    
    # Call to debug(...): (line 786)
    # Processing the call arguments (line 786)
    str_65721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 22), 'str', "_find_existing_fcompiler: compiler_type='%s' not found")
    # Getting the type of 'compiler_type' (line 786)
    compiler_type_65722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 80), 'compiler_type', False)
    # Processing the call keyword arguments (line 786)
    kwargs_65723 = {}
    # Getting the type of 'log' (line 786)
    log_65719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 12), 'log', False)
    # Obtaining the member 'debug' of a type (line 786)
    debug_65720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 12), log_65719, 'debug')
    # Calling debug(args, kwargs) (line 786)
    debug_call_result_65724 = invoke(stypy.reporting.localization.Localization(__file__, 786, 12), debug_65720, *[str_65721, compiler_type_65722], **kwargs_65723)
    
    # SSA join for try-except statement (line 762)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 787)
    # Getting the type of 'v' (line 787)
    v_65725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 8), 'v')
    # Getting the type of 'None' (line 787)
    None_65726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 20), 'None')
    
    (may_be_65727, more_types_in_union_65728) = may_not_be_none(v_65725, None_65726)

    if may_be_65727:

        if more_types_in_union_65728:
            # Runtime conditional SSA (line 787)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'compiler_type' (line 788)
        compiler_type_65729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 19), 'compiler_type')
        # Assigning a type to the variable 'stypy_return_type' (line 788)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 12), 'stypy_return_type', compiler_type_65729)

        if more_types_in_union_65728:
            # SSA join for if statement (line 787)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'None' (line 789)
    None_65730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 789)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 789, 4), 'stypy_return_type', None_65730)
    
    # ################# End of '_find_existing_fcompiler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_find_existing_fcompiler' in the type store
    # Getting the type of 'stypy_return_type' (line 754)
    stypy_return_type_65731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 754, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_65731)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_find_existing_fcompiler'
    return stypy_return_type_65731

# Assigning a type to the variable '_find_existing_fcompiler' (line 754)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 754, 0), '_find_existing_fcompiler', _find_existing_fcompiler)

@norecursion
def available_fcompilers_for_platform(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 791)
    None_65732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 45), 'None')
    # Getting the type of 'None' (line 791)
    None_65733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 60), 'None')
    defaults = [None_65732, None_65733]
    # Create a new context for function 'available_fcompilers_for_platform'
    module_type_store = module_type_store.open_function_context('available_fcompilers_for_platform', 791, 0, False)
    
    # Passed parameters checking function
    available_fcompilers_for_platform.stypy_localization = localization
    available_fcompilers_for_platform.stypy_type_of_self = None
    available_fcompilers_for_platform.stypy_type_store = module_type_store
    available_fcompilers_for_platform.stypy_function_name = 'available_fcompilers_for_platform'
    available_fcompilers_for_platform.stypy_param_names_list = ['osname', 'platform']
    available_fcompilers_for_platform.stypy_varargs_param_name = None
    available_fcompilers_for_platform.stypy_kwargs_param_name = None
    available_fcompilers_for_platform.stypy_call_defaults = defaults
    available_fcompilers_for_platform.stypy_call_varargs = varargs
    available_fcompilers_for_platform.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'available_fcompilers_for_platform', ['osname', 'platform'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'available_fcompilers_for_platform', localization, ['osname', 'platform'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'available_fcompilers_for_platform(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 792)
    # Getting the type of 'osname' (line 792)
    osname_65734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 7), 'osname')
    # Getting the type of 'None' (line 792)
    None_65735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 17), 'None')
    
    (may_be_65736, more_types_in_union_65737) = may_be_none(osname_65734, None_65735)

    if may_be_65736:

        if more_types_in_union_65737:
            # Runtime conditional SSA (line 792)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 793):
        
        # Assigning a Attribute to a Name (line 793):
        # Getting the type of 'os' (line 793)
        os_65738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 793, 17), 'os')
        # Obtaining the member 'name' of a type (line 793)
        name_65739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 793, 17), os_65738, 'name')
        # Assigning a type to the variable 'osname' (line 793)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 793, 8), 'osname', name_65739)

        if more_types_in_union_65737:
            # SSA join for if statement (line 792)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 794)
    # Getting the type of 'platform' (line 794)
    platform_65740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 7), 'platform')
    # Getting the type of 'None' (line 794)
    None_65741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 19), 'None')
    
    (may_be_65742, more_types_in_union_65743) = may_be_none(platform_65740, None_65741)

    if may_be_65742:

        if more_types_in_union_65743:
            # Runtime conditional SSA (line 794)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 795):
        
        # Assigning a Attribute to a Name (line 795):
        # Getting the type of 'sys' (line 795)
        sys_65744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 19), 'sys')
        # Obtaining the member 'platform' of a type (line 795)
        platform_65745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 795, 19), sys_65744, 'platform')
        # Assigning a type to the variable 'platform' (line 795)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 8), 'platform', platform_65745)

        if more_types_in_union_65743:
            # SSA join for if statement (line 794)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a List to a Name (line 796):
    
    # Assigning a List to a Name (line 796):
    
    # Obtaining an instance of the builtin type 'list' (line 796)
    list_65746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 796)
    
    # Assigning a type to the variable 'matching_compiler_types' (line 796)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), 'matching_compiler_types', list_65746)
    
    # Getting the type of '_default_compilers' (line 797)
    _default_compilers_65747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 34), '_default_compilers')
    # Testing the type of a for loop iterable (line 797)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 797, 4), _default_compilers_65747)
    # Getting the type of the for loop variable (line 797)
    for_loop_var_65748 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 797, 4), _default_compilers_65747)
    # Assigning a type to the variable 'pattern' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'pattern', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 797, 4), for_loop_var_65748))
    # Assigning a type to the variable 'compiler_type' (line 797)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'compiler_type', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 797, 4), for_loop_var_65748))
    # SSA begins for a for statement (line 797)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Call to match(...): (line 798)
    # Processing the call arguments (line 798)
    # Getting the type of 'pattern' (line 798)
    pattern_65751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 20), 'pattern', False)
    # Getting the type of 'platform' (line 798)
    platform_65752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 29), 'platform', False)
    # Processing the call keyword arguments (line 798)
    kwargs_65753 = {}
    # Getting the type of 're' (line 798)
    re_65749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 11), 're', False)
    # Obtaining the member 'match' of a type (line 798)
    match_65750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 11), re_65749, 'match')
    # Calling match(args, kwargs) (line 798)
    match_call_result_65754 = invoke(stypy.reporting.localization.Localization(__file__, 798, 11), match_65750, *[pattern_65751, platform_65752], **kwargs_65753)
    
    
    # Call to match(...): (line 798)
    # Processing the call arguments (line 798)
    # Getting the type of 'pattern' (line 798)
    pattern_65757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 51), 'pattern', False)
    # Getting the type of 'osname' (line 798)
    osname_65758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 60), 'osname', False)
    # Processing the call keyword arguments (line 798)
    kwargs_65759 = {}
    # Getting the type of 're' (line 798)
    re_65755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 42), 're', False)
    # Obtaining the member 'match' of a type (line 798)
    match_65756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 798, 42), re_65755, 'match')
    # Calling match(args, kwargs) (line 798)
    match_call_result_65760 = invoke(stypy.reporting.localization.Localization(__file__, 798, 42), match_65756, *[pattern_65757, osname_65758], **kwargs_65759)
    
    # Applying the binary operator 'or' (line 798)
    result_or_keyword_65761 = python_operator(stypy.reporting.localization.Localization(__file__, 798, 11), 'or', match_call_result_65754, match_call_result_65760)
    
    # Testing the type of an if condition (line 798)
    if_condition_65762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 798, 8), result_or_keyword_65761)
    # Assigning a type to the variable 'if_condition_65762' (line 798)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 8), 'if_condition_65762', if_condition_65762)
    # SSA begins for if statement (line 798)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'compiler_type' (line 799)
    compiler_type_65763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 22), 'compiler_type')
    # Testing the type of a for loop iterable (line 799)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 799, 12), compiler_type_65763)
    # Getting the type of the for loop variable (line 799)
    for_loop_var_65764 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 799, 12), compiler_type_65763)
    # Assigning a type to the variable 'ct' (line 799)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 12), 'ct', for_loop_var_65764)
    # SSA begins for a for statement (line 799)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'ct' (line 800)
    ct_65765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 19), 'ct')
    # Getting the type of 'matching_compiler_types' (line 800)
    matching_compiler_types_65766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 29), 'matching_compiler_types')
    # Applying the binary operator 'notin' (line 800)
    result_contains_65767 = python_operator(stypy.reporting.localization.Localization(__file__, 800, 19), 'notin', ct_65765, matching_compiler_types_65766)
    
    # Testing the type of an if condition (line 800)
    if_condition_65768 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 800, 16), result_contains_65767)
    # Assigning a type to the variable 'if_condition_65768' (line 800)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 800, 16), 'if_condition_65768', if_condition_65768)
    # SSA begins for if statement (line 800)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 801)
    # Processing the call arguments (line 801)
    # Getting the type of 'ct' (line 801)
    ct_65771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 51), 'ct', False)
    # Processing the call keyword arguments (line 801)
    kwargs_65772 = {}
    # Getting the type of 'matching_compiler_types' (line 801)
    matching_compiler_types_65769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 20), 'matching_compiler_types', False)
    # Obtaining the member 'append' of a type (line 801)
    append_65770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 801, 20), matching_compiler_types_65769, 'append')
    # Calling append(args, kwargs) (line 801)
    append_call_result_65773 = invoke(stypy.reporting.localization.Localization(__file__, 801, 20), append_65770, *[ct_65771], **kwargs_65772)
    
    # SSA join for if statement (line 800)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 798)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'matching_compiler_types' (line 802)
    matching_compiler_types_65774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 802, 11), 'matching_compiler_types')
    # Applying the 'not' unary operator (line 802)
    result_not__65775 = python_operator(stypy.reporting.localization.Localization(__file__, 802, 7), 'not', matching_compiler_types_65774)
    
    # Testing the type of an if condition (line 802)
    if_condition_65776 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 802, 4), result_not__65775)
    # Assigning a type to the variable 'if_condition_65776' (line 802)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 802, 4), 'if_condition_65776', if_condition_65776)
    # SSA begins for if statement (line 802)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 803)
    # Processing the call arguments (line 803)
    str_65779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 39), 'str', 'gnu')
    # Processing the call keyword arguments (line 803)
    kwargs_65780 = {}
    # Getting the type of 'matching_compiler_types' (line 803)
    matching_compiler_types_65777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 8), 'matching_compiler_types', False)
    # Obtaining the member 'append' of a type (line 803)
    append_65778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 8), matching_compiler_types_65777, 'append')
    # Calling append(args, kwargs) (line 803)
    append_call_result_65781 = invoke(stypy.reporting.localization.Localization(__file__, 803, 8), append_65778, *[str_65779], **kwargs_65780)
    
    # SSA join for if statement (line 802)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'matching_compiler_types' (line 804)
    matching_compiler_types_65782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 11), 'matching_compiler_types')
    # Assigning a type to the variable 'stypy_return_type' (line 804)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'stypy_return_type', matching_compiler_types_65782)
    
    # ################# End of 'available_fcompilers_for_platform(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'available_fcompilers_for_platform' in the type store
    # Getting the type of 'stypy_return_type' (line 791)
    stypy_return_type_65783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_65783)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'available_fcompilers_for_platform'
    return stypy_return_type_65783

# Assigning a type to the variable 'available_fcompilers_for_platform' (line 791)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 791, 0), 'available_fcompilers_for_platform', available_fcompilers_for_platform)

@norecursion
def get_default_fcompiler(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 806)
    None_65784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 33), 'None')
    # Getting the type of 'None' (line 806)
    None_65785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 48), 'None')
    # Getting the type of 'False' (line 806)
    False_65786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 65), 'False')
    # Getting the type of 'None' (line 807)
    None_65787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 37), 'None')
    defaults = [None_65784, None_65785, False_65786, None_65787]
    # Create a new context for function 'get_default_fcompiler'
    module_type_store = module_type_store.open_function_context('get_default_fcompiler', 806, 0, False)
    
    # Passed parameters checking function
    get_default_fcompiler.stypy_localization = localization
    get_default_fcompiler.stypy_type_of_self = None
    get_default_fcompiler.stypy_type_store = module_type_store
    get_default_fcompiler.stypy_function_name = 'get_default_fcompiler'
    get_default_fcompiler.stypy_param_names_list = ['osname', 'platform', 'requiref90', 'c_compiler']
    get_default_fcompiler.stypy_varargs_param_name = None
    get_default_fcompiler.stypy_kwargs_param_name = None
    get_default_fcompiler.stypy_call_defaults = defaults
    get_default_fcompiler.stypy_call_varargs = varargs
    get_default_fcompiler.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_default_fcompiler', ['osname', 'platform', 'requiref90', 'c_compiler'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_default_fcompiler', localization, ['osname', 'platform', 'requiref90', 'c_compiler'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_default_fcompiler(...)' code ##################

    str_65788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, (-1)), 'str', 'Determine the default Fortran compiler to use for the given\n    platform.')
    
    # Assigning a Call to a Name (line 810):
    
    # Assigning a Call to a Name (line 810):
    
    # Call to available_fcompilers_for_platform(...): (line 810)
    # Processing the call arguments (line 810)
    # Getting the type of 'osname' (line 810)
    osname_65790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 64), 'osname', False)
    # Getting the type of 'platform' (line 811)
    platform_65791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 64), 'platform', False)
    # Processing the call keyword arguments (line 810)
    kwargs_65792 = {}
    # Getting the type of 'available_fcompilers_for_platform' (line 810)
    available_fcompilers_for_platform_65789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 810, 30), 'available_fcompilers_for_platform', False)
    # Calling available_fcompilers_for_platform(args, kwargs) (line 810)
    available_fcompilers_for_platform_call_result_65793 = invoke(stypy.reporting.localization.Localization(__file__, 810, 30), available_fcompilers_for_platform_65789, *[osname_65790, platform_65791], **kwargs_65792)
    
    # Assigning a type to the variable 'matching_compiler_types' (line 810)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 810, 4), 'matching_compiler_types', available_fcompilers_for_platform_call_result_65793)
    
    # Assigning a Call to a Name (line 812):
    
    # Assigning a Call to a Name (line 812):
    
    # Call to _find_existing_fcompiler(...): (line 812)
    # Processing the call arguments (line 812)
    # Getting the type of 'matching_compiler_types' (line 812)
    matching_compiler_types_65795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 46), 'matching_compiler_types', False)
    # Processing the call keyword arguments (line 812)
    # Getting the type of 'osname' (line 813)
    osname_65796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 53), 'osname', False)
    keyword_65797 = osname_65796
    # Getting the type of 'platform' (line 814)
    platform_65798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 55), 'platform', False)
    keyword_65799 = platform_65798
    # Getting the type of 'requiref90' (line 815)
    requiref90_65800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 57), 'requiref90', False)
    keyword_65801 = requiref90_65800
    # Getting the type of 'c_compiler' (line 816)
    c_compiler_65802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 57), 'c_compiler', False)
    keyword_65803 = c_compiler_65802
    kwargs_65804 = {'platform': keyword_65799, 'c_compiler': keyword_65803, 'requiref90': keyword_65801, 'osname': keyword_65797}
    # Getting the type of '_find_existing_fcompiler' (line 812)
    _find_existing_fcompiler_65794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 21), '_find_existing_fcompiler', False)
    # Calling _find_existing_fcompiler(args, kwargs) (line 812)
    _find_existing_fcompiler_call_result_65805 = invoke(stypy.reporting.localization.Localization(__file__, 812, 21), _find_existing_fcompiler_65794, *[matching_compiler_types_65795], **kwargs_65804)
    
    # Assigning a type to the variable 'compiler_type' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 4), 'compiler_type', _find_existing_fcompiler_call_result_65805)
    # Getting the type of 'compiler_type' (line 817)
    compiler_type_65806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 11), 'compiler_type')
    # Assigning a type to the variable 'stypy_return_type' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 4), 'stypy_return_type', compiler_type_65806)
    
    # ################# End of 'get_default_fcompiler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_default_fcompiler' in the type store
    # Getting the type of 'stypy_return_type' (line 806)
    stypy_return_type_65807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_65807)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_default_fcompiler'
    return stypy_return_type_65807

# Assigning a type to the variable 'get_default_fcompiler' (line 806)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 0), 'get_default_fcompiler', get_default_fcompiler)

# Assigning a Call to a Name (line 820):

# Assigning a Call to a Name (line 820):

# Call to set(...): (line 820)
# Processing the call keyword arguments (line 820)
kwargs_65809 = {}
# Getting the type of 'set' (line 820)
set_65808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 20), 'set', False)
# Calling set(args, kwargs) (line 820)
set_call_result_65810 = invoke(stypy.reporting.localization.Localization(__file__, 820, 20), set_65808, *[], **kwargs_65809)

# Assigning a type to the variable 'failed_fcompilers' (line 820)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 820, 0), 'failed_fcompilers', set_call_result_65810)

@norecursion
def new_fcompiler(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 822)
    None_65811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 23), 'None')
    # Getting the type of 'None' (line 823)
    None_65812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 27), 'None')
    int_65813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 824, 26), 'int')
    int_65814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 26), 'int')
    int_65815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 24), 'int')
    # Getting the type of 'False' (line 827)
    False_65816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 29), 'False')
    # Getting the type of 'None' (line 828)
    None_65817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 31), 'None')
    defaults = [None_65811, None_65812, int_65813, int_65814, int_65815, False_65816, None_65817]
    # Create a new context for function 'new_fcompiler'
    module_type_store = module_type_store.open_function_context('new_fcompiler', 822, 0, False)
    
    # Passed parameters checking function
    new_fcompiler.stypy_localization = localization
    new_fcompiler.stypy_type_of_self = None
    new_fcompiler.stypy_type_store = module_type_store
    new_fcompiler.stypy_function_name = 'new_fcompiler'
    new_fcompiler.stypy_param_names_list = ['plat', 'compiler', 'verbose', 'dry_run', 'force', 'requiref90', 'c_compiler']
    new_fcompiler.stypy_varargs_param_name = None
    new_fcompiler.stypy_kwargs_param_name = None
    new_fcompiler.stypy_call_defaults = defaults
    new_fcompiler.stypy_call_varargs = varargs
    new_fcompiler.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'new_fcompiler', ['plat', 'compiler', 'verbose', 'dry_run', 'force', 'requiref90', 'c_compiler'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'new_fcompiler', localization, ['plat', 'compiler', 'verbose', 'dry_run', 'force', 'requiref90', 'c_compiler'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'new_fcompiler(...)' code ##################

    str_65818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 831, (-1)), 'str', 'Generate an instance of some FCompiler subclass for the supplied\n    platform/compiler combination.\n    ')
    # Marking variables as global (line 832)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 832, 4), 'failed_fcompilers')
    
    # Assigning a Tuple to a Name (line 833):
    
    # Assigning a Tuple to a Name (line 833):
    
    # Obtaining an instance of the builtin type 'tuple' (line 833)
    tuple_65819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 833)
    # Adding element type (line 833)
    # Getting the type of 'plat' (line 833)
    plat_65820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 21), 'plat')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 833, 21), tuple_65819, plat_65820)
    # Adding element type (line 833)
    # Getting the type of 'compiler' (line 833)
    compiler_65821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 27), 'compiler')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 833, 21), tuple_65819, compiler_65821)
    
    # Assigning a type to the variable 'fcompiler_key' (line 833)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 4), 'fcompiler_key', tuple_65819)
    
    
    # Getting the type of 'fcompiler_key' (line 834)
    fcompiler_key_65822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 7), 'fcompiler_key')
    # Getting the type of 'failed_fcompilers' (line 834)
    failed_fcompilers_65823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 24), 'failed_fcompilers')
    # Applying the binary operator 'in' (line 834)
    result_contains_65824 = python_operator(stypy.reporting.localization.Localization(__file__, 834, 7), 'in', fcompiler_key_65822, failed_fcompilers_65823)
    
    # Testing the type of an if condition (line 834)
    if_condition_65825 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 834, 4), result_contains_65824)
    # Assigning a type to the variable 'if_condition_65825' (line 834)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 4), 'if_condition_65825', if_condition_65825)
    # SSA begins for if statement (line 834)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 835)
    None_65826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 835, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 835)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 835, 8), 'stypy_return_type', None_65826)
    # SSA join for if statement (line 834)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to load_all_fcompiler_classes(...): (line 837)
    # Processing the call keyword arguments (line 837)
    kwargs_65828 = {}
    # Getting the type of 'load_all_fcompiler_classes' (line 837)
    load_all_fcompiler_classes_65827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 4), 'load_all_fcompiler_classes', False)
    # Calling load_all_fcompiler_classes(args, kwargs) (line 837)
    load_all_fcompiler_classes_call_result_65829 = invoke(stypy.reporting.localization.Localization(__file__, 837, 4), load_all_fcompiler_classes_65827, *[], **kwargs_65828)
    
    
    # Type idiom detected: calculating its left and rigth part (line 838)
    # Getting the type of 'plat' (line 838)
    plat_65830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 7), 'plat')
    # Getting the type of 'None' (line 838)
    None_65831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 838, 15), 'None')
    
    (may_be_65832, more_types_in_union_65833) = may_be_none(plat_65830, None_65831)

    if may_be_65832:

        if more_types_in_union_65833:
            # Runtime conditional SSA (line 838)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 839):
        
        # Assigning a Attribute to a Name (line 839):
        # Getting the type of 'os' (line 839)
        os_65834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 15), 'os')
        # Obtaining the member 'name' of a type (line 839)
        name_65835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 15), os_65834, 'name')
        # Assigning a type to the variable 'plat' (line 839)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 8), 'plat', name_65835)

        if more_types_in_union_65833:
            # SSA join for if statement (line 838)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 840)
    # Getting the type of 'compiler' (line 840)
    compiler_65836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 7), 'compiler')
    # Getting the type of 'None' (line 840)
    None_65837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 19), 'None')
    
    (may_be_65838, more_types_in_union_65839) = may_be_none(compiler_65836, None_65837)

    if may_be_65838:

        if more_types_in_union_65839:
            # Runtime conditional SSA (line 840)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 841):
        
        # Assigning a Call to a Name (line 841):
        
        # Call to get_default_fcompiler(...): (line 841)
        # Processing the call arguments (line 841)
        # Getting the type of 'plat' (line 841)
        plat_65841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 41), 'plat', False)
        # Processing the call keyword arguments (line 841)
        # Getting the type of 'requiref90' (line 841)
        requiref90_65842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 58), 'requiref90', False)
        keyword_65843 = requiref90_65842
        # Getting the type of 'c_compiler' (line 842)
        c_compiler_65844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 52), 'c_compiler', False)
        keyword_65845 = c_compiler_65844
        kwargs_65846 = {'c_compiler': keyword_65845, 'requiref90': keyword_65843}
        # Getting the type of 'get_default_fcompiler' (line 841)
        get_default_fcompiler_65840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 19), 'get_default_fcompiler', False)
        # Calling get_default_fcompiler(args, kwargs) (line 841)
        get_default_fcompiler_call_result_65847 = invoke(stypy.reporting.localization.Localization(__file__, 841, 19), get_default_fcompiler_65840, *[plat_65841], **kwargs_65846)
        
        # Assigning a type to the variable 'compiler' (line 841)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 8), 'compiler', get_default_fcompiler_call_result_65847)

        if more_types_in_union_65839:
            # SSA join for if statement (line 840)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'compiler' (line 843)
    compiler_65848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 7), 'compiler')
    # Getting the type of 'fcompiler_class' (line 843)
    fcompiler_class_65849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 19), 'fcompiler_class')
    # Applying the binary operator 'in' (line 843)
    result_contains_65850 = python_operator(stypy.reporting.localization.Localization(__file__, 843, 7), 'in', compiler_65848, fcompiler_class_65849)
    
    # Testing the type of an if condition (line 843)
    if_condition_65851 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 843, 4), result_contains_65850)
    # Assigning a type to the variable 'if_condition_65851' (line 843)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 4), 'if_condition_65851', if_condition_65851)
    # SSA begins for if statement (line 843)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Tuple (line 844):
    
    # Assigning a Subscript to a Name (line 844):
    
    # Obtaining the type of the subscript
    int_65852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'compiler' (line 844)
    compiler_65853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 63), 'compiler')
    # Getting the type of 'fcompiler_class' (line 844)
    fcompiler_class_65854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 47), 'fcompiler_class')
    # Obtaining the member '__getitem__' of a type (line 844)
    getitem___65855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 47), fcompiler_class_65854, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 844)
    subscript_call_result_65856 = invoke(stypy.reporting.localization.Localization(__file__, 844, 47), getitem___65855, compiler_65853)
    
    # Obtaining the member '__getitem__' of a type (line 844)
    getitem___65857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 8), subscript_call_result_65856, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 844)
    subscript_call_result_65858 = invoke(stypy.reporting.localization.Localization(__file__, 844, 8), getitem___65857, int_65852)
    
    # Assigning a type to the variable 'tuple_var_assignment_63494' (line 844)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 8), 'tuple_var_assignment_63494', subscript_call_result_65858)
    
    # Assigning a Subscript to a Name (line 844):
    
    # Obtaining the type of the subscript
    int_65859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'compiler' (line 844)
    compiler_65860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 63), 'compiler')
    # Getting the type of 'fcompiler_class' (line 844)
    fcompiler_class_65861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 47), 'fcompiler_class')
    # Obtaining the member '__getitem__' of a type (line 844)
    getitem___65862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 47), fcompiler_class_65861, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 844)
    subscript_call_result_65863 = invoke(stypy.reporting.localization.Localization(__file__, 844, 47), getitem___65862, compiler_65860)
    
    # Obtaining the member '__getitem__' of a type (line 844)
    getitem___65864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 8), subscript_call_result_65863, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 844)
    subscript_call_result_65865 = invoke(stypy.reporting.localization.Localization(__file__, 844, 8), getitem___65864, int_65859)
    
    # Assigning a type to the variable 'tuple_var_assignment_63495' (line 844)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 8), 'tuple_var_assignment_63495', subscript_call_result_65865)
    
    # Assigning a Subscript to a Name (line 844):
    
    # Obtaining the type of the subscript
    int_65866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 844, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'compiler' (line 844)
    compiler_65867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 63), 'compiler')
    # Getting the type of 'fcompiler_class' (line 844)
    fcompiler_class_65868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 47), 'fcompiler_class')
    # Obtaining the member '__getitem__' of a type (line 844)
    getitem___65869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 47), fcompiler_class_65868, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 844)
    subscript_call_result_65870 = invoke(stypy.reporting.localization.Localization(__file__, 844, 47), getitem___65869, compiler_65867)
    
    # Obtaining the member '__getitem__' of a type (line 844)
    getitem___65871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 844, 8), subscript_call_result_65870, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 844)
    subscript_call_result_65872 = invoke(stypy.reporting.localization.Localization(__file__, 844, 8), getitem___65871, int_65866)
    
    # Assigning a type to the variable 'tuple_var_assignment_63496' (line 844)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 8), 'tuple_var_assignment_63496', subscript_call_result_65872)
    
    # Assigning a Name to a Name (line 844):
    # Getting the type of 'tuple_var_assignment_63494' (line 844)
    tuple_var_assignment_63494_65873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 8), 'tuple_var_assignment_63494')
    # Assigning a type to the variable 'module_name' (line 844)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 8), 'module_name', tuple_var_assignment_63494_65873)
    
    # Assigning a Name to a Name (line 844):
    # Getting the type of 'tuple_var_assignment_63495' (line 844)
    tuple_var_assignment_63495_65874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 8), 'tuple_var_assignment_63495')
    # Assigning a type to the variable 'klass' (line 844)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 21), 'klass', tuple_var_assignment_63495_65874)
    
    # Assigning a Name to a Name (line 844):
    # Getting the type of 'tuple_var_assignment_63496' (line 844)
    tuple_var_assignment_63496_65875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 844, 8), 'tuple_var_assignment_63496')
    # Assigning a type to the variable 'long_description' (line 844)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 844, 28), 'long_description', tuple_var_assignment_63496_65875)
    # SSA branch for the else part of an if statement (line 843)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'compiler' (line 845)
    compiler_65876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 9), 'compiler')
    # Getting the type of 'fcompiler_aliases' (line 845)
    fcompiler_aliases_65877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 845, 21), 'fcompiler_aliases')
    # Applying the binary operator 'in' (line 845)
    result_contains_65878 = python_operator(stypy.reporting.localization.Localization(__file__, 845, 9), 'in', compiler_65876, fcompiler_aliases_65877)
    
    # Testing the type of an if condition (line 845)
    if_condition_65879 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 845, 9), result_contains_65878)
    # Assigning a type to the variable 'if_condition_65879' (line 845)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 845, 9), 'if_condition_65879', if_condition_65879)
    # SSA begins for if statement (line 845)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Tuple (line 846):
    
    # Assigning a Subscript to a Name (line 846):
    
    # Obtaining the type of the subscript
    int_65880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'compiler' (line 846)
    compiler_65881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 65), 'compiler')
    # Getting the type of 'fcompiler_aliases' (line 846)
    fcompiler_aliases_65882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 47), 'fcompiler_aliases')
    # Obtaining the member '__getitem__' of a type (line 846)
    getitem___65883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 47), fcompiler_aliases_65882, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 846)
    subscript_call_result_65884 = invoke(stypy.reporting.localization.Localization(__file__, 846, 47), getitem___65883, compiler_65881)
    
    # Obtaining the member '__getitem__' of a type (line 846)
    getitem___65885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 8), subscript_call_result_65884, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 846)
    subscript_call_result_65886 = invoke(stypy.reporting.localization.Localization(__file__, 846, 8), getitem___65885, int_65880)
    
    # Assigning a type to the variable 'tuple_var_assignment_63497' (line 846)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 8), 'tuple_var_assignment_63497', subscript_call_result_65886)
    
    # Assigning a Subscript to a Name (line 846):
    
    # Obtaining the type of the subscript
    int_65887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'compiler' (line 846)
    compiler_65888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 65), 'compiler')
    # Getting the type of 'fcompiler_aliases' (line 846)
    fcompiler_aliases_65889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 47), 'fcompiler_aliases')
    # Obtaining the member '__getitem__' of a type (line 846)
    getitem___65890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 47), fcompiler_aliases_65889, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 846)
    subscript_call_result_65891 = invoke(stypy.reporting.localization.Localization(__file__, 846, 47), getitem___65890, compiler_65888)
    
    # Obtaining the member '__getitem__' of a type (line 846)
    getitem___65892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 8), subscript_call_result_65891, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 846)
    subscript_call_result_65893 = invoke(stypy.reporting.localization.Localization(__file__, 846, 8), getitem___65892, int_65887)
    
    # Assigning a type to the variable 'tuple_var_assignment_63498' (line 846)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 8), 'tuple_var_assignment_63498', subscript_call_result_65893)
    
    # Assigning a Subscript to a Name (line 846):
    
    # Obtaining the type of the subscript
    int_65894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 846, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'compiler' (line 846)
    compiler_65895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 65), 'compiler')
    # Getting the type of 'fcompiler_aliases' (line 846)
    fcompiler_aliases_65896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 47), 'fcompiler_aliases')
    # Obtaining the member '__getitem__' of a type (line 846)
    getitem___65897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 47), fcompiler_aliases_65896, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 846)
    subscript_call_result_65898 = invoke(stypy.reporting.localization.Localization(__file__, 846, 47), getitem___65897, compiler_65895)
    
    # Obtaining the member '__getitem__' of a type (line 846)
    getitem___65899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 8), subscript_call_result_65898, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 846)
    subscript_call_result_65900 = invoke(stypy.reporting.localization.Localization(__file__, 846, 8), getitem___65899, int_65894)
    
    # Assigning a type to the variable 'tuple_var_assignment_63499' (line 846)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 8), 'tuple_var_assignment_63499', subscript_call_result_65900)
    
    # Assigning a Name to a Name (line 846):
    # Getting the type of 'tuple_var_assignment_63497' (line 846)
    tuple_var_assignment_63497_65901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 8), 'tuple_var_assignment_63497')
    # Assigning a type to the variable 'module_name' (line 846)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 8), 'module_name', tuple_var_assignment_63497_65901)
    
    # Assigning a Name to a Name (line 846):
    # Getting the type of 'tuple_var_assignment_63498' (line 846)
    tuple_var_assignment_63498_65902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 8), 'tuple_var_assignment_63498')
    # Assigning a type to the variable 'klass' (line 846)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 21), 'klass', tuple_var_assignment_63498_65902)
    
    # Assigning a Name to a Name (line 846):
    # Getting the type of 'tuple_var_assignment_63499' (line 846)
    tuple_var_assignment_63499_65903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 8), 'tuple_var_assignment_63499')
    # Assigning a type to the variable 'long_description' (line 846)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 846, 28), 'long_description', tuple_var_assignment_63499_65903)
    # SSA branch for the else part of an if statement (line 845)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 848):
    
    # Assigning a BinOp to a Name (line 848):
    str_65904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 14), 'str', "don't know how to compile Fortran code on platform '%s'")
    # Getting the type of 'plat' (line 848)
    plat_65905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 74), 'plat')
    # Applying the binary operator '%' (line 848)
    result_mod_65906 = python_operator(stypy.reporting.localization.Localization(__file__, 848, 14), '%', str_65904, plat_65905)
    
    # Assigning a type to the variable 'msg' (line 848)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 848, 8), 'msg', result_mod_65906)
    
    # Type idiom detected: calculating its left and rigth part (line 849)
    # Getting the type of 'compiler' (line 849)
    compiler_65907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 8), 'compiler')
    # Getting the type of 'None' (line 849)
    None_65908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 27), 'None')
    
    (may_be_65909, more_types_in_union_65910) = may_not_be_none(compiler_65907, None_65908)

    if may_be_65909:

        if more_types_in_union_65910:
            # Runtime conditional SSA (line 849)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 850):
        
        # Assigning a BinOp to a Name (line 850):
        # Getting the type of 'msg' (line 850)
        msg_65911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 18), 'msg')
        str_65912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, 24), 'str', " with '%s' compiler.")
        # Getting the type of 'compiler' (line 850)
        compiler_65913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 49), 'compiler')
        # Applying the binary operator '%' (line 850)
        result_mod_65914 = python_operator(stypy.reporting.localization.Localization(__file__, 850, 24), '%', str_65912, compiler_65913)
        
        # Applying the binary operator '+' (line 850)
        result_add_65915 = python_operator(stypy.reporting.localization.Localization(__file__, 850, 18), '+', msg_65911, result_mod_65914)
        
        # Assigning a type to the variable 'msg' (line 850)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 12), 'msg', result_add_65915)
        
        # Assigning a BinOp to a Name (line 851):
        
        # Assigning a BinOp to a Name (line 851):
        # Getting the type of 'msg' (line 851)
        msg_65916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 18), 'msg')
        str_65917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, 24), 'str', ' Supported compilers are: %s)')
        
        # Call to join(...): (line 852)
        # Processing the call arguments (line 852)
        
        # Call to keys(...): (line 852)
        # Processing the call keyword arguments (line 852)
        kwargs_65922 = {}
        # Getting the type of 'fcompiler_class' (line 852)
        fcompiler_class_65920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 30), 'fcompiler_class', False)
        # Obtaining the member 'keys' of a type (line 852)
        keys_65921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 852, 30), fcompiler_class_65920, 'keys')
        # Calling keys(args, kwargs) (line 852)
        keys_call_result_65923 = invoke(stypy.reporting.localization.Localization(__file__, 852, 30), keys_65921, *[], **kwargs_65922)
        
        # Processing the call keyword arguments (line 852)
        kwargs_65924 = {}
        str_65918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, 21), 'str', ',')
        # Obtaining the member 'join' of a type (line 852)
        join_65919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 852, 21), str_65918, 'join')
        # Calling join(args, kwargs) (line 852)
        join_call_result_65925 = invoke(stypy.reporting.localization.Localization(__file__, 852, 21), join_65919, *[keys_call_result_65923], **kwargs_65924)
        
        # Applying the binary operator '%' (line 851)
        result_mod_65926 = python_operator(stypy.reporting.localization.Localization(__file__, 851, 24), '%', str_65917, join_call_result_65925)
        
        # Applying the binary operator '+' (line 851)
        result_add_65927 = python_operator(stypy.reporting.localization.Localization(__file__, 851, 18), '+', msg_65916, result_mod_65926)
        
        # Assigning a type to the variable 'msg' (line 851)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 851, 12), 'msg', result_add_65927)

        if more_types_in_union_65910:
            # SSA join for if statement (line 849)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to warn(...): (line 853)
    # Processing the call arguments (line 853)
    # Getting the type of 'msg' (line 853)
    msg_65930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 17), 'msg', False)
    # Processing the call keyword arguments (line 853)
    kwargs_65931 = {}
    # Getting the type of 'log' (line 853)
    log_65928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 8), 'log', False)
    # Obtaining the member 'warn' of a type (line 853)
    warn_65929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 853, 8), log_65928, 'warn')
    # Calling warn(args, kwargs) (line 853)
    warn_call_result_65932 = invoke(stypy.reporting.localization.Localization(__file__, 853, 8), warn_65929, *[msg_65930], **kwargs_65931)
    
    
    # Call to add(...): (line 854)
    # Processing the call arguments (line 854)
    # Getting the type of 'fcompiler_key' (line 854)
    fcompiler_key_65935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 30), 'fcompiler_key', False)
    # Processing the call keyword arguments (line 854)
    kwargs_65936 = {}
    # Getting the type of 'failed_fcompilers' (line 854)
    failed_fcompilers_65933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 8), 'failed_fcompilers', False)
    # Obtaining the member 'add' of a type (line 854)
    add_65934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 854, 8), failed_fcompilers_65933, 'add')
    # Calling add(args, kwargs) (line 854)
    add_call_result_65937 = invoke(stypy.reporting.localization.Localization(__file__, 854, 8), add_65934, *[fcompiler_key_65935], **kwargs_65936)
    
    # Getting the type of 'None' (line 855)
    None_65938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 855, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 855)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 855, 8), 'stypy_return_type', None_65938)
    # SSA join for if statement (line 845)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 843)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 857):
    
    # Assigning a Call to a Name (line 857):
    
    # Call to klass(...): (line 857)
    # Processing the call keyword arguments (line 857)
    # Getting the type of 'verbose' (line 857)
    verbose_65940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 29), 'verbose', False)
    keyword_65941 = verbose_65940
    # Getting the type of 'dry_run' (line 857)
    dry_run_65942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 46), 'dry_run', False)
    keyword_65943 = dry_run_65942
    # Getting the type of 'force' (line 857)
    force_65944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 61), 'force', False)
    keyword_65945 = force_65944
    kwargs_65946 = {'force': keyword_65945, 'verbose': keyword_65941, 'dry_run': keyword_65943}
    # Getting the type of 'klass' (line 857)
    klass_65939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 857, 15), 'klass', False)
    # Calling klass(args, kwargs) (line 857)
    klass_call_result_65947 = invoke(stypy.reporting.localization.Localization(__file__, 857, 15), klass_65939, *[], **kwargs_65946)
    
    # Assigning a type to the variable 'compiler' (line 857)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 857, 4), 'compiler', klass_call_result_65947)
    
    # Assigning a Name to a Attribute (line 858):
    
    # Assigning a Name to a Attribute (line 858):
    # Getting the type of 'c_compiler' (line 858)
    c_compiler_65948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 26), 'c_compiler')
    # Getting the type of 'compiler' (line 858)
    compiler_65949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 858, 4), 'compiler')
    # Setting the type of the member 'c_compiler' of a type (line 858)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 858, 4), compiler_65949, 'c_compiler', c_compiler_65948)
    # Getting the type of 'compiler' (line 859)
    compiler_65950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 859, 11), 'compiler')
    # Assigning a type to the variable 'stypy_return_type' (line 859)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 859, 4), 'stypy_return_type', compiler_65950)
    
    # ################# End of 'new_fcompiler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'new_fcompiler' in the type store
    # Getting the type of 'stypy_return_type' (line 822)
    stypy_return_type_65951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_65951)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'new_fcompiler'
    return stypy_return_type_65951

# Assigning a type to the variable 'new_fcompiler' (line 822)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 822, 0), 'new_fcompiler', new_fcompiler)

@norecursion
def show_fcompilers(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 861)
    None_65952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 25), 'None')
    defaults = [None_65952]
    # Create a new context for function 'show_fcompilers'
    module_type_store = module_type_store.open_function_context('show_fcompilers', 861, 0, False)
    
    # Passed parameters checking function
    show_fcompilers.stypy_localization = localization
    show_fcompilers.stypy_type_of_self = None
    show_fcompilers.stypy_type_store = module_type_store
    show_fcompilers.stypy_function_name = 'show_fcompilers'
    show_fcompilers.stypy_param_names_list = ['dist']
    show_fcompilers.stypy_varargs_param_name = None
    show_fcompilers.stypy_kwargs_param_name = None
    show_fcompilers.stypy_call_defaults = defaults
    show_fcompilers.stypy_call_varargs = varargs
    show_fcompilers.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'show_fcompilers', ['dist'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'show_fcompilers', localization, ['dist'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'show_fcompilers(...)' code ##################

    str_65953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 864, (-1)), 'str', 'Print list of available compilers (used by the "--help-fcompiler"\n    option to "config_fc").\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 865)
    # Getting the type of 'dist' (line 865)
    dist_65954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 7), 'dist')
    # Getting the type of 'None' (line 865)
    None_65955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 15), 'None')
    
    (may_be_65956, more_types_in_union_65957) = may_be_none(dist_65954, None_65955)

    if may_be_65956:

        if more_types_in_union_65957:
            # Runtime conditional SSA (line 865)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 866, 8))
        
        # 'from distutils.dist import Distribution' statement (line 866)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
        import_65958 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 866, 8), 'distutils.dist')

        if (type(import_65958) is not StypyTypeError):

            if (import_65958 != 'pyd_module'):
                __import__(import_65958)
                sys_modules_65959 = sys.modules[import_65958]
                import_from_module(stypy.reporting.localization.Localization(__file__, 866, 8), 'distutils.dist', sys_modules_65959.module_type_store, module_type_store, ['Distribution'])
                nest_module(stypy.reporting.localization.Localization(__file__, 866, 8), __file__, sys_modules_65959, sys_modules_65959.module_type_store, module_type_store)
            else:
                from distutils.dist import Distribution

                import_from_module(stypy.reporting.localization.Localization(__file__, 866, 8), 'distutils.dist', None, module_type_store, ['Distribution'], [Distribution])

        else:
            # Assigning a type to the variable 'distutils.dist' (line 866)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 866, 8), 'distutils.dist', import_65958)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 867, 8))
        
        # 'from numpy.distutils.command.config_compiler import config_fc' statement (line 867)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
        import_65960 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 867, 8), 'numpy.distutils.command.config_compiler')

        if (type(import_65960) is not StypyTypeError):

            if (import_65960 != 'pyd_module'):
                __import__(import_65960)
                sys_modules_65961 = sys.modules[import_65960]
                import_from_module(stypy.reporting.localization.Localization(__file__, 867, 8), 'numpy.distutils.command.config_compiler', sys_modules_65961.module_type_store, module_type_store, ['config_fc'])
                nest_module(stypy.reporting.localization.Localization(__file__, 867, 8), __file__, sys_modules_65961, sys_modules_65961.module_type_store, module_type_store)
            else:
                from numpy.distutils.command.config_compiler import config_fc

                import_from_module(stypy.reporting.localization.Localization(__file__, 867, 8), 'numpy.distutils.command.config_compiler', None, module_type_store, ['config_fc'], [config_fc])

        else:
            # Assigning a type to the variable 'numpy.distutils.command.config_compiler' (line 867)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 867, 8), 'numpy.distutils.command.config_compiler', import_65960)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
        
        
        # Assigning a Call to a Name (line 868):
        
        # Assigning a Call to a Name (line 868):
        
        # Call to Distribution(...): (line 868)
        # Processing the call keyword arguments (line 868)
        kwargs_65963 = {}
        # Getting the type of 'Distribution' (line 868)
        Distribution_65962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 868, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 868)
        Distribution_call_result_65964 = invoke(stypy.reporting.localization.Localization(__file__, 868, 15), Distribution_65962, *[], **kwargs_65963)
        
        # Assigning a type to the variable 'dist' (line 868)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 868, 8), 'dist', Distribution_call_result_65964)
        
        # Assigning a Call to a Attribute (line 869):
        
        # Assigning a Call to a Attribute (line 869):
        
        # Call to basename(...): (line 869)
        # Processing the call arguments (line 869)
        
        # Obtaining the type of the subscript
        int_65968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 53), 'int')
        # Getting the type of 'sys' (line 869)
        sys_65969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 44), 'sys', False)
        # Obtaining the member 'argv' of a type (line 869)
        argv_65970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 44), sys_65969, 'argv')
        # Obtaining the member '__getitem__' of a type (line 869)
        getitem___65971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 44), argv_65970, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 869)
        subscript_call_result_65972 = invoke(stypy.reporting.localization.Localization(__file__, 869, 44), getitem___65971, int_65968)
        
        # Processing the call keyword arguments (line 869)
        kwargs_65973 = {}
        # Getting the type of 'os' (line 869)
        os_65965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 869)
        path_65966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 27), os_65965, 'path')
        # Obtaining the member 'basename' of a type (line 869)
        basename_65967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 27), path_65966, 'basename')
        # Calling basename(args, kwargs) (line 869)
        basename_call_result_65974 = invoke(stypy.reporting.localization.Localization(__file__, 869, 27), basename_65967, *[subscript_call_result_65972], **kwargs_65973)
        
        # Getting the type of 'dist' (line 869)
        dist_65975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 869, 8), 'dist')
        # Setting the type of the member 'script_name' of a type (line 869)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 869, 8), dist_65975, 'script_name', basename_call_result_65974)
        
        # Assigning a BinOp to a Attribute (line 870):
        
        # Assigning a BinOp to a Attribute (line 870):
        
        # Obtaining an instance of the builtin type 'list' (line 870)
        list_65976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 870, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 870)
        # Adding element type (line 870)
        str_65977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 870, 28), 'str', 'config_fc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 870, 27), list_65976, str_65977)
        
        
        # Obtaining the type of the subscript
        int_65978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 870, 52), 'int')
        slice_65979 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 870, 43), int_65978, None, None)
        # Getting the type of 'sys' (line 870)
        sys_65980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 43), 'sys')
        # Obtaining the member 'argv' of a type (line 870)
        argv_65981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 43), sys_65980, 'argv')
        # Obtaining the member '__getitem__' of a type (line 870)
        getitem___65982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 43), argv_65981, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 870)
        subscript_call_result_65983 = invoke(stypy.reporting.localization.Localization(__file__, 870, 43), getitem___65982, slice_65979)
        
        # Applying the binary operator '+' (line 870)
        result_add_65984 = python_operator(stypy.reporting.localization.Localization(__file__, 870, 27), '+', list_65976, subscript_call_result_65983)
        
        # Getting the type of 'dist' (line 870)
        dist_65985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 8), 'dist')
        # Setting the type of the member 'script_args' of a type (line 870)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 8), dist_65985, 'script_args', result_add_65984)
        
        
        # SSA begins for try-except statement (line 871)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to remove(...): (line 872)
        # Processing the call arguments (line 872)
        str_65989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 36), 'str', '--help-fcompiler')
        # Processing the call keyword arguments (line 872)
        kwargs_65990 = {}
        # Getting the type of 'dist' (line 872)
        dist_65986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 872, 12), 'dist', False)
        # Obtaining the member 'script_args' of a type (line 872)
        script_args_65987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 12), dist_65986, 'script_args')
        # Obtaining the member 'remove' of a type (line 872)
        remove_65988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 872, 12), script_args_65987, 'remove')
        # Calling remove(args, kwargs) (line 872)
        remove_call_result_65991 = invoke(stypy.reporting.localization.Localization(__file__, 872, 12), remove_65988, *[str_65989], **kwargs_65990)
        
        # SSA branch for the except part of a try statement (line 871)
        # SSA branch for the except 'ValueError' branch of a try statement (line 871)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 871)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 875):
        
        # Assigning a Name to a Subscript (line 875):
        # Getting the type of 'config_fc' (line 875)
        config_fc_65992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 37), 'config_fc')
        # Getting the type of 'dist' (line 875)
        dist_65993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 8), 'dist')
        # Obtaining the member 'cmdclass' of a type (line 875)
        cmdclass_65994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 875, 8), dist_65993, 'cmdclass')
        str_65995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 875, 22), 'str', 'config_fc')
        # Storing an element on a container (line 875)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 875, 8), cmdclass_65994, (str_65995, config_fc_65992))
        
        # Call to parse_config_files(...): (line 876)
        # Processing the call keyword arguments (line 876)
        kwargs_65998 = {}
        # Getting the type of 'dist' (line 876)
        dist_65996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'dist', False)
        # Obtaining the member 'parse_config_files' of a type (line 876)
        parse_config_files_65997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 876, 8), dist_65996, 'parse_config_files')
        # Calling parse_config_files(args, kwargs) (line 876)
        parse_config_files_call_result_65999 = invoke(stypy.reporting.localization.Localization(__file__, 876, 8), parse_config_files_65997, *[], **kwargs_65998)
        
        
        # Call to parse_command_line(...): (line 877)
        # Processing the call keyword arguments (line 877)
        kwargs_66002 = {}
        # Getting the type of 'dist' (line 877)
        dist_66000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 877, 8), 'dist', False)
        # Obtaining the member 'parse_command_line' of a type (line 877)
        parse_command_line_66001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 877, 8), dist_66000, 'parse_command_line')
        # Calling parse_command_line(args, kwargs) (line 877)
        parse_command_line_call_result_66003 = invoke(stypy.reporting.localization.Localization(__file__, 877, 8), parse_command_line_66001, *[], **kwargs_66002)
        

        if more_types_in_union_65957:
            # SSA join for if statement (line 865)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a List to a Name (line 878):
    
    # Assigning a List to a Name (line 878):
    
    # Obtaining an instance of the builtin type 'list' (line 878)
    list_66004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 878, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 878)
    
    # Assigning a type to the variable 'compilers' (line 878)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 4), 'compilers', list_66004)
    
    # Assigning a List to a Name (line 879):
    
    # Assigning a List to a Name (line 879):
    
    # Obtaining an instance of the builtin type 'list' (line 879)
    list_66005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 879, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 879)
    
    # Assigning a type to the variable 'compilers_na' (line 879)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 4), 'compilers_na', list_66005)
    
    # Assigning a List to a Name (line 880):
    
    # Assigning a List to a Name (line 880):
    
    # Obtaining an instance of the builtin type 'list' (line 880)
    list_66006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 880, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 880)
    
    # Assigning a type to the variable 'compilers_ni' (line 880)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 880, 4), 'compilers_ni', list_66006)
    
    
    # Getting the type of 'fcompiler_class' (line 881)
    fcompiler_class_66007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 881, 11), 'fcompiler_class')
    # Applying the 'not' unary operator (line 881)
    result_not__66008 = python_operator(stypy.reporting.localization.Localization(__file__, 881, 7), 'not', fcompiler_class_66007)
    
    # Testing the type of an if condition (line 881)
    if_condition_66009 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 881, 4), result_not__66008)
    # Assigning a type to the variable 'if_condition_66009' (line 881)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 881, 4), 'if_condition_66009', if_condition_66009)
    # SSA begins for if statement (line 881)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to load_all_fcompiler_classes(...): (line 882)
    # Processing the call keyword arguments (line 882)
    kwargs_66011 = {}
    # Getting the type of 'load_all_fcompiler_classes' (line 882)
    load_all_fcompiler_classes_66010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 882, 8), 'load_all_fcompiler_classes', False)
    # Calling load_all_fcompiler_classes(args, kwargs) (line 882)
    load_all_fcompiler_classes_call_result_66012 = invoke(stypy.reporting.localization.Localization(__file__, 882, 8), load_all_fcompiler_classes_66010, *[], **kwargs_66011)
    
    # SSA join for if statement (line 881)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 883):
    
    # Assigning a Call to a Name (line 883):
    
    # Call to available_fcompilers_for_platform(...): (line 883)
    # Processing the call keyword arguments (line 883)
    kwargs_66014 = {}
    # Getting the type of 'available_fcompilers_for_platform' (line 883)
    available_fcompilers_for_platform_66013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 25), 'available_fcompilers_for_platform', False)
    # Calling available_fcompilers_for_platform(args, kwargs) (line 883)
    available_fcompilers_for_platform_call_result_66015 = invoke(stypy.reporting.localization.Localization(__file__, 883, 25), available_fcompilers_for_platform_66013, *[], **kwargs_66014)
    
    # Assigning a type to the variable 'platform_compilers' (line 883)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 883, 4), 'platform_compilers', available_fcompilers_for_platform_call_result_66015)
    
    # Getting the type of 'platform_compilers' (line 884)
    platform_compilers_66016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 884, 20), 'platform_compilers')
    # Testing the type of a for loop iterable (line 884)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 884, 4), platform_compilers_66016)
    # Getting the type of the for loop variable (line 884)
    for_loop_var_66017 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 884, 4), platform_compilers_66016)
    # Assigning a type to the variable 'compiler' (line 884)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 4), 'compiler', for_loop_var_66017)
    # SSA begins for a for statement (line 884)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Name (line 885):
    
    # Assigning a Name to a Name (line 885):
    # Getting the type of 'None' (line 885)
    None_66018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 885, 12), 'None')
    # Assigning a type to the variable 'v' (line 885)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 885, 8), 'v', None_66018)
    
    # Call to set_verbosity(...): (line 886)
    # Processing the call arguments (line 886)
    int_66021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 886, 26), 'int')
    # Processing the call keyword arguments (line 886)
    kwargs_66022 = {}
    # Getting the type of 'log' (line 886)
    log_66019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 8), 'log', False)
    # Obtaining the member 'set_verbosity' of a type (line 886)
    set_verbosity_66020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 886, 8), log_66019, 'set_verbosity')
    # Calling set_verbosity(args, kwargs) (line 886)
    set_verbosity_call_result_66023 = invoke(stypy.reporting.localization.Localization(__file__, 886, 8), set_verbosity_66020, *[int_66021], **kwargs_66022)
    
    
    
    # SSA begins for try-except statement (line 887)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 888):
    
    # Assigning a Call to a Name (line 888):
    
    # Call to new_fcompiler(...): (line 888)
    # Processing the call keyword arguments (line 888)
    # Getting the type of 'compiler' (line 888)
    compiler_66025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 39), 'compiler', False)
    keyword_66026 = compiler_66025
    # Getting the type of 'dist' (line 888)
    dist_66027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 57), 'dist', False)
    # Obtaining the member 'verbose' of a type (line 888)
    verbose_66028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 888, 57), dist_66027, 'verbose')
    keyword_66029 = verbose_66028
    kwargs_66030 = {'verbose': keyword_66029, 'compiler': keyword_66026}
    # Getting the type of 'new_fcompiler' (line 888)
    new_fcompiler_66024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 16), 'new_fcompiler', False)
    # Calling new_fcompiler(args, kwargs) (line 888)
    new_fcompiler_call_result_66031 = invoke(stypy.reporting.localization.Localization(__file__, 888, 16), new_fcompiler_66024, *[], **kwargs_66030)
    
    # Assigning a type to the variable 'c' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 12), 'c', new_fcompiler_call_result_66031)
    
    # Call to customize(...): (line 889)
    # Processing the call arguments (line 889)
    # Getting the type of 'dist' (line 889)
    dist_66034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 24), 'dist', False)
    # Processing the call keyword arguments (line 889)
    kwargs_66035 = {}
    # Getting the type of 'c' (line 889)
    c_66032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 12), 'c', False)
    # Obtaining the member 'customize' of a type (line 889)
    customize_66033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 889, 12), c_66032, 'customize')
    # Calling customize(args, kwargs) (line 889)
    customize_call_result_66036 = invoke(stypy.reporting.localization.Localization(__file__, 889, 12), customize_66033, *[dist_66034], **kwargs_66035)
    
    
    # Assigning a Call to a Name (line 890):
    
    # Assigning a Call to a Name (line 890):
    
    # Call to get_version(...): (line 890)
    # Processing the call keyword arguments (line 890)
    kwargs_66039 = {}
    # Getting the type of 'c' (line 890)
    c_66037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 890, 16), 'c', False)
    # Obtaining the member 'get_version' of a type (line 890)
    get_version_66038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 890, 16), c_66037, 'get_version')
    # Calling get_version(args, kwargs) (line 890)
    get_version_call_result_66040 = invoke(stypy.reporting.localization.Localization(__file__, 890, 16), get_version_66038, *[], **kwargs_66039)
    
    # Assigning a type to the variable 'v' (line 890)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 890, 12), 'v', get_version_call_result_66040)
    # SSA branch for the except part of a try statement (line 887)
    # SSA branch for the except 'Tuple' branch of a try statement (line 887)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 892):
    
    # Assigning a Call to a Name (line 892):
    
    # Call to get_exception(...): (line 892)
    # Processing the call keyword arguments (line 892)
    kwargs_66042 = {}
    # Getting the type of 'get_exception' (line 892)
    get_exception_66041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 892, 16), 'get_exception', False)
    # Calling get_exception(args, kwargs) (line 892)
    get_exception_call_result_66043 = invoke(stypy.reporting.localization.Localization(__file__, 892, 16), get_exception_66041, *[], **kwargs_66042)
    
    # Assigning a type to the variable 'e' (line 892)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 892, 12), 'e', get_exception_call_result_66043)
    
    # Call to debug(...): (line 893)
    # Processing the call arguments (line 893)
    str_66046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 22), 'str', 'show_fcompilers: %s not found')
    
    # Obtaining an instance of the builtin type 'tuple' (line 893)
    tuple_66047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 893)
    # Adding element type (line 893)
    # Getting the type of 'compiler' (line 893)
    compiler_66048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 57), 'compiler', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 893, 57), tuple_66047, compiler_66048)
    
    # Applying the binary operator '%' (line 893)
    result_mod_66049 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 22), '%', str_66046, tuple_66047)
    
    # Processing the call keyword arguments (line 893)
    kwargs_66050 = {}
    # Getting the type of 'log' (line 893)
    log_66044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 12), 'log', False)
    # Obtaining the member 'debug' of a type (line 893)
    debug_66045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 893, 12), log_66044, 'debug')
    # Calling debug(args, kwargs) (line 893)
    debug_call_result_66051 = invoke(stypy.reporting.localization.Localization(__file__, 893, 12), debug_66045, *[result_mod_66049], **kwargs_66050)
    
    
    # Call to debug(...): (line 894)
    # Processing the call arguments (line 894)
    
    # Call to repr(...): (line 894)
    # Processing the call arguments (line 894)
    # Getting the type of 'e' (line 894)
    e_66055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 27), 'e', False)
    # Processing the call keyword arguments (line 894)
    kwargs_66056 = {}
    # Getting the type of 'repr' (line 894)
    repr_66054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 22), 'repr', False)
    # Calling repr(args, kwargs) (line 894)
    repr_call_result_66057 = invoke(stypy.reporting.localization.Localization(__file__, 894, 22), repr_66054, *[e_66055], **kwargs_66056)
    
    # Processing the call keyword arguments (line 894)
    kwargs_66058 = {}
    # Getting the type of 'log' (line 894)
    log_66052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 894, 12), 'log', False)
    # Obtaining the member 'debug' of a type (line 894)
    debug_66053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 894, 12), log_66052, 'debug')
    # Calling debug(args, kwargs) (line 894)
    debug_call_result_66059 = invoke(stypy.reporting.localization.Localization(__file__, 894, 12), debug_66053, *[repr_call_result_66057], **kwargs_66058)
    
    # SSA join for try-except statement (line 887)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 896)
    # Getting the type of 'v' (line 896)
    v_66060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 11), 'v')
    # Getting the type of 'None' (line 896)
    None_66061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 16), 'None')
    
    (may_be_66062, more_types_in_union_66063) = may_be_none(v_66060, None_66061)

    if may_be_66062:

        if more_types_in_union_66063:
            # Runtime conditional SSA (line 896)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to append(...): (line 897)
        # Processing the call arguments (line 897)
        
        # Obtaining an instance of the builtin type 'tuple' (line 897)
        tuple_66066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 897)
        # Adding element type (line 897)
        str_66067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 33), 'str', 'fcompiler=')
        # Getting the type of 'compiler' (line 897)
        compiler_66068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 46), 'compiler', False)
        # Applying the binary operator '+' (line 897)
        result_add_66069 = python_operator(stypy.reporting.localization.Localization(__file__, 897, 33), '+', str_66067, compiler_66068)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 33), tuple_66066, result_add_66069)
        # Adding element type (line 897)
        # Getting the type of 'None' (line 897)
        None_66070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 56), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 33), tuple_66066, None_66070)
        # Adding element type (line 897)
        
        # Obtaining the type of the subscript
        int_66071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 898, 56), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'compiler' (line 898)
        compiler_66072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 46), 'compiler', False)
        # Getting the type of 'fcompiler_class' (line 898)
        fcompiler_class_66073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 898, 30), 'fcompiler_class', False)
        # Obtaining the member '__getitem__' of a type (line 898)
        getitem___66074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 898, 30), fcompiler_class_66073, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 898)
        subscript_call_result_66075 = invoke(stypy.reporting.localization.Localization(__file__, 898, 30), getitem___66074, compiler_66072)
        
        # Obtaining the member '__getitem__' of a type (line 898)
        getitem___66076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 898, 30), subscript_call_result_66075, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 898)
        subscript_call_result_66077 = invoke(stypy.reporting.localization.Localization(__file__, 898, 30), getitem___66076, int_66071)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 897, 33), tuple_66066, subscript_call_result_66077)
        
        # Processing the call keyword arguments (line 897)
        kwargs_66078 = {}
        # Getting the type of 'compilers_na' (line 897)
        compilers_na_66064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 897, 12), 'compilers_na', False)
        # Obtaining the member 'append' of a type (line 897)
        append_66065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 897, 12), compilers_na_66064, 'append')
        # Calling append(args, kwargs) (line 897)
        append_call_result_66079 = invoke(stypy.reporting.localization.Localization(__file__, 897, 12), append_66065, *[tuple_66066], **kwargs_66078)
        

        if more_types_in_union_66063:
            # Runtime conditional SSA for else branch (line 896)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_66062) or more_types_in_union_66063):
        
        # Call to dump_properties(...): (line 900)
        # Processing the call keyword arguments (line 900)
        kwargs_66082 = {}
        # Getting the type of 'c' (line 900)
        c_66080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 900, 12), 'c', False)
        # Obtaining the member 'dump_properties' of a type (line 900)
        dump_properties_66081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 900, 12), c_66080, 'dump_properties')
        # Calling dump_properties(args, kwargs) (line 900)
        dump_properties_call_result_66083 = invoke(stypy.reporting.localization.Localization(__file__, 900, 12), dump_properties_66081, *[], **kwargs_66082)
        
        
        # Call to append(...): (line 901)
        # Processing the call arguments (line 901)
        
        # Obtaining an instance of the builtin type 'tuple' (line 901)
        tuple_66086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 901, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 901)
        # Adding element type (line 901)
        str_66087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 901, 30), 'str', 'fcompiler=')
        # Getting the type of 'compiler' (line 901)
        compiler_66088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 43), 'compiler', False)
        # Applying the binary operator '+' (line 901)
        result_add_66089 = python_operator(stypy.reporting.localization.Localization(__file__, 901, 30), '+', str_66087, compiler_66088)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 901, 30), tuple_66086, result_add_66089)
        # Adding element type (line 901)
        # Getting the type of 'None' (line 901)
        None_66090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 53), 'None', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 901, 30), tuple_66086, None_66090)
        # Adding element type (line 901)
        
        # Obtaining the type of the subscript
        int_66091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 56), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'compiler' (line 902)
        compiler_66092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 46), 'compiler', False)
        # Getting the type of 'fcompiler_class' (line 902)
        fcompiler_class_66093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 30), 'fcompiler_class', False)
        # Obtaining the member '__getitem__' of a type (line 902)
        getitem___66094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 902, 30), fcompiler_class_66093, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 902)
        subscript_call_result_66095 = invoke(stypy.reporting.localization.Localization(__file__, 902, 30), getitem___66094, compiler_66092)
        
        # Obtaining the member '__getitem__' of a type (line 902)
        getitem___66096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 902, 30), subscript_call_result_66095, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 902)
        subscript_call_result_66097 = invoke(stypy.reporting.localization.Localization(__file__, 902, 30), getitem___66096, int_66091)
        
        str_66098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 902, 61), 'str', ' (%s)')
        # Getting the type of 'v' (line 902)
        v_66099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 902, 71), 'v', False)
        # Applying the binary operator '%' (line 902)
        result_mod_66100 = python_operator(stypy.reporting.localization.Localization(__file__, 902, 61), '%', str_66098, v_66099)
        
        # Applying the binary operator '+' (line 902)
        result_add_66101 = python_operator(stypy.reporting.localization.Localization(__file__, 902, 30), '+', subscript_call_result_66097, result_mod_66100)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 901, 30), tuple_66086, result_add_66101)
        
        # Processing the call keyword arguments (line 901)
        kwargs_66102 = {}
        # Getting the type of 'compilers' (line 901)
        compilers_66084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 12), 'compilers', False)
        # Obtaining the member 'append' of a type (line 901)
        append_66085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 901, 12), compilers_66084, 'append')
        # Calling append(args, kwargs) (line 901)
        append_call_result_66103 = invoke(stypy.reporting.localization.Localization(__file__, 901, 12), append_66085, *[tuple_66086], **kwargs_66102)
        

        if (may_be_66062 and more_types_in_union_66063):
            # SSA join for if statement (line 896)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 904):
    
    # Assigning a Call to a Name (line 904):
    
    # Call to list(...): (line 904)
    # Processing the call arguments (line 904)
    
    # Call to set(...): (line 904)
    # Processing the call arguments (line 904)
    
    # Call to keys(...): (line 904)
    # Processing the call keyword arguments (line 904)
    kwargs_66108 = {}
    # Getting the type of 'fcompiler_class' (line 904)
    fcompiler_class_66106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 28), 'fcompiler_class', False)
    # Obtaining the member 'keys' of a type (line 904)
    keys_66107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 904, 28), fcompiler_class_66106, 'keys')
    # Calling keys(args, kwargs) (line 904)
    keys_call_result_66109 = invoke(stypy.reporting.localization.Localization(__file__, 904, 28), keys_66107, *[], **kwargs_66108)
    
    # Processing the call keyword arguments (line 904)
    kwargs_66110 = {}
    # Getting the type of 'set' (line 904)
    set_66105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 24), 'set', False)
    # Calling set(args, kwargs) (line 904)
    set_call_result_66111 = invoke(stypy.reporting.localization.Localization(__file__, 904, 24), set_66105, *[keys_call_result_66109], **kwargs_66110)
    
    
    # Call to set(...): (line 904)
    # Processing the call arguments (line 904)
    # Getting the type of 'platform_compilers' (line 904)
    platform_compilers_66113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 58), 'platform_compilers', False)
    # Processing the call keyword arguments (line 904)
    kwargs_66114 = {}
    # Getting the type of 'set' (line 904)
    set_66112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 54), 'set', False)
    # Calling set(args, kwargs) (line 904)
    set_call_result_66115 = invoke(stypy.reporting.localization.Localization(__file__, 904, 54), set_66112, *[platform_compilers_66113], **kwargs_66114)
    
    # Applying the binary operator '-' (line 904)
    result_sub_66116 = python_operator(stypy.reporting.localization.Localization(__file__, 904, 24), '-', set_call_result_66111, set_call_result_66115)
    
    # Processing the call keyword arguments (line 904)
    kwargs_66117 = {}
    # Getting the type of 'list' (line 904)
    list_66104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 19), 'list', False)
    # Calling list(args, kwargs) (line 904)
    list_call_result_66118 = invoke(stypy.reporting.localization.Localization(__file__, 904, 19), list_66104, *[result_sub_66116], **kwargs_66117)
    
    # Assigning a type to the variable 'compilers_ni' (line 904)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 904, 4), 'compilers_ni', list_call_result_66118)
    
    # Assigning a ListComp to a Name (line 905):
    
    # Assigning a ListComp to a Name (line 905):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'compilers_ni' (line 906)
    compilers_ni_66131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 30), 'compilers_ni')
    comprehension_66132 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 905, 20), compilers_ni_66131)
    # Assigning a type to the variable 'fc' (line 905)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 905, 20), 'fc', comprehension_66132)
    
    # Obtaining an instance of the builtin type 'tuple' (line 905)
    tuple_66119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 905)
    # Adding element type (line 905)
    str_66120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, 21), 'str', 'fcompiler=')
    # Getting the type of 'fc' (line 905)
    fc_66121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 34), 'fc')
    # Applying the binary operator '+' (line 905)
    result_add_66122 = python_operator(stypy.reporting.localization.Localization(__file__, 905, 21), '+', str_66120, fc_66121)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 905, 21), tuple_66119, result_add_66122)
    # Adding element type (line 905)
    # Getting the type of 'None' (line 905)
    None_66123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 38), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 905, 21), tuple_66119, None_66123)
    # Adding element type (line 905)
    
    # Obtaining the type of the subscript
    int_66124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, 64), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'fc' (line 905)
    fc_66125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 60), 'fc')
    # Getting the type of 'fcompiler_class' (line 905)
    fcompiler_class_66126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 44), 'fcompiler_class')
    # Obtaining the member '__getitem__' of a type (line 905)
    getitem___66127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 44), fcompiler_class_66126, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 905)
    subscript_call_result_66128 = invoke(stypy.reporting.localization.Localization(__file__, 905, 44), getitem___66127, fc_66125)
    
    # Obtaining the member '__getitem__' of a type (line 905)
    getitem___66129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 44), subscript_call_result_66128, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 905)
    subscript_call_result_66130 = invoke(stypy.reporting.localization.Localization(__file__, 905, 44), getitem___66129, int_66124)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 905, 21), tuple_66119, subscript_call_result_66130)
    
    list_66133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 905, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 905, 20), list_66133, tuple_66119)
    # Assigning a type to the variable 'compilers_ni' (line 905)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 905, 4), 'compilers_ni', list_66133)
    
    # Call to sort(...): (line 908)
    # Processing the call keyword arguments (line 908)
    kwargs_66136 = {}
    # Getting the type of 'compilers' (line 908)
    compilers_66134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 4), 'compilers', False)
    # Obtaining the member 'sort' of a type (line 908)
    sort_66135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 908, 4), compilers_66134, 'sort')
    # Calling sort(args, kwargs) (line 908)
    sort_call_result_66137 = invoke(stypy.reporting.localization.Localization(__file__, 908, 4), sort_66135, *[], **kwargs_66136)
    
    
    # Call to sort(...): (line 909)
    # Processing the call keyword arguments (line 909)
    kwargs_66140 = {}
    # Getting the type of 'compilers_na' (line 909)
    compilers_na_66138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 4), 'compilers_na', False)
    # Obtaining the member 'sort' of a type (line 909)
    sort_66139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 4), compilers_na_66138, 'sort')
    # Calling sort(args, kwargs) (line 909)
    sort_call_result_66141 = invoke(stypy.reporting.localization.Localization(__file__, 909, 4), sort_66139, *[], **kwargs_66140)
    
    
    # Call to sort(...): (line 910)
    # Processing the call keyword arguments (line 910)
    kwargs_66144 = {}
    # Getting the type of 'compilers_ni' (line 910)
    compilers_ni_66142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 4), 'compilers_ni', False)
    # Obtaining the member 'sort' of a type (line 910)
    sort_66143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 910, 4), compilers_ni_66142, 'sort')
    # Calling sort(args, kwargs) (line 910)
    sort_call_result_66145 = invoke(stypy.reporting.localization.Localization(__file__, 910, 4), sort_66143, *[], **kwargs_66144)
    
    
    # Assigning a Call to a Name (line 911):
    
    # Assigning a Call to a Name (line 911):
    
    # Call to FancyGetopt(...): (line 911)
    # Processing the call arguments (line 911)
    # Getting the type of 'compilers' (line 911)
    compilers_66147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 33), 'compilers', False)
    # Processing the call keyword arguments (line 911)
    kwargs_66148 = {}
    # Getting the type of 'FancyGetopt' (line 911)
    FancyGetopt_66146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 21), 'FancyGetopt', False)
    # Calling FancyGetopt(args, kwargs) (line 911)
    FancyGetopt_call_result_66149 = invoke(stypy.reporting.localization.Localization(__file__, 911, 21), FancyGetopt_66146, *[compilers_66147], **kwargs_66148)
    
    # Assigning a type to the variable 'pretty_printer' (line 911)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 4), 'pretty_printer', FancyGetopt_call_result_66149)
    
    # Call to print_help(...): (line 912)
    # Processing the call arguments (line 912)
    str_66152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 912, 30), 'str', 'Fortran compilers found:')
    # Processing the call keyword arguments (line 912)
    kwargs_66153 = {}
    # Getting the type of 'pretty_printer' (line 912)
    pretty_printer_66150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 4), 'pretty_printer', False)
    # Obtaining the member 'print_help' of a type (line 912)
    print_help_66151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 912, 4), pretty_printer_66150, 'print_help')
    # Calling print_help(args, kwargs) (line 912)
    print_help_call_result_66154 = invoke(stypy.reporting.localization.Localization(__file__, 912, 4), print_help_66151, *[str_66152], **kwargs_66153)
    
    
    # Assigning a Call to a Name (line 913):
    
    # Assigning a Call to a Name (line 913):
    
    # Call to FancyGetopt(...): (line 913)
    # Processing the call arguments (line 913)
    # Getting the type of 'compilers_na' (line 913)
    compilers_na_66156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 33), 'compilers_na', False)
    # Processing the call keyword arguments (line 913)
    kwargs_66157 = {}
    # Getting the type of 'FancyGetopt' (line 913)
    FancyGetopt_66155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 21), 'FancyGetopt', False)
    # Calling FancyGetopt(args, kwargs) (line 913)
    FancyGetopt_call_result_66158 = invoke(stypy.reporting.localization.Localization(__file__, 913, 21), FancyGetopt_66155, *[compilers_na_66156], **kwargs_66157)
    
    # Assigning a type to the variable 'pretty_printer' (line 913)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 4), 'pretty_printer', FancyGetopt_call_result_66158)
    
    # Call to print_help(...): (line 914)
    # Processing the call arguments (line 914)
    str_66161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 914, 30), 'str', 'Compilers available for this platform, but not found:')
    # Processing the call keyword arguments (line 914)
    kwargs_66162 = {}
    # Getting the type of 'pretty_printer' (line 914)
    pretty_printer_66159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 914, 4), 'pretty_printer', False)
    # Obtaining the member 'print_help' of a type (line 914)
    print_help_66160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 914, 4), pretty_printer_66159, 'print_help')
    # Calling print_help(args, kwargs) (line 914)
    print_help_call_result_66163 = invoke(stypy.reporting.localization.Localization(__file__, 914, 4), print_help_66160, *[str_66161], **kwargs_66162)
    
    
    # Getting the type of 'compilers_ni' (line 916)
    compilers_ni_66164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 7), 'compilers_ni')
    # Testing the type of an if condition (line 916)
    if_condition_66165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 916, 4), compilers_ni_66164)
    # Assigning a type to the variable 'if_condition_66165' (line 916)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 916, 4), 'if_condition_66165', if_condition_66165)
    # SSA begins for if statement (line 916)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 917):
    
    # Assigning a Call to a Name (line 917):
    
    # Call to FancyGetopt(...): (line 917)
    # Processing the call arguments (line 917)
    # Getting the type of 'compilers_ni' (line 917)
    compilers_ni_66167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 37), 'compilers_ni', False)
    # Processing the call keyword arguments (line 917)
    kwargs_66168 = {}
    # Getting the type of 'FancyGetopt' (line 917)
    FancyGetopt_66166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 917, 25), 'FancyGetopt', False)
    # Calling FancyGetopt(args, kwargs) (line 917)
    FancyGetopt_call_result_66169 = invoke(stypy.reporting.localization.Localization(__file__, 917, 25), FancyGetopt_66166, *[compilers_ni_66167], **kwargs_66168)
    
    # Assigning a type to the variable 'pretty_printer' (line 917)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 917, 8), 'pretty_printer', FancyGetopt_call_result_66169)
    
    # Call to print_help(...): (line 918)
    # Processing the call arguments (line 918)
    str_66172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 918, 34), 'str', 'Compilers not available on this platform:')
    # Processing the call keyword arguments (line 918)
    kwargs_66173 = {}
    # Getting the type of 'pretty_printer' (line 918)
    pretty_printer_66170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 918, 8), 'pretty_printer', False)
    # Obtaining the member 'print_help' of a type (line 918)
    print_help_66171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 918, 8), pretty_printer_66170, 'print_help')
    # Calling print_help(args, kwargs) (line 918)
    print_help_call_result_66174 = invoke(stypy.reporting.localization.Localization(__file__, 918, 8), print_help_66171, *[str_66172], **kwargs_66173)
    
    # SSA join for if statement (line 916)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 919)
    # Processing the call arguments (line 919)
    str_66176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 919, 10), 'str', "For compiler details, run 'config_fc --verbose' setup command.")
    # Processing the call keyword arguments (line 919)
    kwargs_66177 = {}
    # Getting the type of 'print' (line 919)
    print_66175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 919, 4), 'print', False)
    # Calling print(args, kwargs) (line 919)
    print_call_result_66178 = invoke(stypy.reporting.localization.Localization(__file__, 919, 4), print_66175, *[str_66176], **kwargs_66177)
    
    
    # ################# End of 'show_fcompilers(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'show_fcompilers' in the type store
    # Getting the type of 'stypy_return_type' (line 861)
    stypy_return_type_66179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 861, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66179)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'show_fcompilers'
    return stypy_return_type_66179

# Assigning a type to the variable 'show_fcompilers' (line 861)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 861, 0), 'show_fcompilers', show_fcompilers)

@norecursion
def dummy_fortran_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dummy_fortran_file'
    module_type_store = module_type_store.open_function_context('dummy_fortran_file', 922, 0, False)
    
    # Passed parameters checking function
    dummy_fortran_file.stypy_localization = localization
    dummy_fortran_file.stypy_type_of_self = None
    dummy_fortran_file.stypy_type_store = module_type_store
    dummy_fortran_file.stypy_function_name = 'dummy_fortran_file'
    dummy_fortran_file.stypy_param_names_list = []
    dummy_fortran_file.stypy_varargs_param_name = None
    dummy_fortran_file.stypy_kwargs_param_name = None
    dummy_fortran_file.stypy_call_defaults = defaults
    dummy_fortran_file.stypy_call_varargs = varargs
    dummy_fortran_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dummy_fortran_file', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dummy_fortran_file', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dummy_fortran_file(...)' code ##################

    
    # Assigning a Call to a Tuple (line 923):
    
    # Assigning a Call to a Name:
    
    # Call to make_temp_file(...): (line 923)
    # Processing the call keyword arguments (line 923)
    str_66181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 37), 'str', '.f')
    keyword_66182 = str_66181
    kwargs_66183 = {'suffix': keyword_66182}
    # Getting the type of 'make_temp_file' (line 923)
    make_temp_file_66180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 15), 'make_temp_file', False)
    # Calling make_temp_file(args, kwargs) (line 923)
    make_temp_file_call_result_66184 = invoke(stypy.reporting.localization.Localization(__file__, 923, 15), make_temp_file_66180, *[], **kwargs_66183)
    
    # Assigning a type to the variable 'call_assignment_63500' (line 923)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 4), 'call_assignment_63500', make_temp_file_call_result_66184)
    
    # Assigning a Call to a Name (line 923):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_66187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 4), 'int')
    # Processing the call keyword arguments
    kwargs_66188 = {}
    # Getting the type of 'call_assignment_63500' (line 923)
    call_assignment_63500_66185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 4), 'call_assignment_63500', False)
    # Obtaining the member '__getitem__' of a type (line 923)
    getitem___66186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 4), call_assignment_63500_66185, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_66189 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___66186, *[int_66187], **kwargs_66188)
    
    # Assigning a type to the variable 'call_assignment_63501' (line 923)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 4), 'call_assignment_63501', getitem___call_result_66189)
    
    # Assigning a Name to a Name (line 923):
    # Getting the type of 'call_assignment_63501' (line 923)
    call_assignment_63501_66190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 4), 'call_assignment_63501')
    # Assigning a type to the variable 'fo' (line 923)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 4), 'fo', call_assignment_63501_66190)
    
    # Assigning a Call to a Name (line 923):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_66193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 923, 4), 'int')
    # Processing the call keyword arguments
    kwargs_66194 = {}
    # Getting the type of 'call_assignment_63500' (line 923)
    call_assignment_63500_66191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 4), 'call_assignment_63500', False)
    # Obtaining the member '__getitem__' of a type (line 923)
    getitem___66192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 923, 4), call_assignment_63500_66191, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_66195 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___66192, *[int_66193], **kwargs_66194)
    
    # Assigning a type to the variable 'call_assignment_63502' (line 923)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 4), 'call_assignment_63502', getitem___call_result_66195)
    
    # Assigning a Name to a Name (line 923):
    # Getting the type of 'call_assignment_63502' (line 923)
    call_assignment_63502_66196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 923, 4), 'call_assignment_63502')
    # Assigning a type to the variable 'name' (line 923)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 923, 8), 'name', call_assignment_63502_66196)
    
    # Call to write(...): (line 924)
    # Processing the call arguments (line 924)
    str_66199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 924, 13), 'str', '      subroutine dummy()\n      end\n')
    # Processing the call keyword arguments (line 924)
    kwargs_66200 = {}
    # Getting the type of 'fo' (line 924)
    fo_66197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 4), 'fo', False)
    # Obtaining the member 'write' of a type (line 924)
    write_66198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 924, 4), fo_66197, 'write')
    # Calling write(args, kwargs) (line 924)
    write_call_result_66201 = invoke(stypy.reporting.localization.Localization(__file__, 924, 4), write_66198, *[str_66199], **kwargs_66200)
    
    
    # Call to close(...): (line 925)
    # Processing the call keyword arguments (line 925)
    kwargs_66204 = {}
    # Getting the type of 'fo' (line 925)
    fo_66202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 925, 4), 'fo', False)
    # Obtaining the member 'close' of a type (line 925)
    close_66203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 925, 4), fo_66202, 'close')
    # Calling close(args, kwargs) (line 925)
    close_call_result_66205 = invoke(stypy.reporting.localization.Localization(__file__, 925, 4), close_66203, *[], **kwargs_66204)
    
    
    # Obtaining the type of the subscript
    int_66206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 926, 17), 'int')
    slice_66207 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 926, 11), None, int_66206, None)
    # Getting the type of 'name' (line 926)
    name_66208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 926, 11), 'name')
    # Obtaining the member '__getitem__' of a type (line 926)
    getitem___66209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 926, 11), name_66208, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 926)
    subscript_call_result_66210 = invoke(stypy.reporting.localization.Localization(__file__, 926, 11), getitem___66209, slice_66207)
    
    # Assigning a type to the variable 'stypy_return_type' (line 926)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 926, 4), 'stypy_return_type', subscript_call_result_66210)
    
    # ################# End of 'dummy_fortran_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dummy_fortran_file' in the type store
    # Getting the type of 'stypy_return_type' (line 922)
    stypy_return_type_66211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 922, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66211)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dummy_fortran_file'
    return stypy_return_type_66211

# Assigning a type to the variable 'dummy_fortran_file' (line 922)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 922, 0), 'dummy_fortran_file', dummy_fortran_file)

# Assigning a Attribute to a Name (line 929):

# Assigning a Attribute to a Name (line 929):

# Call to compile(...): (line 929)
# Processing the call arguments (line 929)
str_66214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 929, 23), 'str', '.*[.](for|ftn|f77|f)\\Z')
# Getting the type of 're' (line 929)
re_66215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 50), 're', False)
# Obtaining the member 'I' of a type (line 929)
I_66216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 929, 50), re_66215, 'I')
# Processing the call keyword arguments (line 929)
kwargs_66217 = {}
# Getting the type of 're' (line 929)
re_66212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 929, 12), 're', False)
# Obtaining the member 'compile' of a type (line 929)
compile_66213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 929, 12), re_66212, 'compile')
# Calling compile(args, kwargs) (line 929)
compile_call_result_66218 = invoke(stypy.reporting.localization.Localization(__file__, 929, 12), compile_66213, *[str_66214, I_66216], **kwargs_66217)

# Obtaining the member 'match' of a type (line 929)
match_66219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 929, 12), compile_call_result_66218, 'match')
# Assigning a type to the variable 'is_f_file' (line 929)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 929, 0), 'is_f_file', match_66219)

# Assigning a Attribute to a Name (line 930):

# Assigning a Attribute to a Name (line 930):

# Call to compile(...): (line 930)
# Processing the call arguments (line 930)
str_66222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 930, 27), 'str', '-[*]-\\s*fortran\\s*-[*]-')
# Getting the type of 're' (line 930)
re_66223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 55), 're', False)
# Obtaining the member 'I' of a type (line 930)
I_66224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 930, 55), re_66223, 'I')
# Processing the call keyword arguments (line 930)
kwargs_66225 = {}
# Getting the type of 're' (line 930)
re_66220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 16), 're', False)
# Obtaining the member 'compile' of a type (line 930)
compile_66221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 930, 16), re_66220, 'compile')
# Calling compile(args, kwargs) (line 930)
compile_call_result_66226 = invoke(stypy.reporting.localization.Localization(__file__, 930, 16), compile_66221, *[str_66222, I_66224], **kwargs_66225)

# Obtaining the member 'search' of a type (line 930)
search_66227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 930, 16), compile_call_result_66226, 'search')
# Assigning a type to the variable '_has_f_header' (line 930)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 930, 0), '_has_f_header', search_66227)

# Assigning a Attribute to a Name (line 931):

# Assigning a Attribute to a Name (line 931):

# Call to compile(...): (line 931)
# Processing the call arguments (line 931)
str_66230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 931, 29), 'str', '-[*]-\\s*f90\\s*-[*]-')
# Getting the type of 're' (line 931)
re_66231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 53), 're', False)
# Obtaining the member 'I' of a type (line 931)
I_66232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 931, 53), re_66231, 'I')
# Processing the call keyword arguments (line 931)
kwargs_66233 = {}
# Getting the type of 're' (line 931)
re_66228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 18), 're', False)
# Obtaining the member 'compile' of a type (line 931)
compile_66229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 931, 18), re_66228, 'compile')
# Calling compile(args, kwargs) (line 931)
compile_call_result_66234 = invoke(stypy.reporting.localization.Localization(__file__, 931, 18), compile_66229, *[str_66230, I_66232], **kwargs_66233)

# Obtaining the member 'search' of a type (line 931)
search_66235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 931, 18), compile_call_result_66234, 'search')
# Assigning a type to the variable '_has_f90_header' (line 931)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 931, 0), '_has_f90_header', search_66235)

# Assigning a Attribute to a Name (line 932):

# Assigning a Attribute to a Name (line 932):

# Call to compile(...): (line 932)
# Processing the call arguments (line 932)
str_66238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 29), 'str', '-[*]-\\s*fix\\s*-[*]-')
# Getting the type of 're' (line 932)
re_66239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 53), 're', False)
# Obtaining the member 'I' of a type (line 932)
I_66240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 932, 53), re_66239, 'I')
# Processing the call keyword arguments (line 932)
kwargs_66241 = {}
# Getting the type of 're' (line 932)
re_66236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 18), 're', False)
# Obtaining the member 'compile' of a type (line 932)
compile_66237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 932, 18), re_66236, 'compile')
# Calling compile(args, kwargs) (line 932)
compile_call_result_66242 = invoke(stypy.reporting.localization.Localization(__file__, 932, 18), compile_66237, *[str_66238, I_66240], **kwargs_66241)

# Obtaining the member 'search' of a type (line 932)
search_66243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 932, 18), compile_call_result_66242, 'search')
# Assigning a type to the variable '_has_fix_header' (line 932)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 932, 0), '_has_fix_header', search_66243)

# Assigning a Attribute to a Name (line 933):

# Assigning a Attribute to a Name (line 933):

# Call to compile(...): (line 933)
# Processing the call arguments (line 933)
str_66246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 933, 29), 'str', '[^c*!]\\s*[^\\s\\d\\t]')
# Getting the type of 're' (line 933)
re_66247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 52), 're', False)
# Obtaining the member 'I' of a type (line 933)
I_66248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 933, 52), re_66247, 'I')
# Processing the call keyword arguments (line 933)
kwargs_66249 = {}
# Getting the type of 're' (line 933)
re_66244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 933, 18), 're', False)
# Obtaining the member 'compile' of a type (line 933)
compile_66245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 933, 18), re_66244, 'compile')
# Calling compile(args, kwargs) (line 933)
compile_call_result_66250 = invoke(stypy.reporting.localization.Localization(__file__, 933, 18), compile_66245, *[str_66246, I_66248], **kwargs_66249)

# Obtaining the member 'match' of a type (line 933)
match_66251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 933, 18), compile_call_result_66250, 'match')
# Assigning a type to the variable '_free_f90_start' (line 933)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 933, 0), '_free_f90_start', match_66251)

@norecursion
def is_free_format(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_free_format'
    module_type_store = module_type_store.open_function_context('is_free_format', 935, 0, False)
    
    # Passed parameters checking function
    is_free_format.stypy_localization = localization
    is_free_format.stypy_type_of_self = None
    is_free_format.stypy_type_store = module_type_store
    is_free_format.stypy_function_name = 'is_free_format'
    is_free_format.stypy_param_names_list = ['file']
    is_free_format.stypy_varargs_param_name = None
    is_free_format.stypy_kwargs_param_name = None
    is_free_format.stypy_call_defaults = defaults
    is_free_format.stypy_call_varargs = varargs
    is_free_format.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_free_format', ['file'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_free_format', localization, ['file'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_free_format(...)' code ##################

    str_66252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 936, 4), 'str', 'Check if file is in free format Fortran.')
    
    # Assigning a Num to a Name (line 939):
    
    # Assigning a Num to a Name (line 939):
    int_66253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 939, 13), 'int')
    # Assigning a type to the variable 'result' (line 939)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 4), 'result', int_66253)
    
    # Assigning a Call to a Name (line 940):
    
    # Assigning a Call to a Name (line 940):
    
    # Call to open_latin1(...): (line 940)
    # Processing the call arguments (line 940)
    # Getting the type of 'file' (line 940)
    file_66255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 20), 'file', False)
    str_66256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 940, 26), 'str', 'r')
    # Processing the call keyword arguments (line 940)
    kwargs_66257 = {}
    # Getting the type of 'open_latin1' (line 940)
    open_latin1_66254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 940, 8), 'open_latin1', False)
    # Calling open_latin1(args, kwargs) (line 940)
    open_latin1_call_result_66258 = invoke(stypy.reporting.localization.Localization(__file__, 940, 8), open_latin1_66254, *[file_66255, str_66256], **kwargs_66257)
    
    # Assigning a type to the variable 'f' (line 940)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 940, 4), 'f', open_latin1_call_result_66258)
    
    # Assigning a Call to a Name (line 941):
    
    # Assigning a Call to a Name (line 941):
    
    # Call to readline(...): (line 941)
    # Processing the call keyword arguments (line 941)
    kwargs_66261 = {}
    # Getting the type of 'f' (line 941)
    f_66259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 11), 'f', False)
    # Obtaining the member 'readline' of a type (line 941)
    readline_66260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 941, 11), f_66259, 'readline')
    # Calling readline(args, kwargs) (line 941)
    readline_call_result_66262 = invoke(stypy.reporting.localization.Localization(__file__, 941, 11), readline_66260, *[], **kwargs_66261)
    
    # Assigning a type to the variable 'line' (line 941)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 941, 4), 'line', readline_call_result_66262)
    
    # Assigning a Num to a Name (line 942):
    
    # Assigning a Num to a Name (line 942):
    int_66263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 8), 'int')
    # Assigning a type to the variable 'n' (line 942)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 942, 4), 'n', int_66263)
    
    
    # Call to _has_f_header(...): (line 943)
    # Processing the call arguments (line 943)
    # Getting the type of 'line' (line 943)
    line_66265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 21), 'line', False)
    # Processing the call keyword arguments (line 943)
    kwargs_66266 = {}
    # Getting the type of '_has_f_header' (line 943)
    _has_f_header_66264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 943, 7), '_has_f_header', False)
    # Calling _has_f_header(args, kwargs) (line 943)
    _has_f_header_call_result_66267 = invoke(stypy.reporting.localization.Localization(__file__, 943, 7), _has_f_header_66264, *[line_66265], **kwargs_66266)
    
    # Testing the type of an if condition (line 943)
    if_condition_66268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 943, 4), _has_f_header_call_result_66267)
    # Assigning a type to the variable 'if_condition_66268' (line 943)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 943, 4), 'if_condition_66268', if_condition_66268)
    # SSA begins for if statement (line 943)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 944):
    
    # Assigning a Num to a Name (line 944):
    int_66269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 12), 'int')
    # Assigning a type to the variable 'n' (line 944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 8), 'n', int_66269)
    # SSA branch for the else part of an if statement (line 943)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to _has_f90_header(...): (line 945)
    # Processing the call arguments (line 945)
    # Getting the type of 'line' (line 945)
    line_66271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 25), 'line', False)
    # Processing the call keyword arguments (line 945)
    kwargs_66272 = {}
    # Getting the type of '_has_f90_header' (line 945)
    _has_f90_header_66270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 945, 9), '_has_f90_header', False)
    # Calling _has_f90_header(args, kwargs) (line 945)
    _has_f90_header_call_result_66273 = invoke(stypy.reporting.localization.Localization(__file__, 945, 9), _has_f90_header_66270, *[line_66271], **kwargs_66272)
    
    # Testing the type of an if condition (line 945)
    if_condition_66274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 945, 9), _has_f90_header_call_result_66273)
    # Assigning a type to the variable 'if_condition_66274' (line 945)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 945, 9), 'if_condition_66274', if_condition_66274)
    # SSA begins for if statement (line 945)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 946):
    
    # Assigning a Num to a Name (line 946):
    int_66275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 12), 'int')
    # Assigning a type to the variable 'n' (line 946)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 8), 'n', int_66275)
    
    # Assigning a Num to a Name (line 947):
    
    # Assigning a Num to a Name (line 947):
    int_66276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 947, 17), 'int')
    # Assigning a type to the variable 'result' (line 947)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 8), 'result', int_66276)
    # SSA join for if statement (line 945)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 943)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'n' (line 948)
    n_66277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 10), 'n')
    int_66278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, 12), 'int')
    # Applying the binary operator '>' (line 948)
    result_gt_66279 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 10), '>', n_66277, int_66278)
    
    # Getting the type of 'line' (line 948)
    line_66280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 18), 'line')
    # Applying the binary operator 'and' (line 948)
    result_and_keyword_66281 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 10), 'and', result_gt_66279, line_66280)
    
    # Testing the type of an if condition (line 948)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 948, 4), result_and_keyword_66281)
    # SSA begins for while statement (line 948)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 949):
    
    # Assigning a Call to a Name (line 949):
    
    # Call to rstrip(...): (line 949)
    # Processing the call keyword arguments (line 949)
    kwargs_66284 = {}
    # Getting the type of 'line' (line 949)
    line_66282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 15), 'line', False)
    # Obtaining the member 'rstrip' of a type (line 949)
    rstrip_66283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 949, 15), line_66282, 'rstrip')
    # Calling rstrip(args, kwargs) (line 949)
    rstrip_call_result_66285 = invoke(stypy.reporting.localization.Localization(__file__, 949, 15), rstrip_66283, *[], **kwargs_66284)
    
    # Assigning a type to the variable 'line' (line 949)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 949, 8), 'line', rstrip_call_result_66285)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'line' (line 950)
    line_66286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 11), 'line')
    
    
    # Obtaining the type of the subscript
    int_66287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, 25), 'int')
    # Getting the type of 'line' (line 950)
    line_66288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 950, 20), 'line')
    # Obtaining the member '__getitem__' of a type (line 950)
    getitem___66289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 950, 20), line_66288, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 950)
    subscript_call_result_66290 = invoke(stypy.reporting.localization.Localization(__file__, 950, 20), getitem___66289, int_66287)
    
    str_66291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, 29), 'str', '!')
    # Applying the binary operator '!=' (line 950)
    result_ne_66292 = python_operator(stypy.reporting.localization.Localization(__file__, 950, 20), '!=', subscript_call_result_66290, str_66291)
    
    # Applying the binary operator 'and' (line 950)
    result_and_keyword_66293 = python_operator(stypy.reporting.localization.Localization(__file__, 950, 11), 'and', line_66286, result_ne_66292)
    
    # Testing the type of an if condition (line 950)
    if_condition_66294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 950, 8), result_and_keyword_66293)
    # Assigning a type to the variable 'if_condition_66294' (line 950)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 950, 8), 'if_condition_66294', if_condition_66294)
    # SSA begins for if statement (line 950)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'n' (line 951)
    n_66295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 12), 'n')
    int_66296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 951, 17), 'int')
    # Applying the binary operator '-=' (line 951)
    result_isub_66297 = python_operator(stypy.reporting.localization.Localization(__file__, 951, 12), '-=', n_66295, int_66296)
    # Assigning a type to the variable 'n' (line 951)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 951, 12), 'n', result_isub_66297)
    
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_66298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 21), 'int')
    # Getting the type of 'line' (line 952)
    line_66299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 16), 'line')
    # Obtaining the member '__getitem__' of a type (line 952)
    getitem___66300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 16), line_66299, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 952)
    subscript_call_result_66301 = invoke(stypy.reporting.localization.Localization(__file__, 952, 16), getitem___66300, int_66298)
    
    str_66302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 25), 'str', '\t')
    # Applying the binary operator '!=' (line 952)
    result_ne_66303 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 16), '!=', subscript_call_result_66301, str_66302)
    
    
    # Call to _free_f90_start(...): (line 952)
    # Processing the call arguments (line 952)
    
    # Obtaining the type of the subscript
    int_66305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 56), 'int')
    slice_66306 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 952, 50), None, int_66305, None)
    # Getting the type of 'line' (line 952)
    line_66307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 50), 'line', False)
    # Obtaining the member '__getitem__' of a type (line 952)
    getitem___66308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 50), line_66307, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 952)
    subscript_call_result_66309 = invoke(stypy.reporting.localization.Localization(__file__, 952, 50), getitem___66308, slice_66306)
    
    # Processing the call keyword arguments (line 952)
    kwargs_66310 = {}
    # Getting the type of '_free_f90_start' (line 952)
    _free_f90_start_66304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 34), '_free_f90_start', False)
    # Calling _free_f90_start(args, kwargs) (line 952)
    _free_f90_start_call_result_66311 = invoke(stypy.reporting.localization.Localization(__file__, 952, 34), _free_f90_start_66304, *[subscript_call_result_66309], **kwargs_66310)
    
    # Applying the binary operator 'and' (line 952)
    result_and_keyword_66312 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 16), 'and', result_ne_66303, _free_f90_start_call_result_66311)
    
    
    
    # Obtaining the type of the subscript
    int_66313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 69), 'int')
    slice_66314 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 952, 64), int_66313, None, None)
    # Getting the type of 'line' (line 952)
    line_66315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 64), 'line')
    # Obtaining the member '__getitem__' of a type (line 952)
    getitem___66316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 952, 64), line_66315, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 952)
    subscript_call_result_66317 = invoke(stypy.reporting.localization.Localization(__file__, 952, 64), getitem___66316, slice_66314)
    
    str_66318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 952, 75), 'str', '&')
    # Applying the binary operator '==' (line 952)
    result_eq_66319 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 64), '==', subscript_call_result_66317, str_66318)
    
    # Applying the binary operator 'or' (line 952)
    result_or_keyword_66320 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 15), 'or', result_and_keyword_66312, result_eq_66319)
    
    # Testing the type of an if condition (line 952)
    if_condition_66321 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 952, 12), result_or_keyword_66320)
    # Assigning a type to the variable 'if_condition_66321' (line 952)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 952, 12), 'if_condition_66321', if_condition_66321)
    # SSA begins for if statement (line 952)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 953):
    
    # Assigning a Num to a Name (line 953):
    int_66322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 953, 25), 'int')
    # Assigning a type to the variable 'result' (line 953)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 953, 16), 'result', int_66322)
    # SSA join for if statement (line 952)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 950)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 955):
    
    # Assigning a Call to a Name (line 955):
    
    # Call to readline(...): (line 955)
    # Processing the call keyword arguments (line 955)
    kwargs_66325 = {}
    # Getting the type of 'f' (line 955)
    f_66323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 15), 'f', False)
    # Obtaining the member 'readline' of a type (line 955)
    readline_66324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 955, 15), f_66323, 'readline')
    # Calling readline(args, kwargs) (line 955)
    readline_call_result_66326 = invoke(stypy.reporting.localization.Localization(__file__, 955, 15), readline_66324, *[], **kwargs_66325)
    
    # Assigning a type to the variable 'line' (line 955)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 8), 'line', readline_call_result_66326)
    # SSA join for while statement (line 948)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to close(...): (line 956)
    # Processing the call keyword arguments (line 956)
    kwargs_66329 = {}
    # Getting the type of 'f' (line 956)
    f_66327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 4), 'f', False)
    # Obtaining the member 'close' of a type (line 956)
    close_66328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 956, 4), f_66327, 'close')
    # Calling close(args, kwargs) (line 956)
    close_call_result_66330 = invoke(stypy.reporting.localization.Localization(__file__, 956, 4), close_66328, *[], **kwargs_66329)
    
    # Getting the type of 'result' (line 957)
    result_66331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 957)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 957, 4), 'stypy_return_type', result_66331)
    
    # ################# End of 'is_free_format(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_free_format' in the type store
    # Getting the type of 'stypy_return_type' (line 935)
    stypy_return_type_66332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66332)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_free_format'
    return stypy_return_type_66332

# Assigning a type to the variable 'is_free_format' (line 935)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 935, 0), 'is_free_format', is_free_format)

@norecursion
def has_f90_header(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'has_f90_header'
    module_type_store = module_type_store.open_function_context('has_f90_header', 959, 0, False)
    
    # Passed parameters checking function
    has_f90_header.stypy_localization = localization
    has_f90_header.stypy_type_of_self = None
    has_f90_header.stypy_type_store = module_type_store
    has_f90_header.stypy_function_name = 'has_f90_header'
    has_f90_header.stypy_param_names_list = ['src']
    has_f90_header.stypy_varargs_param_name = None
    has_f90_header.stypy_kwargs_param_name = None
    has_f90_header.stypy_call_defaults = defaults
    has_f90_header.stypy_call_varargs = varargs
    has_f90_header.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'has_f90_header', ['src'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'has_f90_header', localization, ['src'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'has_f90_header(...)' code ##################

    
    # Assigning a Call to a Name (line 960):
    
    # Assigning a Call to a Name (line 960):
    
    # Call to open_latin1(...): (line 960)
    # Processing the call arguments (line 960)
    # Getting the type of 'src' (line 960)
    src_66334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 20), 'src', False)
    str_66335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 25), 'str', 'r')
    # Processing the call keyword arguments (line 960)
    kwargs_66336 = {}
    # Getting the type of 'open_latin1' (line 960)
    open_latin1_66333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 8), 'open_latin1', False)
    # Calling open_latin1(args, kwargs) (line 960)
    open_latin1_call_result_66337 = invoke(stypy.reporting.localization.Localization(__file__, 960, 8), open_latin1_66333, *[src_66334, str_66335], **kwargs_66336)
    
    # Assigning a type to the variable 'f' (line 960)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 960, 4), 'f', open_latin1_call_result_66337)
    
    # Assigning a Call to a Name (line 961):
    
    # Assigning a Call to a Name (line 961):
    
    # Call to readline(...): (line 961)
    # Processing the call keyword arguments (line 961)
    kwargs_66340 = {}
    # Getting the type of 'f' (line 961)
    f_66338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 11), 'f', False)
    # Obtaining the member 'readline' of a type (line 961)
    readline_66339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 11), f_66338, 'readline')
    # Calling readline(args, kwargs) (line 961)
    readline_call_result_66341 = invoke(stypy.reporting.localization.Localization(__file__, 961, 11), readline_66339, *[], **kwargs_66340)
    
    # Assigning a type to the variable 'line' (line 961)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 961, 4), 'line', readline_call_result_66341)
    
    # Call to close(...): (line 962)
    # Processing the call keyword arguments (line 962)
    kwargs_66344 = {}
    # Getting the type of 'f' (line 962)
    f_66342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 962, 4), 'f', False)
    # Obtaining the member 'close' of a type (line 962)
    close_66343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 962, 4), f_66342, 'close')
    # Calling close(args, kwargs) (line 962)
    close_call_result_66345 = invoke(stypy.reporting.localization.Localization(__file__, 962, 4), close_66343, *[], **kwargs_66344)
    
    
    # Evaluating a boolean operation
    
    # Call to _has_f90_header(...): (line 963)
    # Processing the call arguments (line 963)
    # Getting the type of 'line' (line 963)
    line_66347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 27), 'line', False)
    # Processing the call keyword arguments (line 963)
    kwargs_66348 = {}
    # Getting the type of '_has_f90_header' (line 963)
    _has_f90_header_66346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 11), '_has_f90_header', False)
    # Calling _has_f90_header(args, kwargs) (line 963)
    _has_f90_header_call_result_66349 = invoke(stypy.reporting.localization.Localization(__file__, 963, 11), _has_f90_header_66346, *[line_66347], **kwargs_66348)
    
    
    # Call to _has_fix_header(...): (line 963)
    # Processing the call arguments (line 963)
    # Getting the type of 'line' (line 963)
    line_66351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 52), 'line', False)
    # Processing the call keyword arguments (line 963)
    kwargs_66352 = {}
    # Getting the type of '_has_fix_header' (line 963)
    _has_fix_header_66350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 36), '_has_fix_header', False)
    # Calling _has_fix_header(args, kwargs) (line 963)
    _has_fix_header_call_result_66353 = invoke(stypy.reporting.localization.Localization(__file__, 963, 36), _has_fix_header_66350, *[line_66351], **kwargs_66352)
    
    # Applying the binary operator 'or' (line 963)
    result_or_keyword_66354 = python_operator(stypy.reporting.localization.Localization(__file__, 963, 11), 'or', _has_f90_header_call_result_66349, _has_fix_header_call_result_66353)
    
    # Assigning a type to the variable 'stypy_return_type' (line 963)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 4), 'stypy_return_type', result_or_keyword_66354)
    
    # ################# End of 'has_f90_header(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'has_f90_header' in the type store
    # Getting the type of 'stypy_return_type' (line 959)
    stypy_return_type_66355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66355)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'has_f90_header'
    return stypy_return_type_66355

# Assigning a type to the variable 'has_f90_header' (line 959)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 959, 0), 'has_f90_header', has_f90_header)

# Assigning a Call to a Name (line 965):

# Assigning a Call to a Name (line 965):

# Call to compile(...): (line 965)
# Processing the call arguments (line 965)
str_66358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 965, 26), 'str', '(c|)f77flags\\s*\\(\\s*(?P<fcname>\\w+)\\s*\\)\\s*=\\s*(?P<fflags>.*)')
# Getting the type of 're' (line 965)
re_66359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 92), 're', False)
# Obtaining the member 'I' of a type (line 965)
I_66360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 965, 92), re_66359, 'I')
# Processing the call keyword arguments (line 965)
kwargs_66361 = {}
# Getting the type of 're' (line 965)
re_66356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 15), 're', False)
# Obtaining the member 'compile' of a type (line 965)
compile_66357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 965, 15), re_66356, 'compile')
# Calling compile(args, kwargs) (line 965)
compile_call_result_66362 = invoke(stypy.reporting.localization.Localization(__file__, 965, 15), compile_66357, *[str_66358, I_66360], **kwargs_66361)

# Assigning a type to the variable '_f77flags_re' (line 965)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 965, 0), '_f77flags_re', compile_call_result_66362)

@norecursion
def get_f77flags(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_f77flags'
    module_type_store = module_type_store.open_function_context('get_f77flags', 966, 0, False)
    
    # Passed parameters checking function
    get_f77flags.stypy_localization = localization
    get_f77flags.stypy_type_of_self = None
    get_f77flags.stypy_type_store = module_type_store
    get_f77flags.stypy_function_name = 'get_f77flags'
    get_f77flags.stypy_param_names_list = ['src']
    get_f77flags.stypy_varargs_param_name = None
    get_f77flags.stypy_kwargs_param_name = None
    get_f77flags.stypy_call_defaults = defaults
    get_f77flags.stypy_call_varargs = varargs
    get_f77flags.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_f77flags', ['src'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_f77flags', localization, ['src'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_f77flags(...)' code ##################

    str_66363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 971, (-1)), 'str', '\n    Search the first 20 lines of fortran 77 code for line pattern\n      `CF77FLAGS(<fcompiler type>)=<f77 flags>`\n    Return a dictionary {<fcompiler type>:<f77 flags>}.\n    ')
    
    # Assigning a Dict to a Name (line 972):
    
    # Assigning a Dict to a Name (line 972):
    
    # Obtaining an instance of the builtin type 'dict' (line 972)
    dict_66364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 972, 12), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 972)
    
    # Assigning a type to the variable 'flags' (line 972)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 4), 'flags', dict_66364)
    
    # Assigning a Call to a Name (line 973):
    
    # Assigning a Call to a Name (line 973):
    
    # Call to open_latin1(...): (line 973)
    # Processing the call arguments (line 973)
    # Getting the type of 'src' (line 973)
    src_66366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 20), 'src', False)
    str_66367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 973, 25), 'str', 'r')
    # Processing the call keyword arguments (line 973)
    kwargs_66368 = {}
    # Getting the type of 'open_latin1' (line 973)
    open_latin1_66365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 8), 'open_latin1', False)
    # Calling open_latin1(args, kwargs) (line 973)
    open_latin1_call_result_66369 = invoke(stypy.reporting.localization.Localization(__file__, 973, 8), open_latin1_66365, *[src_66366, str_66367], **kwargs_66368)
    
    # Assigning a type to the variable 'f' (line 973)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 973, 4), 'f', open_latin1_call_result_66369)
    
    # Assigning a Num to a Name (line 974):
    
    # Assigning a Num to a Name (line 974):
    int_66370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 8), 'int')
    # Assigning a type to the variable 'i' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 4), 'i', int_66370)
    
    # Getting the type of 'f' (line 975)
    f_66371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 975, 16), 'f')
    # Testing the type of a for loop iterable (line 975)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 975, 4), f_66371)
    # Getting the type of the for loop variable (line 975)
    for_loop_var_66372 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 975, 4), f_66371)
    # Assigning a type to the variable 'line' (line 975)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 975, 4), 'line', for_loop_var_66372)
    # SSA begins for a for statement (line 975)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'i' (line 976)
    i_66373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 8), 'i')
    int_66374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 976, 13), 'int')
    # Applying the binary operator '+=' (line 976)
    result_iadd_66375 = python_operator(stypy.reporting.localization.Localization(__file__, 976, 8), '+=', i_66373, int_66374)
    # Assigning a type to the variable 'i' (line 976)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 976, 8), 'i', result_iadd_66375)
    
    
    
    # Getting the type of 'i' (line 977)
    i_66376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 11), 'i')
    int_66377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 977, 13), 'int')
    # Applying the binary operator '>' (line 977)
    result_gt_66378 = python_operator(stypy.reporting.localization.Localization(__file__, 977, 11), '>', i_66376, int_66377)
    
    # Testing the type of an if condition (line 977)
    if_condition_66379 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 977, 8), result_gt_66378)
    # Assigning a type to the variable 'if_condition_66379' (line 977)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 977, 8), 'if_condition_66379', if_condition_66379)
    # SSA begins for if statement (line 977)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 977)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 978):
    
    # Assigning a Call to a Name (line 978):
    
    # Call to match(...): (line 978)
    # Processing the call arguments (line 978)
    # Getting the type of 'line' (line 978)
    line_66382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 31), 'line', False)
    # Processing the call keyword arguments (line 978)
    kwargs_66383 = {}
    # Getting the type of '_f77flags_re' (line 978)
    _f77flags_re_66380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 12), '_f77flags_re', False)
    # Obtaining the member 'match' of a type (line 978)
    match_66381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 978, 12), _f77flags_re_66380, 'match')
    # Calling match(args, kwargs) (line 978)
    match_call_result_66384 = invoke(stypy.reporting.localization.Localization(__file__, 978, 12), match_66381, *[line_66382], **kwargs_66383)
    
    # Assigning a type to the variable 'm' (line 978)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 978, 8), 'm', match_call_result_66384)
    
    
    # Getting the type of 'm' (line 979)
    m_66385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 979, 15), 'm')
    # Applying the 'not' unary operator (line 979)
    result_not__66386 = python_operator(stypy.reporting.localization.Localization(__file__, 979, 11), 'not', m_66385)
    
    # Testing the type of an if condition (line 979)
    if_condition_66387 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 979, 8), result_not__66386)
    # Assigning a type to the variable 'if_condition_66387' (line 979)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 979, 8), 'if_condition_66387', if_condition_66387)
    # SSA begins for if statement (line 979)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 979)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 980):
    
    # Assigning a Call to a Name (line 980):
    
    # Call to strip(...): (line 980)
    # Processing the call keyword arguments (line 980)
    kwargs_66394 = {}
    
    # Call to group(...): (line 980)
    # Processing the call arguments (line 980)
    str_66390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 25), 'str', 'fcname')
    # Processing the call keyword arguments (line 980)
    kwargs_66391 = {}
    # Getting the type of 'm' (line 980)
    m_66388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 17), 'm', False)
    # Obtaining the member 'group' of a type (line 980)
    group_66389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 980, 17), m_66388, 'group')
    # Calling group(args, kwargs) (line 980)
    group_call_result_66392 = invoke(stypy.reporting.localization.Localization(__file__, 980, 17), group_66389, *[str_66390], **kwargs_66391)
    
    # Obtaining the member 'strip' of a type (line 980)
    strip_66393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 980, 17), group_call_result_66392, 'strip')
    # Calling strip(args, kwargs) (line 980)
    strip_call_result_66395 = invoke(stypy.reporting.localization.Localization(__file__, 980, 17), strip_66393, *[], **kwargs_66394)
    
    # Assigning a type to the variable 'fcname' (line 980)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 8), 'fcname', strip_call_result_66395)
    
    # Assigning a Call to a Name (line 981):
    
    # Assigning a Call to a Name (line 981):
    
    # Call to strip(...): (line 981)
    # Processing the call keyword arguments (line 981)
    kwargs_66402 = {}
    
    # Call to group(...): (line 981)
    # Processing the call arguments (line 981)
    str_66398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, 25), 'str', 'fflags')
    # Processing the call keyword arguments (line 981)
    kwargs_66399 = {}
    # Getting the type of 'm' (line 981)
    m_66396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 17), 'm', False)
    # Obtaining the member 'group' of a type (line 981)
    group_66397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 17), m_66396, 'group')
    # Calling group(args, kwargs) (line 981)
    group_call_result_66400 = invoke(stypy.reporting.localization.Localization(__file__, 981, 17), group_66397, *[str_66398], **kwargs_66399)
    
    # Obtaining the member 'strip' of a type (line 981)
    strip_66401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 17), group_call_result_66400, 'strip')
    # Calling strip(args, kwargs) (line 981)
    strip_call_result_66403 = invoke(stypy.reporting.localization.Localization(__file__, 981, 17), strip_66401, *[], **kwargs_66402)
    
    # Assigning a type to the variable 'fflags' (line 981)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 981, 8), 'fflags', strip_call_result_66403)
    
    # Assigning a Call to a Subscript (line 982):
    
    # Assigning a Call to a Subscript (line 982):
    
    # Call to split_quoted(...): (line 982)
    # Processing the call arguments (line 982)
    # Getting the type of 'fflags' (line 982)
    fflags_66405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 37), 'fflags', False)
    # Processing the call keyword arguments (line 982)
    kwargs_66406 = {}
    # Getting the type of 'split_quoted' (line 982)
    split_quoted_66404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 24), 'split_quoted', False)
    # Calling split_quoted(args, kwargs) (line 982)
    split_quoted_call_result_66407 = invoke(stypy.reporting.localization.Localization(__file__, 982, 24), split_quoted_66404, *[fflags_66405], **kwargs_66406)
    
    # Getting the type of 'flags' (line 982)
    flags_66408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 8), 'flags')
    # Getting the type of 'fcname' (line 982)
    fcname_66409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 982, 14), 'fcname')
    # Storing an element on a container (line 982)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 982, 8), flags_66408, (fcname_66409, split_quoted_call_result_66407))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to close(...): (line 983)
    # Processing the call keyword arguments (line 983)
    kwargs_66412 = {}
    # Getting the type of 'f' (line 983)
    f_66410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 4), 'f', False)
    # Obtaining the member 'close' of a type (line 983)
    close_66411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 983, 4), f_66410, 'close')
    # Calling close(args, kwargs) (line 983)
    close_call_result_66413 = invoke(stypy.reporting.localization.Localization(__file__, 983, 4), close_66411, *[], **kwargs_66412)
    
    # Getting the type of 'flags' (line 984)
    flags_66414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 11), 'flags')
    # Assigning a type to the variable 'stypy_return_type' (line 984)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 984, 4), 'stypy_return_type', flags_66414)
    
    # ################# End of 'get_f77flags(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_f77flags' in the type store
    # Getting the type of 'stypy_return_type' (line 966)
    stypy_return_type_66415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66415)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_f77flags'
    return stypy_return_type_66415

# Assigning a type to the variable 'get_f77flags' (line 966)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 966, 0), 'get_f77flags', get_f77flags)

if (__name__ == '__main__'):
    
    # Call to show_fcompilers(...): (line 989)
    # Processing the call keyword arguments (line 989)
    kwargs_66417 = {}
    # Getting the type of 'show_fcompilers' (line 989)
    show_fcompilers_66416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 989, 4), 'show_fcompilers', False)
    # Calling show_fcompilers(args, kwargs) (line 989)
    show_fcompilers_call_result_66418 = invoke(stypy.reporting.localization.Localization(__file__, 989, 4), show_fcompilers_66416, *[], **kwargs_66417)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
