
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.cygwinccompiler
2: 
3: Provides the CygwinCCompiler class, a subclass of UnixCCompiler that
4: handles the Cygwin port of the GNU C compiler to Windows.  It also contains
5: the Mingw32CCompiler class which handles the mingw32 port of GCC (same as
6: cygwin in no-cygwin mode).
7: '''
8: 
9: # problems:
10: #
11: # * if you use a msvc compiled python version (1.5.2)
12: #   1. you have to insert a __GNUC__ section in its config.h
13: #   2. you have to generate an import library for its dll
14: #      - create a def-file for python??.dll
15: #      - create an import library using
16: #             dlltool --dllname python15.dll --def python15.def \
17: #                       --output-lib libpython15.a
18: #
19: #   see also http://starship.python.net/crew/kernr/mingw32/Notes.html
20: #
21: # * We put export_symbols in a def-file, and don't use
22: #   --export-all-symbols because it doesn't worked reliable in some
23: #   tested configurations. And because other windows compilers also
24: #   need their symbols specified this no serious problem.
25: #
26: # tested configurations:
27: #
28: # * cygwin gcc 2.91.57/ld 2.9.4/dllwrap 0.2.4 works
29: #   (after patching python's config.h and for C++ some other include files)
30: #   see also http://starship.python.net/crew/kernr/mingw32/Notes.html
31: # * mingw32 gcc 2.95.2/ld 2.9.4/dllwrap 0.2.4 works
32: #   (ld doesn't support -shared, so we use dllwrap)
33: # * cygwin gcc 2.95.2/ld 2.10.90/dllwrap 2.10.90 works now
34: #   - its dllwrap doesn't work, there is a bug in binutils 2.10.90
35: #     see also http://sources.redhat.com/ml/cygwin/2000-06/msg01274.html
36: #   - using gcc -mdll instead dllwrap doesn't work without -static because
37: #     it tries to link against dlls instead their import libraries. (If
38: #     it finds the dll first.)
39: #     By specifying -static we force ld to link against the import libraries,
40: #     this is windows standard and there are normally not the necessary symbols
41: #     in the dlls.
42: #   *** only the version of June 2000 shows these problems
43: # * cygwin gcc 3.2/ld 2.13.90 works
44: #   (ld supports -shared)
45: # * mingw gcc 3.2/ld 2.13 works
46: #   (ld supports -shared)
47: 
48: # This module should be kept compatible with Python 2.1.
49: 
50: __revision__ = "$Id$"
51: 
52: import os,sys,copy
53: from distutils.ccompiler import gen_preprocess_options, gen_lib_options
54: from distutils.unixccompiler import UnixCCompiler
55: from distutils.file_util import write_file
56: from distutils.errors import DistutilsExecError, CompileError, UnknownFileError
57: from distutils import log
58: 
59: def get_msvcr():
60:     '''Include the appropriate MSVC runtime library if Python was built
61:     with MSVC 7.0 or later.
62:     '''
63:     msc_pos = sys.version.find('MSC v.')
64:     if msc_pos != -1:
65:         msc_ver = sys.version[msc_pos+6:msc_pos+10]
66:         if msc_ver == '1300':
67:             # MSVC 7.0
68:             return ['msvcr70']
69:         elif msc_ver == '1310':
70:             # MSVC 7.1
71:             return ['msvcr71']
72:         elif msc_ver == '1400':
73:             # VS2005 / MSVC 8.0
74:             return ['msvcr80']
75:         elif msc_ver == '1500':
76:             # VS2008 / MSVC 9.0
77:             return ['msvcr90']
78:         else:
79:             raise ValueError("Unknown MS Compiler version %s " % msc_ver)
80: 
81: 
82: class CygwinCCompiler (UnixCCompiler):
83: 
84:     compiler_type = 'cygwin'
85:     obj_extension = ".o"
86:     static_lib_extension = ".a"
87:     shared_lib_extension = ".dll"
88:     static_lib_format = "lib%s%s"
89:     shared_lib_format = "%s%s"
90:     exe_extension = ".exe"
91: 
92:     def __init__ (self, verbose=0, dry_run=0, force=0):
93: 
94:         UnixCCompiler.__init__ (self, verbose, dry_run, force)
95: 
96:         (status, details) = check_config_h()
97:         self.debug_print("Python's GCC status: %s (details: %s)" %
98:                          (status, details))
99:         if status is not CONFIG_H_OK:
100:             self.warn(
101:                 "Python's pyconfig.h doesn't seem to support your compiler. "
102:                 "Reason: %s. "
103:                 "Compiling may fail because of undefined preprocessor macros."
104:                 % details)
105: 
106:         self.gcc_version, self.ld_version, self.dllwrap_version = \
107:             get_versions()
108:         self.debug_print(self.compiler_type + ": gcc %s, ld %s, dllwrap %s\n" %
109:                          (self.gcc_version,
110:                           self.ld_version,
111:                           self.dllwrap_version) )
112: 
113:         # ld_version >= "2.10.90" and < "2.13" should also be able to use
114:         # gcc -mdll instead of dllwrap
115:         # Older dllwraps had own version numbers, newer ones use the
116:         # same as the rest of binutils ( also ld )
117:         # dllwrap 2.10.90 is buggy
118:         if self.ld_version >= "2.10.90":
119:             self.linker_dll = "gcc"
120:         else:
121:             self.linker_dll = "dllwrap"
122: 
123:         # ld_version >= "2.13" support -shared so use it instead of
124:         # -mdll -static
125:         if self.ld_version >= "2.13":
126:             shared_option = "-shared"
127:         else:
128:             shared_option = "-mdll -static"
129: 
130:         # Hard-code GCC because that's what this is all about.
131:         # XXX optimization, warnings etc. should be customizable.
132:         self.set_executables(compiler='gcc -mcygwin -O -Wall',
133:                              compiler_so='gcc -mcygwin -mdll -O -Wall',
134:                              compiler_cxx='g++ -mcygwin -O -Wall',
135:                              linker_exe='gcc -mcygwin',
136:                              linker_so=('%s -mcygwin %s' %
137:                                         (self.linker_dll, shared_option)))
138: 
139:         # cygwin and mingw32 need different sets of libraries
140:         if self.gcc_version == "2.91.57":
141:             # cygwin shouldn't need msvcrt, but without the dlls will crash
142:             # (gcc version 2.91.57) -- perhaps something about initialization
143:             self.dll_libraries=["msvcrt"]
144:             self.warn(
145:                 "Consider upgrading to a newer version of gcc")
146:         else:
147:             # Include the appropriate MSVC runtime library if Python was built
148:             # with MSVC 7.0 or later.
149:             self.dll_libraries = get_msvcr()
150: 
151:     # __init__ ()
152: 
153: 
154:     def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
155:         if ext == '.rc' or ext == '.res':
156:             # gcc needs '.res' and '.rc' compiled to object files !!!
157:             try:
158:                 self.spawn(["windres", "-i", src, "-o", obj])
159:             except DistutilsExecError, msg:
160:                 raise CompileError, msg
161:         else: # for other files use the C-compiler
162:             try:
163:                 self.spawn(self.compiler_so + cc_args + [src, '-o', obj] +
164:                            extra_postargs)
165:             except DistutilsExecError, msg:
166:                 raise CompileError, msg
167: 
168:     def link (self,
169:               target_desc,
170:               objects,
171:               output_filename,
172:               output_dir=None,
173:               libraries=None,
174:               library_dirs=None,
175:               runtime_library_dirs=None,
176:               export_symbols=None,
177:               debug=0,
178:               extra_preargs=None,
179:               extra_postargs=None,
180:               build_temp=None,
181:               target_lang=None):
182: 
183:         # use separate copies, so we can modify the lists
184:         extra_preargs = copy.copy(extra_preargs or [])
185:         libraries = copy.copy(libraries or [])
186:         objects = copy.copy(objects or [])
187: 
188:         # Additional libraries
189:         libraries.extend(self.dll_libraries)
190: 
191:         # handle export symbols by creating a def-file
192:         # with executables this only works with gcc/ld as linker
193:         if ((export_symbols is not None) and
194:             (target_desc != self.EXECUTABLE or self.linker_dll == "gcc")):
195:             # (The linker doesn't do anything if output is up-to-date.
196:             # So it would probably better to check if we really need this,
197:             # but for this we had to insert some unchanged parts of
198:             # UnixCCompiler, and this is not what we want.)
199: 
200:             # we want to put some files in the same directory as the
201:             # object files are, build_temp doesn't help much
202:             # where are the object files
203:             temp_dir = os.path.dirname(objects[0])
204:             # name of dll to give the helper files the same base name
205:             (dll_name, dll_extension) = os.path.splitext(
206:                 os.path.basename(output_filename))
207: 
208:             # generate the filenames for these files
209:             def_file = os.path.join(temp_dir, dll_name + ".def")
210:             lib_file = os.path.join(temp_dir, 'lib' + dll_name + ".a")
211: 
212:             # Generate .def file
213:             contents = [
214:                 "LIBRARY %s" % os.path.basename(output_filename),
215:                 "EXPORTS"]
216:             for sym in export_symbols:
217:                 contents.append(sym)
218:             self.execute(write_file, (def_file, contents),
219:                          "writing %s" % def_file)
220: 
221:             # next add options for def-file and to creating import libraries
222: 
223:             # dllwrap uses different options than gcc/ld
224:             if self.linker_dll == "dllwrap":
225:                 extra_preargs.extend(["--output-lib", lib_file])
226:                 # for dllwrap we have to use a special option
227:                 extra_preargs.extend(["--def", def_file])
228:             # we use gcc/ld here and can be sure ld is >= 2.9.10
229:             else:
230:                 # doesn't work: bfd_close build\...\libfoo.a: Invalid operation
231:                 #extra_preargs.extend(["-Wl,--out-implib,%s" % lib_file])
232:                 # for gcc/ld the def-file is specified as any object files
233:                 objects.append(def_file)
234: 
235:         #end: if ((export_symbols is not None) and
236:         #        (target_desc != self.EXECUTABLE or self.linker_dll == "gcc")):
237: 
238:         # who wants symbols and a many times larger output file
239:         # should explicitly switch the debug mode on
240:         # otherwise we let dllwrap/ld strip the output file
241:         # (On my machine: 10KB < stripped_file < ??100KB
242:         #   unstripped_file = stripped_file + XXX KB
243:         #  ( XXX=254 for a typical python extension))
244:         if not debug:
245:             extra_preargs.append("-s")
246: 
247:         UnixCCompiler.link(self,
248:                            target_desc,
249:                            objects,
250:                            output_filename,
251:                            output_dir,
252:                            libraries,
253:                            library_dirs,
254:                            runtime_library_dirs,
255:                            None, # export_symbols, we do this in our def-file
256:                            debug,
257:                            extra_preargs,
258:                            extra_postargs,
259:                            build_temp,
260:                            target_lang)
261: 
262:     # link ()
263: 
264:     # -- Miscellaneous methods -----------------------------------------
265: 
266:     # overwrite the one from CCompiler to support rc and res-files
267:     def object_filenames (self,
268:                           source_filenames,
269:                           strip_dir=0,
270:                           output_dir=''):
271:         if output_dir is None: output_dir = ''
272:         obj_names = []
273:         for src_name in source_filenames:
274:             # use normcase to make sure '.rc' is really '.rc' and not '.RC'
275:             (base, ext) = os.path.splitext (os.path.normcase(src_name))
276:             if ext not in (self.src_extensions + ['.rc','.res']):
277:                 raise UnknownFileError, \
278:                       "unknown file type '%s' (from '%s')" % \
279:                       (ext, src_name)
280:             if strip_dir:
281:                 base = os.path.basename (base)
282:             if ext == '.res' or ext == '.rc':
283:                 # these need to be compiled to object files
284:                 obj_names.append (os.path.join (output_dir,
285:                                             base + ext + self.obj_extension))
286:             else:
287:                 obj_names.append (os.path.join (output_dir,
288:                                             base + self.obj_extension))
289:         return obj_names
290: 
291:     # object_filenames ()
292: 
293: # class CygwinCCompiler
294: 
295: 
296: # the same as cygwin plus some additional parameters
297: class Mingw32CCompiler (CygwinCCompiler):
298: 
299:     compiler_type = 'mingw32'
300: 
301:     def __init__ (self,
302:                   verbose=0,
303:                   dry_run=0,
304:                   force=0):
305: 
306:         CygwinCCompiler.__init__ (self, verbose, dry_run, force)
307: 
308:         # ld_version >= "2.13" support -shared so use it instead of
309:         # -mdll -static
310:         if self.ld_version >= "2.13":
311:             shared_option = "-shared"
312:         else:
313:             shared_option = "-mdll -static"
314: 
315:         # A real mingw32 doesn't need to specify a different entry point,
316:         # but cygwin 2.91.57 in no-cygwin-mode needs it.
317:         if self.gcc_version <= "2.91.57":
318:             entry_point = '--entry _DllMain@12'
319:         else:
320:             entry_point = ''
321: 
322:         if self.gcc_version < '4' or is_cygwingcc():
323:             no_cygwin = ' -mno-cygwin'
324:         else:
325:             no_cygwin = ''
326: 
327:         self.set_executables(compiler='gcc%s -O -Wall' % no_cygwin,
328:                              compiler_so='gcc%s -mdll -O -Wall' % no_cygwin,
329:                              compiler_cxx='g++%s -O -Wall' % no_cygwin,
330:                              linker_exe='gcc%s' % no_cygwin,
331:                              linker_so='%s%s %s %s'
332:                                     % (self.linker_dll, no_cygwin,
333:                                        shared_option, entry_point))
334:         # Maybe we should also append -mthreads, but then the finished
335:         # dlls need another dll (mingwm10.dll see Mingw32 docs)
336:         # (-mthreads: Support thread-safe exception handling on `Mingw32')
337: 
338:         # no additional libraries needed
339:         self.dll_libraries=[]
340: 
341:         # Include the appropriate MSVC runtime library if Python was built
342:         # with MSVC 7.0 or later.
343:         self.dll_libraries = get_msvcr()
344: 
345:     # __init__ ()
346: 
347: # class Mingw32CCompiler
348: 
349: # Because these compilers aren't configured in Python's pyconfig.h file by
350: # default, we should at least warn the user if he is using an unmodified
351: # version.
352: 
353: CONFIG_H_OK = "ok"
354: CONFIG_H_NOTOK = "not ok"
355: CONFIG_H_UNCERTAIN = "uncertain"
356: 
357: def check_config_h():
358: 
359:     '''Check if the current Python installation (specifically, pyconfig.h)
360:     appears amenable to building extensions with GCC.  Returns a tuple
361:     (status, details), where 'status' is one of the following constants:
362:       CONFIG_H_OK
363:         all is well, go ahead and compile
364:       CONFIG_H_NOTOK
365:         doesn't look good
366:       CONFIG_H_UNCERTAIN
367:         not sure -- unable to read pyconfig.h
368:     'details' is a human-readable string explaining the situation.
369: 
370:     Note there are two ways to conclude "OK": either 'sys.version' contains
371:     the string "GCC" (implying that this Python was built with GCC), or the
372:     installed "pyconfig.h" contains the string "__GNUC__".
373:     '''
374: 
375:     # XXX since this function also checks sys.version, it's not strictly a
376:     # "pyconfig.h" check -- should probably be renamed...
377: 
378:     from distutils import sysconfig
379:     import string
380:     # if sys.version contains GCC then python was compiled with
381:     # GCC, and the pyconfig.h file should be OK
382:     if string.find(sys.version,"GCC") >= 0:
383:         return (CONFIG_H_OK, "sys.version mentions 'GCC'")
384: 
385:     fn = sysconfig.get_config_h_filename()
386:     try:
387:         # It would probably better to read single lines to search.
388:         # But we do this only once, and it is fast enough
389:         f = open(fn)
390:         try:
391:             s = f.read()
392:         finally:
393:             f.close()
394: 
395:     except IOError, exc:
396:         # if we can't read this file, we cannot say it is wrong
397:         # the compiler will complain later about this file as missing
398:         return (CONFIG_H_UNCERTAIN,
399:                 "couldn't read '%s': %s" % (fn, exc.strerror))
400: 
401:     else:
402:         # "pyconfig.h" contains an "#ifdef __GNUC__" or something similar
403:         if string.find(s,"__GNUC__") >= 0:
404:             return (CONFIG_H_OK, "'%s' mentions '__GNUC__'" % fn)
405:         else:
406:             return (CONFIG_H_NOTOK, "'%s' does not mention '__GNUC__'" % fn)
407: 
408: 
409: 
410: def get_versions():
411:     ''' Try to find out the versions of gcc, ld and dllwrap.
412:         If not possible it returns None for it.
413:     '''
414:     from distutils.version import LooseVersion
415:     from distutils.spawn import find_executable
416:     import re
417: 
418:     gcc_exe = find_executable('gcc')
419:     if gcc_exe:
420:         out = os.popen(gcc_exe + ' -dumpversion','r')
421:         out_string = out.read()
422:         out.close()
423:         result = re.search('(\d+\.\d+(\.\d+)*)',out_string)
424:         if result:
425:             gcc_version = LooseVersion(result.group(1))
426:         else:
427:             gcc_version = None
428:     else:
429:         gcc_version = None
430:     ld_exe = find_executable('ld')
431:     if ld_exe:
432:         out = os.popen(ld_exe + ' -v','r')
433:         out_string = out.read()
434:         out.close()
435:         result = re.search('(\d+\.\d+(\.\d+)*)',out_string)
436:         if result:
437:             ld_version = LooseVersion(result.group(1))
438:         else:
439:             ld_version = None
440:     else:
441:         ld_version = None
442:     dllwrap_exe = find_executable('dllwrap')
443:     if dllwrap_exe:
444:         out = os.popen(dllwrap_exe + ' --version','r')
445:         out_string = out.read()
446:         out.close()
447:         result = re.search(' (\d+\.\d+(\.\d+)*)',out_string)
448:         if result:
449:             dllwrap_version = LooseVersion(result.group(1))
450:         else:
451:             dllwrap_version = None
452:     else:
453:         dllwrap_version = None
454:     return (gcc_version, ld_version, dllwrap_version)
455: 
456: def is_cygwingcc():
457:     '''Try to determine if the gcc that would be used is from cygwin.'''
458:     out = os.popen('gcc -dumpmachine', 'r')
459:     out_string = out.read()
460:     out.close()
461:     # out_string is the target triplet cpu-vendor-os
462:     # Cygwin's gcc sets the os to 'cygwin'
463:     return out_string.strip().endswith('cygwin')
464: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_306676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', 'distutils.cygwinccompiler\n\nProvides the CygwinCCompiler class, a subclass of UnixCCompiler that\nhandles the Cygwin port of the GNU C compiler to Windows.  It also contains\nthe Mingw32CCompiler class which handles the mingw32 port of GCC (same as\ncygwin in no-cygwin mode).\n')

# Assigning a Str to a Name (line 50):

# Assigning a Str to a Name (line 50):
str_306677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), '__revision__', str_306677)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 52, 0))

# Multiple import statement. import os (1/3) (line 52)
import os

import_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'os', os, module_type_store)
# Multiple import statement. import sys (2/3) (line 52)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'sys', sys, module_type_store)
# Multiple import statement. import copy (3/3) (line 52)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 53, 0))

# 'from distutils.ccompiler import gen_preprocess_options, gen_lib_options' statement (line 53)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_306678 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 53, 0), 'distutils.ccompiler')

if (type(import_306678) is not StypyTypeError):

    if (import_306678 != 'pyd_module'):
        __import__(import_306678)
        sys_modules_306679 = sys.modules[import_306678]
        import_from_module(stypy.reporting.localization.Localization(__file__, 53, 0), 'distutils.ccompiler', sys_modules_306679.module_type_store, module_type_store, ['gen_preprocess_options', 'gen_lib_options'])
        nest_module(stypy.reporting.localization.Localization(__file__, 53, 0), __file__, sys_modules_306679, sys_modules_306679.module_type_store, module_type_store)
    else:
        from distutils.ccompiler import gen_preprocess_options, gen_lib_options

        import_from_module(stypy.reporting.localization.Localization(__file__, 53, 0), 'distutils.ccompiler', None, module_type_store, ['gen_preprocess_options', 'gen_lib_options'], [gen_preprocess_options, gen_lib_options])

else:
    # Assigning a type to the variable 'distutils.ccompiler' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'distutils.ccompiler', import_306678)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 54, 0))

# 'from distutils.unixccompiler import UnixCCompiler' statement (line 54)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_306680 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 54, 0), 'distutils.unixccompiler')

if (type(import_306680) is not StypyTypeError):

    if (import_306680 != 'pyd_module'):
        __import__(import_306680)
        sys_modules_306681 = sys.modules[import_306680]
        import_from_module(stypy.reporting.localization.Localization(__file__, 54, 0), 'distutils.unixccompiler', sys_modules_306681.module_type_store, module_type_store, ['UnixCCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 54, 0), __file__, sys_modules_306681, sys_modules_306681.module_type_store, module_type_store)
    else:
        from distutils.unixccompiler import UnixCCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 54, 0), 'distutils.unixccompiler', None, module_type_store, ['UnixCCompiler'], [UnixCCompiler])

else:
    # Assigning a type to the variable 'distutils.unixccompiler' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'distutils.unixccompiler', import_306680)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 55, 0))

# 'from distutils.file_util import write_file' statement (line 55)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_306682 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 55, 0), 'distutils.file_util')

if (type(import_306682) is not StypyTypeError):

    if (import_306682 != 'pyd_module'):
        __import__(import_306682)
        sys_modules_306683 = sys.modules[import_306682]
        import_from_module(stypy.reporting.localization.Localization(__file__, 55, 0), 'distutils.file_util', sys_modules_306683.module_type_store, module_type_store, ['write_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 55, 0), __file__, sys_modules_306683, sys_modules_306683.module_type_store, module_type_store)
    else:
        from distutils.file_util import write_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 55, 0), 'distutils.file_util', None, module_type_store, ['write_file'], [write_file])

else:
    # Assigning a type to the variable 'distutils.file_util' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'distutils.file_util', import_306682)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 56, 0))

# 'from distutils.errors import DistutilsExecError, CompileError, UnknownFileError' statement (line 56)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_306684 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 56, 0), 'distutils.errors')

if (type(import_306684) is not StypyTypeError):

    if (import_306684 != 'pyd_module'):
        __import__(import_306684)
        sys_modules_306685 = sys.modules[import_306684]
        import_from_module(stypy.reporting.localization.Localization(__file__, 56, 0), 'distutils.errors', sys_modules_306685.module_type_store, module_type_store, ['DistutilsExecError', 'CompileError', 'UnknownFileError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 56, 0), __file__, sys_modules_306685, sys_modules_306685.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsExecError, CompileError, UnknownFileError

        import_from_module(stypy.reporting.localization.Localization(__file__, 56, 0), 'distutils.errors', None, module_type_store, ['DistutilsExecError', 'CompileError', 'UnknownFileError'], [DistutilsExecError, CompileError, UnknownFileError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'distutils.errors', import_306684)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 57, 0))

# 'from distutils import log' statement (line 57)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 57, 0), 'distutils', None, module_type_store, ['log'], [log])


@norecursion
def get_msvcr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_msvcr'
    module_type_store = module_type_store.open_function_context('get_msvcr', 59, 0, False)
    
    # Passed parameters checking function
    get_msvcr.stypy_localization = localization
    get_msvcr.stypy_type_of_self = None
    get_msvcr.stypy_type_store = module_type_store
    get_msvcr.stypy_function_name = 'get_msvcr'
    get_msvcr.stypy_param_names_list = []
    get_msvcr.stypy_varargs_param_name = None
    get_msvcr.stypy_kwargs_param_name = None
    get_msvcr.stypy_call_defaults = defaults
    get_msvcr.stypy_call_varargs = varargs
    get_msvcr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_msvcr', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_msvcr', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_msvcr(...)' code ##################

    str_306686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'str', 'Include the appropriate MSVC runtime library if Python was built\n    with MSVC 7.0 or later.\n    ')
    
    # Assigning a Call to a Name (line 63):
    
    # Assigning a Call to a Name (line 63):
    
    # Call to find(...): (line 63)
    # Processing the call arguments (line 63)
    str_306690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 31), 'str', 'MSC v.')
    # Processing the call keyword arguments (line 63)
    kwargs_306691 = {}
    # Getting the type of 'sys' (line 63)
    sys_306687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 14), 'sys', False)
    # Obtaining the member 'version' of a type (line 63)
    version_306688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 14), sys_306687, 'version')
    # Obtaining the member 'find' of a type (line 63)
    find_306689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 14), version_306688, 'find')
    # Calling find(args, kwargs) (line 63)
    find_call_result_306692 = invoke(stypy.reporting.localization.Localization(__file__, 63, 14), find_306689, *[str_306690], **kwargs_306691)
    
    # Assigning a type to the variable 'msc_pos' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'msc_pos', find_call_result_306692)
    
    
    # Getting the type of 'msc_pos' (line 64)
    msc_pos_306693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 7), 'msc_pos')
    int_306694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 18), 'int')
    # Applying the binary operator '!=' (line 64)
    result_ne_306695 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 7), '!=', msc_pos_306693, int_306694)
    
    # Testing the type of an if condition (line 64)
    if_condition_306696 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 4), result_ne_306695)
    # Assigning a type to the variable 'if_condition_306696' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'if_condition_306696', if_condition_306696)
    # SSA begins for if statement (line 64)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 65):
    
    # Assigning a Subscript to a Name (line 65):
    
    # Obtaining the type of the subscript
    # Getting the type of 'msc_pos' (line 65)
    msc_pos_306697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'msc_pos')
    int_306698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 38), 'int')
    # Applying the binary operator '+' (line 65)
    result_add_306699 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 30), '+', msc_pos_306697, int_306698)
    
    # Getting the type of 'msc_pos' (line 65)
    msc_pos_306700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 40), 'msc_pos')
    int_306701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 48), 'int')
    # Applying the binary operator '+' (line 65)
    result_add_306702 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 40), '+', msc_pos_306700, int_306701)
    
    slice_306703 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 65, 18), result_add_306699, result_add_306702, None)
    # Getting the type of 'sys' (line 65)
    sys_306704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 18), 'sys')
    # Obtaining the member 'version' of a type (line 65)
    version_306705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 18), sys_306704, 'version')
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___306706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 18), version_306705, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_306707 = invoke(stypy.reporting.localization.Localization(__file__, 65, 18), getitem___306706, slice_306703)
    
    # Assigning a type to the variable 'msc_ver' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'msc_ver', subscript_call_result_306707)
    
    
    # Getting the type of 'msc_ver' (line 66)
    msc_ver_306708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'msc_ver')
    str_306709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 22), 'str', '1300')
    # Applying the binary operator '==' (line 66)
    result_eq_306710 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 11), '==', msc_ver_306708, str_306709)
    
    # Testing the type of an if condition (line 66)
    if_condition_306711 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 8), result_eq_306710)
    # Assigning a type to the variable 'if_condition_306711' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'if_condition_306711', if_condition_306711)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_306712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    str_306713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'str', 'msvcr70')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 19), list_306712, str_306713)
    
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'stypy_return_type', list_306712)
    # SSA branch for the else part of an if statement (line 66)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'msc_ver' (line 69)
    msc_ver_306714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'msc_ver')
    str_306715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 24), 'str', '1310')
    # Applying the binary operator '==' (line 69)
    result_eq_306716 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 13), '==', msc_ver_306714, str_306715)
    
    # Testing the type of an if condition (line 69)
    if_condition_306717 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 13), result_eq_306716)
    # Assigning a type to the variable 'if_condition_306717' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'if_condition_306717', if_condition_306717)
    # SSA begins for if statement (line 69)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'list' (line 71)
    list_306718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 71)
    # Adding element type (line 71)
    str_306719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 20), 'str', 'msvcr71')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 19), list_306718, str_306719)
    
    # Assigning a type to the variable 'stypy_return_type' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'stypy_return_type', list_306718)
    # SSA branch for the else part of an if statement (line 69)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'msc_ver' (line 72)
    msc_ver_306720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'msc_ver')
    str_306721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 24), 'str', '1400')
    # Applying the binary operator '==' (line 72)
    result_eq_306722 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 13), '==', msc_ver_306720, str_306721)
    
    # Testing the type of an if condition (line 72)
    if_condition_306723 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 13), result_eq_306722)
    # Assigning a type to the variable 'if_condition_306723' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'if_condition_306723', if_condition_306723)
    # SSA begins for if statement (line 72)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'list' (line 74)
    list_306724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 74)
    # Adding element type (line 74)
    str_306725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 20), 'str', 'msvcr80')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 19), list_306724, str_306725)
    
    # Assigning a type to the variable 'stypy_return_type' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'stypy_return_type', list_306724)
    # SSA branch for the else part of an if statement (line 72)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'msc_ver' (line 75)
    msc_ver_306726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 13), 'msc_ver')
    str_306727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 24), 'str', '1500')
    # Applying the binary operator '==' (line 75)
    result_eq_306728 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 13), '==', msc_ver_306726, str_306727)
    
    # Testing the type of an if condition (line 75)
    if_condition_306729 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 13), result_eq_306728)
    # Assigning a type to the variable 'if_condition_306729' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 13), 'if_condition_306729', if_condition_306729)
    # SSA begins for if statement (line 75)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'list' (line 77)
    list_306730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 77)
    # Adding element type (line 77)
    str_306731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 20), 'str', 'msvcr90')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 19), list_306730, str_306731)
    
    # Assigning a type to the variable 'stypy_return_type' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'stypy_return_type', list_306730)
    # SSA branch for the else part of an if statement (line 75)
    module_type_store.open_ssa_branch('else')
    
    # Call to ValueError(...): (line 79)
    # Processing the call arguments (line 79)
    str_306733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 29), 'str', 'Unknown MS Compiler version %s ')
    # Getting the type of 'msc_ver' (line 79)
    msc_ver_306734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 65), 'msc_ver', False)
    # Applying the binary operator '%' (line 79)
    result_mod_306735 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 29), '%', str_306733, msc_ver_306734)
    
    # Processing the call keyword arguments (line 79)
    kwargs_306736 = {}
    # Getting the type of 'ValueError' (line 79)
    ValueError_306732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 79)
    ValueError_call_result_306737 = invoke(stypy.reporting.localization.Localization(__file__, 79, 18), ValueError_306732, *[result_mod_306735], **kwargs_306736)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 79, 12), ValueError_call_result_306737, 'raise parameter', BaseException)
    # SSA join for if statement (line 75)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 72)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 69)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 64)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_msvcr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_msvcr' in the type store
    # Getting the type of 'stypy_return_type' (line 59)
    stypy_return_type_306738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_306738)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_msvcr'
    return stypy_return_type_306738

# Assigning a type to the variable 'get_msvcr' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'get_msvcr', get_msvcr)
# Declaration of the 'CygwinCCompiler' class
# Getting the type of 'UnixCCompiler' (line 82)
UnixCCompiler_306739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 23), 'UnixCCompiler')

class CygwinCCompiler(UnixCCompiler_306739, ):
    
    # Assigning a Str to a Name (line 84):
    
    # Assigning a Str to a Name (line 85):
    
    # Assigning a Str to a Name (line 86):
    
    # Assigning a Str to a Name (line 87):
    
    # Assigning a Str to a Name (line 88):
    
    # Assigning a Str to a Name (line 89):
    
    # Assigning a Str to a Name (line 90):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_306740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 32), 'int')
        int_306741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 43), 'int')
        int_306742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 52), 'int')
        defaults = [int_306740, int_306741, int_306742]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CygwinCCompiler.__init__', ['verbose', 'dry_run', 'force'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'self' (line 94)
        self_306745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 32), 'self', False)
        # Getting the type of 'verbose' (line 94)
        verbose_306746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 38), 'verbose', False)
        # Getting the type of 'dry_run' (line 94)
        dry_run_306747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 47), 'dry_run', False)
        # Getting the type of 'force' (line 94)
        force_306748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 56), 'force', False)
        # Processing the call keyword arguments (line 94)
        kwargs_306749 = {}
        # Getting the type of 'UnixCCompiler' (line 94)
        UnixCCompiler_306743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'UnixCCompiler', False)
        # Obtaining the member '__init__' of a type (line 94)
        init___306744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), UnixCCompiler_306743, '__init__')
        # Calling __init__(args, kwargs) (line 94)
        init___call_result_306750 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), init___306744, *[self_306745, verbose_306746, dry_run_306747, force_306748], **kwargs_306749)
        
        
        # Assigning a Call to a Tuple (line 96):
        
        # Assigning a Call to a Name:
        
        # Call to check_config_h(...): (line 96)
        # Processing the call keyword arguments (line 96)
        kwargs_306752 = {}
        # Getting the type of 'check_config_h' (line 96)
        check_config_h_306751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 28), 'check_config_h', False)
        # Calling check_config_h(args, kwargs) (line 96)
        check_config_h_call_result_306753 = invoke(stypy.reporting.localization.Localization(__file__, 96, 28), check_config_h_306751, *[], **kwargs_306752)
        
        # Assigning a type to the variable 'call_assignment_306663' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'call_assignment_306663', check_config_h_call_result_306753)
        
        # Assigning a Call to a Name (line 96):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_306756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'int')
        # Processing the call keyword arguments
        kwargs_306757 = {}
        # Getting the type of 'call_assignment_306663' (line 96)
        call_assignment_306663_306754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'call_assignment_306663', False)
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___306755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), call_assignment_306663_306754, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_306758 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___306755, *[int_306756], **kwargs_306757)
        
        # Assigning a type to the variable 'call_assignment_306664' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'call_assignment_306664', getitem___call_result_306758)
        
        # Assigning a Name to a Name (line 96):
        # Getting the type of 'call_assignment_306664' (line 96)
        call_assignment_306664_306759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'call_assignment_306664')
        # Assigning a type to the variable 'status' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 9), 'status', call_assignment_306664_306759)
        
        # Assigning a Call to a Name (line 96):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_306762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'int')
        # Processing the call keyword arguments
        kwargs_306763 = {}
        # Getting the type of 'call_assignment_306663' (line 96)
        call_assignment_306663_306760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'call_assignment_306663', False)
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___306761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), call_assignment_306663_306760, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_306764 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___306761, *[int_306762], **kwargs_306763)
        
        # Assigning a type to the variable 'call_assignment_306665' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'call_assignment_306665', getitem___call_result_306764)
        
        # Assigning a Name to a Name (line 96):
        # Getting the type of 'call_assignment_306665' (line 96)
        call_assignment_306665_306765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'call_assignment_306665')
        # Assigning a type to the variable 'details' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'details', call_assignment_306665_306765)
        
        # Call to debug_print(...): (line 97)
        # Processing the call arguments (line 97)
        str_306768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 25), 'str', "Python's GCC status: %s (details: %s)")
        
        # Obtaining an instance of the builtin type 'tuple' (line 98)
        tuple_306769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 98)
        # Adding element type (line 98)
        # Getting the type of 'status' (line 98)
        status_306770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 'status', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 26), tuple_306769, status_306770)
        # Adding element type (line 98)
        # Getting the type of 'details' (line 98)
        details_306771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 34), 'details', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 26), tuple_306769, details_306771)
        
        # Applying the binary operator '%' (line 97)
        result_mod_306772 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 25), '%', str_306768, tuple_306769)
        
        # Processing the call keyword arguments (line 97)
        kwargs_306773 = {}
        # Getting the type of 'self' (line 97)
        self_306766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 97)
        debug_print_306767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_306766, 'debug_print')
        # Calling debug_print(args, kwargs) (line 97)
        debug_print_call_result_306774 = invoke(stypy.reporting.localization.Localization(__file__, 97, 8), debug_print_306767, *[result_mod_306772], **kwargs_306773)
        
        
        
        # Getting the type of 'status' (line 99)
        status_306775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'status')
        # Getting the type of 'CONFIG_H_OK' (line 99)
        CONFIG_H_OK_306776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 25), 'CONFIG_H_OK')
        # Applying the binary operator 'isnot' (line 99)
        result_is_not_306777 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 11), 'isnot', status_306775, CONFIG_H_OK_306776)
        
        # Testing the type of an if condition (line 99)
        if_condition_306778 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 8), result_is_not_306777)
        # Assigning a type to the variable 'if_condition_306778' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'if_condition_306778', if_condition_306778)
        # SSA begins for if statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 100)
        # Processing the call arguments (line 100)
        str_306781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 16), 'str', "Python's pyconfig.h doesn't seem to support your compiler. Reason: %s. Compiling may fail because of undefined preprocessor macros.")
        # Getting the type of 'details' (line 104)
        details_306782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 18), 'details', False)
        # Applying the binary operator '%' (line 101)
        result_mod_306783 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 16), '%', str_306781, details_306782)
        
        # Processing the call keyword arguments (line 100)
        kwargs_306784 = {}
        # Getting the type of 'self' (line 100)
        self_306779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'self', False)
        # Obtaining the member 'warn' of a type (line 100)
        warn_306780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), self_306779, 'warn')
        # Calling warn(args, kwargs) (line 100)
        warn_call_result_306785 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), warn_306780, *[result_mod_306783], **kwargs_306784)
        
        # SSA join for if statement (line 99)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 106):
        
        # Assigning a Call to a Name:
        
        # Call to get_versions(...): (line 107)
        # Processing the call keyword arguments (line 107)
        kwargs_306787 = {}
        # Getting the type of 'get_versions' (line 107)
        get_versions_306786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'get_versions', False)
        # Calling get_versions(args, kwargs) (line 107)
        get_versions_call_result_306788 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), get_versions_306786, *[], **kwargs_306787)
        
        # Assigning a type to the variable 'call_assignment_306666' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_306666', get_versions_call_result_306788)
        
        # Assigning a Call to a Name (line 106):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_306791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
        # Processing the call keyword arguments
        kwargs_306792 = {}
        # Getting the type of 'call_assignment_306666' (line 106)
        call_assignment_306666_306789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_306666', False)
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___306790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), call_assignment_306666_306789, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_306793 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___306790, *[int_306791], **kwargs_306792)
        
        # Assigning a type to the variable 'call_assignment_306667' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_306667', getitem___call_result_306793)
        
        # Assigning a Name to a Attribute (line 106):
        # Getting the type of 'call_assignment_306667' (line 106)
        call_assignment_306667_306794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_306667')
        # Getting the type of 'self' (line 106)
        self_306795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'self')
        # Setting the type of the member 'gcc_version' of a type (line 106)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), self_306795, 'gcc_version', call_assignment_306667_306794)
        
        # Assigning a Call to a Name (line 106):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_306798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
        # Processing the call keyword arguments
        kwargs_306799 = {}
        # Getting the type of 'call_assignment_306666' (line 106)
        call_assignment_306666_306796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_306666', False)
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___306797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), call_assignment_306666_306796, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_306800 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___306797, *[int_306798], **kwargs_306799)
        
        # Assigning a type to the variable 'call_assignment_306668' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_306668', getitem___call_result_306800)
        
        # Assigning a Name to a Attribute (line 106):
        # Getting the type of 'call_assignment_306668' (line 106)
        call_assignment_306668_306801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_306668')
        # Getting the type of 'self' (line 106)
        self_306802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'self')
        # Setting the type of the member 'ld_version' of a type (line 106)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 26), self_306802, 'ld_version', call_assignment_306668_306801)
        
        # Assigning a Call to a Name (line 106):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_306805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'int')
        # Processing the call keyword arguments
        kwargs_306806 = {}
        # Getting the type of 'call_assignment_306666' (line 106)
        call_assignment_306666_306803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_306666', False)
        # Obtaining the member '__getitem__' of a type (line 106)
        getitem___306804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), call_assignment_306666_306803, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_306807 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___306804, *[int_306805], **kwargs_306806)
        
        # Assigning a type to the variable 'call_assignment_306669' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_306669', getitem___call_result_306807)
        
        # Assigning a Name to a Attribute (line 106):
        # Getting the type of 'call_assignment_306669' (line 106)
        call_assignment_306669_306808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'call_assignment_306669')
        # Getting the type of 'self' (line 106)
        self_306809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 43), 'self')
        # Setting the type of the member 'dllwrap_version' of a type (line 106)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 43), self_306809, 'dllwrap_version', call_assignment_306669_306808)
        
        # Call to debug_print(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'self' (line 108)
        self_306812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'self', False)
        # Obtaining the member 'compiler_type' of a type (line 108)
        compiler_type_306813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 25), self_306812, 'compiler_type')
        str_306814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 46), 'str', ': gcc %s, ld %s, dllwrap %s\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 109)
        tuple_306815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 109)
        # Adding element type (line 109)
        # Getting the type of 'self' (line 109)
        self_306816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 26), 'self', False)
        # Obtaining the member 'gcc_version' of a type (line 109)
        gcc_version_306817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 26), self_306816, 'gcc_version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 26), tuple_306815, gcc_version_306817)
        # Adding element type (line 109)
        # Getting the type of 'self' (line 110)
        self_306818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 26), 'self', False)
        # Obtaining the member 'ld_version' of a type (line 110)
        ld_version_306819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 26), self_306818, 'ld_version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 26), tuple_306815, ld_version_306819)
        # Adding element type (line 109)
        # Getting the type of 'self' (line 111)
        self_306820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'self', False)
        # Obtaining the member 'dllwrap_version' of a type (line 111)
        dllwrap_version_306821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 26), self_306820, 'dllwrap_version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 26), tuple_306815, dllwrap_version_306821)
        
        # Applying the binary operator '%' (line 108)
        result_mod_306822 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 46), '%', str_306814, tuple_306815)
        
        # Applying the binary operator '+' (line 108)
        result_add_306823 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 25), '+', compiler_type_306813, result_mod_306822)
        
        # Processing the call keyword arguments (line 108)
        kwargs_306824 = {}
        # Getting the type of 'self' (line 108)
        self_306810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 108)
        debug_print_306811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), self_306810, 'debug_print')
        # Calling debug_print(args, kwargs) (line 108)
        debug_print_call_result_306825 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), debug_print_306811, *[result_add_306823], **kwargs_306824)
        
        
        
        # Getting the type of 'self' (line 118)
        self_306826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 11), 'self')
        # Obtaining the member 'ld_version' of a type (line 118)
        ld_version_306827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 11), self_306826, 'ld_version')
        str_306828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 30), 'str', '2.10.90')
        # Applying the binary operator '>=' (line 118)
        result_ge_306829 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 11), '>=', ld_version_306827, str_306828)
        
        # Testing the type of an if condition (line 118)
        if_condition_306830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 8), result_ge_306829)
        # Assigning a type to the variable 'if_condition_306830' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'if_condition_306830', if_condition_306830)
        # SSA begins for if statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Attribute (line 119):
        
        # Assigning a Str to a Attribute (line 119):
        str_306831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 30), 'str', 'gcc')
        # Getting the type of 'self' (line 119)
        self_306832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'self')
        # Setting the type of the member 'linker_dll' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), self_306832, 'linker_dll', str_306831)
        # SSA branch for the else part of an if statement (line 118)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Attribute (line 121):
        
        # Assigning a Str to a Attribute (line 121):
        str_306833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 30), 'str', 'dllwrap')
        # Getting the type of 'self' (line 121)
        self_306834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'self')
        # Setting the type of the member 'linker_dll' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), self_306834, 'linker_dll', str_306833)
        # SSA join for if statement (line 118)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 125)
        self_306835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'self')
        # Obtaining the member 'ld_version' of a type (line 125)
        ld_version_306836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 11), self_306835, 'ld_version')
        str_306837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 30), 'str', '2.13')
        # Applying the binary operator '>=' (line 125)
        result_ge_306838 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 11), '>=', ld_version_306836, str_306837)
        
        # Testing the type of an if condition (line 125)
        if_condition_306839 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), result_ge_306838)
        # Assigning a type to the variable 'if_condition_306839' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'if_condition_306839', if_condition_306839)
        # SSA begins for if statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 126):
        
        # Assigning a Str to a Name (line 126):
        str_306840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 28), 'str', '-shared')
        # Assigning a type to the variable 'shared_option' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'shared_option', str_306840)
        # SSA branch for the else part of an if statement (line 125)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 128):
        
        # Assigning a Str to a Name (line 128):
        str_306841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 28), 'str', '-mdll -static')
        # Assigning a type to the variable 'shared_option' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'shared_option', str_306841)
        # SSA join for if statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_executables(...): (line 132)
        # Processing the call keyword arguments (line 132)
        str_306844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 38), 'str', 'gcc -mcygwin -O -Wall')
        keyword_306845 = str_306844
        str_306846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 41), 'str', 'gcc -mcygwin -mdll -O -Wall')
        keyword_306847 = str_306846
        str_306848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 42), 'str', 'g++ -mcygwin -O -Wall')
        keyword_306849 = str_306848
        str_306850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 40), 'str', 'gcc -mcygwin')
        keyword_306851 = str_306850
        str_306852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 40), 'str', '%s -mcygwin %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 137)
        tuple_306853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 137)
        # Adding element type (line 137)
        # Getting the type of 'self' (line 137)
        self_306854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 41), 'self', False)
        # Obtaining the member 'linker_dll' of a type (line 137)
        linker_dll_306855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 41), self_306854, 'linker_dll')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 41), tuple_306853, linker_dll_306855)
        # Adding element type (line 137)
        # Getting the type of 'shared_option' (line 137)
        shared_option_306856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 58), 'shared_option', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 41), tuple_306853, shared_option_306856)
        
        # Applying the binary operator '%' (line 136)
        result_mod_306857 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 40), '%', str_306852, tuple_306853)
        
        keyword_306858 = result_mod_306857
        kwargs_306859 = {'compiler_cxx': keyword_306849, 'linker_exe': keyword_306851, 'compiler_so': keyword_306847, 'linker_so': keyword_306858, 'compiler': keyword_306845}
        # Getting the type of 'self' (line 132)
        self_306842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'self', False)
        # Obtaining the member 'set_executables' of a type (line 132)
        set_executables_306843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), self_306842, 'set_executables')
        # Calling set_executables(args, kwargs) (line 132)
        set_executables_call_result_306860 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), set_executables_306843, *[], **kwargs_306859)
        
        
        
        # Getting the type of 'self' (line 140)
        self_306861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'self')
        # Obtaining the member 'gcc_version' of a type (line 140)
        gcc_version_306862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 11), self_306861, 'gcc_version')
        str_306863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 31), 'str', '2.91.57')
        # Applying the binary operator '==' (line 140)
        result_eq_306864 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 11), '==', gcc_version_306862, str_306863)
        
        # Testing the type of an if condition (line 140)
        if_condition_306865 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 8), result_eq_306864)
        # Assigning a type to the variable 'if_condition_306865' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'if_condition_306865', if_condition_306865)
        # SSA begins for if statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Attribute (line 143):
        
        # Assigning a List to a Attribute (line 143):
        
        # Obtaining an instance of the builtin type 'list' (line 143)
        list_306866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 143)
        # Adding element type (line 143)
        str_306867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 32), 'str', 'msvcrt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 31), list_306866, str_306867)
        
        # Getting the type of 'self' (line 143)
        self_306868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'self')
        # Setting the type of the member 'dll_libraries' of a type (line 143)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 12), self_306868, 'dll_libraries', list_306866)
        
        # Call to warn(...): (line 144)
        # Processing the call arguments (line 144)
        str_306871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 16), 'str', 'Consider upgrading to a newer version of gcc')
        # Processing the call keyword arguments (line 144)
        kwargs_306872 = {}
        # Getting the type of 'self' (line 144)
        self_306869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'self', False)
        # Obtaining the member 'warn' of a type (line 144)
        warn_306870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), self_306869, 'warn')
        # Calling warn(args, kwargs) (line 144)
        warn_call_result_306873 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), warn_306870, *[str_306871], **kwargs_306872)
        
        # SSA branch for the else part of an if statement (line 140)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 149):
        
        # Assigning a Call to a Attribute (line 149):
        
        # Call to get_msvcr(...): (line 149)
        # Processing the call keyword arguments (line 149)
        kwargs_306875 = {}
        # Getting the type of 'get_msvcr' (line 149)
        get_msvcr_306874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 33), 'get_msvcr', False)
        # Calling get_msvcr(args, kwargs) (line 149)
        get_msvcr_call_result_306876 = invoke(stypy.reporting.localization.Localization(__file__, 149, 33), get_msvcr_306874, *[], **kwargs_306875)
        
        # Getting the type of 'self' (line 149)
        self_306877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'self')
        # Setting the type of the member 'dll_libraries' of a type (line 149)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), self_306877, 'dll_libraries', get_msvcr_call_result_306876)
        # SSA join for if statement (line 140)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_compile'
        module_type_store = module_type_store.open_function_context('_compile', 154, 4, False)
        # Assigning a type to the variable 'self' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CygwinCCompiler._compile.__dict__.__setitem__('stypy_localization', localization)
        CygwinCCompiler._compile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CygwinCCompiler._compile.__dict__.__setitem__('stypy_type_store', module_type_store)
        CygwinCCompiler._compile.__dict__.__setitem__('stypy_function_name', 'CygwinCCompiler._compile')
        CygwinCCompiler._compile.__dict__.__setitem__('stypy_param_names_list', ['obj', 'src', 'ext', 'cc_args', 'extra_postargs', 'pp_opts'])
        CygwinCCompiler._compile.__dict__.__setitem__('stypy_varargs_param_name', None)
        CygwinCCompiler._compile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CygwinCCompiler._compile.__dict__.__setitem__('stypy_call_defaults', defaults)
        CygwinCCompiler._compile.__dict__.__setitem__('stypy_call_varargs', varargs)
        CygwinCCompiler._compile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CygwinCCompiler._compile.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CygwinCCompiler._compile', ['obj', 'src', 'ext', 'cc_args', 'extra_postargs', 'pp_opts'], None, None, defaults, varargs, kwargs)

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

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'ext' (line 155)
        ext_306878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'ext')
        str_306879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 18), 'str', '.rc')
        # Applying the binary operator '==' (line 155)
        result_eq_306880 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 11), '==', ext_306878, str_306879)
        
        
        # Getting the type of 'ext' (line 155)
        ext_306881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 27), 'ext')
        str_306882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 34), 'str', '.res')
        # Applying the binary operator '==' (line 155)
        result_eq_306883 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 27), '==', ext_306881, str_306882)
        
        # Applying the binary operator 'or' (line 155)
        result_or_keyword_306884 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 11), 'or', result_eq_306880, result_eq_306883)
        
        # Testing the type of an if condition (line 155)
        if_condition_306885 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 8), result_or_keyword_306884)
        # Assigning a type to the variable 'if_condition_306885' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'if_condition_306885', if_condition_306885)
        # SSA begins for if statement (line 155)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 158)
        # Processing the call arguments (line 158)
        
        # Obtaining an instance of the builtin type 'list' (line 158)
        list_306888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 158)
        # Adding element type (line 158)
        str_306889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 28), 'str', 'windres')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 27), list_306888, str_306889)
        # Adding element type (line 158)
        str_306890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 39), 'str', '-i')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 27), list_306888, str_306890)
        # Adding element type (line 158)
        # Getting the type of 'src' (line 158)
        src_306891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 45), 'src', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 27), list_306888, src_306891)
        # Adding element type (line 158)
        str_306892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 50), 'str', '-o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 27), list_306888, str_306892)
        # Adding element type (line 158)
        # Getting the type of 'obj' (line 158)
        obj_306893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 56), 'obj', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 27), list_306888, obj_306893)
        
        # Processing the call keyword arguments (line 158)
        kwargs_306894 = {}
        # Getting the type of 'self' (line 158)
        self_306886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 158)
        spawn_306887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 16), self_306886, 'spawn')
        # Calling spawn(args, kwargs) (line 158)
        spawn_call_result_306895 = invoke(stypy.reporting.localization.Localization(__file__, 158, 16), spawn_306887, *[list_306888], **kwargs_306894)
        
        # SSA branch for the except part of a try statement (line 157)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 157)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 159)
        DistutilsExecError_306896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'msg', DistutilsExecError_306896)
        # Getting the type of 'CompileError' (line 160)
        CompileError_306897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 22), 'CompileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 160, 16), CompileError_306897, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 157)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 155)
        module_type_store.open_ssa_branch('else')
        
        
        # SSA begins for try-except statement (line 162)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'self' (line 163)
        self_306900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 27), 'self', False)
        # Obtaining the member 'compiler_so' of a type (line 163)
        compiler_so_306901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 27), self_306900, 'compiler_so')
        # Getting the type of 'cc_args' (line 163)
        cc_args_306902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 46), 'cc_args', False)
        # Applying the binary operator '+' (line 163)
        result_add_306903 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 27), '+', compiler_so_306901, cc_args_306902)
        
        
        # Obtaining an instance of the builtin type 'list' (line 163)
        list_306904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 163)
        # Adding element type (line 163)
        # Getting the type of 'src' (line 163)
        src_306905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 57), 'src', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 56), list_306904, src_306905)
        # Adding element type (line 163)
        str_306906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 62), 'str', '-o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 56), list_306904, str_306906)
        # Adding element type (line 163)
        # Getting the type of 'obj' (line 163)
        obj_306907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 68), 'obj', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 56), list_306904, obj_306907)
        
        # Applying the binary operator '+' (line 163)
        result_add_306908 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 54), '+', result_add_306903, list_306904)
        
        # Getting the type of 'extra_postargs' (line 164)
        extra_postargs_306909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), 'extra_postargs', False)
        # Applying the binary operator '+' (line 163)
        result_add_306910 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 73), '+', result_add_306908, extra_postargs_306909)
        
        # Processing the call keyword arguments (line 163)
        kwargs_306911 = {}
        # Getting the type of 'self' (line 163)
        self_306898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 163)
        spawn_306899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 16), self_306898, 'spawn')
        # Calling spawn(args, kwargs) (line 163)
        spawn_call_result_306912 = invoke(stypy.reporting.localization.Localization(__file__, 163, 16), spawn_306899, *[result_add_306910], **kwargs_306911)
        
        # SSA branch for the except part of a try statement (line 162)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 162)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 165)
        DistutilsExecError_306913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 19), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'msg', DistutilsExecError_306913)
        # Getting the type of 'CompileError' (line 166)
        CompileError_306914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 22), 'CompileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 166, 16), CompileError_306914, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 162)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 155)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_compile' in the type store
        # Getting the type of 'stypy_return_type' (line 154)
        stypy_return_type_306915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_306915)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_compile'
        return stypy_return_type_306915


    @norecursion
    def link(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 172)
        None_306916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 25), 'None')
        # Getting the type of 'None' (line 173)
        None_306917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'None')
        # Getting the type of 'None' (line 174)
        None_306918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'None')
        # Getting the type of 'None' (line 175)
        None_306919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 35), 'None')
        # Getting the type of 'None' (line 176)
        None_306920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 29), 'None')
        int_306921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 20), 'int')
        # Getting the type of 'None' (line 178)
        None_306922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 28), 'None')
        # Getting the type of 'None' (line 179)
        None_306923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 29), 'None')
        # Getting the type of 'None' (line 180)
        None_306924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 25), 'None')
        # Getting the type of 'None' (line 181)
        None_306925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 26), 'None')
        defaults = [None_306916, None_306917, None_306918, None_306919, None_306920, int_306921, None_306922, None_306923, None_306924, None_306925]
        # Create a new context for function 'link'
        module_type_store = module_type_store.open_function_context('link', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CygwinCCompiler.link.__dict__.__setitem__('stypy_localization', localization)
        CygwinCCompiler.link.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CygwinCCompiler.link.__dict__.__setitem__('stypy_type_store', module_type_store)
        CygwinCCompiler.link.__dict__.__setitem__('stypy_function_name', 'CygwinCCompiler.link')
        CygwinCCompiler.link.__dict__.__setitem__('stypy_param_names_list', ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'])
        CygwinCCompiler.link.__dict__.__setitem__('stypy_varargs_param_name', None)
        CygwinCCompiler.link.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CygwinCCompiler.link.__dict__.__setitem__('stypy_call_defaults', defaults)
        CygwinCCompiler.link.__dict__.__setitem__('stypy_call_varargs', varargs)
        CygwinCCompiler.link.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CygwinCCompiler.link.__dict__.__setitem__('stypy_declared_arg_number', 14)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CygwinCCompiler.link', ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 184):
        
        # Assigning a Call to a Name (line 184):
        
        # Call to copy(...): (line 184)
        # Processing the call arguments (line 184)
        
        # Evaluating a boolean operation
        # Getting the type of 'extra_preargs' (line 184)
        extra_preargs_306928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 34), 'extra_preargs', False)
        
        # Obtaining an instance of the builtin type 'list' (line 184)
        list_306929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 51), 'list')
        # Adding type elements to the builtin type 'list' instance (line 184)
        
        # Applying the binary operator 'or' (line 184)
        result_or_keyword_306930 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 34), 'or', extra_preargs_306928, list_306929)
        
        # Processing the call keyword arguments (line 184)
        kwargs_306931 = {}
        # Getting the type of 'copy' (line 184)
        copy_306926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 24), 'copy', False)
        # Obtaining the member 'copy' of a type (line 184)
        copy_306927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 24), copy_306926, 'copy')
        # Calling copy(args, kwargs) (line 184)
        copy_call_result_306932 = invoke(stypy.reporting.localization.Localization(__file__, 184, 24), copy_306927, *[result_or_keyword_306930], **kwargs_306931)
        
        # Assigning a type to the variable 'extra_preargs' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'extra_preargs', copy_call_result_306932)
        
        # Assigning a Call to a Name (line 185):
        
        # Assigning a Call to a Name (line 185):
        
        # Call to copy(...): (line 185)
        # Processing the call arguments (line 185)
        
        # Evaluating a boolean operation
        # Getting the type of 'libraries' (line 185)
        libraries_306935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 30), 'libraries', False)
        
        # Obtaining an instance of the builtin type 'list' (line 185)
        list_306936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 185)
        
        # Applying the binary operator 'or' (line 185)
        result_or_keyword_306937 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 30), 'or', libraries_306935, list_306936)
        
        # Processing the call keyword arguments (line 185)
        kwargs_306938 = {}
        # Getting the type of 'copy' (line 185)
        copy_306933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 20), 'copy', False)
        # Obtaining the member 'copy' of a type (line 185)
        copy_306934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 20), copy_306933, 'copy')
        # Calling copy(args, kwargs) (line 185)
        copy_call_result_306939 = invoke(stypy.reporting.localization.Localization(__file__, 185, 20), copy_306934, *[result_or_keyword_306937], **kwargs_306938)
        
        # Assigning a type to the variable 'libraries' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'libraries', copy_call_result_306939)
        
        # Assigning a Call to a Name (line 186):
        
        # Assigning a Call to a Name (line 186):
        
        # Call to copy(...): (line 186)
        # Processing the call arguments (line 186)
        
        # Evaluating a boolean operation
        # Getting the type of 'objects' (line 186)
        objects_306942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 28), 'objects', False)
        
        # Obtaining an instance of the builtin type 'list' (line 186)
        list_306943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 186)
        
        # Applying the binary operator 'or' (line 186)
        result_or_keyword_306944 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 28), 'or', objects_306942, list_306943)
        
        # Processing the call keyword arguments (line 186)
        kwargs_306945 = {}
        # Getting the type of 'copy' (line 186)
        copy_306940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 18), 'copy', False)
        # Obtaining the member 'copy' of a type (line 186)
        copy_306941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 18), copy_306940, 'copy')
        # Calling copy(args, kwargs) (line 186)
        copy_call_result_306946 = invoke(stypy.reporting.localization.Localization(__file__, 186, 18), copy_306941, *[result_or_keyword_306944], **kwargs_306945)
        
        # Assigning a type to the variable 'objects' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'objects', copy_call_result_306946)
        
        # Call to extend(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'self' (line 189)
        self_306949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 25), 'self', False)
        # Obtaining the member 'dll_libraries' of a type (line 189)
        dll_libraries_306950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 25), self_306949, 'dll_libraries')
        # Processing the call keyword arguments (line 189)
        kwargs_306951 = {}
        # Getting the type of 'libraries' (line 189)
        libraries_306947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'libraries', False)
        # Obtaining the member 'extend' of a type (line 189)
        extend_306948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), libraries_306947, 'extend')
        # Calling extend(args, kwargs) (line 189)
        extend_call_result_306952 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), extend_306948, *[dll_libraries_306950], **kwargs_306951)
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'export_symbols' (line 193)
        export_symbols_306953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 13), 'export_symbols')
        # Getting the type of 'None' (line 193)
        None_306954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 35), 'None')
        # Applying the binary operator 'isnot' (line 193)
        result_is_not_306955 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 13), 'isnot', export_symbols_306953, None_306954)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'target_desc' (line 194)
        target_desc_306956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 13), 'target_desc')
        # Getting the type of 'self' (line 194)
        self_306957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 28), 'self')
        # Obtaining the member 'EXECUTABLE' of a type (line 194)
        EXECUTABLE_306958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 28), self_306957, 'EXECUTABLE')
        # Applying the binary operator '!=' (line 194)
        result_ne_306959 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 13), '!=', target_desc_306956, EXECUTABLE_306958)
        
        
        # Getting the type of 'self' (line 194)
        self_306960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 47), 'self')
        # Obtaining the member 'linker_dll' of a type (line 194)
        linker_dll_306961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 47), self_306960, 'linker_dll')
        str_306962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 66), 'str', 'gcc')
        # Applying the binary operator '==' (line 194)
        result_eq_306963 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 47), '==', linker_dll_306961, str_306962)
        
        # Applying the binary operator 'or' (line 194)
        result_or_keyword_306964 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 13), 'or', result_ne_306959, result_eq_306963)
        
        # Applying the binary operator 'and' (line 193)
        result_and_keyword_306965 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 12), 'and', result_is_not_306955, result_or_keyword_306964)
        
        # Testing the type of an if condition (line 193)
        if_condition_306966 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 8), result_and_keyword_306965)
        # Assigning a type to the variable 'if_condition_306966' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'if_condition_306966', if_condition_306966)
        # SSA begins for if statement (line 193)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 203):
        
        # Assigning a Call to a Name (line 203):
        
        # Call to dirname(...): (line 203)
        # Processing the call arguments (line 203)
        
        # Obtaining the type of the subscript
        int_306970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 47), 'int')
        # Getting the type of 'objects' (line 203)
        objects_306971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 39), 'objects', False)
        # Obtaining the member '__getitem__' of a type (line 203)
        getitem___306972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 39), objects_306971, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 203)
        subscript_call_result_306973 = invoke(stypy.reporting.localization.Localization(__file__, 203, 39), getitem___306972, int_306970)
        
        # Processing the call keyword arguments (line 203)
        kwargs_306974 = {}
        # Getting the type of 'os' (line 203)
        os_306967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 203)
        path_306968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 23), os_306967, 'path')
        # Obtaining the member 'dirname' of a type (line 203)
        dirname_306969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 23), path_306968, 'dirname')
        # Calling dirname(args, kwargs) (line 203)
        dirname_call_result_306975 = invoke(stypy.reporting.localization.Localization(__file__, 203, 23), dirname_306969, *[subscript_call_result_306973], **kwargs_306974)
        
        # Assigning a type to the variable 'temp_dir' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'temp_dir', dirname_call_result_306975)
        
        # Assigning a Call to a Tuple (line 205):
        
        # Assigning a Call to a Name:
        
        # Call to splitext(...): (line 205)
        # Processing the call arguments (line 205)
        
        # Call to basename(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'output_filename' (line 206)
        output_filename_306982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 33), 'output_filename', False)
        # Processing the call keyword arguments (line 206)
        kwargs_306983 = {}
        # Getting the type of 'os' (line 206)
        os_306979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 206)
        path_306980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 16), os_306979, 'path')
        # Obtaining the member 'basename' of a type (line 206)
        basename_306981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 16), path_306980, 'basename')
        # Calling basename(args, kwargs) (line 206)
        basename_call_result_306984 = invoke(stypy.reporting.localization.Localization(__file__, 206, 16), basename_306981, *[output_filename_306982], **kwargs_306983)
        
        # Processing the call keyword arguments (line 205)
        kwargs_306985 = {}
        # Getting the type of 'os' (line 205)
        os_306976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 40), 'os', False)
        # Obtaining the member 'path' of a type (line 205)
        path_306977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 40), os_306976, 'path')
        # Obtaining the member 'splitext' of a type (line 205)
        splitext_306978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 40), path_306977, 'splitext')
        # Calling splitext(args, kwargs) (line 205)
        splitext_call_result_306986 = invoke(stypy.reporting.localization.Localization(__file__, 205, 40), splitext_306978, *[basename_call_result_306984], **kwargs_306985)
        
        # Assigning a type to the variable 'call_assignment_306670' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'call_assignment_306670', splitext_call_result_306986)
        
        # Assigning a Call to a Name (line 205):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_306989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 12), 'int')
        # Processing the call keyword arguments
        kwargs_306990 = {}
        # Getting the type of 'call_assignment_306670' (line 205)
        call_assignment_306670_306987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'call_assignment_306670', False)
        # Obtaining the member '__getitem__' of a type (line 205)
        getitem___306988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 12), call_assignment_306670_306987, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_306991 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___306988, *[int_306989], **kwargs_306990)
        
        # Assigning a type to the variable 'call_assignment_306671' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'call_assignment_306671', getitem___call_result_306991)
        
        # Assigning a Name to a Name (line 205):
        # Getting the type of 'call_assignment_306671' (line 205)
        call_assignment_306671_306992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'call_assignment_306671')
        # Assigning a type to the variable 'dll_name' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 13), 'dll_name', call_assignment_306671_306992)
        
        # Assigning a Call to a Name (line 205):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_306995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 12), 'int')
        # Processing the call keyword arguments
        kwargs_306996 = {}
        # Getting the type of 'call_assignment_306670' (line 205)
        call_assignment_306670_306993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'call_assignment_306670', False)
        # Obtaining the member '__getitem__' of a type (line 205)
        getitem___306994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 12), call_assignment_306670_306993, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_306997 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___306994, *[int_306995], **kwargs_306996)
        
        # Assigning a type to the variable 'call_assignment_306672' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'call_assignment_306672', getitem___call_result_306997)
        
        # Assigning a Name to a Name (line 205):
        # Getting the type of 'call_assignment_306672' (line 205)
        call_assignment_306672_306998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'call_assignment_306672')
        # Assigning a type to the variable 'dll_extension' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 23), 'dll_extension', call_assignment_306672_306998)
        
        # Assigning a Call to a Name (line 209):
        
        # Assigning a Call to a Name (line 209):
        
        # Call to join(...): (line 209)
        # Processing the call arguments (line 209)
        # Getting the type of 'temp_dir' (line 209)
        temp_dir_307002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 36), 'temp_dir', False)
        # Getting the type of 'dll_name' (line 209)
        dll_name_307003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 46), 'dll_name', False)
        str_307004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 57), 'str', '.def')
        # Applying the binary operator '+' (line 209)
        result_add_307005 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 46), '+', dll_name_307003, str_307004)
        
        # Processing the call keyword arguments (line 209)
        kwargs_307006 = {}
        # Getting the type of 'os' (line 209)
        os_306999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 209)
        path_307000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 23), os_306999, 'path')
        # Obtaining the member 'join' of a type (line 209)
        join_307001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 23), path_307000, 'join')
        # Calling join(args, kwargs) (line 209)
        join_call_result_307007 = invoke(stypy.reporting.localization.Localization(__file__, 209, 23), join_307001, *[temp_dir_307002, result_add_307005], **kwargs_307006)
        
        # Assigning a type to the variable 'def_file' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'def_file', join_call_result_307007)
        
        # Assigning a Call to a Name (line 210):
        
        # Assigning a Call to a Name (line 210):
        
        # Call to join(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'temp_dir' (line 210)
        temp_dir_307011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 36), 'temp_dir', False)
        str_307012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 46), 'str', 'lib')
        # Getting the type of 'dll_name' (line 210)
        dll_name_307013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 54), 'dll_name', False)
        # Applying the binary operator '+' (line 210)
        result_add_307014 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 46), '+', str_307012, dll_name_307013)
        
        str_307015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 65), 'str', '.a')
        # Applying the binary operator '+' (line 210)
        result_add_307016 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 63), '+', result_add_307014, str_307015)
        
        # Processing the call keyword arguments (line 210)
        kwargs_307017 = {}
        # Getting the type of 'os' (line 210)
        os_307008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 210)
        path_307009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 23), os_307008, 'path')
        # Obtaining the member 'join' of a type (line 210)
        join_307010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 23), path_307009, 'join')
        # Calling join(args, kwargs) (line 210)
        join_call_result_307018 = invoke(stypy.reporting.localization.Localization(__file__, 210, 23), join_307010, *[temp_dir_307011, result_add_307016], **kwargs_307017)
        
        # Assigning a type to the variable 'lib_file' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'lib_file', join_call_result_307018)
        
        # Assigning a List to a Name (line 213):
        
        # Assigning a List to a Name (line 213):
        
        # Obtaining an instance of the builtin type 'list' (line 213)
        list_307019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 213)
        # Adding element type (line 213)
        str_307020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 16), 'str', 'LIBRARY %s')
        
        # Call to basename(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'output_filename' (line 214)
        output_filename_307024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 48), 'output_filename', False)
        # Processing the call keyword arguments (line 214)
        kwargs_307025 = {}
        # Getting the type of 'os' (line 214)
        os_307021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 214)
        path_307022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 31), os_307021, 'path')
        # Obtaining the member 'basename' of a type (line 214)
        basename_307023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 31), path_307022, 'basename')
        # Calling basename(args, kwargs) (line 214)
        basename_call_result_307026 = invoke(stypy.reporting.localization.Localization(__file__, 214, 31), basename_307023, *[output_filename_307024], **kwargs_307025)
        
        # Applying the binary operator '%' (line 214)
        result_mod_307027 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 16), '%', str_307020, basename_call_result_307026)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 23), list_307019, result_mod_307027)
        # Adding element type (line 213)
        str_307028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 16), 'str', 'EXPORTS')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 23), list_307019, str_307028)
        
        # Assigning a type to the variable 'contents' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'contents', list_307019)
        
        # Getting the type of 'export_symbols' (line 216)
        export_symbols_307029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 23), 'export_symbols')
        # Testing the type of a for loop iterable (line 216)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 216, 12), export_symbols_307029)
        # Getting the type of the for loop variable (line 216)
        for_loop_var_307030 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 216, 12), export_symbols_307029)
        # Assigning a type to the variable 'sym' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'sym', for_loop_var_307030)
        # SSA begins for a for statement (line 216)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'sym' (line 217)
        sym_307033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 32), 'sym', False)
        # Processing the call keyword arguments (line 217)
        kwargs_307034 = {}
        # Getting the type of 'contents' (line 217)
        contents_307031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 16), 'contents', False)
        # Obtaining the member 'append' of a type (line 217)
        append_307032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 16), contents_307031, 'append')
        # Calling append(args, kwargs) (line 217)
        append_call_result_307035 = invoke(stypy.reporting.localization.Localization(__file__, 217, 16), append_307032, *[sym_307033], **kwargs_307034)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to execute(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'write_file' (line 218)
        write_file_307038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 25), 'write_file', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 218)
        tuple_307039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 218)
        # Adding element type (line 218)
        # Getting the type of 'def_file' (line 218)
        def_file_307040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 38), 'def_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 38), tuple_307039, def_file_307040)
        # Adding element type (line 218)
        # Getting the type of 'contents' (line 218)
        contents_307041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 48), 'contents', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 38), tuple_307039, contents_307041)
        
        str_307042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 25), 'str', 'writing %s')
        # Getting the type of 'def_file' (line 219)
        def_file_307043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 40), 'def_file', False)
        # Applying the binary operator '%' (line 219)
        result_mod_307044 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 25), '%', str_307042, def_file_307043)
        
        # Processing the call keyword arguments (line 218)
        kwargs_307045 = {}
        # Getting the type of 'self' (line 218)
        self_307036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'self', False)
        # Obtaining the member 'execute' of a type (line 218)
        execute_307037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), self_307036, 'execute')
        # Calling execute(args, kwargs) (line 218)
        execute_call_result_307046 = invoke(stypy.reporting.localization.Localization(__file__, 218, 12), execute_307037, *[write_file_307038, tuple_307039, result_mod_307044], **kwargs_307045)
        
        
        
        # Getting the type of 'self' (line 224)
        self_307047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 15), 'self')
        # Obtaining the member 'linker_dll' of a type (line 224)
        linker_dll_307048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 15), self_307047, 'linker_dll')
        str_307049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 34), 'str', 'dllwrap')
        # Applying the binary operator '==' (line 224)
        result_eq_307050 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 15), '==', linker_dll_307048, str_307049)
        
        # Testing the type of an if condition (line 224)
        if_condition_307051 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 12), result_eq_307050)
        # Assigning a type to the variable 'if_condition_307051' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'if_condition_307051', if_condition_307051)
        # SSA begins for if statement (line 224)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 225)
        # Processing the call arguments (line 225)
        
        # Obtaining an instance of the builtin type 'list' (line 225)
        list_307054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 225)
        # Adding element type (line 225)
        str_307055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 38), 'str', '--output-lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 37), list_307054, str_307055)
        # Adding element type (line 225)
        # Getting the type of 'lib_file' (line 225)
        lib_file_307056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 54), 'lib_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 37), list_307054, lib_file_307056)
        
        # Processing the call keyword arguments (line 225)
        kwargs_307057 = {}
        # Getting the type of 'extra_preargs' (line 225)
        extra_preargs_307052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'extra_preargs', False)
        # Obtaining the member 'extend' of a type (line 225)
        extend_307053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 16), extra_preargs_307052, 'extend')
        # Calling extend(args, kwargs) (line 225)
        extend_call_result_307058 = invoke(stypy.reporting.localization.Localization(__file__, 225, 16), extend_307053, *[list_307054], **kwargs_307057)
        
        
        # Call to extend(...): (line 227)
        # Processing the call arguments (line 227)
        
        # Obtaining an instance of the builtin type 'list' (line 227)
        list_307061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 227)
        # Adding element type (line 227)
        str_307062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 38), 'str', '--def')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 37), list_307061, str_307062)
        # Adding element type (line 227)
        # Getting the type of 'def_file' (line 227)
        def_file_307063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 47), 'def_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 37), list_307061, def_file_307063)
        
        # Processing the call keyword arguments (line 227)
        kwargs_307064 = {}
        # Getting the type of 'extra_preargs' (line 227)
        extra_preargs_307059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'extra_preargs', False)
        # Obtaining the member 'extend' of a type (line 227)
        extend_307060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), extra_preargs_307059, 'extend')
        # Calling extend(args, kwargs) (line 227)
        extend_call_result_307065 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), extend_307060, *[list_307061], **kwargs_307064)
        
        # SSA branch for the else part of an if statement (line 224)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'def_file' (line 233)
        def_file_307068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 31), 'def_file', False)
        # Processing the call keyword arguments (line 233)
        kwargs_307069 = {}
        # Getting the type of 'objects' (line 233)
        objects_307066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'objects', False)
        # Obtaining the member 'append' of a type (line 233)
        append_307067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 16), objects_307066, 'append')
        # Calling append(args, kwargs) (line 233)
        append_call_result_307070 = invoke(stypy.reporting.localization.Localization(__file__, 233, 16), append_307067, *[def_file_307068], **kwargs_307069)
        
        # SSA join for if statement (line 224)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 193)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'debug' (line 244)
        debug_307071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 15), 'debug')
        # Applying the 'not' unary operator (line 244)
        result_not__307072 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 11), 'not', debug_307071)
        
        # Testing the type of an if condition (line 244)
        if_condition_307073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 244, 8), result_not__307072)
        # Assigning a type to the variable 'if_condition_307073' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'if_condition_307073', if_condition_307073)
        # SSA begins for if statement (line 244)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 245)
        # Processing the call arguments (line 245)
        str_307076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 33), 'str', '-s')
        # Processing the call keyword arguments (line 245)
        kwargs_307077 = {}
        # Getting the type of 'extra_preargs' (line 245)
        extra_preargs_307074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'extra_preargs', False)
        # Obtaining the member 'append' of a type (line 245)
        append_307075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 12), extra_preargs_307074, 'append')
        # Calling append(args, kwargs) (line 245)
        append_call_result_307078 = invoke(stypy.reporting.localization.Localization(__file__, 245, 12), append_307075, *[str_307076], **kwargs_307077)
        
        # SSA join for if statement (line 244)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to link(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'self' (line 247)
        self_307081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 27), 'self', False)
        # Getting the type of 'target_desc' (line 248)
        target_desc_307082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 27), 'target_desc', False)
        # Getting the type of 'objects' (line 249)
        objects_307083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 27), 'objects', False)
        # Getting the type of 'output_filename' (line 250)
        output_filename_307084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 27), 'output_filename', False)
        # Getting the type of 'output_dir' (line 251)
        output_dir_307085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 27), 'output_dir', False)
        # Getting the type of 'libraries' (line 252)
        libraries_307086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 27), 'libraries', False)
        # Getting the type of 'library_dirs' (line 253)
        library_dirs_307087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 27), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 254)
        runtime_library_dirs_307088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 27), 'runtime_library_dirs', False)
        # Getting the type of 'None' (line 255)
        None_307089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 27), 'None', False)
        # Getting the type of 'debug' (line 256)
        debug_307090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 27), 'debug', False)
        # Getting the type of 'extra_preargs' (line 257)
        extra_preargs_307091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 27), 'extra_preargs', False)
        # Getting the type of 'extra_postargs' (line 258)
        extra_postargs_307092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 27), 'extra_postargs', False)
        # Getting the type of 'build_temp' (line 259)
        build_temp_307093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 27), 'build_temp', False)
        # Getting the type of 'target_lang' (line 260)
        target_lang_307094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 27), 'target_lang', False)
        # Processing the call keyword arguments (line 247)
        kwargs_307095 = {}
        # Getting the type of 'UnixCCompiler' (line 247)
        UnixCCompiler_307079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'UnixCCompiler', False)
        # Obtaining the member 'link' of a type (line 247)
        link_307080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), UnixCCompiler_307079, 'link')
        # Calling link(args, kwargs) (line 247)
        link_call_result_307096 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), link_307080, *[self_307081, target_desc_307082, objects_307083, output_filename_307084, output_dir_307085, libraries_307086, library_dirs_307087, runtime_library_dirs_307088, None_307089, debug_307090, extra_preargs_307091, extra_postargs_307092, build_temp_307093, target_lang_307094], **kwargs_307095)
        
        
        # ################# End of 'link(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'link' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_307097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_307097)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'link'
        return stypy_return_type_307097


    @norecursion
    def object_filenames(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_307098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 36), 'int')
        str_307099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 37), 'str', '')
        defaults = [int_307098, str_307099]
        # Create a new context for function 'object_filenames'
        module_type_store = module_type_store.open_function_context('object_filenames', 267, 4, False)
        # Assigning a type to the variable 'self' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CygwinCCompiler.object_filenames.__dict__.__setitem__('stypy_localization', localization)
        CygwinCCompiler.object_filenames.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CygwinCCompiler.object_filenames.__dict__.__setitem__('stypy_type_store', module_type_store)
        CygwinCCompiler.object_filenames.__dict__.__setitem__('stypy_function_name', 'CygwinCCompiler.object_filenames')
        CygwinCCompiler.object_filenames.__dict__.__setitem__('stypy_param_names_list', ['source_filenames', 'strip_dir', 'output_dir'])
        CygwinCCompiler.object_filenames.__dict__.__setitem__('stypy_varargs_param_name', None)
        CygwinCCompiler.object_filenames.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CygwinCCompiler.object_filenames.__dict__.__setitem__('stypy_call_defaults', defaults)
        CygwinCCompiler.object_filenames.__dict__.__setitem__('stypy_call_varargs', varargs)
        CygwinCCompiler.object_filenames.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CygwinCCompiler.object_filenames.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CygwinCCompiler.object_filenames', ['source_filenames', 'strip_dir', 'output_dir'], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 271)
        # Getting the type of 'output_dir' (line 271)
        output_dir_307100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 11), 'output_dir')
        # Getting the type of 'None' (line 271)
        None_307101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 25), 'None')
        
        (may_be_307102, more_types_in_union_307103) = may_be_none(output_dir_307100, None_307101)

        if may_be_307102:

            if more_types_in_union_307103:
                # Runtime conditional SSA (line 271)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 271):
            
            # Assigning a Str to a Name (line 271):
            str_307104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 44), 'str', '')
            # Assigning a type to the variable 'output_dir' (line 271)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 31), 'output_dir', str_307104)

            if more_types_in_union_307103:
                # SSA join for if statement (line 271)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Name (line 272):
        
        # Assigning a List to a Name (line 272):
        
        # Obtaining an instance of the builtin type 'list' (line 272)
        list_307105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 272)
        
        # Assigning a type to the variable 'obj_names' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'obj_names', list_307105)
        
        # Getting the type of 'source_filenames' (line 273)
        source_filenames_307106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 24), 'source_filenames')
        # Testing the type of a for loop iterable (line 273)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 273, 8), source_filenames_307106)
        # Getting the type of the for loop variable (line 273)
        for_loop_var_307107 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 273, 8), source_filenames_307106)
        # Assigning a type to the variable 'src_name' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'src_name', for_loop_var_307107)
        # SSA begins for a for statement (line 273)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 275):
        
        # Assigning a Call to a Name:
        
        # Call to splitext(...): (line 275)
        # Processing the call arguments (line 275)
        
        # Call to normcase(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'src_name' (line 275)
        src_name_307114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 61), 'src_name', False)
        # Processing the call keyword arguments (line 275)
        kwargs_307115 = {}
        # Getting the type of 'os' (line 275)
        os_307111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 44), 'os', False)
        # Obtaining the member 'path' of a type (line 275)
        path_307112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 44), os_307111, 'path')
        # Obtaining the member 'normcase' of a type (line 275)
        normcase_307113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 44), path_307112, 'normcase')
        # Calling normcase(args, kwargs) (line 275)
        normcase_call_result_307116 = invoke(stypy.reporting.localization.Localization(__file__, 275, 44), normcase_307113, *[src_name_307114], **kwargs_307115)
        
        # Processing the call keyword arguments (line 275)
        kwargs_307117 = {}
        # Getting the type of 'os' (line 275)
        os_307108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 275)
        path_307109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 26), os_307108, 'path')
        # Obtaining the member 'splitext' of a type (line 275)
        splitext_307110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 26), path_307109, 'splitext')
        # Calling splitext(args, kwargs) (line 275)
        splitext_call_result_307118 = invoke(stypy.reporting.localization.Localization(__file__, 275, 26), splitext_307110, *[normcase_call_result_307116], **kwargs_307117)
        
        # Assigning a type to the variable 'call_assignment_306673' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'call_assignment_306673', splitext_call_result_307118)
        
        # Assigning a Call to a Name (line 275):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_307121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 12), 'int')
        # Processing the call keyword arguments
        kwargs_307122 = {}
        # Getting the type of 'call_assignment_306673' (line 275)
        call_assignment_306673_307119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'call_assignment_306673', False)
        # Obtaining the member '__getitem__' of a type (line 275)
        getitem___307120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 12), call_assignment_306673_307119, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_307123 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___307120, *[int_307121], **kwargs_307122)
        
        # Assigning a type to the variable 'call_assignment_306674' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'call_assignment_306674', getitem___call_result_307123)
        
        # Assigning a Name to a Name (line 275):
        # Getting the type of 'call_assignment_306674' (line 275)
        call_assignment_306674_307124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'call_assignment_306674')
        # Assigning a type to the variable 'base' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 13), 'base', call_assignment_306674_307124)
        
        # Assigning a Call to a Name (line 275):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_307127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 12), 'int')
        # Processing the call keyword arguments
        kwargs_307128 = {}
        # Getting the type of 'call_assignment_306673' (line 275)
        call_assignment_306673_307125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'call_assignment_306673', False)
        # Obtaining the member '__getitem__' of a type (line 275)
        getitem___307126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 12), call_assignment_306673_307125, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_307129 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___307126, *[int_307127], **kwargs_307128)
        
        # Assigning a type to the variable 'call_assignment_306675' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'call_assignment_306675', getitem___call_result_307129)
        
        # Assigning a Name to a Name (line 275):
        # Getting the type of 'call_assignment_306675' (line 275)
        call_assignment_306675_307130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'call_assignment_306675')
        # Assigning a type to the variable 'ext' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 19), 'ext', call_assignment_306675_307130)
        
        
        # Getting the type of 'ext' (line 276)
        ext_307131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 15), 'ext')
        # Getting the type of 'self' (line 276)
        self_307132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 27), 'self')
        # Obtaining the member 'src_extensions' of a type (line 276)
        src_extensions_307133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 27), self_307132, 'src_extensions')
        
        # Obtaining an instance of the builtin type 'list' (line 276)
        list_307134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 276)
        # Adding element type (line 276)
        str_307135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 50), 'str', '.rc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 49), list_307134, str_307135)
        # Adding element type (line 276)
        str_307136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 56), 'str', '.res')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 276, 49), list_307134, str_307136)
        
        # Applying the binary operator '+' (line 276)
        result_add_307137 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 27), '+', src_extensions_307133, list_307134)
        
        # Applying the binary operator 'notin' (line 276)
        result_contains_307138 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 15), 'notin', ext_307131, result_add_307137)
        
        # Testing the type of an if condition (line 276)
        if_condition_307139 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 276, 12), result_contains_307138)
        # Assigning a type to the variable 'if_condition_307139' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 12), 'if_condition_307139', if_condition_307139)
        # SSA begins for if statement (line 276)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'UnknownFileError' (line 277)
        UnknownFileError_307140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 22), 'UnknownFileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 277, 16), UnknownFileError_307140, 'raise parameter', BaseException)
        # SSA join for if statement (line 276)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'strip_dir' (line 280)
        strip_dir_307141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'strip_dir')
        # Testing the type of an if condition (line 280)
        if_condition_307142 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 12), strip_dir_307141)
        # Assigning a type to the variable 'if_condition_307142' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'if_condition_307142', if_condition_307142)
        # SSA begins for if statement (line 280)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 281):
        
        # Assigning a Call to a Name (line 281):
        
        # Call to basename(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'base' (line 281)
        base_307146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 41), 'base', False)
        # Processing the call keyword arguments (line 281)
        kwargs_307147 = {}
        # Getting the type of 'os' (line 281)
        os_307143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 281)
        path_307144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 23), os_307143, 'path')
        # Obtaining the member 'basename' of a type (line 281)
        basename_307145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 23), path_307144, 'basename')
        # Calling basename(args, kwargs) (line 281)
        basename_call_result_307148 = invoke(stypy.reporting.localization.Localization(__file__, 281, 23), basename_307145, *[base_307146], **kwargs_307147)
        
        # Assigning a type to the variable 'base' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 16), 'base', basename_call_result_307148)
        # SSA join for if statement (line 280)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'ext' (line 282)
        ext_307149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'ext')
        str_307150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 22), 'str', '.res')
        # Applying the binary operator '==' (line 282)
        result_eq_307151 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 15), '==', ext_307149, str_307150)
        
        
        # Getting the type of 'ext' (line 282)
        ext_307152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 32), 'ext')
        str_307153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 39), 'str', '.rc')
        # Applying the binary operator '==' (line 282)
        result_eq_307154 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 32), '==', ext_307152, str_307153)
        
        # Applying the binary operator 'or' (line 282)
        result_or_keyword_307155 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 15), 'or', result_eq_307151, result_eq_307154)
        
        # Testing the type of an if condition (line 282)
        if_condition_307156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 12), result_or_keyword_307155)
        # Assigning a type to the variable 'if_condition_307156' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'if_condition_307156', if_condition_307156)
        # SSA begins for if statement (line 282)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 284)
        # Processing the call arguments (line 284)
        
        # Call to join(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'output_dir' (line 284)
        output_dir_307162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 48), 'output_dir', False)
        # Getting the type of 'base' (line 285)
        base_307163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 44), 'base', False)
        # Getting the type of 'ext' (line 285)
        ext_307164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 51), 'ext', False)
        # Applying the binary operator '+' (line 285)
        result_add_307165 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 44), '+', base_307163, ext_307164)
        
        # Getting the type of 'self' (line 285)
        self_307166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 57), 'self', False)
        # Obtaining the member 'obj_extension' of a type (line 285)
        obj_extension_307167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 57), self_307166, 'obj_extension')
        # Applying the binary operator '+' (line 285)
        result_add_307168 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 55), '+', result_add_307165, obj_extension_307167)
        
        # Processing the call keyword arguments (line 284)
        kwargs_307169 = {}
        # Getting the type of 'os' (line 284)
        os_307159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 284)
        path_307160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 34), os_307159, 'path')
        # Obtaining the member 'join' of a type (line 284)
        join_307161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 34), path_307160, 'join')
        # Calling join(args, kwargs) (line 284)
        join_call_result_307170 = invoke(stypy.reporting.localization.Localization(__file__, 284, 34), join_307161, *[output_dir_307162, result_add_307168], **kwargs_307169)
        
        # Processing the call keyword arguments (line 284)
        kwargs_307171 = {}
        # Getting the type of 'obj_names' (line 284)
        obj_names_307157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'obj_names', False)
        # Obtaining the member 'append' of a type (line 284)
        append_307158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 16), obj_names_307157, 'append')
        # Calling append(args, kwargs) (line 284)
        append_call_result_307172 = invoke(stypy.reporting.localization.Localization(__file__, 284, 16), append_307158, *[join_call_result_307170], **kwargs_307171)
        
        # SSA branch for the else part of an if statement (line 282)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 287)
        # Processing the call arguments (line 287)
        
        # Call to join(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'output_dir' (line 287)
        output_dir_307178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 48), 'output_dir', False)
        # Getting the type of 'base' (line 288)
        base_307179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 44), 'base', False)
        # Getting the type of 'self' (line 288)
        self_307180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 51), 'self', False)
        # Obtaining the member 'obj_extension' of a type (line 288)
        obj_extension_307181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 51), self_307180, 'obj_extension')
        # Applying the binary operator '+' (line 288)
        result_add_307182 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 44), '+', base_307179, obj_extension_307181)
        
        # Processing the call keyword arguments (line 287)
        kwargs_307183 = {}
        # Getting the type of 'os' (line 287)
        os_307175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 287)
        path_307176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 34), os_307175, 'path')
        # Obtaining the member 'join' of a type (line 287)
        join_307177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 34), path_307176, 'join')
        # Calling join(args, kwargs) (line 287)
        join_call_result_307184 = invoke(stypy.reporting.localization.Localization(__file__, 287, 34), join_307177, *[output_dir_307178, result_add_307182], **kwargs_307183)
        
        # Processing the call keyword arguments (line 287)
        kwargs_307185 = {}
        # Getting the type of 'obj_names' (line 287)
        obj_names_307173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 16), 'obj_names', False)
        # Obtaining the member 'append' of a type (line 287)
        append_307174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 16), obj_names_307173, 'append')
        # Calling append(args, kwargs) (line 287)
        append_call_result_307186 = invoke(stypy.reporting.localization.Localization(__file__, 287, 16), append_307174, *[join_call_result_307184], **kwargs_307185)
        
        # SSA join for if statement (line 282)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'obj_names' (line 289)
        obj_names_307187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), 'obj_names')
        # Assigning a type to the variable 'stypy_return_type' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'stypy_return_type', obj_names_307187)
        
        # ################# End of 'object_filenames(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'object_filenames' in the type store
        # Getting the type of 'stypy_return_type' (line 267)
        stypy_return_type_307188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_307188)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'object_filenames'
        return stypy_return_type_307188


# Assigning a type to the variable 'CygwinCCompiler' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'CygwinCCompiler', CygwinCCompiler)

# Assigning a Str to a Name (line 84):
str_307189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 20), 'str', 'cygwin')
# Getting the type of 'CygwinCCompiler'
CygwinCCompiler_307190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CygwinCCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CygwinCCompiler_307190, 'compiler_type', str_307189)

# Assigning a Str to a Name (line 85):
str_307191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 20), 'str', '.o')
# Getting the type of 'CygwinCCompiler'
CygwinCCompiler_307192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CygwinCCompiler')
# Setting the type of the member 'obj_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CygwinCCompiler_307192, 'obj_extension', str_307191)

# Assigning a Str to a Name (line 86):
str_307193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 27), 'str', '.a')
# Getting the type of 'CygwinCCompiler'
CygwinCCompiler_307194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CygwinCCompiler')
# Setting the type of the member 'static_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CygwinCCompiler_307194, 'static_lib_extension', str_307193)

# Assigning a Str to a Name (line 87):
str_307195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 27), 'str', '.dll')
# Getting the type of 'CygwinCCompiler'
CygwinCCompiler_307196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CygwinCCompiler')
# Setting the type of the member 'shared_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CygwinCCompiler_307196, 'shared_lib_extension', str_307195)

# Assigning a Str to a Name (line 88):
str_307197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 24), 'str', 'lib%s%s')
# Getting the type of 'CygwinCCompiler'
CygwinCCompiler_307198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CygwinCCompiler')
# Setting the type of the member 'static_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CygwinCCompiler_307198, 'static_lib_format', str_307197)

# Assigning a Str to a Name (line 89):
str_307199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 24), 'str', '%s%s')
# Getting the type of 'CygwinCCompiler'
CygwinCCompiler_307200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CygwinCCompiler')
# Setting the type of the member 'shared_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CygwinCCompiler_307200, 'shared_lib_format', str_307199)

# Assigning a Str to a Name (line 90):
str_307201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 20), 'str', '.exe')
# Getting the type of 'CygwinCCompiler'
CygwinCCompiler_307202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CygwinCCompiler')
# Setting the type of the member 'exe_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CygwinCCompiler_307202, 'exe_extension', str_307201)
# Declaration of the 'Mingw32CCompiler' class
# Getting the type of 'CygwinCCompiler' (line 297)
CygwinCCompiler_307203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 24), 'CygwinCCompiler')

class Mingw32CCompiler(CygwinCCompiler_307203, ):
    
    # Assigning a Str to a Name (line 299):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_307204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 26), 'int')
        int_307205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 26), 'int')
        int_307206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 24), 'int')
        defaults = [int_307204, int_307205, int_307206]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 301, 4, False)
        # Assigning a type to the variable 'self' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Mingw32CCompiler.__init__', ['verbose', 'dry_run', 'force'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'self' (line 306)
        self_307209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 34), 'self', False)
        # Getting the type of 'verbose' (line 306)
        verbose_307210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 40), 'verbose', False)
        # Getting the type of 'dry_run' (line 306)
        dry_run_307211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 49), 'dry_run', False)
        # Getting the type of 'force' (line 306)
        force_307212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 58), 'force', False)
        # Processing the call keyword arguments (line 306)
        kwargs_307213 = {}
        # Getting the type of 'CygwinCCompiler' (line 306)
        CygwinCCompiler_307207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'CygwinCCompiler', False)
        # Obtaining the member '__init__' of a type (line 306)
        init___307208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 8), CygwinCCompiler_307207, '__init__')
        # Calling __init__(args, kwargs) (line 306)
        init___call_result_307214 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), init___307208, *[self_307209, verbose_307210, dry_run_307211, force_307212], **kwargs_307213)
        
        
        
        # Getting the type of 'self' (line 310)
        self_307215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 11), 'self')
        # Obtaining the member 'ld_version' of a type (line 310)
        ld_version_307216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 11), self_307215, 'ld_version')
        str_307217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 30), 'str', '2.13')
        # Applying the binary operator '>=' (line 310)
        result_ge_307218 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 11), '>=', ld_version_307216, str_307217)
        
        # Testing the type of an if condition (line 310)
        if_condition_307219 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 8), result_ge_307218)
        # Assigning a type to the variable 'if_condition_307219' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'if_condition_307219', if_condition_307219)
        # SSA begins for if statement (line 310)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 311):
        
        # Assigning a Str to a Name (line 311):
        str_307220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 28), 'str', '-shared')
        # Assigning a type to the variable 'shared_option' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'shared_option', str_307220)
        # SSA branch for the else part of an if statement (line 310)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 313):
        
        # Assigning a Str to a Name (line 313):
        str_307221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 28), 'str', '-mdll -static')
        # Assigning a type to the variable 'shared_option' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'shared_option', str_307221)
        # SSA join for if statement (line 310)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 317)
        self_307222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 11), 'self')
        # Obtaining the member 'gcc_version' of a type (line 317)
        gcc_version_307223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 11), self_307222, 'gcc_version')
        str_307224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 31), 'str', '2.91.57')
        # Applying the binary operator '<=' (line 317)
        result_le_307225 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 11), '<=', gcc_version_307223, str_307224)
        
        # Testing the type of an if condition (line 317)
        if_condition_307226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 8), result_le_307225)
        # Assigning a type to the variable 'if_condition_307226' (line 317)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'if_condition_307226', if_condition_307226)
        # SSA begins for if statement (line 317)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 318):
        
        # Assigning a Str to a Name (line 318):
        str_307227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 26), 'str', '--entry _DllMain@12')
        # Assigning a type to the variable 'entry_point' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'entry_point', str_307227)
        # SSA branch for the else part of an if statement (line 317)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 320):
        
        # Assigning a Str to a Name (line 320):
        str_307228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 26), 'str', '')
        # Assigning a type to the variable 'entry_point' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'entry_point', str_307228)
        # SSA join for if statement (line 317)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 322)
        self_307229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 11), 'self')
        # Obtaining the member 'gcc_version' of a type (line 322)
        gcc_version_307230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 11), self_307229, 'gcc_version')
        str_307231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 30), 'str', '4')
        # Applying the binary operator '<' (line 322)
        result_lt_307232 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 11), '<', gcc_version_307230, str_307231)
        
        
        # Call to is_cygwingcc(...): (line 322)
        # Processing the call keyword arguments (line 322)
        kwargs_307234 = {}
        # Getting the type of 'is_cygwingcc' (line 322)
        is_cygwingcc_307233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 37), 'is_cygwingcc', False)
        # Calling is_cygwingcc(args, kwargs) (line 322)
        is_cygwingcc_call_result_307235 = invoke(stypy.reporting.localization.Localization(__file__, 322, 37), is_cygwingcc_307233, *[], **kwargs_307234)
        
        # Applying the binary operator 'or' (line 322)
        result_or_keyword_307236 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 11), 'or', result_lt_307232, is_cygwingcc_call_result_307235)
        
        # Testing the type of an if condition (line 322)
        if_condition_307237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 8), result_or_keyword_307236)
        # Assigning a type to the variable 'if_condition_307237' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'if_condition_307237', if_condition_307237)
        # SSA begins for if statement (line 322)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 323):
        
        # Assigning a Str to a Name (line 323):
        str_307238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 24), 'str', ' -mno-cygwin')
        # Assigning a type to the variable 'no_cygwin' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'no_cygwin', str_307238)
        # SSA branch for the else part of an if statement (line 322)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 325):
        
        # Assigning a Str to a Name (line 325):
        str_307239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 24), 'str', '')
        # Assigning a type to the variable 'no_cygwin' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'no_cygwin', str_307239)
        # SSA join for if statement (line 322)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_executables(...): (line 327)
        # Processing the call keyword arguments (line 327)
        str_307242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 38), 'str', 'gcc%s -O -Wall')
        # Getting the type of 'no_cygwin' (line 327)
        no_cygwin_307243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 57), 'no_cygwin', False)
        # Applying the binary operator '%' (line 327)
        result_mod_307244 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 38), '%', str_307242, no_cygwin_307243)
        
        keyword_307245 = result_mod_307244
        str_307246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 41), 'str', 'gcc%s -mdll -O -Wall')
        # Getting the type of 'no_cygwin' (line 328)
        no_cygwin_307247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 66), 'no_cygwin', False)
        # Applying the binary operator '%' (line 328)
        result_mod_307248 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 41), '%', str_307246, no_cygwin_307247)
        
        keyword_307249 = result_mod_307248
        str_307250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 42), 'str', 'g++%s -O -Wall')
        # Getting the type of 'no_cygwin' (line 329)
        no_cygwin_307251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 61), 'no_cygwin', False)
        # Applying the binary operator '%' (line 329)
        result_mod_307252 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 42), '%', str_307250, no_cygwin_307251)
        
        keyword_307253 = result_mod_307252
        str_307254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 40), 'str', 'gcc%s')
        # Getting the type of 'no_cygwin' (line 330)
        no_cygwin_307255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 50), 'no_cygwin', False)
        # Applying the binary operator '%' (line 330)
        result_mod_307256 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 40), '%', str_307254, no_cygwin_307255)
        
        keyword_307257 = result_mod_307256
        str_307258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 39), 'str', '%s%s %s %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 332)
        tuple_307259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 332)
        # Adding element type (line 332)
        # Getting the type of 'self' (line 332)
        self_307260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 39), 'self', False)
        # Obtaining the member 'linker_dll' of a type (line 332)
        linker_dll_307261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 39), self_307260, 'linker_dll')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 39), tuple_307259, linker_dll_307261)
        # Adding element type (line 332)
        # Getting the type of 'no_cygwin' (line 332)
        no_cygwin_307262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 56), 'no_cygwin', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 39), tuple_307259, no_cygwin_307262)
        # Adding element type (line 332)
        # Getting the type of 'shared_option' (line 333)
        shared_option_307263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 39), 'shared_option', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 39), tuple_307259, shared_option_307263)
        # Adding element type (line 332)
        # Getting the type of 'entry_point' (line 333)
        entry_point_307264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 54), 'entry_point', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 39), tuple_307259, entry_point_307264)
        
        # Applying the binary operator '%' (line 331)
        result_mod_307265 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 39), '%', str_307258, tuple_307259)
        
        keyword_307266 = result_mod_307265
        kwargs_307267 = {'compiler_cxx': keyword_307253, 'linker_exe': keyword_307257, 'compiler_so': keyword_307249, 'linker_so': keyword_307266, 'compiler': keyword_307245}
        # Getting the type of 'self' (line 327)
        self_307240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'self', False)
        # Obtaining the member 'set_executables' of a type (line 327)
        set_executables_307241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 8), self_307240, 'set_executables')
        # Calling set_executables(args, kwargs) (line 327)
        set_executables_call_result_307268 = invoke(stypy.reporting.localization.Localization(__file__, 327, 8), set_executables_307241, *[], **kwargs_307267)
        
        
        # Assigning a List to a Attribute (line 339):
        
        # Assigning a List to a Attribute (line 339):
        
        # Obtaining an instance of the builtin type 'list' (line 339)
        list_307269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 339)
        
        # Getting the type of 'self' (line 339)
        self_307270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'self')
        # Setting the type of the member 'dll_libraries' of a type (line 339)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 8), self_307270, 'dll_libraries', list_307269)
        
        # Assigning a Call to a Attribute (line 343):
        
        # Assigning a Call to a Attribute (line 343):
        
        # Call to get_msvcr(...): (line 343)
        # Processing the call keyword arguments (line 343)
        kwargs_307272 = {}
        # Getting the type of 'get_msvcr' (line 343)
        get_msvcr_307271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 29), 'get_msvcr', False)
        # Calling get_msvcr(args, kwargs) (line 343)
        get_msvcr_call_result_307273 = invoke(stypy.reporting.localization.Localization(__file__, 343, 29), get_msvcr_307271, *[], **kwargs_307272)
        
        # Getting the type of 'self' (line 343)
        self_307274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'self')
        # Setting the type of the member 'dll_libraries' of a type (line 343)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), self_307274, 'dll_libraries', get_msvcr_call_result_307273)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Mingw32CCompiler' (line 297)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 0), 'Mingw32CCompiler', Mingw32CCompiler)

# Assigning a Str to a Name (line 299):
str_307275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 20), 'str', 'mingw32')
# Getting the type of 'Mingw32CCompiler'
Mingw32CCompiler_307276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Mingw32CCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Mingw32CCompiler_307276, 'compiler_type', str_307275)

# Assigning a Str to a Name (line 353):

# Assigning a Str to a Name (line 353):
str_307277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 353, 14), 'str', 'ok')
# Assigning a type to the variable 'CONFIG_H_OK' (line 353)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 0), 'CONFIG_H_OK', str_307277)

# Assigning a Str to a Name (line 354):

# Assigning a Str to a Name (line 354):
str_307278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 17), 'str', 'not ok')
# Assigning a type to the variable 'CONFIG_H_NOTOK' (line 354)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), 'CONFIG_H_NOTOK', str_307278)

# Assigning a Str to a Name (line 355):

# Assigning a Str to a Name (line 355):
str_307279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 21), 'str', 'uncertain')
# Assigning a type to the variable 'CONFIG_H_UNCERTAIN' (line 355)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 0), 'CONFIG_H_UNCERTAIN', str_307279)

@norecursion
def check_config_h(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_config_h'
    module_type_store = module_type_store.open_function_context('check_config_h', 357, 0, False)
    
    # Passed parameters checking function
    check_config_h.stypy_localization = localization
    check_config_h.stypy_type_of_self = None
    check_config_h.stypy_type_store = module_type_store
    check_config_h.stypy_function_name = 'check_config_h'
    check_config_h.stypy_param_names_list = []
    check_config_h.stypy_varargs_param_name = None
    check_config_h.stypy_kwargs_param_name = None
    check_config_h.stypy_call_defaults = defaults
    check_config_h.stypy_call_varargs = varargs
    check_config_h.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_config_h', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_config_h', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_config_h(...)' code ##################

    str_307280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, (-1)), 'str', 'Check if the current Python installation (specifically, pyconfig.h)\n    appears amenable to building extensions with GCC.  Returns a tuple\n    (status, details), where \'status\' is one of the following constants:\n      CONFIG_H_OK\n        all is well, go ahead and compile\n      CONFIG_H_NOTOK\n        doesn\'t look good\n      CONFIG_H_UNCERTAIN\n        not sure -- unable to read pyconfig.h\n    \'details\' is a human-readable string explaining the situation.\n\n    Note there are two ways to conclude "OK": either \'sys.version\' contains\n    the string "GCC" (implying that this Python was built with GCC), or the\n    installed "pyconfig.h" contains the string "__GNUC__".\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 378, 4))
    
    # 'from distutils import sysconfig' statement (line 378)
    try:
        from distutils import sysconfig

    except:
        sysconfig = UndefinedType
    import_from_module(stypy.reporting.localization.Localization(__file__, 378, 4), 'distutils', None, module_type_store, ['sysconfig'], [sysconfig])
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 379, 4))
    
    # 'import string' statement (line 379)
    import string

    import_module(stypy.reporting.localization.Localization(__file__, 379, 4), 'string', string, module_type_store)
    
    
    
    
    # Call to find(...): (line 382)
    # Processing the call arguments (line 382)
    # Getting the type of 'sys' (line 382)
    sys_307283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 19), 'sys', False)
    # Obtaining the member 'version' of a type (line 382)
    version_307284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 19), sys_307283, 'version')
    str_307285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 31), 'str', 'GCC')
    # Processing the call keyword arguments (line 382)
    kwargs_307286 = {}
    # Getting the type of 'string' (line 382)
    string_307281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 7), 'string', False)
    # Obtaining the member 'find' of a type (line 382)
    find_307282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 7), string_307281, 'find')
    # Calling find(args, kwargs) (line 382)
    find_call_result_307287 = invoke(stypy.reporting.localization.Localization(__file__, 382, 7), find_307282, *[version_307284, str_307285], **kwargs_307286)
    
    int_307288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 41), 'int')
    # Applying the binary operator '>=' (line 382)
    result_ge_307289 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 7), '>=', find_call_result_307287, int_307288)
    
    # Testing the type of an if condition (line 382)
    if_condition_307290 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 382, 4), result_ge_307289)
    # Assigning a type to the variable 'if_condition_307290' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'if_condition_307290', if_condition_307290)
    # SSA begins for if statement (line 382)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 383)
    tuple_307291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 383)
    # Adding element type (line 383)
    # Getting the type of 'CONFIG_H_OK' (line 383)
    CONFIG_H_OK_307292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'CONFIG_H_OK')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 16), tuple_307291, CONFIG_H_OK_307292)
    # Adding element type (line 383)
    str_307293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 29), 'str', "sys.version mentions 'GCC'")
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 383, 16), tuple_307291, str_307293)
    
    # Assigning a type to the variable 'stypy_return_type' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'stypy_return_type', tuple_307291)
    # SSA join for if statement (line 382)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 385):
    
    # Assigning a Call to a Name (line 385):
    
    # Call to get_config_h_filename(...): (line 385)
    # Processing the call keyword arguments (line 385)
    kwargs_307296 = {}
    # Getting the type of 'sysconfig' (line 385)
    sysconfig_307294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 9), 'sysconfig', False)
    # Obtaining the member 'get_config_h_filename' of a type (line 385)
    get_config_h_filename_307295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 9), sysconfig_307294, 'get_config_h_filename')
    # Calling get_config_h_filename(args, kwargs) (line 385)
    get_config_h_filename_call_result_307297 = invoke(stypy.reporting.localization.Localization(__file__, 385, 9), get_config_h_filename_307295, *[], **kwargs_307296)
    
    # Assigning a type to the variable 'fn' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'fn', get_config_h_filename_call_result_307297)
    
    
    # SSA begins for try-except statement (line 386)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 389):
    
    # Assigning a Call to a Name (line 389):
    
    # Call to open(...): (line 389)
    # Processing the call arguments (line 389)
    # Getting the type of 'fn' (line 389)
    fn_307299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 17), 'fn', False)
    # Processing the call keyword arguments (line 389)
    kwargs_307300 = {}
    # Getting the type of 'open' (line 389)
    open_307298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 'open', False)
    # Calling open(args, kwargs) (line 389)
    open_call_result_307301 = invoke(stypy.reporting.localization.Localization(__file__, 389, 12), open_307298, *[fn_307299], **kwargs_307300)
    
    # Assigning a type to the variable 'f' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'f', open_call_result_307301)
    
    # Try-finally block (line 390)
    
    # Assigning a Call to a Name (line 391):
    
    # Assigning a Call to a Name (line 391):
    
    # Call to read(...): (line 391)
    # Processing the call keyword arguments (line 391)
    kwargs_307304 = {}
    # Getting the type of 'f' (line 391)
    f_307302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 16), 'f', False)
    # Obtaining the member 'read' of a type (line 391)
    read_307303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 16), f_307302, 'read')
    # Calling read(args, kwargs) (line 391)
    read_call_result_307305 = invoke(stypy.reporting.localization.Localization(__file__, 391, 16), read_307303, *[], **kwargs_307304)
    
    # Assigning a type to the variable 's' (line 391)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 's', read_call_result_307305)
    
    # finally branch of the try-finally block (line 390)
    
    # Call to close(...): (line 393)
    # Processing the call keyword arguments (line 393)
    kwargs_307308 = {}
    # Getting the type of 'f' (line 393)
    f_307306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'f', False)
    # Obtaining the member 'close' of a type (line 393)
    close_307307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 12), f_307306, 'close')
    # Calling close(args, kwargs) (line 393)
    close_call_result_307309 = invoke(stypy.reporting.localization.Localization(__file__, 393, 12), close_307307, *[], **kwargs_307308)
    
    
    # SSA branch for the except part of a try statement (line 386)
    # SSA branch for the except 'IOError' branch of a try statement (line 386)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'IOError' (line 395)
    IOError_307310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 11), 'IOError')
    # Assigning a type to the variable 'exc' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'exc', IOError_307310)
    
    # Obtaining an instance of the builtin type 'tuple' (line 398)
    tuple_307311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 398)
    # Adding element type (line 398)
    # Getting the type of 'CONFIG_H_UNCERTAIN' (line 398)
    CONFIG_H_UNCERTAIN_307312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 16), 'CONFIG_H_UNCERTAIN')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 16), tuple_307311, CONFIG_H_UNCERTAIN_307312)
    # Adding element type (line 398)
    str_307313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 16), 'str', "couldn't read '%s': %s")
    
    # Obtaining an instance of the builtin type 'tuple' (line 399)
    tuple_307314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 44), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 399)
    # Adding element type (line 399)
    # Getting the type of 'fn' (line 399)
    fn_307315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 44), 'fn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 44), tuple_307314, fn_307315)
    # Adding element type (line 399)
    # Getting the type of 'exc' (line 399)
    exc_307316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 48), 'exc')
    # Obtaining the member 'strerror' of a type (line 399)
    strerror_307317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 48), exc_307316, 'strerror')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 399, 44), tuple_307314, strerror_307317)
    
    # Applying the binary operator '%' (line 399)
    result_mod_307318 = python_operator(stypy.reporting.localization.Localization(__file__, 399, 16), '%', str_307313, tuple_307314)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 398, 16), tuple_307311, result_mod_307318)
    
    # Assigning a type to the variable 'stypy_return_type' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'stypy_return_type', tuple_307311)
    # SSA branch for the else branch of a try statement (line 386)
    module_type_store.open_ssa_branch('except else')
    
    
    
    # Call to find(...): (line 403)
    # Processing the call arguments (line 403)
    # Getting the type of 's' (line 403)
    s_307321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 23), 's', False)
    str_307322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 25), 'str', '__GNUC__')
    # Processing the call keyword arguments (line 403)
    kwargs_307323 = {}
    # Getting the type of 'string' (line 403)
    string_307319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 11), 'string', False)
    # Obtaining the member 'find' of a type (line 403)
    find_307320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 11), string_307319, 'find')
    # Calling find(args, kwargs) (line 403)
    find_call_result_307324 = invoke(stypy.reporting.localization.Localization(__file__, 403, 11), find_307320, *[s_307321, str_307322], **kwargs_307323)
    
    int_307325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 40), 'int')
    # Applying the binary operator '>=' (line 403)
    result_ge_307326 = python_operator(stypy.reporting.localization.Localization(__file__, 403, 11), '>=', find_call_result_307324, int_307325)
    
    # Testing the type of an if condition (line 403)
    if_condition_307327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 403, 8), result_ge_307326)
    # Assigning a type to the variable 'if_condition_307327' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'if_condition_307327', if_condition_307327)
    # SSA begins for if statement (line 403)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 404)
    tuple_307328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 404)
    # Adding element type (line 404)
    # Getting the type of 'CONFIG_H_OK' (line 404)
    CONFIG_H_OK_307329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 20), 'CONFIG_H_OK')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 20), tuple_307328, CONFIG_H_OK_307329)
    # Adding element type (line 404)
    str_307330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 33), 'str', "'%s' mentions '__GNUC__'")
    # Getting the type of 'fn' (line 404)
    fn_307331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 62), 'fn')
    # Applying the binary operator '%' (line 404)
    result_mod_307332 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 33), '%', str_307330, fn_307331)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 404, 20), tuple_307328, result_mod_307332)
    
    # Assigning a type to the variable 'stypy_return_type' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 12), 'stypy_return_type', tuple_307328)
    # SSA branch for the else part of an if statement (line 403)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 406)
    tuple_307333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 406)
    # Adding element type (line 406)
    # Getting the type of 'CONFIG_H_NOTOK' (line 406)
    CONFIG_H_NOTOK_307334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 20), 'CONFIG_H_NOTOK')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 20), tuple_307333, CONFIG_H_NOTOK_307334)
    # Adding element type (line 406)
    str_307335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 36), 'str', "'%s' does not mention '__GNUC__'")
    # Getting the type of 'fn' (line 406)
    fn_307336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 73), 'fn')
    # Applying the binary operator '%' (line 406)
    result_mod_307337 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 36), '%', str_307335, fn_307336)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 20), tuple_307333, result_mod_307337)
    
    # Assigning a type to the variable 'stypy_return_type' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 12), 'stypy_return_type', tuple_307333)
    # SSA join for if statement (line 403)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 386)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_config_h(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_config_h' in the type store
    # Getting the type of 'stypy_return_type' (line 357)
    stypy_return_type_307338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_307338)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_config_h'
    return stypy_return_type_307338

# Assigning a type to the variable 'check_config_h' (line 357)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 0), 'check_config_h', check_config_h)

@norecursion
def get_versions(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_versions'
    module_type_store = module_type_store.open_function_context('get_versions', 410, 0, False)
    
    # Passed parameters checking function
    get_versions.stypy_localization = localization
    get_versions.stypy_type_of_self = None
    get_versions.stypy_type_store = module_type_store
    get_versions.stypy_function_name = 'get_versions'
    get_versions.stypy_param_names_list = []
    get_versions.stypy_varargs_param_name = None
    get_versions.stypy_kwargs_param_name = None
    get_versions.stypy_call_defaults = defaults
    get_versions.stypy_call_varargs = varargs
    get_versions.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_versions', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_versions', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_versions(...)' code ##################

    str_307339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, (-1)), 'str', ' Try to find out the versions of gcc, ld and dllwrap.\n        If not possible it returns None for it.\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 414, 4))
    
    # 'from distutils.version import LooseVersion' statement (line 414)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/')
    import_307340 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 414, 4), 'distutils.version')

    if (type(import_307340) is not StypyTypeError):

        if (import_307340 != 'pyd_module'):
            __import__(import_307340)
            sys_modules_307341 = sys.modules[import_307340]
            import_from_module(stypy.reporting.localization.Localization(__file__, 414, 4), 'distutils.version', sys_modules_307341.module_type_store, module_type_store, ['LooseVersion'])
            nest_module(stypy.reporting.localization.Localization(__file__, 414, 4), __file__, sys_modules_307341, sys_modules_307341.module_type_store, module_type_store)
        else:
            from distutils.version import LooseVersion

            import_from_module(stypy.reporting.localization.Localization(__file__, 414, 4), 'distutils.version', None, module_type_store, ['LooseVersion'], [LooseVersion])

    else:
        # Assigning a type to the variable 'distutils.version' (line 414)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'distutils.version', import_307340)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 415, 4))
    
    # 'from distutils.spawn import find_executable' statement (line 415)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/')
    import_307342 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 415, 4), 'distutils.spawn')

    if (type(import_307342) is not StypyTypeError):

        if (import_307342 != 'pyd_module'):
            __import__(import_307342)
            sys_modules_307343 = sys.modules[import_307342]
            import_from_module(stypy.reporting.localization.Localization(__file__, 415, 4), 'distutils.spawn', sys_modules_307343.module_type_store, module_type_store, ['find_executable'])
            nest_module(stypy.reporting.localization.Localization(__file__, 415, 4), __file__, sys_modules_307343, sys_modules_307343.module_type_store, module_type_store)
        else:
            from distutils.spawn import find_executable

            import_from_module(stypy.reporting.localization.Localization(__file__, 415, 4), 'distutils.spawn', None, module_type_store, ['find_executable'], [find_executable])

    else:
        # Assigning a type to the variable 'distutils.spawn' (line 415)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'distutils.spawn', import_307342)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 416, 4))
    
    # 'import re' statement (line 416)
    import re

    import_module(stypy.reporting.localization.Localization(__file__, 416, 4), 're', re, module_type_store)
    
    
    # Assigning a Call to a Name (line 418):
    
    # Assigning a Call to a Name (line 418):
    
    # Call to find_executable(...): (line 418)
    # Processing the call arguments (line 418)
    str_307345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 30), 'str', 'gcc')
    # Processing the call keyword arguments (line 418)
    kwargs_307346 = {}
    # Getting the type of 'find_executable' (line 418)
    find_executable_307344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 14), 'find_executable', False)
    # Calling find_executable(args, kwargs) (line 418)
    find_executable_call_result_307347 = invoke(stypy.reporting.localization.Localization(__file__, 418, 14), find_executable_307344, *[str_307345], **kwargs_307346)
    
    # Assigning a type to the variable 'gcc_exe' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'gcc_exe', find_executable_call_result_307347)
    
    # Getting the type of 'gcc_exe' (line 419)
    gcc_exe_307348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 7), 'gcc_exe')
    # Testing the type of an if condition (line 419)
    if_condition_307349 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 419, 4), gcc_exe_307348)
    # Assigning a type to the variable 'if_condition_307349' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'if_condition_307349', if_condition_307349)
    # SSA begins for if statement (line 419)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 420):
    
    # Assigning a Call to a Name (line 420):
    
    # Call to popen(...): (line 420)
    # Processing the call arguments (line 420)
    # Getting the type of 'gcc_exe' (line 420)
    gcc_exe_307352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 23), 'gcc_exe', False)
    str_307353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 33), 'str', ' -dumpversion')
    # Applying the binary operator '+' (line 420)
    result_add_307354 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 23), '+', gcc_exe_307352, str_307353)
    
    str_307355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 49), 'str', 'r')
    # Processing the call keyword arguments (line 420)
    kwargs_307356 = {}
    # Getting the type of 'os' (line 420)
    os_307350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 14), 'os', False)
    # Obtaining the member 'popen' of a type (line 420)
    popen_307351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 14), os_307350, 'popen')
    # Calling popen(args, kwargs) (line 420)
    popen_call_result_307357 = invoke(stypy.reporting.localization.Localization(__file__, 420, 14), popen_307351, *[result_add_307354, str_307355], **kwargs_307356)
    
    # Assigning a type to the variable 'out' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'out', popen_call_result_307357)
    
    # Assigning a Call to a Name (line 421):
    
    # Assigning a Call to a Name (line 421):
    
    # Call to read(...): (line 421)
    # Processing the call keyword arguments (line 421)
    kwargs_307360 = {}
    # Getting the type of 'out' (line 421)
    out_307358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 21), 'out', False)
    # Obtaining the member 'read' of a type (line 421)
    read_307359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 21), out_307358, 'read')
    # Calling read(args, kwargs) (line 421)
    read_call_result_307361 = invoke(stypy.reporting.localization.Localization(__file__, 421, 21), read_307359, *[], **kwargs_307360)
    
    # Assigning a type to the variable 'out_string' (line 421)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'out_string', read_call_result_307361)
    
    # Call to close(...): (line 422)
    # Processing the call keyword arguments (line 422)
    kwargs_307364 = {}
    # Getting the type of 'out' (line 422)
    out_307362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'out', False)
    # Obtaining the member 'close' of a type (line 422)
    close_307363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 8), out_307362, 'close')
    # Calling close(args, kwargs) (line 422)
    close_call_result_307365 = invoke(stypy.reporting.localization.Localization(__file__, 422, 8), close_307363, *[], **kwargs_307364)
    
    
    # Assigning a Call to a Name (line 423):
    
    # Assigning a Call to a Name (line 423):
    
    # Call to search(...): (line 423)
    # Processing the call arguments (line 423)
    str_307368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 27), 'str', '(\\d+\\.\\d+(\\.\\d+)*)')
    # Getting the type of 'out_string' (line 423)
    out_string_307369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 48), 'out_string', False)
    # Processing the call keyword arguments (line 423)
    kwargs_307370 = {}
    # Getting the type of 're' (line 423)
    re_307366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 17), 're', False)
    # Obtaining the member 'search' of a type (line 423)
    search_307367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 17), re_307366, 'search')
    # Calling search(args, kwargs) (line 423)
    search_call_result_307371 = invoke(stypy.reporting.localization.Localization(__file__, 423, 17), search_307367, *[str_307368, out_string_307369], **kwargs_307370)
    
    # Assigning a type to the variable 'result' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'result', search_call_result_307371)
    
    # Getting the type of 'result' (line 424)
    result_307372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 11), 'result')
    # Testing the type of an if condition (line 424)
    if_condition_307373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 424, 8), result_307372)
    # Assigning a type to the variable 'if_condition_307373' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'if_condition_307373', if_condition_307373)
    # SSA begins for if statement (line 424)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 425):
    
    # Assigning a Call to a Name (line 425):
    
    # Call to LooseVersion(...): (line 425)
    # Processing the call arguments (line 425)
    
    # Call to group(...): (line 425)
    # Processing the call arguments (line 425)
    int_307377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 52), 'int')
    # Processing the call keyword arguments (line 425)
    kwargs_307378 = {}
    # Getting the type of 'result' (line 425)
    result_307375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 39), 'result', False)
    # Obtaining the member 'group' of a type (line 425)
    group_307376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 39), result_307375, 'group')
    # Calling group(args, kwargs) (line 425)
    group_call_result_307379 = invoke(stypy.reporting.localization.Localization(__file__, 425, 39), group_307376, *[int_307377], **kwargs_307378)
    
    # Processing the call keyword arguments (line 425)
    kwargs_307380 = {}
    # Getting the type of 'LooseVersion' (line 425)
    LooseVersion_307374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 26), 'LooseVersion', False)
    # Calling LooseVersion(args, kwargs) (line 425)
    LooseVersion_call_result_307381 = invoke(stypy.reporting.localization.Localization(__file__, 425, 26), LooseVersion_307374, *[group_call_result_307379], **kwargs_307380)
    
    # Assigning a type to the variable 'gcc_version' (line 425)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 12), 'gcc_version', LooseVersion_call_result_307381)
    # SSA branch for the else part of an if statement (line 424)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 427):
    
    # Assigning a Name to a Name (line 427):
    # Getting the type of 'None' (line 427)
    None_307382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 26), 'None')
    # Assigning a type to the variable 'gcc_version' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 12), 'gcc_version', None_307382)
    # SSA join for if statement (line 424)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 419)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 429):
    
    # Assigning a Name to a Name (line 429):
    # Getting the type of 'None' (line 429)
    None_307383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 22), 'None')
    # Assigning a type to the variable 'gcc_version' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'gcc_version', None_307383)
    # SSA join for if statement (line 419)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 430):
    
    # Assigning a Call to a Name (line 430):
    
    # Call to find_executable(...): (line 430)
    # Processing the call arguments (line 430)
    str_307385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 29), 'str', 'ld')
    # Processing the call keyword arguments (line 430)
    kwargs_307386 = {}
    # Getting the type of 'find_executable' (line 430)
    find_executable_307384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 13), 'find_executable', False)
    # Calling find_executable(args, kwargs) (line 430)
    find_executable_call_result_307387 = invoke(stypy.reporting.localization.Localization(__file__, 430, 13), find_executable_307384, *[str_307385], **kwargs_307386)
    
    # Assigning a type to the variable 'ld_exe' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'ld_exe', find_executable_call_result_307387)
    
    # Getting the type of 'ld_exe' (line 431)
    ld_exe_307388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 7), 'ld_exe')
    # Testing the type of an if condition (line 431)
    if_condition_307389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 431, 4), ld_exe_307388)
    # Assigning a type to the variable 'if_condition_307389' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'if_condition_307389', if_condition_307389)
    # SSA begins for if statement (line 431)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 432):
    
    # Assigning a Call to a Name (line 432):
    
    # Call to popen(...): (line 432)
    # Processing the call arguments (line 432)
    # Getting the type of 'ld_exe' (line 432)
    ld_exe_307392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 23), 'ld_exe', False)
    str_307393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 32), 'str', ' -v')
    # Applying the binary operator '+' (line 432)
    result_add_307394 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 23), '+', ld_exe_307392, str_307393)
    
    str_307395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 38), 'str', 'r')
    # Processing the call keyword arguments (line 432)
    kwargs_307396 = {}
    # Getting the type of 'os' (line 432)
    os_307390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 14), 'os', False)
    # Obtaining the member 'popen' of a type (line 432)
    popen_307391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 14), os_307390, 'popen')
    # Calling popen(args, kwargs) (line 432)
    popen_call_result_307397 = invoke(stypy.reporting.localization.Localization(__file__, 432, 14), popen_307391, *[result_add_307394, str_307395], **kwargs_307396)
    
    # Assigning a type to the variable 'out' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'out', popen_call_result_307397)
    
    # Assigning a Call to a Name (line 433):
    
    # Assigning a Call to a Name (line 433):
    
    # Call to read(...): (line 433)
    # Processing the call keyword arguments (line 433)
    kwargs_307400 = {}
    # Getting the type of 'out' (line 433)
    out_307398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 21), 'out', False)
    # Obtaining the member 'read' of a type (line 433)
    read_307399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 21), out_307398, 'read')
    # Calling read(args, kwargs) (line 433)
    read_call_result_307401 = invoke(stypy.reporting.localization.Localization(__file__, 433, 21), read_307399, *[], **kwargs_307400)
    
    # Assigning a type to the variable 'out_string' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'out_string', read_call_result_307401)
    
    # Call to close(...): (line 434)
    # Processing the call keyword arguments (line 434)
    kwargs_307404 = {}
    # Getting the type of 'out' (line 434)
    out_307402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'out', False)
    # Obtaining the member 'close' of a type (line 434)
    close_307403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 8), out_307402, 'close')
    # Calling close(args, kwargs) (line 434)
    close_call_result_307405 = invoke(stypy.reporting.localization.Localization(__file__, 434, 8), close_307403, *[], **kwargs_307404)
    
    
    # Assigning a Call to a Name (line 435):
    
    # Assigning a Call to a Name (line 435):
    
    # Call to search(...): (line 435)
    # Processing the call arguments (line 435)
    str_307408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 27), 'str', '(\\d+\\.\\d+(\\.\\d+)*)')
    # Getting the type of 'out_string' (line 435)
    out_string_307409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 48), 'out_string', False)
    # Processing the call keyword arguments (line 435)
    kwargs_307410 = {}
    # Getting the type of 're' (line 435)
    re_307406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 17), 're', False)
    # Obtaining the member 'search' of a type (line 435)
    search_307407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 17), re_307406, 'search')
    # Calling search(args, kwargs) (line 435)
    search_call_result_307411 = invoke(stypy.reporting.localization.Localization(__file__, 435, 17), search_307407, *[str_307408, out_string_307409], **kwargs_307410)
    
    # Assigning a type to the variable 'result' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'result', search_call_result_307411)
    
    # Getting the type of 'result' (line 436)
    result_307412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 11), 'result')
    # Testing the type of an if condition (line 436)
    if_condition_307413 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 436, 8), result_307412)
    # Assigning a type to the variable 'if_condition_307413' (line 436)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'if_condition_307413', if_condition_307413)
    # SSA begins for if statement (line 436)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 437):
    
    # Assigning a Call to a Name (line 437):
    
    # Call to LooseVersion(...): (line 437)
    # Processing the call arguments (line 437)
    
    # Call to group(...): (line 437)
    # Processing the call arguments (line 437)
    int_307417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 51), 'int')
    # Processing the call keyword arguments (line 437)
    kwargs_307418 = {}
    # Getting the type of 'result' (line 437)
    result_307415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 38), 'result', False)
    # Obtaining the member 'group' of a type (line 437)
    group_307416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 38), result_307415, 'group')
    # Calling group(args, kwargs) (line 437)
    group_call_result_307419 = invoke(stypy.reporting.localization.Localization(__file__, 437, 38), group_307416, *[int_307417], **kwargs_307418)
    
    # Processing the call keyword arguments (line 437)
    kwargs_307420 = {}
    # Getting the type of 'LooseVersion' (line 437)
    LooseVersion_307414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 25), 'LooseVersion', False)
    # Calling LooseVersion(args, kwargs) (line 437)
    LooseVersion_call_result_307421 = invoke(stypy.reporting.localization.Localization(__file__, 437, 25), LooseVersion_307414, *[group_call_result_307419], **kwargs_307420)
    
    # Assigning a type to the variable 'ld_version' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'ld_version', LooseVersion_call_result_307421)
    # SSA branch for the else part of an if statement (line 436)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 439):
    
    # Assigning a Name to a Name (line 439):
    # Getting the type of 'None' (line 439)
    None_307422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 25), 'None')
    # Assigning a type to the variable 'ld_version' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'ld_version', None_307422)
    # SSA join for if statement (line 436)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 431)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 441):
    
    # Assigning a Name to a Name (line 441):
    # Getting the type of 'None' (line 441)
    None_307423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 21), 'None')
    # Assigning a type to the variable 'ld_version' (line 441)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'ld_version', None_307423)
    # SSA join for if statement (line 431)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 442):
    
    # Assigning a Call to a Name (line 442):
    
    # Call to find_executable(...): (line 442)
    # Processing the call arguments (line 442)
    str_307425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 34), 'str', 'dllwrap')
    # Processing the call keyword arguments (line 442)
    kwargs_307426 = {}
    # Getting the type of 'find_executable' (line 442)
    find_executable_307424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 18), 'find_executable', False)
    # Calling find_executable(args, kwargs) (line 442)
    find_executable_call_result_307427 = invoke(stypy.reporting.localization.Localization(__file__, 442, 18), find_executable_307424, *[str_307425], **kwargs_307426)
    
    # Assigning a type to the variable 'dllwrap_exe' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'dllwrap_exe', find_executable_call_result_307427)
    
    # Getting the type of 'dllwrap_exe' (line 443)
    dllwrap_exe_307428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 7), 'dllwrap_exe')
    # Testing the type of an if condition (line 443)
    if_condition_307429 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 443, 4), dllwrap_exe_307428)
    # Assigning a type to the variable 'if_condition_307429' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'if_condition_307429', if_condition_307429)
    # SSA begins for if statement (line 443)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 444):
    
    # Assigning a Call to a Name (line 444):
    
    # Call to popen(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'dllwrap_exe' (line 444)
    dllwrap_exe_307432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 23), 'dllwrap_exe', False)
    str_307433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 37), 'str', ' --version')
    # Applying the binary operator '+' (line 444)
    result_add_307434 = python_operator(stypy.reporting.localization.Localization(__file__, 444, 23), '+', dllwrap_exe_307432, str_307433)
    
    str_307435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 444, 50), 'str', 'r')
    # Processing the call keyword arguments (line 444)
    kwargs_307436 = {}
    # Getting the type of 'os' (line 444)
    os_307430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 14), 'os', False)
    # Obtaining the member 'popen' of a type (line 444)
    popen_307431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 14), os_307430, 'popen')
    # Calling popen(args, kwargs) (line 444)
    popen_call_result_307437 = invoke(stypy.reporting.localization.Localization(__file__, 444, 14), popen_307431, *[result_add_307434, str_307435], **kwargs_307436)
    
    # Assigning a type to the variable 'out' (line 444)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'out', popen_call_result_307437)
    
    # Assigning a Call to a Name (line 445):
    
    # Assigning a Call to a Name (line 445):
    
    # Call to read(...): (line 445)
    # Processing the call keyword arguments (line 445)
    kwargs_307440 = {}
    # Getting the type of 'out' (line 445)
    out_307438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 21), 'out', False)
    # Obtaining the member 'read' of a type (line 445)
    read_307439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 21), out_307438, 'read')
    # Calling read(args, kwargs) (line 445)
    read_call_result_307441 = invoke(stypy.reporting.localization.Localization(__file__, 445, 21), read_307439, *[], **kwargs_307440)
    
    # Assigning a type to the variable 'out_string' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'out_string', read_call_result_307441)
    
    # Call to close(...): (line 446)
    # Processing the call keyword arguments (line 446)
    kwargs_307444 = {}
    # Getting the type of 'out' (line 446)
    out_307442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'out', False)
    # Obtaining the member 'close' of a type (line 446)
    close_307443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 8), out_307442, 'close')
    # Calling close(args, kwargs) (line 446)
    close_call_result_307445 = invoke(stypy.reporting.localization.Localization(__file__, 446, 8), close_307443, *[], **kwargs_307444)
    
    
    # Assigning a Call to a Name (line 447):
    
    # Assigning a Call to a Name (line 447):
    
    # Call to search(...): (line 447)
    # Processing the call arguments (line 447)
    str_307448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 27), 'str', ' (\\d+\\.\\d+(\\.\\d+)*)')
    # Getting the type of 'out_string' (line 447)
    out_string_307449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 49), 'out_string', False)
    # Processing the call keyword arguments (line 447)
    kwargs_307450 = {}
    # Getting the type of 're' (line 447)
    re_307446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 17), 're', False)
    # Obtaining the member 'search' of a type (line 447)
    search_307447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 17), re_307446, 'search')
    # Calling search(args, kwargs) (line 447)
    search_call_result_307451 = invoke(stypy.reporting.localization.Localization(__file__, 447, 17), search_307447, *[str_307448, out_string_307449], **kwargs_307450)
    
    # Assigning a type to the variable 'result' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'result', search_call_result_307451)
    
    # Getting the type of 'result' (line 448)
    result_307452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 11), 'result')
    # Testing the type of an if condition (line 448)
    if_condition_307453 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 448, 8), result_307452)
    # Assigning a type to the variable 'if_condition_307453' (line 448)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'if_condition_307453', if_condition_307453)
    # SSA begins for if statement (line 448)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 449):
    
    # Assigning a Call to a Name (line 449):
    
    # Call to LooseVersion(...): (line 449)
    # Processing the call arguments (line 449)
    
    # Call to group(...): (line 449)
    # Processing the call arguments (line 449)
    int_307457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 449, 56), 'int')
    # Processing the call keyword arguments (line 449)
    kwargs_307458 = {}
    # Getting the type of 'result' (line 449)
    result_307455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 43), 'result', False)
    # Obtaining the member 'group' of a type (line 449)
    group_307456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 43), result_307455, 'group')
    # Calling group(args, kwargs) (line 449)
    group_call_result_307459 = invoke(stypy.reporting.localization.Localization(__file__, 449, 43), group_307456, *[int_307457], **kwargs_307458)
    
    # Processing the call keyword arguments (line 449)
    kwargs_307460 = {}
    # Getting the type of 'LooseVersion' (line 449)
    LooseVersion_307454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 30), 'LooseVersion', False)
    # Calling LooseVersion(args, kwargs) (line 449)
    LooseVersion_call_result_307461 = invoke(stypy.reporting.localization.Localization(__file__, 449, 30), LooseVersion_307454, *[group_call_result_307459], **kwargs_307460)
    
    # Assigning a type to the variable 'dllwrap_version' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'dllwrap_version', LooseVersion_call_result_307461)
    # SSA branch for the else part of an if statement (line 448)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 451):
    
    # Assigning a Name to a Name (line 451):
    # Getting the type of 'None' (line 451)
    None_307462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 30), 'None')
    # Assigning a type to the variable 'dllwrap_version' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 12), 'dllwrap_version', None_307462)
    # SSA join for if statement (line 448)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 443)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 453):
    
    # Assigning a Name to a Name (line 453):
    # Getting the type of 'None' (line 453)
    None_307463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 26), 'None')
    # Assigning a type to the variable 'dllwrap_version' (line 453)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'dllwrap_version', None_307463)
    # SSA join for if statement (line 443)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 454)
    tuple_307464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 454)
    # Adding element type (line 454)
    # Getting the type of 'gcc_version' (line 454)
    gcc_version_307465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 12), 'gcc_version')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 12), tuple_307464, gcc_version_307465)
    # Adding element type (line 454)
    # Getting the type of 'ld_version' (line 454)
    ld_version_307466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 25), 'ld_version')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 12), tuple_307464, ld_version_307466)
    # Adding element type (line 454)
    # Getting the type of 'dllwrap_version' (line 454)
    dllwrap_version_307467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 37), 'dllwrap_version')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 454, 12), tuple_307464, dllwrap_version_307467)
    
    # Assigning a type to the variable 'stypy_return_type' (line 454)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 4), 'stypy_return_type', tuple_307464)
    
    # ################# End of 'get_versions(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_versions' in the type store
    # Getting the type of 'stypy_return_type' (line 410)
    stypy_return_type_307468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_307468)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_versions'
    return stypy_return_type_307468

# Assigning a type to the variable 'get_versions' (line 410)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 0), 'get_versions', get_versions)

@norecursion
def is_cygwingcc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_cygwingcc'
    module_type_store = module_type_store.open_function_context('is_cygwingcc', 456, 0, False)
    
    # Passed parameters checking function
    is_cygwingcc.stypy_localization = localization
    is_cygwingcc.stypy_type_of_self = None
    is_cygwingcc.stypy_type_store = module_type_store
    is_cygwingcc.stypy_function_name = 'is_cygwingcc'
    is_cygwingcc.stypy_param_names_list = []
    is_cygwingcc.stypy_varargs_param_name = None
    is_cygwingcc.stypy_kwargs_param_name = None
    is_cygwingcc.stypy_call_defaults = defaults
    is_cygwingcc.stypy_call_varargs = varargs
    is_cygwingcc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_cygwingcc', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_cygwingcc', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_cygwingcc(...)' code ##################

    str_307469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 4), 'str', 'Try to determine if the gcc that would be used is from cygwin.')
    
    # Assigning a Call to a Name (line 458):
    
    # Assigning a Call to a Name (line 458):
    
    # Call to popen(...): (line 458)
    # Processing the call arguments (line 458)
    str_307472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 19), 'str', 'gcc -dumpmachine')
    str_307473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 39), 'str', 'r')
    # Processing the call keyword arguments (line 458)
    kwargs_307474 = {}
    # Getting the type of 'os' (line 458)
    os_307470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 10), 'os', False)
    # Obtaining the member 'popen' of a type (line 458)
    popen_307471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 10), os_307470, 'popen')
    # Calling popen(args, kwargs) (line 458)
    popen_call_result_307475 = invoke(stypy.reporting.localization.Localization(__file__, 458, 10), popen_307471, *[str_307472, str_307473], **kwargs_307474)
    
    # Assigning a type to the variable 'out' (line 458)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 4), 'out', popen_call_result_307475)
    
    # Assigning a Call to a Name (line 459):
    
    # Assigning a Call to a Name (line 459):
    
    # Call to read(...): (line 459)
    # Processing the call keyword arguments (line 459)
    kwargs_307478 = {}
    # Getting the type of 'out' (line 459)
    out_307476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 17), 'out', False)
    # Obtaining the member 'read' of a type (line 459)
    read_307477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 17), out_307476, 'read')
    # Calling read(args, kwargs) (line 459)
    read_call_result_307479 = invoke(stypy.reporting.localization.Localization(__file__, 459, 17), read_307477, *[], **kwargs_307478)
    
    # Assigning a type to the variable 'out_string' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'out_string', read_call_result_307479)
    
    # Call to close(...): (line 460)
    # Processing the call keyword arguments (line 460)
    kwargs_307482 = {}
    # Getting the type of 'out' (line 460)
    out_307480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 4), 'out', False)
    # Obtaining the member 'close' of a type (line 460)
    close_307481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 4), out_307480, 'close')
    # Calling close(args, kwargs) (line 460)
    close_call_result_307483 = invoke(stypy.reporting.localization.Localization(__file__, 460, 4), close_307481, *[], **kwargs_307482)
    
    
    # Call to endswith(...): (line 463)
    # Processing the call arguments (line 463)
    str_307489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 39), 'str', 'cygwin')
    # Processing the call keyword arguments (line 463)
    kwargs_307490 = {}
    
    # Call to strip(...): (line 463)
    # Processing the call keyword arguments (line 463)
    kwargs_307486 = {}
    # Getting the type of 'out_string' (line 463)
    out_string_307484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 11), 'out_string', False)
    # Obtaining the member 'strip' of a type (line 463)
    strip_307485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 11), out_string_307484, 'strip')
    # Calling strip(args, kwargs) (line 463)
    strip_call_result_307487 = invoke(stypy.reporting.localization.Localization(__file__, 463, 11), strip_307485, *[], **kwargs_307486)
    
    # Obtaining the member 'endswith' of a type (line 463)
    endswith_307488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 11), strip_call_result_307487, 'endswith')
    # Calling endswith(args, kwargs) (line 463)
    endswith_call_result_307491 = invoke(stypy.reporting.localization.Localization(__file__, 463, 11), endswith_307488, *[str_307489], **kwargs_307490)
    
    # Assigning a type to the variable 'stypy_return_type' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'stypy_return_type', endswith_call_result_307491)
    
    # ################# End of 'is_cygwingcc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_cygwingcc' in the type store
    # Getting the type of 'stypy_return_type' (line 456)
    stypy_return_type_307492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_307492)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_cygwingcc'
    return stypy_return_type_307492

# Assigning a type to the variable 'is_cygwingcc' (line 456)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 0), 'is_cygwingcc', is_cygwingcc)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
