
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.bcppcompiler
2: 
3: Contains BorlandCCompiler, an implementation of the abstract CCompiler class
4: for the Borland C++ compiler.
5: '''
6: 
7: # This implementation by Lyle Johnson, based on the original msvccompiler.py
8: # module and using the directions originally published by Gordon Williams.
9: 
10: # XXX looks like there's a LOT of overlap between these two classes:
11: # someone should sit down and factor out the common code as
12: # WindowsCCompiler!  --GPW
13: 
14: __revision__ = "$Id$"
15: 
16: import os
17: 
18: from distutils.errors import (DistutilsExecError, CompileError, LibError,
19:                               LinkError, UnknownFileError)
20: from distutils.ccompiler import CCompiler, gen_preprocess_options
21: from distutils.file_util import write_file
22: from distutils.dep_util import newer
23: from distutils import log
24: 
25: class BCPPCompiler(CCompiler) :
26:     '''Concrete class that implements an interface to the Borland C/C++
27:     compiler, as defined by the CCompiler abstract class.
28:     '''
29: 
30:     compiler_type = 'bcpp'
31: 
32:     # Just set this so CCompiler's constructor doesn't barf.  We currently
33:     # don't use the 'set_executables()' bureaucracy provided by CCompiler,
34:     # as it really isn't necessary for this sort of single-compiler class.
35:     # Would be nice to have a consistent interface with UnixCCompiler,
36:     # though, so it's worth thinking about.
37:     executables = {}
38: 
39:     # Private class data (need to distinguish C from C++ source for compiler)
40:     _c_extensions = ['.c']
41:     _cpp_extensions = ['.cc', '.cpp', '.cxx']
42: 
43:     # Needed for the filename generation methods provided by the
44:     # base class, CCompiler.
45:     src_extensions = _c_extensions + _cpp_extensions
46:     obj_extension = '.obj'
47:     static_lib_extension = '.lib'
48:     shared_lib_extension = '.dll'
49:     static_lib_format = shared_lib_format = '%s%s'
50:     exe_extension = '.exe'
51: 
52: 
53:     def __init__ (self,
54:                   verbose=0,
55:                   dry_run=0,
56:                   force=0):
57: 
58:         CCompiler.__init__ (self, verbose, dry_run, force)
59: 
60:         # These executables are assumed to all be in the path.
61:         # Borland doesn't seem to use any special registry settings to
62:         # indicate their installation locations.
63: 
64:         self.cc = "bcc32.exe"
65:         self.linker = "ilink32.exe"
66:         self.lib = "tlib.exe"
67: 
68:         self.preprocess_options = None
69:         self.compile_options = ['/tWM', '/O2', '/q', '/g0']
70:         self.compile_options_debug = ['/tWM', '/Od', '/q', '/g0']
71: 
72:         self.ldflags_shared = ['/Tpd', '/Gn', '/q', '/x']
73:         self.ldflags_shared_debug = ['/Tpd', '/Gn', '/q', '/x']
74:         self.ldflags_static = []
75:         self.ldflags_exe = ['/Gn', '/q', '/x']
76:         self.ldflags_exe_debug = ['/Gn', '/q', '/x','/r']
77: 
78: 
79:     # -- Worker methods ------------------------------------------------
80: 
81:     def compile(self, sources,
82:                 output_dir=None, macros=None, include_dirs=None, debug=0,
83:                 extra_preargs=None, extra_postargs=None, depends=None):
84: 
85:         macros, objects, extra_postargs, pp_opts, build = \
86:                 self._setup_compile(output_dir, macros, include_dirs, sources,
87:                                     depends, extra_postargs)
88:         compile_opts = extra_preargs or []
89:         compile_opts.append ('-c')
90:         if debug:
91:             compile_opts.extend (self.compile_options_debug)
92:         else:
93:             compile_opts.extend (self.compile_options)
94: 
95:         for obj in objects:
96:             try:
97:                 src, ext = build[obj]
98:             except KeyError:
99:                 continue
100:             # XXX why do the normpath here?
101:             src = os.path.normpath(src)
102:             obj = os.path.normpath(obj)
103:             # XXX _setup_compile() did a mkpath() too but before the normpath.
104:             # Is it possible to skip the normpath?
105:             self.mkpath(os.path.dirname(obj))
106: 
107:             if ext == '.res':
108:                 # This is already a binary file -- skip it.
109:                 continue # the 'for' loop
110:             if ext == '.rc':
111:                 # This needs to be compiled to a .res file -- do it now.
112:                 try:
113:                     self.spawn (["brcc32", "-fo", obj, src])
114:                 except DistutilsExecError, msg:
115:                     raise CompileError, msg
116:                 continue # the 'for' loop
117: 
118:             # The next two are both for the real compiler.
119:             if ext in self._c_extensions:
120:                 input_opt = ""
121:             elif ext in self._cpp_extensions:
122:                 input_opt = "-P"
123:             else:
124:                 # Unknown file type -- no extra options.  The compiler
125:                 # will probably fail, but let it just in case this is a
126:                 # file the compiler recognizes even if we don't.
127:                 input_opt = ""
128: 
129:             output_opt = "-o" + obj
130: 
131:             # Compiler command line syntax is: "bcc32 [options] file(s)".
132:             # Note that the source file names must appear at the end of
133:             # the command line.
134:             try:
135:                 self.spawn ([self.cc] + compile_opts + pp_opts +
136:                             [input_opt, output_opt] +
137:                             extra_postargs + [src])
138:             except DistutilsExecError, msg:
139:                 raise CompileError, msg
140: 
141:         return objects
142: 
143:     # compile ()
144: 
145: 
146:     def create_static_lib (self,
147:                            objects,
148:                            output_libname,
149:                            output_dir=None,
150:                            debug=0,
151:                            target_lang=None):
152: 
153:         (objects, output_dir) = self._fix_object_args (objects, output_dir)
154:         output_filename = \
155:             self.library_filename (output_libname, output_dir=output_dir)
156: 
157:         if self._need_link (objects, output_filename):
158:             lib_args = [output_filename, '/u'] + objects
159:             if debug:
160:                 pass                    # XXX what goes here?
161:             try:
162:                 self.spawn ([self.lib] + lib_args)
163:             except DistutilsExecError, msg:
164:                 raise LibError, msg
165:         else:
166:             log.debug("skipping %s (up-to-date)", output_filename)
167: 
168:     # create_static_lib ()
169: 
170: 
171:     def link (self,
172:               target_desc,
173:               objects,
174:               output_filename,
175:               output_dir=None,
176:               libraries=None,
177:               library_dirs=None,
178:               runtime_library_dirs=None,
179:               export_symbols=None,
180:               debug=0,
181:               extra_preargs=None,
182:               extra_postargs=None,
183:               build_temp=None,
184:               target_lang=None):
185: 
186:         # XXX this ignores 'build_temp'!  should follow the lead of
187:         # msvccompiler.py
188: 
189:         (objects, output_dir) = self._fix_object_args (objects, output_dir)
190:         (libraries, library_dirs, runtime_library_dirs) = \
191:             self._fix_lib_args (libraries, library_dirs, runtime_library_dirs)
192: 
193:         if runtime_library_dirs:
194:             log.warn("I don't know what to do with 'runtime_library_dirs': %s",
195:                      str(runtime_library_dirs))
196: 
197:         if output_dir is not None:
198:             output_filename = os.path.join (output_dir, output_filename)
199: 
200:         if self._need_link (objects, output_filename):
201: 
202:             # Figure out linker args based on type of target.
203:             if target_desc == CCompiler.EXECUTABLE:
204:                 startup_obj = 'c0w32'
205:                 if debug:
206:                     ld_args = self.ldflags_exe_debug[:]
207:                 else:
208:                     ld_args = self.ldflags_exe[:]
209:             else:
210:                 startup_obj = 'c0d32'
211:                 if debug:
212:                     ld_args = self.ldflags_shared_debug[:]
213:                 else:
214:                     ld_args = self.ldflags_shared[:]
215: 
216: 
217:             # Create a temporary exports file for use by the linker
218:             if export_symbols is None:
219:                 def_file = ''
220:             else:
221:                 head, tail = os.path.split (output_filename)
222:                 modname, ext = os.path.splitext (tail)
223:                 temp_dir = os.path.dirname(objects[0]) # preserve tree structure
224:                 def_file = os.path.join (temp_dir, '%s.def' % modname)
225:                 contents = ['EXPORTS']
226:                 for sym in (export_symbols or []):
227:                     contents.append('  %s=_%s' % (sym, sym))
228:                 self.execute(write_file, (def_file, contents),
229:                              "writing %s" % def_file)
230: 
231:             # Borland C++ has problems with '/' in paths
232:             objects2 = map(os.path.normpath, objects)
233:             # split objects in .obj and .res files
234:             # Borland C++ needs them at different positions in the command line
235:             objects = [startup_obj]
236:             resources = []
237:             for file in objects2:
238:                 (base, ext) = os.path.splitext(os.path.normcase(file))
239:                 if ext == '.res':
240:                     resources.append(file)
241:                 else:
242:                     objects.append(file)
243: 
244: 
245:             for l in library_dirs:
246:                 ld_args.append("/L%s" % os.path.normpath(l))
247:             ld_args.append("/L.") # we sometimes use relative paths
248: 
249:             # list of object files
250:             ld_args.extend(objects)
251: 
252:             # XXX the command-line syntax for Borland C++ is a bit wonky;
253:             # certain filenames are jammed together in one big string, but
254:             # comma-delimited.  This doesn't mesh too well with the
255:             # Unix-centric attitude (with a DOS/Windows quoting hack) of
256:             # 'spawn()', so constructing the argument list is a bit
257:             # awkward.  Note that doing the obvious thing and jamming all
258:             # the filenames and commas into one argument would be wrong,
259:             # because 'spawn()' would quote any filenames with spaces in
260:             # them.  Arghghh!.  Apparently it works fine as coded...
261: 
262:             # name of dll/exe file
263:             ld_args.extend([',',output_filename])
264:             # no map file and start libraries
265:             ld_args.append(',,')
266: 
267:             for lib in libraries:
268:                 # see if we find it and if there is a bcpp specific lib
269:                 # (xxx_bcpp.lib)
270:                 libfile = self.find_library_file(library_dirs, lib, debug)
271:                 if libfile is None:
272:                     ld_args.append(lib)
273:                     # probably a BCPP internal library -- don't warn
274:                 else:
275:                     # full name which prefers bcpp_xxx.lib over xxx.lib
276:                     ld_args.append(libfile)
277: 
278:             # some default libraries
279:             ld_args.append ('import32')
280:             ld_args.append ('cw32mt')
281: 
282:             # def file for export symbols
283:             ld_args.extend([',',def_file])
284:             # add resource files
285:             ld_args.append(',')
286:             ld_args.extend(resources)
287: 
288: 
289:             if extra_preargs:
290:                 ld_args[:0] = extra_preargs
291:             if extra_postargs:
292:                 ld_args.extend(extra_postargs)
293: 
294:             self.mkpath (os.path.dirname (output_filename))
295:             try:
296:                 self.spawn ([self.linker] + ld_args)
297:             except DistutilsExecError, msg:
298:                 raise LinkError, msg
299: 
300:         else:
301:             log.debug("skipping %s (up-to-date)", output_filename)
302: 
303:     # link ()
304: 
305:     # -- Miscellaneous methods -----------------------------------------
306: 
307: 
308:     def find_library_file (self, dirs, lib, debug=0):
309:         # List of effective library names to try, in order of preference:
310:         # xxx_bcpp.lib is better than xxx.lib
311:         # and xxx_d.lib is better than xxx.lib if debug is set
312:         #
313:         # The "_bcpp" suffix is to handle a Python installation for people
314:         # with multiple compilers (primarily Distutils hackers, I suspect
315:         # ;-).  The idea is they'd have one static library for each
316:         # compiler they care about, since (almost?) every Windows compiler
317:         # seems to have a different format for static libraries.
318:         if debug:
319:             dlib = (lib + "_d")
320:             try_names = (dlib + "_bcpp", lib + "_bcpp", dlib, lib)
321:         else:
322:             try_names = (lib + "_bcpp", lib)
323: 
324:         for dir in dirs:
325:             for name in try_names:
326:                 libfile = os.path.join(dir, self.library_filename(name))
327:                 if os.path.exists(libfile):
328:                     return libfile
329:         else:
330:             # Oops, didn't find it in *any* of 'dirs'
331:             return None
332: 
333:     # overwrite the one from CCompiler to support rc and res-files
334:     def object_filenames (self,
335:                           source_filenames,
336:                           strip_dir=0,
337:                           output_dir=''):
338:         if output_dir is None: output_dir = ''
339:         obj_names = []
340:         for src_name in source_filenames:
341:             # use normcase to make sure '.rc' is really '.rc' and not '.RC'
342:             (base, ext) = os.path.splitext (os.path.normcase(src_name))
343:             if ext not in (self.src_extensions + ['.rc','.res']):
344:                 raise UnknownFileError, \
345:                       "unknown file type '%s' (from '%s')" % \
346:                       (ext, src_name)
347:             if strip_dir:
348:                 base = os.path.basename (base)
349:             if ext == '.res':
350:                 # these can go unchanged
351:                 obj_names.append (os.path.join (output_dir, base + ext))
352:             elif ext == '.rc':
353:                 # these need to be compiled to .res-files
354:                 obj_names.append (os.path.join (output_dir, base + '.res'))
355:             else:
356:                 obj_names.append (os.path.join (output_dir,
357:                                             base + self.obj_extension))
358:         return obj_names
359: 
360:     # object_filenames ()
361: 
362:     def preprocess (self,
363:                     source,
364:                     output_file=None,
365:                     macros=None,
366:                     include_dirs=None,
367:                     extra_preargs=None,
368:                     extra_postargs=None):
369: 
370:         (_, macros, include_dirs) = \
371:             self._fix_compile_args(None, macros, include_dirs)
372:         pp_opts = gen_preprocess_options(macros, include_dirs)
373:         pp_args = ['cpp32.exe'] + pp_opts
374:         if output_file is not None:
375:             pp_args.append('-o' + output_file)
376:         if extra_preargs:
377:             pp_args[:0] = extra_preargs
378:         if extra_postargs:
379:             pp_args.extend(extra_postargs)
380:         pp_args.append(source)
381: 
382:         # We need to preprocess: either we're being forced to, or the
383:         # source file is newer than the target (or the target doesn't
384:         # exist).
385:         if self.force or output_file is None or newer(source, output_file):
386:             if output_file:
387:                 self.mkpath(os.path.dirname(output_file))
388:             try:
389:                 self.spawn(pp_args)
390:             except DistutilsExecError, msg:
391:                 print msg
392:                 raise CompileError, msg
393: 
394:     # preprocess()
395: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_302859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', 'distutils.bcppcompiler\n\nContains BorlandCCompiler, an implementation of the abstract CCompiler class\nfor the Borland C++ compiler.\n')

# Assigning a Str to a Name (line 14):

# Assigning a Str to a Name (line 14):
str_302860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '__revision__', str_302860)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import os' statement (line 16)
import os

import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from distutils.errors import DistutilsExecError, CompileError, LibError, LinkError, UnknownFileError' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_302861 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.errors')

if (type(import_302861) is not StypyTypeError):

    if (import_302861 != 'pyd_module'):
        __import__(import_302861)
        sys_modules_302862 = sys.modules[import_302861]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.errors', sys_modules_302862.module_type_store, module_type_store, ['DistutilsExecError', 'CompileError', 'LibError', 'LinkError', 'UnknownFileError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_302862, sys_modules_302862.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsExecError, CompileError, LibError, LinkError, UnknownFileError

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.errors', None, module_type_store, ['DistutilsExecError', 'CompileError', 'LibError', 'LinkError', 'UnknownFileError'], [DistutilsExecError, CompileError, LibError, LinkError, UnknownFileError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.errors', import_302861)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from distutils.ccompiler import CCompiler, gen_preprocess_options' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_302863 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.ccompiler')

if (type(import_302863) is not StypyTypeError):

    if (import_302863 != 'pyd_module'):
        __import__(import_302863)
        sys_modules_302864 = sys.modules[import_302863]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.ccompiler', sys_modules_302864.module_type_store, module_type_store, ['CCompiler', 'gen_preprocess_options'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_302864, sys_modules_302864.module_type_store, module_type_store)
    else:
        from distutils.ccompiler import CCompiler, gen_preprocess_options

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.ccompiler', None, module_type_store, ['CCompiler', 'gen_preprocess_options'], [CCompiler, gen_preprocess_options])

else:
    # Assigning a type to the variable 'distutils.ccompiler' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.ccompiler', import_302863)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from distutils.file_util import write_file' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_302865 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.file_util')

if (type(import_302865) is not StypyTypeError):

    if (import_302865 != 'pyd_module'):
        __import__(import_302865)
        sys_modules_302866 = sys.modules[import_302865]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.file_util', sys_modules_302866.module_type_store, module_type_store, ['write_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_302866, sys_modules_302866.module_type_store, module_type_store)
    else:
        from distutils.file_util import write_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.file_util', None, module_type_store, ['write_file'], [write_file])

else:
    # Assigning a type to the variable 'distutils.file_util' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.file_util', import_302865)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from distutils.dep_util import newer' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_302867 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.dep_util')

if (type(import_302867) is not StypyTypeError):

    if (import_302867 != 'pyd_module'):
        __import__(import_302867)
        sys_modules_302868 = sys.modules[import_302867]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.dep_util', sys_modules_302868.module_type_store, module_type_store, ['newer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_302868, sys_modules_302868.module_type_store, module_type_store)
    else:
        from distutils.dep_util import newer

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.dep_util', None, module_type_store, ['newer'], [newer])

else:
    # Assigning a type to the variable 'distutils.dep_util' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.dep_util', import_302867)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from distutils import log' statement (line 23)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'distutils', None, module_type_store, ['log'], [log])

# Declaration of the 'BCPPCompiler' class
# Getting the type of 'CCompiler' (line 25)
CCompiler_302869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'CCompiler')

class BCPPCompiler(CCompiler_302869, ):
    str_302870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, (-1)), 'str', 'Concrete class that implements an interface to the Borland C/C++\n    compiler, as defined by the CCompiler abstract class.\n    ')
    
    # Assigning a Str to a Name (line 30):
    
    # Assigning a Dict to a Name (line 37):
    
    # Assigning a List to a Name (line 40):
    
    # Assigning a List to a Name (line 41):
    
    # Assigning a BinOp to a Name (line 45):
    
    # Assigning a Str to a Name (line 46):
    
    # Assigning a Str to a Name (line 47):
    
    # Assigning a Str to a Name (line 48):
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Str to a Name (line 50):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_302871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 26), 'int')
        int_302872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 26), 'int')
        int_302873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 24), 'int')
        defaults = [int_302871, int_302872, int_302873]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BCPPCompiler.__init__', ['verbose', 'dry_run', 'force'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'self' (line 58)
        self_302876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 28), 'self', False)
        # Getting the type of 'verbose' (line 58)
        verbose_302877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 34), 'verbose', False)
        # Getting the type of 'dry_run' (line 58)
        dry_run_302878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 43), 'dry_run', False)
        # Getting the type of 'force' (line 58)
        force_302879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 52), 'force', False)
        # Processing the call keyword arguments (line 58)
        kwargs_302880 = {}
        # Getting the type of 'CCompiler' (line 58)
        CCompiler_302874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'CCompiler', False)
        # Obtaining the member '__init__' of a type (line 58)
        init___302875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), CCompiler_302874, '__init__')
        # Calling __init__(args, kwargs) (line 58)
        init___call_result_302881 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), init___302875, *[self_302876, verbose_302877, dry_run_302878, force_302879], **kwargs_302880)
        
        
        # Assigning a Str to a Attribute (line 64):
        
        # Assigning a Str to a Attribute (line 64):
        str_302882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 18), 'str', 'bcc32.exe')
        # Getting the type of 'self' (line 64)
        self_302883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Setting the type of the member 'cc' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_302883, 'cc', str_302882)
        
        # Assigning a Str to a Attribute (line 65):
        
        # Assigning a Str to a Attribute (line 65):
        str_302884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 22), 'str', 'ilink32.exe')
        # Getting the type of 'self' (line 65)
        self_302885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self')
        # Setting the type of the member 'linker' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_302885, 'linker', str_302884)
        
        # Assigning a Str to a Attribute (line 66):
        
        # Assigning a Str to a Attribute (line 66):
        str_302886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 19), 'str', 'tlib.exe')
        # Getting the type of 'self' (line 66)
        self_302887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self')
        # Setting the type of the member 'lib' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_302887, 'lib', str_302886)
        
        # Assigning a Name to a Attribute (line 68):
        
        # Assigning a Name to a Attribute (line 68):
        # Getting the type of 'None' (line 68)
        None_302888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 34), 'None')
        # Getting the type of 'self' (line 68)
        self_302889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self')
        # Setting the type of the member 'preprocess_options' of a type (line 68)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_302889, 'preprocess_options', None_302888)
        
        # Assigning a List to a Attribute (line 69):
        
        # Assigning a List to a Attribute (line 69):
        
        # Obtaining an instance of the builtin type 'list' (line 69)
        list_302890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 69)
        # Adding element type (line 69)
        str_302891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 32), 'str', '/tWM')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 31), list_302890, str_302891)
        # Adding element type (line 69)
        str_302892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 40), 'str', '/O2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 31), list_302890, str_302892)
        # Adding element type (line 69)
        str_302893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 47), 'str', '/q')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 31), list_302890, str_302893)
        # Adding element type (line 69)
        str_302894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 53), 'str', '/g0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 31), list_302890, str_302894)
        
        # Getting the type of 'self' (line 69)
        self_302895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self')
        # Setting the type of the member 'compile_options' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_302895, 'compile_options', list_302890)
        
        # Assigning a List to a Attribute (line 70):
        
        # Assigning a List to a Attribute (line 70):
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_302896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        str_302897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 38), 'str', '/tWM')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 37), list_302896, str_302897)
        # Adding element type (line 70)
        str_302898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 46), 'str', '/Od')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 37), list_302896, str_302898)
        # Adding element type (line 70)
        str_302899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 53), 'str', '/q')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 37), list_302896, str_302899)
        # Adding element type (line 70)
        str_302900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 59), 'str', '/g0')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 37), list_302896, str_302900)
        
        # Getting the type of 'self' (line 70)
        self_302901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self')
        # Setting the type of the member 'compile_options_debug' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_302901, 'compile_options_debug', list_302896)
        
        # Assigning a List to a Attribute (line 72):
        
        # Assigning a List to a Attribute (line 72):
        
        # Obtaining an instance of the builtin type 'list' (line 72)
        list_302902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 72)
        # Adding element type (line 72)
        str_302903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 31), 'str', '/Tpd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 30), list_302902, str_302903)
        # Adding element type (line 72)
        str_302904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 39), 'str', '/Gn')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 30), list_302902, str_302904)
        # Adding element type (line 72)
        str_302905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 46), 'str', '/q')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 30), list_302902, str_302905)
        # Adding element type (line 72)
        str_302906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 52), 'str', '/x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 30), list_302902, str_302906)
        
        # Getting the type of 'self' (line 72)
        self_302907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self')
        # Setting the type of the member 'ldflags_shared' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_302907, 'ldflags_shared', list_302902)
        
        # Assigning a List to a Attribute (line 73):
        
        # Assigning a List to a Attribute (line 73):
        
        # Obtaining an instance of the builtin type 'list' (line 73)
        list_302908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 73)
        # Adding element type (line 73)
        str_302909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 37), 'str', '/Tpd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 36), list_302908, str_302909)
        # Adding element type (line 73)
        str_302910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 45), 'str', '/Gn')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 36), list_302908, str_302910)
        # Adding element type (line 73)
        str_302911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 52), 'str', '/q')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 36), list_302908, str_302911)
        # Adding element type (line 73)
        str_302912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 58), 'str', '/x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 36), list_302908, str_302912)
        
        # Getting the type of 'self' (line 73)
        self_302913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self')
        # Setting the type of the member 'ldflags_shared_debug' of a type (line 73)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_302913, 'ldflags_shared_debug', list_302908)
        
        # Assigning a List to a Attribute (line 74):
        
        # Assigning a List to a Attribute (line 74):
        
        # Obtaining an instance of the builtin type 'list' (line 74)
        list_302914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 74)
        
        # Getting the type of 'self' (line 74)
        self_302915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member 'ldflags_static' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_302915, 'ldflags_static', list_302914)
        
        # Assigning a List to a Attribute (line 75):
        
        # Assigning a List to a Attribute (line 75):
        
        # Obtaining an instance of the builtin type 'list' (line 75)
        list_302916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 75)
        # Adding element type (line 75)
        str_302917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 28), 'str', '/Gn')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 27), list_302916, str_302917)
        # Adding element type (line 75)
        str_302918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 35), 'str', '/q')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 27), list_302916, str_302918)
        # Adding element type (line 75)
        str_302919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 41), 'str', '/x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 27), list_302916, str_302919)
        
        # Getting the type of 'self' (line 75)
        self_302920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self')
        # Setting the type of the member 'ldflags_exe' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_302920, 'ldflags_exe', list_302916)
        
        # Assigning a List to a Attribute (line 76):
        
        # Assigning a List to a Attribute (line 76):
        
        # Obtaining an instance of the builtin type 'list' (line 76)
        list_302921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 76)
        # Adding element type (line 76)
        str_302922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 34), 'str', '/Gn')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 33), list_302921, str_302922)
        # Adding element type (line 76)
        str_302923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 41), 'str', '/q')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 33), list_302921, str_302923)
        # Adding element type (line 76)
        str_302924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 47), 'str', '/x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 33), list_302921, str_302924)
        # Adding element type (line 76)
        str_302925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 52), 'str', '/r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 33), list_302921, str_302925)
        
        # Getting the type of 'self' (line 76)
        self_302926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self')
        # Setting the type of the member 'ldflags_exe_debug' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_302926, 'ldflags_exe_debug', list_302921)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 82)
        None_302927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'None')
        # Getting the type of 'None' (line 82)
        None_302928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 40), 'None')
        # Getting the type of 'None' (line 82)
        None_302929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 59), 'None')
        int_302930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 71), 'int')
        # Getting the type of 'None' (line 83)
        None_302931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'None')
        # Getting the type of 'None' (line 83)
        None_302932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 51), 'None')
        # Getting the type of 'None' (line 83)
        None_302933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 65), 'None')
        defaults = [None_302927, None_302928, None_302929, int_302930, None_302931, None_302932, None_302933]
        # Create a new context for function 'compile'
        module_type_store = module_type_store.open_function_context('compile', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BCPPCompiler.compile.__dict__.__setitem__('stypy_localization', localization)
        BCPPCompiler.compile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BCPPCompiler.compile.__dict__.__setitem__('stypy_type_store', module_type_store)
        BCPPCompiler.compile.__dict__.__setitem__('stypy_function_name', 'BCPPCompiler.compile')
        BCPPCompiler.compile.__dict__.__setitem__('stypy_param_names_list', ['sources', 'output_dir', 'macros', 'include_dirs', 'debug', 'extra_preargs', 'extra_postargs', 'depends'])
        BCPPCompiler.compile.__dict__.__setitem__('stypy_varargs_param_name', None)
        BCPPCompiler.compile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BCPPCompiler.compile.__dict__.__setitem__('stypy_call_defaults', defaults)
        BCPPCompiler.compile.__dict__.__setitem__('stypy_call_varargs', varargs)
        BCPPCompiler.compile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BCPPCompiler.compile.__dict__.__setitem__('stypy_declared_arg_number', 9)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BCPPCompiler.compile', ['sources', 'output_dir', 'macros', 'include_dirs', 'debug', 'extra_preargs', 'extra_postargs', 'depends'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Tuple (line 85):
        
        # Assigning a Call to a Name:
        
        # Call to _setup_compile(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'output_dir' (line 86)
        output_dir_302936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 36), 'output_dir', False)
        # Getting the type of 'macros' (line 86)
        macros_302937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 48), 'macros', False)
        # Getting the type of 'include_dirs' (line 86)
        include_dirs_302938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 56), 'include_dirs', False)
        # Getting the type of 'sources' (line 86)
        sources_302939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 70), 'sources', False)
        # Getting the type of 'depends' (line 87)
        depends_302940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 36), 'depends', False)
        # Getting the type of 'extra_postargs' (line 87)
        extra_postargs_302941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 45), 'extra_postargs', False)
        # Processing the call keyword arguments (line 86)
        kwargs_302942 = {}
        # Getting the type of 'self' (line 86)
        self_302934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'self', False)
        # Obtaining the member '_setup_compile' of a type (line 86)
        _setup_compile_302935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 16), self_302934, '_setup_compile')
        # Calling _setup_compile(args, kwargs) (line 86)
        _setup_compile_call_result_302943 = invoke(stypy.reporting.localization.Localization(__file__, 86, 16), _setup_compile_302935, *[output_dir_302936, macros_302937, include_dirs_302938, sources_302939, depends_302940, extra_postargs_302941], **kwargs_302942)
        
        # Assigning a type to the variable 'call_assignment_302825' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'call_assignment_302825', _setup_compile_call_result_302943)
        
        # Assigning a Call to a Name (line 85):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_302946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
        # Processing the call keyword arguments
        kwargs_302947 = {}
        # Getting the type of 'call_assignment_302825' (line 85)
        call_assignment_302825_302944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'call_assignment_302825', False)
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___302945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), call_assignment_302825_302944, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_302948 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___302945, *[int_302946], **kwargs_302947)
        
        # Assigning a type to the variable 'call_assignment_302826' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'call_assignment_302826', getitem___call_result_302948)
        
        # Assigning a Name to a Name (line 85):
        # Getting the type of 'call_assignment_302826' (line 85)
        call_assignment_302826_302949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'call_assignment_302826')
        # Assigning a type to the variable 'macros' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'macros', call_assignment_302826_302949)
        
        # Assigning a Call to a Name (line 85):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_302952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
        # Processing the call keyword arguments
        kwargs_302953 = {}
        # Getting the type of 'call_assignment_302825' (line 85)
        call_assignment_302825_302950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'call_assignment_302825', False)
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___302951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), call_assignment_302825_302950, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_302954 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___302951, *[int_302952], **kwargs_302953)
        
        # Assigning a type to the variable 'call_assignment_302827' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'call_assignment_302827', getitem___call_result_302954)
        
        # Assigning a Name to a Name (line 85):
        # Getting the type of 'call_assignment_302827' (line 85)
        call_assignment_302827_302955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'call_assignment_302827')
        # Assigning a type to the variable 'objects' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'objects', call_assignment_302827_302955)
        
        # Assigning a Call to a Name (line 85):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_302958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
        # Processing the call keyword arguments
        kwargs_302959 = {}
        # Getting the type of 'call_assignment_302825' (line 85)
        call_assignment_302825_302956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'call_assignment_302825', False)
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___302957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), call_assignment_302825_302956, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_302960 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___302957, *[int_302958], **kwargs_302959)
        
        # Assigning a type to the variable 'call_assignment_302828' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'call_assignment_302828', getitem___call_result_302960)
        
        # Assigning a Name to a Name (line 85):
        # Getting the type of 'call_assignment_302828' (line 85)
        call_assignment_302828_302961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'call_assignment_302828')
        # Assigning a type to the variable 'extra_postargs' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'extra_postargs', call_assignment_302828_302961)
        
        # Assigning a Call to a Name (line 85):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_302964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
        # Processing the call keyword arguments
        kwargs_302965 = {}
        # Getting the type of 'call_assignment_302825' (line 85)
        call_assignment_302825_302962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'call_assignment_302825', False)
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___302963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), call_assignment_302825_302962, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_302966 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___302963, *[int_302964], **kwargs_302965)
        
        # Assigning a type to the variable 'call_assignment_302829' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'call_assignment_302829', getitem___call_result_302966)
        
        # Assigning a Name to a Name (line 85):
        # Getting the type of 'call_assignment_302829' (line 85)
        call_assignment_302829_302967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'call_assignment_302829')
        # Assigning a type to the variable 'pp_opts' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 41), 'pp_opts', call_assignment_302829_302967)
        
        # Assigning a Call to a Name (line 85):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_302970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'int')
        # Processing the call keyword arguments
        kwargs_302971 = {}
        # Getting the type of 'call_assignment_302825' (line 85)
        call_assignment_302825_302968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'call_assignment_302825', False)
        # Obtaining the member '__getitem__' of a type (line 85)
        getitem___302969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), call_assignment_302825_302968, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_302972 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___302969, *[int_302970], **kwargs_302971)
        
        # Assigning a type to the variable 'call_assignment_302830' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'call_assignment_302830', getitem___call_result_302972)
        
        # Assigning a Name to a Name (line 85):
        # Getting the type of 'call_assignment_302830' (line 85)
        call_assignment_302830_302973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'call_assignment_302830')
        # Assigning a type to the variable 'build' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 50), 'build', call_assignment_302830_302973)
        
        # Assigning a BoolOp to a Name (line 88):
        
        # Assigning a BoolOp to a Name (line 88):
        
        # Evaluating a boolean operation
        # Getting the type of 'extra_preargs' (line 88)
        extra_preargs_302974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'extra_preargs')
        
        # Obtaining an instance of the builtin type 'list' (line 88)
        list_302975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 40), 'list')
        # Adding type elements to the builtin type 'list' instance (line 88)
        
        # Applying the binary operator 'or' (line 88)
        result_or_keyword_302976 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 23), 'or', extra_preargs_302974, list_302975)
        
        # Assigning a type to the variable 'compile_opts' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'compile_opts', result_or_keyword_302976)
        
        # Call to append(...): (line 89)
        # Processing the call arguments (line 89)
        str_302979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 29), 'str', '-c')
        # Processing the call keyword arguments (line 89)
        kwargs_302980 = {}
        # Getting the type of 'compile_opts' (line 89)
        compile_opts_302977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'compile_opts', False)
        # Obtaining the member 'append' of a type (line 89)
        append_302978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), compile_opts_302977, 'append')
        # Calling append(args, kwargs) (line 89)
        append_call_result_302981 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), append_302978, *[str_302979], **kwargs_302980)
        
        
        # Getting the type of 'debug' (line 90)
        debug_302982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 11), 'debug')
        # Testing the type of an if condition (line 90)
        if_condition_302983 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 8), debug_302982)
        # Assigning a type to the variable 'if_condition_302983' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'if_condition_302983', if_condition_302983)
        # SSA begins for if statement (line 90)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'self' (line 91)
        self_302986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 33), 'self', False)
        # Obtaining the member 'compile_options_debug' of a type (line 91)
        compile_options_debug_302987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 33), self_302986, 'compile_options_debug')
        # Processing the call keyword arguments (line 91)
        kwargs_302988 = {}
        # Getting the type of 'compile_opts' (line 91)
        compile_opts_302984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'compile_opts', False)
        # Obtaining the member 'extend' of a type (line 91)
        extend_302985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), compile_opts_302984, 'extend')
        # Calling extend(args, kwargs) (line 91)
        extend_call_result_302989 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), extend_302985, *[compile_options_debug_302987], **kwargs_302988)
        
        # SSA branch for the else part of an if statement (line 90)
        module_type_store.open_ssa_branch('else')
        
        # Call to extend(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'self' (line 93)
        self_302992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 33), 'self', False)
        # Obtaining the member 'compile_options' of a type (line 93)
        compile_options_302993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 33), self_302992, 'compile_options')
        # Processing the call keyword arguments (line 93)
        kwargs_302994 = {}
        # Getting the type of 'compile_opts' (line 93)
        compile_opts_302990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'compile_opts', False)
        # Obtaining the member 'extend' of a type (line 93)
        extend_302991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), compile_opts_302990, 'extend')
        # Calling extend(args, kwargs) (line 93)
        extend_call_result_302995 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), extend_302991, *[compile_options_302993], **kwargs_302994)
        
        # SSA join for if statement (line 90)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'objects' (line 95)
        objects_302996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 19), 'objects')
        # Testing the type of a for loop iterable (line 95)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 95, 8), objects_302996)
        # Getting the type of the for loop variable (line 95)
        for_loop_var_302997 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 95, 8), objects_302996)
        # Assigning a type to the variable 'obj' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'obj', for_loop_var_302997)
        # SSA begins for a for statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 96)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Tuple (line 97):
        
        # Assigning a Subscript to a Name (line 97):
        
        # Obtaining the type of the subscript
        int_302998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 16), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'obj' (line 97)
        obj_302999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 33), 'obj')
        # Getting the type of 'build' (line 97)
        build_303000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 27), 'build')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___303001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 27), build_303000, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_303002 = invoke(stypy.reporting.localization.Localization(__file__, 97, 27), getitem___303001, obj_302999)
        
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___303003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 16), subscript_call_result_303002, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_303004 = invoke(stypy.reporting.localization.Localization(__file__, 97, 16), getitem___303003, int_302998)
        
        # Assigning a type to the variable 'tuple_var_assignment_302831' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'tuple_var_assignment_302831', subscript_call_result_303004)
        
        # Assigning a Subscript to a Name (line 97):
        
        # Obtaining the type of the subscript
        int_303005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 16), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'obj' (line 97)
        obj_303006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 33), 'obj')
        # Getting the type of 'build' (line 97)
        build_303007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 27), 'build')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___303008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 27), build_303007, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_303009 = invoke(stypy.reporting.localization.Localization(__file__, 97, 27), getitem___303008, obj_303006)
        
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___303010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 16), subscript_call_result_303009, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_303011 = invoke(stypy.reporting.localization.Localization(__file__, 97, 16), getitem___303010, int_303005)
        
        # Assigning a type to the variable 'tuple_var_assignment_302832' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'tuple_var_assignment_302832', subscript_call_result_303011)
        
        # Assigning a Name to a Name (line 97):
        # Getting the type of 'tuple_var_assignment_302831' (line 97)
        tuple_var_assignment_302831_303012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'tuple_var_assignment_302831')
        # Assigning a type to the variable 'src' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'src', tuple_var_assignment_302831_303012)
        
        # Assigning a Name to a Name (line 97):
        # Getting the type of 'tuple_var_assignment_302832' (line 97)
        tuple_var_assignment_302832_303013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'tuple_var_assignment_302832')
        # Assigning a type to the variable 'ext' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 21), 'ext', tuple_var_assignment_302832_303013)
        # SSA branch for the except part of a try statement (line 96)
        # SSA branch for the except 'KeyError' branch of a try statement (line 96)
        module_type_store.open_ssa_branch('except')
        # SSA join for try-except statement (line 96)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 101):
        
        # Assigning a Call to a Name (line 101):
        
        # Call to normpath(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'src' (line 101)
        src_303017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 35), 'src', False)
        # Processing the call keyword arguments (line 101)
        kwargs_303018 = {}
        # Getting the type of 'os' (line 101)
        os_303014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 101)
        path_303015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 18), os_303014, 'path')
        # Obtaining the member 'normpath' of a type (line 101)
        normpath_303016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 18), path_303015, 'normpath')
        # Calling normpath(args, kwargs) (line 101)
        normpath_call_result_303019 = invoke(stypy.reporting.localization.Localization(__file__, 101, 18), normpath_303016, *[src_303017], **kwargs_303018)
        
        # Assigning a type to the variable 'src' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'src', normpath_call_result_303019)
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to normpath(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'obj' (line 102)
        obj_303023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 35), 'obj', False)
        # Processing the call keyword arguments (line 102)
        kwargs_303024 = {}
        # Getting the type of 'os' (line 102)
        os_303020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 102)
        path_303021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 18), os_303020, 'path')
        # Obtaining the member 'normpath' of a type (line 102)
        normpath_303022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 18), path_303021, 'normpath')
        # Calling normpath(args, kwargs) (line 102)
        normpath_call_result_303025 = invoke(stypy.reporting.localization.Localization(__file__, 102, 18), normpath_303022, *[obj_303023], **kwargs_303024)
        
        # Assigning a type to the variable 'obj' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'obj', normpath_call_result_303025)
        
        # Call to mkpath(...): (line 105)
        # Processing the call arguments (line 105)
        
        # Call to dirname(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'obj' (line 105)
        obj_303031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 40), 'obj', False)
        # Processing the call keyword arguments (line 105)
        kwargs_303032 = {}
        # Getting the type of 'os' (line 105)
        os_303028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 105)
        path_303029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 24), os_303028, 'path')
        # Obtaining the member 'dirname' of a type (line 105)
        dirname_303030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 24), path_303029, 'dirname')
        # Calling dirname(args, kwargs) (line 105)
        dirname_call_result_303033 = invoke(stypy.reporting.localization.Localization(__file__, 105, 24), dirname_303030, *[obj_303031], **kwargs_303032)
        
        # Processing the call keyword arguments (line 105)
        kwargs_303034 = {}
        # Getting the type of 'self' (line 105)
        self_303026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 105)
        mkpath_303027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), self_303026, 'mkpath')
        # Calling mkpath(args, kwargs) (line 105)
        mkpath_call_result_303035 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), mkpath_303027, *[dirname_call_result_303033], **kwargs_303034)
        
        
        
        # Getting the type of 'ext' (line 107)
        ext_303036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'ext')
        str_303037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 22), 'str', '.res')
        # Applying the binary operator '==' (line 107)
        result_eq_303038 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 15), '==', ext_303036, str_303037)
        
        # Testing the type of an if condition (line 107)
        if_condition_303039 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 12), result_eq_303038)
        # Assigning a type to the variable 'if_condition_303039' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'if_condition_303039', if_condition_303039)
        # SSA begins for if statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 107)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'ext' (line 110)
        ext_303040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'ext')
        str_303041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 22), 'str', '.rc')
        # Applying the binary operator '==' (line 110)
        result_eq_303042 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 15), '==', ext_303040, str_303041)
        
        # Testing the type of an if condition (line 110)
        if_condition_303043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 12), result_eq_303042)
        # Assigning a type to the variable 'if_condition_303043' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'if_condition_303043', if_condition_303043)
        # SSA begins for if statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_303046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        # Adding element type (line 113)
        str_303047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 33), 'str', 'brcc32')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 32), list_303046, str_303047)
        # Adding element type (line 113)
        str_303048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 43), 'str', '-fo')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 32), list_303046, str_303048)
        # Adding element type (line 113)
        # Getting the type of 'obj' (line 113)
        obj_303049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 50), 'obj', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 32), list_303046, obj_303049)
        # Adding element type (line 113)
        # Getting the type of 'src' (line 113)
        src_303050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 55), 'src', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 32), list_303046, src_303050)
        
        # Processing the call keyword arguments (line 113)
        kwargs_303051 = {}
        # Getting the type of 'self' (line 113)
        self_303044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'self', False)
        # Obtaining the member 'spawn' of a type (line 113)
        spawn_303045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 20), self_303044, 'spawn')
        # Calling spawn(args, kwargs) (line 113)
        spawn_call_result_303052 = invoke(stypy.reporting.localization.Localization(__file__, 113, 20), spawn_303045, *[list_303046], **kwargs_303051)
        
        # SSA branch for the except part of a try statement (line 112)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 112)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 114)
        DistutilsExecError_303053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'msg', DistutilsExecError_303053)
        # Getting the type of 'CompileError' (line 115)
        CompileError_303054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 26), 'CompileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 115, 20), CompileError_303054, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'ext' (line 119)
        ext_303055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'ext')
        # Getting the type of 'self' (line 119)
        self_303056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 22), 'self')
        # Obtaining the member '_c_extensions' of a type (line 119)
        _c_extensions_303057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 22), self_303056, '_c_extensions')
        # Applying the binary operator 'in' (line 119)
        result_contains_303058 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 15), 'in', ext_303055, _c_extensions_303057)
        
        # Testing the type of an if condition (line 119)
        if_condition_303059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 12), result_contains_303058)
        # Assigning a type to the variable 'if_condition_303059' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'if_condition_303059', if_condition_303059)
        # SSA begins for if statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 120):
        
        # Assigning a Str to a Name (line 120):
        str_303060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 28), 'str', '')
        # Assigning a type to the variable 'input_opt' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'input_opt', str_303060)
        # SSA branch for the else part of an if statement (line 119)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'ext' (line 121)
        ext_303061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'ext')
        # Getting the type of 'self' (line 121)
        self_303062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 24), 'self')
        # Obtaining the member '_cpp_extensions' of a type (line 121)
        _cpp_extensions_303063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 24), self_303062, '_cpp_extensions')
        # Applying the binary operator 'in' (line 121)
        result_contains_303064 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 17), 'in', ext_303061, _cpp_extensions_303063)
        
        # Testing the type of an if condition (line 121)
        if_condition_303065 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 17), result_contains_303064)
        # Assigning a type to the variable 'if_condition_303065' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'if_condition_303065', if_condition_303065)
        # SSA begins for if statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 122):
        
        # Assigning a Str to a Name (line 122):
        str_303066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 28), 'str', '-P')
        # Assigning a type to the variable 'input_opt' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'input_opt', str_303066)
        # SSA branch for the else part of an if statement (line 121)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 127):
        
        # Assigning a Str to a Name (line 127):
        str_303067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 28), 'str', '')
        # Assigning a type to the variable 'input_opt' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'input_opt', str_303067)
        # SSA join for if statement (line 121)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 119)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 129):
        
        # Assigning a BinOp to a Name (line 129):
        str_303068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 25), 'str', '-o')
        # Getting the type of 'obj' (line 129)
        obj_303069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 32), 'obj')
        # Applying the binary operator '+' (line 129)
        result_add_303070 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 25), '+', str_303068, obj_303069)
        
        # Assigning a type to the variable 'output_opt' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'output_opt', result_add_303070)
        
        
        # SSA begins for try-except statement (line 134)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 135)
        # Processing the call arguments (line 135)
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_303073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        # Getting the type of 'self' (line 135)
        self_303074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'self', False)
        # Obtaining the member 'cc' of a type (line 135)
        cc_303075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 29), self_303074, 'cc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 28), list_303073, cc_303075)
        
        # Getting the type of 'compile_opts' (line 135)
        compile_opts_303076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 40), 'compile_opts', False)
        # Applying the binary operator '+' (line 135)
        result_add_303077 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 28), '+', list_303073, compile_opts_303076)
        
        # Getting the type of 'pp_opts' (line 135)
        pp_opts_303078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 55), 'pp_opts', False)
        # Applying the binary operator '+' (line 135)
        result_add_303079 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 53), '+', result_add_303077, pp_opts_303078)
        
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_303080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        # Adding element type (line 136)
        # Getting the type of 'input_opt' (line 136)
        input_opt_303081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 29), 'input_opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 28), list_303080, input_opt_303081)
        # Adding element type (line 136)
        # Getting the type of 'output_opt' (line 136)
        output_opt_303082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 40), 'output_opt', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 28), list_303080, output_opt_303082)
        
        # Applying the binary operator '+' (line 135)
        result_add_303083 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 63), '+', result_add_303079, list_303080)
        
        # Getting the type of 'extra_postargs' (line 137)
        extra_postargs_303084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 28), 'extra_postargs', False)
        # Applying the binary operator '+' (line 136)
        result_add_303085 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 52), '+', result_add_303083, extra_postargs_303084)
        
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_303086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        # Adding element type (line 137)
        # Getting the type of 'src' (line 137)
        src_303087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 46), 'src', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 45), list_303086, src_303087)
        
        # Applying the binary operator '+' (line 137)
        result_add_303088 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 43), '+', result_add_303085, list_303086)
        
        # Processing the call keyword arguments (line 135)
        kwargs_303089 = {}
        # Getting the type of 'self' (line 135)
        self_303071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 135)
        spawn_303072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 16), self_303071, 'spawn')
        # Calling spawn(args, kwargs) (line 135)
        spawn_call_result_303090 = invoke(stypy.reporting.localization.Localization(__file__, 135, 16), spawn_303072, *[result_add_303088], **kwargs_303089)
        
        # SSA branch for the except part of a try statement (line 134)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 134)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 138)
        DistutilsExecError_303091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'msg', DistutilsExecError_303091)
        # Getting the type of 'CompileError' (line 139)
        CompileError_303092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 22), 'CompileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 139, 16), CompileError_303092, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 134)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'objects' (line 141)
        objects_303093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'objects')
        # Assigning a type to the variable 'stypy_return_type' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'stypy_return_type', objects_303093)
        
        # ################# End of 'compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'compile' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_303094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_303094)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'compile'
        return stypy_return_type_303094


    @norecursion
    def create_static_lib(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 149)
        None_303095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 38), 'None')
        int_303096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 33), 'int')
        # Getting the type of 'None' (line 151)
        None_303097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 39), 'None')
        defaults = [None_303095, int_303096, None_303097]
        # Create a new context for function 'create_static_lib'
        module_type_store = module_type_store.open_function_context('create_static_lib', 146, 4, False)
        # Assigning a type to the variable 'self' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BCPPCompiler.create_static_lib.__dict__.__setitem__('stypy_localization', localization)
        BCPPCompiler.create_static_lib.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BCPPCompiler.create_static_lib.__dict__.__setitem__('stypy_type_store', module_type_store)
        BCPPCompiler.create_static_lib.__dict__.__setitem__('stypy_function_name', 'BCPPCompiler.create_static_lib')
        BCPPCompiler.create_static_lib.__dict__.__setitem__('stypy_param_names_list', ['objects', 'output_libname', 'output_dir', 'debug', 'target_lang'])
        BCPPCompiler.create_static_lib.__dict__.__setitem__('stypy_varargs_param_name', None)
        BCPPCompiler.create_static_lib.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BCPPCompiler.create_static_lib.__dict__.__setitem__('stypy_call_defaults', defaults)
        BCPPCompiler.create_static_lib.__dict__.__setitem__('stypy_call_varargs', varargs)
        BCPPCompiler.create_static_lib.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BCPPCompiler.create_static_lib.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BCPPCompiler.create_static_lib', ['objects', 'output_libname', 'output_dir', 'debug', 'target_lang'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Tuple (line 153):
        
        # Assigning a Call to a Name:
        
        # Call to _fix_object_args(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'objects' (line 153)
        objects_303100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 55), 'objects', False)
        # Getting the type of 'output_dir' (line 153)
        output_dir_303101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 64), 'output_dir', False)
        # Processing the call keyword arguments (line 153)
        kwargs_303102 = {}
        # Getting the type of 'self' (line 153)
        self_303098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 32), 'self', False)
        # Obtaining the member '_fix_object_args' of a type (line 153)
        _fix_object_args_303099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 32), self_303098, '_fix_object_args')
        # Calling _fix_object_args(args, kwargs) (line 153)
        _fix_object_args_call_result_303103 = invoke(stypy.reporting.localization.Localization(__file__, 153, 32), _fix_object_args_303099, *[objects_303100, output_dir_303101], **kwargs_303102)
        
        # Assigning a type to the variable 'call_assignment_302833' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'call_assignment_302833', _fix_object_args_call_result_303103)
        
        # Assigning a Call to a Name (line 153):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_303106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 8), 'int')
        # Processing the call keyword arguments
        kwargs_303107 = {}
        # Getting the type of 'call_assignment_302833' (line 153)
        call_assignment_302833_303104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'call_assignment_302833', False)
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___303105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), call_assignment_302833_303104, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_303108 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303105, *[int_303106], **kwargs_303107)
        
        # Assigning a type to the variable 'call_assignment_302834' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'call_assignment_302834', getitem___call_result_303108)
        
        # Assigning a Name to a Name (line 153):
        # Getting the type of 'call_assignment_302834' (line 153)
        call_assignment_302834_303109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'call_assignment_302834')
        # Assigning a type to the variable 'objects' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 9), 'objects', call_assignment_302834_303109)
        
        # Assigning a Call to a Name (line 153):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_303112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 8), 'int')
        # Processing the call keyword arguments
        kwargs_303113 = {}
        # Getting the type of 'call_assignment_302833' (line 153)
        call_assignment_302833_303110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'call_assignment_302833', False)
        # Obtaining the member '__getitem__' of a type (line 153)
        getitem___303111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), call_assignment_302833_303110, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_303114 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303111, *[int_303112], **kwargs_303113)
        
        # Assigning a type to the variable 'call_assignment_302835' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'call_assignment_302835', getitem___call_result_303114)
        
        # Assigning a Name to a Name (line 153):
        # Getting the type of 'call_assignment_302835' (line 153)
        call_assignment_302835_303115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'call_assignment_302835')
        # Assigning a type to the variable 'output_dir' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 18), 'output_dir', call_assignment_302835_303115)
        
        # Assigning a Call to a Name (line 154):
        
        # Assigning a Call to a Name (line 154):
        
        # Call to library_filename(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'output_libname' (line 155)
        output_libname_303118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 35), 'output_libname', False)
        # Processing the call keyword arguments (line 155)
        # Getting the type of 'output_dir' (line 155)
        output_dir_303119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 62), 'output_dir', False)
        keyword_303120 = output_dir_303119
        kwargs_303121 = {'output_dir': keyword_303120}
        # Getting the type of 'self' (line 155)
        self_303116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'self', False)
        # Obtaining the member 'library_filename' of a type (line 155)
        library_filename_303117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), self_303116, 'library_filename')
        # Calling library_filename(args, kwargs) (line 155)
        library_filename_call_result_303122 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), library_filename_303117, *[output_libname_303118], **kwargs_303121)
        
        # Assigning a type to the variable 'output_filename' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'output_filename', library_filename_call_result_303122)
        
        
        # Call to _need_link(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'objects' (line 157)
        objects_303125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 28), 'objects', False)
        # Getting the type of 'output_filename' (line 157)
        output_filename_303126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 37), 'output_filename', False)
        # Processing the call keyword arguments (line 157)
        kwargs_303127 = {}
        # Getting the type of 'self' (line 157)
        self_303123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 11), 'self', False)
        # Obtaining the member '_need_link' of a type (line 157)
        _need_link_303124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 11), self_303123, '_need_link')
        # Calling _need_link(args, kwargs) (line 157)
        _need_link_call_result_303128 = invoke(stypy.reporting.localization.Localization(__file__, 157, 11), _need_link_303124, *[objects_303125, output_filename_303126], **kwargs_303127)
        
        # Testing the type of an if condition (line 157)
        if_condition_303129 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 8), _need_link_call_result_303128)
        # Assigning a type to the variable 'if_condition_303129' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'if_condition_303129', if_condition_303129)
        # SSA begins for if statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 158):
        
        # Assigning a BinOp to a Name (line 158):
        
        # Obtaining an instance of the builtin type 'list' (line 158)
        list_303130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 158)
        # Adding element type (line 158)
        # Getting the type of 'output_filename' (line 158)
        output_filename_303131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), 'output_filename')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 23), list_303130, output_filename_303131)
        # Adding element type (line 158)
        str_303132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 41), 'str', '/u')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 23), list_303130, str_303132)
        
        # Getting the type of 'objects' (line 158)
        objects_303133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 49), 'objects')
        # Applying the binary operator '+' (line 158)
        result_add_303134 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 23), '+', list_303130, objects_303133)
        
        # Assigning a type to the variable 'lib_args' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'lib_args', result_add_303134)
        
        # Getting the type of 'debug' (line 159)
        debug_303135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), 'debug')
        # Testing the type of an if condition (line 159)
        if_condition_303136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 12), debug_303135)
        # Assigning a type to the variable 'if_condition_303136' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'if_condition_303136', if_condition_303136)
        # SSA begins for if statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        pass
        # SSA join for if statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 162)
        # Processing the call arguments (line 162)
        
        # Obtaining an instance of the builtin type 'list' (line 162)
        list_303139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 162)
        # Adding element type (line 162)
        # Getting the type of 'self' (line 162)
        self_303140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 29), 'self', False)
        # Obtaining the member 'lib' of a type (line 162)
        lib_303141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 29), self_303140, 'lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 28), list_303139, lib_303141)
        
        # Getting the type of 'lib_args' (line 162)
        lib_args_303142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 41), 'lib_args', False)
        # Applying the binary operator '+' (line 162)
        result_add_303143 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 28), '+', list_303139, lib_args_303142)
        
        # Processing the call keyword arguments (line 162)
        kwargs_303144 = {}
        # Getting the type of 'self' (line 162)
        self_303137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 162)
        spawn_303138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 16), self_303137, 'spawn')
        # Calling spawn(args, kwargs) (line 162)
        spawn_call_result_303145 = invoke(stypy.reporting.localization.Localization(__file__, 162, 16), spawn_303138, *[result_add_303143], **kwargs_303144)
        
        # SSA branch for the except part of a try statement (line 161)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 161)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 163)
        DistutilsExecError_303146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'msg', DistutilsExecError_303146)
        # Getting the type of 'LibError' (line 164)
        LibError_303147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 22), 'LibError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 164, 16), LibError_303147, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 161)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 157)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug(...): (line 166)
        # Processing the call arguments (line 166)
        str_303150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 22), 'str', 'skipping %s (up-to-date)')
        # Getting the type of 'output_filename' (line 166)
        output_filename_303151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 50), 'output_filename', False)
        # Processing the call keyword arguments (line 166)
        kwargs_303152 = {}
        # Getting the type of 'log' (line 166)
        log_303148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 166)
        debug_303149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), log_303148, 'debug')
        # Calling debug(args, kwargs) (line 166)
        debug_call_result_303153 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), debug_303149, *[str_303150, output_filename_303151], **kwargs_303152)
        
        # SSA join for if statement (line 157)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'create_static_lib(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_static_lib' in the type store
        # Getting the type of 'stypy_return_type' (line 146)
        stypy_return_type_303154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_303154)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_static_lib'
        return stypy_return_type_303154


    @norecursion
    def link(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 175)
        None_303155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'None')
        # Getting the type of 'None' (line 176)
        None_303156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'None')
        # Getting the type of 'None' (line 177)
        None_303157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 27), 'None')
        # Getting the type of 'None' (line 178)
        None_303158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 35), 'None')
        # Getting the type of 'None' (line 179)
        None_303159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 29), 'None')
        int_303160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 20), 'int')
        # Getting the type of 'None' (line 181)
        None_303161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 28), 'None')
        # Getting the type of 'None' (line 182)
        None_303162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 29), 'None')
        # Getting the type of 'None' (line 183)
        None_303163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 25), 'None')
        # Getting the type of 'None' (line 184)
        None_303164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 26), 'None')
        defaults = [None_303155, None_303156, None_303157, None_303158, None_303159, int_303160, None_303161, None_303162, None_303163, None_303164]
        # Create a new context for function 'link'
        module_type_store = module_type_store.open_function_context('link', 171, 4, False)
        # Assigning a type to the variable 'self' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BCPPCompiler.link.__dict__.__setitem__('stypy_localization', localization)
        BCPPCompiler.link.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BCPPCompiler.link.__dict__.__setitem__('stypy_type_store', module_type_store)
        BCPPCompiler.link.__dict__.__setitem__('stypy_function_name', 'BCPPCompiler.link')
        BCPPCompiler.link.__dict__.__setitem__('stypy_param_names_list', ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'])
        BCPPCompiler.link.__dict__.__setitem__('stypy_varargs_param_name', None)
        BCPPCompiler.link.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BCPPCompiler.link.__dict__.__setitem__('stypy_call_defaults', defaults)
        BCPPCompiler.link.__dict__.__setitem__('stypy_call_varargs', varargs)
        BCPPCompiler.link.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BCPPCompiler.link.__dict__.__setitem__('stypy_declared_arg_number', 14)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BCPPCompiler.link', ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Tuple (line 189):
        
        # Assigning a Call to a Name:
        
        # Call to _fix_object_args(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'objects' (line 189)
        objects_303167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 55), 'objects', False)
        # Getting the type of 'output_dir' (line 189)
        output_dir_303168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 64), 'output_dir', False)
        # Processing the call keyword arguments (line 189)
        kwargs_303169 = {}
        # Getting the type of 'self' (line 189)
        self_303165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 32), 'self', False)
        # Obtaining the member '_fix_object_args' of a type (line 189)
        _fix_object_args_303166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 32), self_303165, '_fix_object_args')
        # Calling _fix_object_args(args, kwargs) (line 189)
        _fix_object_args_call_result_303170 = invoke(stypy.reporting.localization.Localization(__file__, 189, 32), _fix_object_args_303166, *[objects_303167, output_dir_303168], **kwargs_303169)
        
        # Assigning a type to the variable 'call_assignment_302836' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'call_assignment_302836', _fix_object_args_call_result_303170)
        
        # Assigning a Call to a Name (line 189):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_303173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 8), 'int')
        # Processing the call keyword arguments
        kwargs_303174 = {}
        # Getting the type of 'call_assignment_302836' (line 189)
        call_assignment_302836_303171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'call_assignment_302836', False)
        # Obtaining the member '__getitem__' of a type (line 189)
        getitem___303172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), call_assignment_302836_303171, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_303175 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303172, *[int_303173], **kwargs_303174)
        
        # Assigning a type to the variable 'call_assignment_302837' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'call_assignment_302837', getitem___call_result_303175)
        
        # Assigning a Name to a Name (line 189):
        # Getting the type of 'call_assignment_302837' (line 189)
        call_assignment_302837_303176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'call_assignment_302837')
        # Assigning a type to the variable 'objects' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 9), 'objects', call_assignment_302837_303176)
        
        # Assigning a Call to a Name (line 189):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_303179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 8), 'int')
        # Processing the call keyword arguments
        kwargs_303180 = {}
        # Getting the type of 'call_assignment_302836' (line 189)
        call_assignment_302836_303177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'call_assignment_302836', False)
        # Obtaining the member '__getitem__' of a type (line 189)
        getitem___303178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), call_assignment_302836_303177, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_303181 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303178, *[int_303179], **kwargs_303180)
        
        # Assigning a type to the variable 'call_assignment_302838' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'call_assignment_302838', getitem___call_result_303181)
        
        # Assigning a Name to a Name (line 189):
        # Getting the type of 'call_assignment_302838' (line 189)
        call_assignment_302838_303182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'call_assignment_302838')
        # Assigning a type to the variable 'output_dir' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 18), 'output_dir', call_assignment_302838_303182)
        
        # Assigning a Call to a Tuple (line 190):
        
        # Assigning a Call to a Name:
        
        # Call to _fix_lib_args(...): (line 191)
        # Processing the call arguments (line 191)
        # Getting the type of 'libraries' (line 191)
        libraries_303185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 32), 'libraries', False)
        # Getting the type of 'library_dirs' (line 191)
        library_dirs_303186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 43), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 191)
        runtime_library_dirs_303187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 57), 'runtime_library_dirs', False)
        # Processing the call keyword arguments (line 191)
        kwargs_303188 = {}
        # Getting the type of 'self' (line 191)
        self_303183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'self', False)
        # Obtaining the member '_fix_lib_args' of a type (line 191)
        _fix_lib_args_303184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 12), self_303183, '_fix_lib_args')
        # Calling _fix_lib_args(args, kwargs) (line 191)
        _fix_lib_args_call_result_303189 = invoke(stypy.reporting.localization.Localization(__file__, 191, 12), _fix_lib_args_303184, *[libraries_303185, library_dirs_303186, runtime_library_dirs_303187], **kwargs_303188)
        
        # Assigning a type to the variable 'call_assignment_302839' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_302839', _fix_lib_args_call_result_303189)
        
        # Assigning a Call to a Name (line 190):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_303192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 8), 'int')
        # Processing the call keyword arguments
        kwargs_303193 = {}
        # Getting the type of 'call_assignment_302839' (line 190)
        call_assignment_302839_303190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_302839', False)
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___303191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), call_assignment_302839_303190, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_303194 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303191, *[int_303192], **kwargs_303193)
        
        # Assigning a type to the variable 'call_assignment_302840' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_302840', getitem___call_result_303194)
        
        # Assigning a Name to a Name (line 190):
        # Getting the type of 'call_assignment_302840' (line 190)
        call_assignment_302840_303195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_302840')
        # Assigning a type to the variable 'libraries' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 9), 'libraries', call_assignment_302840_303195)
        
        # Assigning a Call to a Name (line 190):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_303198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 8), 'int')
        # Processing the call keyword arguments
        kwargs_303199 = {}
        # Getting the type of 'call_assignment_302839' (line 190)
        call_assignment_302839_303196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_302839', False)
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___303197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), call_assignment_302839_303196, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_303200 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303197, *[int_303198], **kwargs_303199)
        
        # Assigning a type to the variable 'call_assignment_302841' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_302841', getitem___call_result_303200)
        
        # Assigning a Name to a Name (line 190):
        # Getting the type of 'call_assignment_302841' (line 190)
        call_assignment_302841_303201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_302841')
        # Assigning a type to the variable 'library_dirs' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 20), 'library_dirs', call_assignment_302841_303201)
        
        # Assigning a Call to a Name (line 190):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_303204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 8), 'int')
        # Processing the call keyword arguments
        kwargs_303205 = {}
        # Getting the type of 'call_assignment_302839' (line 190)
        call_assignment_302839_303202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_302839', False)
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___303203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), call_assignment_302839_303202, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_303206 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303203, *[int_303204], **kwargs_303205)
        
        # Assigning a type to the variable 'call_assignment_302842' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_302842', getitem___call_result_303206)
        
        # Assigning a Name to a Name (line 190):
        # Getting the type of 'call_assignment_302842' (line 190)
        call_assignment_302842_303207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'call_assignment_302842')
        # Assigning a type to the variable 'runtime_library_dirs' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 34), 'runtime_library_dirs', call_assignment_302842_303207)
        
        # Getting the type of 'runtime_library_dirs' (line 193)
        runtime_library_dirs_303208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'runtime_library_dirs')
        # Testing the type of an if condition (line 193)
        if_condition_303209 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 8), runtime_library_dirs_303208)
        # Assigning a type to the variable 'if_condition_303209' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'if_condition_303209', if_condition_303209)
        # SSA begins for if statement (line 193)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 194)
        # Processing the call arguments (line 194)
        str_303212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 21), 'str', "I don't know what to do with 'runtime_library_dirs': %s")
        
        # Call to str(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'runtime_library_dirs' (line 195)
        runtime_library_dirs_303214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 25), 'runtime_library_dirs', False)
        # Processing the call keyword arguments (line 195)
        kwargs_303215 = {}
        # Getting the type of 'str' (line 195)
        str_303213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 21), 'str', False)
        # Calling str(args, kwargs) (line 195)
        str_call_result_303216 = invoke(stypy.reporting.localization.Localization(__file__, 195, 21), str_303213, *[runtime_library_dirs_303214], **kwargs_303215)
        
        # Processing the call keyword arguments (line 194)
        kwargs_303217 = {}
        # Getting the type of 'log' (line 194)
        log_303210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'log', False)
        # Obtaining the member 'warn' of a type (line 194)
        warn_303211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), log_303210, 'warn')
        # Calling warn(args, kwargs) (line 194)
        warn_call_result_303218 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), warn_303211, *[str_303212, str_call_result_303216], **kwargs_303217)
        
        # SSA join for if statement (line 193)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 197)
        # Getting the type of 'output_dir' (line 197)
        output_dir_303219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'output_dir')
        # Getting the type of 'None' (line 197)
        None_303220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 29), 'None')
        
        (may_be_303221, more_types_in_union_303222) = may_not_be_none(output_dir_303219, None_303220)

        if may_be_303221:

            if more_types_in_union_303222:
                # Runtime conditional SSA (line 197)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 198):
            
            # Assigning a Call to a Name (line 198):
            
            # Call to join(...): (line 198)
            # Processing the call arguments (line 198)
            # Getting the type of 'output_dir' (line 198)
            output_dir_303226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 44), 'output_dir', False)
            # Getting the type of 'output_filename' (line 198)
            output_filename_303227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 56), 'output_filename', False)
            # Processing the call keyword arguments (line 198)
            kwargs_303228 = {}
            # Getting the type of 'os' (line 198)
            os_303223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 30), 'os', False)
            # Obtaining the member 'path' of a type (line 198)
            path_303224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 30), os_303223, 'path')
            # Obtaining the member 'join' of a type (line 198)
            join_303225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 30), path_303224, 'join')
            # Calling join(args, kwargs) (line 198)
            join_call_result_303229 = invoke(stypy.reporting.localization.Localization(__file__, 198, 30), join_303225, *[output_dir_303226, output_filename_303227], **kwargs_303228)
            
            # Assigning a type to the variable 'output_filename' (line 198)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'output_filename', join_call_result_303229)

            if more_types_in_union_303222:
                # SSA join for if statement (line 197)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to _need_link(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'objects' (line 200)
        objects_303232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 28), 'objects', False)
        # Getting the type of 'output_filename' (line 200)
        output_filename_303233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 37), 'output_filename', False)
        # Processing the call keyword arguments (line 200)
        kwargs_303234 = {}
        # Getting the type of 'self' (line 200)
        self_303230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 11), 'self', False)
        # Obtaining the member '_need_link' of a type (line 200)
        _need_link_303231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 11), self_303230, '_need_link')
        # Calling _need_link(args, kwargs) (line 200)
        _need_link_call_result_303235 = invoke(stypy.reporting.localization.Localization(__file__, 200, 11), _need_link_303231, *[objects_303232, output_filename_303233], **kwargs_303234)
        
        # Testing the type of an if condition (line 200)
        if_condition_303236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 8), _need_link_call_result_303235)
        # Assigning a type to the variable 'if_condition_303236' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'if_condition_303236', if_condition_303236)
        # SSA begins for if statement (line 200)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'target_desc' (line 203)
        target_desc_303237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'target_desc')
        # Getting the type of 'CCompiler' (line 203)
        CCompiler_303238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 30), 'CCompiler')
        # Obtaining the member 'EXECUTABLE' of a type (line 203)
        EXECUTABLE_303239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 30), CCompiler_303238, 'EXECUTABLE')
        # Applying the binary operator '==' (line 203)
        result_eq_303240 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 15), '==', target_desc_303237, EXECUTABLE_303239)
        
        # Testing the type of an if condition (line 203)
        if_condition_303241 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 12), result_eq_303240)
        # Assigning a type to the variable 'if_condition_303241' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'if_condition_303241', if_condition_303241)
        # SSA begins for if statement (line 203)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 204):
        
        # Assigning a Str to a Name (line 204):
        str_303242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 30), 'str', 'c0w32')
        # Assigning a type to the variable 'startup_obj' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'startup_obj', str_303242)
        
        # Getting the type of 'debug' (line 205)
        debug_303243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 19), 'debug')
        # Testing the type of an if condition (line 205)
        if_condition_303244 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 16), debug_303243)
        # Assigning a type to the variable 'if_condition_303244' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 16), 'if_condition_303244', if_condition_303244)
        # SSA begins for if statement (line 205)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 206):
        
        # Assigning a Subscript to a Name (line 206):
        
        # Obtaining the type of the subscript
        slice_303245 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 206, 30), None, None, None)
        # Getting the type of 'self' (line 206)
        self_303246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 30), 'self')
        # Obtaining the member 'ldflags_exe_debug' of a type (line 206)
        ldflags_exe_debug_303247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 30), self_303246, 'ldflags_exe_debug')
        # Obtaining the member '__getitem__' of a type (line 206)
        getitem___303248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 30), ldflags_exe_debug_303247, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
        subscript_call_result_303249 = invoke(stypy.reporting.localization.Localization(__file__, 206, 30), getitem___303248, slice_303245)
        
        # Assigning a type to the variable 'ld_args' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 20), 'ld_args', subscript_call_result_303249)
        # SSA branch for the else part of an if statement (line 205)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 208):
        
        # Assigning a Subscript to a Name (line 208):
        
        # Obtaining the type of the subscript
        slice_303250 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 208, 30), None, None, None)
        # Getting the type of 'self' (line 208)
        self_303251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 30), 'self')
        # Obtaining the member 'ldflags_exe' of a type (line 208)
        ldflags_exe_303252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 30), self_303251, 'ldflags_exe')
        # Obtaining the member '__getitem__' of a type (line 208)
        getitem___303253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 30), ldflags_exe_303252, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 208)
        subscript_call_result_303254 = invoke(stypy.reporting.localization.Localization(__file__, 208, 30), getitem___303253, slice_303250)
        
        # Assigning a type to the variable 'ld_args' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 20), 'ld_args', subscript_call_result_303254)
        # SSA join for if statement (line 205)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 203)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 210):
        
        # Assigning a Str to a Name (line 210):
        str_303255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 30), 'str', 'c0d32')
        # Assigning a type to the variable 'startup_obj' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'startup_obj', str_303255)
        
        # Getting the type of 'debug' (line 211)
        debug_303256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 19), 'debug')
        # Testing the type of an if condition (line 211)
        if_condition_303257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 16), debug_303256)
        # Assigning a type to the variable 'if_condition_303257' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'if_condition_303257', if_condition_303257)
        # SSA begins for if statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 212):
        
        # Assigning a Subscript to a Name (line 212):
        
        # Obtaining the type of the subscript
        slice_303258 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 212, 30), None, None, None)
        # Getting the type of 'self' (line 212)
        self_303259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 30), 'self')
        # Obtaining the member 'ldflags_shared_debug' of a type (line 212)
        ldflags_shared_debug_303260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 30), self_303259, 'ldflags_shared_debug')
        # Obtaining the member '__getitem__' of a type (line 212)
        getitem___303261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 30), ldflags_shared_debug_303260, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 212)
        subscript_call_result_303262 = invoke(stypy.reporting.localization.Localization(__file__, 212, 30), getitem___303261, slice_303258)
        
        # Assigning a type to the variable 'ld_args' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'ld_args', subscript_call_result_303262)
        # SSA branch for the else part of an if statement (line 211)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 214):
        
        # Assigning a Subscript to a Name (line 214):
        
        # Obtaining the type of the subscript
        slice_303263 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 214, 30), None, None, None)
        # Getting the type of 'self' (line 214)
        self_303264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 30), 'self')
        # Obtaining the member 'ldflags_shared' of a type (line 214)
        ldflags_shared_303265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 30), self_303264, 'ldflags_shared')
        # Obtaining the member '__getitem__' of a type (line 214)
        getitem___303266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 30), ldflags_shared_303265, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 214)
        subscript_call_result_303267 = invoke(stypy.reporting.localization.Localization(__file__, 214, 30), getitem___303266, slice_303263)
        
        # Assigning a type to the variable 'ld_args' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'ld_args', subscript_call_result_303267)
        # SSA join for if statement (line 211)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 203)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 218)
        # Getting the type of 'export_symbols' (line 218)
        export_symbols_303268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'export_symbols')
        # Getting the type of 'None' (line 218)
        None_303269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 33), 'None')
        
        (may_be_303270, more_types_in_union_303271) = may_be_none(export_symbols_303268, None_303269)

        if may_be_303270:

            if more_types_in_union_303271:
                # Runtime conditional SSA (line 218)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 219):
            
            # Assigning a Str to a Name (line 219):
            str_303272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 27), 'str', '')
            # Assigning a type to the variable 'def_file' (line 219)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'def_file', str_303272)

            if more_types_in_union_303271:
                # Runtime conditional SSA for else branch (line 218)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_303270) or more_types_in_union_303271):
            
            # Assigning a Call to a Tuple (line 221):
            
            # Assigning a Call to a Name:
            
            # Call to split(...): (line 221)
            # Processing the call arguments (line 221)
            # Getting the type of 'output_filename' (line 221)
            output_filename_303276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 44), 'output_filename', False)
            # Processing the call keyword arguments (line 221)
            kwargs_303277 = {}
            # Getting the type of 'os' (line 221)
            os_303273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 29), 'os', False)
            # Obtaining the member 'path' of a type (line 221)
            path_303274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 29), os_303273, 'path')
            # Obtaining the member 'split' of a type (line 221)
            split_303275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 29), path_303274, 'split')
            # Calling split(args, kwargs) (line 221)
            split_call_result_303278 = invoke(stypy.reporting.localization.Localization(__file__, 221, 29), split_303275, *[output_filename_303276], **kwargs_303277)
            
            # Assigning a type to the variable 'call_assignment_302843' (line 221)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'call_assignment_302843', split_call_result_303278)
            
            # Assigning a Call to a Name (line 221):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_303281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 16), 'int')
            # Processing the call keyword arguments
            kwargs_303282 = {}
            # Getting the type of 'call_assignment_302843' (line 221)
            call_assignment_302843_303279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'call_assignment_302843', False)
            # Obtaining the member '__getitem__' of a type (line 221)
            getitem___303280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), call_assignment_302843_303279, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_303283 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303280, *[int_303281], **kwargs_303282)
            
            # Assigning a type to the variable 'call_assignment_302844' (line 221)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'call_assignment_302844', getitem___call_result_303283)
            
            # Assigning a Name to a Name (line 221):
            # Getting the type of 'call_assignment_302844' (line 221)
            call_assignment_302844_303284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'call_assignment_302844')
            # Assigning a type to the variable 'head' (line 221)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'head', call_assignment_302844_303284)
            
            # Assigning a Call to a Name (line 221):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_303287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 16), 'int')
            # Processing the call keyword arguments
            kwargs_303288 = {}
            # Getting the type of 'call_assignment_302843' (line 221)
            call_assignment_302843_303285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'call_assignment_302843', False)
            # Obtaining the member '__getitem__' of a type (line 221)
            getitem___303286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), call_assignment_302843_303285, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_303289 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303286, *[int_303287], **kwargs_303288)
            
            # Assigning a type to the variable 'call_assignment_302845' (line 221)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'call_assignment_302845', getitem___call_result_303289)
            
            # Assigning a Name to a Name (line 221):
            # Getting the type of 'call_assignment_302845' (line 221)
            call_assignment_302845_303290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'call_assignment_302845')
            # Assigning a type to the variable 'tail' (line 221)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 22), 'tail', call_assignment_302845_303290)
            
            # Assigning a Call to a Tuple (line 222):
            
            # Assigning a Call to a Name:
            
            # Call to splitext(...): (line 222)
            # Processing the call arguments (line 222)
            # Getting the type of 'tail' (line 222)
            tail_303294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 49), 'tail', False)
            # Processing the call keyword arguments (line 222)
            kwargs_303295 = {}
            # Getting the type of 'os' (line 222)
            os_303291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 31), 'os', False)
            # Obtaining the member 'path' of a type (line 222)
            path_303292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 31), os_303291, 'path')
            # Obtaining the member 'splitext' of a type (line 222)
            splitext_303293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 31), path_303292, 'splitext')
            # Calling splitext(args, kwargs) (line 222)
            splitext_call_result_303296 = invoke(stypy.reporting.localization.Localization(__file__, 222, 31), splitext_303293, *[tail_303294], **kwargs_303295)
            
            # Assigning a type to the variable 'call_assignment_302846' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'call_assignment_302846', splitext_call_result_303296)
            
            # Assigning a Call to a Name (line 222):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_303299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 16), 'int')
            # Processing the call keyword arguments
            kwargs_303300 = {}
            # Getting the type of 'call_assignment_302846' (line 222)
            call_assignment_302846_303297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'call_assignment_302846', False)
            # Obtaining the member '__getitem__' of a type (line 222)
            getitem___303298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 16), call_assignment_302846_303297, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_303301 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303298, *[int_303299], **kwargs_303300)
            
            # Assigning a type to the variable 'call_assignment_302847' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'call_assignment_302847', getitem___call_result_303301)
            
            # Assigning a Name to a Name (line 222):
            # Getting the type of 'call_assignment_302847' (line 222)
            call_assignment_302847_303302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'call_assignment_302847')
            # Assigning a type to the variable 'modname' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'modname', call_assignment_302847_303302)
            
            # Assigning a Call to a Name (line 222):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_303305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 16), 'int')
            # Processing the call keyword arguments
            kwargs_303306 = {}
            # Getting the type of 'call_assignment_302846' (line 222)
            call_assignment_302846_303303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'call_assignment_302846', False)
            # Obtaining the member '__getitem__' of a type (line 222)
            getitem___303304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 16), call_assignment_302846_303303, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_303307 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303304, *[int_303305], **kwargs_303306)
            
            # Assigning a type to the variable 'call_assignment_302848' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'call_assignment_302848', getitem___call_result_303307)
            
            # Assigning a Name to a Name (line 222):
            # Getting the type of 'call_assignment_302848' (line 222)
            call_assignment_302848_303308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'call_assignment_302848')
            # Assigning a type to the variable 'ext' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 25), 'ext', call_assignment_302848_303308)
            
            # Assigning a Call to a Name (line 223):
            
            # Assigning a Call to a Name (line 223):
            
            # Call to dirname(...): (line 223)
            # Processing the call arguments (line 223)
            
            # Obtaining the type of the subscript
            int_303312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 51), 'int')
            # Getting the type of 'objects' (line 223)
            objects_303313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 43), 'objects', False)
            # Obtaining the member '__getitem__' of a type (line 223)
            getitem___303314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 43), objects_303313, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 223)
            subscript_call_result_303315 = invoke(stypy.reporting.localization.Localization(__file__, 223, 43), getitem___303314, int_303312)
            
            # Processing the call keyword arguments (line 223)
            kwargs_303316 = {}
            # Getting the type of 'os' (line 223)
            os_303309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 27), 'os', False)
            # Obtaining the member 'path' of a type (line 223)
            path_303310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 27), os_303309, 'path')
            # Obtaining the member 'dirname' of a type (line 223)
            dirname_303311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 27), path_303310, 'dirname')
            # Calling dirname(args, kwargs) (line 223)
            dirname_call_result_303317 = invoke(stypy.reporting.localization.Localization(__file__, 223, 27), dirname_303311, *[subscript_call_result_303315], **kwargs_303316)
            
            # Assigning a type to the variable 'temp_dir' (line 223)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), 'temp_dir', dirname_call_result_303317)
            
            # Assigning a Call to a Name (line 224):
            
            # Assigning a Call to a Name (line 224):
            
            # Call to join(...): (line 224)
            # Processing the call arguments (line 224)
            # Getting the type of 'temp_dir' (line 224)
            temp_dir_303321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 41), 'temp_dir', False)
            str_303322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 51), 'str', '%s.def')
            # Getting the type of 'modname' (line 224)
            modname_303323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 62), 'modname', False)
            # Applying the binary operator '%' (line 224)
            result_mod_303324 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 51), '%', str_303322, modname_303323)
            
            # Processing the call keyword arguments (line 224)
            kwargs_303325 = {}
            # Getting the type of 'os' (line 224)
            os_303318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 27), 'os', False)
            # Obtaining the member 'path' of a type (line 224)
            path_303319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 27), os_303318, 'path')
            # Obtaining the member 'join' of a type (line 224)
            join_303320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 27), path_303319, 'join')
            # Calling join(args, kwargs) (line 224)
            join_call_result_303326 = invoke(stypy.reporting.localization.Localization(__file__, 224, 27), join_303320, *[temp_dir_303321, result_mod_303324], **kwargs_303325)
            
            # Assigning a type to the variable 'def_file' (line 224)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'def_file', join_call_result_303326)
            
            # Assigning a List to a Name (line 225):
            
            # Assigning a List to a Name (line 225):
            
            # Obtaining an instance of the builtin type 'list' (line 225)
            list_303327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 27), 'list')
            # Adding type elements to the builtin type 'list' instance (line 225)
            # Adding element type (line 225)
            str_303328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 28), 'str', 'EXPORTS')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 27), list_303327, str_303328)
            
            # Assigning a type to the variable 'contents' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'contents', list_303327)
            
            
            # Evaluating a boolean operation
            # Getting the type of 'export_symbols' (line 226)
            export_symbols_303329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 28), 'export_symbols')
            
            # Obtaining an instance of the builtin type 'list' (line 226)
            list_303330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 46), 'list')
            # Adding type elements to the builtin type 'list' instance (line 226)
            
            # Applying the binary operator 'or' (line 226)
            result_or_keyword_303331 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 28), 'or', export_symbols_303329, list_303330)
            
            # Testing the type of a for loop iterable (line 226)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 226, 16), result_or_keyword_303331)
            # Getting the type of the for loop variable (line 226)
            for_loop_var_303332 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 226, 16), result_or_keyword_303331)
            # Assigning a type to the variable 'sym' (line 226)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'sym', for_loop_var_303332)
            # SSA begins for a for statement (line 226)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 227)
            # Processing the call arguments (line 227)
            str_303335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 36), 'str', '  %s=_%s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 227)
            tuple_303336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 50), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 227)
            # Adding element type (line 227)
            # Getting the type of 'sym' (line 227)
            sym_303337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 50), 'sym', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 50), tuple_303336, sym_303337)
            # Adding element type (line 227)
            # Getting the type of 'sym' (line 227)
            sym_303338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 55), 'sym', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 50), tuple_303336, sym_303338)
            
            # Applying the binary operator '%' (line 227)
            result_mod_303339 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 36), '%', str_303335, tuple_303336)
            
            # Processing the call keyword arguments (line 227)
            kwargs_303340 = {}
            # Getting the type of 'contents' (line 227)
            contents_303333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'contents', False)
            # Obtaining the member 'append' of a type (line 227)
            append_303334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 20), contents_303333, 'append')
            # Calling append(args, kwargs) (line 227)
            append_call_result_303341 = invoke(stypy.reporting.localization.Localization(__file__, 227, 20), append_303334, *[result_mod_303339], **kwargs_303340)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to execute(...): (line 228)
            # Processing the call arguments (line 228)
            # Getting the type of 'write_file' (line 228)
            write_file_303344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 29), 'write_file', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 228)
            tuple_303345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 42), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 228)
            # Adding element type (line 228)
            # Getting the type of 'def_file' (line 228)
            def_file_303346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 42), 'def_file', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 42), tuple_303345, def_file_303346)
            # Adding element type (line 228)
            # Getting the type of 'contents' (line 228)
            contents_303347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 52), 'contents', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 42), tuple_303345, contents_303347)
            
            str_303348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 29), 'str', 'writing %s')
            # Getting the type of 'def_file' (line 229)
            def_file_303349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 44), 'def_file', False)
            # Applying the binary operator '%' (line 229)
            result_mod_303350 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 29), '%', str_303348, def_file_303349)
            
            # Processing the call keyword arguments (line 228)
            kwargs_303351 = {}
            # Getting the type of 'self' (line 228)
            self_303342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'self', False)
            # Obtaining the member 'execute' of a type (line 228)
            execute_303343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 16), self_303342, 'execute')
            # Calling execute(args, kwargs) (line 228)
            execute_call_result_303352 = invoke(stypy.reporting.localization.Localization(__file__, 228, 16), execute_303343, *[write_file_303344, tuple_303345, result_mod_303350], **kwargs_303351)
            

            if (may_be_303270 and more_types_in_union_303271):
                # SSA join for if statement (line 218)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 232):
        
        # Assigning a Call to a Name (line 232):
        
        # Call to map(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'os' (line 232)
        os_303354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 232)
        path_303355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 27), os_303354, 'path')
        # Obtaining the member 'normpath' of a type (line 232)
        normpath_303356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 27), path_303355, 'normpath')
        # Getting the type of 'objects' (line 232)
        objects_303357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 45), 'objects', False)
        # Processing the call keyword arguments (line 232)
        kwargs_303358 = {}
        # Getting the type of 'map' (line 232)
        map_303353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 23), 'map', False)
        # Calling map(args, kwargs) (line 232)
        map_call_result_303359 = invoke(stypy.reporting.localization.Localization(__file__, 232, 23), map_303353, *[normpath_303356, objects_303357], **kwargs_303358)
        
        # Assigning a type to the variable 'objects2' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'objects2', map_call_result_303359)
        
        # Assigning a List to a Name (line 235):
        
        # Assigning a List to a Name (line 235):
        
        # Obtaining an instance of the builtin type 'list' (line 235)
        list_303360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 235)
        # Adding element type (line 235)
        # Getting the type of 'startup_obj' (line 235)
        startup_obj_303361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 23), 'startup_obj')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 22), list_303360, startup_obj_303361)
        
        # Assigning a type to the variable 'objects' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'objects', list_303360)
        
        # Assigning a List to a Name (line 236):
        
        # Assigning a List to a Name (line 236):
        
        # Obtaining an instance of the builtin type 'list' (line 236)
        list_303362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 236)
        
        # Assigning a type to the variable 'resources' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'resources', list_303362)
        
        # Getting the type of 'objects2' (line 237)
        objects2_303363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 24), 'objects2')
        # Testing the type of a for loop iterable (line 237)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 237, 12), objects2_303363)
        # Getting the type of the for loop variable (line 237)
        for_loop_var_303364 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 237, 12), objects2_303363)
        # Assigning a type to the variable 'file' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'file', for_loop_var_303364)
        # SSA begins for a for statement (line 237)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 238):
        
        # Assigning a Call to a Name:
        
        # Call to splitext(...): (line 238)
        # Processing the call arguments (line 238)
        
        # Call to normcase(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'file' (line 238)
        file_303371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 64), 'file', False)
        # Processing the call keyword arguments (line 238)
        kwargs_303372 = {}
        # Getting the type of 'os' (line 238)
        os_303368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 47), 'os', False)
        # Obtaining the member 'path' of a type (line 238)
        path_303369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 47), os_303368, 'path')
        # Obtaining the member 'normcase' of a type (line 238)
        normcase_303370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 47), path_303369, 'normcase')
        # Calling normcase(args, kwargs) (line 238)
        normcase_call_result_303373 = invoke(stypy.reporting.localization.Localization(__file__, 238, 47), normcase_303370, *[file_303371], **kwargs_303372)
        
        # Processing the call keyword arguments (line 238)
        kwargs_303374 = {}
        # Getting the type of 'os' (line 238)
        os_303365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 238)
        path_303366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 30), os_303365, 'path')
        # Obtaining the member 'splitext' of a type (line 238)
        splitext_303367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 30), path_303366, 'splitext')
        # Calling splitext(args, kwargs) (line 238)
        splitext_call_result_303375 = invoke(stypy.reporting.localization.Localization(__file__, 238, 30), splitext_303367, *[normcase_call_result_303373], **kwargs_303374)
        
        # Assigning a type to the variable 'call_assignment_302849' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'call_assignment_302849', splitext_call_result_303375)
        
        # Assigning a Call to a Name (line 238):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_303378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 16), 'int')
        # Processing the call keyword arguments
        kwargs_303379 = {}
        # Getting the type of 'call_assignment_302849' (line 238)
        call_assignment_302849_303376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'call_assignment_302849', False)
        # Obtaining the member '__getitem__' of a type (line 238)
        getitem___303377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 16), call_assignment_302849_303376, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_303380 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303377, *[int_303378], **kwargs_303379)
        
        # Assigning a type to the variable 'call_assignment_302850' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'call_assignment_302850', getitem___call_result_303380)
        
        # Assigning a Name to a Name (line 238):
        # Getting the type of 'call_assignment_302850' (line 238)
        call_assignment_302850_303381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'call_assignment_302850')
        # Assigning a type to the variable 'base' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 17), 'base', call_assignment_302850_303381)
        
        # Assigning a Call to a Name (line 238):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_303384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 16), 'int')
        # Processing the call keyword arguments
        kwargs_303385 = {}
        # Getting the type of 'call_assignment_302849' (line 238)
        call_assignment_302849_303382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'call_assignment_302849', False)
        # Obtaining the member '__getitem__' of a type (line 238)
        getitem___303383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 16), call_assignment_302849_303382, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_303386 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303383, *[int_303384], **kwargs_303385)
        
        # Assigning a type to the variable 'call_assignment_302851' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'call_assignment_302851', getitem___call_result_303386)
        
        # Assigning a Name to a Name (line 238):
        # Getting the type of 'call_assignment_302851' (line 238)
        call_assignment_302851_303387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'call_assignment_302851')
        # Assigning a type to the variable 'ext' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 23), 'ext', call_assignment_302851_303387)
        
        
        # Getting the type of 'ext' (line 239)
        ext_303388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 19), 'ext')
        str_303389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 26), 'str', '.res')
        # Applying the binary operator '==' (line 239)
        result_eq_303390 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 19), '==', ext_303388, str_303389)
        
        # Testing the type of an if condition (line 239)
        if_condition_303391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 16), result_eq_303390)
        # Assigning a type to the variable 'if_condition_303391' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'if_condition_303391', if_condition_303391)
        # SSA begins for if statement (line 239)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'file' (line 240)
        file_303394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 37), 'file', False)
        # Processing the call keyword arguments (line 240)
        kwargs_303395 = {}
        # Getting the type of 'resources' (line 240)
        resources_303392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'resources', False)
        # Obtaining the member 'append' of a type (line 240)
        append_303393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 20), resources_303392, 'append')
        # Calling append(args, kwargs) (line 240)
        append_call_result_303396 = invoke(stypy.reporting.localization.Localization(__file__, 240, 20), append_303393, *[file_303394], **kwargs_303395)
        
        # SSA branch for the else part of an if statement (line 239)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'file' (line 242)
        file_303399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 35), 'file', False)
        # Processing the call keyword arguments (line 242)
        kwargs_303400 = {}
        # Getting the type of 'objects' (line 242)
        objects_303397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 20), 'objects', False)
        # Obtaining the member 'append' of a type (line 242)
        append_303398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 20), objects_303397, 'append')
        # Calling append(args, kwargs) (line 242)
        append_call_result_303401 = invoke(stypy.reporting.localization.Localization(__file__, 242, 20), append_303398, *[file_303399], **kwargs_303400)
        
        # SSA join for if statement (line 239)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'library_dirs' (line 245)
        library_dirs_303402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 21), 'library_dirs')
        # Testing the type of a for loop iterable (line 245)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 245, 12), library_dirs_303402)
        # Getting the type of the for loop variable (line 245)
        for_loop_var_303403 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 245, 12), library_dirs_303402)
        # Assigning a type to the variable 'l' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'l', for_loop_var_303403)
        # SSA begins for a for statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 246)
        # Processing the call arguments (line 246)
        str_303406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 31), 'str', '/L%s')
        
        # Call to normpath(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'l' (line 246)
        l_303410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 57), 'l', False)
        # Processing the call keyword arguments (line 246)
        kwargs_303411 = {}
        # Getting the type of 'os' (line 246)
        os_303407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 40), 'os', False)
        # Obtaining the member 'path' of a type (line 246)
        path_303408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 40), os_303407, 'path')
        # Obtaining the member 'normpath' of a type (line 246)
        normpath_303409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 40), path_303408, 'normpath')
        # Calling normpath(args, kwargs) (line 246)
        normpath_call_result_303412 = invoke(stypy.reporting.localization.Localization(__file__, 246, 40), normpath_303409, *[l_303410], **kwargs_303411)
        
        # Applying the binary operator '%' (line 246)
        result_mod_303413 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 31), '%', str_303406, normpath_call_result_303412)
        
        # Processing the call keyword arguments (line 246)
        kwargs_303414 = {}
        # Getting the type of 'ld_args' (line 246)
        ld_args_303404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'ld_args', False)
        # Obtaining the member 'append' of a type (line 246)
        append_303405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 16), ld_args_303404, 'append')
        # Calling append(args, kwargs) (line 246)
        append_call_result_303415 = invoke(stypy.reporting.localization.Localization(__file__, 246, 16), append_303405, *[result_mod_303413], **kwargs_303414)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 247)
        # Processing the call arguments (line 247)
        str_303418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 27), 'str', '/L.')
        # Processing the call keyword arguments (line 247)
        kwargs_303419 = {}
        # Getting the type of 'ld_args' (line 247)
        ld_args_303416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'ld_args', False)
        # Obtaining the member 'append' of a type (line 247)
        append_303417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 12), ld_args_303416, 'append')
        # Calling append(args, kwargs) (line 247)
        append_call_result_303420 = invoke(stypy.reporting.localization.Localization(__file__, 247, 12), append_303417, *[str_303418], **kwargs_303419)
        
        
        # Call to extend(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'objects' (line 250)
        objects_303423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 27), 'objects', False)
        # Processing the call keyword arguments (line 250)
        kwargs_303424 = {}
        # Getting the type of 'ld_args' (line 250)
        ld_args_303421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'ld_args', False)
        # Obtaining the member 'extend' of a type (line 250)
        extend_303422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 12), ld_args_303421, 'extend')
        # Calling extend(args, kwargs) (line 250)
        extend_call_result_303425 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), extend_303422, *[objects_303423], **kwargs_303424)
        
        
        # Call to extend(...): (line 263)
        # Processing the call arguments (line 263)
        
        # Obtaining an instance of the builtin type 'list' (line 263)
        list_303428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 263)
        # Adding element type (line 263)
        str_303429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 28), 'str', ',')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 27), list_303428, str_303429)
        # Adding element type (line 263)
        # Getting the type of 'output_filename' (line 263)
        output_filename_303430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 32), 'output_filename', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 27), list_303428, output_filename_303430)
        
        # Processing the call keyword arguments (line 263)
        kwargs_303431 = {}
        # Getting the type of 'ld_args' (line 263)
        ld_args_303426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'ld_args', False)
        # Obtaining the member 'extend' of a type (line 263)
        extend_303427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 12), ld_args_303426, 'extend')
        # Calling extend(args, kwargs) (line 263)
        extend_call_result_303432 = invoke(stypy.reporting.localization.Localization(__file__, 263, 12), extend_303427, *[list_303428], **kwargs_303431)
        
        
        # Call to append(...): (line 265)
        # Processing the call arguments (line 265)
        str_303435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 27), 'str', ',,')
        # Processing the call keyword arguments (line 265)
        kwargs_303436 = {}
        # Getting the type of 'ld_args' (line 265)
        ld_args_303433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'ld_args', False)
        # Obtaining the member 'append' of a type (line 265)
        append_303434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 12), ld_args_303433, 'append')
        # Calling append(args, kwargs) (line 265)
        append_call_result_303437 = invoke(stypy.reporting.localization.Localization(__file__, 265, 12), append_303434, *[str_303435], **kwargs_303436)
        
        
        # Getting the type of 'libraries' (line 267)
        libraries_303438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 23), 'libraries')
        # Testing the type of a for loop iterable (line 267)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 267, 12), libraries_303438)
        # Getting the type of the for loop variable (line 267)
        for_loop_var_303439 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 267, 12), libraries_303438)
        # Assigning a type to the variable 'lib' (line 267)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'lib', for_loop_var_303439)
        # SSA begins for a for statement (line 267)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 270):
        
        # Assigning a Call to a Name (line 270):
        
        # Call to find_library_file(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'library_dirs' (line 270)
        library_dirs_303442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 49), 'library_dirs', False)
        # Getting the type of 'lib' (line 270)
        lib_303443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 63), 'lib', False)
        # Getting the type of 'debug' (line 270)
        debug_303444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 68), 'debug', False)
        # Processing the call keyword arguments (line 270)
        kwargs_303445 = {}
        # Getting the type of 'self' (line 270)
        self_303440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 26), 'self', False)
        # Obtaining the member 'find_library_file' of a type (line 270)
        find_library_file_303441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 26), self_303440, 'find_library_file')
        # Calling find_library_file(args, kwargs) (line 270)
        find_library_file_call_result_303446 = invoke(stypy.reporting.localization.Localization(__file__, 270, 26), find_library_file_303441, *[library_dirs_303442, lib_303443, debug_303444], **kwargs_303445)
        
        # Assigning a type to the variable 'libfile' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'libfile', find_library_file_call_result_303446)
        
        # Type idiom detected: calculating its left and rigth part (line 271)
        # Getting the type of 'libfile' (line 271)
        libfile_303447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 19), 'libfile')
        # Getting the type of 'None' (line 271)
        None_303448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 30), 'None')
        
        (may_be_303449, more_types_in_union_303450) = may_be_none(libfile_303447, None_303448)

        if may_be_303449:

            if more_types_in_union_303450:
                # Runtime conditional SSA (line 271)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 272)
            # Processing the call arguments (line 272)
            # Getting the type of 'lib' (line 272)
            lib_303453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 35), 'lib', False)
            # Processing the call keyword arguments (line 272)
            kwargs_303454 = {}
            # Getting the type of 'ld_args' (line 272)
            ld_args_303451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 20), 'ld_args', False)
            # Obtaining the member 'append' of a type (line 272)
            append_303452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 20), ld_args_303451, 'append')
            # Calling append(args, kwargs) (line 272)
            append_call_result_303455 = invoke(stypy.reporting.localization.Localization(__file__, 272, 20), append_303452, *[lib_303453], **kwargs_303454)
            

            if more_types_in_union_303450:
                # Runtime conditional SSA for else branch (line 271)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_303449) or more_types_in_union_303450):
            
            # Call to append(...): (line 276)
            # Processing the call arguments (line 276)
            # Getting the type of 'libfile' (line 276)
            libfile_303458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 35), 'libfile', False)
            # Processing the call keyword arguments (line 276)
            kwargs_303459 = {}
            # Getting the type of 'ld_args' (line 276)
            ld_args_303456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'ld_args', False)
            # Obtaining the member 'append' of a type (line 276)
            append_303457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 20), ld_args_303456, 'append')
            # Calling append(args, kwargs) (line 276)
            append_call_result_303460 = invoke(stypy.reporting.localization.Localization(__file__, 276, 20), append_303457, *[libfile_303458], **kwargs_303459)
            

            if (may_be_303449 and more_types_in_union_303450):
                # SSA join for if statement (line 271)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 279)
        # Processing the call arguments (line 279)
        str_303463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 28), 'str', 'import32')
        # Processing the call keyword arguments (line 279)
        kwargs_303464 = {}
        # Getting the type of 'ld_args' (line 279)
        ld_args_303461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'ld_args', False)
        # Obtaining the member 'append' of a type (line 279)
        append_303462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), ld_args_303461, 'append')
        # Calling append(args, kwargs) (line 279)
        append_call_result_303465 = invoke(stypy.reporting.localization.Localization(__file__, 279, 12), append_303462, *[str_303463], **kwargs_303464)
        
        
        # Call to append(...): (line 280)
        # Processing the call arguments (line 280)
        str_303468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 28), 'str', 'cw32mt')
        # Processing the call keyword arguments (line 280)
        kwargs_303469 = {}
        # Getting the type of 'ld_args' (line 280)
        ld_args_303466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'ld_args', False)
        # Obtaining the member 'append' of a type (line 280)
        append_303467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), ld_args_303466, 'append')
        # Calling append(args, kwargs) (line 280)
        append_call_result_303470 = invoke(stypy.reporting.localization.Localization(__file__, 280, 12), append_303467, *[str_303468], **kwargs_303469)
        
        
        # Call to extend(...): (line 283)
        # Processing the call arguments (line 283)
        
        # Obtaining an instance of the builtin type 'list' (line 283)
        list_303473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 283)
        # Adding element type (line 283)
        str_303474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 28), 'str', ',')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 27), list_303473, str_303474)
        # Adding element type (line 283)
        # Getting the type of 'def_file' (line 283)
        def_file_303475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 32), 'def_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 283, 27), list_303473, def_file_303475)
        
        # Processing the call keyword arguments (line 283)
        kwargs_303476 = {}
        # Getting the type of 'ld_args' (line 283)
        ld_args_303471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'ld_args', False)
        # Obtaining the member 'extend' of a type (line 283)
        extend_303472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 12), ld_args_303471, 'extend')
        # Calling extend(args, kwargs) (line 283)
        extend_call_result_303477 = invoke(stypy.reporting.localization.Localization(__file__, 283, 12), extend_303472, *[list_303473], **kwargs_303476)
        
        
        # Call to append(...): (line 285)
        # Processing the call arguments (line 285)
        str_303480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 27), 'str', ',')
        # Processing the call keyword arguments (line 285)
        kwargs_303481 = {}
        # Getting the type of 'ld_args' (line 285)
        ld_args_303478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'ld_args', False)
        # Obtaining the member 'append' of a type (line 285)
        append_303479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 12), ld_args_303478, 'append')
        # Calling append(args, kwargs) (line 285)
        append_call_result_303482 = invoke(stypy.reporting.localization.Localization(__file__, 285, 12), append_303479, *[str_303480], **kwargs_303481)
        
        
        # Call to extend(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'resources' (line 286)
        resources_303485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 27), 'resources', False)
        # Processing the call keyword arguments (line 286)
        kwargs_303486 = {}
        # Getting the type of 'ld_args' (line 286)
        ld_args_303483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'ld_args', False)
        # Obtaining the member 'extend' of a type (line 286)
        extend_303484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 12), ld_args_303483, 'extend')
        # Calling extend(args, kwargs) (line 286)
        extend_call_result_303487 = invoke(stypy.reporting.localization.Localization(__file__, 286, 12), extend_303484, *[resources_303485], **kwargs_303486)
        
        
        # Getting the type of 'extra_preargs' (line 289)
        extra_preargs_303488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), 'extra_preargs')
        # Testing the type of an if condition (line 289)
        if_condition_303489 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 12), extra_preargs_303488)
        # Assigning a type to the variable 'if_condition_303489' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'if_condition_303489', if_condition_303489)
        # SSA begins for if statement (line 289)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 290):
        
        # Assigning a Name to a Subscript (line 290):
        # Getting the type of 'extra_preargs' (line 290)
        extra_preargs_303490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 30), 'extra_preargs')
        # Getting the type of 'ld_args' (line 290)
        ld_args_303491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'ld_args')
        int_303492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 25), 'int')
        slice_303493 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 290, 16), None, int_303492, None)
        # Storing an element on a container (line 290)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 16), ld_args_303491, (slice_303493, extra_preargs_303490))
        # SSA join for if statement (line 289)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'extra_postargs' (line 291)
        extra_postargs_303494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'extra_postargs')
        # Testing the type of an if condition (line 291)
        if_condition_303495 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 291, 12), extra_postargs_303494)
        # Assigning a type to the variable 'if_condition_303495' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'if_condition_303495', if_condition_303495)
        # SSA begins for if statement (line 291)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'extra_postargs' (line 292)
        extra_postargs_303498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 31), 'extra_postargs', False)
        # Processing the call keyword arguments (line 292)
        kwargs_303499 = {}
        # Getting the type of 'ld_args' (line 292)
        ld_args_303496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 16), 'ld_args', False)
        # Obtaining the member 'extend' of a type (line 292)
        extend_303497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 16), ld_args_303496, 'extend')
        # Calling extend(args, kwargs) (line 292)
        extend_call_result_303500 = invoke(stypy.reporting.localization.Localization(__file__, 292, 16), extend_303497, *[extra_postargs_303498], **kwargs_303499)
        
        # SSA join for if statement (line 291)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to mkpath(...): (line 294)
        # Processing the call arguments (line 294)
        
        # Call to dirname(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'output_filename' (line 294)
        output_filename_303506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 42), 'output_filename', False)
        # Processing the call keyword arguments (line 294)
        kwargs_303507 = {}
        # Getting the type of 'os' (line 294)
        os_303503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 294)
        path_303504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 25), os_303503, 'path')
        # Obtaining the member 'dirname' of a type (line 294)
        dirname_303505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 25), path_303504, 'dirname')
        # Calling dirname(args, kwargs) (line 294)
        dirname_call_result_303508 = invoke(stypy.reporting.localization.Localization(__file__, 294, 25), dirname_303505, *[output_filename_303506], **kwargs_303507)
        
        # Processing the call keyword arguments (line 294)
        kwargs_303509 = {}
        # Getting the type of 'self' (line 294)
        self_303501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 294)
        mkpath_303502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 12), self_303501, 'mkpath')
        # Calling mkpath(args, kwargs) (line 294)
        mkpath_call_result_303510 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), mkpath_303502, *[dirname_call_result_303508], **kwargs_303509)
        
        
        
        # SSA begins for try-except statement (line 295)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 296)
        # Processing the call arguments (line 296)
        
        # Obtaining an instance of the builtin type 'list' (line 296)
        list_303513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 296)
        # Adding element type (line 296)
        # Getting the type of 'self' (line 296)
        self_303514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 29), 'self', False)
        # Obtaining the member 'linker' of a type (line 296)
        linker_303515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 29), self_303514, 'linker')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 28), list_303513, linker_303515)
        
        # Getting the type of 'ld_args' (line 296)
        ld_args_303516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 44), 'ld_args', False)
        # Applying the binary operator '+' (line 296)
        result_add_303517 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 28), '+', list_303513, ld_args_303516)
        
        # Processing the call keyword arguments (line 296)
        kwargs_303518 = {}
        # Getting the type of 'self' (line 296)
        self_303511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 296)
        spawn_303512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 16), self_303511, 'spawn')
        # Calling spawn(args, kwargs) (line 296)
        spawn_call_result_303519 = invoke(stypy.reporting.localization.Localization(__file__, 296, 16), spawn_303512, *[result_add_303517], **kwargs_303518)
        
        # SSA branch for the except part of a try statement (line 295)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 295)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 297)
        DistutilsExecError_303520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 19), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'msg', DistutilsExecError_303520)
        # Getting the type of 'LinkError' (line 298)
        LinkError_303521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 22), 'LinkError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 298, 16), LinkError_303521, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 295)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 200)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug(...): (line 301)
        # Processing the call arguments (line 301)
        str_303524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 22), 'str', 'skipping %s (up-to-date)')
        # Getting the type of 'output_filename' (line 301)
        output_filename_303525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 50), 'output_filename', False)
        # Processing the call keyword arguments (line 301)
        kwargs_303526 = {}
        # Getting the type of 'log' (line 301)
        log_303522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 301)
        debug_303523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 12), log_303522, 'debug')
        # Calling debug(args, kwargs) (line 301)
        debug_call_result_303527 = invoke(stypy.reporting.localization.Localization(__file__, 301, 12), debug_303523, *[str_303524, output_filename_303525], **kwargs_303526)
        
        # SSA join for if statement (line 200)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'link(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'link' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_303528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_303528)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'link'
        return stypy_return_type_303528


    @norecursion
    def find_library_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_303529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 50), 'int')
        defaults = [int_303529]
        # Create a new context for function 'find_library_file'
        module_type_store = module_type_store.open_function_context('find_library_file', 308, 4, False)
        # Assigning a type to the variable 'self' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BCPPCompiler.find_library_file.__dict__.__setitem__('stypy_localization', localization)
        BCPPCompiler.find_library_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BCPPCompiler.find_library_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        BCPPCompiler.find_library_file.__dict__.__setitem__('stypy_function_name', 'BCPPCompiler.find_library_file')
        BCPPCompiler.find_library_file.__dict__.__setitem__('stypy_param_names_list', ['dirs', 'lib', 'debug'])
        BCPPCompiler.find_library_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        BCPPCompiler.find_library_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BCPPCompiler.find_library_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        BCPPCompiler.find_library_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        BCPPCompiler.find_library_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BCPPCompiler.find_library_file.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BCPPCompiler.find_library_file', ['dirs', 'lib', 'debug'], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'debug' (line 318)
        debug_303530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 11), 'debug')
        # Testing the type of an if condition (line 318)
        if_condition_303531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 8), debug_303530)
        # Assigning a type to the variable 'if_condition_303531' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 8), 'if_condition_303531', if_condition_303531)
        # SSA begins for if statement (line 318)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 319):
        
        # Assigning a BinOp to a Name (line 319):
        # Getting the type of 'lib' (line 319)
        lib_303532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 20), 'lib')
        str_303533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 26), 'str', '_d')
        # Applying the binary operator '+' (line 319)
        result_add_303534 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 20), '+', lib_303532, str_303533)
        
        # Assigning a type to the variable 'dlib' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'dlib', result_add_303534)
        
        # Assigning a Tuple to a Name (line 320):
        
        # Assigning a Tuple to a Name (line 320):
        
        # Obtaining an instance of the builtin type 'tuple' (line 320)
        tuple_303535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 320)
        # Adding element type (line 320)
        # Getting the type of 'dlib' (line 320)
        dlib_303536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 25), 'dlib')
        str_303537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 32), 'str', '_bcpp')
        # Applying the binary operator '+' (line 320)
        result_add_303538 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 25), '+', dlib_303536, str_303537)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 25), tuple_303535, result_add_303538)
        # Adding element type (line 320)
        # Getting the type of 'lib' (line 320)
        lib_303539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 41), 'lib')
        str_303540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 47), 'str', '_bcpp')
        # Applying the binary operator '+' (line 320)
        result_add_303541 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 41), '+', lib_303539, str_303540)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 25), tuple_303535, result_add_303541)
        # Adding element type (line 320)
        # Getting the type of 'dlib' (line 320)
        dlib_303542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 56), 'dlib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 25), tuple_303535, dlib_303542)
        # Adding element type (line 320)
        # Getting the type of 'lib' (line 320)
        lib_303543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 62), 'lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 320, 25), tuple_303535, lib_303543)
        
        # Assigning a type to the variable 'try_names' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'try_names', tuple_303535)
        # SSA branch for the else part of an if statement (line 318)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Tuple to a Name (line 322):
        
        # Assigning a Tuple to a Name (line 322):
        
        # Obtaining an instance of the builtin type 'tuple' (line 322)
        tuple_303544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 322)
        # Adding element type (line 322)
        # Getting the type of 'lib' (line 322)
        lib_303545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 25), 'lib')
        str_303546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 31), 'str', '_bcpp')
        # Applying the binary operator '+' (line 322)
        result_add_303547 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 25), '+', lib_303545, str_303546)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 25), tuple_303544, result_add_303547)
        # Adding element type (line 322)
        # Getting the type of 'lib' (line 322)
        lib_303548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 40), 'lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 25), tuple_303544, lib_303548)
        
        # Assigning a type to the variable 'try_names' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'try_names', tuple_303544)
        # SSA join for if statement (line 318)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'dirs' (line 324)
        dirs_303549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 19), 'dirs')
        # Testing the type of a for loop iterable (line 324)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 324, 8), dirs_303549)
        # Getting the type of the for loop variable (line 324)
        for_loop_var_303550 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 324, 8), dirs_303549)
        # Assigning a type to the variable 'dir' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'dir', for_loop_var_303550)
        # SSA begins for a for statement (line 324)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'try_names' (line 325)
        try_names_303551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 24), 'try_names')
        # Testing the type of a for loop iterable (line 325)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 325, 12), try_names_303551)
        # Getting the type of the for loop variable (line 325)
        for_loop_var_303552 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 325, 12), try_names_303551)
        # Assigning a type to the variable 'name' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'name', for_loop_var_303552)
        # SSA begins for a for statement (line 325)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 326):
        
        # Assigning a Call to a Name (line 326):
        
        # Call to join(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'dir' (line 326)
        dir_303556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 39), 'dir', False)
        
        # Call to library_filename(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'name' (line 326)
        name_303559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 66), 'name', False)
        # Processing the call keyword arguments (line 326)
        kwargs_303560 = {}
        # Getting the type of 'self' (line 326)
        self_303557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 44), 'self', False)
        # Obtaining the member 'library_filename' of a type (line 326)
        library_filename_303558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 44), self_303557, 'library_filename')
        # Calling library_filename(args, kwargs) (line 326)
        library_filename_call_result_303561 = invoke(stypy.reporting.localization.Localization(__file__, 326, 44), library_filename_303558, *[name_303559], **kwargs_303560)
        
        # Processing the call keyword arguments (line 326)
        kwargs_303562 = {}
        # Getting the type of 'os' (line 326)
        os_303553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 326)
        path_303554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 26), os_303553, 'path')
        # Obtaining the member 'join' of a type (line 326)
        join_303555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 26), path_303554, 'join')
        # Calling join(args, kwargs) (line 326)
        join_call_result_303563 = invoke(stypy.reporting.localization.Localization(__file__, 326, 26), join_303555, *[dir_303556, library_filename_call_result_303561], **kwargs_303562)
        
        # Assigning a type to the variable 'libfile' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'libfile', join_call_result_303563)
        
        
        # Call to exists(...): (line 327)
        # Processing the call arguments (line 327)
        # Getting the type of 'libfile' (line 327)
        libfile_303567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 34), 'libfile', False)
        # Processing the call keyword arguments (line 327)
        kwargs_303568 = {}
        # Getting the type of 'os' (line 327)
        os_303564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 327)
        path_303565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 19), os_303564, 'path')
        # Obtaining the member 'exists' of a type (line 327)
        exists_303566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 19), path_303565, 'exists')
        # Calling exists(args, kwargs) (line 327)
        exists_call_result_303569 = invoke(stypy.reporting.localization.Localization(__file__, 327, 19), exists_303566, *[libfile_303567], **kwargs_303568)
        
        # Testing the type of an if condition (line 327)
        if_condition_303570 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 327, 16), exists_call_result_303569)
        # Assigning a type to the variable 'if_condition_303570' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 16), 'if_condition_303570', if_condition_303570)
        # SSA begins for if statement (line 327)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'libfile' (line 328)
        libfile_303571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 27), 'libfile')
        # Assigning a type to the variable 'stypy_return_type' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 20), 'stypy_return_type', libfile_303571)
        # SSA join for if statement (line 327)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 324)
        module_type_store.open_ssa_branch('for loop else')
        # Getting the type of 'None' (line 331)
        None_303572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 331)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'stypy_return_type', None_303572)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'find_library_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_library_file' in the type store
        # Getting the type of 'stypy_return_type' (line 308)
        stypy_return_type_303573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_303573)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_library_file'
        return stypy_return_type_303573


    @norecursion
    def object_filenames(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_303574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 36), 'int')
        str_303575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 37), 'str', '')
        defaults = [int_303574, str_303575]
        # Create a new context for function 'object_filenames'
        module_type_store = module_type_store.open_function_context('object_filenames', 334, 4, False)
        # Assigning a type to the variable 'self' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BCPPCompiler.object_filenames.__dict__.__setitem__('stypy_localization', localization)
        BCPPCompiler.object_filenames.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BCPPCompiler.object_filenames.__dict__.__setitem__('stypy_type_store', module_type_store)
        BCPPCompiler.object_filenames.__dict__.__setitem__('stypy_function_name', 'BCPPCompiler.object_filenames')
        BCPPCompiler.object_filenames.__dict__.__setitem__('stypy_param_names_list', ['source_filenames', 'strip_dir', 'output_dir'])
        BCPPCompiler.object_filenames.__dict__.__setitem__('stypy_varargs_param_name', None)
        BCPPCompiler.object_filenames.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BCPPCompiler.object_filenames.__dict__.__setitem__('stypy_call_defaults', defaults)
        BCPPCompiler.object_filenames.__dict__.__setitem__('stypy_call_varargs', varargs)
        BCPPCompiler.object_filenames.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BCPPCompiler.object_filenames.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BCPPCompiler.object_filenames', ['source_filenames', 'strip_dir', 'output_dir'], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 338)
        # Getting the type of 'output_dir' (line 338)
        output_dir_303576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 11), 'output_dir')
        # Getting the type of 'None' (line 338)
        None_303577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 25), 'None')
        
        (may_be_303578, more_types_in_union_303579) = may_be_none(output_dir_303576, None_303577)

        if may_be_303578:

            if more_types_in_union_303579:
                # Runtime conditional SSA (line 338)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 338):
            
            # Assigning a Str to a Name (line 338):
            str_303580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 44), 'str', '')
            # Assigning a type to the variable 'output_dir' (line 338)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 31), 'output_dir', str_303580)

            if more_types_in_union_303579:
                # SSA join for if statement (line 338)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Name (line 339):
        
        # Assigning a List to a Name (line 339):
        
        # Obtaining an instance of the builtin type 'list' (line 339)
        list_303581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 339)
        
        # Assigning a type to the variable 'obj_names' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'obj_names', list_303581)
        
        # Getting the type of 'source_filenames' (line 340)
        source_filenames_303582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 24), 'source_filenames')
        # Testing the type of a for loop iterable (line 340)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 340, 8), source_filenames_303582)
        # Getting the type of the for loop variable (line 340)
        for_loop_var_303583 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 340, 8), source_filenames_303582)
        # Assigning a type to the variable 'src_name' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'src_name', for_loop_var_303583)
        # SSA begins for a for statement (line 340)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 342):
        
        # Assigning a Call to a Name:
        
        # Call to splitext(...): (line 342)
        # Processing the call arguments (line 342)
        
        # Call to normcase(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'src_name' (line 342)
        src_name_303590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 61), 'src_name', False)
        # Processing the call keyword arguments (line 342)
        kwargs_303591 = {}
        # Getting the type of 'os' (line 342)
        os_303587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 44), 'os', False)
        # Obtaining the member 'path' of a type (line 342)
        path_303588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 44), os_303587, 'path')
        # Obtaining the member 'normcase' of a type (line 342)
        normcase_303589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 44), path_303588, 'normcase')
        # Calling normcase(args, kwargs) (line 342)
        normcase_call_result_303592 = invoke(stypy.reporting.localization.Localization(__file__, 342, 44), normcase_303589, *[src_name_303590], **kwargs_303591)
        
        # Processing the call keyword arguments (line 342)
        kwargs_303593 = {}
        # Getting the type of 'os' (line 342)
        os_303584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 342)
        path_303585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 26), os_303584, 'path')
        # Obtaining the member 'splitext' of a type (line 342)
        splitext_303586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 26), path_303585, 'splitext')
        # Calling splitext(args, kwargs) (line 342)
        splitext_call_result_303594 = invoke(stypy.reporting.localization.Localization(__file__, 342, 26), splitext_303586, *[normcase_call_result_303592], **kwargs_303593)
        
        # Assigning a type to the variable 'call_assignment_302852' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'call_assignment_302852', splitext_call_result_303594)
        
        # Assigning a Call to a Name (line 342):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_303597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 12), 'int')
        # Processing the call keyword arguments
        kwargs_303598 = {}
        # Getting the type of 'call_assignment_302852' (line 342)
        call_assignment_302852_303595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'call_assignment_302852', False)
        # Obtaining the member '__getitem__' of a type (line 342)
        getitem___303596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 12), call_assignment_302852_303595, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_303599 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303596, *[int_303597], **kwargs_303598)
        
        # Assigning a type to the variable 'call_assignment_302853' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'call_assignment_302853', getitem___call_result_303599)
        
        # Assigning a Name to a Name (line 342):
        # Getting the type of 'call_assignment_302853' (line 342)
        call_assignment_302853_303600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'call_assignment_302853')
        # Assigning a type to the variable 'base' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 13), 'base', call_assignment_302853_303600)
        
        # Assigning a Call to a Name (line 342):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_303603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 12), 'int')
        # Processing the call keyword arguments
        kwargs_303604 = {}
        # Getting the type of 'call_assignment_302852' (line 342)
        call_assignment_302852_303601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'call_assignment_302852', False)
        # Obtaining the member '__getitem__' of a type (line 342)
        getitem___303602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 12), call_assignment_302852_303601, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_303605 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303602, *[int_303603], **kwargs_303604)
        
        # Assigning a type to the variable 'call_assignment_302854' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'call_assignment_302854', getitem___call_result_303605)
        
        # Assigning a Name to a Name (line 342):
        # Getting the type of 'call_assignment_302854' (line 342)
        call_assignment_302854_303606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 12), 'call_assignment_302854')
        # Assigning a type to the variable 'ext' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 19), 'ext', call_assignment_302854_303606)
        
        
        # Getting the type of 'ext' (line 343)
        ext_303607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 15), 'ext')
        # Getting the type of 'self' (line 343)
        self_303608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 27), 'self')
        # Obtaining the member 'src_extensions' of a type (line 343)
        src_extensions_303609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 27), self_303608, 'src_extensions')
        
        # Obtaining an instance of the builtin type 'list' (line 343)
        list_303610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 343)
        # Adding element type (line 343)
        str_303611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 50), 'str', '.rc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 49), list_303610, str_303611)
        # Adding element type (line 343)
        str_303612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 56), 'str', '.res')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 343, 49), list_303610, str_303612)
        
        # Applying the binary operator '+' (line 343)
        result_add_303613 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 27), '+', src_extensions_303609, list_303610)
        
        # Applying the binary operator 'notin' (line 343)
        result_contains_303614 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 15), 'notin', ext_303607, result_add_303613)
        
        # Testing the type of an if condition (line 343)
        if_condition_303615 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 343, 12), result_contains_303614)
        # Assigning a type to the variable 'if_condition_303615' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'if_condition_303615', if_condition_303615)
        # SSA begins for if statement (line 343)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'UnknownFileError' (line 344)
        UnknownFileError_303616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 22), 'UnknownFileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 344, 16), UnknownFileError_303616, 'raise parameter', BaseException)
        # SSA join for if statement (line 343)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'strip_dir' (line 347)
        strip_dir_303617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 15), 'strip_dir')
        # Testing the type of an if condition (line 347)
        if_condition_303618 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 347, 12), strip_dir_303617)
        # Assigning a type to the variable 'if_condition_303618' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'if_condition_303618', if_condition_303618)
        # SSA begins for if statement (line 347)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 348):
        
        # Assigning a Call to a Name (line 348):
        
        # Call to basename(...): (line 348)
        # Processing the call arguments (line 348)
        # Getting the type of 'base' (line 348)
        base_303622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 41), 'base', False)
        # Processing the call keyword arguments (line 348)
        kwargs_303623 = {}
        # Getting the type of 'os' (line 348)
        os_303619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 348)
        path_303620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 23), os_303619, 'path')
        # Obtaining the member 'basename' of a type (line 348)
        basename_303621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 23), path_303620, 'basename')
        # Calling basename(args, kwargs) (line 348)
        basename_call_result_303624 = invoke(stypy.reporting.localization.Localization(__file__, 348, 23), basename_303621, *[base_303622], **kwargs_303623)
        
        # Assigning a type to the variable 'base' (line 348)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 16), 'base', basename_call_result_303624)
        # SSA join for if statement (line 347)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'ext' (line 349)
        ext_303625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 15), 'ext')
        str_303626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 22), 'str', '.res')
        # Applying the binary operator '==' (line 349)
        result_eq_303627 = python_operator(stypy.reporting.localization.Localization(__file__, 349, 15), '==', ext_303625, str_303626)
        
        # Testing the type of an if condition (line 349)
        if_condition_303628 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 349, 12), result_eq_303627)
        # Assigning a type to the variable 'if_condition_303628' (line 349)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'if_condition_303628', if_condition_303628)
        # SSA begins for if statement (line 349)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 351)
        # Processing the call arguments (line 351)
        
        # Call to join(...): (line 351)
        # Processing the call arguments (line 351)
        # Getting the type of 'output_dir' (line 351)
        output_dir_303634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 48), 'output_dir', False)
        # Getting the type of 'base' (line 351)
        base_303635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 60), 'base', False)
        # Getting the type of 'ext' (line 351)
        ext_303636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 67), 'ext', False)
        # Applying the binary operator '+' (line 351)
        result_add_303637 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 60), '+', base_303635, ext_303636)
        
        # Processing the call keyword arguments (line 351)
        kwargs_303638 = {}
        # Getting the type of 'os' (line 351)
        os_303631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 351)
        path_303632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 34), os_303631, 'path')
        # Obtaining the member 'join' of a type (line 351)
        join_303633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 34), path_303632, 'join')
        # Calling join(args, kwargs) (line 351)
        join_call_result_303639 = invoke(stypy.reporting.localization.Localization(__file__, 351, 34), join_303633, *[output_dir_303634, result_add_303637], **kwargs_303638)
        
        # Processing the call keyword arguments (line 351)
        kwargs_303640 = {}
        # Getting the type of 'obj_names' (line 351)
        obj_names_303629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 16), 'obj_names', False)
        # Obtaining the member 'append' of a type (line 351)
        append_303630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 16), obj_names_303629, 'append')
        # Calling append(args, kwargs) (line 351)
        append_call_result_303641 = invoke(stypy.reporting.localization.Localization(__file__, 351, 16), append_303630, *[join_call_result_303639], **kwargs_303640)
        
        # SSA branch for the else part of an if statement (line 349)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'ext' (line 352)
        ext_303642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 17), 'ext')
        str_303643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 24), 'str', '.rc')
        # Applying the binary operator '==' (line 352)
        result_eq_303644 = python_operator(stypy.reporting.localization.Localization(__file__, 352, 17), '==', ext_303642, str_303643)
        
        # Testing the type of an if condition (line 352)
        if_condition_303645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 352, 17), result_eq_303644)
        # Assigning a type to the variable 'if_condition_303645' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 17), 'if_condition_303645', if_condition_303645)
        # SSA begins for if statement (line 352)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 354)
        # Processing the call arguments (line 354)
        
        # Call to join(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'output_dir' (line 354)
        output_dir_303651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 48), 'output_dir', False)
        # Getting the type of 'base' (line 354)
        base_303652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 60), 'base', False)
        str_303653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 67), 'str', '.res')
        # Applying the binary operator '+' (line 354)
        result_add_303654 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 60), '+', base_303652, str_303653)
        
        # Processing the call keyword arguments (line 354)
        kwargs_303655 = {}
        # Getting the type of 'os' (line 354)
        os_303648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 354)
        path_303649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 34), os_303648, 'path')
        # Obtaining the member 'join' of a type (line 354)
        join_303650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 34), path_303649, 'join')
        # Calling join(args, kwargs) (line 354)
        join_call_result_303656 = invoke(stypy.reporting.localization.Localization(__file__, 354, 34), join_303650, *[output_dir_303651, result_add_303654], **kwargs_303655)
        
        # Processing the call keyword arguments (line 354)
        kwargs_303657 = {}
        # Getting the type of 'obj_names' (line 354)
        obj_names_303646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'obj_names', False)
        # Obtaining the member 'append' of a type (line 354)
        append_303647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 16), obj_names_303646, 'append')
        # Calling append(args, kwargs) (line 354)
        append_call_result_303658 = invoke(stypy.reporting.localization.Localization(__file__, 354, 16), append_303647, *[join_call_result_303656], **kwargs_303657)
        
        # SSA branch for the else part of an if statement (line 352)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 356)
        # Processing the call arguments (line 356)
        
        # Call to join(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'output_dir' (line 356)
        output_dir_303664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 48), 'output_dir', False)
        # Getting the type of 'base' (line 357)
        base_303665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 44), 'base', False)
        # Getting the type of 'self' (line 357)
        self_303666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 51), 'self', False)
        # Obtaining the member 'obj_extension' of a type (line 357)
        obj_extension_303667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 51), self_303666, 'obj_extension')
        # Applying the binary operator '+' (line 357)
        result_add_303668 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 44), '+', base_303665, obj_extension_303667)
        
        # Processing the call keyword arguments (line 356)
        kwargs_303669 = {}
        # Getting the type of 'os' (line 356)
        os_303661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 34), 'os', False)
        # Obtaining the member 'path' of a type (line 356)
        path_303662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 34), os_303661, 'path')
        # Obtaining the member 'join' of a type (line 356)
        join_303663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 34), path_303662, 'join')
        # Calling join(args, kwargs) (line 356)
        join_call_result_303670 = invoke(stypy.reporting.localization.Localization(__file__, 356, 34), join_303663, *[output_dir_303664, result_add_303668], **kwargs_303669)
        
        # Processing the call keyword arguments (line 356)
        kwargs_303671 = {}
        # Getting the type of 'obj_names' (line 356)
        obj_names_303659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 16), 'obj_names', False)
        # Obtaining the member 'append' of a type (line 356)
        append_303660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 16), obj_names_303659, 'append')
        # Calling append(args, kwargs) (line 356)
        append_call_result_303672 = invoke(stypy.reporting.localization.Localization(__file__, 356, 16), append_303660, *[join_call_result_303670], **kwargs_303671)
        
        # SSA join for if statement (line 352)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 349)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'obj_names' (line 358)
        obj_names_303673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 15), 'obj_names')
        # Assigning a type to the variable 'stypy_return_type' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'stypy_return_type', obj_names_303673)
        
        # ################# End of 'object_filenames(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'object_filenames' in the type store
        # Getting the type of 'stypy_return_type' (line 334)
        stypy_return_type_303674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_303674)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'object_filenames'
        return stypy_return_type_303674


    @norecursion
    def preprocess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 364)
        None_303675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 32), 'None')
        # Getting the type of 'None' (line 365)
        None_303676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 27), 'None')
        # Getting the type of 'None' (line 366)
        None_303677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 33), 'None')
        # Getting the type of 'None' (line 367)
        None_303678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 34), 'None')
        # Getting the type of 'None' (line 368)
        None_303679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 35), 'None')
        defaults = [None_303675, None_303676, None_303677, None_303678, None_303679]
        # Create a new context for function 'preprocess'
        module_type_store = module_type_store.open_function_context('preprocess', 362, 4, False)
        # Assigning a type to the variable 'self' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BCPPCompiler.preprocess.__dict__.__setitem__('stypy_localization', localization)
        BCPPCompiler.preprocess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BCPPCompiler.preprocess.__dict__.__setitem__('stypy_type_store', module_type_store)
        BCPPCompiler.preprocess.__dict__.__setitem__('stypy_function_name', 'BCPPCompiler.preprocess')
        BCPPCompiler.preprocess.__dict__.__setitem__('stypy_param_names_list', ['source', 'output_file', 'macros', 'include_dirs', 'extra_preargs', 'extra_postargs'])
        BCPPCompiler.preprocess.__dict__.__setitem__('stypy_varargs_param_name', None)
        BCPPCompiler.preprocess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BCPPCompiler.preprocess.__dict__.__setitem__('stypy_call_defaults', defaults)
        BCPPCompiler.preprocess.__dict__.__setitem__('stypy_call_varargs', varargs)
        BCPPCompiler.preprocess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BCPPCompiler.preprocess.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BCPPCompiler.preprocess', ['source', 'output_file', 'macros', 'include_dirs', 'extra_preargs', 'extra_postargs'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Tuple (line 370):
        
        # Assigning a Call to a Name:
        
        # Call to _fix_compile_args(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'None' (line 371)
        None_303682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 35), 'None', False)
        # Getting the type of 'macros' (line 371)
        macros_303683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 41), 'macros', False)
        # Getting the type of 'include_dirs' (line 371)
        include_dirs_303684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 49), 'include_dirs', False)
        # Processing the call keyword arguments (line 371)
        kwargs_303685 = {}
        # Getting the type of 'self' (line 371)
        self_303680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'self', False)
        # Obtaining the member '_fix_compile_args' of a type (line 371)
        _fix_compile_args_303681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 12), self_303680, '_fix_compile_args')
        # Calling _fix_compile_args(args, kwargs) (line 371)
        _fix_compile_args_call_result_303686 = invoke(stypy.reporting.localization.Localization(__file__, 371, 12), _fix_compile_args_303681, *[None_303682, macros_303683, include_dirs_303684], **kwargs_303685)
        
        # Assigning a type to the variable 'call_assignment_302855' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'call_assignment_302855', _fix_compile_args_call_result_303686)
        
        # Assigning a Call to a Name (line 370):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_303689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 8), 'int')
        # Processing the call keyword arguments
        kwargs_303690 = {}
        # Getting the type of 'call_assignment_302855' (line 370)
        call_assignment_302855_303687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'call_assignment_302855', False)
        # Obtaining the member '__getitem__' of a type (line 370)
        getitem___303688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 8), call_assignment_302855_303687, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_303691 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303688, *[int_303689], **kwargs_303690)
        
        # Assigning a type to the variable 'call_assignment_302856' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'call_assignment_302856', getitem___call_result_303691)
        
        # Assigning a Name to a Name (line 370):
        # Getting the type of 'call_assignment_302856' (line 370)
        call_assignment_302856_303692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'call_assignment_302856')
        # Assigning a type to the variable '_' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 9), '_', call_assignment_302856_303692)
        
        # Assigning a Call to a Name (line 370):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_303695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 8), 'int')
        # Processing the call keyword arguments
        kwargs_303696 = {}
        # Getting the type of 'call_assignment_302855' (line 370)
        call_assignment_302855_303693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'call_assignment_302855', False)
        # Obtaining the member '__getitem__' of a type (line 370)
        getitem___303694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 8), call_assignment_302855_303693, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_303697 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303694, *[int_303695], **kwargs_303696)
        
        # Assigning a type to the variable 'call_assignment_302857' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'call_assignment_302857', getitem___call_result_303697)
        
        # Assigning a Name to a Name (line 370):
        # Getting the type of 'call_assignment_302857' (line 370)
        call_assignment_302857_303698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'call_assignment_302857')
        # Assigning a type to the variable 'macros' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'macros', call_assignment_302857_303698)
        
        # Assigning a Call to a Name (line 370):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_303701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 8), 'int')
        # Processing the call keyword arguments
        kwargs_303702 = {}
        # Getting the type of 'call_assignment_302855' (line 370)
        call_assignment_302855_303699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'call_assignment_302855', False)
        # Obtaining the member '__getitem__' of a type (line 370)
        getitem___303700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 8), call_assignment_302855_303699, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_303703 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___303700, *[int_303701], **kwargs_303702)
        
        # Assigning a type to the variable 'call_assignment_302858' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'call_assignment_302858', getitem___call_result_303703)
        
        # Assigning a Name to a Name (line 370):
        # Getting the type of 'call_assignment_302858' (line 370)
        call_assignment_302858_303704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'call_assignment_302858')
        # Assigning a type to the variable 'include_dirs' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 20), 'include_dirs', call_assignment_302858_303704)
        
        # Assigning a Call to a Name (line 372):
        
        # Assigning a Call to a Name (line 372):
        
        # Call to gen_preprocess_options(...): (line 372)
        # Processing the call arguments (line 372)
        # Getting the type of 'macros' (line 372)
        macros_303706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 41), 'macros', False)
        # Getting the type of 'include_dirs' (line 372)
        include_dirs_303707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 49), 'include_dirs', False)
        # Processing the call keyword arguments (line 372)
        kwargs_303708 = {}
        # Getting the type of 'gen_preprocess_options' (line 372)
        gen_preprocess_options_303705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 18), 'gen_preprocess_options', False)
        # Calling gen_preprocess_options(args, kwargs) (line 372)
        gen_preprocess_options_call_result_303709 = invoke(stypy.reporting.localization.Localization(__file__, 372, 18), gen_preprocess_options_303705, *[macros_303706, include_dirs_303707], **kwargs_303708)
        
        # Assigning a type to the variable 'pp_opts' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'pp_opts', gen_preprocess_options_call_result_303709)
        
        # Assigning a BinOp to a Name (line 373):
        
        # Assigning a BinOp to a Name (line 373):
        
        # Obtaining an instance of the builtin type 'list' (line 373)
        list_303710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 373)
        # Adding element type (line 373)
        str_303711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 19), 'str', 'cpp32.exe')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 18), list_303710, str_303711)
        
        # Getting the type of 'pp_opts' (line 373)
        pp_opts_303712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 34), 'pp_opts')
        # Applying the binary operator '+' (line 373)
        result_add_303713 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 18), '+', list_303710, pp_opts_303712)
        
        # Assigning a type to the variable 'pp_args' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'pp_args', result_add_303713)
        
        # Type idiom detected: calculating its left and rigth part (line 374)
        # Getting the type of 'output_file' (line 374)
        output_file_303714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'output_file')
        # Getting the type of 'None' (line 374)
        None_303715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 30), 'None')
        
        (may_be_303716, more_types_in_union_303717) = may_not_be_none(output_file_303714, None_303715)

        if may_be_303716:

            if more_types_in_union_303717:
                # Runtime conditional SSA (line 374)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to append(...): (line 375)
            # Processing the call arguments (line 375)
            str_303720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 27), 'str', '-o')
            # Getting the type of 'output_file' (line 375)
            output_file_303721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 34), 'output_file', False)
            # Applying the binary operator '+' (line 375)
            result_add_303722 = python_operator(stypy.reporting.localization.Localization(__file__, 375, 27), '+', str_303720, output_file_303721)
            
            # Processing the call keyword arguments (line 375)
            kwargs_303723 = {}
            # Getting the type of 'pp_args' (line 375)
            pp_args_303718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'pp_args', False)
            # Obtaining the member 'append' of a type (line 375)
            append_303719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 12), pp_args_303718, 'append')
            # Calling append(args, kwargs) (line 375)
            append_call_result_303724 = invoke(stypy.reporting.localization.Localization(__file__, 375, 12), append_303719, *[result_add_303722], **kwargs_303723)
            

            if more_types_in_union_303717:
                # SSA join for if statement (line 374)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'extra_preargs' (line 376)
        extra_preargs_303725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 11), 'extra_preargs')
        # Testing the type of an if condition (line 376)
        if_condition_303726 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 376, 8), extra_preargs_303725)
        # Assigning a type to the variable 'if_condition_303726' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'if_condition_303726', if_condition_303726)
        # SSA begins for if statement (line 376)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 377):
        
        # Assigning a Name to a Subscript (line 377):
        # Getting the type of 'extra_preargs' (line 377)
        extra_preargs_303727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 26), 'extra_preargs')
        # Getting the type of 'pp_args' (line 377)
        pp_args_303728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'pp_args')
        int_303729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 21), 'int')
        slice_303730 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 377, 12), None, int_303729, None)
        # Storing an element on a container (line 377)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 377, 12), pp_args_303728, (slice_303730, extra_preargs_303727))
        # SSA join for if statement (line 376)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'extra_postargs' (line 378)
        extra_postargs_303731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 11), 'extra_postargs')
        # Testing the type of an if condition (line 378)
        if_condition_303732 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 378, 8), extra_postargs_303731)
        # Assigning a type to the variable 'if_condition_303732' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'if_condition_303732', if_condition_303732)
        # SSA begins for if statement (line 378)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 'extra_postargs' (line 379)
        extra_postargs_303735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 27), 'extra_postargs', False)
        # Processing the call keyword arguments (line 379)
        kwargs_303736 = {}
        # Getting the type of 'pp_args' (line 379)
        pp_args_303733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'pp_args', False)
        # Obtaining the member 'extend' of a type (line 379)
        extend_303734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 12), pp_args_303733, 'extend')
        # Calling extend(args, kwargs) (line 379)
        extend_call_result_303737 = invoke(stypy.reporting.localization.Localization(__file__, 379, 12), extend_303734, *[extra_postargs_303735], **kwargs_303736)
        
        # SSA join for if statement (line 378)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 380)
        # Processing the call arguments (line 380)
        # Getting the type of 'source' (line 380)
        source_303740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 23), 'source', False)
        # Processing the call keyword arguments (line 380)
        kwargs_303741 = {}
        # Getting the type of 'pp_args' (line 380)
        pp_args_303738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'pp_args', False)
        # Obtaining the member 'append' of a type (line 380)
        append_303739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 8), pp_args_303738, 'append')
        # Calling append(args, kwargs) (line 380)
        append_call_result_303742 = invoke(stypy.reporting.localization.Localization(__file__, 380, 8), append_303739, *[source_303740], **kwargs_303741)
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 385)
        self_303743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 11), 'self')
        # Obtaining the member 'force' of a type (line 385)
        force_303744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 11), self_303743, 'force')
        
        # Getting the type of 'output_file' (line 385)
        output_file_303745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 25), 'output_file')
        # Getting the type of 'None' (line 385)
        None_303746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 40), 'None')
        # Applying the binary operator 'is' (line 385)
        result_is__303747 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 25), 'is', output_file_303745, None_303746)
        
        # Applying the binary operator 'or' (line 385)
        result_or_keyword_303748 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 11), 'or', force_303744, result_is__303747)
        
        # Call to newer(...): (line 385)
        # Processing the call arguments (line 385)
        # Getting the type of 'source' (line 385)
        source_303750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 54), 'source', False)
        # Getting the type of 'output_file' (line 385)
        output_file_303751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 62), 'output_file', False)
        # Processing the call keyword arguments (line 385)
        kwargs_303752 = {}
        # Getting the type of 'newer' (line 385)
        newer_303749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 48), 'newer', False)
        # Calling newer(args, kwargs) (line 385)
        newer_call_result_303753 = invoke(stypy.reporting.localization.Localization(__file__, 385, 48), newer_303749, *[source_303750, output_file_303751], **kwargs_303752)
        
        # Applying the binary operator 'or' (line 385)
        result_or_keyword_303754 = python_operator(stypy.reporting.localization.Localization(__file__, 385, 11), 'or', result_or_keyword_303748, newer_call_result_303753)
        
        # Testing the type of an if condition (line 385)
        if_condition_303755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 385, 8), result_or_keyword_303754)
        # Assigning a type to the variable 'if_condition_303755' (line 385)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'if_condition_303755', if_condition_303755)
        # SSA begins for if statement (line 385)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'output_file' (line 386)
        output_file_303756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 15), 'output_file')
        # Testing the type of an if condition (line 386)
        if_condition_303757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 386, 12), output_file_303756)
        # Assigning a type to the variable 'if_condition_303757' (line 386)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 12), 'if_condition_303757', if_condition_303757)
        # SSA begins for if statement (line 386)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to mkpath(...): (line 387)
        # Processing the call arguments (line 387)
        
        # Call to dirname(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 'output_file' (line 387)
        output_file_303763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 44), 'output_file', False)
        # Processing the call keyword arguments (line 387)
        kwargs_303764 = {}
        # Getting the type of 'os' (line 387)
        os_303760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 387)
        path_303761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 28), os_303760, 'path')
        # Obtaining the member 'dirname' of a type (line 387)
        dirname_303762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 28), path_303761, 'dirname')
        # Calling dirname(args, kwargs) (line 387)
        dirname_call_result_303765 = invoke(stypy.reporting.localization.Localization(__file__, 387, 28), dirname_303762, *[output_file_303763], **kwargs_303764)
        
        # Processing the call keyword arguments (line 387)
        kwargs_303766 = {}
        # Getting the type of 'self' (line 387)
        self_303758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 16), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 387)
        mkpath_303759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 16), self_303758, 'mkpath')
        # Calling mkpath(args, kwargs) (line 387)
        mkpath_call_result_303767 = invoke(stypy.reporting.localization.Localization(__file__, 387, 16), mkpath_303759, *[dirname_call_result_303765], **kwargs_303766)
        
        # SSA join for if statement (line 386)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 388)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 'pp_args' (line 389)
        pp_args_303770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 27), 'pp_args', False)
        # Processing the call keyword arguments (line 389)
        kwargs_303771 = {}
        # Getting the type of 'self' (line 389)
        self_303768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 389)
        spawn_303769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 16), self_303768, 'spawn')
        # Calling spawn(args, kwargs) (line 389)
        spawn_call_result_303772 = invoke(stypy.reporting.localization.Localization(__file__, 389, 16), spawn_303769, *[pp_args_303770], **kwargs_303771)
        
        # SSA branch for the except part of a try statement (line 388)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 388)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 390)
        DistutilsExecError_303773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'msg', DistutilsExecError_303773)
        # Getting the type of 'msg' (line 391)
        msg_303774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 22), 'msg')
        # Getting the type of 'CompileError' (line 392)
        CompileError_303775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 22), 'CompileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 392, 16), CompileError_303775, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 388)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 385)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'preprocess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'preprocess' in the type store
        # Getting the type of 'stypy_return_type' (line 362)
        stypy_return_type_303776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_303776)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'preprocess'
        return stypy_return_type_303776


# Assigning a type to the variable 'BCPPCompiler' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'BCPPCompiler', BCPPCompiler)

# Assigning a Str to a Name (line 30):
str_303777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 20), 'str', 'bcpp')
# Getting the type of 'BCPPCompiler'
BCPPCompiler_303778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BCPPCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BCPPCompiler_303778, 'compiler_type', str_303777)

# Assigning a Dict to a Name (line 37):

# Obtaining an instance of the builtin type 'dict' (line 37)
dict_303779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 37)

# Getting the type of 'BCPPCompiler'
BCPPCompiler_303780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BCPPCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BCPPCompiler_303780, 'executables', dict_303779)

# Assigning a List to a Name (line 40):

# Obtaining an instance of the builtin type 'list' (line 40)
list_303781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 40)
# Adding element type (line 40)
str_303782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 21), 'str', '.c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 20), list_303781, str_303782)

# Getting the type of 'BCPPCompiler'
BCPPCompiler_303783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BCPPCompiler')
# Setting the type of the member '_c_extensions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BCPPCompiler_303783, '_c_extensions', list_303781)

# Assigning a List to a Name (line 41):

# Obtaining an instance of the builtin type 'list' (line 41)
list_303784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 41)
# Adding element type (line 41)
str_303785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 23), 'str', '.cc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 22), list_303784, str_303785)
# Adding element type (line 41)
str_303786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'str', '.cpp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 22), list_303784, str_303786)
# Adding element type (line 41)
str_303787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 38), 'str', '.cxx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 22), list_303784, str_303787)

# Getting the type of 'BCPPCompiler'
BCPPCompiler_303788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BCPPCompiler')
# Setting the type of the member '_cpp_extensions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BCPPCompiler_303788, '_cpp_extensions', list_303784)

# Assigning a BinOp to a Name (line 45):
# Getting the type of 'BCPPCompiler'
BCPPCompiler_303789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BCPPCompiler')
# Obtaining the member '_c_extensions' of a type
_c_extensions_303790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BCPPCompiler_303789, '_c_extensions')
# Getting the type of 'BCPPCompiler'
BCPPCompiler_303791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BCPPCompiler')
# Obtaining the member '_cpp_extensions' of a type
_cpp_extensions_303792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BCPPCompiler_303791, '_cpp_extensions')
# Applying the binary operator '+' (line 45)
result_add_303793 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 21), '+', _c_extensions_303790, _cpp_extensions_303792)

# Getting the type of 'BCPPCompiler'
BCPPCompiler_303794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BCPPCompiler')
# Setting the type of the member 'src_extensions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BCPPCompiler_303794, 'src_extensions', result_add_303793)

# Assigning a Str to a Name (line 46):
str_303795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 20), 'str', '.obj')
# Getting the type of 'BCPPCompiler'
BCPPCompiler_303796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BCPPCompiler')
# Setting the type of the member 'obj_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BCPPCompiler_303796, 'obj_extension', str_303795)

# Assigning a Str to a Name (line 47):
str_303797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 27), 'str', '.lib')
# Getting the type of 'BCPPCompiler'
BCPPCompiler_303798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BCPPCompiler')
# Setting the type of the member 'static_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BCPPCompiler_303798, 'static_lib_extension', str_303797)

# Assigning a Str to a Name (line 48):
str_303799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'str', '.dll')
# Getting the type of 'BCPPCompiler'
BCPPCompiler_303800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BCPPCompiler')
# Setting the type of the member 'shared_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BCPPCompiler_303800, 'shared_lib_extension', str_303799)

# Assigning a Str to a Name (line 49):
str_303801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 44), 'str', '%s%s')
# Getting the type of 'BCPPCompiler'
BCPPCompiler_303802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BCPPCompiler')
# Setting the type of the member 'shared_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BCPPCompiler_303802, 'shared_lib_format', str_303801)

# Assigning a Name to a Name (line 49):
# Getting the type of 'BCPPCompiler'
BCPPCompiler_303803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BCPPCompiler')
# Obtaining the member 'shared_lib_format' of a type
shared_lib_format_303804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BCPPCompiler_303803, 'shared_lib_format')
# Getting the type of 'BCPPCompiler'
BCPPCompiler_303805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BCPPCompiler')
# Setting the type of the member 'static_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BCPPCompiler_303805, 'static_lib_format', shared_lib_format_303804)

# Assigning a Str to a Name (line 50):
str_303806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 20), 'str', '.exe')
# Getting the type of 'BCPPCompiler'
BCPPCompiler_303807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'BCPPCompiler')
# Setting the type of the member 'exe_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), BCPPCompiler_303807, 'exe_extension', str_303806)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
