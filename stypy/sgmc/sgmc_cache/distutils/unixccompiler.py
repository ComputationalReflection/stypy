
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.unixccompiler
2: 
3: Contains the UnixCCompiler class, a subclass of CCompiler that handles
4: the "typical" Unix-style command-line C compiler:
5:   * macros defined with -Dname[=value]
6:   * macros undefined with -Uname
7:   * include search directories specified with -Idir
8:   * libraries specified with -lllib
9:   * library search directories specified with -Ldir
10:   * compile handled by 'cc' (or similar) executable with -c option:
11:     compiles .c to .o
12:   * link static library handled by 'ar' command (possibly with 'ranlib')
13:   * link shared library handled by 'cc -shared'
14: '''
15: 
16: __revision__ = "$Id$"
17: 
18: import os, sys, re
19: from types import StringType, NoneType
20: 
21: from distutils import sysconfig
22: from distutils.dep_util import newer
23: from distutils.ccompiler import \
24:      CCompiler, gen_preprocess_options, gen_lib_options
25: from distutils.errors import \
26:      DistutilsExecError, CompileError, LibError, LinkError
27: from distutils import log
28: 
29: if sys.platform == 'darwin':
30:     import _osx_support
31: 
32: # XXX Things not currently handled:
33: #   * optimization/debug/warning flags; we just use whatever's in Python's
34: #     Makefile and live with it.  Is this adequate?  If not, we might
35: #     have to have a bunch of subclasses GNUCCompiler, SGICCompiler,
36: #     SunCCompiler, and I suspect down that road lies madness.
37: #   * even if we don't know a warning flag from an optimization flag,
38: #     we need some way for outsiders to feed preprocessor/compiler/linker
39: #     flags in to us -- eg. a sysadmin might want to mandate certain flags
40: #     via a site config file, or a user might want to set something for
41: #     compiling this module distribution only via the setup.py command
42: #     line, whatever.  As long as these options come from something on the
43: #     current system, they can be as system-dependent as they like, and we
44: #     should just happily stuff them into the preprocessor/compiler/linker
45: #     options and carry on.
46: 
47: 
48: class UnixCCompiler(CCompiler):
49: 
50:     compiler_type = 'unix'
51: 
52:     # These are used by CCompiler in two places: the constructor sets
53:     # instance attributes 'preprocessor', 'compiler', etc. from them, and
54:     # 'set_executable()' allows any of these to be set.  The defaults here
55:     # are pretty generic; they will probably have to be set by an outsider
56:     # (eg. using information discovered by the sysconfig about building
57:     # Python extensions).
58:     executables = {'preprocessor' : None,
59:                    'compiler'     : ["cc"],
60:                    'compiler_so'  : ["cc"],
61:                    'compiler_cxx' : ["cc"],
62:                    'linker_so'    : ["cc", "-shared"],
63:                    'linker_exe'   : ["cc"],
64:                    'archiver'     : ["ar", "-cr"],
65:                    'ranlib'       : None,
66:                   }
67: 
68:     if sys.platform[:6] == "darwin":
69:         executables['ranlib'] = ["ranlib"]
70: 
71:     # Needed for the filename generation methods provided by the base
72:     # class, CCompiler.  NB. whoever instantiates/uses a particular
73:     # UnixCCompiler instance should set 'shared_lib_ext' -- we set a
74:     # reasonable common default here, but it's not necessarily used on all
75:     # Unices!
76: 
77:     src_extensions = [".c",".C",".cc",".cxx",".cpp",".m"]
78:     obj_extension = ".o"
79:     static_lib_extension = ".a"
80:     shared_lib_extension = ".so"
81:     dylib_lib_extension = ".dylib"
82:     xcode_stub_lib_extension = ".tbd"
83:     static_lib_format = shared_lib_format = dylib_lib_format = "lib%s%s"
84:     xcode_stub_lib_format = dylib_lib_format
85:     if sys.platform == "cygwin":
86:         exe_extension = ".exe"
87: 
88:     def preprocess(self, source,
89:                    output_file=None, macros=None, include_dirs=None,
90:                    extra_preargs=None, extra_postargs=None):
91:         ignore, macros, include_dirs = \
92:             self._fix_compile_args(None, macros, include_dirs)
93:         pp_opts = gen_preprocess_options(macros, include_dirs)
94:         pp_args = self.preprocessor + pp_opts
95:         if output_file:
96:             pp_args.extend(['-o', output_file])
97:         if extra_preargs:
98:             pp_args[:0] = extra_preargs
99:         if extra_postargs:
100:             pp_args.extend(extra_postargs)
101:         pp_args.append(source)
102: 
103:         # We need to preprocess: either we're being forced to, or we're
104:         # generating output to stdout, or there's a target output file and
105:         # the source file is newer than the target (or the target doesn't
106:         # exist).
107:         if self.force or output_file is None or newer(source, output_file):
108:             if output_file:
109:                 self.mkpath(os.path.dirname(output_file))
110:             try:
111:                 self.spawn(pp_args)
112:             except DistutilsExecError, msg:
113:                 raise CompileError, msg
114: 
115:     def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
116:         compiler_so = self.compiler_so
117:         if sys.platform == 'darwin':
118:             compiler_so = _osx_support.compiler_fixup(compiler_so,
119:                                                     cc_args + extra_postargs)
120:         try:
121:             self.spawn(compiler_so + cc_args + [src, '-o', obj] +
122:                        extra_postargs)
123:         except DistutilsExecError, msg:
124:             raise CompileError, msg
125: 
126:     def create_static_lib(self, objects, output_libname,
127:                           output_dir=None, debug=0, target_lang=None):
128:         objects, output_dir = self._fix_object_args(objects, output_dir)
129: 
130:         output_filename = \
131:             self.library_filename(output_libname, output_dir=output_dir)
132: 
133:         if self._need_link(objects, output_filename):
134:             self.mkpath(os.path.dirname(output_filename))
135:             self.spawn(self.archiver +
136:                        [output_filename] +
137:                        objects + self.objects)
138: 
139:             # Not many Unices required ranlib anymore -- SunOS 4.x is, I
140:             # think the only major Unix that does.  Maybe we need some
141:             # platform intelligence here to skip ranlib if it's not
142:             # needed -- or maybe Python's configure script took care of
143:             # it for us, hence the check for leading colon.
144:             if self.ranlib:
145:                 try:
146:                     self.spawn(self.ranlib + [output_filename])
147:                 except DistutilsExecError, msg:
148:                     raise LibError, msg
149:         else:
150:             log.debug("skipping %s (up-to-date)", output_filename)
151: 
152:     def link(self, target_desc, objects,
153:              output_filename, output_dir=None, libraries=None,
154:              library_dirs=None, runtime_library_dirs=None,
155:              export_symbols=None, debug=0, extra_preargs=None,
156:              extra_postargs=None, build_temp=None, target_lang=None):
157:         objects, output_dir = self._fix_object_args(objects, output_dir)
158:         libraries, library_dirs, runtime_library_dirs = \
159:             self._fix_lib_args(libraries, library_dirs, runtime_library_dirs)
160: 
161:         lib_opts = gen_lib_options(self, library_dirs, runtime_library_dirs,
162:                                    libraries)
163:         if type(output_dir) not in (StringType, NoneType):
164:             raise TypeError, "'output_dir' must be a string or None"
165:         if output_dir is not None:
166:             output_filename = os.path.join(output_dir, output_filename)
167: 
168:         if self._need_link(objects, output_filename):
169:             ld_args = (objects + self.objects +
170:                        lib_opts + ['-o', output_filename])
171:             if debug:
172:                 ld_args[:0] = ['-g']
173:             if extra_preargs:
174:                 ld_args[:0] = extra_preargs
175:             if extra_postargs:
176:                 ld_args.extend(extra_postargs)
177:             self.mkpath(os.path.dirname(output_filename))
178:             try:
179:                 if target_desc == CCompiler.EXECUTABLE:
180:                     linker = self.linker_exe[:]
181:                 else:
182:                     linker = self.linker_so[:]
183:                 if target_lang == "c++" and self.compiler_cxx:
184:                     # skip over environment variable settings if /usr/bin/env
185:                     # is used to set up the linker's environment.
186:                     # This is needed on OSX. Note: this assumes that the
187:                     # normal and C++ compiler have the same environment
188:                     # settings.
189:                     i = 0
190:                     if os.path.basename(linker[0]) == "env":
191:                         i = 1
192:                         while '=' in linker[i]:
193:                             i = i + 1
194: 
195:                     linker[i] = self.compiler_cxx[i]
196: 
197:                 if sys.platform == 'darwin':
198:                     linker = _osx_support.compiler_fixup(linker, ld_args)
199: 
200:                 self.spawn(linker + ld_args)
201:             except DistutilsExecError, msg:
202:                 raise LinkError, msg
203:         else:
204:             log.debug("skipping %s (up-to-date)", output_filename)
205: 
206:     # -- Miscellaneous methods -----------------------------------------
207:     # These are all used by the 'gen_lib_options() function, in
208:     # ccompiler.py.
209: 
210:     def library_dir_option(self, dir):
211:         return "-L" + dir
212: 
213:     def _is_gcc(self, compiler_name):
214:         return "gcc" in compiler_name or "g++" in compiler_name
215: 
216:     def runtime_library_dir_option(self, dir):
217:         # XXX Hackish, at the very least.  See Python bug #445902:
218:         # http://sourceforge.net/tracker/index.php
219:         #   ?func=detail&aid=445902&group_id=5470&atid=105470
220:         # Linkers on different platforms need different options to
221:         # specify that directories need to be added to the list of
222:         # directories searched for dependencies when a dynamic library
223:         # is sought.  GCC has to be told to pass the -R option through
224:         # to the linker, whereas other compilers just know this.
225:         # Other compilers may need something slightly different.  At
226:         # this time, there's no way to determine this information from
227:         # the configuration data stored in the Python installation, so
228:         # we use this hack.
229:         compiler = os.path.basename(sysconfig.get_config_var("CC"))
230:         if sys.platform[:6] == "darwin":
231:             # MacOSX's linker doesn't understand the -R flag at all
232:             return "-L" + dir
233:         elif sys.platform[:7] == "freebsd":
234:             return "-Wl,-rpath=" + dir
235:         elif sys.platform[:5] == "hp-ux":
236:             if self._is_gcc(compiler):
237:                 return ["-Wl,+s", "-L" + dir]
238:             return ["+s", "-L" + dir]
239:         elif sys.platform[:7] == "irix646" or sys.platform[:6] == "osf1V5":
240:             return ["-rpath", dir]
241:         elif self._is_gcc(compiler):
242:             return "-Wl,-R" + dir
243:         else:
244:             return "-R" + dir
245: 
246:     def library_option(self, lib):
247:         return "-l" + lib
248: 
249:     def find_library_file(self, dirs, lib, debug=0):
250:         shared_f = self.library_filename(lib, lib_type='shared')
251:         dylib_f = self.library_filename(lib, lib_type='dylib')
252:         xcode_stub_f = self.library_filename(lib, lib_type='xcode_stub')
253:         static_f = self.library_filename(lib, lib_type='static')
254: 
255:         if sys.platform == 'darwin':
256:             # On OSX users can specify an alternate SDK using
257:             # '-isysroot', calculate the SDK root if it is specified
258:             # (and use it further on)
259:             #
260:             # Note that, as of Xcode 7, Apple SDKs may contain textual stub
261:             # libraries with .tbd extensions rather than the normal .dylib
262:             # shared libraries installed in /.  The Apple compiler tool
263:             # chain handles this transparently but it can cause problems
264:             # for programs that are being built with an SDK and searching
265:             # for specific libraries.  Callers of find_library_file need to
266:             # keep in mind that the base filename of the returned SDK library
267:             # file might have a different extension from that of the library
268:             # file installed on the running system, for example:
269:             #   /Applications/Xcode.app/Contents/Developer/Platforms/
270:             #       MacOSX.platform/Developer/SDKs/MacOSX10.11.sdk/
271:             #       usr/lib/libedit.tbd
272:             # vs
273:             #   /usr/lib/libedit.dylib
274:             cflags = sysconfig.get_config_var('CFLAGS')
275:             m = re.search(r'-isysroot\s+(\S+)', cflags)
276:             if m is None:
277:                 sysroot = '/'
278:             else:
279:                 sysroot = m.group(1)
280: 
281: 
282: 
283:         for dir in dirs:
284:             shared = os.path.join(dir, shared_f)
285:             dylib = os.path.join(dir, dylib_f)
286:             static = os.path.join(dir, static_f)
287:             xcode_stub = os.path.join(dir, xcode_stub_f)
288: 
289:             if sys.platform == 'darwin' and (
290:                 dir.startswith('/System/') or (
291:                 dir.startswith('/usr/') and not dir.startswith('/usr/local/'))):
292: 
293:                 shared = os.path.join(sysroot, dir[1:], shared_f)
294:                 dylib = os.path.join(sysroot, dir[1:], dylib_f)
295:                 static = os.path.join(sysroot, dir[1:], static_f)
296:                 xcode_stub = os.path.join(sysroot, dir[1:], xcode_stub_f)
297: 
298:             # We're second-guessing the linker here, with not much hard
299:             # data to go on: GCC seems to prefer the shared library, so I'm
300:             # assuming that *all* Unix C compilers do.  And of course I'm
301:             # ignoring even GCC's "-static" option.  So sue me.
302:             if os.path.exists(dylib):
303:                 return dylib
304:             elif os.path.exists(xcode_stub):
305:                 return xcode_stub
306:             elif os.path.exists(shared):
307:                 return shared
308:             elif os.path.exists(static):
309:                 return static
310: 
311:         # Oops, didn't find it in *any* of 'dirs'
312:         return None
313: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_9051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, (-1)), 'str', 'distutils.unixccompiler\n\nContains the UnixCCompiler class, a subclass of CCompiler that handles\nthe "typical" Unix-style command-line C compiler:\n  * macros defined with -Dname[=value]\n  * macros undefined with -Uname\n  * include search directories specified with -Idir\n  * libraries specified with -lllib\n  * library search directories specified with -Ldir\n  * compile handled by \'cc\' (or similar) executable with -c option:\n    compiles .c to .o\n  * link static library handled by \'ar\' command (possibly with \'ranlib\')\n  * link shared library handled by \'cc -shared\'\n')

# Assigning a Str to a Name (line 16):

# Assigning a Str to a Name (line 16):
str_9052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '__revision__', str_9052)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# Multiple import statement. import os (1/3) (line 18)
import os

import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'os', os, module_type_store)
# Multiple import statement. import sys (2/3) (line 18)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'sys', sys, module_type_store)
# Multiple import statement. import re (3/3) (line 18)
import re

import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from types import StringType, NoneType' statement (line 19)
try:
    from types import StringType, NoneType

except:
    StringType = UndefinedType
    NoneType = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'types', None, module_type_store, ['StringType', 'NoneType'], [StringType, NoneType])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from distutils import sysconfig' statement (line 21)
try:
    from distutils import sysconfig

except:
    sysconfig = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils', None, module_type_store, ['sysconfig'], [sysconfig])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from distutils.dep_util import newer' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_9053 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.dep_util')

if (type(import_9053) is not StypyTypeError):

    if (import_9053 != 'pyd_module'):
        __import__(import_9053)
        sys_modules_9054 = sys.modules[import_9053]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.dep_util', sys_modules_9054.module_type_store, module_type_store, ['newer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_9054, sys_modules_9054.module_type_store, module_type_store)
    else:
        from distutils.dep_util import newer

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.dep_util', None, module_type_store, ['newer'], [newer])

else:
    # Assigning a type to the variable 'distutils.dep_util' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.dep_util', import_9053)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from distutils.ccompiler import CCompiler, gen_preprocess_options, gen_lib_options' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_9055 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'distutils.ccompiler')

if (type(import_9055) is not StypyTypeError):

    if (import_9055 != 'pyd_module'):
        __import__(import_9055)
        sys_modules_9056 = sys.modules[import_9055]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'distutils.ccompiler', sys_modules_9056.module_type_store, module_type_store, ['CCompiler', 'gen_preprocess_options', 'gen_lib_options'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 0), __file__, sys_modules_9056, sys_modules_9056.module_type_store, module_type_store)
    else:
        from distutils.ccompiler import CCompiler, gen_preprocess_options, gen_lib_options

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'distutils.ccompiler', None, module_type_store, ['CCompiler', 'gen_preprocess_options', 'gen_lib_options'], [CCompiler, gen_preprocess_options, gen_lib_options])

else:
    # Assigning a type to the variable 'distutils.ccompiler' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'distutils.ccompiler', import_9055)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from distutils.errors import DistutilsExecError, CompileError, LibError, LinkError' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_9057 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'distutils.errors')

if (type(import_9057) is not StypyTypeError):

    if (import_9057 != 'pyd_module'):
        __import__(import_9057)
        sys_modules_9058 = sys.modules[import_9057]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'distutils.errors', sys_modules_9058.module_type_store, module_type_store, ['DistutilsExecError', 'CompileError', 'LibError', 'LinkError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_9058, sys_modules_9058.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsExecError, CompileError, LibError, LinkError

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'distutils.errors', None, module_type_store, ['DistutilsExecError', 'CompileError', 'LibError', 'LinkError'], [DistutilsExecError, CompileError, LibError, LinkError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'distutils.errors', import_9057)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from distutils import log' statement (line 27)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'distutils', None, module_type_store, ['log'], [log])



# Getting the type of 'sys' (line 29)
sys_9059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 3), 'sys')
# Obtaining the member 'platform' of a type (line 29)
platform_9060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 3), sys_9059, 'platform')
str_9061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 19), 'str', 'darwin')
# Applying the binary operator '==' (line 29)
result_eq_9062 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 3), '==', platform_9060, str_9061)

# Testing the type of an if condition (line 29)
if_condition_9063 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 0), result_eq_9062)
# Assigning a type to the variable 'if_condition_9063' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'if_condition_9063', if_condition_9063)
# SSA begins for if statement (line 29)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 4))

# 'import _osx_support' statement (line 30)
import _osx_support

import_module(stypy.reporting.localization.Localization(__file__, 30, 4), '_osx_support', _osx_support, module_type_store)

# SSA join for if statement (line 29)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'UnixCCompiler' class
# Getting the type of 'CCompiler' (line 48)
CCompiler_9064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'CCompiler')

class UnixCCompiler(CCompiler_9064, ):
    
    # Assigning a Str to a Name (line 50):
    
    # Assigning a List to a Name (line 77):
    
    # Assigning a Str to a Name (line 78):
    
    # Assigning a Str to a Name (line 79):
    
    # Assigning a Str to a Name (line 80):
    
    # Assigning a Str to a Name (line 81):
    
    # Assigning a Str to a Name (line 82):
    
    # Multiple assignment of 3 elements.

    @norecursion
    def preprocess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 89)
        None_9065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 31), 'None')
        # Getting the type of 'None' (line 89)
        None_9066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 44), 'None')
        # Getting the type of 'None' (line 89)
        None_9067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 63), 'None')
        # Getting the type of 'None' (line 90)
        None_9068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 33), 'None')
        # Getting the type of 'None' (line 90)
        None_9069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 54), 'None')
        defaults = [None_9065, None_9066, None_9067, None_9068, None_9069]
        # Create a new context for function 'preprocess'
        module_type_store = module_type_store.open_function_context('preprocess', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnixCCompiler.preprocess.__dict__.__setitem__('stypy_localization', localization)
        UnixCCompiler.preprocess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnixCCompiler.preprocess.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnixCCompiler.preprocess.__dict__.__setitem__('stypy_function_name', 'UnixCCompiler.preprocess')
        UnixCCompiler.preprocess.__dict__.__setitem__('stypy_param_names_list', ['source', 'output_file', 'macros', 'include_dirs', 'extra_preargs', 'extra_postargs'])
        UnixCCompiler.preprocess.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnixCCompiler.preprocess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnixCCompiler.preprocess.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnixCCompiler.preprocess.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnixCCompiler.preprocess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnixCCompiler.preprocess.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnixCCompiler.preprocess', ['source', 'output_file', 'macros', 'include_dirs', 'extra_preargs', 'extra_postargs'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Tuple (line 91):
        
        # Assigning a Subscript to a Name (line 91):
        
        # Obtaining the type of the subscript
        int_9070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
        
        # Call to _fix_compile_args(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'None' (line 92)
        None_9073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 35), 'None', False)
        # Getting the type of 'macros' (line 92)
        macros_9074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 41), 'macros', False)
        # Getting the type of 'include_dirs' (line 92)
        include_dirs_9075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 49), 'include_dirs', False)
        # Processing the call keyword arguments (line 92)
        kwargs_9076 = {}
        # Getting the type of 'self' (line 92)
        self_9071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'self', False)
        # Obtaining the member '_fix_compile_args' of a type (line 92)
        _fix_compile_args_9072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), self_9071, '_fix_compile_args')
        # Calling _fix_compile_args(args, kwargs) (line 92)
        _fix_compile_args_call_result_9077 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), _fix_compile_args_9072, *[None_9073, macros_9074, include_dirs_9075], **kwargs_9076)
        
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___9078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), _fix_compile_args_call_result_9077, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_9079 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), getitem___9078, int_9070)
        
        # Assigning a type to the variable 'tuple_var_assignment_9041' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_9041', subscript_call_result_9079)
        
        # Assigning a Subscript to a Name (line 91):
        
        # Obtaining the type of the subscript
        int_9080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
        
        # Call to _fix_compile_args(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'None' (line 92)
        None_9083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 35), 'None', False)
        # Getting the type of 'macros' (line 92)
        macros_9084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 41), 'macros', False)
        # Getting the type of 'include_dirs' (line 92)
        include_dirs_9085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 49), 'include_dirs', False)
        # Processing the call keyword arguments (line 92)
        kwargs_9086 = {}
        # Getting the type of 'self' (line 92)
        self_9081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'self', False)
        # Obtaining the member '_fix_compile_args' of a type (line 92)
        _fix_compile_args_9082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), self_9081, '_fix_compile_args')
        # Calling _fix_compile_args(args, kwargs) (line 92)
        _fix_compile_args_call_result_9087 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), _fix_compile_args_9082, *[None_9083, macros_9084, include_dirs_9085], **kwargs_9086)
        
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___9088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), _fix_compile_args_call_result_9087, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_9089 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), getitem___9088, int_9080)
        
        # Assigning a type to the variable 'tuple_var_assignment_9042' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_9042', subscript_call_result_9089)
        
        # Assigning a Subscript to a Name (line 91):
        
        # Obtaining the type of the subscript
        int_9090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'int')
        
        # Call to _fix_compile_args(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'None' (line 92)
        None_9093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 35), 'None', False)
        # Getting the type of 'macros' (line 92)
        macros_9094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 41), 'macros', False)
        # Getting the type of 'include_dirs' (line 92)
        include_dirs_9095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 49), 'include_dirs', False)
        # Processing the call keyword arguments (line 92)
        kwargs_9096 = {}
        # Getting the type of 'self' (line 92)
        self_9091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'self', False)
        # Obtaining the member '_fix_compile_args' of a type (line 92)
        _fix_compile_args_9092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), self_9091, '_fix_compile_args')
        # Calling _fix_compile_args(args, kwargs) (line 92)
        _fix_compile_args_call_result_9097 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), _fix_compile_args_9092, *[None_9093, macros_9094, include_dirs_9095], **kwargs_9096)
        
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___9098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), _fix_compile_args_call_result_9097, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_9099 = invoke(stypy.reporting.localization.Localization(__file__, 91, 8), getitem___9098, int_9090)
        
        # Assigning a type to the variable 'tuple_var_assignment_9043' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_9043', subscript_call_result_9099)
        
        # Assigning a Name to a Name (line 91):
        # Getting the type of 'tuple_var_assignment_9041' (line 91)
        tuple_var_assignment_9041_9100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_9041')
        # Assigning a type to the variable 'ignore' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'ignore', tuple_var_assignment_9041_9100)
        
        # Assigning a Name to a Name (line 91):
        # Getting the type of 'tuple_var_assignment_9042' (line 91)
        tuple_var_assignment_9042_9101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_9042')
        # Assigning a type to the variable 'macros' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'macros', tuple_var_assignment_9042_9101)
        
        # Assigning a Name to a Name (line 91):
        # Getting the type of 'tuple_var_assignment_9043' (line 91)
        tuple_var_assignment_9043_9102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tuple_var_assignment_9043')
        # Assigning a type to the variable 'include_dirs' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'include_dirs', tuple_var_assignment_9043_9102)
        
        # Assigning a Call to a Name (line 93):
        
        # Assigning a Call to a Name (line 93):
        
        # Call to gen_preprocess_options(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'macros' (line 93)
        macros_9104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 41), 'macros', False)
        # Getting the type of 'include_dirs' (line 93)
        include_dirs_9105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 49), 'include_dirs', False)
        # Processing the call keyword arguments (line 93)
        kwargs_9106 = {}
        # Getting the type of 'gen_preprocess_options' (line 93)
        gen_preprocess_options_9103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), 'gen_preprocess_options', False)
        # Calling gen_preprocess_options(args, kwargs) (line 93)
        gen_preprocess_options_call_result_9107 = invoke(stypy.reporting.localization.Localization(__file__, 93, 18), gen_preprocess_options_9103, *[macros_9104, include_dirs_9105], **kwargs_9106)
        
        # Assigning a type to the variable 'pp_opts' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'pp_opts', gen_preprocess_options_call_result_9107)
        
        # Assigning a BinOp to a Name (line 94):
        
        # Assigning a BinOp to a Name (line 94):
        # Getting the type of 'self' (line 94)
        self_9108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 18), 'self')
        # Obtaining the member 'preprocessor' of a type (line 94)
        preprocessor_9109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 18), self_9108, 'preprocessor')
        # Getting the type of 'pp_opts' (line 94)
        pp_opts_9110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 38), 'pp_opts')
        # Applying the binary operator '+' (line 94)
        result_add_9111 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 18), '+', preprocessor_9109, pp_opts_9110)
        
        # Assigning a type to the variable 'pp_args' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'pp_args', result_add_9111)
        
        # Getting the type of 'output_file' (line 95)
        output_file_9112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'output_file')
        # Testing the type of an if condition (line 95)
        if_condition_9113 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 8), output_file_9112)
        # Assigning a type to the variable 'if_condition_9113' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'if_condition_9113', if_condition_9113)
        # SSA begins for if statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Obtaining an instance of the builtin type 'list' (line 96)
        list_9116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 96)
        # Adding element type (line 96)
        str_9117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 28), 'str', '-o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 27), list_9116, str_9117)
        # Adding element type (line 96)
        # Getting the type of 'output_file' (line 96)
        output_file_9118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 34), 'output_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 27), list_9116, output_file_9118)
        
        # Processing the call keyword arguments (line 96)
        kwargs_9119 = {}
        # Getting the type of 'pp_args' (line 96)
        pp_args_9114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'pp_args', False)
        # Obtaining the member 'extend' of a type (line 96)
        extend_9115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), pp_args_9114, 'extend')
        # Calling extend(args, kwargs) (line 96)
        extend_call_result_9120 = invoke(stypy.reporting.localization.Localization(__file__, 96, 12), extend_9115, *[list_9116], **kwargs_9119)
        
        # SSA join for if statement (line 95)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'extra_preargs' (line 97)
        extra_preargs_9121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'extra_preargs')
        # Testing the type of an if condition (line 97)
        if_condition_9122 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), extra_preargs_9121)
        # Assigning a type to the variable 'if_condition_9122' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_9122', if_condition_9122)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 98):
        
        # Assigning a Name to a Subscript (line 98):
        # Getting the type of 'extra_preargs' (line 98)
        extra_preargs_9123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 26), 'extra_preargs')
        # Getting the type of 'pp_args' (line 98)
        pp_args_9124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'pp_args')
        int_9125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 21), 'int')
        slice_9126 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 98, 12), None, int_9125, None)
        # Storing an element on a container (line 98)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 12), pp_args_9124, (slice_9126, extra_preargs_9123))
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'extra_postargs' (line 99)
        extra_postargs_9127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'extra_postargs')
        # Testing the type of an if condition (line 99)
        if_condition_9128 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 8), extra_postargs_9127)
        # Assigning a type to the variable 'if_condition_9128' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'if_condition_9128', if_condition_9128)
        # SSA begins for if statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'extra_postargs' (line 100)
        extra_postargs_9131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'extra_postargs', False)
        # Processing the call keyword arguments (line 100)
        kwargs_9132 = {}
        # Getting the type of 'pp_args' (line 100)
        pp_args_9129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'pp_args', False)
        # Obtaining the member 'extend' of a type (line 100)
        extend_9130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), pp_args_9129, 'extend')
        # Calling extend(args, kwargs) (line 100)
        extend_call_result_9133 = invoke(stypy.reporting.localization.Localization(__file__, 100, 12), extend_9130, *[extra_postargs_9131], **kwargs_9132)
        
        # SSA join for if statement (line 99)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'source' (line 101)
        source_9136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'source', False)
        # Processing the call keyword arguments (line 101)
        kwargs_9137 = {}
        # Getting the type of 'pp_args' (line 101)
        pp_args_9134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'pp_args', False)
        # Obtaining the member 'append' of a type (line 101)
        append_9135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), pp_args_9134, 'append')
        # Calling append(args, kwargs) (line 101)
        append_call_result_9138 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), append_9135, *[source_9136], **kwargs_9137)
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 107)
        self_9139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'self')
        # Obtaining the member 'force' of a type (line 107)
        force_9140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 11), self_9139, 'force')
        
        # Getting the type of 'output_file' (line 107)
        output_file_9141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 25), 'output_file')
        # Getting the type of 'None' (line 107)
        None_9142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 40), 'None')
        # Applying the binary operator 'is' (line 107)
        result_is__9143 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 25), 'is', output_file_9141, None_9142)
        
        # Applying the binary operator 'or' (line 107)
        result_or_keyword_9144 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 11), 'or', force_9140, result_is__9143)
        
        # Call to newer(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'source' (line 107)
        source_9146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 54), 'source', False)
        # Getting the type of 'output_file' (line 107)
        output_file_9147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 62), 'output_file', False)
        # Processing the call keyword arguments (line 107)
        kwargs_9148 = {}
        # Getting the type of 'newer' (line 107)
        newer_9145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 48), 'newer', False)
        # Calling newer(args, kwargs) (line 107)
        newer_call_result_9149 = invoke(stypy.reporting.localization.Localization(__file__, 107, 48), newer_9145, *[source_9146, output_file_9147], **kwargs_9148)
        
        # Applying the binary operator 'or' (line 107)
        result_or_keyword_9150 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 11), 'or', result_or_keyword_9144, newer_call_result_9149)
        
        # Testing the type of an if condition (line 107)
        if_condition_9151 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 8), result_or_keyword_9150)
        # Assigning a type to the variable 'if_condition_9151' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'if_condition_9151', if_condition_9151)
        # SSA begins for if statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'output_file' (line 108)
        output_file_9152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'output_file')
        # Testing the type of an if condition (line 108)
        if_condition_9153 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 12), output_file_9152)
        # Assigning a type to the variable 'if_condition_9153' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'if_condition_9153', if_condition_9153)
        # SSA begins for if statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to mkpath(...): (line 109)
        # Processing the call arguments (line 109)
        
        # Call to dirname(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'output_file' (line 109)
        output_file_9159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 44), 'output_file', False)
        # Processing the call keyword arguments (line 109)
        kwargs_9160 = {}
        # Getting the type of 'os' (line 109)
        os_9156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 109)
        path_9157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 28), os_9156, 'path')
        # Obtaining the member 'dirname' of a type (line 109)
        dirname_9158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 28), path_9157, 'dirname')
        # Calling dirname(args, kwargs) (line 109)
        dirname_call_result_9161 = invoke(stypy.reporting.localization.Localization(__file__, 109, 28), dirname_9158, *[output_file_9159], **kwargs_9160)
        
        # Processing the call keyword arguments (line 109)
        kwargs_9162 = {}
        # Getting the type of 'self' (line 109)
        self_9154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 109)
        mkpath_9155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 16), self_9154, 'mkpath')
        # Calling mkpath(args, kwargs) (line 109)
        mkpath_call_result_9163 = invoke(stypy.reporting.localization.Localization(__file__, 109, 16), mkpath_9155, *[dirname_call_result_9161], **kwargs_9162)
        
        # SSA join for if statement (line 108)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'pp_args' (line 111)
        pp_args_9166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 27), 'pp_args', False)
        # Processing the call keyword arguments (line 111)
        kwargs_9167 = {}
        # Getting the type of 'self' (line 111)
        self_9164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 111)
        spawn_9165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), self_9164, 'spawn')
        # Calling spawn(args, kwargs) (line 111)
        spawn_call_result_9168 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), spawn_9165, *[pp_args_9166], **kwargs_9167)
        
        # SSA branch for the except part of a try statement (line 110)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 110)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 112)
        DistutilsExecError_9169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'msg', DistutilsExecError_9169)
        # Getting the type of 'CompileError' (line 113)
        CompileError_9170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 22), 'CompileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 113, 16), CompileError_9170, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 107)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'preprocess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'preprocess' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_9171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9171)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'preprocess'
        return stypy_return_type_9171


    @norecursion
    def _compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_compile'
        module_type_store = module_type_store.open_function_context('_compile', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnixCCompiler._compile.__dict__.__setitem__('stypy_localization', localization)
        UnixCCompiler._compile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnixCCompiler._compile.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnixCCompiler._compile.__dict__.__setitem__('stypy_function_name', 'UnixCCompiler._compile')
        UnixCCompiler._compile.__dict__.__setitem__('stypy_param_names_list', ['obj', 'src', 'ext', 'cc_args', 'extra_postargs', 'pp_opts'])
        UnixCCompiler._compile.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnixCCompiler._compile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnixCCompiler._compile.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnixCCompiler._compile.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnixCCompiler._compile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnixCCompiler._compile.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnixCCompiler._compile', ['obj', 'src', 'ext', 'cc_args', 'extra_postargs', 'pp_opts'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Name (line 116):
        
        # Assigning a Attribute to a Name (line 116):
        # Getting the type of 'self' (line 116)
        self_9172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 22), 'self')
        # Obtaining the member 'compiler_so' of a type (line 116)
        compiler_so_9173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 22), self_9172, 'compiler_so')
        # Assigning a type to the variable 'compiler_so' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'compiler_so', compiler_so_9173)
        
        
        # Getting the type of 'sys' (line 117)
        sys_9174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 117)
        platform_9175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 11), sys_9174, 'platform')
        str_9176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 27), 'str', 'darwin')
        # Applying the binary operator '==' (line 117)
        result_eq_9177 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 11), '==', platform_9175, str_9176)
        
        # Testing the type of an if condition (line 117)
        if_condition_9178 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 8), result_eq_9177)
        # Assigning a type to the variable 'if_condition_9178' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'if_condition_9178', if_condition_9178)
        # SSA begins for if statement (line 117)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 118):
        
        # Assigning a Call to a Name (line 118):
        
        # Call to compiler_fixup(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'compiler_so' (line 118)
        compiler_so_9181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 54), 'compiler_so', False)
        # Getting the type of 'cc_args' (line 119)
        cc_args_9182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 52), 'cc_args', False)
        # Getting the type of 'extra_postargs' (line 119)
        extra_postargs_9183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 62), 'extra_postargs', False)
        # Applying the binary operator '+' (line 119)
        result_add_9184 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 52), '+', cc_args_9182, extra_postargs_9183)
        
        # Processing the call keyword arguments (line 118)
        kwargs_9185 = {}
        # Getting the type of '_osx_support' (line 118)
        _osx_support_9179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 26), '_osx_support', False)
        # Obtaining the member 'compiler_fixup' of a type (line 118)
        compiler_fixup_9180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 26), _osx_support_9179, 'compiler_fixup')
        # Calling compiler_fixup(args, kwargs) (line 118)
        compiler_fixup_call_result_9186 = invoke(stypy.reporting.localization.Localization(__file__, 118, 26), compiler_fixup_9180, *[compiler_so_9181, result_add_9184], **kwargs_9185)
        
        # Assigning a type to the variable 'compiler_so' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'compiler_so', compiler_fixup_call_result_9186)
        # SSA join for if statement (line 117)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 120)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'compiler_so' (line 121)
        compiler_so_9189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'compiler_so', False)
        # Getting the type of 'cc_args' (line 121)
        cc_args_9190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 37), 'cc_args', False)
        # Applying the binary operator '+' (line 121)
        result_add_9191 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 23), '+', compiler_so_9189, cc_args_9190)
        
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_9192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 47), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        # Adding element type (line 121)
        # Getting the type of 'src' (line 121)
        src_9193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 48), 'src', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 47), list_9192, src_9193)
        # Adding element type (line 121)
        str_9194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 53), 'str', '-o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 47), list_9192, str_9194)
        # Adding element type (line 121)
        # Getting the type of 'obj' (line 121)
        obj_9195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 59), 'obj', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 47), list_9192, obj_9195)
        
        # Applying the binary operator '+' (line 121)
        result_add_9196 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 45), '+', result_add_9191, list_9192)
        
        # Getting the type of 'extra_postargs' (line 122)
        extra_postargs_9197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 23), 'extra_postargs', False)
        # Applying the binary operator '+' (line 121)
        result_add_9198 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 64), '+', result_add_9196, extra_postargs_9197)
        
        # Processing the call keyword arguments (line 121)
        kwargs_9199 = {}
        # Getting the type of 'self' (line 121)
        self_9187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'self', False)
        # Obtaining the member 'spawn' of a type (line 121)
        spawn_9188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), self_9187, 'spawn')
        # Calling spawn(args, kwargs) (line 121)
        spawn_call_result_9200 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), spawn_9188, *[result_add_9198], **kwargs_9199)
        
        # SSA branch for the except part of a try statement (line 120)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 120)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 123)
        DistutilsExecError_9201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'msg', DistutilsExecError_9201)
        # Getting the type of 'CompileError' (line 124)
        CompileError_9202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 18), 'CompileError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 124, 12), CompileError_9202, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 120)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_compile' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_9203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9203)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_compile'
        return stypy_return_type_9203


    @norecursion
    def create_static_lib(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 127)
        None_9204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 37), 'None')
        int_9205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 49), 'int')
        # Getting the type of 'None' (line 127)
        None_9206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 64), 'None')
        defaults = [None_9204, int_9205, None_9206]
        # Create a new context for function 'create_static_lib'
        module_type_store = module_type_store.open_function_context('create_static_lib', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnixCCompiler.create_static_lib.__dict__.__setitem__('stypy_localization', localization)
        UnixCCompiler.create_static_lib.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnixCCompiler.create_static_lib.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnixCCompiler.create_static_lib.__dict__.__setitem__('stypy_function_name', 'UnixCCompiler.create_static_lib')
        UnixCCompiler.create_static_lib.__dict__.__setitem__('stypy_param_names_list', ['objects', 'output_libname', 'output_dir', 'debug', 'target_lang'])
        UnixCCompiler.create_static_lib.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnixCCompiler.create_static_lib.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnixCCompiler.create_static_lib.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnixCCompiler.create_static_lib.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnixCCompiler.create_static_lib.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnixCCompiler.create_static_lib.__dict__.__setitem__('stypy_declared_arg_number', 6)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnixCCompiler.create_static_lib', ['objects', 'output_libname', 'output_dir', 'debug', 'target_lang'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Tuple (line 128):
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        int_9207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 8), 'int')
        
        # Call to _fix_object_args(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'objects' (line 128)
        objects_9210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 52), 'objects', False)
        # Getting the type of 'output_dir' (line 128)
        output_dir_9211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 61), 'output_dir', False)
        # Processing the call keyword arguments (line 128)
        kwargs_9212 = {}
        # Getting the type of 'self' (line 128)
        self_9208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 30), 'self', False)
        # Obtaining the member '_fix_object_args' of a type (line 128)
        _fix_object_args_9209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 30), self_9208, '_fix_object_args')
        # Calling _fix_object_args(args, kwargs) (line 128)
        _fix_object_args_call_result_9213 = invoke(stypy.reporting.localization.Localization(__file__, 128, 30), _fix_object_args_9209, *[objects_9210, output_dir_9211], **kwargs_9212)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___9214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), _fix_object_args_call_result_9213, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_9215 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), getitem___9214, int_9207)
        
        # Assigning a type to the variable 'tuple_var_assignment_9044' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_9044', subscript_call_result_9215)
        
        # Assigning a Subscript to a Name (line 128):
        
        # Obtaining the type of the subscript
        int_9216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 8), 'int')
        
        # Call to _fix_object_args(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'objects' (line 128)
        objects_9219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 52), 'objects', False)
        # Getting the type of 'output_dir' (line 128)
        output_dir_9220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 61), 'output_dir', False)
        # Processing the call keyword arguments (line 128)
        kwargs_9221 = {}
        # Getting the type of 'self' (line 128)
        self_9217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 30), 'self', False)
        # Obtaining the member '_fix_object_args' of a type (line 128)
        _fix_object_args_9218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 30), self_9217, '_fix_object_args')
        # Calling _fix_object_args(args, kwargs) (line 128)
        _fix_object_args_call_result_9222 = invoke(stypy.reporting.localization.Localization(__file__, 128, 30), _fix_object_args_9218, *[objects_9219, output_dir_9220], **kwargs_9221)
        
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___9223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), _fix_object_args_call_result_9222, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_9224 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), getitem___9223, int_9216)
        
        # Assigning a type to the variable 'tuple_var_assignment_9045' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_9045', subscript_call_result_9224)
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'tuple_var_assignment_9044' (line 128)
        tuple_var_assignment_9044_9225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_9044')
        # Assigning a type to the variable 'objects' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'objects', tuple_var_assignment_9044_9225)
        
        # Assigning a Name to a Name (line 128):
        # Getting the type of 'tuple_var_assignment_9045' (line 128)
        tuple_var_assignment_9045_9226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'tuple_var_assignment_9045')
        # Assigning a type to the variable 'output_dir' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 17), 'output_dir', tuple_var_assignment_9045_9226)
        
        # Assigning a Call to a Name (line 130):
        
        # Assigning a Call to a Name (line 130):
        
        # Call to library_filename(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'output_libname' (line 131)
        output_libname_9229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 34), 'output_libname', False)
        # Processing the call keyword arguments (line 131)
        # Getting the type of 'output_dir' (line 131)
        output_dir_9230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 61), 'output_dir', False)
        keyword_9231 = output_dir_9230
        kwargs_9232 = {'output_dir': keyword_9231}
        # Getting the type of 'self' (line 131)
        self_9227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'self', False)
        # Obtaining the member 'library_filename' of a type (line 131)
        library_filename_9228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), self_9227, 'library_filename')
        # Calling library_filename(args, kwargs) (line 131)
        library_filename_call_result_9233 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), library_filename_9228, *[output_libname_9229], **kwargs_9232)
        
        # Assigning a type to the variable 'output_filename' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'output_filename', library_filename_call_result_9233)
        
        
        # Call to _need_link(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'objects' (line 133)
        objects_9236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 27), 'objects', False)
        # Getting the type of 'output_filename' (line 133)
        output_filename_9237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 36), 'output_filename', False)
        # Processing the call keyword arguments (line 133)
        kwargs_9238 = {}
        # Getting the type of 'self' (line 133)
        self_9234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'self', False)
        # Obtaining the member '_need_link' of a type (line 133)
        _need_link_9235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 11), self_9234, '_need_link')
        # Calling _need_link(args, kwargs) (line 133)
        _need_link_call_result_9239 = invoke(stypy.reporting.localization.Localization(__file__, 133, 11), _need_link_9235, *[objects_9236, output_filename_9237], **kwargs_9238)
        
        # Testing the type of an if condition (line 133)
        if_condition_9240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 8), _need_link_call_result_9239)
        # Assigning a type to the variable 'if_condition_9240' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'if_condition_9240', if_condition_9240)
        # SSA begins for if statement (line 133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to mkpath(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Call to dirname(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'output_filename' (line 134)
        output_filename_9246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 40), 'output_filename', False)
        # Processing the call keyword arguments (line 134)
        kwargs_9247 = {}
        # Getting the type of 'os' (line 134)
        os_9243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 134)
        path_9244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 24), os_9243, 'path')
        # Obtaining the member 'dirname' of a type (line 134)
        dirname_9245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 24), path_9244, 'dirname')
        # Calling dirname(args, kwargs) (line 134)
        dirname_call_result_9248 = invoke(stypy.reporting.localization.Localization(__file__, 134, 24), dirname_9245, *[output_filename_9246], **kwargs_9247)
        
        # Processing the call keyword arguments (line 134)
        kwargs_9249 = {}
        # Getting the type of 'self' (line 134)
        self_9241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 134)
        mkpath_9242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 12), self_9241, 'mkpath')
        # Calling mkpath(args, kwargs) (line 134)
        mkpath_call_result_9250 = invoke(stypy.reporting.localization.Localization(__file__, 134, 12), mkpath_9242, *[dirname_call_result_9248], **kwargs_9249)
        
        
        # Call to spawn(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'self' (line 135)
        self_9253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'self', False)
        # Obtaining the member 'archiver' of a type (line 135)
        archiver_9254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 23), self_9253, 'archiver')
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_9255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        # Adding element type (line 136)
        # Getting the type of 'output_filename' (line 136)
        output_filename_9256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'output_filename', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 23), list_9255, output_filename_9256)
        
        # Applying the binary operator '+' (line 135)
        result_add_9257 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 23), '+', archiver_9254, list_9255)
        
        # Getting the type of 'objects' (line 137)
        objects_9258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 23), 'objects', False)
        # Applying the binary operator '+' (line 136)
        result_add_9259 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 41), '+', result_add_9257, objects_9258)
        
        # Getting the type of 'self' (line 137)
        self_9260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 33), 'self', False)
        # Obtaining the member 'objects' of a type (line 137)
        objects_9261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 33), self_9260, 'objects')
        # Applying the binary operator '+' (line 137)
        result_add_9262 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 31), '+', result_add_9259, objects_9261)
        
        # Processing the call keyword arguments (line 135)
        kwargs_9263 = {}
        # Getting the type of 'self' (line 135)
        self_9251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'self', False)
        # Obtaining the member 'spawn' of a type (line 135)
        spawn_9252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 12), self_9251, 'spawn')
        # Calling spawn(args, kwargs) (line 135)
        spawn_call_result_9264 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), spawn_9252, *[result_add_9262], **kwargs_9263)
        
        
        # Getting the type of 'self' (line 144)
        self_9265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'self')
        # Obtaining the member 'ranlib' of a type (line 144)
        ranlib_9266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 15), self_9265, 'ranlib')
        # Testing the type of an if condition (line 144)
        if_condition_9267 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 12), ranlib_9266)
        # Assigning a type to the variable 'if_condition_9267' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'if_condition_9267', if_condition_9267)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 145)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'self' (line 146)
        self_9270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 31), 'self', False)
        # Obtaining the member 'ranlib' of a type (line 146)
        ranlib_9271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 31), self_9270, 'ranlib')
        
        # Obtaining an instance of the builtin type 'list' (line 146)
        list_9272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 146)
        # Adding element type (line 146)
        # Getting the type of 'output_filename' (line 146)
        output_filename_9273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 46), 'output_filename', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 45), list_9272, output_filename_9273)
        
        # Applying the binary operator '+' (line 146)
        result_add_9274 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 31), '+', ranlib_9271, list_9272)
        
        # Processing the call keyword arguments (line 146)
        kwargs_9275 = {}
        # Getting the type of 'self' (line 146)
        self_9268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'self', False)
        # Obtaining the member 'spawn' of a type (line 146)
        spawn_9269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 20), self_9268, 'spawn')
        # Calling spawn(args, kwargs) (line 146)
        spawn_call_result_9276 = invoke(stypy.reporting.localization.Localization(__file__, 146, 20), spawn_9269, *[result_add_9274], **kwargs_9275)
        
        # SSA branch for the except part of a try statement (line 145)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 145)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 147)
        DistutilsExecError_9277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), 'msg', DistutilsExecError_9277)
        # Getting the type of 'LibError' (line 148)
        LibError_9278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 26), 'LibError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 148, 20), LibError_9278, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 145)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 133)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug(...): (line 150)
        # Processing the call arguments (line 150)
        str_9281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 22), 'str', 'skipping %s (up-to-date)')
        # Getting the type of 'output_filename' (line 150)
        output_filename_9282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 50), 'output_filename', False)
        # Processing the call keyword arguments (line 150)
        kwargs_9283 = {}
        # Getting the type of 'log' (line 150)
        log_9279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 150)
        debug_9280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 12), log_9279, 'debug')
        # Calling debug(args, kwargs) (line 150)
        debug_call_result_9284 = invoke(stypy.reporting.localization.Localization(__file__, 150, 12), debug_9280, *[str_9281, output_filename_9282], **kwargs_9283)
        
        # SSA join for if statement (line 133)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'create_static_lib(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_static_lib' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_9285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9285)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_static_lib'
        return stypy_return_type_9285


    @norecursion
    def link(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 153)
        None_9286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 41), 'None')
        # Getting the type of 'None' (line 153)
        None_9287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 57), 'None')
        # Getting the type of 'None' (line 154)
        None_9288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 26), 'None')
        # Getting the type of 'None' (line 154)
        None_9289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 53), 'None')
        # Getting the type of 'None' (line 155)
        None_9290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 28), 'None')
        int_9291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 40), 'int')
        # Getting the type of 'None' (line 155)
        None_9292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 57), 'None')
        # Getting the type of 'None' (line 156)
        None_9293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 28), 'None')
        # Getting the type of 'None' (line 156)
        None_9294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 45), 'None')
        # Getting the type of 'None' (line 156)
        None_9295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 63), 'None')
        defaults = [None_9286, None_9287, None_9288, None_9289, None_9290, int_9291, None_9292, None_9293, None_9294, None_9295]
        # Create a new context for function 'link'
        module_type_store = module_type_store.open_function_context('link', 152, 4, False)
        # Assigning a type to the variable 'self' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnixCCompiler.link.__dict__.__setitem__('stypy_localization', localization)
        UnixCCompiler.link.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnixCCompiler.link.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnixCCompiler.link.__dict__.__setitem__('stypy_function_name', 'UnixCCompiler.link')
        UnixCCompiler.link.__dict__.__setitem__('stypy_param_names_list', ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'])
        UnixCCompiler.link.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnixCCompiler.link.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnixCCompiler.link.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnixCCompiler.link.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnixCCompiler.link.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnixCCompiler.link.__dict__.__setitem__('stypy_declared_arg_number', 14)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnixCCompiler.link', ['target_desc', 'objects', 'output_filename', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Tuple (line 157):
        
        # Assigning a Subscript to a Name (line 157):
        
        # Obtaining the type of the subscript
        int_9296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 8), 'int')
        
        # Call to _fix_object_args(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'objects' (line 157)
        objects_9299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 52), 'objects', False)
        # Getting the type of 'output_dir' (line 157)
        output_dir_9300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 61), 'output_dir', False)
        # Processing the call keyword arguments (line 157)
        kwargs_9301 = {}
        # Getting the type of 'self' (line 157)
        self_9297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 30), 'self', False)
        # Obtaining the member '_fix_object_args' of a type (line 157)
        _fix_object_args_9298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 30), self_9297, '_fix_object_args')
        # Calling _fix_object_args(args, kwargs) (line 157)
        _fix_object_args_call_result_9302 = invoke(stypy.reporting.localization.Localization(__file__, 157, 30), _fix_object_args_9298, *[objects_9299, output_dir_9300], **kwargs_9301)
        
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___9303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), _fix_object_args_call_result_9302, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_9304 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), getitem___9303, int_9296)
        
        # Assigning a type to the variable 'tuple_var_assignment_9046' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_9046', subscript_call_result_9304)
        
        # Assigning a Subscript to a Name (line 157):
        
        # Obtaining the type of the subscript
        int_9305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 8), 'int')
        
        # Call to _fix_object_args(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'objects' (line 157)
        objects_9308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 52), 'objects', False)
        # Getting the type of 'output_dir' (line 157)
        output_dir_9309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 61), 'output_dir', False)
        # Processing the call keyword arguments (line 157)
        kwargs_9310 = {}
        # Getting the type of 'self' (line 157)
        self_9306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 30), 'self', False)
        # Obtaining the member '_fix_object_args' of a type (line 157)
        _fix_object_args_9307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 30), self_9306, '_fix_object_args')
        # Calling _fix_object_args(args, kwargs) (line 157)
        _fix_object_args_call_result_9311 = invoke(stypy.reporting.localization.Localization(__file__, 157, 30), _fix_object_args_9307, *[objects_9308, output_dir_9309], **kwargs_9310)
        
        # Obtaining the member '__getitem__' of a type (line 157)
        getitem___9312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), _fix_object_args_call_result_9311, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 157)
        subscript_call_result_9313 = invoke(stypy.reporting.localization.Localization(__file__, 157, 8), getitem___9312, int_9305)
        
        # Assigning a type to the variable 'tuple_var_assignment_9047' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_9047', subscript_call_result_9313)
        
        # Assigning a Name to a Name (line 157):
        # Getting the type of 'tuple_var_assignment_9046' (line 157)
        tuple_var_assignment_9046_9314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_9046')
        # Assigning a type to the variable 'objects' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'objects', tuple_var_assignment_9046_9314)
        
        # Assigning a Name to a Name (line 157):
        # Getting the type of 'tuple_var_assignment_9047' (line 157)
        tuple_var_assignment_9047_9315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'tuple_var_assignment_9047')
        # Assigning a type to the variable 'output_dir' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 17), 'output_dir', tuple_var_assignment_9047_9315)
        
        # Assigning a Call to a Tuple (line 158):
        
        # Assigning a Subscript to a Name (line 158):
        
        # Obtaining the type of the subscript
        int_9316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 8), 'int')
        
        # Call to _fix_lib_args(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'libraries' (line 159)
        libraries_9319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 31), 'libraries', False)
        # Getting the type of 'library_dirs' (line 159)
        library_dirs_9320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 42), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 159)
        runtime_library_dirs_9321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 56), 'runtime_library_dirs', False)
        # Processing the call keyword arguments (line 159)
        kwargs_9322 = {}
        # Getting the type of 'self' (line 159)
        self_9317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'self', False)
        # Obtaining the member '_fix_lib_args' of a type (line 159)
        _fix_lib_args_9318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), self_9317, '_fix_lib_args')
        # Calling _fix_lib_args(args, kwargs) (line 159)
        _fix_lib_args_call_result_9323 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), _fix_lib_args_9318, *[libraries_9319, library_dirs_9320, runtime_library_dirs_9321], **kwargs_9322)
        
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___9324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), _fix_lib_args_call_result_9323, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_9325 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), getitem___9324, int_9316)
        
        # Assigning a type to the variable 'tuple_var_assignment_9048' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'tuple_var_assignment_9048', subscript_call_result_9325)
        
        # Assigning a Subscript to a Name (line 158):
        
        # Obtaining the type of the subscript
        int_9326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 8), 'int')
        
        # Call to _fix_lib_args(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'libraries' (line 159)
        libraries_9329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 31), 'libraries', False)
        # Getting the type of 'library_dirs' (line 159)
        library_dirs_9330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 42), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 159)
        runtime_library_dirs_9331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 56), 'runtime_library_dirs', False)
        # Processing the call keyword arguments (line 159)
        kwargs_9332 = {}
        # Getting the type of 'self' (line 159)
        self_9327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'self', False)
        # Obtaining the member '_fix_lib_args' of a type (line 159)
        _fix_lib_args_9328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), self_9327, '_fix_lib_args')
        # Calling _fix_lib_args(args, kwargs) (line 159)
        _fix_lib_args_call_result_9333 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), _fix_lib_args_9328, *[libraries_9329, library_dirs_9330, runtime_library_dirs_9331], **kwargs_9332)
        
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___9334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), _fix_lib_args_call_result_9333, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_9335 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), getitem___9334, int_9326)
        
        # Assigning a type to the variable 'tuple_var_assignment_9049' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'tuple_var_assignment_9049', subscript_call_result_9335)
        
        # Assigning a Subscript to a Name (line 158):
        
        # Obtaining the type of the subscript
        int_9336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 8), 'int')
        
        # Call to _fix_lib_args(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'libraries' (line 159)
        libraries_9339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 31), 'libraries', False)
        # Getting the type of 'library_dirs' (line 159)
        library_dirs_9340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 42), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 159)
        runtime_library_dirs_9341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 56), 'runtime_library_dirs', False)
        # Processing the call keyword arguments (line 159)
        kwargs_9342 = {}
        # Getting the type of 'self' (line 159)
        self_9337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'self', False)
        # Obtaining the member '_fix_lib_args' of a type (line 159)
        _fix_lib_args_9338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 12), self_9337, '_fix_lib_args')
        # Calling _fix_lib_args(args, kwargs) (line 159)
        _fix_lib_args_call_result_9343 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), _fix_lib_args_9338, *[libraries_9339, library_dirs_9340, runtime_library_dirs_9341], **kwargs_9342)
        
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___9344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), _fix_lib_args_call_result_9343, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_9345 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), getitem___9344, int_9336)
        
        # Assigning a type to the variable 'tuple_var_assignment_9050' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'tuple_var_assignment_9050', subscript_call_result_9345)
        
        # Assigning a Name to a Name (line 158):
        # Getting the type of 'tuple_var_assignment_9048' (line 158)
        tuple_var_assignment_9048_9346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'tuple_var_assignment_9048')
        # Assigning a type to the variable 'libraries' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'libraries', tuple_var_assignment_9048_9346)
        
        # Assigning a Name to a Name (line 158):
        # Getting the type of 'tuple_var_assignment_9049' (line 158)
        tuple_var_assignment_9049_9347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'tuple_var_assignment_9049')
        # Assigning a type to the variable 'library_dirs' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 19), 'library_dirs', tuple_var_assignment_9049_9347)
        
        # Assigning a Name to a Name (line 158):
        # Getting the type of 'tuple_var_assignment_9050' (line 158)
        tuple_var_assignment_9050_9348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'tuple_var_assignment_9050')
        # Assigning a type to the variable 'runtime_library_dirs' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 33), 'runtime_library_dirs', tuple_var_assignment_9050_9348)
        
        # Assigning a Call to a Name (line 161):
        
        # Assigning a Call to a Name (line 161):
        
        # Call to gen_lib_options(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'self' (line 161)
        self_9350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 35), 'self', False)
        # Getting the type of 'library_dirs' (line 161)
        library_dirs_9351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 41), 'library_dirs', False)
        # Getting the type of 'runtime_library_dirs' (line 161)
        runtime_library_dirs_9352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 55), 'runtime_library_dirs', False)
        # Getting the type of 'libraries' (line 162)
        libraries_9353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 35), 'libraries', False)
        # Processing the call keyword arguments (line 161)
        kwargs_9354 = {}
        # Getting the type of 'gen_lib_options' (line 161)
        gen_lib_options_9349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 19), 'gen_lib_options', False)
        # Calling gen_lib_options(args, kwargs) (line 161)
        gen_lib_options_call_result_9355 = invoke(stypy.reporting.localization.Localization(__file__, 161, 19), gen_lib_options_9349, *[self_9350, library_dirs_9351, runtime_library_dirs_9352, libraries_9353], **kwargs_9354)
        
        # Assigning a type to the variable 'lib_opts' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'lib_opts', gen_lib_options_call_result_9355)
        
        
        
        # Call to type(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'output_dir' (line 163)
        output_dir_9357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'output_dir', False)
        # Processing the call keyword arguments (line 163)
        kwargs_9358 = {}
        # Getting the type of 'type' (line 163)
        type_9356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'type', False)
        # Calling type(args, kwargs) (line 163)
        type_call_result_9359 = invoke(stypy.reporting.localization.Localization(__file__, 163, 11), type_9356, *[output_dir_9357], **kwargs_9358)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 163)
        tuple_9360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 163)
        # Adding element type (line 163)
        # Getting the type of 'StringType' (line 163)
        StringType_9361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 36), 'StringType')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 36), tuple_9360, StringType_9361)
        # Adding element type (line 163)
        # Getting the type of 'NoneType' (line 163)
        NoneType_9362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 48), 'NoneType')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 36), tuple_9360, NoneType_9362)
        
        # Applying the binary operator 'notin' (line 163)
        result_contains_9363 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 11), 'notin', type_call_result_9359, tuple_9360)
        
        # Testing the type of an if condition (line 163)
        if_condition_9364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 8), result_contains_9363)
        # Assigning a type to the variable 'if_condition_9364' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'if_condition_9364', if_condition_9364)
        # SSA begins for if statement (line 163)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'TypeError' (line 164)
        TypeError_9365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 18), 'TypeError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 164, 12), TypeError_9365, 'raise parameter', BaseException)
        # SSA join for if statement (line 163)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 165)
        # Getting the type of 'output_dir' (line 165)
        output_dir_9366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'output_dir')
        # Getting the type of 'None' (line 165)
        None_9367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 29), 'None')
        
        (may_be_9368, more_types_in_union_9369) = may_not_be_none(output_dir_9366, None_9367)

        if may_be_9368:

            if more_types_in_union_9369:
                # Runtime conditional SSA (line 165)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 166):
            
            # Assigning a Call to a Name (line 166):
            
            # Call to join(...): (line 166)
            # Processing the call arguments (line 166)
            # Getting the type of 'output_dir' (line 166)
            output_dir_9373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 43), 'output_dir', False)
            # Getting the type of 'output_filename' (line 166)
            output_filename_9374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 55), 'output_filename', False)
            # Processing the call keyword arguments (line 166)
            kwargs_9375 = {}
            # Getting the type of 'os' (line 166)
            os_9370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 30), 'os', False)
            # Obtaining the member 'path' of a type (line 166)
            path_9371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 30), os_9370, 'path')
            # Obtaining the member 'join' of a type (line 166)
            join_9372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 30), path_9371, 'join')
            # Calling join(args, kwargs) (line 166)
            join_call_result_9376 = invoke(stypy.reporting.localization.Localization(__file__, 166, 30), join_9372, *[output_dir_9373, output_filename_9374], **kwargs_9375)
            
            # Assigning a type to the variable 'output_filename' (line 166)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'output_filename', join_call_result_9376)

            if more_types_in_union_9369:
                # SSA join for if statement (line 165)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to _need_link(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'objects' (line 168)
        objects_9379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'objects', False)
        # Getting the type of 'output_filename' (line 168)
        output_filename_9380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 36), 'output_filename', False)
        # Processing the call keyword arguments (line 168)
        kwargs_9381 = {}
        # Getting the type of 'self' (line 168)
        self_9377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 11), 'self', False)
        # Obtaining the member '_need_link' of a type (line 168)
        _need_link_9378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 11), self_9377, '_need_link')
        # Calling _need_link(args, kwargs) (line 168)
        _need_link_call_result_9382 = invoke(stypy.reporting.localization.Localization(__file__, 168, 11), _need_link_9378, *[objects_9379, output_filename_9380], **kwargs_9381)
        
        # Testing the type of an if condition (line 168)
        if_condition_9383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 8), _need_link_call_result_9382)
        # Assigning a type to the variable 'if_condition_9383' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'if_condition_9383', if_condition_9383)
        # SSA begins for if statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 169):
        
        # Assigning a BinOp to a Name (line 169):
        # Getting the type of 'objects' (line 169)
        objects_9384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'objects')
        # Getting the type of 'self' (line 169)
        self_9385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 33), 'self')
        # Obtaining the member 'objects' of a type (line 169)
        objects_9386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 33), self_9385, 'objects')
        # Applying the binary operator '+' (line 169)
        result_add_9387 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 23), '+', objects_9384, objects_9386)
        
        # Getting the type of 'lib_opts' (line 170)
        lib_opts_9388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 23), 'lib_opts')
        # Applying the binary operator '+' (line 169)
        result_add_9389 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 46), '+', result_add_9387, lib_opts_9388)
        
        
        # Obtaining an instance of the builtin type 'list' (line 170)
        list_9390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 170)
        # Adding element type (line 170)
        str_9391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 35), 'str', '-o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 34), list_9390, str_9391)
        # Adding element type (line 170)
        # Getting the type of 'output_filename' (line 170)
        output_filename_9392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 41), 'output_filename')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 34), list_9390, output_filename_9392)
        
        # Applying the binary operator '+' (line 170)
        result_add_9393 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 32), '+', result_add_9389, list_9390)
        
        # Assigning a type to the variable 'ld_args' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'ld_args', result_add_9393)
        
        # Getting the type of 'debug' (line 171)
        debug_9394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'debug')
        # Testing the type of an if condition (line 171)
        if_condition_9395 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 12), debug_9394)
        # Assigning a type to the variable 'if_condition_9395' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'if_condition_9395', if_condition_9395)
        # SSA begins for if statement (line 171)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Subscript (line 172):
        
        # Assigning a List to a Subscript (line 172):
        
        # Obtaining an instance of the builtin type 'list' (line 172)
        list_9396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 172)
        # Adding element type (line 172)
        str_9397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 31), 'str', '-g')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 30), list_9396, str_9397)
        
        # Getting the type of 'ld_args' (line 172)
        ld_args_9398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'ld_args')
        int_9399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 25), 'int')
        slice_9400 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 172, 16), None, int_9399, None)
        # Storing an element on a container (line 172)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 16), ld_args_9398, (slice_9400, list_9396))
        # SSA join for if statement (line 171)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'extra_preargs' (line 173)
        extra_preargs_9401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'extra_preargs')
        # Testing the type of an if condition (line 173)
        if_condition_9402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 12), extra_preargs_9401)
        # Assigning a type to the variable 'if_condition_9402' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'if_condition_9402', if_condition_9402)
        # SSA begins for if statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 174):
        
        # Assigning a Name to a Subscript (line 174):
        # Getting the type of 'extra_preargs' (line 174)
        extra_preargs_9403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 30), 'extra_preargs')
        # Getting the type of 'ld_args' (line 174)
        ld_args_9404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'ld_args')
        int_9405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 25), 'int')
        slice_9406 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 174, 16), None, int_9405, None)
        # Storing an element on a container (line 174)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 16), ld_args_9404, (slice_9406, extra_preargs_9403))
        # SSA join for if statement (line 173)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'extra_postargs' (line 175)
        extra_postargs_9407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'extra_postargs')
        # Testing the type of an if condition (line 175)
        if_condition_9408 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 12), extra_postargs_9407)
        # Assigning a type to the variable 'if_condition_9408' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'if_condition_9408', if_condition_9408)
        # SSA begins for if statement (line 175)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'extra_postargs' (line 176)
        extra_postargs_9411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 31), 'extra_postargs', False)
        # Processing the call keyword arguments (line 176)
        kwargs_9412 = {}
        # Getting the type of 'ld_args' (line 176)
        ld_args_9409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'ld_args', False)
        # Obtaining the member 'extend' of a type (line 176)
        extend_9410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 16), ld_args_9409, 'extend')
        # Calling extend(args, kwargs) (line 176)
        extend_call_result_9413 = invoke(stypy.reporting.localization.Localization(__file__, 176, 16), extend_9410, *[extra_postargs_9411], **kwargs_9412)
        
        # SSA join for if statement (line 175)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to mkpath(...): (line 177)
        # Processing the call arguments (line 177)
        
        # Call to dirname(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'output_filename' (line 177)
        output_filename_9419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 40), 'output_filename', False)
        # Processing the call keyword arguments (line 177)
        kwargs_9420 = {}
        # Getting the type of 'os' (line 177)
        os_9416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 177)
        path_9417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 24), os_9416, 'path')
        # Obtaining the member 'dirname' of a type (line 177)
        dirname_9418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 24), path_9417, 'dirname')
        # Calling dirname(args, kwargs) (line 177)
        dirname_call_result_9421 = invoke(stypy.reporting.localization.Localization(__file__, 177, 24), dirname_9418, *[output_filename_9419], **kwargs_9420)
        
        # Processing the call keyword arguments (line 177)
        kwargs_9422 = {}
        # Getting the type of 'self' (line 177)
        self_9414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 177)
        mkpath_9415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 12), self_9414, 'mkpath')
        # Calling mkpath(args, kwargs) (line 177)
        mkpath_call_result_9423 = invoke(stypy.reporting.localization.Localization(__file__, 177, 12), mkpath_9415, *[dirname_call_result_9421], **kwargs_9422)
        
        
        
        # SSA begins for try-except statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        
        # Getting the type of 'target_desc' (line 179)
        target_desc_9424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 19), 'target_desc')
        # Getting the type of 'CCompiler' (line 179)
        CCompiler_9425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 34), 'CCompiler')
        # Obtaining the member 'EXECUTABLE' of a type (line 179)
        EXECUTABLE_9426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 34), CCompiler_9425, 'EXECUTABLE')
        # Applying the binary operator '==' (line 179)
        result_eq_9427 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 19), '==', target_desc_9424, EXECUTABLE_9426)
        
        # Testing the type of an if condition (line 179)
        if_condition_9428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 16), result_eq_9427)
        # Assigning a type to the variable 'if_condition_9428' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'if_condition_9428', if_condition_9428)
        # SSA begins for if statement (line 179)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 180):
        
        # Assigning a Subscript to a Name (line 180):
        
        # Obtaining the type of the subscript
        slice_9429 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 180, 29), None, None, None)
        # Getting the type of 'self' (line 180)
        self_9430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 29), 'self')
        # Obtaining the member 'linker_exe' of a type (line 180)
        linker_exe_9431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 29), self_9430, 'linker_exe')
        # Obtaining the member '__getitem__' of a type (line 180)
        getitem___9432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 29), linker_exe_9431, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 180)
        subscript_call_result_9433 = invoke(stypy.reporting.localization.Localization(__file__, 180, 29), getitem___9432, slice_9429)
        
        # Assigning a type to the variable 'linker' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 20), 'linker', subscript_call_result_9433)
        # SSA branch for the else part of an if statement (line 179)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 182):
        
        # Assigning a Subscript to a Name (line 182):
        
        # Obtaining the type of the subscript
        slice_9434 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 182, 29), None, None, None)
        # Getting the type of 'self' (line 182)
        self_9435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 29), 'self')
        # Obtaining the member 'linker_so' of a type (line 182)
        linker_so_9436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 29), self_9435, 'linker_so')
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___9437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 29), linker_so_9436, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_9438 = invoke(stypy.reporting.localization.Localization(__file__, 182, 29), getitem___9437, slice_9434)
        
        # Assigning a type to the variable 'linker' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 20), 'linker', subscript_call_result_9438)
        # SSA join for if statement (line 179)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'target_lang' (line 183)
        target_lang_9439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 19), 'target_lang')
        str_9440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 34), 'str', 'c++')
        # Applying the binary operator '==' (line 183)
        result_eq_9441 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 19), '==', target_lang_9439, str_9440)
        
        # Getting the type of 'self' (line 183)
        self_9442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 44), 'self')
        # Obtaining the member 'compiler_cxx' of a type (line 183)
        compiler_cxx_9443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 44), self_9442, 'compiler_cxx')
        # Applying the binary operator 'and' (line 183)
        result_and_keyword_9444 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 19), 'and', result_eq_9441, compiler_cxx_9443)
        
        # Testing the type of an if condition (line 183)
        if_condition_9445 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 16), result_and_keyword_9444)
        # Assigning a type to the variable 'if_condition_9445' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'if_condition_9445', if_condition_9445)
        # SSA begins for if statement (line 183)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 189):
        
        # Assigning a Num to a Name (line 189):
        int_9446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 24), 'int')
        # Assigning a type to the variable 'i' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'i', int_9446)
        
        
        
        # Call to basename(...): (line 190)
        # Processing the call arguments (line 190)
        
        # Obtaining the type of the subscript
        int_9450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 47), 'int')
        # Getting the type of 'linker' (line 190)
        linker_9451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 40), 'linker', False)
        # Obtaining the member '__getitem__' of a type (line 190)
        getitem___9452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 40), linker_9451, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 190)
        subscript_call_result_9453 = invoke(stypy.reporting.localization.Localization(__file__, 190, 40), getitem___9452, int_9450)
        
        # Processing the call keyword arguments (line 190)
        kwargs_9454 = {}
        # Getting the type of 'os' (line 190)
        os_9447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 190)
        path_9448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 23), os_9447, 'path')
        # Obtaining the member 'basename' of a type (line 190)
        basename_9449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 23), path_9448, 'basename')
        # Calling basename(args, kwargs) (line 190)
        basename_call_result_9455 = invoke(stypy.reporting.localization.Localization(__file__, 190, 23), basename_9449, *[subscript_call_result_9453], **kwargs_9454)
        
        str_9456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 54), 'str', 'env')
        # Applying the binary operator '==' (line 190)
        result_eq_9457 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 23), '==', basename_call_result_9455, str_9456)
        
        # Testing the type of an if condition (line 190)
        if_condition_9458 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 20), result_eq_9457)
        # Assigning a type to the variable 'if_condition_9458' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 20), 'if_condition_9458', if_condition_9458)
        # SSA begins for if statement (line 190)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 191):
        
        # Assigning a Num to a Name (line 191):
        int_9459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 28), 'int')
        # Assigning a type to the variable 'i' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 24), 'i', int_9459)
        
        
        str_9460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 30), 'str', '=')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 192)
        i_9461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 44), 'i')
        # Getting the type of 'linker' (line 192)
        linker_9462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 37), 'linker')
        # Obtaining the member '__getitem__' of a type (line 192)
        getitem___9463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 37), linker_9462, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 192)
        subscript_call_result_9464 = invoke(stypy.reporting.localization.Localization(__file__, 192, 37), getitem___9463, i_9461)
        
        # Applying the binary operator 'in' (line 192)
        result_contains_9465 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 30), 'in', str_9460, subscript_call_result_9464)
        
        # Testing the type of an if condition (line 192)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 24), result_contains_9465)
        # SSA begins for while statement (line 192)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a BinOp to a Name (line 193):
        
        # Assigning a BinOp to a Name (line 193):
        # Getting the type of 'i' (line 193)
        i_9466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 32), 'i')
        int_9467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 36), 'int')
        # Applying the binary operator '+' (line 193)
        result_add_9468 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 32), '+', i_9466, int_9467)
        
        # Assigning a type to the variable 'i' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 28), 'i', result_add_9468)
        # SSA join for while statement (line 192)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 190)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Subscript (line 195):
        
        # Assigning a Subscript to a Subscript (line 195):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 195)
        i_9469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 50), 'i')
        # Getting the type of 'self' (line 195)
        self_9470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 32), 'self')
        # Obtaining the member 'compiler_cxx' of a type (line 195)
        compiler_cxx_9471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 32), self_9470, 'compiler_cxx')
        # Obtaining the member '__getitem__' of a type (line 195)
        getitem___9472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 32), compiler_cxx_9471, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 195)
        subscript_call_result_9473 = invoke(stypy.reporting.localization.Localization(__file__, 195, 32), getitem___9472, i_9469)
        
        # Getting the type of 'linker' (line 195)
        linker_9474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 20), 'linker')
        # Getting the type of 'i' (line 195)
        i_9475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 27), 'i')
        # Storing an element on a container (line 195)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 20), linker_9474, (i_9475, subscript_call_result_9473))
        # SSA join for if statement (line 183)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'sys' (line 197)
        sys_9476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'sys')
        # Obtaining the member 'platform' of a type (line 197)
        platform_9477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 19), sys_9476, 'platform')
        str_9478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 35), 'str', 'darwin')
        # Applying the binary operator '==' (line 197)
        result_eq_9479 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 19), '==', platform_9477, str_9478)
        
        # Testing the type of an if condition (line 197)
        if_condition_9480 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 16), result_eq_9479)
        # Assigning a type to the variable 'if_condition_9480' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'if_condition_9480', if_condition_9480)
        # SSA begins for if statement (line 197)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 198):
        
        # Assigning a Call to a Name (line 198):
        
        # Call to compiler_fixup(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'linker' (line 198)
        linker_9483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 57), 'linker', False)
        # Getting the type of 'ld_args' (line 198)
        ld_args_9484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 65), 'ld_args', False)
        # Processing the call keyword arguments (line 198)
        kwargs_9485 = {}
        # Getting the type of '_osx_support' (line 198)
        _osx_support_9481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 29), '_osx_support', False)
        # Obtaining the member 'compiler_fixup' of a type (line 198)
        compiler_fixup_9482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 29), _osx_support_9481, 'compiler_fixup')
        # Calling compiler_fixup(args, kwargs) (line 198)
        compiler_fixup_call_result_9486 = invoke(stypy.reporting.localization.Localization(__file__, 198, 29), compiler_fixup_9482, *[linker_9483, ld_args_9484], **kwargs_9485)
        
        # Assigning a type to the variable 'linker' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 20), 'linker', compiler_fixup_call_result_9486)
        # SSA join for if statement (line 197)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to spawn(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'linker' (line 200)
        linker_9489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 27), 'linker', False)
        # Getting the type of 'ld_args' (line 200)
        ld_args_9490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 36), 'ld_args', False)
        # Applying the binary operator '+' (line 200)
        result_add_9491 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 27), '+', linker_9489, ld_args_9490)
        
        # Processing the call keyword arguments (line 200)
        kwargs_9492 = {}
        # Getting the type of 'self' (line 200)
        self_9487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 16), 'self', False)
        # Obtaining the member 'spawn' of a type (line 200)
        spawn_9488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 16), self_9487, 'spawn')
        # Calling spawn(args, kwargs) (line 200)
        spawn_call_result_9493 = invoke(stypy.reporting.localization.Localization(__file__, 200, 16), spawn_9488, *[result_add_9491], **kwargs_9492)
        
        # SSA branch for the except part of a try statement (line 178)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 178)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 201)
        DistutilsExecError_9494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 'DistutilsExecError')
        # Assigning a type to the variable 'msg' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'msg', DistutilsExecError_9494)
        # Getting the type of 'LinkError' (line 202)
        LinkError_9495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 22), 'LinkError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 202, 16), LinkError_9495, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 178)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 168)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug(...): (line 204)
        # Processing the call arguments (line 204)
        str_9498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 22), 'str', 'skipping %s (up-to-date)')
        # Getting the type of 'output_filename' (line 204)
        output_filename_9499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 50), 'output_filename', False)
        # Processing the call keyword arguments (line 204)
        kwargs_9500 = {}
        # Getting the type of 'log' (line 204)
        log_9496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 204)
        debug_9497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), log_9496, 'debug')
        # Calling debug(args, kwargs) (line 204)
        debug_call_result_9501 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), debug_9497, *[str_9498, output_filename_9499], **kwargs_9500)
        
        # SSA join for if statement (line 168)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'link(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'link' in the type store
        # Getting the type of 'stypy_return_type' (line 152)
        stypy_return_type_9502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9502)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'link'
        return stypy_return_type_9502


    @norecursion
    def library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'library_dir_option'
        module_type_store = module_type_store.open_function_context('library_dir_option', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnixCCompiler.library_dir_option.__dict__.__setitem__('stypy_localization', localization)
        UnixCCompiler.library_dir_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnixCCompiler.library_dir_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnixCCompiler.library_dir_option.__dict__.__setitem__('stypy_function_name', 'UnixCCompiler.library_dir_option')
        UnixCCompiler.library_dir_option.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        UnixCCompiler.library_dir_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnixCCompiler.library_dir_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnixCCompiler.library_dir_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnixCCompiler.library_dir_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnixCCompiler.library_dir_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnixCCompiler.library_dir_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnixCCompiler.library_dir_option', ['dir'], None, None, defaults, varargs, kwargs)

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

        str_9503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 15), 'str', '-L')
        # Getting the type of 'dir' (line 211)
        dir_9504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 22), 'dir')
        # Applying the binary operator '+' (line 211)
        result_add_9505 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 15), '+', str_9503, dir_9504)
        
        # Assigning a type to the variable 'stypy_return_type' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'stypy_return_type', result_add_9505)
        
        # ################# End of 'library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_9506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9506)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'library_dir_option'
        return stypy_return_type_9506


    @norecursion
    def _is_gcc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_is_gcc'
        module_type_store = module_type_store.open_function_context('_is_gcc', 213, 4, False)
        # Assigning a type to the variable 'self' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnixCCompiler._is_gcc.__dict__.__setitem__('stypy_localization', localization)
        UnixCCompiler._is_gcc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnixCCompiler._is_gcc.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnixCCompiler._is_gcc.__dict__.__setitem__('stypy_function_name', 'UnixCCompiler._is_gcc')
        UnixCCompiler._is_gcc.__dict__.__setitem__('stypy_param_names_list', ['compiler_name'])
        UnixCCompiler._is_gcc.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnixCCompiler._is_gcc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnixCCompiler._is_gcc.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnixCCompiler._is_gcc.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnixCCompiler._is_gcc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnixCCompiler._is_gcc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnixCCompiler._is_gcc', ['compiler_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_is_gcc', localization, ['compiler_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_is_gcc(...)' code ##################

        
        # Evaluating a boolean operation
        
        str_9507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 15), 'str', 'gcc')
        # Getting the type of 'compiler_name' (line 214)
        compiler_name_9508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 24), 'compiler_name')
        # Applying the binary operator 'in' (line 214)
        result_contains_9509 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 15), 'in', str_9507, compiler_name_9508)
        
        
        str_9510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 41), 'str', 'g++')
        # Getting the type of 'compiler_name' (line 214)
        compiler_name_9511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 50), 'compiler_name')
        # Applying the binary operator 'in' (line 214)
        result_contains_9512 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 41), 'in', str_9510, compiler_name_9511)
        
        # Applying the binary operator 'or' (line 214)
        result_or_keyword_9513 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 15), 'or', result_contains_9509, result_contains_9512)
        
        # Assigning a type to the variable 'stypy_return_type' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'stypy_return_type', result_or_keyword_9513)
        
        # ################# End of '_is_gcc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_is_gcc' in the type store
        # Getting the type of 'stypy_return_type' (line 213)
        stypy_return_type_9514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9514)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_is_gcc'
        return stypy_return_type_9514


    @norecursion
    def runtime_library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runtime_library_dir_option'
        module_type_store = module_type_store.open_function_context('runtime_library_dir_option', 216, 4, False)
        # Assigning a type to the variable 'self' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnixCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_localization', localization)
        UnixCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnixCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnixCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_function_name', 'UnixCCompiler.runtime_library_dir_option')
        UnixCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        UnixCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnixCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnixCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnixCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnixCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnixCCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnixCCompiler.runtime_library_dir_option', ['dir'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 229):
        
        # Assigning a Call to a Name (line 229):
        
        # Call to basename(...): (line 229)
        # Processing the call arguments (line 229)
        
        # Call to get_config_var(...): (line 229)
        # Processing the call arguments (line 229)
        str_9520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 61), 'str', 'CC')
        # Processing the call keyword arguments (line 229)
        kwargs_9521 = {}
        # Getting the type of 'sysconfig' (line 229)
        sysconfig_9518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 36), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 229)
        get_config_var_9519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 36), sysconfig_9518, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 229)
        get_config_var_call_result_9522 = invoke(stypy.reporting.localization.Localization(__file__, 229, 36), get_config_var_9519, *[str_9520], **kwargs_9521)
        
        # Processing the call keyword arguments (line 229)
        kwargs_9523 = {}
        # Getting the type of 'os' (line 229)
        os_9515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 229)
        path_9516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 19), os_9515, 'path')
        # Obtaining the member 'basename' of a type (line 229)
        basename_9517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 19), path_9516, 'basename')
        # Calling basename(args, kwargs) (line 229)
        basename_call_result_9524 = invoke(stypy.reporting.localization.Localization(__file__, 229, 19), basename_9517, *[get_config_var_call_result_9522], **kwargs_9523)
        
        # Assigning a type to the variable 'compiler' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'compiler', basename_call_result_9524)
        
        
        
        # Obtaining the type of the subscript
        int_9525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 25), 'int')
        slice_9526 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 230, 11), None, int_9525, None)
        # Getting the type of 'sys' (line 230)
        sys_9527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 230)
        platform_9528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 11), sys_9527, 'platform')
        # Obtaining the member '__getitem__' of a type (line 230)
        getitem___9529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 11), platform_9528, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 230)
        subscript_call_result_9530 = invoke(stypy.reporting.localization.Localization(__file__, 230, 11), getitem___9529, slice_9526)
        
        str_9531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 31), 'str', 'darwin')
        # Applying the binary operator '==' (line 230)
        result_eq_9532 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 11), '==', subscript_call_result_9530, str_9531)
        
        # Testing the type of an if condition (line 230)
        if_condition_9533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 8), result_eq_9532)
        # Assigning a type to the variable 'if_condition_9533' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'if_condition_9533', if_condition_9533)
        # SSA begins for if statement (line 230)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_9534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 19), 'str', '-L')
        # Getting the type of 'dir' (line 232)
        dir_9535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 26), 'dir')
        # Applying the binary operator '+' (line 232)
        result_add_9536 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 19), '+', str_9534, dir_9535)
        
        # Assigning a type to the variable 'stypy_return_type' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'stypy_return_type', result_add_9536)
        # SSA branch for the else part of an if statement (line 230)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        int_9537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 27), 'int')
        slice_9538 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 233, 13), None, int_9537, None)
        # Getting the type of 'sys' (line 233)
        sys_9539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 13), 'sys')
        # Obtaining the member 'platform' of a type (line 233)
        platform_9540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 13), sys_9539, 'platform')
        # Obtaining the member '__getitem__' of a type (line 233)
        getitem___9541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 13), platform_9540, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 233)
        subscript_call_result_9542 = invoke(stypy.reporting.localization.Localization(__file__, 233, 13), getitem___9541, slice_9538)
        
        str_9543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 33), 'str', 'freebsd')
        # Applying the binary operator '==' (line 233)
        result_eq_9544 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 13), '==', subscript_call_result_9542, str_9543)
        
        # Testing the type of an if condition (line 233)
        if_condition_9545 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 13), result_eq_9544)
        # Assigning a type to the variable 'if_condition_9545' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 13), 'if_condition_9545', if_condition_9545)
        # SSA begins for if statement (line 233)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_9546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 19), 'str', '-Wl,-rpath=')
        # Getting the type of 'dir' (line 234)
        dir_9547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 35), 'dir')
        # Applying the binary operator '+' (line 234)
        result_add_9548 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 19), '+', str_9546, dir_9547)
        
        # Assigning a type to the variable 'stypy_return_type' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'stypy_return_type', result_add_9548)
        # SSA branch for the else part of an if statement (line 233)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Obtaining the type of the subscript
        int_9549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 27), 'int')
        slice_9550 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 235, 13), None, int_9549, None)
        # Getting the type of 'sys' (line 235)
        sys_9551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 13), 'sys')
        # Obtaining the member 'platform' of a type (line 235)
        platform_9552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 13), sys_9551, 'platform')
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___9553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 13), platform_9552, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_9554 = invoke(stypy.reporting.localization.Localization(__file__, 235, 13), getitem___9553, slice_9550)
        
        str_9555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 33), 'str', 'hp-ux')
        # Applying the binary operator '==' (line 235)
        result_eq_9556 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 13), '==', subscript_call_result_9554, str_9555)
        
        # Testing the type of an if condition (line 235)
        if_condition_9557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 13), result_eq_9556)
        # Assigning a type to the variable 'if_condition_9557' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 13), 'if_condition_9557', if_condition_9557)
        # SSA begins for if statement (line 235)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to _is_gcc(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'compiler' (line 236)
        compiler_9560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 28), 'compiler', False)
        # Processing the call keyword arguments (line 236)
        kwargs_9561 = {}
        # Getting the type of 'self' (line 236)
        self_9558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 'self', False)
        # Obtaining the member '_is_gcc' of a type (line 236)
        _is_gcc_9559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 15), self_9558, '_is_gcc')
        # Calling _is_gcc(args, kwargs) (line 236)
        _is_gcc_call_result_9562 = invoke(stypy.reporting.localization.Localization(__file__, 236, 15), _is_gcc_9559, *[compiler_9560], **kwargs_9561)
        
        # Testing the type of an if condition (line 236)
        if_condition_9563 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 236, 12), _is_gcc_call_result_9562)
        # Assigning a type to the variable 'if_condition_9563' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'if_condition_9563', if_condition_9563)
        # SSA begins for if statement (line 236)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 237)
        list_9564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 237)
        # Adding element type (line 237)
        str_9565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 24), 'str', '-Wl,+s')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 23), list_9564, str_9565)
        # Adding element type (line 237)
        str_9566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 34), 'str', '-L')
        # Getting the type of 'dir' (line 237)
        dir_9567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 41), 'dir')
        # Applying the binary operator '+' (line 237)
        result_add_9568 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 34), '+', str_9566, dir_9567)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 23), list_9564, result_add_9568)
        
        # Assigning a type to the variable 'stypy_return_type' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'stypy_return_type', list_9564)
        # SSA join for if statement (line 236)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'list' (line 238)
        list_9569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 238)
        # Adding element type (line 238)
        str_9570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 20), 'str', '+s')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 19), list_9569, str_9570)
        # Adding element type (line 238)
        str_9571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 26), 'str', '-L')
        # Getting the type of 'dir' (line 238)
        dir_9572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 33), 'dir')
        # Applying the binary operator '+' (line 238)
        result_add_9573 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 26), '+', str_9571, dir_9572)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 19), list_9569, result_add_9573)
        
        # Assigning a type to the variable 'stypy_return_type' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'stypy_return_type', list_9569)
        # SSA branch for the else part of an if statement (line 235)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        
        # Obtaining the type of the subscript
        int_9574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 27), 'int')
        slice_9575 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 239, 13), None, int_9574, None)
        # Getting the type of 'sys' (line 239)
        sys_9576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 13), 'sys')
        # Obtaining the member 'platform' of a type (line 239)
        platform_9577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 13), sys_9576, 'platform')
        # Obtaining the member '__getitem__' of a type (line 239)
        getitem___9578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 13), platform_9577, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 239)
        subscript_call_result_9579 = invoke(stypy.reporting.localization.Localization(__file__, 239, 13), getitem___9578, slice_9575)
        
        str_9580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 33), 'str', 'irix646')
        # Applying the binary operator '==' (line 239)
        result_eq_9581 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 13), '==', subscript_call_result_9579, str_9580)
        
        
        
        # Obtaining the type of the subscript
        int_9582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 60), 'int')
        slice_9583 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 239, 46), None, int_9582, None)
        # Getting the type of 'sys' (line 239)
        sys_9584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 46), 'sys')
        # Obtaining the member 'platform' of a type (line 239)
        platform_9585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 46), sys_9584, 'platform')
        # Obtaining the member '__getitem__' of a type (line 239)
        getitem___9586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 46), platform_9585, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 239)
        subscript_call_result_9587 = invoke(stypy.reporting.localization.Localization(__file__, 239, 46), getitem___9586, slice_9583)
        
        str_9588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 66), 'str', 'osf1V5')
        # Applying the binary operator '==' (line 239)
        result_eq_9589 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 46), '==', subscript_call_result_9587, str_9588)
        
        # Applying the binary operator 'or' (line 239)
        result_or_keyword_9590 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 13), 'or', result_eq_9581, result_eq_9589)
        
        # Testing the type of an if condition (line 239)
        if_condition_9591 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 13), result_or_keyword_9590)
        # Assigning a type to the variable 'if_condition_9591' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 13), 'if_condition_9591', if_condition_9591)
        # SSA begins for if statement (line 239)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 240)
        list_9592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 240)
        # Adding element type (line 240)
        str_9593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 20), 'str', '-rpath')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 19), list_9592, str_9593)
        # Adding element type (line 240)
        # Getting the type of 'dir' (line 240)
        dir_9594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 30), 'dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 240, 19), list_9592, dir_9594)
        
        # Assigning a type to the variable 'stypy_return_type' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'stypy_return_type', list_9592)
        # SSA branch for the else part of an if statement (line 239)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to _is_gcc(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'compiler' (line 241)
        compiler_9597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 26), 'compiler', False)
        # Processing the call keyword arguments (line 241)
        kwargs_9598 = {}
        # Getting the type of 'self' (line 241)
        self_9595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 13), 'self', False)
        # Obtaining the member '_is_gcc' of a type (line 241)
        _is_gcc_9596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 13), self_9595, '_is_gcc')
        # Calling _is_gcc(args, kwargs) (line 241)
        _is_gcc_call_result_9599 = invoke(stypy.reporting.localization.Localization(__file__, 241, 13), _is_gcc_9596, *[compiler_9597], **kwargs_9598)
        
        # Testing the type of an if condition (line 241)
        if_condition_9600 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 13), _is_gcc_call_result_9599)
        # Assigning a type to the variable 'if_condition_9600' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 13), 'if_condition_9600', if_condition_9600)
        # SSA begins for if statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_9601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 19), 'str', '-Wl,-R')
        # Getting the type of 'dir' (line 242)
        dir_9602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 30), 'dir')
        # Applying the binary operator '+' (line 242)
        result_add_9603 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 19), '+', str_9601, dir_9602)
        
        # Assigning a type to the variable 'stypy_return_type' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'stypy_return_type', result_add_9603)
        # SSA branch for the else part of an if statement (line 241)
        module_type_store.open_ssa_branch('else')
        str_9604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 19), 'str', '-R')
        # Getting the type of 'dir' (line 244)
        dir_9605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 26), 'dir')
        # Applying the binary operator '+' (line 244)
        result_add_9606 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 19), '+', str_9604, dir_9605)
        
        # Assigning a type to the variable 'stypy_return_type' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'stypy_return_type', result_add_9606)
        # SSA join for if statement (line 241)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 239)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 235)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 233)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 230)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'runtime_library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runtime_library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 216)
        stypy_return_type_9607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9607)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runtime_library_dir_option'
        return stypy_return_type_9607


    @norecursion
    def library_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'library_option'
        module_type_store = module_type_store.open_function_context('library_option', 246, 4, False)
        # Assigning a type to the variable 'self' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnixCCompiler.library_option.__dict__.__setitem__('stypy_localization', localization)
        UnixCCompiler.library_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnixCCompiler.library_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnixCCompiler.library_option.__dict__.__setitem__('stypy_function_name', 'UnixCCompiler.library_option')
        UnixCCompiler.library_option.__dict__.__setitem__('stypy_param_names_list', ['lib'])
        UnixCCompiler.library_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnixCCompiler.library_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnixCCompiler.library_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnixCCompiler.library_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnixCCompiler.library_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnixCCompiler.library_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnixCCompiler.library_option', ['lib'], None, None, defaults, varargs, kwargs)

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

        str_9608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 15), 'str', '-l')
        # Getting the type of 'lib' (line 247)
        lib_9609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 22), 'lib')
        # Applying the binary operator '+' (line 247)
        result_add_9610 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 15), '+', str_9608, lib_9609)
        
        # Assigning a type to the variable 'stypy_return_type' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'stypy_return_type', result_add_9610)
        
        # ################# End of 'library_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'library_option' in the type store
        # Getting the type of 'stypy_return_type' (line 246)
        stypy_return_type_9611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9611)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'library_option'
        return stypy_return_type_9611


    @norecursion
    def find_library_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_9612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 49), 'int')
        defaults = [int_9612]
        # Create a new context for function 'find_library_file'
        module_type_store = module_type_store.open_function_context('find_library_file', 249, 4, False)
        # Assigning a type to the variable 'self' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UnixCCompiler.find_library_file.__dict__.__setitem__('stypy_localization', localization)
        UnixCCompiler.find_library_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnixCCompiler.find_library_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnixCCompiler.find_library_file.__dict__.__setitem__('stypy_function_name', 'UnixCCompiler.find_library_file')
        UnixCCompiler.find_library_file.__dict__.__setitem__('stypy_param_names_list', ['dirs', 'lib', 'debug'])
        UnixCCompiler.find_library_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnixCCompiler.find_library_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnixCCompiler.find_library_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnixCCompiler.find_library_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnixCCompiler.find_library_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnixCCompiler.find_library_file.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnixCCompiler.find_library_file', ['dirs', 'lib', 'debug'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 250):
        
        # Assigning a Call to a Name (line 250):
        
        # Call to library_filename(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'lib' (line 250)
        lib_9615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 41), 'lib', False)
        # Processing the call keyword arguments (line 250)
        str_9616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 55), 'str', 'shared')
        keyword_9617 = str_9616
        kwargs_9618 = {'lib_type': keyword_9617}
        # Getting the type of 'self' (line 250)
        self_9613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 19), 'self', False)
        # Obtaining the member 'library_filename' of a type (line 250)
        library_filename_9614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 19), self_9613, 'library_filename')
        # Calling library_filename(args, kwargs) (line 250)
        library_filename_call_result_9619 = invoke(stypy.reporting.localization.Localization(__file__, 250, 19), library_filename_9614, *[lib_9615], **kwargs_9618)
        
        # Assigning a type to the variable 'shared_f' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'shared_f', library_filename_call_result_9619)
        
        # Assigning a Call to a Name (line 251):
        
        # Assigning a Call to a Name (line 251):
        
        # Call to library_filename(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'lib' (line 251)
        lib_9622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 40), 'lib', False)
        # Processing the call keyword arguments (line 251)
        str_9623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 54), 'str', 'dylib')
        keyword_9624 = str_9623
        kwargs_9625 = {'lib_type': keyword_9624}
        # Getting the type of 'self' (line 251)
        self_9620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 18), 'self', False)
        # Obtaining the member 'library_filename' of a type (line 251)
        library_filename_9621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 18), self_9620, 'library_filename')
        # Calling library_filename(args, kwargs) (line 251)
        library_filename_call_result_9626 = invoke(stypy.reporting.localization.Localization(__file__, 251, 18), library_filename_9621, *[lib_9622], **kwargs_9625)
        
        # Assigning a type to the variable 'dylib_f' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'dylib_f', library_filename_call_result_9626)
        
        # Assigning a Call to a Name (line 252):
        
        # Assigning a Call to a Name (line 252):
        
        # Call to library_filename(...): (line 252)
        # Processing the call arguments (line 252)
        # Getting the type of 'lib' (line 252)
        lib_9629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 45), 'lib', False)
        # Processing the call keyword arguments (line 252)
        str_9630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 59), 'str', 'xcode_stub')
        keyword_9631 = str_9630
        kwargs_9632 = {'lib_type': keyword_9631}
        # Getting the type of 'self' (line 252)
        self_9627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 23), 'self', False)
        # Obtaining the member 'library_filename' of a type (line 252)
        library_filename_9628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 23), self_9627, 'library_filename')
        # Calling library_filename(args, kwargs) (line 252)
        library_filename_call_result_9633 = invoke(stypy.reporting.localization.Localization(__file__, 252, 23), library_filename_9628, *[lib_9629], **kwargs_9632)
        
        # Assigning a type to the variable 'xcode_stub_f' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'xcode_stub_f', library_filename_call_result_9633)
        
        # Assigning a Call to a Name (line 253):
        
        # Assigning a Call to a Name (line 253):
        
        # Call to library_filename(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'lib' (line 253)
        lib_9636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 41), 'lib', False)
        # Processing the call keyword arguments (line 253)
        str_9637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 55), 'str', 'static')
        keyword_9638 = str_9637
        kwargs_9639 = {'lib_type': keyword_9638}
        # Getting the type of 'self' (line 253)
        self_9634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 19), 'self', False)
        # Obtaining the member 'library_filename' of a type (line 253)
        library_filename_9635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 19), self_9634, 'library_filename')
        # Calling library_filename(args, kwargs) (line 253)
        library_filename_call_result_9640 = invoke(stypy.reporting.localization.Localization(__file__, 253, 19), library_filename_9635, *[lib_9636], **kwargs_9639)
        
        # Assigning a type to the variable 'static_f' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'static_f', library_filename_call_result_9640)
        
        
        # Getting the type of 'sys' (line 255)
        sys_9641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 255)
        platform_9642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 11), sys_9641, 'platform')
        str_9643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 27), 'str', 'darwin')
        # Applying the binary operator '==' (line 255)
        result_eq_9644 = python_operator(stypy.reporting.localization.Localization(__file__, 255, 11), '==', platform_9642, str_9643)
        
        # Testing the type of an if condition (line 255)
        if_condition_9645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 255, 8), result_eq_9644)
        # Assigning a type to the variable 'if_condition_9645' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'if_condition_9645', if_condition_9645)
        # SSA begins for if statement (line 255)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 274):
        
        # Assigning a Call to a Name (line 274):
        
        # Call to get_config_var(...): (line 274)
        # Processing the call arguments (line 274)
        str_9648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 46), 'str', 'CFLAGS')
        # Processing the call keyword arguments (line 274)
        kwargs_9649 = {}
        # Getting the type of 'sysconfig' (line 274)
        sysconfig_9646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 21), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 274)
        get_config_var_9647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 21), sysconfig_9646, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 274)
        get_config_var_call_result_9650 = invoke(stypy.reporting.localization.Localization(__file__, 274, 21), get_config_var_9647, *[str_9648], **kwargs_9649)
        
        # Assigning a type to the variable 'cflags' (line 274)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'cflags', get_config_var_call_result_9650)
        
        # Assigning a Call to a Name (line 275):
        
        # Assigning a Call to a Name (line 275):
        
        # Call to search(...): (line 275)
        # Processing the call arguments (line 275)
        str_9653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 26), 'str', '-isysroot\\s+(\\S+)')
        # Getting the type of 'cflags' (line 275)
        cflags_9654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 48), 'cflags', False)
        # Processing the call keyword arguments (line 275)
        kwargs_9655 = {}
        # Getting the type of 're' (line 275)
        re_9651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 're', False)
        # Obtaining the member 'search' of a type (line 275)
        search_9652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 16), re_9651, 'search')
        # Calling search(args, kwargs) (line 275)
        search_call_result_9656 = invoke(stypy.reporting.localization.Localization(__file__, 275, 16), search_9652, *[str_9653, cflags_9654], **kwargs_9655)
        
        # Assigning a type to the variable 'm' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'm', search_call_result_9656)
        
        # Type idiom detected: calculating its left and rigth part (line 276)
        # Getting the type of 'm' (line 276)
        m_9657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 15), 'm')
        # Getting the type of 'None' (line 276)
        None_9658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 20), 'None')
        
        (may_be_9659, more_types_in_union_9660) = may_be_none(m_9657, None_9658)

        if may_be_9659:

            if more_types_in_union_9660:
                # Runtime conditional SSA (line 276)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Name (line 277):
            
            # Assigning a Str to a Name (line 277):
            str_9661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 26), 'str', '/')
            # Assigning a type to the variable 'sysroot' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 16), 'sysroot', str_9661)

            if more_types_in_union_9660:
                # Runtime conditional SSA for else branch (line 276)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_9659) or more_types_in_union_9660):
            
            # Assigning a Call to a Name (line 279):
            
            # Assigning a Call to a Name (line 279):
            
            # Call to group(...): (line 279)
            # Processing the call arguments (line 279)
            int_9664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 34), 'int')
            # Processing the call keyword arguments (line 279)
            kwargs_9665 = {}
            # Getting the type of 'm' (line 279)
            m_9662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 26), 'm', False)
            # Obtaining the member 'group' of a type (line 279)
            group_9663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 26), m_9662, 'group')
            # Calling group(args, kwargs) (line 279)
            group_call_result_9666 = invoke(stypy.reporting.localization.Localization(__file__, 279, 26), group_9663, *[int_9664], **kwargs_9665)
            
            # Assigning a type to the variable 'sysroot' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 16), 'sysroot', group_call_result_9666)

            if (may_be_9659 and more_types_in_union_9660):
                # SSA join for if statement (line 276)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 255)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'dirs' (line 283)
        dirs_9667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 19), 'dirs')
        # Testing the type of a for loop iterable (line 283)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 283, 8), dirs_9667)
        # Getting the type of the for loop variable (line 283)
        for_loop_var_9668 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 283, 8), dirs_9667)
        # Assigning a type to the variable 'dir' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'dir', for_loop_var_9668)
        # SSA begins for a for statement (line 283)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 284):
        
        # Assigning a Call to a Name (line 284):
        
        # Call to join(...): (line 284)
        # Processing the call arguments (line 284)
        # Getting the type of 'dir' (line 284)
        dir_9672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 34), 'dir', False)
        # Getting the type of 'shared_f' (line 284)
        shared_f_9673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 39), 'shared_f', False)
        # Processing the call keyword arguments (line 284)
        kwargs_9674 = {}
        # Getting the type of 'os' (line 284)
        os_9669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 284)
        path_9670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 21), os_9669, 'path')
        # Obtaining the member 'join' of a type (line 284)
        join_9671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 21), path_9670, 'join')
        # Calling join(args, kwargs) (line 284)
        join_call_result_9675 = invoke(stypy.reporting.localization.Localization(__file__, 284, 21), join_9671, *[dir_9672, shared_f_9673], **kwargs_9674)
        
        # Assigning a type to the variable 'shared' (line 284)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'shared', join_call_result_9675)
        
        # Assigning a Call to a Name (line 285):
        
        # Assigning a Call to a Name (line 285):
        
        # Call to join(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'dir' (line 285)
        dir_9679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 33), 'dir', False)
        # Getting the type of 'dylib_f' (line 285)
        dylib_f_9680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 38), 'dylib_f', False)
        # Processing the call keyword arguments (line 285)
        kwargs_9681 = {}
        # Getting the type of 'os' (line 285)
        os_9676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 285)
        path_9677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 20), os_9676, 'path')
        # Obtaining the member 'join' of a type (line 285)
        join_9678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 20), path_9677, 'join')
        # Calling join(args, kwargs) (line 285)
        join_call_result_9682 = invoke(stypy.reporting.localization.Localization(__file__, 285, 20), join_9678, *[dir_9679, dylib_f_9680], **kwargs_9681)
        
        # Assigning a type to the variable 'dylib' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'dylib', join_call_result_9682)
        
        # Assigning a Call to a Name (line 286):
        
        # Assigning a Call to a Name (line 286):
        
        # Call to join(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 'dir' (line 286)
        dir_9686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 34), 'dir', False)
        # Getting the type of 'static_f' (line 286)
        static_f_9687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 39), 'static_f', False)
        # Processing the call keyword arguments (line 286)
        kwargs_9688 = {}
        # Getting the type of 'os' (line 286)
        os_9683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 286)
        path_9684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 21), os_9683, 'path')
        # Obtaining the member 'join' of a type (line 286)
        join_9685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 21), path_9684, 'join')
        # Calling join(args, kwargs) (line 286)
        join_call_result_9689 = invoke(stypy.reporting.localization.Localization(__file__, 286, 21), join_9685, *[dir_9686, static_f_9687], **kwargs_9688)
        
        # Assigning a type to the variable 'static' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'static', join_call_result_9689)
        
        # Assigning a Call to a Name (line 287):
        
        # Assigning a Call to a Name (line 287):
        
        # Call to join(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'dir' (line 287)
        dir_9693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 38), 'dir', False)
        # Getting the type of 'xcode_stub_f' (line 287)
        xcode_stub_f_9694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 43), 'xcode_stub_f', False)
        # Processing the call keyword arguments (line 287)
        kwargs_9695 = {}
        # Getting the type of 'os' (line 287)
        os_9690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 287)
        path_9691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 25), os_9690, 'path')
        # Obtaining the member 'join' of a type (line 287)
        join_9692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 25), path_9691, 'join')
        # Calling join(args, kwargs) (line 287)
        join_call_result_9696 = invoke(stypy.reporting.localization.Localization(__file__, 287, 25), join_9692, *[dir_9693, xcode_stub_f_9694], **kwargs_9695)
        
        # Assigning a type to the variable 'xcode_stub' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'xcode_stub', join_call_result_9696)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sys' (line 289)
        sys_9697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 15), 'sys')
        # Obtaining the member 'platform' of a type (line 289)
        platform_9698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 15), sys_9697, 'platform')
        str_9699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 31), 'str', 'darwin')
        # Applying the binary operator '==' (line 289)
        result_eq_9700 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 15), '==', platform_9698, str_9699)
        
        
        # Evaluating a boolean operation
        
        # Call to startswith(...): (line 290)
        # Processing the call arguments (line 290)
        str_9703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 31), 'str', '/System/')
        # Processing the call keyword arguments (line 290)
        kwargs_9704 = {}
        # Getting the type of 'dir' (line 290)
        dir_9701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 16), 'dir', False)
        # Obtaining the member 'startswith' of a type (line 290)
        startswith_9702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 16), dir_9701, 'startswith')
        # Calling startswith(args, kwargs) (line 290)
        startswith_call_result_9705 = invoke(stypy.reporting.localization.Localization(__file__, 290, 16), startswith_9702, *[str_9703], **kwargs_9704)
        
        
        # Evaluating a boolean operation
        
        # Call to startswith(...): (line 291)
        # Processing the call arguments (line 291)
        str_9708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 31), 'str', '/usr/')
        # Processing the call keyword arguments (line 291)
        kwargs_9709 = {}
        # Getting the type of 'dir' (line 291)
        dir_9706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'dir', False)
        # Obtaining the member 'startswith' of a type (line 291)
        startswith_9707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 16), dir_9706, 'startswith')
        # Calling startswith(args, kwargs) (line 291)
        startswith_call_result_9710 = invoke(stypy.reporting.localization.Localization(__file__, 291, 16), startswith_9707, *[str_9708], **kwargs_9709)
        
        
        
        # Call to startswith(...): (line 291)
        # Processing the call arguments (line 291)
        str_9713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 63), 'str', '/usr/local/')
        # Processing the call keyword arguments (line 291)
        kwargs_9714 = {}
        # Getting the type of 'dir' (line 291)
        dir_9711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 48), 'dir', False)
        # Obtaining the member 'startswith' of a type (line 291)
        startswith_9712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 48), dir_9711, 'startswith')
        # Calling startswith(args, kwargs) (line 291)
        startswith_call_result_9715 = invoke(stypy.reporting.localization.Localization(__file__, 291, 48), startswith_9712, *[str_9713], **kwargs_9714)
        
        # Applying the 'not' unary operator (line 291)
        result_not__9716 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 44), 'not', startswith_call_result_9715)
        
        # Applying the binary operator 'and' (line 291)
        result_and_keyword_9717 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 16), 'and', startswith_call_result_9710, result_not__9716)
        
        # Applying the binary operator 'or' (line 290)
        result_or_keyword_9718 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 16), 'or', startswith_call_result_9705, result_and_keyword_9717)
        
        # Applying the binary operator 'and' (line 289)
        result_and_keyword_9719 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 15), 'and', result_eq_9700, result_or_keyword_9718)
        
        # Testing the type of an if condition (line 289)
        if_condition_9720 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 12), result_and_keyword_9719)
        # Assigning a type to the variable 'if_condition_9720' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 12), 'if_condition_9720', if_condition_9720)
        # SSA begins for if statement (line 289)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Call to join(...): (line 293)
        # Processing the call arguments (line 293)
        # Getting the type of 'sysroot' (line 293)
        sysroot_9724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 38), 'sysroot', False)
        
        # Obtaining the type of the subscript
        int_9725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 51), 'int')
        slice_9726 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 293, 47), int_9725, None, None)
        # Getting the type of 'dir' (line 293)
        dir_9727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 47), 'dir', False)
        # Obtaining the member '__getitem__' of a type (line 293)
        getitem___9728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 47), dir_9727, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 293)
        subscript_call_result_9729 = invoke(stypy.reporting.localization.Localization(__file__, 293, 47), getitem___9728, slice_9726)
        
        # Getting the type of 'shared_f' (line 293)
        shared_f_9730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 56), 'shared_f', False)
        # Processing the call keyword arguments (line 293)
        kwargs_9731 = {}
        # Getting the type of 'os' (line 293)
        os_9721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 293)
        path_9722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 25), os_9721, 'path')
        # Obtaining the member 'join' of a type (line 293)
        join_9723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 25), path_9722, 'join')
        # Calling join(args, kwargs) (line 293)
        join_call_result_9732 = invoke(stypy.reporting.localization.Localization(__file__, 293, 25), join_9723, *[sysroot_9724, subscript_call_result_9729, shared_f_9730], **kwargs_9731)
        
        # Assigning a type to the variable 'shared' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 16), 'shared', join_call_result_9732)
        
        # Assigning a Call to a Name (line 294):
        
        # Assigning a Call to a Name (line 294):
        
        # Call to join(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'sysroot' (line 294)
        sysroot_9736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 37), 'sysroot', False)
        
        # Obtaining the type of the subscript
        int_9737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 50), 'int')
        slice_9738 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 294, 46), int_9737, None, None)
        # Getting the type of 'dir' (line 294)
        dir_9739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 46), 'dir', False)
        # Obtaining the member '__getitem__' of a type (line 294)
        getitem___9740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 46), dir_9739, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 294)
        subscript_call_result_9741 = invoke(stypy.reporting.localization.Localization(__file__, 294, 46), getitem___9740, slice_9738)
        
        # Getting the type of 'dylib_f' (line 294)
        dylib_f_9742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 55), 'dylib_f', False)
        # Processing the call keyword arguments (line 294)
        kwargs_9743 = {}
        # Getting the type of 'os' (line 294)
        os_9733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 294)
        path_9734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 24), os_9733, 'path')
        # Obtaining the member 'join' of a type (line 294)
        join_9735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 24), path_9734, 'join')
        # Calling join(args, kwargs) (line 294)
        join_call_result_9744 = invoke(stypy.reporting.localization.Localization(__file__, 294, 24), join_9735, *[sysroot_9736, subscript_call_result_9741, dylib_f_9742], **kwargs_9743)
        
        # Assigning a type to the variable 'dylib' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 16), 'dylib', join_call_result_9744)
        
        # Assigning a Call to a Name (line 295):
        
        # Assigning a Call to a Name (line 295):
        
        # Call to join(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'sysroot' (line 295)
        sysroot_9748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 38), 'sysroot', False)
        
        # Obtaining the type of the subscript
        int_9749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 51), 'int')
        slice_9750 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 295, 47), int_9749, None, None)
        # Getting the type of 'dir' (line 295)
        dir_9751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 47), 'dir', False)
        # Obtaining the member '__getitem__' of a type (line 295)
        getitem___9752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 47), dir_9751, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 295)
        subscript_call_result_9753 = invoke(stypy.reporting.localization.Localization(__file__, 295, 47), getitem___9752, slice_9750)
        
        # Getting the type of 'static_f' (line 295)
        static_f_9754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 56), 'static_f', False)
        # Processing the call keyword arguments (line 295)
        kwargs_9755 = {}
        # Getting the type of 'os' (line 295)
        os_9745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 295)
        path_9746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 25), os_9745, 'path')
        # Obtaining the member 'join' of a type (line 295)
        join_9747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 25), path_9746, 'join')
        # Calling join(args, kwargs) (line 295)
        join_call_result_9756 = invoke(stypy.reporting.localization.Localization(__file__, 295, 25), join_9747, *[sysroot_9748, subscript_call_result_9753, static_f_9754], **kwargs_9755)
        
        # Assigning a type to the variable 'static' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'static', join_call_result_9756)
        
        # Assigning a Call to a Name (line 296):
        
        # Assigning a Call to a Name (line 296):
        
        # Call to join(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'sysroot' (line 296)
        sysroot_9760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 42), 'sysroot', False)
        
        # Obtaining the type of the subscript
        int_9761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 55), 'int')
        slice_9762 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 296, 51), int_9761, None, None)
        # Getting the type of 'dir' (line 296)
        dir_9763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 51), 'dir', False)
        # Obtaining the member '__getitem__' of a type (line 296)
        getitem___9764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 51), dir_9763, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 296)
        subscript_call_result_9765 = invoke(stypy.reporting.localization.Localization(__file__, 296, 51), getitem___9764, slice_9762)
        
        # Getting the type of 'xcode_stub_f' (line 296)
        xcode_stub_f_9766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 60), 'xcode_stub_f', False)
        # Processing the call keyword arguments (line 296)
        kwargs_9767 = {}
        # Getting the type of 'os' (line 296)
        os_9757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 29), 'os', False)
        # Obtaining the member 'path' of a type (line 296)
        path_9758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 29), os_9757, 'path')
        # Obtaining the member 'join' of a type (line 296)
        join_9759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 29), path_9758, 'join')
        # Calling join(args, kwargs) (line 296)
        join_call_result_9768 = invoke(stypy.reporting.localization.Localization(__file__, 296, 29), join_9759, *[sysroot_9760, subscript_call_result_9765, xcode_stub_f_9766], **kwargs_9767)
        
        # Assigning a type to the variable 'xcode_stub' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'xcode_stub', join_call_result_9768)
        # SSA join for if statement (line 289)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to exists(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'dylib' (line 302)
        dylib_9772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 30), 'dylib', False)
        # Processing the call keyword arguments (line 302)
        kwargs_9773 = {}
        # Getting the type of 'os' (line 302)
        os_9769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 302)
        path_9770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 15), os_9769, 'path')
        # Obtaining the member 'exists' of a type (line 302)
        exists_9771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 15), path_9770, 'exists')
        # Calling exists(args, kwargs) (line 302)
        exists_call_result_9774 = invoke(stypy.reporting.localization.Localization(__file__, 302, 15), exists_9771, *[dylib_9772], **kwargs_9773)
        
        # Testing the type of an if condition (line 302)
        if_condition_9775 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 302, 12), exists_call_result_9774)
        # Assigning a type to the variable 'if_condition_9775' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'if_condition_9775', if_condition_9775)
        # SSA begins for if statement (line 302)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'dylib' (line 303)
        dylib_9776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 23), 'dylib')
        # Assigning a type to the variable 'stypy_return_type' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 16), 'stypy_return_type', dylib_9776)
        # SSA branch for the else part of an if statement (line 302)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to exists(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'xcode_stub' (line 304)
        xcode_stub_9780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 32), 'xcode_stub', False)
        # Processing the call keyword arguments (line 304)
        kwargs_9781 = {}
        # Getting the type of 'os' (line 304)
        os_9777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 304)
        path_9778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 17), os_9777, 'path')
        # Obtaining the member 'exists' of a type (line 304)
        exists_9779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 17), path_9778, 'exists')
        # Calling exists(args, kwargs) (line 304)
        exists_call_result_9782 = invoke(stypy.reporting.localization.Localization(__file__, 304, 17), exists_9779, *[xcode_stub_9780], **kwargs_9781)
        
        # Testing the type of an if condition (line 304)
        if_condition_9783 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 304, 17), exists_call_result_9782)
        # Assigning a type to the variable 'if_condition_9783' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 17), 'if_condition_9783', if_condition_9783)
        # SSA begins for if statement (line 304)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'xcode_stub' (line 305)
        xcode_stub_9784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 23), 'xcode_stub')
        # Assigning a type to the variable 'stypy_return_type' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'stypy_return_type', xcode_stub_9784)
        # SSA branch for the else part of an if statement (line 304)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to exists(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 'shared' (line 306)
        shared_9788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 32), 'shared', False)
        # Processing the call keyword arguments (line 306)
        kwargs_9789 = {}
        # Getting the type of 'os' (line 306)
        os_9785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 306)
        path_9786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 17), os_9785, 'path')
        # Obtaining the member 'exists' of a type (line 306)
        exists_9787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 17), path_9786, 'exists')
        # Calling exists(args, kwargs) (line 306)
        exists_call_result_9790 = invoke(stypy.reporting.localization.Localization(__file__, 306, 17), exists_9787, *[shared_9788], **kwargs_9789)
        
        # Testing the type of an if condition (line 306)
        if_condition_9791 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 17), exists_call_result_9790)
        # Assigning a type to the variable 'if_condition_9791' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 17), 'if_condition_9791', if_condition_9791)
        # SSA begins for if statement (line 306)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'shared' (line 307)
        shared_9792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 23), 'shared')
        # Assigning a type to the variable 'stypy_return_type' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 16), 'stypy_return_type', shared_9792)
        # SSA branch for the else part of an if statement (line 306)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to exists(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'static' (line 308)
        static_9796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 32), 'static', False)
        # Processing the call keyword arguments (line 308)
        kwargs_9797 = {}
        # Getting the type of 'os' (line 308)
        os_9793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 308)
        path_9794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 17), os_9793, 'path')
        # Obtaining the member 'exists' of a type (line 308)
        exists_9795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 17), path_9794, 'exists')
        # Calling exists(args, kwargs) (line 308)
        exists_call_result_9798 = invoke(stypy.reporting.localization.Localization(__file__, 308, 17), exists_9795, *[static_9796], **kwargs_9797)
        
        # Testing the type of an if condition (line 308)
        if_condition_9799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 308, 17), exists_call_result_9798)
        # Assigning a type to the variable 'if_condition_9799' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 17), 'if_condition_9799', if_condition_9799)
        # SSA begins for if statement (line 308)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'static' (line 309)
        static_9800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 23), 'static')
        # Assigning a type to the variable 'stypy_return_type' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 16), 'stypy_return_type', static_9800)
        # SSA join for if statement (line 308)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 306)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 304)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 302)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'None' (line 312)
        None_9801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'stypy_return_type', None_9801)
        
        # ################# End of 'find_library_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_library_file' in the type store
        # Getting the type of 'stypy_return_type' (line 249)
        stypy_return_type_9802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9802)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_library_file'
        return stypy_return_type_9802


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnixCCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'UnixCCompiler' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'UnixCCompiler', UnixCCompiler)

# Assigning a Str to a Name (line 50):
str_9803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 20), 'str', 'unix')
# Getting the type of 'UnixCCompiler'
UnixCCompiler_9804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnixCCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnixCCompiler_9804, 'compiler_type', str_9803)

# Assigning a Dict to a Name (line 58):

# Obtaining an instance of the builtin type 'dict' (line 58)
dict_9805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 58)
# Adding element type (key, value) (line 58)
str_9806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 19), 'str', 'preprocessor')
# Getting the type of 'None' (line 58)
None_9807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 36), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 18), dict_9805, (str_9806, None_9807))
# Adding element type (key, value) (line 58)
str_9808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 19), 'str', 'compiler')

# Obtaining an instance of the builtin type 'list' (line 59)
list_9809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 59)
# Adding element type (line 59)
str_9810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 37), 'str', 'cc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 36), list_9809, str_9810)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 18), dict_9805, (str_9808, list_9809))
# Adding element type (key, value) (line 58)
str_9811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 19), 'str', 'compiler_so')

# Obtaining an instance of the builtin type 'list' (line 60)
list_9812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 60)
# Adding element type (line 60)
str_9813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 37), 'str', 'cc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 36), list_9812, str_9813)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 18), dict_9805, (str_9811, list_9812))
# Adding element type (key, value) (line 58)
str_9814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 19), 'str', 'compiler_cxx')

# Obtaining an instance of the builtin type 'list' (line 61)
list_9815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 61)
# Adding element type (line 61)
str_9816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 37), 'str', 'cc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 36), list_9815, str_9816)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 18), dict_9805, (str_9814, list_9815))
# Adding element type (key, value) (line 58)
str_9817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 19), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 62)
list_9818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 62)
# Adding element type (line 62)
str_9819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 37), 'str', 'cc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 36), list_9818, str_9819)
# Adding element type (line 62)
str_9820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 43), 'str', '-shared')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 36), list_9818, str_9820)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 18), dict_9805, (str_9817, list_9818))
# Adding element type (key, value) (line 58)
str_9821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 19), 'str', 'linker_exe')

# Obtaining an instance of the builtin type 'list' (line 63)
list_9822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 63)
# Adding element type (line 63)
str_9823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 37), 'str', 'cc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 36), list_9822, str_9823)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 18), dict_9805, (str_9821, list_9822))
# Adding element type (key, value) (line 58)
str_9824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 19), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 64)
list_9825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 64)
# Adding element type (line 64)
str_9826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 37), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 36), list_9825, str_9826)
# Adding element type (line 64)
str_9827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 43), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 36), list_9825, str_9827)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 18), dict_9805, (str_9824, list_9825))
# Adding element type (key, value) (line 58)
str_9828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'str', 'ranlib')
# Getting the type of 'None' (line 65)
None_9829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 36), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 18), dict_9805, (str_9828, None_9829))

# Getting the type of 'UnixCCompiler'
UnixCCompiler_9830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnixCCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnixCCompiler_9830, 'executables', dict_9805)

# Assigning a Dict to a Name (line 58):



# Obtaining the type of the subscript
int_9831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 21), 'int')
slice_9832 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 68, 7), None, int_9831, None)
# Getting the type of 'sys' (line 68)
sys_9833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 7), 'sys')
# Obtaining the member 'platform' of a type (line 68)
platform_9834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 7), sys_9833, 'platform')
# Obtaining the member '__getitem__' of a type (line 68)
getitem___9835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 7), platform_9834, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 68)
subscript_call_result_9836 = invoke(stypy.reporting.localization.Localization(__file__, 68, 7), getitem___9835, slice_9832)

str_9837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 27), 'str', 'darwin')
# Applying the binary operator '==' (line 68)
result_eq_9838 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 7), '==', subscript_call_result_9836, str_9837)

# Testing the type of an if condition (line 68)
if_condition_9839 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 4), result_eq_9838)
# Assigning a type to the variable 'if_condition_9839' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'if_condition_9839', if_condition_9839)
# SSA begins for if statement (line 68)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a List to a Subscript (line 69):

# Assigning a List to a Subscript (line 69):

# Obtaining an instance of the builtin type 'list' (line 69)
list_9840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 69)
# Adding element type (line 69)
str_9841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 33), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 32), list_9840, str_9841)

# Getting the type of 'UnixCCompiler'
UnixCCompiler_9842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnixCCompiler')
# Obtaining the member 'executables' of a type
executables_9843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnixCCompiler_9842, 'executables')
str_9844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 20), 'str', 'ranlib')
# Storing an element on a container (line 69)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 8), executables_9843, (str_9844, list_9840))
# SSA join for if statement (line 68)
module_type_store = module_type_store.join_ssa_context()


# Assigning a List to a Name (line 77):

# Obtaining an instance of the builtin type 'list' (line 77)
list_9845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 21), 'list')
# Adding type elements to the builtin type 'list' instance (line 77)
# Adding element type (line 77)
str_9846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 22), 'str', '.c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 21), list_9845, str_9846)
# Adding element type (line 77)
str_9847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 27), 'str', '.C')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 21), list_9845, str_9847)
# Adding element type (line 77)
str_9848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 32), 'str', '.cc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 21), list_9845, str_9848)
# Adding element type (line 77)
str_9849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 38), 'str', '.cxx')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 21), list_9845, str_9849)
# Adding element type (line 77)
str_9850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 45), 'str', '.cpp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 21), list_9845, str_9850)
# Adding element type (line 77)
str_9851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 52), 'str', '.m')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 21), list_9845, str_9851)

# Getting the type of 'UnixCCompiler'
UnixCCompiler_9852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnixCCompiler')
# Setting the type of the member 'src_extensions' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnixCCompiler_9852, 'src_extensions', list_9845)

# Assigning a Str to a Name (line 78):
str_9853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 20), 'str', '.o')
# Getting the type of 'UnixCCompiler'
UnixCCompiler_9854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnixCCompiler')
# Setting the type of the member 'obj_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnixCCompiler_9854, 'obj_extension', str_9853)

# Assigning a Str to a Name (line 79):
str_9855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 27), 'str', '.a')
# Getting the type of 'UnixCCompiler'
UnixCCompiler_9856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnixCCompiler')
# Setting the type of the member 'static_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnixCCompiler_9856, 'static_lib_extension', str_9855)

# Assigning a Str to a Name (line 80):
str_9857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 27), 'str', '.so')
# Getting the type of 'UnixCCompiler'
UnixCCompiler_9858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnixCCompiler')
# Setting the type of the member 'shared_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnixCCompiler_9858, 'shared_lib_extension', str_9857)

# Assigning a Str to a Name (line 81):
str_9859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 26), 'str', '.dylib')
# Getting the type of 'UnixCCompiler'
UnixCCompiler_9860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnixCCompiler')
# Setting the type of the member 'dylib_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnixCCompiler_9860, 'dylib_lib_extension', str_9859)

# Assigning a Str to a Name (line 82):
str_9861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 31), 'str', '.tbd')
# Getting the type of 'UnixCCompiler'
UnixCCompiler_9862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnixCCompiler')
# Setting the type of the member 'xcode_stub_lib_extension' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnixCCompiler_9862, 'xcode_stub_lib_extension', str_9861)

# Assigning a Str to a Name (line 83):
str_9863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 63), 'str', 'lib%s%s')
# Getting the type of 'UnixCCompiler'
UnixCCompiler_9864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnixCCompiler')
# Setting the type of the member 'dylib_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnixCCompiler_9864, 'dylib_lib_format', str_9863)

# Assigning a Name to a Name (line 83):
# Getting the type of 'UnixCCompiler'
UnixCCompiler_9865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnixCCompiler')
# Obtaining the member 'dylib_lib_format' of a type
dylib_lib_format_9866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnixCCompiler_9865, 'dylib_lib_format')
# Getting the type of 'UnixCCompiler'
UnixCCompiler_9867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnixCCompiler')
# Setting the type of the member 'shared_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnixCCompiler_9867, 'shared_lib_format', dylib_lib_format_9866)

# Assigning a Name to a Name (line 83):
# Getting the type of 'UnixCCompiler'
UnixCCompiler_9868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnixCCompiler')
# Obtaining the member 'shared_lib_format' of a type
shared_lib_format_9869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnixCCompiler_9868, 'shared_lib_format')
# Getting the type of 'UnixCCompiler'
UnixCCompiler_9870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnixCCompiler')
# Setting the type of the member 'static_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnixCCompiler_9870, 'static_lib_format', shared_lib_format_9869)

# Assigning a Name to a Name (line 84):
# Getting the type of 'UnixCCompiler'
UnixCCompiler_9871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnixCCompiler')
# Obtaining the member 'dylib_lib_format' of a type
dylib_lib_format_9872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnixCCompiler_9871, 'dylib_lib_format')
# Getting the type of 'UnixCCompiler'
UnixCCompiler_9873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'UnixCCompiler')
# Setting the type of the member 'xcode_stub_lib_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), UnixCCompiler_9873, 'xcode_stub_lib_format', dylib_lib_format_9872)

# Assigning a Name to a Name (line 84):


# Getting the type of 'sys' (line 85)
sys_9874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 7), 'sys')
# Obtaining the member 'platform' of a type (line 85)
platform_9875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 7), sys_9874, 'platform')
str_9876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 23), 'str', 'cygwin')
# Applying the binary operator '==' (line 85)
result_eq_9877 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 7), '==', platform_9875, str_9876)

# Testing the type of an if condition (line 85)
if_condition_9878 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 4), result_eq_9877)
# Assigning a type to the variable 'if_condition_9878' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'if_condition_9878', if_condition_9878)
# SSA begins for if statement (line 85)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Str to a Name (line 86):

# Assigning a Str to a Name (line 86):
str_9879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 24), 'str', '.exe')
# Assigning a type to the variable 'exe_extension' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'exe_extension', str_9879)
# SSA join for if statement (line 85)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
