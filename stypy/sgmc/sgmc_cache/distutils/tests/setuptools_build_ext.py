
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from distutils.command.build_ext import build_ext as _du_build_ext
2: try:
3:     # Attempt to use Pyrex for building extensions, if available
4:     from Pyrex.Distutils.build_ext import build_ext as _build_ext
5: except ImportError:
6:     _build_ext = _du_build_ext
7: 
8: import os, sys
9: from distutils.file_util import copy_file
10: 
11: from distutils.tests.setuptools_extension import Library
12: 
13: from distutils.ccompiler import new_compiler
14: from distutils.sysconfig import customize_compiler, get_config_var
15: get_config_var("LDSHARED")  # make sure _config_vars is initialized
16: from distutils.sysconfig import _config_vars
17: from distutils import log
18: from distutils.errors import *
19: 
20: have_rtld = False
21: use_stubs = False
22: libtype = 'shared'
23: 
24: if sys.platform == "darwin":
25:     use_stubs = True
26: elif os.name != 'nt':
27:     try:
28:         from dl import RTLD_NOW
29:         have_rtld = True
30:         use_stubs = True
31:     except ImportError:
32:         pass
33: 
34: def if_dl(s):
35:     if have_rtld:
36:         return s
37:     return ''
38: 
39: 
40: 
41: 
42: 
43: 
44: class build_ext(_build_ext):
45:     def run(self):
46:         '''Build extensions in build directory, then copy if --inplace'''
47:         old_inplace, self.inplace = self.inplace, 0
48:         _build_ext.run(self)
49:         self.inplace = old_inplace
50:         if old_inplace:
51:             self.copy_extensions_to_source()
52: 
53:     def copy_extensions_to_source(self):
54:         build_py = self.get_finalized_command('build_py')
55:         for ext in self.extensions:
56:             fullname = self.get_ext_fullname(ext.name)
57:             filename = self.get_ext_filename(fullname)
58:             modpath = fullname.split('.')
59:             package = '.'.join(modpath[:-1])
60:             package_dir = build_py.get_package_dir(package)
61:             dest_filename = os.path.join(package_dir,os.path.basename(filename))
62:             src_filename = os.path.join(self.build_lib,filename)
63: 
64:             # Always copy, even if source is older than destination, to ensure
65:             # that the right extensions for the current Python/platform are
66:             # used.
67:             copy_file(
68:                 src_filename, dest_filename, verbose=self.verbose,
69:                 dry_run=self.dry_run
70:             )
71:             if ext._needs_stub:
72:                 self.write_stub(package_dir or os.curdir, ext, True)
73: 
74: 
75:     if _build_ext is not _du_build_ext and not hasattr(_build_ext,'pyrex_sources'):
76:         # Workaround for problems using some Pyrex versions w/SWIG and/or 2.4
77:         def swig_sources(self, sources, *otherargs):
78:             # first do any Pyrex processing
79:             sources = _build_ext.swig_sources(self, sources) or sources
80:             # Then do any actual SWIG stuff on the remainder
81:             return _du_build_ext.swig_sources(self, sources, *otherargs)
82: 
83: 
84: 
85:     def get_ext_filename(self, fullname):
86:         filename = _build_ext.get_ext_filename(self,fullname)
87:         ext = self.ext_map[fullname]
88:         if isinstance(ext,Library):
89:             fn, ext = os.path.splitext(filename)
90:             return self.shlib_compiler.library_filename(fn,libtype)
91:         elif use_stubs and ext._links_to_dynamic:
92:             d,fn = os.path.split(filename)
93:             return os.path.join(d,'dl-'+fn)
94:         else:
95:             return filename
96: 
97:     def initialize_options(self):
98:         _build_ext.initialize_options(self)
99:         self.shlib_compiler = None
100:         self.shlibs = []
101:         self.ext_map = {}
102: 
103:     def finalize_options(self):
104:         _build_ext.finalize_options(self)
105:         self.extensions = self.extensions or []
106:         self.check_extensions_list(self.extensions)
107:         self.shlibs = [ext for ext in self.extensions
108:                         if isinstance(ext,Library)]
109:         if self.shlibs:
110:             self.setup_shlib_compiler()
111:         for ext in self.extensions:
112:             ext._full_name = self.get_ext_fullname(ext.name)
113:         for ext in self.extensions:
114:             fullname = ext._full_name
115:             self.ext_map[fullname] = ext
116:             ltd = ext._links_to_dynamic = \
117:                 self.shlibs and self.links_to_dynamic(ext) or False
118:             ext._needs_stub = ltd and use_stubs and not isinstance(ext,Library)
119:             filename = ext._file_name = self.get_ext_filename(fullname)
120:             libdir = os.path.dirname(os.path.join(self.build_lib,filename))
121:             if ltd and libdir not in ext.library_dirs:
122:                 ext.library_dirs.append(libdir)
123:             if ltd and use_stubs and os.curdir not in ext.runtime_library_dirs:
124:                 ext.runtime_library_dirs.append(os.curdir)
125: 
126:     def setup_shlib_compiler(self):
127:         compiler = self.shlib_compiler = new_compiler(
128:             compiler=self.compiler, dry_run=self.dry_run, force=self.force
129:         )
130:         if sys.platform == "darwin":
131:             tmp = _config_vars.copy()
132:             try:
133:                 # XXX Help!  I don't have any idea whether these are right...
134:                 _config_vars['LDSHARED'] = "gcc -Wl,-x -dynamiclib -undefined dynamic_lookup"
135:                 _config_vars['CCSHARED'] = " -dynamiclib"
136:                 _config_vars['SO'] = ".dylib"
137:                 customize_compiler(compiler)
138:             finally:
139:                 _config_vars.clear()
140:                 _config_vars.update(tmp)
141:         else:
142:             customize_compiler(compiler)
143: 
144:         if self.include_dirs is not None:
145:             compiler.set_include_dirs(self.include_dirs)
146:         if self.define is not None:
147:             # 'define' option is a list of (name,value) tuples
148:             for (name,value) in self.define:
149:                 compiler.define_macro(name, value)
150:         if self.undef is not None:
151:             for macro in self.undef:
152:                 compiler.undefine_macro(macro)
153:         if self.libraries is not None:
154:             compiler.set_libraries(self.libraries)
155:         if self.library_dirs is not None:
156:             compiler.set_library_dirs(self.library_dirs)
157:         if self.rpath is not None:
158:             compiler.set_runtime_library_dirs(self.rpath)
159:         if self.link_objects is not None:
160:             compiler.set_link_objects(self.link_objects)
161: 
162:         # hack so distutils' build_extension() builds a library instead
163:         compiler.link_shared_object = link_shared_object.__get__(compiler)
164: 
165: 
166: 
167:     def get_export_symbols(self, ext):
168:         if isinstance(ext,Library):
169:             return ext.export_symbols
170:         return _build_ext.get_export_symbols(self,ext)
171: 
172:     def build_extension(self, ext):
173:         _compiler = self.compiler
174:         try:
175:             if isinstance(ext,Library):
176:                 self.compiler = self.shlib_compiler
177:             _build_ext.build_extension(self,ext)
178:             if ext._needs_stub:
179:                 self.write_stub(
180:                     self.get_finalized_command('build_py').build_lib, ext
181:                 )
182:         finally:
183:             self.compiler = _compiler
184: 
185:     def links_to_dynamic(self, ext):
186:         '''Return true if 'ext' links to a dynamic lib in the same package'''
187:         # XXX this should check to ensure the lib is actually being built
188:         # XXX as dynamic, and not just using a locally-found version or a
189:         # XXX static-compiled version
190:         libnames = dict.fromkeys([lib._full_name for lib in self.shlibs])
191:         pkg = '.'.join(ext._full_name.split('.')[:-1]+[''])
192:         for libname in ext.libraries:
193:             if pkg+libname in libnames: return True
194:         return False
195: 
196:     def get_outputs(self):
197:         outputs = _build_ext.get_outputs(self)
198:         optimize = self.get_finalized_command('build_py').optimize
199:         for ext in self.extensions:
200:             if ext._needs_stub:
201:                 base = os.path.join(self.build_lib, *ext._full_name.split('.'))
202:                 outputs.append(base+'.py')
203:                 outputs.append(base+'.pyc')
204:                 if optimize:
205:                     outputs.append(base+'.pyo')
206:         return outputs
207: 
208:     def write_stub(self, output_dir, ext, compile=False):
209:         log.info("writing stub loader for %s to %s",ext._full_name, output_dir)
210:         stub_file = os.path.join(output_dir, *ext._full_name.split('.'))+'.py'
211:         if compile and os.path.exists(stub_file):
212:             raise DistutilsError(stub_file+" already exists! Please delete.")
213:         if not self.dry_run:
214:             f = open(stub_file,'w')
215:             f.write('\n'.join([
216:                 "def __bootstrap__():",
217:                 "   global __bootstrap__, __file__, __loader__",
218:                 "   import sys, os, pkg_resources, imp"+if_dl(", dl"),
219:                 "   __file__ = pkg_resources.resource_filename(__name__,%r)"
220:                    % os.path.basename(ext._file_name),
221:                 "   del __bootstrap__",
222:                 "   if '__loader__' in globals():",
223:                 "       del __loader__",
224:                 if_dl("   old_flags = sys.getdlopenflags()"),
225:                 "   old_dir = os.getcwd()",
226:                 "   try:",
227:                 "     os.chdir(os.path.dirname(__file__))",
228:                 if_dl("     sys.setdlopenflags(dl.RTLD_NOW)"),
229:                 "     imp.load_dynamic(__name__,__file__)",
230:                 "   finally:",
231:                 if_dl("     sys.setdlopenflags(old_flags)"),
232:                 "     os.chdir(old_dir)",
233:                 "__bootstrap__()",
234:                 "" # terminal \n
235:             ]))
236:             f.close()
237:         if compile:
238:             from distutils.util import byte_compile
239:             byte_compile([stub_file], optimize=0,
240:                          force=True, dry_run=self.dry_run)
241:             optimize = self.get_finalized_command('install_lib').optimize
242:             if optimize > 0:
243:                 byte_compile([stub_file], optimize=optimize,
244:                              force=True, dry_run=self.dry_run)
245:             if os.path.exists(stub_file) and not self.dry_run:
246:                 os.unlink(stub_file)
247: 
248: 
249: if use_stubs or os.name=='nt':
250:     # Build shared libraries
251:     #
252:     def link_shared_object(self, objects, output_libname, output_dir=None,
253:         libraries=None, library_dirs=None, runtime_library_dirs=None,
254:         export_symbols=None, debug=0, extra_preargs=None,
255:         extra_postargs=None, build_temp=None, target_lang=None
256:     ):  self.link(
257:             self.SHARED_LIBRARY, objects, output_libname,
258:             output_dir, libraries, library_dirs, runtime_library_dirs,
259:             export_symbols, debug, extra_preargs, extra_postargs,
260:             build_temp, target_lang
261:         )
262: else:
263:     # Build static libraries everywhere else
264:     libtype = 'static'
265: 
266:     def link_shared_object(self, objects, output_libname, output_dir=None,
267:         libraries=None, library_dirs=None, runtime_library_dirs=None,
268:         export_symbols=None, debug=0, extra_preargs=None,
269:         extra_postargs=None, build_temp=None, target_lang=None
270:     ):
271:         # XXX we need to either disallow these attrs on Library instances,
272:         #     or warn/abort here if set, or something...
273:         #libraries=None, library_dirs=None, runtime_library_dirs=None,
274:         #export_symbols=None, extra_preargs=None, extra_postargs=None,
275:         #build_temp=None
276: 
277:         assert output_dir is None   # distutils build_ext doesn't pass this
278:         output_dir,filename = os.path.split(output_libname)
279:         basename, ext = os.path.splitext(filename)
280:         if self.library_filename("x").startswith('lib'):
281:             # strip 'lib' prefix; this is kludgy if some platform uses
282:             # a different prefix
283:             basename = basename[3:]
284: 
285:         self.create_static_lib(
286:             objects, basename, output_dir, debug, target_lang
287:         )
288: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from distutils.command.build_ext import _du_build_ext' statement (line 1)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_27349 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'distutils.command.build_ext')

if (type(import_27349) is not StypyTypeError):

    if (import_27349 != 'pyd_module'):
        __import__(import_27349)
        sys_modules_27350 = sys.modules[import_27349]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'distutils.command.build_ext', sys_modules_27350.module_type_store, module_type_store, ['build_ext'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_27350, sys_modules_27350.module_type_store, module_type_store)
    else:
        from distutils.command.build_ext import build_ext as _du_build_ext

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'distutils.command.build_ext', None, module_type_store, ['build_ext'], [_du_build_ext])

else:
    # Assigning a type to the variable 'distutils.command.build_ext' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'distutils.command.build_ext', import_27349)

# Adding an alias
module_type_store.add_alias('_du_build_ext', 'build_ext')
remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')



# SSA begins for try-except statement (line 2)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 4))

# 'from Pyrex.Distutils.build_ext import _build_ext' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_27351 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 4), 'Pyrex.Distutils.build_ext')

if (type(import_27351) is not StypyTypeError):

    if (import_27351 != 'pyd_module'):
        __import__(import_27351)
        sys_modules_27352 = sys.modules[import_27351]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 4), 'Pyrex.Distutils.build_ext', sys_modules_27352.module_type_store, module_type_store, ['build_ext'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 4), __file__, sys_modules_27352, sys_modules_27352.module_type_store, module_type_store)
    else:
        from Pyrex.Distutils.build_ext import build_ext as _build_ext

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 4), 'Pyrex.Distutils.build_ext', None, module_type_store, ['build_ext'], [_build_ext])

else:
    # Assigning a type to the variable 'Pyrex.Distutils.build_ext' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'Pyrex.Distutils.build_ext', import_27351)

# Adding an alias
module_type_store.add_alias('_build_ext', 'build_ext')
remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# SSA branch for the except part of a try statement (line 2)
# SSA branch for the except 'ImportError' branch of a try statement (line 2)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 6):

# Assigning a Name to a Name (line 6):
# Getting the type of '_du_build_ext' (line 6)
_du_build_ext_27353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 17), '_du_build_ext')
# Assigning a type to the variable '_build_ext' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), '_build_ext', _du_build_ext_27353)
# SSA join for try-except statement (line 2)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# Multiple import statement. import os (1/2) (line 8)
import os

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os', os, module_type_store)
# Multiple import statement. import sys (2/2) (line 8)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.file_util import copy_file' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_27354 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.file_util')

if (type(import_27354) is not StypyTypeError):

    if (import_27354 != 'pyd_module'):
        __import__(import_27354)
        sys_modules_27355 = sys.modules[import_27354]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.file_util', sys_modules_27355.module_type_store, module_type_store, ['copy_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_27355, sys_modules_27355.module_type_store, module_type_store)
    else:
        from distutils.file_util import copy_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.file_util', None, module_type_store, ['copy_file'], [copy_file])

else:
    # Assigning a type to the variable 'distutils.file_util' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.file_util', import_27354)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.tests.setuptools_extension import Library' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_27356 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.tests.setuptools_extension')

if (type(import_27356) is not StypyTypeError):

    if (import_27356 != 'pyd_module'):
        __import__(import_27356)
        sys_modules_27357 = sys.modules[import_27356]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.tests.setuptools_extension', sys_modules_27357.module_type_store, module_type_store, ['Library'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_27357, sys_modules_27357.module_type_store, module_type_store)
    else:
        from distutils.tests.setuptools_extension import Library

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.tests.setuptools_extension', None, module_type_store, ['Library'], [Library])

else:
    # Assigning a type to the variable 'distutils.tests.setuptools_extension' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.tests.setuptools_extension', import_27356)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.ccompiler import new_compiler' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_27358 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.ccompiler')

if (type(import_27358) is not StypyTypeError):

    if (import_27358 != 'pyd_module'):
        __import__(import_27358)
        sys_modules_27359 = sys.modules[import_27358]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.ccompiler', sys_modules_27359.module_type_store, module_type_store, ['new_compiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_27359, sys_modules_27359.module_type_store, module_type_store)
    else:
        from distutils.ccompiler import new_compiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.ccompiler', None, module_type_store, ['new_compiler'], [new_compiler])

else:
    # Assigning a type to the variable 'distutils.ccompiler' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.ccompiler', import_27358)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.sysconfig import customize_compiler, get_config_var' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_27360 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.sysconfig')

if (type(import_27360) is not StypyTypeError):

    if (import_27360 != 'pyd_module'):
        __import__(import_27360)
        sys_modules_27361 = sys.modules[import_27360]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.sysconfig', sys_modules_27361.module_type_store, module_type_store, ['customize_compiler', 'get_config_var'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_27361, sys_modules_27361.module_type_store, module_type_store)
    else:
        from distutils.sysconfig import customize_compiler, get_config_var

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.sysconfig', None, module_type_store, ['customize_compiler', 'get_config_var'], [customize_compiler, get_config_var])

else:
    # Assigning a type to the variable 'distutils.sysconfig' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.sysconfig', import_27360)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


# Call to get_config_var(...): (line 15)
# Processing the call arguments (line 15)
str_27363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'str', 'LDSHARED')
# Processing the call keyword arguments (line 15)
kwargs_27364 = {}
# Getting the type of 'get_config_var' (line 15)
get_config_var_27362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'get_config_var', False)
# Calling get_config_var(args, kwargs) (line 15)
get_config_var_call_result_27365 = invoke(stypy.reporting.localization.Localization(__file__, 15, 0), get_config_var_27362, *[str_27363], **kwargs_27364)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils.sysconfig import _config_vars' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_27366 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.sysconfig')

if (type(import_27366) is not StypyTypeError):

    if (import_27366 != 'pyd_module'):
        __import__(import_27366)
        sys_modules_27367 = sys.modules[import_27366]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.sysconfig', sys_modules_27367.module_type_store, module_type_store, ['_config_vars'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_27367, sys_modules_27367.module_type_store, module_type_store)
    else:
        from distutils.sysconfig import _config_vars

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.sysconfig', None, module_type_store, ['_config_vars'], [_config_vars])

else:
    # Assigning a type to the variable 'distutils.sysconfig' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.sysconfig', import_27366)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from distutils import log' statement (line 17)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils', None, module_type_store, ['log'], [log])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from distutils.errors import ' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_27368 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.errors')

if (type(import_27368) is not StypyTypeError):

    if (import_27368 != 'pyd_module'):
        __import__(import_27368)
        sys_modules_27369 = sys.modules[import_27368]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.errors', sys_modules_27369.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_27369, sys_modules_27369.module_type_store, module_type_store)
    else:
        from distutils.errors import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.errors', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'distutils.errors' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.errors', import_27368)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


# Assigning a Name to a Name (line 20):

# Assigning a Name to a Name (line 20):
# Getting the type of 'False' (line 20)
False_27370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'False')
# Assigning a type to the variable 'have_rtld' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'have_rtld', False_27370)

# Assigning a Name to a Name (line 21):

# Assigning a Name to a Name (line 21):
# Getting the type of 'False' (line 21)
False_27371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'False')
# Assigning a type to the variable 'use_stubs' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'use_stubs', False_27371)

# Assigning a Str to a Name (line 22):

# Assigning a Str to a Name (line 22):
str_27372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 10), 'str', 'shared')
# Assigning a type to the variable 'libtype' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'libtype', str_27372)


# Getting the type of 'sys' (line 24)
sys_27373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 3), 'sys')
# Obtaining the member 'platform' of a type (line 24)
platform_27374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 3), sys_27373, 'platform')
str_27375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'str', 'darwin')
# Applying the binary operator '==' (line 24)
result_eq_27376 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 3), '==', platform_27374, str_27375)

# Testing the type of an if condition (line 24)
if_condition_27377 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 0), result_eq_27376)
# Assigning a type to the variable 'if_condition_27377' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'if_condition_27377', if_condition_27377)
# SSA begins for if statement (line 24)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 25):

# Assigning a Name to a Name (line 25):
# Getting the type of 'True' (line 25)
True_27378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'True')
# Assigning a type to the variable 'use_stubs' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'use_stubs', True_27378)
# SSA branch for the else part of an if statement (line 24)
module_type_store.open_ssa_branch('else')


# Getting the type of 'os' (line 26)
os_27379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 5), 'os')
# Obtaining the member 'name' of a type (line 26)
name_27380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 5), os_27379, 'name')
str_27381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'str', 'nt')
# Applying the binary operator '!=' (line 26)
result_ne_27382 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 5), '!=', name_27380, str_27381)

# Testing the type of an if condition (line 26)
if_condition_27383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 5), result_ne_27382)
# Assigning a type to the variable 'if_condition_27383' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 5), 'if_condition_27383', if_condition_27383)
# SSA begins for if statement (line 26)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')


# SSA begins for try-except statement (line 27)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 8))

# 'from dl import RTLD_NOW' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_27384 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 8), 'dl')

if (type(import_27384) is not StypyTypeError):

    if (import_27384 != 'pyd_module'):
        __import__(import_27384)
        sys_modules_27385 = sys.modules[import_27384]
        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 8), 'dl', sys_modules_27385.module_type_store, module_type_store, ['RTLD_NOW'])
        nest_module(stypy.reporting.localization.Localization(__file__, 28, 8), __file__, sys_modules_27385, sys_modules_27385.module_type_store, module_type_store)
    else:
        from dl import RTLD_NOW

        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 8), 'dl', None, module_type_store, ['RTLD_NOW'], [RTLD_NOW])

else:
    # Assigning a type to the variable 'dl' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'dl', import_27384)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


# Assigning a Name to a Name (line 29):

# Assigning a Name to a Name (line 29):
# Getting the type of 'True' (line 29)
True_27386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 20), 'True')
# Assigning a type to the variable 'have_rtld' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'have_rtld', True_27386)

# Assigning a Name to a Name (line 30):

# Assigning a Name to a Name (line 30):
# Getting the type of 'True' (line 30)
True_27387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 20), 'True')
# Assigning a type to the variable 'use_stubs' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'use_stubs', True_27387)
# SSA branch for the except part of a try statement (line 27)
# SSA branch for the except 'ImportError' branch of a try statement (line 27)
module_type_store.open_ssa_branch('except')
pass
# SSA join for try-except statement (line 27)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 26)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 24)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def if_dl(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'if_dl'
    module_type_store = module_type_store.open_function_context('if_dl', 34, 0, False)
    
    # Passed parameters checking function
    if_dl.stypy_localization = localization
    if_dl.stypy_type_of_self = None
    if_dl.stypy_type_store = module_type_store
    if_dl.stypy_function_name = 'if_dl'
    if_dl.stypy_param_names_list = ['s']
    if_dl.stypy_varargs_param_name = None
    if_dl.stypy_kwargs_param_name = None
    if_dl.stypy_call_defaults = defaults
    if_dl.stypy_call_varargs = varargs
    if_dl.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'if_dl', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'if_dl', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'if_dl(...)' code ##################

    
    # Getting the type of 'have_rtld' (line 35)
    have_rtld_27388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 7), 'have_rtld')
    # Testing the type of an if condition (line 35)
    if_condition_27389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 4), have_rtld_27388)
    # Assigning a type to the variable 'if_condition_27389' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'if_condition_27389', if_condition_27389)
    # SSA begins for if statement (line 35)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 's' (line 36)
    s_27390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 's')
    # Assigning a type to the variable 'stypy_return_type' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type', s_27390)
    # SSA join for if statement (line 35)
    module_type_store = module_type_store.join_ssa_context()
    
    str_27391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 11), 'str', '')
    # Assigning a type to the variable 'stypy_return_type' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type', str_27391)
    
    # ################# End of 'if_dl(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'if_dl' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_27392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_27392)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'if_dl'
    return stypy_return_type_27392

# Assigning a type to the variable 'if_dl' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'if_dl', if_dl)
# Declaration of the 'build_ext' class
# Getting the type of '_build_ext' (line 44)
_build_ext_27393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), '_build_ext')

class build_ext(_build_ext_27393, ):

    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
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

        str_27394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 8), 'str', 'Build extensions in build directory, then copy if --inplace')
        
        # Assigning a Tuple to a Tuple (line 47):
        
        # Assigning a Attribute to a Name (line 47):
        # Getting the type of 'self' (line 47)
        self_27395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 36), 'self')
        # Obtaining the member 'inplace' of a type (line 47)
        inplace_27396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 36), self_27395, 'inplace')
        # Assigning a type to the variable 'tuple_assignment_27339' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'tuple_assignment_27339', inplace_27396)
        
        # Assigning a Num to a Name (line 47):
        int_27397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 50), 'int')
        # Assigning a type to the variable 'tuple_assignment_27340' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'tuple_assignment_27340', int_27397)
        
        # Assigning a Name to a Name (line 47):
        # Getting the type of 'tuple_assignment_27339' (line 47)
        tuple_assignment_27339_27398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'tuple_assignment_27339')
        # Assigning a type to the variable 'old_inplace' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'old_inplace', tuple_assignment_27339_27398)
        
        # Assigning a Name to a Attribute (line 47):
        # Getting the type of 'tuple_assignment_27340' (line 47)
        tuple_assignment_27340_27399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'tuple_assignment_27340')
        # Getting the type of 'self' (line 47)
        self_27400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 21), 'self')
        # Setting the type of the member 'inplace' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 21), self_27400, 'inplace', tuple_assignment_27340_27399)
        
        # Call to run(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'self' (line 48)
        self_27403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'self', False)
        # Processing the call keyword arguments (line 48)
        kwargs_27404 = {}
        # Getting the type of '_build_ext' (line 48)
        _build_ext_27401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), '_build_ext', False)
        # Obtaining the member 'run' of a type (line 48)
        run_27402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), _build_ext_27401, 'run')
        # Calling run(args, kwargs) (line 48)
        run_call_result_27405 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), run_27402, *[self_27403], **kwargs_27404)
        
        
        # Assigning a Name to a Attribute (line 49):
        
        # Assigning a Name to a Attribute (line 49):
        # Getting the type of 'old_inplace' (line 49)
        old_inplace_27406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 23), 'old_inplace')
        # Getting the type of 'self' (line 49)
        self_27407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'self')
        # Setting the type of the member 'inplace' of a type (line 49)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), self_27407, 'inplace', old_inplace_27406)
        
        # Getting the type of 'old_inplace' (line 50)
        old_inplace_27408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'old_inplace')
        # Testing the type of an if condition (line 50)
        if_condition_27409 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 8), old_inplace_27408)
        # Assigning a type to the variable 'if_condition_27409' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'if_condition_27409', if_condition_27409)
        # SSA begins for if statement (line 50)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to copy_extensions_to_source(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_27412 = {}
        # Getting the type of 'self' (line 51)
        self_27410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self', False)
        # Obtaining the member 'copy_extensions_to_source' of a type (line 51)
        copy_extensions_to_source_27411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_27410, 'copy_extensions_to_source')
        # Calling copy_extensions_to_source(args, kwargs) (line 51)
        copy_extensions_to_source_call_result_27413 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), copy_extensions_to_source_27411, *[], **kwargs_27412)
        
        # SSA join for if statement (line 50)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_27414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27414)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_27414


    @norecursion
    def copy_extensions_to_source(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy_extensions_to_source'
        module_type_store = module_type_store.open_function_context('copy_extensions_to_source', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.copy_extensions_to_source.__dict__.__setitem__('stypy_localization', localization)
        build_ext.copy_extensions_to_source.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.copy_extensions_to_source.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.copy_extensions_to_source.__dict__.__setitem__('stypy_function_name', 'build_ext.copy_extensions_to_source')
        build_ext.copy_extensions_to_source.__dict__.__setitem__('stypy_param_names_list', [])
        build_ext.copy_extensions_to_source.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.copy_extensions_to_source.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.copy_extensions_to_source.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.copy_extensions_to_source.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.copy_extensions_to_source.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.copy_extensions_to_source.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.copy_extensions_to_source', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy_extensions_to_source', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy_extensions_to_source(...)' code ##################

        
        # Assigning a Call to a Name (line 54):
        
        # Assigning a Call to a Name (line 54):
        
        # Call to get_finalized_command(...): (line 54)
        # Processing the call arguments (line 54)
        str_27417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 46), 'str', 'build_py')
        # Processing the call keyword arguments (line 54)
        kwargs_27418 = {}
        # Getting the type of 'self' (line 54)
        self_27415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 54)
        get_finalized_command_27416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 19), self_27415, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 54)
        get_finalized_command_call_result_27419 = invoke(stypy.reporting.localization.Localization(__file__, 54, 19), get_finalized_command_27416, *[str_27417], **kwargs_27418)
        
        # Assigning a type to the variable 'build_py' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'build_py', get_finalized_command_call_result_27419)
        
        # Getting the type of 'self' (line 55)
        self_27420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'self')
        # Obtaining the member 'extensions' of a type (line 55)
        extensions_27421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 19), self_27420, 'extensions')
        # Testing the type of a for loop iterable (line 55)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 55, 8), extensions_27421)
        # Getting the type of the for loop variable (line 55)
        for_loop_var_27422 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 55, 8), extensions_27421)
        # Assigning a type to the variable 'ext' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'ext', for_loop_var_27422)
        # SSA begins for a for statement (line 55)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to get_ext_fullname(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'ext' (line 56)
        ext_27425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 45), 'ext', False)
        # Obtaining the member 'name' of a type (line 56)
        name_27426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 45), ext_27425, 'name')
        # Processing the call keyword arguments (line 56)
        kwargs_27427 = {}
        # Getting the type of 'self' (line 56)
        self_27423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 23), 'self', False)
        # Obtaining the member 'get_ext_fullname' of a type (line 56)
        get_ext_fullname_27424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 23), self_27423, 'get_ext_fullname')
        # Calling get_ext_fullname(args, kwargs) (line 56)
        get_ext_fullname_call_result_27428 = invoke(stypy.reporting.localization.Localization(__file__, 56, 23), get_ext_fullname_27424, *[name_27426], **kwargs_27427)
        
        # Assigning a type to the variable 'fullname' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'fullname', get_ext_fullname_call_result_27428)
        
        # Assigning a Call to a Name (line 57):
        
        # Assigning a Call to a Name (line 57):
        
        # Call to get_ext_filename(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'fullname' (line 57)
        fullname_27431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 45), 'fullname', False)
        # Processing the call keyword arguments (line 57)
        kwargs_27432 = {}
        # Getting the type of 'self' (line 57)
        self_27429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'self', False)
        # Obtaining the member 'get_ext_filename' of a type (line 57)
        get_ext_filename_27430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 23), self_27429, 'get_ext_filename')
        # Calling get_ext_filename(args, kwargs) (line 57)
        get_ext_filename_call_result_27433 = invoke(stypy.reporting.localization.Localization(__file__, 57, 23), get_ext_filename_27430, *[fullname_27431], **kwargs_27432)
        
        # Assigning a type to the variable 'filename' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'filename', get_ext_filename_call_result_27433)
        
        # Assigning a Call to a Name (line 58):
        
        # Assigning a Call to a Name (line 58):
        
        # Call to split(...): (line 58)
        # Processing the call arguments (line 58)
        str_27436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 37), 'str', '.')
        # Processing the call keyword arguments (line 58)
        kwargs_27437 = {}
        # Getting the type of 'fullname' (line 58)
        fullname_27434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 22), 'fullname', False)
        # Obtaining the member 'split' of a type (line 58)
        split_27435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 22), fullname_27434, 'split')
        # Calling split(args, kwargs) (line 58)
        split_call_result_27438 = invoke(stypy.reporting.localization.Localization(__file__, 58, 22), split_27435, *[str_27436], **kwargs_27437)
        
        # Assigning a type to the variable 'modpath' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'modpath', split_call_result_27438)
        
        # Assigning a Call to a Name (line 59):
        
        # Assigning a Call to a Name (line 59):
        
        # Call to join(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Obtaining the type of the subscript
        int_27441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 40), 'int')
        slice_27442 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 59, 31), None, int_27441, None)
        # Getting the type of 'modpath' (line 59)
        modpath_27443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 31), 'modpath', False)
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___27444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 31), modpath_27443, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_27445 = invoke(stypy.reporting.localization.Localization(__file__, 59, 31), getitem___27444, slice_27442)
        
        # Processing the call keyword arguments (line 59)
        kwargs_27446 = {}
        str_27439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 22), 'str', '.')
        # Obtaining the member 'join' of a type (line 59)
        join_27440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 22), str_27439, 'join')
        # Calling join(args, kwargs) (line 59)
        join_call_result_27447 = invoke(stypy.reporting.localization.Localization(__file__, 59, 22), join_27440, *[subscript_call_result_27445], **kwargs_27446)
        
        # Assigning a type to the variable 'package' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'package', join_call_result_27447)
        
        # Assigning a Call to a Name (line 60):
        
        # Assigning a Call to a Name (line 60):
        
        # Call to get_package_dir(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'package' (line 60)
        package_27450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 51), 'package', False)
        # Processing the call keyword arguments (line 60)
        kwargs_27451 = {}
        # Getting the type of 'build_py' (line 60)
        build_py_27448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 26), 'build_py', False)
        # Obtaining the member 'get_package_dir' of a type (line 60)
        get_package_dir_27449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 26), build_py_27448, 'get_package_dir')
        # Calling get_package_dir(args, kwargs) (line 60)
        get_package_dir_call_result_27452 = invoke(stypy.reporting.localization.Localization(__file__, 60, 26), get_package_dir_27449, *[package_27450], **kwargs_27451)
        
        # Assigning a type to the variable 'package_dir' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'package_dir', get_package_dir_call_result_27452)
        
        # Assigning a Call to a Name (line 61):
        
        # Assigning a Call to a Name (line 61):
        
        # Call to join(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'package_dir' (line 61)
        package_dir_27456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 41), 'package_dir', False)
        
        # Call to basename(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'filename' (line 61)
        filename_27460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 70), 'filename', False)
        # Processing the call keyword arguments (line 61)
        kwargs_27461 = {}
        # Getting the type of 'os' (line 61)
        os_27457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 53), 'os', False)
        # Obtaining the member 'path' of a type (line 61)
        path_27458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 53), os_27457, 'path')
        # Obtaining the member 'basename' of a type (line 61)
        basename_27459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 53), path_27458, 'basename')
        # Calling basename(args, kwargs) (line 61)
        basename_call_result_27462 = invoke(stypy.reporting.localization.Localization(__file__, 61, 53), basename_27459, *[filename_27460], **kwargs_27461)
        
        # Processing the call keyword arguments (line 61)
        kwargs_27463 = {}
        # Getting the type of 'os' (line 61)
        os_27453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 61)
        path_27454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 28), os_27453, 'path')
        # Obtaining the member 'join' of a type (line 61)
        join_27455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 28), path_27454, 'join')
        # Calling join(args, kwargs) (line 61)
        join_call_result_27464 = invoke(stypy.reporting.localization.Localization(__file__, 61, 28), join_27455, *[package_dir_27456, basename_call_result_27462], **kwargs_27463)
        
        # Assigning a type to the variable 'dest_filename' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'dest_filename', join_call_result_27464)
        
        # Assigning a Call to a Name (line 62):
        
        # Assigning a Call to a Name (line 62):
        
        # Call to join(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_27468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 40), 'self', False)
        # Obtaining the member 'build_lib' of a type (line 62)
        build_lib_27469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 40), self_27468, 'build_lib')
        # Getting the type of 'filename' (line 62)
        filename_27470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 55), 'filename', False)
        # Processing the call keyword arguments (line 62)
        kwargs_27471 = {}
        # Getting the type of 'os' (line 62)
        os_27465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 62)
        path_27466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 27), os_27465, 'path')
        # Obtaining the member 'join' of a type (line 62)
        join_27467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 27), path_27466, 'join')
        # Calling join(args, kwargs) (line 62)
        join_call_result_27472 = invoke(stypy.reporting.localization.Localization(__file__, 62, 27), join_27467, *[build_lib_27469, filename_27470], **kwargs_27471)
        
        # Assigning a type to the variable 'src_filename' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'src_filename', join_call_result_27472)
        
        # Call to copy_file(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'src_filename' (line 68)
        src_filename_27474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'src_filename', False)
        # Getting the type of 'dest_filename' (line 68)
        dest_filename_27475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'dest_filename', False)
        # Processing the call keyword arguments (line 67)
        # Getting the type of 'self' (line 68)
        self_27476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 53), 'self', False)
        # Obtaining the member 'verbose' of a type (line 68)
        verbose_27477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 53), self_27476, 'verbose')
        keyword_27478 = verbose_27477
        # Getting the type of 'self' (line 69)
        self_27479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 69)
        dry_run_27480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 24), self_27479, 'dry_run')
        keyword_27481 = dry_run_27480
        kwargs_27482 = {'verbose': keyword_27478, 'dry_run': keyword_27481}
        # Getting the type of 'copy_file' (line 67)
        copy_file_27473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'copy_file', False)
        # Calling copy_file(args, kwargs) (line 67)
        copy_file_call_result_27483 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), copy_file_27473, *[src_filename_27474, dest_filename_27475], **kwargs_27482)
        
        
        # Getting the type of 'ext' (line 71)
        ext_27484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 'ext')
        # Obtaining the member '_needs_stub' of a type (line 71)
        _needs_stub_27485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 15), ext_27484, '_needs_stub')
        # Testing the type of an if condition (line 71)
        if_condition_27486 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 12), _needs_stub_27485)
        # Assigning a type to the variable 'if_condition_27486' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'if_condition_27486', if_condition_27486)
        # SSA begins for if statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write_stub(...): (line 72)
        # Processing the call arguments (line 72)
        
        # Evaluating a boolean operation
        # Getting the type of 'package_dir' (line 72)
        package_dir_27489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'package_dir', False)
        # Getting the type of 'os' (line 72)
        os_27490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 47), 'os', False)
        # Obtaining the member 'curdir' of a type (line 72)
        curdir_27491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 47), os_27490, 'curdir')
        # Applying the binary operator 'or' (line 72)
        result_or_keyword_27492 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 32), 'or', package_dir_27489, curdir_27491)
        
        # Getting the type of 'ext' (line 72)
        ext_27493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 58), 'ext', False)
        # Getting the type of 'True' (line 72)
        True_27494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 63), 'True', False)
        # Processing the call keyword arguments (line 72)
        kwargs_27495 = {}
        # Getting the type of 'self' (line 72)
        self_27487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'self', False)
        # Obtaining the member 'write_stub' of a type (line 72)
        write_stub_27488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 16), self_27487, 'write_stub')
        # Calling write_stub(args, kwargs) (line 72)
        write_stub_call_result_27496 = invoke(stypy.reporting.localization.Localization(__file__, 72, 16), write_stub_27488, *[result_or_keyword_27492, ext_27493, True_27494], **kwargs_27495)
        
        # SSA join for if statement (line 71)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'copy_extensions_to_source(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy_extensions_to_source' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_27497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27497)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy_extensions_to_source'
        return stypy_return_type_27497


    @norecursion
    def get_ext_filename(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_ext_filename'
        module_type_store = module_type_store.open_function_context('get_ext_filename', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_localization', localization)
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_function_name', 'build_ext.get_ext_filename')
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_param_names_list', ['fullname'])
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.get_ext_filename.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.get_ext_filename', ['fullname'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_ext_filename', localization, ['fullname'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_ext_filename(...)' code ##################

        
        # Assigning a Call to a Name (line 86):
        
        # Assigning a Call to a Name (line 86):
        
        # Call to get_ext_filename(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'self' (line 86)
        self_27500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 47), 'self', False)
        # Getting the type of 'fullname' (line 86)
        fullname_27501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 52), 'fullname', False)
        # Processing the call keyword arguments (line 86)
        kwargs_27502 = {}
        # Getting the type of '_build_ext' (line 86)
        _build_ext_27498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), '_build_ext', False)
        # Obtaining the member 'get_ext_filename' of a type (line 86)
        get_ext_filename_27499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 19), _build_ext_27498, 'get_ext_filename')
        # Calling get_ext_filename(args, kwargs) (line 86)
        get_ext_filename_call_result_27503 = invoke(stypy.reporting.localization.Localization(__file__, 86, 19), get_ext_filename_27499, *[self_27500, fullname_27501], **kwargs_27502)
        
        # Assigning a type to the variable 'filename' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'filename', get_ext_filename_call_result_27503)
        
        # Assigning a Subscript to a Name (line 87):
        
        # Assigning a Subscript to a Name (line 87):
        
        # Obtaining the type of the subscript
        # Getting the type of 'fullname' (line 87)
        fullname_27504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 27), 'fullname')
        # Getting the type of 'self' (line 87)
        self_27505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 14), 'self')
        # Obtaining the member 'ext_map' of a type (line 87)
        ext_map_27506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 14), self_27505, 'ext_map')
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___27507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 14), ext_map_27506, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_27508 = invoke(stypy.reporting.localization.Localization(__file__, 87, 14), getitem___27507, fullname_27504)
        
        # Assigning a type to the variable 'ext' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'ext', subscript_call_result_27508)
        
        
        # Call to isinstance(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'ext' (line 88)
        ext_27510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 22), 'ext', False)
        # Getting the type of 'Library' (line 88)
        Library_27511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 26), 'Library', False)
        # Processing the call keyword arguments (line 88)
        kwargs_27512 = {}
        # Getting the type of 'isinstance' (line 88)
        isinstance_27509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 88)
        isinstance_call_result_27513 = invoke(stypy.reporting.localization.Localization(__file__, 88, 11), isinstance_27509, *[ext_27510, Library_27511], **kwargs_27512)
        
        # Testing the type of an if condition (line 88)
        if_condition_27514 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 8), isinstance_call_result_27513)
        # Assigning a type to the variable 'if_condition_27514' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'if_condition_27514', if_condition_27514)
        # SSA begins for if statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 89):
        
        # Assigning a Subscript to a Name (line 89):
        
        # Obtaining the type of the subscript
        int_27515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 12), 'int')
        
        # Call to splitext(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'filename' (line 89)
        filename_27519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 39), 'filename', False)
        # Processing the call keyword arguments (line 89)
        kwargs_27520 = {}
        # Getting the type of 'os' (line 89)
        os_27516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 89)
        path_27517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 22), os_27516, 'path')
        # Obtaining the member 'splitext' of a type (line 89)
        splitext_27518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 22), path_27517, 'splitext')
        # Calling splitext(args, kwargs) (line 89)
        splitext_call_result_27521 = invoke(stypy.reporting.localization.Localization(__file__, 89, 22), splitext_27518, *[filename_27519], **kwargs_27520)
        
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___27522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), splitext_call_result_27521, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_27523 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), getitem___27522, int_27515)
        
        # Assigning a type to the variable 'tuple_var_assignment_27341' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'tuple_var_assignment_27341', subscript_call_result_27523)
        
        # Assigning a Subscript to a Name (line 89):
        
        # Obtaining the type of the subscript
        int_27524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 12), 'int')
        
        # Call to splitext(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'filename' (line 89)
        filename_27528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 39), 'filename', False)
        # Processing the call keyword arguments (line 89)
        kwargs_27529 = {}
        # Getting the type of 'os' (line 89)
        os_27525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 89)
        path_27526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 22), os_27525, 'path')
        # Obtaining the member 'splitext' of a type (line 89)
        splitext_27527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 22), path_27526, 'splitext')
        # Calling splitext(args, kwargs) (line 89)
        splitext_call_result_27530 = invoke(stypy.reporting.localization.Localization(__file__, 89, 22), splitext_27527, *[filename_27528], **kwargs_27529)
        
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___27531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), splitext_call_result_27530, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_27532 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), getitem___27531, int_27524)
        
        # Assigning a type to the variable 'tuple_var_assignment_27342' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'tuple_var_assignment_27342', subscript_call_result_27532)
        
        # Assigning a Name to a Name (line 89):
        # Getting the type of 'tuple_var_assignment_27341' (line 89)
        tuple_var_assignment_27341_27533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'tuple_var_assignment_27341')
        # Assigning a type to the variable 'fn' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'fn', tuple_var_assignment_27341_27533)
        
        # Assigning a Name to a Name (line 89):
        # Getting the type of 'tuple_var_assignment_27342' (line 89)
        tuple_var_assignment_27342_27534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'tuple_var_assignment_27342')
        # Assigning a type to the variable 'ext' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'ext', tuple_var_assignment_27342_27534)
        
        # Call to library_filename(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'fn' (line 90)
        fn_27538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 56), 'fn', False)
        # Getting the type of 'libtype' (line 90)
        libtype_27539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 59), 'libtype', False)
        # Processing the call keyword arguments (line 90)
        kwargs_27540 = {}
        # Getting the type of 'self' (line 90)
        self_27535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'self', False)
        # Obtaining the member 'shlib_compiler' of a type (line 90)
        shlib_compiler_27536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 19), self_27535, 'shlib_compiler')
        # Obtaining the member 'library_filename' of a type (line 90)
        library_filename_27537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 19), shlib_compiler_27536, 'library_filename')
        # Calling library_filename(args, kwargs) (line 90)
        library_filename_call_result_27541 = invoke(stypy.reporting.localization.Localization(__file__, 90, 19), library_filename_27537, *[fn_27538, libtype_27539], **kwargs_27540)
        
        # Assigning a type to the variable 'stypy_return_type' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'stypy_return_type', library_filename_call_result_27541)
        # SSA branch for the else part of an if statement (line 88)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'use_stubs' (line 91)
        use_stubs_27542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 13), 'use_stubs')
        # Getting the type of 'ext' (line 91)
        ext_27543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 27), 'ext')
        # Obtaining the member '_links_to_dynamic' of a type (line 91)
        _links_to_dynamic_27544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 27), ext_27543, '_links_to_dynamic')
        # Applying the binary operator 'and' (line 91)
        result_and_keyword_27545 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 13), 'and', use_stubs_27542, _links_to_dynamic_27544)
        
        # Testing the type of an if condition (line 91)
        if_condition_27546 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 13), result_and_keyword_27545)
        # Assigning a type to the variable 'if_condition_27546' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 13), 'if_condition_27546', if_condition_27546)
        # SSA begins for if statement (line 91)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 92):
        
        # Assigning a Subscript to a Name (line 92):
        
        # Obtaining the type of the subscript
        int_27547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 12), 'int')
        
        # Call to split(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'filename' (line 92)
        filename_27551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 33), 'filename', False)
        # Processing the call keyword arguments (line 92)
        kwargs_27552 = {}
        # Getting the type of 'os' (line 92)
        os_27548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 92)
        path_27549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 19), os_27548, 'path')
        # Obtaining the member 'split' of a type (line 92)
        split_27550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 19), path_27549, 'split')
        # Calling split(args, kwargs) (line 92)
        split_call_result_27553 = invoke(stypy.reporting.localization.Localization(__file__, 92, 19), split_27550, *[filename_27551], **kwargs_27552)
        
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___27554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), split_call_result_27553, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_27555 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), getitem___27554, int_27547)
        
        # Assigning a type to the variable 'tuple_var_assignment_27343' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'tuple_var_assignment_27343', subscript_call_result_27555)
        
        # Assigning a Subscript to a Name (line 92):
        
        # Obtaining the type of the subscript
        int_27556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 12), 'int')
        
        # Call to split(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'filename' (line 92)
        filename_27560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 33), 'filename', False)
        # Processing the call keyword arguments (line 92)
        kwargs_27561 = {}
        # Getting the type of 'os' (line 92)
        os_27557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 92)
        path_27558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 19), os_27557, 'path')
        # Obtaining the member 'split' of a type (line 92)
        split_27559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 19), path_27558, 'split')
        # Calling split(args, kwargs) (line 92)
        split_call_result_27562 = invoke(stypy.reporting.localization.Localization(__file__, 92, 19), split_27559, *[filename_27560], **kwargs_27561)
        
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___27563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), split_call_result_27562, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_27564 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), getitem___27563, int_27556)
        
        # Assigning a type to the variable 'tuple_var_assignment_27344' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'tuple_var_assignment_27344', subscript_call_result_27564)
        
        # Assigning a Name to a Name (line 92):
        # Getting the type of 'tuple_var_assignment_27343' (line 92)
        tuple_var_assignment_27343_27565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'tuple_var_assignment_27343')
        # Assigning a type to the variable 'd' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'd', tuple_var_assignment_27343_27565)
        
        # Assigning a Name to a Name (line 92):
        # Getting the type of 'tuple_var_assignment_27344' (line 92)
        tuple_var_assignment_27344_27566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'tuple_var_assignment_27344')
        # Assigning a type to the variable 'fn' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 14), 'fn', tuple_var_assignment_27344_27566)
        
        # Call to join(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'd' (line 93)
        d_27570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 32), 'd', False)
        str_27571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 34), 'str', 'dl-')
        # Getting the type of 'fn' (line 93)
        fn_27572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 40), 'fn', False)
        # Applying the binary operator '+' (line 93)
        result_add_27573 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 34), '+', str_27571, fn_27572)
        
        # Processing the call keyword arguments (line 93)
        kwargs_27574 = {}
        # Getting the type of 'os' (line 93)
        os_27567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 93)
        path_27568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 19), os_27567, 'path')
        # Obtaining the member 'join' of a type (line 93)
        join_27569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 19), path_27568, 'join')
        # Calling join(args, kwargs) (line 93)
        join_call_result_27575 = invoke(stypy.reporting.localization.Localization(__file__, 93, 19), join_27569, *[d_27570, result_add_27573], **kwargs_27574)
        
        # Assigning a type to the variable 'stypy_return_type' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'stypy_return_type', join_call_result_27575)
        # SSA branch for the else part of an if statement (line 91)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'filename' (line 95)
        filename_27576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 19), 'filename')
        # Assigning a type to the variable 'stypy_return_type' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'stypy_return_type', filename_27576)
        # SSA join for if statement (line 91)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 88)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_ext_filename(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_ext_filename' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_27577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27577)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_ext_filename'
        return stypy_return_type_27577


    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
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

        
        # Call to initialize_options(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'self' (line 98)
        self_27580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 38), 'self', False)
        # Processing the call keyword arguments (line 98)
        kwargs_27581 = {}
        # Getting the type of '_build_ext' (line 98)
        _build_ext_27578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), '_build_ext', False)
        # Obtaining the member 'initialize_options' of a type (line 98)
        initialize_options_27579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), _build_ext_27578, 'initialize_options')
        # Calling initialize_options(args, kwargs) (line 98)
        initialize_options_call_result_27582 = invoke(stypy.reporting.localization.Localization(__file__, 98, 8), initialize_options_27579, *[self_27580], **kwargs_27581)
        
        
        # Assigning a Name to a Attribute (line 99):
        
        # Assigning a Name to a Attribute (line 99):
        # Getting the type of 'None' (line 99)
        None_27583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 30), 'None')
        # Getting the type of 'self' (line 99)
        self_27584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self')
        # Setting the type of the member 'shlib_compiler' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_27584, 'shlib_compiler', None_27583)
        
        # Assigning a List to a Attribute (line 100):
        
        # Assigning a List to a Attribute (line 100):
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_27585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        
        # Getting the type of 'self' (line 100)
        self_27586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'self')
        # Setting the type of the member 'shlibs' of a type (line 100)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), self_27586, 'shlibs', list_27585)
        
        # Assigning a Dict to a Attribute (line 101):
        
        # Assigning a Dict to a Attribute (line 101):
        
        # Obtaining an instance of the builtin type 'dict' (line 101)
        dict_27587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 23), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 101)
        
        # Getting the type of 'self' (line 101)
        self_27588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'self')
        # Setting the type of the member 'ext_map' of a type (line 101)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), self_27588, 'ext_map', dict_27587)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_27589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27589)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_27589


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 103, 4, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'self', type_of_self)
        
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

        
        # Call to finalize_options(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'self' (line 104)
        self_27592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 36), 'self', False)
        # Processing the call keyword arguments (line 104)
        kwargs_27593 = {}
        # Getting the type of '_build_ext' (line 104)
        _build_ext_27590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), '_build_ext', False)
        # Obtaining the member 'finalize_options' of a type (line 104)
        finalize_options_27591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), _build_ext_27590, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 104)
        finalize_options_call_result_27594 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), finalize_options_27591, *[self_27592], **kwargs_27593)
        
        
        # Assigning a BoolOp to a Attribute (line 105):
        
        # Assigning a BoolOp to a Attribute (line 105):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 105)
        self_27595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'self')
        # Obtaining the member 'extensions' of a type (line 105)
        extensions_27596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 26), self_27595, 'extensions')
        
        # Obtaining an instance of the builtin type 'list' (line 105)
        list_27597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 105)
        
        # Applying the binary operator 'or' (line 105)
        result_or_keyword_27598 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 26), 'or', extensions_27596, list_27597)
        
        # Getting the type of 'self' (line 105)
        self_27599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'self')
        # Setting the type of the member 'extensions' of a type (line 105)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), self_27599, 'extensions', result_or_keyword_27598)
        
        # Call to check_extensions_list(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'self' (line 106)
        self_27602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 35), 'self', False)
        # Obtaining the member 'extensions' of a type (line 106)
        extensions_27603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 35), self_27602, 'extensions')
        # Processing the call keyword arguments (line 106)
        kwargs_27604 = {}
        # Getting the type of 'self' (line 106)
        self_27600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'self', False)
        # Obtaining the member 'check_extensions_list' of a type (line 106)
        check_extensions_list_27601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), self_27600, 'check_extensions_list')
        # Calling check_extensions_list(args, kwargs) (line 106)
        check_extensions_list_call_result_27605 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), check_extensions_list_27601, *[extensions_27603], **kwargs_27604)
        
        
        # Assigning a ListComp to a Attribute (line 107):
        
        # Assigning a ListComp to a Attribute (line 107):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 107)
        self_27612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 38), 'self')
        # Obtaining the member 'extensions' of a type (line 107)
        extensions_27613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 38), self_27612, 'extensions')
        comprehension_27614 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 23), extensions_27613)
        # Assigning a type to the variable 'ext' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 23), 'ext', comprehension_27614)
        
        # Call to isinstance(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'ext' (line 108)
        ext_27608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 38), 'ext', False)
        # Getting the type of 'Library' (line 108)
        Library_27609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 42), 'Library', False)
        # Processing the call keyword arguments (line 108)
        kwargs_27610 = {}
        # Getting the type of 'isinstance' (line 108)
        isinstance_27607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 108)
        isinstance_call_result_27611 = invoke(stypy.reporting.localization.Localization(__file__, 108, 27), isinstance_27607, *[ext_27608, Library_27609], **kwargs_27610)
        
        # Getting the type of 'ext' (line 107)
        ext_27606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 23), 'ext')
        list_27615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 23), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 23), list_27615, ext_27606)
        # Getting the type of 'self' (line 107)
        self_27616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'self')
        # Setting the type of the member 'shlibs' of a type (line 107)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 8), self_27616, 'shlibs', list_27615)
        
        # Getting the type of 'self' (line 109)
        self_27617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'self')
        # Obtaining the member 'shlibs' of a type (line 109)
        shlibs_27618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 11), self_27617, 'shlibs')
        # Testing the type of an if condition (line 109)
        if_condition_27619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 8), shlibs_27618)
        # Assigning a type to the variable 'if_condition_27619' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'if_condition_27619', if_condition_27619)
        # SSA begins for if statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to setup_shlib_compiler(...): (line 110)
        # Processing the call keyword arguments (line 110)
        kwargs_27622 = {}
        # Getting the type of 'self' (line 110)
        self_27620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'self', False)
        # Obtaining the member 'setup_shlib_compiler' of a type (line 110)
        setup_shlib_compiler_27621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 12), self_27620, 'setup_shlib_compiler')
        # Calling setup_shlib_compiler(args, kwargs) (line 110)
        setup_shlib_compiler_call_result_27623 = invoke(stypy.reporting.localization.Localization(__file__, 110, 12), setup_shlib_compiler_27621, *[], **kwargs_27622)
        
        # SSA join for if statement (line 109)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 111)
        self_27624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'self')
        # Obtaining the member 'extensions' of a type (line 111)
        extensions_27625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 19), self_27624, 'extensions')
        # Testing the type of a for loop iterable (line 111)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 8), extensions_27625)
        # Getting the type of the for loop variable (line 111)
        for_loop_var_27626 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 8), extensions_27625)
        # Assigning a type to the variable 'ext' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'ext', for_loop_var_27626)
        # SSA begins for a for statement (line 111)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Attribute (line 112):
        
        # Assigning a Call to a Attribute (line 112):
        
        # Call to get_ext_fullname(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'ext' (line 112)
        ext_27629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 51), 'ext', False)
        # Obtaining the member 'name' of a type (line 112)
        name_27630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 51), ext_27629, 'name')
        # Processing the call keyword arguments (line 112)
        kwargs_27631 = {}
        # Getting the type of 'self' (line 112)
        self_27627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 29), 'self', False)
        # Obtaining the member 'get_ext_fullname' of a type (line 112)
        get_ext_fullname_27628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 29), self_27627, 'get_ext_fullname')
        # Calling get_ext_fullname(args, kwargs) (line 112)
        get_ext_fullname_call_result_27632 = invoke(stypy.reporting.localization.Localization(__file__, 112, 29), get_ext_fullname_27628, *[name_27630], **kwargs_27631)
        
        # Getting the type of 'ext' (line 112)
        ext_27633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'ext')
        # Setting the type of the member '_full_name' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 12), ext_27633, '_full_name', get_ext_fullname_call_result_27632)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 113)
        self_27634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'self')
        # Obtaining the member 'extensions' of a type (line 113)
        extensions_27635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 19), self_27634, 'extensions')
        # Testing the type of a for loop iterable (line 113)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 113, 8), extensions_27635)
        # Getting the type of the for loop variable (line 113)
        for_loop_var_27636 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 113, 8), extensions_27635)
        # Assigning a type to the variable 'ext' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'ext', for_loop_var_27636)
        # SSA begins for a for statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Attribute to a Name (line 114):
        
        # Assigning a Attribute to a Name (line 114):
        # Getting the type of 'ext' (line 114)
        ext_27637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'ext')
        # Obtaining the member '_full_name' of a type (line 114)
        _full_name_27638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 23), ext_27637, '_full_name')
        # Assigning a type to the variable 'fullname' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'fullname', _full_name_27638)
        
        # Assigning a Name to a Subscript (line 115):
        
        # Assigning a Name to a Subscript (line 115):
        # Getting the type of 'ext' (line 115)
        ext_27639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 37), 'ext')
        # Getting the type of 'self' (line 115)
        self_27640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'self')
        # Obtaining the member 'ext_map' of a type (line 115)
        ext_map_27641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), self_27640, 'ext_map')
        # Getting the type of 'fullname' (line 115)
        fullname_27642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 25), 'fullname')
        # Storing an element on a container (line 115)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 12), ext_map_27641, (fullname_27642, ext_27639))
        
        # Multiple assignment of 2 elements.
        
        # Assigning a BoolOp to a Attribute (line 116):
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 117)
        self_27643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'self')
        # Obtaining the member 'shlibs' of a type (line 117)
        shlibs_27644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 16), self_27643, 'shlibs')
        
        # Call to links_to_dynamic(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'ext' (line 117)
        ext_27647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 54), 'ext', False)
        # Processing the call keyword arguments (line 117)
        kwargs_27648 = {}
        # Getting the type of 'self' (line 117)
        self_27645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 32), 'self', False)
        # Obtaining the member 'links_to_dynamic' of a type (line 117)
        links_to_dynamic_27646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 32), self_27645, 'links_to_dynamic')
        # Calling links_to_dynamic(args, kwargs) (line 117)
        links_to_dynamic_call_result_27649 = invoke(stypy.reporting.localization.Localization(__file__, 117, 32), links_to_dynamic_27646, *[ext_27647], **kwargs_27648)
        
        # Applying the binary operator 'and' (line 117)
        result_and_keyword_27650 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 16), 'and', shlibs_27644, links_to_dynamic_call_result_27649)
        
        # Getting the type of 'False' (line 117)
        False_27651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 62), 'False')
        # Applying the binary operator 'or' (line 117)
        result_or_keyword_27652 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 16), 'or', result_and_keyword_27650, False_27651)
        
        # Getting the type of 'ext' (line 116)
        ext_27653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 18), 'ext')
        # Setting the type of the member '_links_to_dynamic' of a type (line 116)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 18), ext_27653, '_links_to_dynamic', result_or_keyword_27652)
        
        # Assigning a Attribute to a Name (line 116):
        # Getting the type of 'ext' (line 116)
        ext_27654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 18), 'ext')
        # Obtaining the member '_links_to_dynamic' of a type (line 116)
        _links_to_dynamic_27655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 18), ext_27654, '_links_to_dynamic')
        # Assigning a type to the variable 'ltd' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'ltd', _links_to_dynamic_27655)
        
        # Assigning a BoolOp to a Attribute (line 118):
        
        # Assigning a BoolOp to a Attribute (line 118):
        
        # Evaluating a boolean operation
        # Getting the type of 'ltd' (line 118)
        ltd_27656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 30), 'ltd')
        # Getting the type of 'use_stubs' (line 118)
        use_stubs_27657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 38), 'use_stubs')
        # Applying the binary operator 'and' (line 118)
        result_and_keyword_27658 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 30), 'and', ltd_27656, use_stubs_27657)
        
        
        # Call to isinstance(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'ext' (line 118)
        ext_27660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 67), 'ext', False)
        # Getting the type of 'Library' (line 118)
        Library_27661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 71), 'Library', False)
        # Processing the call keyword arguments (line 118)
        kwargs_27662 = {}
        # Getting the type of 'isinstance' (line 118)
        isinstance_27659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 56), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 118)
        isinstance_call_result_27663 = invoke(stypy.reporting.localization.Localization(__file__, 118, 56), isinstance_27659, *[ext_27660, Library_27661], **kwargs_27662)
        
        # Applying the 'not' unary operator (line 118)
        result_not__27664 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 52), 'not', isinstance_call_result_27663)
        
        # Applying the binary operator 'and' (line 118)
        result_and_keyword_27665 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 30), 'and', result_and_keyword_27658, result_not__27664)
        
        # Getting the type of 'ext' (line 118)
        ext_27666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'ext')
        # Setting the type of the member '_needs_stub' of a type (line 118)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), ext_27666, '_needs_stub', result_and_keyword_27665)
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Call to a Attribute (line 119):
        
        # Call to get_ext_filename(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'fullname' (line 119)
        fullname_27669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 62), 'fullname', False)
        # Processing the call keyword arguments (line 119)
        kwargs_27670 = {}
        # Getting the type of 'self' (line 119)
        self_27667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 40), 'self', False)
        # Obtaining the member 'get_ext_filename' of a type (line 119)
        get_ext_filename_27668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 40), self_27667, 'get_ext_filename')
        # Calling get_ext_filename(args, kwargs) (line 119)
        get_ext_filename_call_result_27671 = invoke(stypy.reporting.localization.Localization(__file__, 119, 40), get_ext_filename_27668, *[fullname_27669], **kwargs_27670)
        
        # Getting the type of 'ext' (line 119)
        ext_27672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'ext')
        # Setting the type of the member '_file_name' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 23), ext_27672, '_file_name', get_ext_filename_call_result_27671)
        
        # Assigning a Attribute to a Name (line 119):
        # Getting the type of 'ext' (line 119)
        ext_27673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'ext')
        # Obtaining the member '_file_name' of a type (line 119)
        _file_name_27674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 23), ext_27673, '_file_name')
        # Assigning a type to the variable 'filename' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'filename', _file_name_27674)
        
        # Assigning a Call to a Name (line 120):
        
        # Assigning a Call to a Name (line 120):
        
        # Call to dirname(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Call to join(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'self' (line 120)
        self_27681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 50), 'self', False)
        # Obtaining the member 'build_lib' of a type (line 120)
        build_lib_27682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 50), self_27681, 'build_lib')
        # Getting the type of 'filename' (line 120)
        filename_27683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 65), 'filename', False)
        # Processing the call keyword arguments (line 120)
        kwargs_27684 = {}
        # Getting the type of 'os' (line 120)
        os_27678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 37), 'os', False)
        # Obtaining the member 'path' of a type (line 120)
        path_27679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 37), os_27678, 'path')
        # Obtaining the member 'join' of a type (line 120)
        join_27680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 37), path_27679, 'join')
        # Calling join(args, kwargs) (line 120)
        join_call_result_27685 = invoke(stypy.reporting.localization.Localization(__file__, 120, 37), join_27680, *[build_lib_27682, filename_27683], **kwargs_27684)
        
        # Processing the call keyword arguments (line 120)
        kwargs_27686 = {}
        # Getting the type of 'os' (line 120)
        os_27675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 120)
        path_27676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 21), os_27675, 'path')
        # Obtaining the member 'dirname' of a type (line 120)
        dirname_27677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 21), path_27676, 'dirname')
        # Calling dirname(args, kwargs) (line 120)
        dirname_call_result_27687 = invoke(stypy.reporting.localization.Localization(__file__, 120, 21), dirname_27677, *[join_call_result_27685], **kwargs_27686)
        
        # Assigning a type to the variable 'libdir' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'libdir', dirname_call_result_27687)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'ltd' (line 121)
        ltd_27688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'ltd')
        
        # Getting the type of 'libdir' (line 121)
        libdir_27689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'libdir')
        # Getting the type of 'ext' (line 121)
        ext_27690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 37), 'ext')
        # Obtaining the member 'library_dirs' of a type (line 121)
        library_dirs_27691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 37), ext_27690, 'library_dirs')
        # Applying the binary operator 'notin' (line 121)
        result_contains_27692 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 23), 'notin', libdir_27689, library_dirs_27691)
        
        # Applying the binary operator 'and' (line 121)
        result_and_keyword_27693 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 15), 'and', ltd_27688, result_contains_27692)
        
        # Testing the type of an if condition (line 121)
        if_condition_27694 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 12), result_and_keyword_27693)
        # Assigning a type to the variable 'if_condition_27694' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'if_condition_27694', if_condition_27694)
        # SSA begins for if statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'libdir' (line 122)
        libdir_27698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 40), 'libdir', False)
        # Processing the call keyword arguments (line 122)
        kwargs_27699 = {}
        # Getting the type of 'ext' (line 122)
        ext_27695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'ext', False)
        # Obtaining the member 'library_dirs' of a type (line 122)
        library_dirs_27696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 16), ext_27695, 'library_dirs')
        # Obtaining the member 'append' of a type (line 122)
        append_27697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 16), library_dirs_27696, 'append')
        # Calling append(args, kwargs) (line 122)
        append_call_result_27700 = invoke(stypy.reporting.localization.Localization(__file__, 122, 16), append_27697, *[libdir_27698], **kwargs_27699)
        
        # SSA join for if statement (line 121)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'ltd' (line 123)
        ltd_27701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'ltd')
        # Getting the type of 'use_stubs' (line 123)
        use_stubs_27702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'use_stubs')
        # Applying the binary operator 'and' (line 123)
        result_and_keyword_27703 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 15), 'and', ltd_27701, use_stubs_27702)
        
        # Getting the type of 'os' (line 123)
        os_27704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 37), 'os')
        # Obtaining the member 'curdir' of a type (line 123)
        curdir_27705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 37), os_27704, 'curdir')
        # Getting the type of 'ext' (line 123)
        ext_27706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 54), 'ext')
        # Obtaining the member 'runtime_library_dirs' of a type (line 123)
        runtime_library_dirs_27707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 54), ext_27706, 'runtime_library_dirs')
        # Applying the binary operator 'notin' (line 123)
        result_contains_27708 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 37), 'notin', curdir_27705, runtime_library_dirs_27707)
        
        # Applying the binary operator 'and' (line 123)
        result_and_keyword_27709 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 15), 'and', result_and_keyword_27703, result_contains_27708)
        
        # Testing the type of an if condition (line 123)
        if_condition_27710 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 12), result_and_keyword_27709)
        # Assigning a type to the variable 'if_condition_27710' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'if_condition_27710', if_condition_27710)
        # SSA begins for if statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'os' (line 124)
        os_27714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 48), 'os', False)
        # Obtaining the member 'curdir' of a type (line 124)
        curdir_27715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 48), os_27714, 'curdir')
        # Processing the call keyword arguments (line 124)
        kwargs_27716 = {}
        # Getting the type of 'ext' (line 124)
        ext_27711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'ext', False)
        # Obtaining the member 'runtime_library_dirs' of a type (line 124)
        runtime_library_dirs_27712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 16), ext_27711, 'runtime_library_dirs')
        # Obtaining the member 'append' of a type (line 124)
        append_27713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 16), runtime_library_dirs_27712, 'append')
        # Calling append(args, kwargs) (line 124)
        append_call_result_27717 = invoke(stypy.reporting.localization.Localization(__file__, 124, 16), append_27713, *[curdir_27715], **kwargs_27716)
        
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_27718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27718)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_27718


    @norecursion
    def setup_shlib_compiler(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_shlib_compiler'
        module_type_store = module_type_store.open_function_context('setup_shlib_compiler', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.setup_shlib_compiler.__dict__.__setitem__('stypy_localization', localization)
        build_ext.setup_shlib_compiler.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.setup_shlib_compiler.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.setup_shlib_compiler.__dict__.__setitem__('stypy_function_name', 'build_ext.setup_shlib_compiler')
        build_ext.setup_shlib_compiler.__dict__.__setitem__('stypy_param_names_list', [])
        build_ext.setup_shlib_compiler.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.setup_shlib_compiler.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.setup_shlib_compiler.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.setup_shlib_compiler.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.setup_shlib_compiler.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.setup_shlib_compiler.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.setup_shlib_compiler', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_shlib_compiler', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_shlib_compiler(...)' code ##################

        
        # Multiple assignment of 2 elements.
        
        # Assigning a Call to a Attribute (line 127):
        
        # Call to new_compiler(...): (line 127)
        # Processing the call keyword arguments (line 127)
        # Getting the type of 'self' (line 128)
        self_27720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 21), 'self', False)
        # Obtaining the member 'compiler' of a type (line 128)
        compiler_27721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 21), self_27720, 'compiler')
        keyword_27722 = compiler_27721
        # Getting the type of 'self' (line 128)
        self_27723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 44), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 128)
        dry_run_27724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 44), self_27723, 'dry_run')
        keyword_27725 = dry_run_27724
        # Getting the type of 'self' (line 128)
        self_27726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 64), 'self', False)
        # Obtaining the member 'force' of a type (line 128)
        force_27727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 64), self_27726, 'force')
        keyword_27728 = force_27727
        kwargs_27729 = {'force': keyword_27728, 'dry_run': keyword_27725, 'compiler': keyword_27722}
        # Getting the type of 'new_compiler' (line 127)
        new_compiler_27719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 41), 'new_compiler', False)
        # Calling new_compiler(args, kwargs) (line 127)
        new_compiler_call_result_27730 = invoke(stypy.reporting.localization.Localization(__file__, 127, 41), new_compiler_27719, *[], **kwargs_27729)
        
        # Getting the type of 'self' (line 127)
        self_27731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'self')
        # Setting the type of the member 'shlib_compiler' of a type (line 127)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 19), self_27731, 'shlib_compiler', new_compiler_call_result_27730)
        
        # Assigning a Attribute to a Name (line 127):
        # Getting the type of 'self' (line 127)
        self_27732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'self')
        # Obtaining the member 'shlib_compiler' of a type (line 127)
        shlib_compiler_27733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 19), self_27732, 'shlib_compiler')
        # Assigning a type to the variable 'compiler' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'compiler', shlib_compiler_27733)
        
        
        # Getting the type of 'sys' (line 130)
        sys_27734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 130)
        platform_27735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 11), sys_27734, 'platform')
        str_27736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 27), 'str', 'darwin')
        # Applying the binary operator '==' (line 130)
        result_eq_27737 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 11), '==', platform_27735, str_27736)
        
        # Testing the type of an if condition (line 130)
        if_condition_27738 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), result_eq_27737)
        # Assigning a type to the variable 'if_condition_27738' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_27738', if_condition_27738)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 131):
        
        # Assigning a Call to a Name (line 131):
        
        # Call to copy(...): (line 131)
        # Processing the call keyword arguments (line 131)
        kwargs_27741 = {}
        # Getting the type of '_config_vars' (line 131)
        _config_vars_27739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 18), '_config_vars', False)
        # Obtaining the member 'copy' of a type (line 131)
        copy_27740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 18), _config_vars_27739, 'copy')
        # Calling copy(args, kwargs) (line 131)
        copy_call_result_27742 = invoke(stypy.reporting.localization.Localization(__file__, 131, 18), copy_27740, *[], **kwargs_27741)
        
        # Assigning a type to the variable 'tmp' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'tmp', copy_call_result_27742)
        
        # Try-finally block (line 132)
        
        # Assigning a Str to a Subscript (line 134):
        
        # Assigning a Str to a Subscript (line 134):
        str_27743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 43), 'str', 'gcc -Wl,-x -dynamiclib -undefined dynamic_lookup')
        # Getting the type of '_config_vars' (line 134)
        _config_vars_27744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 16), '_config_vars')
        str_27745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 29), 'str', 'LDSHARED')
        # Storing an element on a container (line 134)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 16), _config_vars_27744, (str_27745, str_27743))
        
        # Assigning a Str to a Subscript (line 135):
        
        # Assigning a Str to a Subscript (line 135):
        str_27746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 43), 'str', ' -dynamiclib')
        # Getting the type of '_config_vars' (line 135)
        _config_vars_27747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), '_config_vars')
        str_27748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 29), 'str', 'CCSHARED')
        # Storing an element on a container (line 135)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 16), _config_vars_27747, (str_27748, str_27746))
        
        # Assigning a Str to a Subscript (line 136):
        
        # Assigning a Str to a Subscript (line 136):
        str_27749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 37), 'str', '.dylib')
        # Getting the type of '_config_vars' (line 136)
        _config_vars_27750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), '_config_vars')
        str_27751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 29), 'str', 'SO')
        # Storing an element on a container (line 136)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 16), _config_vars_27750, (str_27751, str_27749))
        
        # Call to customize_compiler(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'compiler' (line 137)
        compiler_27753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 35), 'compiler', False)
        # Processing the call keyword arguments (line 137)
        kwargs_27754 = {}
        # Getting the type of 'customize_compiler' (line 137)
        customize_compiler_27752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'customize_compiler', False)
        # Calling customize_compiler(args, kwargs) (line 137)
        customize_compiler_call_result_27755 = invoke(stypy.reporting.localization.Localization(__file__, 137, 16), customize_compiler_27752, *[compiler_27753], **kwargs_27754)
        
        
        # finally branch of the try-finally block (line 132)
        
        # Call to clear(...): (line 139)
        # Processing the call keyword arguments (line 139)
        kwargs_27758 = {}
        # Getting the type of '_config_vars' (line 139)
        _config_vars_27756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), '_config_vars', False)
        # Obtaining the member 'clear' of a type (line 139)
        clear_27757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 16), _config_vars_27756, 'clear')
        # Calling clear(args, kwargs) (line 139)
        clear_call_result_27759 = invoke(stypy.reporting.localization.Localization(__file__, 139, 16), clear_27757, *[], **kwargs_27758)
        
        
        # Call to update(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'tmp' (line 140)
        tmp_27762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 36), 'tmp', False)
        # Processing the call keyword arguments (line 140)
        kwargs_27763 = {}
        # Getting the type of '_config_vars' (line 140)
        _config_vars_27760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), '_config_vars', False)
        # Obtaining the member 'update' of a type (line 140)
        update_27761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 16), _config_vars_27760, 'update')
        # Calling update(args, kwargs) (line 140)
        update_call_result_27764 = invoke(stypy.reporting.localization.Localization(__file__, 140, 16), update_27761, *[tmp_27762], **kwargs_27763)
        
        
        # SSA branch for the else part of an if statement (line 130)
        module_type_store.open_ssa_branch('else')
        
        # Call to customize_compiler(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'compiler' (line 142)
        compiler_27766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 31), 'compiler', False)
        # Processing the call keyword arguments (line 142)
        kwargs_27767 = {}
        # Getting the type of 'customize_compiler' (line 142)
        customize_compiler_27765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'customize_compiler', False)
        # Calling customize_compiler(args, kwargs) (line 142)
        customize_compiler_call_result_27768 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), customize_compiler_27765, *[compiler_27766], **kwargs_27767)
        
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 144)
        self_27769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'self')
        # Obtaining the member 'include_dirs' of a type (line 144)
        include_dirs_27770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 11), self_27769, 'include_dirs')
        # Getting the type of 'None' (line 144)
        None_27771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 36), 'None')
        # Applying the binary operator 'isnot' (line 144)
        result_is_not_27772 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 11), 'isnot', include_dirs_27770, None_27771)
        
        # Testing the type of an if condition (line 144)
        if_condition_27773 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 8), result_is_not_27772)
        # Assigning a type to the variable 'if_condition_27773' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'if_condition_27773', if_condition_27773)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_include_dirs(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'self' (line 145)
        self_27776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 38), 'self', False)
        # Obtaining the member 'include_dirs' of a type (line 145)
        include_dirs_27777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 38), self_27776, 'include_dirs')
        # Processing the call keyword arguments (line 145)
        kwargs_27778 = {}
        # Getting the type of 'compiler' (line 145)
        compiler_27774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'compiler', False)
        # Obtaining the member 'set_include_dirs' of a type (line 145)
        set_include_dirs_27775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), compiler_27774, 'set_include_dirs')
        # Calling set_include_dirs(args, kwargs) (line 145)
        set_include_dirs_call_result_27779 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), set_include_dirs_27775, *[include_dirs_27777], **kwargs_27778)
        
        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 146)
        self_27780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'self')
        # Obtaining the member 'define' of a type (line 146)
        define_27781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 11), self_27780, 'define')
        # Getting the type of 'None' (line 146)
        None_27782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 30), 'None')
        # Applying the binary operator 'isnot' (line 146)
        result_is_not_27783 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 11), 'isnot', define_27781, None_27782)
        
        # Testing the type of an if condition (line 146)
        if_condition_27784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 8), result_is_not_27783)
        # Assigning a type to the variable 'if_condition_27784' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'if_condition_27784', if_condition_27784)
        # SSA begins for if statement (line 146)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 148)
        self_27785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 32), 'self')
        # Obtaining the member 'define' of a type (line 148)
        define_27786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 32), self_27785, 'define')
        # Testing the type of a for loop iterable (line 148)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 148, 12), define_27786)
        # Getting the type of the for loop variable (line 148)
        for_loop_var_27787 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 148, 12), define_27786)
        # Assigning a type to the variable 'name' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 12), for_loop_var_27787))
        # Assigning a type to the variable 'value' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 12), for_loop_var_27787))
        # SSA begins for a for statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to define_macro(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'name' (line 149)
        name_27790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 38), 'name', False)
        # Getting the type of 'value' (line 149)
        value_27791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 44), 'value', False)
        # Processing the call keyword arguments (line 149)
        kwargs_27792 = {}
        # Getting the type of 'compiler' (line 149)
        compiler_27788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'compiler', False)
        # Obtaining the member 'define_macro' of a type (line 149)
        define_macro_27789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 16), compiler_27788, 'define_macro')
        # Calling define_macro(args, kwargs) (line 149)
        define_macro_call_result_27793 = invoke(stypy.reporting.localization.Localization(__file__, 149, 16), define_macro_27789, *[name_27790, value_27791], **kwargs_27792)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 146)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 150)
        self_27794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'self')
        # Obtaining the member 'undef' of a type (line 150)
        undef_27795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 11), self_27794, 'undef')
        # Getting the type of 'None' (line 150)
        None_27796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 29), 'None')
        # Applying the binary operator 'isnot' (line 150)
        result_is_not_27797 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 11), 'isnot', undef_27795, None_27796)
        
        # Testing the type of an if condition (line 150)
        if_condition_27798 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 8), result_is_not_27797)
        # Assigning a type to the variable 'if_condition_27798' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'if_condition_27798', if_condition_27798)
        # SSA begins for if statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 151)
        self_27799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'self')
        # Obtaining the member 'undef' of a type (line 151)
        undef_27800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 25), self_27799, 'undef')
        # Testing the type of a for loop iterable (line 151)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 151, 12), undef_27800)
        # Getting the type of the for loop variable (line 151)
        for_loop_var_27801 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 151, 12), undef_27800)
        # Assigning a type to the variable 'macro' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'macro', for_loop_var_27801)
        # SSA begins for a for statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to undefine_macro(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'macro' (line 152)
        macro_27804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 40), 'macro', False)
        # Processing the call keyword arguments (line 152)
        kwargs_27805 = {}
        # Getting the type of 'compiler' (line 152)
        compiler_27802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'compiler', False)
        # Obtaining the member 'undefine_macro' of a type (line 152)
        undefine_macro_27803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), compiler_27802, 'undefine_macro')
        # Calling undefine_macro(args, kwargs) (line 152)
        undefine_macro_call_result_27806 = invoke(stypy.reporting.localization.Localization(__file__, 152, 16), undefine_macro_27803, *[macro_27804], **kwargs_27805)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 150)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 153)
        self_27807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'self')
        # Obtaining the member 'libraries' of a type (line 153)
        libraries_27808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 11), self_27807, 'libraries')
        # Getting the type of 'None' (line 153)
        None_27809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'None')
        # Applying the binary operator 'isnot' (line 153)
        result_is_not_27810 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 11), 'isnot', libraries_27808, None_27809)
        
        # Testing the type of an if condition (line 153)
        if_condition_27811 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 8), result_is_not_27810)
        # Assigning a type to the variable 'if_condition_27811' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'if_condition_27811', if_condition_27811)
        # SSA begins for if statement (line 153)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_libraries(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'self' (line 154)
        self_27814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 35), 'self', False)
        # Obtaining the member 'libraries' of a type (line 154)
        libraries_27815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 35), self_27814, 'libraries')
        # Processing the call keyword arguments (line 154)
        kwargs_27816 = {}
        # Getting the type of 'compiler' (line 154)
        compiler_27812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'compiler', False)
        # Obtaining the member 'set_libraries' of a type (line 154)
        set_libraries_27813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), compiler_27812, 'set_libraries')
        # Calling set_libraries(args, kwargs) (line 154)
        set_libraries_call_result_27817 = invoke(stypy.reporting.localization.Localization(__file__, 154, 12), set_libraries_27813, *[libraries_27815], **kwargs_27816)
        
        # SSA join for if statement (line 153)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 155)
        self_27818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'self')
        # Obtaining the member 'library_dirs' of a type (line 155)
        library_dirs_27819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 11), self_27818, 'library_dirs')
        # Getting the type of 'None' (line 155)
        None_27820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 36), 'None')
        # Applying the binary operator 'isnot' (line 155)
        result_is_not_27821 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 11), 'isnot', library_dirs_27819, None_27820)
        
        # Testing the type of an if condition (line 155)
        if_condition_27822 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 8), result_is_not_27821)
        # Assigning a type to the variable 'if_condition_27822' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'if_condition_27822', if_condition_27822)
        # SSA begins for if statement (line 155)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_library_dirs(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'self' (line 156)
        self_27825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 38), 'self', False)
        # Obtaining the member 'library_dirs' of a type (line 156)
        library_dirs_27826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 38), self_27825, 'library_dirs')
        # Processing the call keyword arguments (line 156)
        kwargs_27827 = {}
        # Getting the type of 'compiler' (line 156)
        compiler_27823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'compiler', False)
        # Obtaining the member 'set_library_dirs' of a type (line 156)
        set_library_dirs_27824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 12), compiler_27823, 'set_library_dirs')
        # Calling set_library_dirs(args, kwargs) (line 156)
        set_library_dirs_call_result_27828 = invoke(stypy.reporting.localization.Localization(__file__, 156, 12), set_library_dirs_27824, *[library_dirs_27826], **kwargs_27827)
        
        # SSA join for if statement (line 155)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 157)
        self_27829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 11), 'self')
        # Obtaining the member 'rpath' of a type (line 157)
        rpath_27830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 11), self_27829, 'rpath')
        # Getting the type of 'None' (line 157)
        None_27831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 29), 'None')
        # Applying the binary operator 'isnot' (line 157)
        result_is_not_27832 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 11), 'isnot', rpath_27830, None_27831)
        
        # Testing the type of an if condition (line 157)
        if_condition_27833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 8), result_is_not_27832)
        # Assigning a type to the variable 'if_condition_27833' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'if_condition_27833', if_condition_27833)
        # SSA begins for if statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_runtime_library_dirs(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'self' (line 158)
        self_27836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 46), 'self', False)
        # Obtaining the member 'rpath' of a type (line 158)
        rpath_27837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 46), self_27836, 'rpath')
        # Processing the call keyword arguments (line 158)
        kwargs_27838 = {}
        # Getting the type of 'compiler' (line 158)
        compiler_27834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'compiler', False)
        # Obtaining the member 'set_runtime_library_dirs' of a type (line 158)
        set_runtime_library_dirs_27835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 12), compiler_27834, 'set_runtime_library_dirs')
        # Calling set_runtime_library_dirs(args, kwargs) (line 158)
        set_runtime_library_dirs_call_result_27839 = invoke(stypy.reporting.localization.Localization(__file__, 158, 12), set_runtime_library_dirs_27835, *[rpath_27837], **kwargs_27838)
        
        # SSA join for if statement (line 157)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 159)
        self_27840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'self')
        # Obtaining the member 'link_objects' of a type (line 159)
        link_objects_27841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 11), self_27840, 'link_objects')
        # Getting the type of 'None' (line 159)
        None_27842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 36), 'None')
        # Applying the binary operator 'isnot' (line 159)
        result_is_not_27843 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 11), 'isnot', link_objects_27841, None_27842)
        
        # Testing the type of an if condition (line 159)
        if_condition_27844 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 8), result_is_not_27843)
        # Assigning a type to the variable 'if_condition_27844' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'if_condition_27844', if_condition_27844)
        # SSA begins for if statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_link_objects(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'self' (line 160)
        self_27847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 38), 'self', False)
        # Obtaining the member 'link_objects' of a type (line 160)
        link_objects_27848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 38), self_27847, 'link_objects')
        # Processing the call keyword arguments (line 160)
        kwargs_27849 = {}
        # Getting the type of 'compiler' (line 160)
        compiler_27845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'compiler', False)
        # Obtaining the member 'set_link_objects' of a type (line 160)
        set_link_objects_27846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), compiler_27845, 'set_link_objects')
        # Calling set_link_objects(args, kwargs) (line 160)
        set_link_objects_call_result_27850 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), set_link_objects_27846, *[link_objects_27848], **kwargs_27849)
        
        # SSA join for if statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 163):
        
        # Assigning a Call to a Attribute (line 163):
        
        # Call to __get__(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'compiler' (line 163)
        compiler_27853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 65), 'compiler', False)
        # Processing the call keyword arguments (line 163)
        kwargs_27854 = {}
        # Getting the type of 'link_shared_object' (line 163)
        link_shared_object_27851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 38), 'link_shared_object', False)
        # Obtaining the member '__get__' of a type (line 163)
        get___27852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 38), link_shared_object_27851, '__get__')
        # Calling __get__(args, kwargs) (line 163)
        get___call_result_27855 = invoke(stypy.reporting.localization.Localization(__file__, 163, 38), get___27852, *[compiler_27853], **kwargs_27854)
        
        # Getting the type of 'compiler' (line 163)
        compiler_27856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'compiler')
        # Setting the type of the member 'link_shared_object' of a type (line 163)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), compiler_27856, 'link_shared_object', get___call_result_27855)
        
        # ################# End of 'setup_shlib_compiler(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_shlib_compiler' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_27857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27857)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_shlib_compiler'
        return stypy_return_type_27857


    @norecursion
    def get_export_symbols(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_export_symbols'
        module_type_store = module_type_store.open_function_context('get_export_symbols', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
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

        
        
        # Call to isinstance(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'ext' (line 168)
        ext_27859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 22), 'ext', False)
        # Getting the type of 'Library' (line 168)
        Library_27860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 26), 'Library', False)
        # Processing the call keyword arguments (line 168)
        kwargs_27861 = {}
        # Getting the type of 'isinstance' (line 168)
        isinstance_27858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 168)
        isinstance_call_result_27862 = invoke(stypy.reporting.localization.Localization(__file__, 168, 11), isinstance_27858, *[ext_27859, Library_27860], **kwargs_27861)
        
        # Testing the type of an if condition (line 168)
        if_condition_27863 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 8), isinstance_call_result_27862)
        # Assigning a type to the variable 'if_condition_27863' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'if_condition_27863', if_condition_27863)
        # SSA begins for if statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'ext' (line 169)
        ext_27864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 19), 'ext')
        # Obtaining the member 'export_symbols' of a type (line 169)
        export_symbols_27865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 19), ext_27864, 'export_symbols')
        # Assigning a type to the variable 'stypy_return_type' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'stypy_return_type', export_symbols_27865)
        # SSA join for if statement (line 168)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to get_export_symbols(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'self' (line 170)
        self_27868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 45), 'self', False)
        # Getting the type of 'ext' (line 170)
        ext_27869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 50), 'ext', False)
        # Processing the call keyword arguments (line 170)
        kwargs_27870 = {}
        # Getting the type of '_build_ext' (line 170)
        _build_ext_27866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), '_build_ext', False)
        # Obtaining the member 'get_export_symbols' of a type (line 170)
        get_export_symbols_27867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 15), _build_ext_27866, 'get_export_symbols')
        # Calling get_export_symbols(args, kwargs) (line 170)
        get_export_symbols_call_result_27871 = invoke(stypy.reporting.localization.Localization(__file__, 170, 15), get_export_symbols_27867, *[self_27868, ext_27869], **kwargs_27870)
        
        # Assigning a type to the variable 'stypy_return_type' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'stypy_return_type', get_export_symbols_call_result_27871)
        
        # ################# End of 'get_export_symbols(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_export_symbols' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_27872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27872)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_export_symbols'
        return stypy_return_type_27872


    @norecursion
    def build_extension(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_extension'
        module_type_store = module_type_store.open_function_context('build_extension', 172, 4, False)
        # Assigning a type to the variable 'self' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'self', type_of_self)
        
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

        
        # Assigning a Attribute to a Name (line 173):
        
        # Assigning a Attribute to a Name (line 173):
        # Getting the type of 'self' (line 173)
        self_27873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 20), 'self')
        # Obtaining the member 'compiler' of a type (line 173)
        compiler_27874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 20), self_27873, 'compiler')
        # Assigning a type to the variable '_compiler' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), '_compiler', compiler_27874)
        
        # Try-finally block (line 174)
        
        
        # Call to isinstance(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'ext' (line 175)
        ext_27876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 26), 'ext', False)
        # Getting the type of 'Library' (line 175)
        Library_27877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 30), 'Library', False)
        # Processing the call keyword arguments (line 175)
        kwargs_27878 = {}
        # Getting the type of 'isinstance' (line 175)
        isinstance_27875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 175)
        isinstance_call_result_27879 = invoke(stypy.reporting.localization.Localization(__file__, 175, 15), isinstance_27875, *[ext_27876, Library_27877], **kwargs_27878)
        
        # Testing the type of an if condition (line 175)
        if_condition_27880 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 12), isinstance_call_result_27879)
        # Assigning a type to the variable 'if_condition_27880' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'if_condition_27880', if_condition_27880)
        # SSA begins for if statement (line 175)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 176):
        
        # Assigning a Attribute to a Attribute (line 176):
        # Getting the type of 'self' (line 176)
        self_27881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 32), 'self')
        # Obtaining the member 'shlib_compiler' of a type (line 176)
        shlib_compiler_27882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 32), self_27881, 'shlib_compiler')
        # Getting the type of 'self' (line 176)
        self_27883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'self')
        # Setting the type of the member 'compiler' of a type (line 176)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 16), self_27883, 'compiler', shlib_compiler_27882)
        # SSA join for if statement (line 175)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to build_extension(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'self' (line 177)
        self_27886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 39), 'self', False)
        # Getting the type of 'ext' (line 177)
        ext_27887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 44), 'ext', False)
        # Processing the call keyword arguments (line 177)
        kwargs_27888 = {}
        # Getting the type of '_build_ext' (line 177)
        _build_ext_27884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), '_build_ext', False)
        # Obtaining the member 'build_extension' of a type (line 177)
        build_extension_27885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 12), _build_ext_27884, 'build_extension')
        # Calling build_extension(args, kwargs) (line 177)
        build_extension_call_result_27889 = invoke(stypy.reporting.localization.Localization(__file__, 177, 12), build_extension_27885, *[self_27886, ext_27887], **kwargs_27888)
        
        
        # Getting the type of 'ext' (line 178)
        ext_27890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'ext')
        # Obtaining the member '_needs_stub' of a type (line 178)
        _needs_stub_27891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 15), ext_27890, '_needs_stub')
        # Testing the type of an if condition (line 178)
        if_condition_27892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 178, 12), _needs_stub_27891)
        # Assigning a type to the variable 'if_condition_27892' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'if_condition_27892', if_condition_27892)
        # SSA begins for if statement (line 178)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write_stub(...): (line 179)
        # Processing the call arguments (line 179)
        
        # Call to get_finalized_command(...): (line 180)
        # Processing the call arguments (line 180)
        str_27897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 47), 'str', 'build_py')
        # Processing the call keyword arguments (line 180)
        kwargs_27898 = {}
        # Getting the type of 'self' (line 180)
        self_27895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 20), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 180)
        get_finalized_command_27896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 20), self_27895, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 180)
        get_finalized_command_call_result_27899 = invoke(stypy.reporting.localization.Localization(__file__, 180, 20), get_finalized_command_27896, *[str_27897], **kwargs_27898)
        
        # Obtaining the member 'build_lib' of a type (line 180)
        build_lib_27900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 20), get_finalized_command_call_result_27899, 'build_lib')
        # Getting the type of 'ext' (line 180)
        ext_27901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 70), 'ext', False)
        # Processing the call keyword arguments (line 179)
        kwargs_27902 = {}
        # Getting the type of 'self' (line 179)
        self_27893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'self', False)
        # Obtaining the member 'write_stub' of a type (line 179)
        write_stub_27894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 16), self_27893, 'write_stub')
        # Calling write_stub(args, kwargs) (line 179)
        write_stub_call_result_27903 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), write_stub_27894, *[build_lib_27900, ext_27901], **kwargs_27902)
        
        # SSA join for if statement (line 178)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # finally branch of the try-finally block (line 174)
        
        # Assigning a Name to a Attribute (line 183):
        
        # Assigning a Name to a Attribute (line 183):
        # Getting the type of '_compiler' (line 183)
        _compiler_27904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), '_compiler')
        # Getting the type of 'self' (line 183)
        self_27905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'self')
        # Setting the type of the member 'compiler' of a type (line 183)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), self_27905, 'compiler', _compiler_27904)
        
        
        # ################# End of 'build_extension(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_extension' in the type store
        # Getting the type of 'stypy_return_type' (line 172)
        stypy_return_type_27906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27906)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_extension'
        return stypy_return_type_27906


    @norecursion
    def links_to_dynamic(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'links_to_dynamic'
        module_type_store = module_type_store.open_function_context('links_to_dynamic', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.links_to_dynamic.__dict__.__setitem__('stypy_localization', localization)
        build_ext.links_to_dynamic.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.links_to_dynamic.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.links_to_dynamic.__dict__.__setitem__('stypy_function_name', 'build_ext.links_to_dynamic')
        build_ext.links_to_dynamic.__dict__.__setitem__('stypy_param_names_list', ['ext'])
        build_ext.links_to_dynamic.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.links_to_dynamic.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.links_to_dynamic.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.links_to_dynamic.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.links_to_dynamic.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.links_to_dynamic.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.links_to_dynamic', ['ext'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'links_to_dynamic', localization, ['ext'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'links_to_dynamic(...)' code ##################

        str_27907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 8), 'str', "Return true if 'ext' links to a dynamic lib in the same package")
        
        # Assigning a Call to a Name (line 190):
        
        # Assigning a Call to a Name (line 190):
        
        # Call to fromkeys(...): (line 190)
        # Processing the call arguments (line 190)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 190)
        self_27912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 60), 'self', False)
        # Obtaining the member 'shlibs' of a type (line 190)
        shlibs_27913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 60), self_27912, 'shlibs')
        comprehension_27914 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 34), shlibs_27913)
        # Assigning a type to the variable 'lib' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 34), 'lib', comprehension_27914)
        # Getting the type of 'lib' (line 190)
        lib_27910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 34), 'lib', False)
        # Obtaining the member '_full_name' of a type (line 190)
        _full_name_27911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 34), lib_27910, '_full_name')
        list_27915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 34), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 34), list_27915, _full_name_27911)
        # Processing the call keyword arguments (line 190)
        kwargs_27916 = {}
        # Getting the type of 'dict' (line 190)
        dict_27908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 19), 'dict', False)
        # Obtaining the member 'fromkeys' of a type (line 190)
        fromkeys_27909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 19), dict_27908, 'fromkeys')
        # Calling fromkeys(args, kwargs) (line 190)
        fromkeys_call_result_27917 = invoke(stypy.reporting.localization.Localization(__file__, 190, 19), fromkeys_27909, *[list_27915], **kwargs_27916)
        
        # Assigning a type to the variable 'libnames' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'libnames', fromkeys_call_result_27917)
        
        # Assigning a Call to a Name (line 191):
        
        # Assigning a Call to a Name (line 191):
        
        # Call to join(...): (line 191)
        # Processing the call arguments (line 191)
        
        # Obtaining the type of the subscript
        int_27920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 50), 'int')
        slice_27921 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 191, 23), None, int_27920, None)
        
        # Call to split(...): (line 191)
        # Processing the call arguments (line 191)
        str_27925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 44), 'str', '.')
        # Processing the call keyword arguments (line 191)
        kwargs_27926 = {}
        # Getting the type of 'ext' (line 191)
        ext_27922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 23), 'ext', False)
        # Obtaining the member '_full_name' of a type (line 191)
        _full_name_27923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 23), ext_27922, '_full_name')
        # Obtaining the member 'split' of a type (line 191)
        split_27924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 23), _full_name_27923, 'split')
        # Calling split(args, kwargs) (line 191)
        split_call_result_27927 = invoke(stypy.reporting.localization.Localization(__file__, 191, 23), split_27924, *[str_27925], **kwargs_27926)
        
        # Obtaining the member '__getitem__' of a type (line 191)
        getitem___27928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 23), split_call_result_27927, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 191)
        subscript_call_result_27929 = invoke(stypy.reporting.localization.Localization(__file__, 191, 23), getitem___27928, slice_27921)
        
        
        # Obtaining an instance of the builtin type 'list' (line 191)
        list_27930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 191)
        # Adding element type (line 191)
        str_27931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 55), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 54), list_27930, str_27931)
        
        # Applying the binary operator '+' (line 191)
        result_add_27932 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 23), '+', subscript_call_result_27929, list_27930)
        
        # Processing the call keyword arguments (line 191)
        kwargs_27933 = {}
        str_27918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 14), 'str', '.')
        # Obtaining the member 'join' of a type (line 191)
        join_27919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 14), str_27918, 'join')
        # Calling join(args, kwargs) (line 191)
        join_call_result_27934 = invoke(stypy.reporting.localization.Localization(__file__, 191, 14), join_27919, *[result_add_27932], **kwargs_27933)
        
        # Assigning a type to the variable 'pkg' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'pkg', join_call_result_27934)
        
        # Getting the type of 'ext' (line 192)
        ext_27935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 23), 'ext')
        # Obtaining the member 'libraries' of a type (line 192)
        libraries_27936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 23), ext_27935, 'libraries')
        # Testing the type of a for loop iterable (line 192)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 192, 8), libraries_27936)
        # Getting the type of the for loop variable (line 192)
        for_loop_var_27937 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 192, 8), libraries_27936)
        # Assigning a type to the variable 'libname' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'libname', for_loop_var_27937)
        # SSA begins for a for statement (line 192)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'pkg' (line 193)
        pkg_27938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'pkg')
        # Getting the type of 'libname' (line 193)
        libname_27939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 19), 'libname')
        # Applying the binary operator '+' (line 193)
        result_add_27940 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 15), '+', pkg_27938, libname_27939)
        
        # Getting the type of 'libnames' (line 193)
        libnames_27941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 30), 'libnames')
        # Applying the binary operator 'in' (line 193)
        result_contains_27942 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 15), 'in', result_add_27940, libnames_27941)
        
        # Testing the type of an if condition (line 193)
        if_condition_27943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 12), result_contains_27942)
        # Assigning a type to the variable 'if_condition_27943' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'if_condition_27943', if_condition_27943)
        # SSA begins for if statement (line 193)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 193)
        True_27944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 47), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 40), 'stypy_return_type', True_27944)
        # SSA join for if statement (line 193)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'False' (line 194)
        False_27945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'stypy_return_type', False_27945)
        
        # ################# End of 'links_to_dynamic(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'links_to_dynamic' in the type store
        # Getting the type of 'stypy_return_type' (line 185)
        stypy_return_type_27946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_27946)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'links_to_dynamic'
        return stypy_return_type_27946


    @norecursion
    def get_outputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_outputs'
        module_type_store = module_type_store.open_function_context('get_outputs', 196, 4, False)
        # Assigning a type to the variable 'self' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'self', type_of_self)
        
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

        
        # Assigning a Call to a Name (line 197):
        
        # Assigning a Call to a Name (line 197):
        
        # Call to get_outputs(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'self' (line 197)
        self_27949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 41), 'self', False)
        # Processing the call keyword arguments (line 197)
        kwargs_27950 = {}
        # Getting the type of '_build_ext' (line 197)
        _build_ext_27947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 18), '_build_ext', False)
        # Obtaining the member 'get_outputs' of a type (line 197)
        get_outputs_27948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 18), _build_ext_27947, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 197)
        get_outputs_call_result_27951 = invoke(stypy.reporting.localization.Localization(__file__, 197, 18), get_outputs_27948, *[self_27949], **kwargs_27950)
        
        # Assigning a type to the variable 'outputs' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'outputs', get_outputs_call_result_27951)
        
        # Assigning a Attribute to a Name (line 198):
        
        # Assigning a Attribute to a Name (line 198):
        
        # Call to get_finalized_command(...): (line 198)
        # Processing the call arguments (line 198)
        str_27954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 46), 'str', 'build_py')
        # Processing the call keyword arguments (line 198)
        kwargs_27955 = {}
        # Getting the type of 'self' (line 198)
        self_27952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 198)
        get_finalized_command_27953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 19), self_27952, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 198)
        get_finalized_command_call_result_27956 = invoke(stypy.reporting.localization.Localization(__file__, 198, 19), get_finalized_command_27953, *[str_27954], **kwargs_27955)
        
        # Obtaining the member 'optimize' of a type (line 198)
        optimize_27957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 19), get_finalized_command_call_result_27956, 'optimize')
        # Assigning a type to the variable 'optimize' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'optimize', optimize_27957)
        
        # Getting the type of 'self' (line 199)
        self_27958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'self')
        # Obtaining the member 'extensions' of a type (line 199)
        extensions_27959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 19), self_27958, 'extensions')
        # Testing the type of a for loop iterable (line 199)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 199, 8), extensions_27959)
        # Getting the type of the for loop variable (line 199)
        for_loop_var_27960 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 199, 8), extensions_27959)
        # Assigning a type to the variable 'ext' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'ext', for_loop_var_27960)
        # SSA begins for a for statement (line 199)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'ext' (line 200)
        ext_27961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'ext')
        # Obtaining the member '_needs_stub' of a type (line 200)
        _needs_stub_27962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 15), ext_27961, '_needs_stub')
        # Testing the type of an if condition (line 200)
        if_condition_27963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 12), _needs_stub_27962)
        # Assigning a type to the variable 'if_condition_27963' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'if_condition_27963', if_condition_27963)
        # SSA begins for if statement (line 200)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 201):
        
        # Assigning a Call to a Name (line 201):
        
        # Call to join(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'self' (line 201)
        self_27967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 36), 'self', False)
        # Obtaining the member 'build_lib' of a type (line 201)
        build_lib_27968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 36), self_27967, 'build_lib')
        
        # Call to split(...): (line 201)
        # Processing the call arguments (line 201)
        str_27972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 74), 'str', '.')
        # Processing the call keyword arguments (line 201)
        kwargs_27973 = {}
        # Getting the type of 'ext' (line 201)
        ext_27969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 53), 'ext', False)
        # Obtaining the member '_full_name' of a type (line 201)
        _full_name_27970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 53), ext_27969, '_full_name')
        # Obtaining the member 'split' of a type (line 201)
        split_27971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 53), _full_name_27970, 'split')
        # Calling split(args, kwargs) (line 201)
        split_call_result_27974 = invoke(stypy.reporting.localization.Localization(__file__, 201, 53), split_27971, *[str_27972], **kwargs_27973)
        
        # Processing the call keyword arguments (line 201)
        kwargs_27975 = {}
        # Getting the type of 'os' (line 201)
        os_27964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 201)
        path_27965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 23), os_27964, 'path')
        # Obtaining the member 'join' of a type (line 201)
        join_27966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 23), path_27965, 'join')
        # Calling join(args, kwargs) (line 201)
        join_call_result_27976 = invoke(stypy.reporting.localization.Localization(__file__, 201, 23), join_27966, *[build_lib_27968, split_call_result_27974], **kwargs_27975)
        
        # Assigning a type to the variable 'base' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'base', join_call_result_27976)
        
        # Call to append(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'base' (line 202)
        base_27979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 31), 'base', False)
        str_27980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 36), 'str', '.py')
        # Applying the binary operator '+' (line 202)
        result_add_27981 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 31), '+', base_27979, str_27980)
        
        # Processing the call keyword arguments (line 202)
        kwargs_27982 = {}
        # Getting the type of 'outputs' (line 202)
        outputs_27977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'outputs', False)
        # Obtaining the member 'append' of a type (line 202)
        append_27978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 16), outputs_27977, 'append')
        # Calling append(args, kwargs) (line 202)
        append_call_result_27983 = invoke(stypy.reporting.localization.Localization(__file__, 202, 16), append_27978, *[result_add_27981], **kwargs_27982)
        
        
        # Call to append(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'base' (line 203)
        base_27986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 31), 'base', False)
        str_27987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 36), 'str', '.pyc')
        # Applying the binary operator '+' (line 203)
        result_add_27988 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 31), '+', base_27986, str_27987)
        
        # Processing the call keyword arguments (line 203)
        kwargs_27989 = {}
        # Getting the type of 'outputs' (line 203)
        outputs_27984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'outputs', False)
        # Obtaining the member 'append' of a type (line 203)
        append_27985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 16), outputs_27984, 'append')
        # Calling append(args, kwargs) (line 203)
        append_call_result_27990 = invoke(stypy.reporting.localization.Localization(__file__, 203, 16), append_27985, *[result_add_27988], **kwargs_27989)
        
        
        # Getting the type of 'optimize' (line 204)
        optimize_27991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 19), 'optimize')
        # Testing the type of an if condition (line 204)
        if_condition_27992 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 16), optimize_27991)
        # Assigning a type to the variable 'if_condition_27992' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'if_condition_27992', if_condition_27992)
        # SSA begins for if statement (line 204)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'base' (line 205)
        base_27995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 35), 'base', False)
        str_27996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 40), 'str', '.pyo')
        # Applying the binary operator '+' (line 205)
        result_add_27997 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 35), '+', base_27995, str_27996)
        
        # Processing the call keyword arguments (line 205)
        kwargs_27998 = {}
        # Getting the type of 'outputs' (line 205)
        outputs_27993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'outputs', False)
        # Obtaining the member 'append' of a type (line 205)
        append_27994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 20), outputs_27993, 'append')
        # Calling append(args, kwargs) (line 205)
        append_call_result_27999 = invoke(stypy.reporting.localization.Localization(__file__, 205, 20), append_27994, *[result_add_27997], **kwargs_27998)
        
        # SSA join for if statement (line 204)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 200)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'outputs' (line 206)
        outputs_28000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 15), 'outputs')
        # Assigning a type to the variable 'stypy_return_type' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'stypy_return_type', outputs_28000)
        
        # ################# End of 'get_outputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_outputs' in the type store
        # Getting the type of 'stypy_return_type' (line 196)
        stypy_return_type_28001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28001)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_outputs'
        return stypy_return_type_28001


    @norecursion
    def write_stub(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 208)
        False_28002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 50), 'False')
        defaults = [False_28002]
        # Create a new context for function 'write_stub'
        module_type_store = module_type_store.open_function_context('write_stub', 208, 4, False)
        # Assigning a type to the variable 'self' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_ext.write_stub.__dict__.__setitem__('stypy_localization', localization)
        build_ext.write_stub.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_ext.write_stub.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_ext.write_stub.__dict__.__setitem__('stypy_function_name', 'build_ext.write_stub')
        build_ext.write_stub.__dict__.__setitem__('stypy_param_names_list', ['output_dir', 'ext', 'compile'])
        build_ext.write_stub.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_ext.write_stub.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_ext.write_stub.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_ext.write_stub.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_ext.write_stub.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_ext.write_stub.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_ext.write_stub', ['output_dir', 'ext', 'compile'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_stub', localization, ['output_dir', 'ext', 'compile'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_stub(...)' code ##################

        
        # Call to info(...): (line 209)
        # Processing the call arguments (line 209)
        str_28005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 17), 'str', 'writing stub loader for %s to %s')
        # Getting the type of 'ext' (line 209)
        ext_28006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 52), 'ext', False)
        # Obtaining the member '_full_name' of a type (line 209)
        _full_name_28007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 52), ext_28006, '_full_name')
        # Getting the type of 'output_dir' (line 209)
        output_dir_28008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 68), 'output_dir', False)
        # Processing the call keyword arguments (line 209)
        kwargs_28009 = {}
        # Getting the type of 'log' (line 209)
        log_28003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 209)
        info_28004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), log_28003, 'info')
        # Calling info(args, kwargs) (line 209)
        info_call_result_28010 = invoke(stypy.reporting.localization.Localization(__file__, 209, 8), info_28004, *[str_28005, _full_name_28007, output_dir_28008], **kwargs_28009)
        
        
        # Assigning a BinOp to a Name (line 210):
        
        # Assigning a BinOp to a Name (line 210):
        
        # Call to join(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'output_dir' (line 210)
        output_dir_28014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 33), 'output_dir', False)
        
        # Call to split(...): (line 210)
        # Processing the call arguments (line 210)
        str_28018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 67), 'str', '.')
        # Processing the call keyword arguments (line 210)
        kwargs_28019 = {}
        # Getting the type of 'ext' (line 210)
        ext_28015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 46), 'ext', False)
        # Obtaining the member '_full_name' of a type (line 210)
        _full_name_28016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 46), ext_28015, '_full_name')
        # Obtaining the member 'split' of a type (line 210)
        split_28017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 46), _full_name_28016, 'split')
        # Calling split(args, kwargs) (line 210)
        split_call_result_28020 = invoke(stypy.reporting.localization.Localization(__file__, 210, 46), split_28017, *[str_28018], **kwargs_28019)
        
        # Processing the call keyword arguments (line 210)
        kwargs_28021 = {}
        # Getting the type of 'os' (line 210)
        os_28011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 210)
        path_28012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 20), os_28011, 'path')
        # Obtaining the member 'join' of a type (line 210)
        join_28013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 20), path_28012, 'join')
        # Calling join(args, kwargs) (line 210)
        join_call_result_28022 = invoke(stypy.reporting.localization.Localization(__file__, 210, 20), join_28013, *[output_dir_28014, split_call_result_28020], **kwargs_28021)
        
        str_28023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 73), 'str', '.py')
        # Applying the binary operator '+' (line 210)
        result_add_28024 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 20), '+', join_call_result_28022, str_28023)
        
        # Assigning a type to the variable 'stub_file' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'stub_file', result_add_28024)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'compile' (line 211)
        compile_28025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'compile')
        
        # Call to exists(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'stub_file' (line 211)
        stub_file_28029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 38), 'stub_file', False)
        # Processing the call keyword arguments (line 211)
        kwargs_28030 = {}
        # Getting the type of 'os' (line 211)
        os_28026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 211)
        path_28027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 23), os_28026, 'path')
        # Obtaining the member 'exists' of a type (line 211)
        exists_28028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 23), path_28027, 'exists')
        # Calling exists(args, kwargs) (line 211)
        exists_call_result_28031 = invoke(stypy.reporting.localization.Localization(__file__, 211, 23), exists_28028, *[stub_file_28029], **kwargs_28030)
        
        # Applying the binary operator 'and' (line 211)
        result_and_keyword_28032 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 11), 'and', compile_28025, exists_call_result_28031)
        
        # Testing the type of an if condition (line 211)
        if_condition_28033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 8), result_and_keyword_28032)
        # Assigning a type to the variable 'if_condition_28033' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'if_condition_28033', if_condition_28033)
        # SSA begins for if statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsError(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'stub_file' (line 212)
        stub_file_28035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 33), 'stub_file', False)
        str_28036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 43), 'str', ' already exists! Please delete.')
        # Applying the binary operator '+' (line 212)
        result_add_28037 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 33), '+', stub_file_28035, str_28036)
        
        # Processing the call keyword arguments (line 212)
        kwargs_28038 = {}
        # Getting the type of 'DistutilsError' (line 212)
        DistutilsError_28034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 18), 'DistutilsError', False)
        # Calling DistutilsError(args, kwargs) (line 212)
        DistutilsError_call_result_28039 = invoke(stypy.reporting.localization.Localization(__file__, 212, 18), DistutilsError_28034, *[result_add_28037], **kwargs_28038)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 212, 12), DistutilsError_call_result_28039, 'raise parameter', BaseException)
        # SSA join for if statement (line 211)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 213)
        self_28040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 15), 'self')
        # Obtaining the member 'dry_run' of a type (line 213)
        dry_run_28041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 15), self_28040, 'dry_run')
        # Applying the 'not' unary operator (line 213)
        result_not__28042 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 11), 'not', dry_run_28041)
        
        # Testing the type of an if condition (line 213)
        if_condition_28043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 8), result_not__28042)
        # Assigning a type to the variable 'if_condition_28043' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'if_condition_28043', if_condition_28043)
        # SSA begins for if statement (line 213)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 214):
        
        # Assigning a Call to a Name (line 214):
        
        # Call to open(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'stub_file' (line 214)
        stub_file_28045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 21), 'stub_file', False)
        str_28046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 31), 'str', 'w')
        # Processing the call keyword arguments (line 214)
        kwargs_28047 = {}
        # Getting the type of 'open' (line 214)
        open_28044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'open', False)
        # Calling open(args, kwargs) (line 214)
        open_call_result_28048 = invoke(stypy.reporting.localization.Localization(__file__, 214, 16), open_28044, *[stub_file_28045, str_28046], **kwargs_28047)
        
        # Assigning a type to the variable 'f' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'f', open_call_result_28048)
        
        # Call to write(...): (line 215)
        # Processing the call arguments (line 215)
        
        # Call to join(...): (line 215)
        # Processing the call arguments (line 215)
        
        # Obtaining an instance of the builtin type 'list' (line 215)
        list_28053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 215)
        # Adding element type (line 215)
        str_28054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 16), 'str', 'def __bootstrap__():')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, str_28054)
        # Adding element type (line 215)
        str_28055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 16), 'str', '   global __bootstrap__, __file__, __loader__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, str_28055)
        # Adding element type (line 215)
        str_28056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 16), 'str', '   import sys, os, pkg_resources, imp')
        
        # Call to if_dl(...): (line 218)
        # Processing the call arguments (line 218)
        str_28058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 62), 'str', ', dl')
        # Processing the call keyword arguments (line 218)
        kwargs_28059 = {}
        # Getting the type of 'if_dl' (line 218)
        if_dl_28057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 56), 'if_dl', False)
        # Calling if_dl(args, kwargs) (line 218)
        if_dl_call_result_28060 = invoke(stypy.reporting.localization.Localization(__file__, 218, 56), if_dl_28057, *[str_28058], **kwargs_28059)
        
        # Applying the binary operator '+' (line 218)
        result_add_28061 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 16), '+', str_28056, if_dl_call_result_28060)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, result_add_28061)
        # Adding element type (line 215)
        str_28062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 16), 'str', '   __file__ = pkg_resources.resource_filename(__name__,%r)')
        
        # Call to basename(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'ext' (line 220)
        ext_28066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'ext', False)
        # Obtaining the member '_file_name' of a type (line 220)
        _file_name_28067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 38), ext_28066, '_file_name')
        # Processing the call keyword arguments (line 220)
        kwargs_28068 = {}
        # Getting the type of 'os' (line 220)
        os_28063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 220)
        path_28064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 21), os_28063, 'path')
        # Obtaining the member 'basename' of a type (line 220)
        basename_28065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 21), path_28064, 'basename')
        # Calling basename(args, kwargs) (line 220)
        basename_call_result_28069 = invoke(stypy.reporting.localization.Localization(__file__, 220, 21), basename_28065, *[_file_name_28067], **kwargs_28068)
        
        # Applying the binary operator '%' (line 219)
        result_mod_28070 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 16), '%', str_28062, basename_call_result_28069)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, result_mod_28070)
        # Adding element type (line 215)
        str_28071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 16), 'str', '   del __bootstrap__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, str_28071)
        # Adding element type (line 215)
        str_28072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 16), 'str', "   if '__loader__' in globals():")
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, str_28072)
        # Adding element type (line 215)
        str_28073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 16), 'str', '       del __loader__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, str_28073)
        # Adding element type (line 215)
        
        # Call to if_dl(...): (line 224)
        # Processing the call arguments (line 224)
        str_28075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 22), 'str', '   old_flags = sys.getdlopenflags()')
        # Processing the call keyword arguments (line 224)
        kwargs_28076 = {}
        # Getting the type of 'if_dl' (line 224)
        if_dl_28074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'if_dl', False)
        # Calling if_dl(args, kwargs) (line 224)
        if_dl_call_result_28077 = invoke(stypy.reporting.localization.Localization(__file__, 224, 16), if_dl_28074, *[str_28075], **kwargs_28076)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, if_dl_call_result_28077)
        # Adding element type (line 215)
        str_28078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 16), 'str', '   old_dir = os.getcwd()')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, str_28078)
        # Adding element type (line 215)
        str_28079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 16), 'str', '   try:')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, str_28079)
        # Adding element type (line 215)
        str_28080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 16), 'str', '     os.chdir(os.path.dirname(__file__))')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, str_28080)
        # Adding element type (line 215)
        
        # Call to if_dl(...): (line 228)
        # Processing the call arguments (line 228)
        str_28082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 22), 'str', '     sys.setdlopenflags(dl.RTLD_NOW)')
        # Processing the call keyword arguments (line 228)
        kwargs_28083 = {}
        # Getting the type of 'if_dl' (line 228)
        if_dl_28081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'if_dl', False)
        # Calling if_dl(args, kwargs) (line 228)
        if_dl_call_result_28084 = invoke(stypy.reporting.localization.Localization(__file__, 228, 16), if_dl_28081, *[str_28082], **kwargs_28083)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, if_dl_call_result_28084)
        # Adding element type (line 215)
        str_28085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 16), 'str', '     imp.load_dynamic(__name__,__file__)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, str_28085)
        # Adding element type (line 215)
        str_28086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 16), 'str', '   finally:')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, str_28086)
        # Adding element type (line 215)
        
        # Call to if_dl(...): (line 231)
        # Processing the call arguments (line 231)
        str_28088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 22), 'str', '     sys.setdlopenflags(old_flags)')
        # Processing the call keyword arguments (line 231)
        kwargs_28089 = {}
        # Getting the type of 'if_dl' (line 231)
        if_dl_28087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'if_dl', False)
        # Calling if_dl(args, kwargs) (line 231)
        if_dl_call_result_28090 = invoke(stypy.reporting.localization.Localization(__file__, 231, 16), if_dl_28087, *[str_28088], **kwargs_28089)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, if_dl_call_result_28090)
        # Adding element type (line 215)
        str_28091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 16), 'str', '     os.chdir(old_dir)')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, str_28091)
        # Adding element type (line 215)
        str_28092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 16), 'str', '__bootstrap__()')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, str_28092)
        # Adding element type (line 215)
        str_28093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 16), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 30), list_28053, str_28093)
        
        # Processing the call keyword arguments (line 215)
        kwargs_28094 = {}
        str_28051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 20), 'str', '\n')
        # Obtaining the member 'join' of a type (line 215)
        join_28052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 20), str_28051, 'join')
        # Calling join(args, kwargs) (line 215)
        join_call_result_28095 = invoke(stypy.reporting.localization.Localization(__file__, 215, 20), join_28052, *[list_28053], **kwargs_28094)
        
        # Processing the call keyword arguments (line 215)
        kwargs_28096 = {}
        # Getting the type of 'f' (line 215)
        f_28049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 215)
        write_28050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 12), f_28049, 'write')
        # Calling write(args, kwargs) (line 215)
        write_call_result_28097 = invoke(stypy.reporting.localization.Localization(__file__, 215, 12), write_28050, *[join_call_result_28095], **kwargs_28096)
        
        
        # Call to close(...): (line 236)
        # Processing the call keyword arguments (line 236)
        kwargs_28100 = {}
        # Getting the type of 'f' (line 236)
        f_28098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 236)
        close_28099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 12), f_28098, 'close')
        # Calling close(args, kwargs) (line 236)
        close_call_result_28101 = invoke(stypy.reporting.localization.Localization(__file__, 236, 12), close_28099, *[], **kwargs_28100)
        
        # SSA join for if statement (line 213)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'compile' (line 237)
        compile_28102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 11), 'compile')
        # Testing the type of an if condition (line 237)
        if_condition_28103 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 8), compile_28102)
        # Assigning a type to the variable 'if_condition_28103' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'if_condition_28103', if_condition_28103)
        # SSA begins for if statement (line 237)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 238, 12))
        
        # 'from distutils.util import byte_compile' statement (line 238)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
        import_28104 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 238, 12), 'distutils.util')

        if (type(import_28104) is not StypyTypeError):

            if (import_28104 != 'pyd_module'):
                __import__(import_28104)
                sys_modules_28105 = sys.modules[import_28104]
                import_from_module(stypy.reporting.localization.Localization(__file__, 238, 12), 'distutils.util', sys_modules_28105.module_type_store, module_type_store, ['byte_compile'])
                nest_module(stypy.reporting.localization.Localization(__file__, 238, 12), __file__, sys_modules_28105, sys_modules_28105.module_type_store, module_type_store)
            else:
                from distutils.util import byte_compile

                import_from_module(stypy.reporting.localization.Localization(__file__, 238, 12), 'distutils.util', None, module_type_store, ['byte_compile'], [byte_compile])

        else:
            # Assigning a type to the variable 'distutils.util' (line 238)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'distutils.util', import_28104)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')
        
        
        # Call to byte_compile(...): (line 239)
        # Processing the call arguments (line 239)
        
        # Obtaining an instance of the builtin type 'list' (line 239)
        list_28107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 239)
        # Adding element type (line 239)
        # Getting the type of 'stub_file' (line 239)
        stub_file_28108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 26), 'stub_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 25), list_28107, stub_file_28108)
        
        # Processing the call keyword arguments (line 239)
        int_28109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 47), 'int')
        keyword_28110 = int_28109
        # Getting the type of 'True' (line 240)
        True_28111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 31), 'True', False)
        keyword_28112 = True_28111
        # Getting the type of 'self' (line 240)
        self_28113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 45), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 240)
        dry_run_28114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 45), self_28113, 'dry_run')
        keyword_28115 = dry_run_28114
        kwargs_28116 = {'force': keyword_28112, 'optimize': keyword_28110, 'dry_run': keyword_28115}
        # Getting the type of 'byte_compile' (line 239)
        byte_compile_28106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'byte_compile', False)
        # Calling byte_compile(args, kwargs) (line 239)
        byte_compile_call_result_28117 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), byte_compile_28106, *[list_28107], **kwargs_28116)
        
        
        # Assigning a Attribute to a Name (line 241):
        
        # Assigning a Attribute to a Name (line 241):
        
        # Call to get_finalized_command(...): (line 241)
        # Processing the call arguments (line 241)
        str_28120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 50), 'str', 'install_lib')
        # Processing the call keyword arguments (line 241)
        kwargs_28121 = {}
        # Getting the type of 'self' (line 241)
        self_28118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 23), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 241)
        get_finalized_command_28119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 23), self_28118, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 241)
        get_finalized_command_call_result_28122 = invoke(stypy.reporting.localization.Localization(__file__, 241, 23), get_finalized_command_28119, *[str_28120], **kwargs_28121)
        
        # Obtaining the member 'optimize' of a type (line 241)
        optimize_28123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 23), get_finalized_command_call_result_28122, 'optimize')
        # Assigning a type to the variable 'optimize' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'optimize', optimize_28123)
        
        
        # Getting the type of 'optimize' (line 242)
        optimize_28124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 15), 'optimize')
        int_28125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 26), 'int')
        # Applying the binary operator '>' (line 242)
        result_gt_28126 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 15), '>', optimize_28124, int_28125)
        
        # Testing the type of an if condition (line 242)
        if_condition_28127 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 12), result_gt_28126)
        # Assigning a type to the variable 'if_condition_28127' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'if_condition_28127', if_condition_28127)
        # SSA begins for if statement (line 242)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to byte_compile(...): (line 243)
        # Processing the call arguments (line 243)
        
        # Obtaining an instance of the builtin type 'list' (line 243)
        list_28129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 243)
        # Adding element type (line 243)
        # Getting the type of 'stub_file' (line 243)
        stub_file_28130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 30), 'stub_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 29), list_28129, stub_file_28130)
        
        # Processing the call keyword arguments (line 243)
        # Getting the type of 'optimize' (line 243)
        optimize_28131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 51), 'optimize', False)
        keyword_28132 = optimize_28131
        # Getting the type of 'True' (line 244)
        True_28133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 35), 'True', False)
        keyword_28134 = True_28133
        # Getting the type of 'self' (line 244)
        self_28135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 49), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 244)
        dry_run_28136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 49), self_28135, 'dry_run')
        keyword_28137 = dry_run_28136
        kwargs_28138 = {'force': keyword_28134, 'optimize': keyword_28132, 'dry_run': keyword_28137}
        # Getting the type of 'byte_compile' (line 243)
        byte_compile_28128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'byte_compile', False)
        # Calling byte_compile(args, kwargs) (line 243)
        byte_compile_call_result_28139 = invoke(stypy.reporting.localization.Localization(__file__, 243, 16), byte_compile_28128, *[list_28129], **kwargs_28138)
        
        # SSA join for if statement (line 242)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Call to exists(...): (line 245)
        # Processing the call arguments (line 245)
        # Getting the type of 'stub_file' (line 245)
        stub_file_28143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 30), 'stub_file', False)
        # Processing the call keyword arguments (line 245)
        kwargs_28144 = {}
        # Getting the type of 'os' (line 245)
        os_28140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 245)
        path_28141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 15), os_28140, 'path')
        # Obtaining the member 'exists' of a type (line 245)
        exists_28142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 15), path_28141, 'exists')
        # Calling exists(args, kwargs) (line 245)
        exists_call_result_28145 = invoke(stypy.reporting.localization.Localization(__file__, 245, 15), exists_28142, *[stub_file_28143], **kwargs_28144)
        
        
        # Getting the type of 'self' (line 245)
        self_28146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 49), 'self')
        # Obtaining the member 'dry_run' of a type (line 245)
        dry_run_28147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 49), self_28146, 'dry_run')
        # Applying the 'not' unary operator (line 245)
        result_not__28148 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 45), 'not', dry_run_28147)
        
        # Applying the binary operator 'and' (line 245)
        result_and_keyword_28149 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 15), 'and', exists_call_result_28145, result_not__28148)
        
        # Testing the type of an if condition (line 245)
        if_condition_28150 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 12), result_and_keyword_28149)
        # Assigning a type to the variable 'if_condition_28150' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'if_condition_28150', if_condition_28150)
        # SSA begins for if statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to unlink(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'stub_file' (line 246)
        stub_file_28153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 26), 'stub_file', False)
        # Processing the call keyword arguments (line 246)
        kwargs_28154 = {}
        # Getting the type of 'os' (line 246)
        os_28151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'os', False)
        # Obtaining the member 'unlink' of a type (line 246)
        unlink_28152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 16), os_28151, 'unlink')
        # Calling unlink(args, kwargs) (line 246)
        unlink_call_result_28155 = invoke(stypy.reporting.localization.Localization(__file__, 246, 16), unlink_28152, *[stub_file_28153], **kwargs_28154)
        
        # SSA join for if statement (line 245)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 237)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'write_stub(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_stub' in the type store
        # Getting the type of 'stypy_return_type' (line 208)
        stypy_return_type_28156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28156)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_stub'
        return stypy_return_type_28156


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 44, 0, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'self', type_of_self)
        
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


# Assigning a type to the variable 'build_ext' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'build_ext', build_ext)


# Evaluating a boolean operation

# Getting the type of '_build_ext' (line 75)
_build_ext_28157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 7), '_build_ext')
# Getting the type of '_du_build_ext' (line 75)
_du_build_ext_28158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 25), '_du_build_ext')
# Applying the binary operator 'isnot' (line 75)
result_is_not_28159 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 7), 'isnot', _build_ext_28157, _du_build_ext_28158)



# Call to hasattr(...): (line 75)
# Processing the call arguments (line 75)
# Getting the type of '_build_ext' (line 75)
_build_ext_28161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 55), '_build_ext', False)
str_28162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 66), 'str', 'pyrex_sources')
# Processing the call keyword arguments (line 75)
kwargs_28163 = {}
# Getting the type of 'hasattr' (line 75)
hasattr_28160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 47), 'hasattr', False)
# Calling hasattr(args, kwargs) (line 75)
hasattr_call_result_28164 = invoke(stypy.reporting.localization.Localization(__file__, 75, 47), hasattr_28160, *[_build_ext_28161, str_28162], **kwargs_28163)

# Applying the 'not' unary operator (line 75)
result_not__28165 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 43), 'not', hasattr_call_result_28164)

# Applying the binary operator 'and' (line 75)
result_and_keyword_28166 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 7), 'and', result_is_not_28159, result_not__28165)

# Testing the type of an if condition (line 75)
if_condition_28167 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 4), result_and_keyword_28166)
# Assigning a type to the variable 'if_condition_28167' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'if_condition_28167', if_condition_28167)
# SSA begins for if statement (line 75)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def swig_sources(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'swig_sources'
    module_type_store = module_type_store.open_function_context('swig_sources', 77, 8, False)
    
    # Passed parameters checking function
    swig_sources.stypy_localization = localization
    swig_sources.stypy_type_of_self = None
    swig_sources.stypy_type_store = module_type_store
    swig_sources.stypy_function_name = 'swig_sources'
    swig_sources.stypy_param_names_list = ['self', 'sources']
    swig_sources.stypy_varargs_param_name = 'otherargs'
    swig_sources.stypy_kwargs_param_name = None
    swig_sources.stypy_call_defaults = defaults
    swig_sources.stypy_call_varargs = varargs
    swig_sources.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'swig_sources', ['self', 'sources'], 'otherargs', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'swig_sources', localization, ['self', 'sources'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'swig_sources(...)' code ##################

    
    # Assigning a BoolOp to a Name (line 79):
    
    # Assigning a BoolOp to a Name (line 79):
    
    # Evaluating a boolean operation
    
    # Call to swig_sources(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'self' (line 79)
    self_28170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 46), 'self', False)
    # Getting the type of 'sources' (line 79)
    sources_28171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 52), 'sources', False)
    # Processing the call keyword arguments (line 79)
    kwargs_28172 = {}
    # Getting the type of '_build_ext' (line 79)
    _build_ext_28168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 22), '_build_ext', False)
    # Obtaining the member 'swig_sources' of a type (line 79)
    swig_sources_28169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 22), _build_ext_28168, 'swig_sources')
    # Calling swig_sources(args, kwargs) (line 79)
    swig_sources_call_result_28173 = invoke(stypy.reporting.localization.Localization(__file__, 79, 22), swig_sources_28169, *[self_28170, sources_28171], **kwargs_28172)
    
    # Getting the type of 'sources' (line 79)
    sources_28174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 64), 'sources')
    # Applying the binary operator 'or' (line 79)
    result_or_keyword_28175 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 22), 'or', swig_sources_call_result_28173, sources_28174)
    
    # Assigning a type to the variable 'sources' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'sources', result_or_keyword_28175)
    
    # Call to swig_sources(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'self' (line 81)
    self_28178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 46), 'self', False)
    # Getting the type of 'sources' (line 81)
    sources_28179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 52), 'sources', False)
    # Getting the type of 'otherargs' (line 81)
    otherargs_28180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 62), 'otherargs', False)
    # Processing the call keyword arguments (line 81)
    kwargs_28181 = {}
    # Getting the type of '_du_build_ext' (line 81)
    _du_build_ext_28176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 19), '_du_build_ext', False)
    # Obtaining the member 'swig_sources' of a type (line 81)
    swig_sources_28177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 19), _du_build_ext_28176, 'swig_sources')
    # Calling swig_sources(args, kwargs) (line 81)
    swig_sources_call_result_28182 = invoke(stypy.reporting.localization.Localization(__file__, 81, 19), swig_sources_28177, *[self_28178, sources_28179, otherargs_28180], **kwargs_28181)
    
    # Assigning a type to the variable 'stypy_return_type' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'stypy_return_type', swig_sources_call_result_28182)
    
    # ################# End of 'swig_sources(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'swig_sources' in the type store
    # Getting the type of 'stypy_return_type' (line 77)
    stypy_return_type_28183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28183)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'swig_sources'
    return stypy_return_type_28183

# Assigning a type to the variable 'swig_sources' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'swig_sources', swig_sources)
# SSA join for if statement (line 75)
module_type_store = module_type_store.join_ssa_context()



# Evaluating a boolean operation
# Getting the type of 'use_stubs' (line 249)
use_stubs_28184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 3), 'use_stubs')

# Getting the type of 'os' (line 249)
os_28185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 16), 'os')
# Obtaining the member 'name' of a type (line 249)
name_28186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 16), os_28185, 'name')
str_28187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 25), 'str', 'nt')
# Applying the binary operator '==' (line 249)
result_eq_28188 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 16), '==', name_28186, str_28187)

# Applying the binary operator 'or' (line 249)
result_or_keyword_28189 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 3), 'or', use_stubs_28184, result_eq_28188)

# Testing the type of an if condition (line 249)
if_condition_28190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 0), result_or_keyword_28189)
# Assigning a type to the variable 'if_condition_28190' (line 249)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 0), 'if_condition_28190', if_condition_28190)
# SSA begins for if statement (line 249)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def link_shared_object(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 252)
    None_28191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 69), 'None')
    # Getting the type of 'None' (line 253)
    None_28192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 18), 'None')
    # Getting the type of 'None' (line 253)
    None_28193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 37), 'None')
    # Getting the type of 'None' (line 253)
    None_28194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 64), 'None')
    # Getting the type of 'None' (line 254)
    None_28195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 23), 'None')
    int_28196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 35), 'int')
    # Getting the type of 'None' (line 254)
    None_28197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 52), 'None')
    # Getting the type of 'None' (line 255)
    None_28198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 23), 'None')
    # Getting the type of 'None' (line 255)
    None_28199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 40), 'None')
    # Getting the type of 'None' (line 255)
    None_28200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 58), 'None')
    defaults = [None_28191, None_28192, None_28193, None_28194, None_28195, int_28196, None_28197, None_28198, None_28199, None_28200]
    # Create a new context for function 'link_shared_object'
    module_type_store = module_type_store.open_function_context('link_shared_object', 252, 4, False)
    
    # Passed parameters checking function
    link_shared_object.stypy_localization = localization
    link_shared_object.stypy_type_of_self = None
    link_shared_object.stypy_type_store = module_type_store
    link_shared_object.stypy_function_name = 'link_shared_object'
    link_shared_object.stypy_param_names_list = ['self', 'objects', 'output_libname', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang']
    link_shared_object.stypy_varargs_param_name = None
    link_shared_object.stypy_kwargs_param_name = None
    link_shared_object.stypy_call_defaults = defaults
    link_shared_object.stypy_call_varargs = varargs
    link_shared_object.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'link_shared_object', ['self', 'objects', 'output_libname', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'link_shared_object', localization, ['self', 'objects', 'output_libname', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'link_shared_object(...)' code ##################

    
    # Call to link(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'self' (line 257)
    self_28203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 12), 'self', False)
    # Obtaining the member 'SHARED_LIBRARY' of a type (line 257)
    SHARED_LIBRARY_28204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 12), self_28203, 'SHARED_LIBRARY')
    # Getting the type of 'objects' (line 257)
    objects_28205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 33), 'objects', False)
    # Getting the type of 'output_libname' (line 257)
    output_libname_28206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 42), 'output_libname', False)
    # Getting the type of 'output_dir' (line 258)
    output_dir_28207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'output_dir', False)
    # Getting the type of 'libraries' (line 258)
    libraries_28208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 24), 'libraries', False)
    # Getting the type of 'library_dirs' (line 258)
    library_dirs_28209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 35), 'library_dirs', False)
    # Getting the type of 'runtime_library_dirs' (line 258)
    runtime_library_dirs_28210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 49), 'runtime_library_dirs', False)
    # Getting the type of 'export_symbols' (line 259)
    export_symbols_28211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'export_symbols', False)
    # Getting the type of 'debug' (line 259)
    debug_28212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 28), 'debug', False)
    # Getting the type of 'extra_preargs' (line 259)
    extra_preargs_28213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 35), 'extra_preargs', False)
    # Getting the type of 'extra_postargs' (line 259)
    extra_postargs_28214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 50), 'extra_postargs', False)
    # Getting the type of 'build_temp' (line 260)
    build_temp_28215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'build_temp', False)
    # Getting the type of 'target_lang' (line 260)
    target_lang_28216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 24), 'target_lang', False)
    # Processing the call keyword arguments (line 256)
    kwargs_28217 = {}
    # Getting the type of 'self' (line 256)
    self_28201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'self', False)
    # Obtaining the member 'link' of a type (line 256)
    link_28202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), self_28201, 'link')
    # Calling link(args, kwargs) (line 256)
    link_call_result_28218 = invoke(stypy.reporting.localization.Localization(__file__, 256, 8), link_28202, *[SHARED_LIBRARY_28204, objects_28205, output_libname_28206, output_dir_28207, libraries_28208, library_dirs_28209, runtime_library_dirs_28210, export_symbols_28211, debug_28212, extra_preargs_28213, extra_postargs_28214, build_temp_28215, target_lang_28216], **kwargs_28217)
    
    
    # ################# End of 'link_shared_object(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'link_shared_object' in the type store
    # Getting the type of 'stypy_return_type' (line 252)
    stypy_return_type_28219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28219)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'link_shared_object'
    return stypy_return_type_28219

# Assigning a type to the variable 'link_shared_object' (line 252)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'link_shared_object', link_shared_object)
# SSA branch for the else part of an if statement (line 249)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 264):

# Assigning a Str to a Name (line 264):
str_28220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 14), 'str', 'static')
# Assigning a type to the variable 'libtype' (line 264)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'libtype', str_28220)

@norecursion
def link_shared_object(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 266)
    None_28221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 69), 'None')
    # Getting the type of 'None' (line 267)
    None_28222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 18), 'None')
    # Getting the type of 'None' (line 267)
    None_28223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 37), 'None')
    # Getting the type of 'None' (line 267)
    None_28224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 64), 'None')
    # Getting the type of 'None' (line 268)
    None_28225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 23), 'None')
    int_28226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 35), 'int')
    # Getting the type of 'None' (line 268)
    None_28227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 52), 'None')
    # Getting the type of 'None' (line 269)
    None_28228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 23), 'None')
    # Getting the type of 'None' (line 269)
    None_28229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 40), 'None')
    # Getting the type of 'None' (line 269)
    None_28230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 58), 'None')
    defaults = [None_28221, None_28222, None_28223, None_28224, None_28225, int_28226, None_28227, None_28228, None_28229, None_28230]
    # Create a new context for function 'link_shared_object'
    module_type_store = module_type_store.open_function_context('link_shared_object', 266, 4, False)
    
    # Passed parameters checking function
    link_shared_object.stypy_localization = localization
    link_shared_object.stypy_type_of_self = None
    link_shared_object.stypy_type_store = module_type_store
    link_shared_object.stypy_function_name = 'link_shared_object'
    link_shared_object.stypy_param_names_list = ['self', 'objects', 'output_libname', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang']
    link_shared_object.stypy_varargs_param_name = None
    link_shared_object.stypy_kwargs_param_name = None
    link_shared_object.stypy_call_defaults = defaults
    link_shared_object.stypy_call_varargs = varargs
    link_shared_object.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'link_shared_object', ['self', 'objects', 'output_libname', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'link_shared_object', localization, ['self', 'objects', 'output_libname', 'output_dir', 'libraries', 'library_dirs', 'runtime_library_dirs', 'export_symbols', 'debug', 'extra_preargs', 'extra_postargs', 'build_temp', 'target_lang'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'link_shared_object(...)' code ##################

    # Evaluating assert statement condition
    
    # Getting the type of 'output_dir' (line 277)
    output_dir_28231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 15), 'output_dir')
    # Getting the type of 'None' (line 277)
    None_28232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 29), 'None')
    # Applying the binary operator 'is' (line 277)
    result_is__28233 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 15), 'is', output_dir_28231, None_28232)
    
    
    # Assigning a Call to a Tuple (line 278):
    
    # Assigning a Subscript to a Name (line 278):
    
    # Obtaining the type of the subscript
    int_28234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 8), 'int')
    
    # Call to split(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'output_libname' (line 278)
    output_libname_28238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 44), 'output_libname', False)
    # Processing the call keyword arguments (line 278)
    kwargs_28239 = {}
    # Getting the type of 'os' (line 278)
    os_28235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 30), 'os', False)
    # Obtaining the member 'path' of a type (line 278)
    path_28236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 30), os_28235, 'path')
    # Obtaining the member 'split' of a type (line 278)
    split_28237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 30), path_28236, 'split')
    # Calling split(args, kwargs) (line 278)
    split_call_result_28240 = invoke(stypy.reporting.localization.Localization(__file__, 278, 30), split_28237, *[output_libname_28238], **kwargs_28239)
    
    # Obtaining the member '__getitem__' of a type (line 278)
    getitem___28241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), split_call_result_28240, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 278)
    subscript_call_result_28242 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), getitem___28241, int_28234)
    
    # Assigning a type to the variable 'tuple_var_assignment_27345' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'tuple_var_assignment_27345', subscript_call_result_28242)
    
    # Assigning a Subscript to a Name (line 278):
    
    # Obtaining the type of the subscript
    int_28243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 8), 'int')
    
    # Call to split(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'output_libname' (line 278)
    output_libname_28247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 44), 'output_libname', False)
    # Processing the call keyword arguments (line 278)
    kwargs_28248 = {}
    # Getting the type of 'os' (line 278)
    os_28244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 30), 'os', False)
    # Obtaining the member 'path' of a type (line 278)
    path_28245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 30), os_28244, 'path')
    # Obtaining the member 'split' of a type (line 278)
    split_28246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 30), path_28245, 'split')
    # Calling split(args, kwargs) (line 278)
    split_call_result_28249 = invoke(stypy.reporting.localization.Localization(__file__, 278, 30), split_28246, *[output_libname_28247], **kwargs_28248)
    
    # Obtaining the member '__getitem__' of a type (line 278)
    getitem___28250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), split_call_result_28249, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 278)
    subscript_call_result_28251 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), getitem___28250, int_28243)
    
    # Assigning a type to the variable 'tuple_var_assignment_27346' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'tuple_var_assignment_27346', subscript_call_result_28251)
    
    # Assigning a Name to a Name (line 278):
    # Getting the type of 'tuple_var_assignment_27345' (line 278)
    tuple_var_assignment_27345_28252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'tuple_var_assignment_27345')
    # Assigning a type to the variable 'output_dir' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'output_dir', tuple_var_assignment_27345_28252)
    
    # Assigning a Name to a Name (line 278):
    # Getting the type of 'tuple_var_assignment_27346' (line 278)
    tuple_var_assignment_27346_28253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'tuple_var_assignment_27346')
    # Assigning a type to the variable 'filename' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'filename', tuple_var_assignment_27346_28253)
    
    # Assigning a Call to a Tuple (line 279):
    
    # Assigning a Subscript to a Name (line 279):
    
    # Obtaining the type of the subscript
    int_28254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 8), 'int')
    
    # Call to splitext(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'filename' (line 279)
    filename_28258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 41), 'filename', False)
    # Processing the call keyword arguments (line 279)
    kwargs_28259 = {}
    # Getting the type of 'os' (line 279)
    os_28255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 279)
    path_28256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 24), os_28255, 'path')
    # Obtaining the member 'splitext' of a type (line 279)
    splitext_28257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 24), path_28256, 'splitext')
    # Calling splitext(args, kwargs) (line 279)
    splitext_call_result_28260 = invoke(stypy.reporting.localization.Localization(__file__, 279, 24), splitext_28257, *[filename_28258], **kwargs_28259)
    
    # Obtaining the member '__getitem__' of a type (line 279)
    getitem___28261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 8), splitext_call_result_28260, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 279)
    subscript_call_result_28262 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), getitem___28261, int_28254)
    
    # Assigning a type to the variable 'tuple_var_assignment_27347' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'tuple_var_assignment_27347', subscript_call_result_28262)
    
    # Assigning a Subscript to a Name (line 279):
    
    # Obtaining the type of the subscript
    int_28263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 8), 'int')
    
    # Call to splitext(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'filename' (line 279)
    filename_28267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 41), 'filename', False)
    # Processing the call keyword arguments (line 279)
    kwargs_28268 = {}
    # Getting the type of 'os' (line 279)
    os_28264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 279)
    path_28265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 24), os_28264, 'path')
    # Obtaining the member 'splitext' of a type (line 279)
    splitext_28266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 24), path_28265, 'splitext')
    # Calling splitext(args, kwargs) (line 279)
    splitext_call_result_28269 = invoke(stypy.reporting.localization.Localization(__file__, 279, 24), splitext_28266, *[filename_28267], **kwargs_28268)
    
    # Obtaining the member '__getitem__' of a type (line 279)
    getitem___28270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 8), splitext_call_result_28269, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 279)
    subscript_call_result_28271 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), getitem___28270, int_28263)
    
    # Assigning a type to the variable 'tuple_var_assignment_27348' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'tuple_var_assignment_27348', subscript_call_result_28271)
    
    # Assigning a Name to a Name (line 279):
    # Getting the type of 'tuple_var_assignment_27347' (line 279)
    tuple_var_assignment_27347_28272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'tuple_var_assignment_27347')
    # Assigning a type to the variable 'basename' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'basename', tuple_var_assignment_27347_28272)
    
    # Assigning a Name to a Name (line 279):
    # Getting the type of 'tuple_var_assignment_27348' (line 279)
    tuple_var_assignment_27348_28273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'tuple_var_assignment_27348')
    # Assigning a type to the variable 'ext' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 18), 'ext', tuple_var_assignment_27348_28273)
    
    
    # Call to startswith(...): (line 280)
    # Processing the call arguments (line 280)
    str_28280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 49), 'str', 'lib')
    # Processing the call keyword arguments (line 280)
    kwargs_28281 = {}
    
    # Call to library_filename(...): (line 280)
    # Processing the call arguments (line 280)
    str_28276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 33), 'str', 'x')
    # Processing the call keyword arguments (line 280)
    kwargs_28277 = {}
    # Getting the type of 'self' (line 280)
    self_28274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 11), 'self', False)
    # Obtaining the member 'library_filename' of a type (line 280)
    library_filename_28275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 11), self_28274, 'library_filename')
    # Calling library_filename(args, kwargs) (line 280)
    library_filename_call_result_28278 = invoke(stypy.reporting.localization.Localization(__file__, 280, 11), library_filename_28275, *[str_28276], **kwargs_28277)
    
    # Obtaining the member 'startswith' of a type (line 280)
    startswith_28279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 11), library_filename_call_result_28278, 'startswith')
    # Calling startswith(args, kwargs) (line 280)
    startswith_call_result_28282 = invoke(stypy.reporting.localization.Localization(__file__, 280, 11), startswith_28279, *[str_28280], **kwargs_28281)
    
    # Testing the type of an if condition (line 280)
    if_condition_28283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 280, 8), startswith_call_result_28282)
    # Assigning a type to the variable 'if_condition_28283' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'if_condition_28283', if_condition_28283)
    # SSA begins for if statement (line 280)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 283):
    
    # Assigning a Subscript to a Name (line 283):
    
    # Obtaining the type of the subscript
    int_28284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 32), 'int')
    slice_28285 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 283, 23), int_28284, None, None)
    # Getting the type of 'basename' (line 283)
    basename_28286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 23), 'basename')
    # Obtaining the member '__getitem__' of a type (line 283)
    getitem___28287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 23), basename_28286, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 283)
    subscript_call_result_28288 = invoke(stypy.reporting.localization.Localization(__file__, 283, 23), getitem___28287, slice_28285)
    
    # Assigning a type to the variable 'basename' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'basename', subscript_call_result_28288)
    # SSA join for if statement (line 280)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to create_static_lib(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'objects' (line 286)
    objects_28291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'objects', False)
    # Getting the type of 'basename' (line 286)
    basename_28292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 21), 'basename', False)
    # Getting the type of 'output_dir' (line 286)
    output_dir_28293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 31), 'output_dir', False)
    # Getting the type of 'debug' (line 286)
    debug_28294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 43), 'debug', False)
    # Getting the type of 'target_lang' (line 286)
    target_lang_28295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 50), 'target_lang', False)
    # Processing the call keyword arguments (line 285)
    kwargs_28296 = {}
    # Getting the type of 'self' (line 285)
    self_28289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'self', False)
    # Obtaining the member 'create_static_lib' of a type (line 285)
    create_static_lib_28290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), self_28289, 'create_static_lib')
    # Calling create_static_lib(args, kwargs) (line 285)
    create_static_lib_call_result_28297 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), create_static_lib_28290, *[objects_28291, basename_28292, output_dir_28293, debug_28294, target_lang_28295], **kwargs_28296)
    
    
    # ################# End of 'link_shared_object(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'link_shared_object' in the type store
    # Getting the type of 'stypy_return_type' (line 266)
    stypy_return_type_28298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28298)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'link_shared_object'
    return stypy_return_type_28298

# Assigning a type to the variable 'link_shared_object' (line 266)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'link_shared_object', link_shared_object)
# SSA join for if statement (line 249)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
