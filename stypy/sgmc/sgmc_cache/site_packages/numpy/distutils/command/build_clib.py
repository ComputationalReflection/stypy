
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Modified version of build_clib that handles fortran source files.
2: '''
3: from __future__ import division, absolute_import, print_function
4: 
5: import os
6: from glob import glob
7: import shutil
8: from distutils.command.build_clib import build_clib as old_build_clib
9: from distutils.errors import DistutilsSetupError, DistutilsError, \
10:      DistutilsFileError
11: 
12: from numpy.distutils import log
13: from distutils.dep_util import newer_group
14: from numpy.distutils.misc_util import filter_sources, has_f_sources,\
15:      has_cxx_sources, all_strings, get_lib_source_files, is_sequence, \
16:      get_numpy_include_dirs
17: 
18: # Fix Python distutils bug sf #1718574:
19: _l = old_build_clib.user_options
20: for _i in range(len(_l)):
21:     if _l[_i][0] in ['build-clib', 'build-temp']:
22:         _l[_i] = (_l[_i][0]+'=',)+_l[_i][1:]
23: #
24: 
25: class build_clib(old_build_clib):
26: 
27:     description = "build C/C++/F libraries used by Python extensions"
28: 
29:     user_options = old_build_clib.user_options + [
30:         ('fcompiler=', None,
31:          "specify the Fortran compiler type"),
32:         ('inplace', 'i', 'Build in-place'),
33:         ('parallel=', 'j',
34:          "number of parallel jobs"),
35:         ]
36: 
37:     boolean_options = old_build_clib.boolean_options + ['inplace']
38: 
39:     def initialize_options(self):
40:         old_build_clib.initialize_options(self)
41:         self.fcompiler = None
42:         self.inplace = 0
43:         self.parallel = None
44: 
45:     def finalize_options(self):
46:         if self.parallel:
47:             try:
48:                 self.parallel = int(self.parallel)
49:             except ValueError:
50:                 raise ValueError("--parallel/-j argument must be an integer")
51:         old_build_clib.finalize_options(self)
52:         self.set_undefined_options('build', ('parallel', 'parallel'))
53: 
54:     def have_f_sources(self):
55:         for (lib_name, build_info) in self.libraries:
56:             if has_f_sources(build_info.get('sources', [])):
57:                 return True
58:         return False
59: 
60:     def have_cxx_sources(self):
61:         for (lib_name, build_info) in self.libraries:
62:             if has_cxx_sources(build_info.get('sources', [])):
63:                 return True
64:         return False
65: 
66:     def run(self):
67:         if not self.libraries:
68:             return
69: 
70:         # Make sure that library sources are complete.
71:         languages = []
72: 
73:         # Make sure that extension sources are complete.
74:         self.run_command('build_src')
75: 
76:         for (lib_name, build_info) in self.libraries:
77:             l = build_info.get('language', None)
78:             if l and l not in languages: languages.append(l)
79: 
80:         from distutils.ccompiler import new_compiler
81:         self.compiler = new_compiler(compiler=self.compiler,
82:                                      dry_run=self.dry_run,
83:                                      force=self.force)
84:         self.compiler.customize(self.distribution,
85:                                 need_cxx=self.have_cxx_sources())
86: 
87:         libraries = self.libraries
88:         self.libraries = None
89:         self.compiler.customize_cmd(self)
90:         self.libraries = libraries
91: 
92:         self.compiler.show_customization()
93: 
94:         if self.have_f_sources():
95:             from numpy.distutils.fcompiler import new_fcompiler
96:             self._f_compiler = new_fcompiler(compiler=self.fcompiler,
97:                                                verbose=self.verbose,
98:                                                dry_run=self.dry_run,
99:                                                force=self.force,
100:                                                requiref90='f90' in languages,
101:                                                c_compiler=self.compiler)
102:             if self._f_compiler is not None:
103:                 self._f_compiler.customize(self.distribution)
104: 
105:                 libraries = self.libraries
106:                 self.libraries = None
107:                 self._f_compiler.customize_cmd(self)
108:                 self.libraries = libraries
109: 
110:                 self._f_compiler.show_customization()
111:         else:
112:             self._f_compiler = None
113: 
114:         self.build_libraries(self.libraries)
115: 
116:         if self.inplace:
117:             for l in  self.distribution.installed_libraries:
118:                 libname = self.compiler.library_filename(l.name)
119:                 source = os.path.join(self.build_clib, libname)
120:                 target =  os.path.join(l.target_dir, libname)
121:                 self.mkpath(l.target_dir)
122:                 shutil.copy(source, target)
123: 
124:     def get_source_files(self):
125:         self.check_library_list(self.libraries)
126:         filenames = []
127:         for lib in self.libraries:
128:             filenames.extend(get_lib_source_files(lib))
129:         return filenames
130: 
131:     def build_libraries(self, libraries):
132:         for (lib_name, build_info) in libraries:
133:             self.build_a_library(build_info, lib_name, libraries)
134: 
135:     def build_a_library(self, build_info, lib_name, libraries):
136:         # default compilers
137:         compiler = self.compiler
138:         fcompiler = self._f_compiler
139: 
140:         sources = build_info.get('sources')
141:         if sources is None or not is_sequence(sources):
142:             raise DistutilsSetupError(("in 'libraries' option (library '%s'), " +
143:                    "'sources' must be present and must be " +
144:                    "a list of source filenames") % lib_name)
145:         sources = list(sources)
146: 
147:         c_sources, cxx_sources, f_sources, fmodule_sources \
148:                    = filter_sources(sources)
149:         requiref90 = not not fmodule_sources or \
150:                      build_info.get('language', 'c')=='f90'
151: 
152:         # save source type information so that build_ext can use it.
153:         source_languages = []
154:         if c_sources: source_languages.append('c')
155:         if cxx_sources: source_languages.append('c++')
156:         if requiref90: source_languages.append('f90')
157:         elif f_sources: source_languages.append('f77')
158:         build_info['source_languages'] = source_languages
159: 
160:         lib_file = compiler.library_filename(lib_name,
161:                                              output_dir=self.build_clib)
162:         depends = sources + build_info.get('depends', [])
163:         if not (self.force or newer_group(depends, lib_file, 'newer')):
164:             log.debug("skipping '%s' library (up-to-date)", lib_name)
165:             return
166:         else:
167:             log.info("building '%s' library", lib_name)
168: 
169:         config_fc = build_info.get('config_fc', {})
170:         if fcompiler is not None and config_fc:
171:             log.info('using additional config_fc from setup script '\
172:                      'for fortran compiler: %s' \
173:                      % (config_fc,))
174:             from numpy.distutils.fcompiler import new_fcompiler
175:             fcompiler = new_fcompiler(compiler=fcompiler.compiler_type,
176:                                       verbose=self.verbose,
177:                                       dry_run=self.dry_run,
178:                                       force=self.force,
179:                                       requiref90=requiref90,
180:                                       c_compiler=self.compiler)
181:             if fcompiler is not None:
182:                 dist = self.distribution
183:                 base_config_fc = dist.get_option_dict('config_fc').copy()
184:                 base_config_fc.update(config_fc)
185:                 fcompiler.customize(base_config_fc)
186: 
187:         # check availability of Fortran compilers
188:         if (f_sources or fmodule_sources) and fcompiler is None:
189:             raise DistutilsError("library %s has Fortran sources"\
190:                   " but no Fortran compiler found" % (lib_name))
191: 
192:         if fcompiler is not None:
193:             fcompiler.extra_f77_compile_args = build_info.get('extra_f77_compile_args') or []
194:             fcompiler.extra_f90_compile_args = build_info.get('extra_f90_compile_args') or []
195: 
196:         macros = build_info.get('macros')
197:         include_dirs = build_info.get('include_dirs')
198:         if include_dirs is None:
199:             include_dirs = []
200:         extra_postargs = build_info.get('extra_compiler_args') or []
201: 
202:         include_dirs.extend(get_numpy_include_dirs())
203:         # where compiled F90 module files are:
204:         module_dirs = build_info.get('module_dirs') or []
205:         module_build_dir = os.path.dirname(lib_file)
206:         if requiref90: self.mkpath(module_build_dir)
207: 
208:         if compiler.compiler_type=='msvc':
209:             # this hack works around the msvc compiler attributes
210:             # problem, msvc uses its own convention :(
211:             c_sources += cxx_sources
212:             cxx_sources = []
213: 
214:         objects = []
215:         if c_sources:
216:             log.info("compiling C sources")
217:             objects = compiler.compile(c_sources,
218:                                        output_dir=self.build_temp,
219:                                        macros=macros,
220:                                        include_dirs=include_dirs,
221:                                        debug=self.debug,
222:                                        extra_postargs=extra_postargs)
223: 
224:         if cxx_sources:
225:             log.info("compiling C++ sources")
226:             cxx_compiler = compiler.cxx_compiler()
227:             cxx_objects = cxx_compiler.compile(cxx_sources,
228:                                                output_dir=self.build_temp,
229:                                                macros=macros,
230:                                                include_dirs=include_dirs,
231:                                                debug=self.debug,
232:                                                extra_postargs=extra_postargs)
233:             objects.extend(cxx_objects)
234: 
235:         if f_sources or fmodule_sources:
236:             extra_postargs = []
237:             f_objects = []
238: 
239:             if requiref90:
240:                 if fcompiler.module_dir_switch is None:
241:                     existing_modules = glob('*.mod')
242:                 extra_postargs += fcompiler.module_options(\
243:                     module_dirs, module_build_dir)
244: 
245:             if fmodule_sources:
246:                 log.info("compiling Fortran 90 module sources")
247:                 f_objects += fcompiler.compile(fmodule_sources,
248:                                                output_dir=self.build_temp,
249:                                                macros=macros,
250:                                                include_dirs=include_dirs,
251:                                                debug=self.debug,
252:                                                extra_postargs=extra_postargs)
253: 
254:             if requiref90 and self._f_compiler.module_dir_switch is None:
255:                 # move new compiled F90 module files to module_build_dir
256:                 for f in glob('*.mod'):
257:                     if f in existing_modules:
258:                         continue
259:                     t = os.path.join(module_build_dir, f)
260:                     if os.path.abspath(f)==os.path.abspath(t):
261:                         continue
262:                     if os.path.isfile(t):
263:                         os.remove(t)
264:                     try:
265:                         self.move_file(f, module_build_dir)
266:                     except DistutilsFileError:
267:                         log.warn('failed to move %r to %r' \
268:                                  % (f, module_build_dir))
269: 
270:             if f_sources:
271:                 log.info("compiling Fortran sources")
272:                 f_objects += fcompiler.compile(f_sources,
273:                                                output_dir=self.build_temp,
274:                                                macros=macros,
275:                                                include_dirs=include_dirs,
276:                                                debug=self.debug,
277:                                                extra_postargs=extra_postargs)
278:         else:
279:             f_objects = []
280: 
281:         objects.extend(f_objects)
282: 
283:         # assume that default linker is suitable for
284:         # linking Fortran object files
285:         compiler.create_static_lib(objects, lib_name,
286:                                    output_dir=self.build_clib,
287:                                    debug=self.debug)
288: 
289:         # fix library dependencies
290:         clib_libraries = build_info.get('libraries', [])
291:         for lname, binfo in libraries:
292:             if lname in clib_libraries:
293:                 clib_libraries.extend(binfo.get('libraries', []))
294:         if clib_libraries:
295:             build_info['libraries'] = clib_libraries
296: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_52551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', ' Modified version of build_clib that handles fortran source files.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import os' statement (line 5)
import os

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from glob import glob' statement (line 6)
from glob import glob

import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'glob', None, module_type_store, ['glob'], [glob])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import shutil' statement (line 7)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'shutil', shutil, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.command.build_clib import old_build_clib' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_52552 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.command.build_clib')

if (type(import_52552) is not StypyTypeError):

    if (import_52552 != 'pyd_module'):
        __import__(import_52552)
        sys_modules_52553 = sys.modules[import_52552]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.command.build_clib', sys_modules_52553.module_type_store, module_type_store, ['build_clib'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_52553, sys_modules_52553.module_type_store, module_type_store)
    else:
        from distutils.command.build_clib import build_clib as old_build_clib

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.command.build_clib', None, module_type_store, ['build_clib'], [old_build_clib])

else:
    # Assigning a type to the variable 'distutils.command.build_clib' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.command.build_clib', import_52552)

# Adding an alias
module_type_store.add_alias('old_build_clib', 'build_clib')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.errors import DistutilsSetupError, DistutilsError, DistutilsFileError' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_52554 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors')

if (type(import_52554) is not StypyTypeError):

    if (import_52554 != 'pyd_module'):
        __import__(import_52554)
        sys_modules_52555 = sys.modules[import_52554]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', sys_modules_52555.module_type_store, module_type_store, ['DistutilsSetupError', 'DistutilsError', 'DistutilsFileError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_52555, sys_modules_52555.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsSetupError, DistutilsError, DistutilsFileError

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', None, module_type_store, ['DistutilsSetupError', 'DistutilsError', 'DistutilsFileError'], [DistutilsSetupError, DistutilsError, DistutilsFileError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', import_52554)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.distutils import log' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_52556 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils')

if (type(import_52556) is not StypyTypeError):

    if (import_52556 != 'pyd_module'):
        __import__(import_52556)
        sys_modules_52557 = sys.modules[import_52556]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils', sys_modules_52557.module_type_store, module_type_store, ['log'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_52557, sys_modules_52557.module_type_store, module_type_store)
    else:
        from numpy.distutils import log

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils', None, module_type_store, ['log'], [log])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils', import_52556)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.dep_util import newer_group' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_52558 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.dep_util')

if (type(import_52558) is not StypyTypeError):

    if (import_52558 != 'pyd_module'):
        __import__(import_52558)
        sys_modules_52559 = sys.modules[import_52558]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.dep_util', sys_modules_52559.module_type_store, module_type_store, ['newer_group'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_52559, sys_modules_52559.module_type_store, module_type_store)
    else:
        from distutils.dep_util import newer_group

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.dep_util', None, module_type_store, ['newer_group'], [newer_group])

else:
    # Assigning a type to the variable 'distutils.dep_util' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.dep_util', import_52558)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from numpy.distutils.misc_util import filter_sources, has_f_sources, has_cxx_sources, all_strings, get_lib_source_files, is_sequence, get_numpy_include_dirs' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_52560 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.distutils.misc_util')

if (type(import_52560) is not StypyTypeError):

    if (import_52560 != 'pyd_module'):
        __import__(import_52560)
        sys_modules_52561 = sys.modules[import_52560]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.distutils.misc_util', sys_modules_52561.module_type_store, module_type_store, ['filter_sources', 'has_f_sources', 'has_cxx_sources', 'all_strings', 'get_lib_source_files', 'is_sequence', 'get_numpy_include_dirs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_52561, sys_modules_52561.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import filter_sources, has_f_sources, has_cxx_sources, all_strings, get_lib_source_files, is_sequence, get_numpy_include_dirs

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.distutils.misc_util', None, module_type_store, ['filter_sources', 'has_f_sources', 'has_cxx_sources', 'all_strings', 'get_lib_source_files', 'is_sequence', 'get_numpy_include_dirs'], [filter_sources, has_f_sources, has_cxx_sources, all_strings, get_lib_source_files, is_sequence, get_numpy_include_dirs])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'numpy.distutils.misc_util', import_52560)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')


# Assigning a Attribute to a Name (line 19):

# Assigning a Attribute to a Name (line 19):
# Getting the type of 'old_build_clib' (line 19)
old_build_clib_52562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'old_build_clib')
# Obtaining the member 'user_options' of a type (line 19)
user_options_52563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), old_build_clib_52562, 'user_options')
# Assigning a type to the variable '_l' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), '_l', user_options_52563)


# Call to range(...): (line 20)
# Processing the call arguments (line 20)

# Call to len(...): (line 20)
# Processing the call arguments (line 20)
# Getting the type of '_l' (line 20)
_l_52566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 20), '_l', False)
# Processing the call keyword arguments (line 20)
kwargs_52567 = {}
# Getting the type of 'len' (line 20)
len_52565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'len', False)
# Calling len(args, kwargs) (line 20)
len_call_result_52568 = invoke(stypy.reporting.localization.Localization(__file__, 20, 16), len_52565, *[_l_52566], **kwargs_52567)

# Processing the call keyword arguments (line 20)
kwargs_52569 = {}
# Getting the type of 'range' (line 20)
range_52564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'range', False)
# Calling range(args, kwargs) (line 20)
range_call_result_52570 = invoke(stypy.reporting.localization.Localization(__file__, 20, 10), range_52564, *[len_call_result_52568], **kwargs_52569)

# Testing the type of a for loop iterable (line 20)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 20, 0), range_call_result_52570)
# Getting the type of the for loop variable (line 20)
for_loop_var_52571 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 20, 0), range_call_result_52570)
# Assigning a type to the variable '_i' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), '_i', for_loop_var_52571)
# SSA begins for a for statement (line 20)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')



# Obtaining the type of the subscript
int_52572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 14), 'int')

# Obtaining the type of the subscript
# Getting the type of '_i' (line 21)
_i_52573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), '_i')
# Getting the type of '_l' (line 21)
_l_52574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 7), '_l')
# Obtaining the member '__getitem__' of a type (line 21)
getitem___52575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 7), _l_52574, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 21)
subscript_call_result_52576 = invoke(stypy.reporting.localization.Localization(__file__, 21, 7), getitem___52575, _i_52573)

# Obtaining the member '__getitem__' of a type (line 21)
getitem___52577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 7), subscript_call_result_52576, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 21)
subscript_call_result_52578 = invoke(stypy.reporting.localization.Localization(__file__, 21, 7), getitem___52577, int_52572)


# Obtaining an instance of the builtin type 'list' (line 21)
list_52579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_52580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 21), 'str', 'build-clib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 20), list_52579, str_52580)
# Adding element type (line 21)
str_52581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 35), 'str', 'build-temp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 20), list_52579, str_52581)

# Applying the binary operator 'in' (line 21)
result_contains_52582 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 7), 'in', subscript_call_result_52578, list_52579)

# Testing the type of an if condition (line 21)
if_condition_52583 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 4), result_contains_52582)
# Assigning a type to the variable 'if_condition_52583' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'if_condition_52583', if_condition_52583)
# SSA begins for if statement (line 21)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a BinOp to a Subscript (line 22):

# Assigning a BinOp to a Subscript (line 22):

# Obtaining an instance of the builtin type 'tuple' (line 22)
tuple_52584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 22)
# Adding element type (line 22)

# Obtaining the type of the subscript
int_52585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'int')

# Obtaining the type of the subscript
# Getting the type of '_i' (line 22)
_i_52586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 21), '_i')
# Getting the type of '_l' (line 22)
_l_52587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 18), '_l')
# Obtaining the member '__getitem__' of a type (line 22)
getitem___52588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 18), _l_52587, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 22)
subscript_call_result_52589 = invoke(stypy.reporting.localization.Localization(__file__, 22, 18), getitem___52588, _i_52586)

# Obtaining the member '__getitem__' of a type (line 22)
getitem___52590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 18), subscript_call_result_52589, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 22)
subscript_call_result_52591 = invoke(stypy.reporting.localization.Localization(__file__, 22, 18), getitem___52590, int_52585)

str_52592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 28), 'str', '=')
# Applying the binary operator '+' (line 22)
result_add_52593 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 18), '+', subscript_call_result_52591, str_52592)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 18), tuple_52584, result_add_52593)


# Obtaining the type of the subscript
int_52594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 41), 'int')
slice_52595 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 22, 34), int_52594, None, None)

# Obtaining the type of the subscript
# Getting the type of '_i' (line 22)
_i_52596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 37), '_i')
# Getting the type of '_l' (line 22)
_l_52597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 34), '_l')
# Obtaining the member '__getitem__' of a type (line 22)
getitem___52598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 34), _l_52597, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 22)
subscript_call_result_52599 = invoke(stypy.reporting.localization.Localization(__file__, 22, 34), getitem___52598, _i_52596)

# Obtaining the member '__getitem__' of a type (line 22)
getitem___52600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 34), subscript_call_result_52599, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 22)
subscript_call_result_52601 = invoke(stypy.reporting.localization.Localization(__file__, 22, 34), getitem___52600, slice_52595)

# Applying the binary operator '+' (line 22)
result_add_52602 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 17), '+', tuple_52584, subscript_call_result_52601)

# Getting the type of '_l' (line 22)
_l_52603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), '_l')
# Getting the type of '_i' (line 22)
_i_52604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), '_i')
# Storing an element on a container (line 22)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 8), _l_52603, (_i_52604, result_add_52602))
# SSA join for if statement (line 21)
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'build_clib' class
# Getting the type of 'old_build_clib' (line 25)
old_build_clib_52605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'old_build_clib')

class build_clib(old_build_clib_52605, ):
    
    # Assigning a Str to a Name (line 27):
    
    # Assigning a BinOp to a Name (line 29):
    
    # Assigning a BinOp to a Name (line 37):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_clib.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        build_clib.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_clib.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_clib.initialize_options.__dict__.__setitem__('stypy_function_name', 'build_clib.initialize_options')
        build_clib.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build_clib.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_clib.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_clib.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_clib.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_clib.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_clib.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to initialize_options(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'self' (line 40)
        self_52608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 42), 'self', False)
        # Processing the call keyword arguments (line 40)
        kwargs_52609 = {}
        # Getting the type of 'old_build_clib' (line 40)
        old_build_clib_52606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'old_build_clib', False)
        # Obtaining the member 'initialize_options' of a type (line 40)
        initialize_options_52607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), old_build_clib_52606, 'initialize_options')
        # Calling initialize_options(args, kwargs) (line 40)
        initialize_options_call_result_52610 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), initialize_options_52607, *[self_52608], **kwargs_52609)
        
        
        # Assigning a Name to a Attribute (line 41):
        
        # Assigning a Name to a Attribute (line 41):
        # Getting the type of 'None' (line 41)
        None_52611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'None')
        # Getting the type of 'self' (line 41)
        self_52612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member 'fcompiler' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_52612, 'fcompiler', None_52611)
        
        # Assigning a Num to a Attribute (line 42):
        
        # Assigning a Num to a Attribute (line 42):
        int_52613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 23), 'int')
        # Getting the type of 'self' (line 42)
        self_52614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self')
        # Setting the type of the member 'inplace' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_52614, 'inplace', int_52613)
        
        # Assigning a Name to a Attribute (line 43):
        
        # Assigning a Name to a Attribute (line 43):
        # Getting the type of 'None' (line 43)
        None_52615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'None')
        # Getting the type of 'self' (line 43)
        self_52616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member 'parallel' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_52616, 'parallel', None_52615)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_52617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52617)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_52617


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_clib.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        build_clib.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_clib.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_clib.finalize_options.__dict__.__setitem__('stypy_function_name', 'build_clib.finalize_options')
        build_clib.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build_clib.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_clib.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_clib.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_clib.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_clib.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_clib.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'self' (line 46)
        self_52618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'self')
        # Obtaining the member 'parallel' of a type (line 46)
        parallel_52619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 11), self_52618, 'parallel')
        # Testing the type of an if condition (line 46)
        if_condition_52620 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 8), parallel_52619)
        # Assigning a type to the variable 'if_condition_52620' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'if_condition_52620', if_condition_52620)
        # SSA begins for if statement (line 46)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Attribute (line 48):
        
        # Assigning a Call to a Attribute (line 48):
        
        # Call to int(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'self' (line 48)
        self_52622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 36), 'self', False)
        # Obtaining the member 'parallel' of a type (line 48)
        parallel_52623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 36), self_52622, 'parallel')
        # Processing the call keyword arguments (line 48)
        kwargs_52624 = {}
        # Getting the type of 'int' (line 48)
        int_52621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 32), 'int', False)
        # Calling int(args, kwargs) (line 48)
        int_call_result_52625 = invoke(stypy.reporting.localization.Localization(__file__, 48, 32), int_52621, *[parallel_52623], **kwargs_52624)
        
        # Getting the type of 'self' (line 48)
        self_52626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'self')
        # Setting the type of the member 'parallel' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 16), self_52626, 'parallel', int_call_result_52625)
        # SSA branch for the except part of a try statement (line 47)
        # SSA branch for the except 'ValueError' branch of a try statement (line 47)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 50)
        # Processing the call arguments (line 50)
        str_52628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 33), 'str', '--parallel/-j argument must be an integer')
        # Processing the call keyword arguments (line 50)
        kwargs_52629 = {}
        # Getting the type of 'ValueError' (line 50)
        ValueError_52627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 50)
        ValueError_call_result_52630 = invoke(stypy.reporting.localization.Localization(__file__, 50, 22), ValueError_52627, *[str_52628], **kwargs_52629)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 50, 16), ValueError_call_result_52630, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 47)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 46)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to finalize_options(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'self' (line 51)
        self_52633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 40), 'self', False)
        # Processing the call keyword arguments (line 51)
        kwargs_52634 = {}
        # Getting the type of 'old_build_clib' (line 51)
        old_build_clib_52631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'old_build_clib', False)
        # Obtaining the member 'finalize_options' of a type (line 51)
        finalize_options_52632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), old_build_clib_52631, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 51)
        finalize_options_call_result_52635 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), finalize_options_52632, *[self_52633], **kwargs_52634)
        
        
        # Call to set_undefined_options(...): (line 52)
        # Processing the call arguments (line 52)
        str_52638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 35), 'str', 'build')
        
        # Obtaining an instance of the builtin type 'tuple' (line 52)
        tuple_52639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 52)
        # Adding element type (line 52)
        str_52640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 45), 'str', 'parallel')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 45), tuple_52639, str_52640)
        # Adding element type (line 52)
        str_52641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 57), 'str', 'parallel')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 45), tuple_52639, str_52641)
        
        # Processing the call keyword arguments (line 52)
        kwargs_52642 = {}
        # Getting the type of 'self' (line 52)
        self_52636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 52)
        set_undefined_options_52637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_52636, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 52)
        set_undefined_options_call_result_52643 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), set_undefined_options_52637, *[str_52638, tuple_52639], **kwargs_52642)
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_52644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52644)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_52644


    @norecursion
    def have_f_sources(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'have_f_sources'
        module_type_store = module_type_store.open_function_context('have_f_sources', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_clib.have_f_sources.__dict__.__setitem__('stypy_localization', localization)
        build_clib.have_f_sources.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_clib.have_f_sources.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_clib.have_f_sources.__dict__.__setitem__('stypy_function_name', 'build_clib.have_f_sources')
        build_clib.have_f_sources.__dict__.__setitem__('stypy_param_names_list', [])
        build_clib.have_f_sources.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_clib.have_f_sources.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_clib.have_f_sources.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_clib.have_f_sources.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_clib.have_f_sources.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_clib.have_f_sources.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.have_f_sources', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'have_f_sources', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'have_f_sources(...)' code ##################

        
        # Getting the type of 'self' (line 55)
        self_52645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 38), 'self')
        # Obtaining the member 'libraries' of a type (line 55)
        libraries_52646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 38), self_52645, 'libraries')
        # Testing the type of a for loop iterable (line 55)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 55, 8), libraries_52646)
        # Getting the type of the for loop variable (line 55)
        for_loop_var_52647 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 55, 8), libraries_52646)
        # Assigning a type to the variable 'lib_name' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'lib_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 8), for_loop_var_52647))
        # Assigning a type to the variable 'build_info' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'build_info', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 8), for_loop_var_52647))
        # SSA begins for a for statement (line 55)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to has_f_sources(...): (line 56)
        # Processing the call arguments (line 56)
        
        # Call to get(...): (line 56)
        # Processing the call arguments (line 56)
        str_52651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 44), 'str', 'sources')
        
        # Obtaining an instance of the builtin type 'list' (line 56)
        list_52652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 56)
        
        # Processing the call keyword arguments (line 56)
        kwargs_52653 = {}
        # Getting the type of 'build_info' (line 56)
        build_info_52649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 29), 'build_info', False)
        # Obtaining the member 'get' of a type (line 56)
        get_52650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 29), build_info_52649, 'get')
        # Calling get(args, kwargs) (line 56)
        get_call_result_52654 = invoke(stypy.reporting.localization.Localization(__file__, 56, 29), get_52650, *[str_52651, list_52652], **kwargs_52653)
        
        # Processing the call keyword arguments (line 56)
        kwargs_52655 = {}
        # Getting the type of 'has_f_sources' (line 56)
        has_f_sources_52648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'has_f_sources', False)
        # Calling has_f_sources(args, kwargs) (line 56)
        has_f_sources_call_result_52656 = invoke(stypy.reporting.localization.Localization(__file__, 56, 15), has_f_sources_52648, *[get_call_result_52654], **kwargs_52655)
        
        # Testing the type of an if condition (line 56)
        if_condition_52657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 12), has_f_sources_call_result_52656)
        # Assigning a type to the variable 'if_condition_52657' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'if_condition_52657', if_condition_52657)
        # SSA begins for if statement (line 56)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 57)
        True_52658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'stypy_return_type', True_52658)
        # SSA join for if statement (line 56)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'False' (line 58)
        False_52659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'stypy_return_type', False_52659)
        
        # ################# End of 'have_f_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'have_f_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_52660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52660)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'have_f_sources'
        return stypy_return_type_52660


    @norecursion
    def have_cxx_sources(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'have_cxx_sources'
        module_type_store = module_type_store.open_function_context('have_cxx_sources', 60, 4, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_clib.have_cxx_sources.__dict__.__setitem__('stypy_localization', localization)
        build_clib.have_cxx_sources.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_clib.have_cxx_sources.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_clib.have_cxx_sources.__dict__.__setitem__('stypy_function_name', 'build_clib.have_cxx_sources')
        build_clib.have_cxx_sources.__dict__.__setitem__('stypy_param_names_list', [])
        build_clib.have_cxx_sources.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_clib.have_cxx_sources.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_clib.have_cxx_sources.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_clib.have_cxx_sources.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_clib.have_cxx_sources.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_clib.have_cxx_sources.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.have_cxx_sources', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'have_cxx_sources', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'have_cxx_sources(...)' code ##################

        
        # Getting the type of 'self' (line 61)
        self_52661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 38), 'self')
        # Obtaining the member 'libraries' of a type (line 61)
        libraries_52662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 38), self_52661, 'libraries')
        # Testing the type of a for loop iterable (line 61)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 61, 8), libraries_52662)
        # Getting the type of the for loop variable (line 61)
        for_loop_var_52663 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 61, 8), libraries_52662)
        # Assigning a type to the variable 'lib_name' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'lib_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 8), for_loop_var_52663))
        # Assigning a type to the variable 'build_info' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'build_info', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 8), for_loop_var_52663))
        # SSA begins for a for statement (line 61)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to has_cxx_sources(...): (line 62)
        # Processing the call arguments (line 62)
        
        # Call to get(...): (line 62)
        # Processing the call arguments (line 62)
        str_52667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 46), 'str', 'sources')
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_52668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        
        # Processing the call keyword arguments (line 62)
        kwargs_52669 = {}
        # Getting the type of 'build_info' (line 62)
        build_info_52665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 31), 'build_info', False)
        # Obtaining the member 'get' of a type (line 62)
        get_52666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 31), build_info_52665, 'get')
        # Calling get(args, kwargs) (line 62)
        get_call_result_52670 = invoke(stypy.reporting.localization.Localization(__file__, 62, 31), get_52666, *[str_52667, list_52668], **kwargs_52669)
        
        # Processing the call keyword arguments (line 62)
        kwargs_52671 = {}
        # Getting the type of 'has_cxx_sources' (line 62)
        has_cxx_sources_52664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'has_cxx_sources', False)
        # Calling has_cxx_sources(args, kwargs) (line 62)
        has_cxx_sources_call_result_52672 = invoke(stypy.reporting.localization.Localization(__file__, 62, 15), has_cxx_sources_52664, *[get_call_result_52670], **kwargs_52671)
        
        # Testing the type of an if condition (line 62)
        if_condition_52673 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 12), has_cxx_sources_call_result_52672)
        # Assigning a type to the variable 'if_condition_52673' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'if_condition_52673', if_condition_52673)
        # SSA begins for if statement (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 63)
        True_52674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 23), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'stypy_return_type', True_52674)
        # SSA join for if statement (line 62)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'False' (line 64)
        False_52675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'stypy_return_type', False_52675)
        
        # ################# End of 'have_cxx_sources(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'have_cxx_sources' in the type store
        # Getting the type of 'stypy_return_type' (line 60)
        stypy_return_type_52676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52676)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'have_cxx_sources'
        return stypy_return_type_52676


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_clib.run.__dict__.__setitem__('stypy_localization', localization)
        build_clib.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_clib.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_clib.run.__dict__.__setitem__('stypy_function_name', 'build_clib.run')
        build_clib.run.__dict__.__setitem__('stypy_param_names_list', [])
        build_clib.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_clib.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_clib.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_clib.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_clib.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_clib.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.run', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 67)
        self_52677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'self')
        # Obtaining the member 'libraries' of a type (line 67)
        libraries_52678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 15), self_52677, 'libraries')
        # Applying the 'not' unary operator (line 67)
        result_not__52679 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 11), 'not', libraries_52678)
        
        # Testing the type of an if condition (line 67)
        if_condition_52680 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), result_not__52679)
        # Assigning a type to the variable 'if_condition_52680' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'if_condition_52680', if_condition_52680)
        # SSA begins for if statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 67)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 71):
        
        # Assigning a List to a Name (line 71):
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_52681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        
        # Assigning a type to the variable 'languages' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'languages', list_52681)
        
        # Call to run_command(...): (line 74)
        # Processing the call arguments (line 74)
        str_52684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 25), 'str', 'build_src')
        # Processing the call keyword arguments (line 74)
        kwargs_52685 = {}
        # Getting the type of 'self' (line 74)
        self_52682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self', False)
        # Obtaining the member 'run_command' of a type (line 74)
        run_command_52683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_52682, 'run_command')
        # Calling run_command(args, kwargs) (line 74)
        run_command_call_result_52686 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), run_command_52683, *[str_52684], **kwargs_52685)
        
        
        # Getting the type of 'self' (line 76)
        self_52687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 38), 'self')
        # Obtaining the member 'libraries' of a type (line 76)
        libraries_52688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 38), self_52687, 'libraries')
        # Testing the type of a for loop iterable (line 76)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 76, 8), libraries_52688)
        # Getting the type of the for loop variable (line 76)
        for_loop_var_52689 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 76, 8), libraries_52688)
        # Assigning a type to the variable 'lib_name' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'lib_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 8), for_loop_var_52689))
        # Assigning a type to the variable 'build_info' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'build_info', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 8), for_loop_var_52689))
        # SSA begins for a for statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 77):
        
        # Assigning a Call to a Name (line 77):
        
        # Call to get(...): (line 77)
        # Processing the call arguments (line 77)
        str_52692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 31), 'str', 'language')
        # Getting the type of 'None' (line 77)
        None_52693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 43), 'None', False)
        # Processing the call keyword arguments (line 77)
        kwargs_52694 = {}
        # Getting the type of 'build_info' (line 77)
        build_info_52690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'build_info', False)
        # Obtaining the member 'get' of a type (line 77)
        get_52691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 16), build_info_52690, 'get')
        # Calling get(args, kwargs) (line 77)
        get_call_result_52695 = invoke(stypy.reporting.localization.Localization(__file__, 77, 16), get_52691, *[str_52692, None_52693], **kwargs_52694)
        
        # Assigning a type to the variable 'l' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'l', get_call_result_52695)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'l' (line 78)
        l_52696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'l')
        
        # Getting the type of 'l' (line 78)
        l_52697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 21), 'l')
        # Getting the type of 'languages' (line 78)
        languages_52698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'languages')
        # Applying the binary operator 'notin' (line 78)
        result_contains_52699 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 21), 'notin', l_52697, languages_52698)
        
        # Applying the binary operator 'and' (line 78)
        result_and_keyword_52700 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 15), 'and', l_52696, result_contains_52699)
        
        # Testing the type of an if condition (line 78)
        if_condition_52701 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 12), result_and_keyword_52700)
        # Assigning a type to the variable 'if_condition_52701' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'if_condition_52701', if_condition_52701)
        # SSA begins for if statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'l' (line 78)
        l_52704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 58), 'l', False)
        # Processing the call keyword arguments (line 78)
        kwargs_52705 = {}
        # Getting the type of 'languages' (line 78)
        languages_52702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 41), 'languages', False)
        # Obtaining the member 'append' of a type (line 78)
        append_52703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 41), languages_52702, 'append')
        # Calling append(args, kwargs) (line 78)
        append_call_result_52706 = invoke(stypy.reporting.localization.Localization(__file__, 78, 41), append_52703, *[l_52704], **kwargs_52705)
        
        # SSA join for if statement (line 78)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 80, 8))
        
        # 'from distutils.ccompiler import new_compiler' statement (line 80)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
        import_52707 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 80, 8), 'distutils.ccompiler')

        if (type(import_52707) is not StypyTypeError):

            if (import_52707 != 'pyd_module'):
                __import__(import_52707)
                sys_modules_52708 = sys.modules[import_52707]
                import_from_module(stypy.reporting.localization.Localization(__file__, 80, 8), 'distutils.ccompiler', sys_modules_52708.module_type_store, module_type_store, ['new_compiler'])
                nest_module(stypy.reporting.localization.Localization(__file__, 80, 8), __file__, sys_modules_52708, sys_modules_52708.module_type_store, module_type_store)
            else:
                from distutils.ccompiler import new_compiler

                import_from_module(stypy.reporting.localization.Localization(__file__, 80, 8), 'distutils.ccompiler', None, module_type_store, ['new_compiler'], [new_compiler])

        else:
            # Assigning a type to the variable 'distutils.ccompiler' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'distutils.ccompiler', import_52707)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')
        
        
        # Assigning a Call to a Attribute (line 81):
        
        # Assigning a Call to a Attribute (line 81):
        
        # Call to new_compiler(...): (line 81)
        # Processing the call keyword arguments (line 81)
        # Getting the type of 'self' (line 81)
        self_52710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 46), 'self', False)
        # Obtaining the member 'compiler' of a type (line 81)
        compiler_52711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 46), self_52710, 'compiler')
        keyword_52712 = compiler_52711
        # Getting the type of 'self' (line 82)
        self_52713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 45), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 82)
        dry_run_52714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 45), self_52713, 'dry_run')
        keyword_52715 = dry_run_52714
        # Getting the type of 'self' (line 83)
        self_52716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 43), 'self', False)
        # Obtaining the member 'force' of a type (line 83)
        force_52717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 43), self_52716, 'force')
        keyword_52718 = force_52717
        kwargs_52719 = {'force': keyword_52718, 'dry_run': keyword_52715, 'compiler': keyword_52712}
        # Getting the type of 'new_compiler' (line 81)
        new_compiler_52709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 24), 'new_compiler', False)
        # Calling new_compiler(args, kwargs) (line 81)
        new_compiler_call_result_52720 = invoke(stypy.reporting.localization.Localization(__file__, 81, 24), new_compiler_52709, *[], **kwargs_52719)
        
        # Getting the type of 'self' (line 81)
        self_52721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self')
        # Setting the type of the member 'compiler' of a type (line 81)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_52721, 'compiler', new_compiler_call_result_52720)
        
        # Call to customize(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'self' (line 84)
        self_52725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'self', False)
        # Obtaining the member 'distribution' of a type (line 84)
        distribution_52726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 32), self_52725, 'distribution')
        # Processing the call keyword arguments (line 84)
        
        # Call to have_cxx_sources(...): (line 85)
        # Processing the call keyword arguments (line 85)
        kwargs_52729 = {}
        # Getting the type of 'self' (line 85)
        self_52727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 41), 'self', False)
        # Obtaining the member 'have_cxx_sources' of a type (line 85)
        have_cxx_sources_52728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 41), self_52727, 'have_cxx_sources')
        # Calling have_cxx_sources(args, kwargs) (line 85)
        have_cxx_sources_call_result_52730 = invoke(stypy.reporting.localization.Localization(__file__, 85, 41), have_cxx_sources_52728, *[], **kwargs_52729)
        
        keyword_52731 = have_cxx_sources_call_result_52730
        kwargs_52732 = {'need_cxx': keyword_52731}
        # Getting the type of 'self' (line 84)
        self_52722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'self', False)
        # Obtaining the member 'compiler' of a type (line 84)
        compiler_52723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), self_52722, 'compiler')
        # Obtaining the member 'customize' of a type (line 84)
        customize_52724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), compiler_52723, 'customize')
        # Calling customize(args, kwargs) (line 84)
        customize_call_result_52733 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), customize_52724, *[distribution_52726], **kwargs_52732)
        
        
        # Assigning a Attribute to a Name (line 87):
        
        # Assigning a Attribute to a Name (line 87):
        # Getting the type of 'self' (line 87)
        self_52734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'self')
        # Obtaining the member 'libraries' of a type (line 87)
        libraries_52735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 20), self_52734, 'libraries')
        # Assigning a type to the variable 'libraries' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'libraries', libraries_52735)
        
        # Assigning a Name to a Attribute (line 88):
        
        # Assigning a Name to a Attribute (line 88):
        # Getting the type of 'None' (line 88)
        None_52736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 25), 'None')
        # Getting the type of 'self' (line 88)
        self_52737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self')
        # Setting the type of the member 'libraries' of a type (line 88)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_52737, 'libraries', None_52736)
        
        # Call to customize_cmd(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'self' (line 89)
        self_52741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 36), 'self', False)
        # Processing the call keyword arguments (line 89)
        kwargs_52742 = {}
        # Getting the type of 'self' (line 89)
        self_52738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self', False)
        # Obtaining the member 'compiler' of a type (line 89)
        compiler_52739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_52738, 'compiler')
        # Obtaining the member 'customize_cmd' of a type (line 89)
        customize_cmd_52740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), compiler_52739, 'customize_cmd')
        # Calling customize_cmd(args, kwargs) (line 89)
        customize_cmd_call_result_52743 = invoke(stypy.reporting.localization.Localization(__file__, 89, 8), customize_cmd_52740, *[self_52741], **kwargs_52742)
        
        
        # Assigning a Name to a Attribute (line 90):
        
        # Assigning a Name to a Attribute (line 90):
        # Getting the type of 'libraries' (line 90)
        libraries_52744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'libraries')
        # Getting the type of 'self' (line 90)
        self_52745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self')
        # Setting the type of the member 'libraries' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_52745, 'libraries', libraries_52744)
        
        # Call to show_customization(...): (line 92)
        # Processing the call keyword arguments (line 92)
        kwargs_52749 = {}
        # Getting the type of 'self' (line 92)
        self_52746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'self', False)
        # Obtaining the member 'compiler' of a type (line 92)
        compiler_52747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), self_52746, 'compiler')
        # Obtaining the member 'show_customization' of a type (line 92)
        show_customization_52748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), compiler_52747, 'show_customization')
        # Calling show_customization(args, kwargs) (line 92)
        show_customization_call_result_52750 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), show_customization_52748, *[], **kwargs_52749)
        
        
        
        # Call to have_f_sources(...): (line 94)
        # Processing the call keyword arguments (line 94)
        kwargs_52753 = {}
        # Getting the type of 'self' (line 94)
        self_52751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'self', False)
        # Obtaining the member 'have_f_sources' of a type (line 94)
        have_f_sources_52752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 11), self_52751, 'have_f_sources')
        # Calling have_f_sources(args, kwargs) (line 94)
        have_f_sources_call_result_52754 = invoke(stypy.reporting.localization.Localization(__file__, 94, 11), have_f_sources_52752, *[], **kwargs_52753)
        
        # Testing the type of an if condition (line 94)
        if_condition_52755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 8), have_f_sources_call_result_52754)
        # Assigning a type to the variable 'if_condition_52755' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'if_condition_52755', if_condition_52755)
        # SSA begins for if statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 95, 12))
        
        # 'from numpy.distutils.fcompiler import new_fcompiler' statement (line 95)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
        import_52756 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 95, 12), 'numpy.distutils.fcompiler')

        if (type(import_52756) is not StypyTypeError):

            if (import_52756 != 'pyd_module'):
                __import__(import_52756)
                sys_modules_52757 = sys.modules[import_52756]
                import_from_module(stypy.reporting.localization.Localization(__file__, 95, 12), 'numpy.distutils.fcompiler', sys_modules_52757.module_type_store, module_type_store, ['new_fcompiler'])
                nest_module(stypy.reporting.localization.Localization(__file__, 95, 12), __file__, sys_modules_52757, sys_modules_52757.module_type_store, module_type_store)
            else:
                from numpy.distutils.fcompiler import new_fcompiler

                import_from_module(stypy.reporting.localization.Localization(__file__, 95, 12), 'numpy.distutils.fcompiler', None, module_type_store, ['new_fcompiler'], [new_fcompiler])

        else:
            # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'numpy.distutils.fcompiler', import_52756)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')
        
        
        # Assigning a Call to a Attribute (line 96):
        
        # Assigning a Call to a Attribute (line 96):
        
        # Call to new_fcompiler(...): (line 96)
        # Processing the call keyword arguments (line 96)
        # Getting the type of 'self' (line 96)
        self_52759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 54), 'self', False)
        # Obtaining the member 'fcompiler' of a type (line 96)
        fcompiler_52760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 54), self_52759, 'fcompiler')
        keyword_52761 = fcompiler_52760
        # Getting the type of 'self' (line 97)
        self_52762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 55), 'self', False)
        # Obtaining the member 'verbose' of a type (line 97)
        verbose_52763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 55), self_52762, 'verbose')
        keyword_52764 = verbose_52763
        # Getting the type of 'self' (line 98)
        self_52765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 55), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 98)
        dry_run_52766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 55), self_52765, 'dry_run')
        keyword_52767 = dry_run_52766
        # Getting the type of 'self' (line 99)
        self_52768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 53), 'self', False)
        # Obtaining the member 'force' of a type (line 99)
        force_52769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 53), self_52768, 'force')
        keyword_52770 = force_52769
        
        str_52771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 58), 'str', 'f90')
        # Getting the type of 'languages' (line 100)
        languages_52772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 67), 'languages', False)
        # Applying the binary operator 'in' (line 100)
        result_contains_52773 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 58), 'in', str_52771, languages_52772)
        
        keyword_52774 = result_contains_52773
        # Getting the type of 'self' (line 101)
        self_52775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 58), 'self', False)
        # Obtaining the member 'compiler' of a type (line 101)
        compiler_52776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 58), self_52775, 'compiler')
        keyword_52777 = compiler_52776
        kwargs_52778 = {'force': keyword_52770, 'verbose': keyword_52764, 'dry_run': keyword_52767, 'c_compiler': keyword_52777, 'requiref90': keyword_52774, 'compiler': keyword_52761}
        # Getting the type of 'new_fcompiler' (line 96)
        new_fcompiler_52758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 31), 'new_fcompiler', False)
        # Calling new_fcompiler(args, kwargs) (line 96)
        new_fcompiler_call_result_52779 = invoke(stypy.reporting.localization.Localization(__file__, 96, 31), new_fcompiler_52758, *[], **kwargs_52778)
        
        # Getting the type of 'self' (line 96)
        self_52780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'self')
        # Setting the type of the member '_f_compiler' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), self_52780, '_f_compiler', new_fcompiler_call_result_52779)
        
        
        # Getting the type of 'self' (line 102)
        self_52781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'self')
        # Obtaining the member '_f_compiler' of a type (line 102)
        _f_compiler_52782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), self_52781, '_f_compiler')
        # Getting the type of 'None' (line 102)
        None_52783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 39), 'None')
        # Applying the binary operator 'isnot' (line 102)
        result_is_not_52784 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 15), 'isnot', _f_compiler_52782, None_52783)
        
        # Testing the type of an if condition (line 102)
        if_condition_52785 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 12), result_is_not_52784)
        # Assigning a type to the variable 'if_condition_52785' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'if_condition_52785', if_condition_52785)
        # SSA begins for if statement (line 102)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to customize(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'self' (line 103)
        self_52789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 43), 'self', False)
        # Obtaining the member 'distribution' of a type (line 103)
        distribution_52790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 43), self_52789, 'distribution')
        # Processing the call keyword arguments (line 103)
        kwargs_52791 = {}
        # Getting the type of 'self' (line 103)
        self_52786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'self', False)
        # Obtaining the member '_f_compiler' of a type (line 103)
        _f_compiler_52787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 16), self_52786, '_f_compiler')
        # Obtaining the member 'customize' of a type (line 103)
        customize_52788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 16), _f_compiler_52787, 'customize')
        # Calling customize(args, kwargs) (line 103)
        customize_call_result_52792 = invoke(stypy.reporting.localization.Localization(__file__, 103, 16), customize_52788, *[distribution_52790], **kwargs_52791)
        
        
        # Assigning a Attribute to a Name (line 105):
        
        # Assigning a Attribute to a Name (line 105):
        # Getting the type of 'self' (line 105)
        self_52793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 28), 'self')
        # Obtaining the member 'libraries' of a type (line 105)
        libraries_52794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 28), self_52793, 'libraries')
        # Assigning a type to the variable 'libraries' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'libraries', libraries_52794)
        
        # Assigning a Name to a Attribute (line 106):
        
        # Assigning a Name to a Attribute (line 106):
        # Getting the type of 'None' (line 106)
        None_52795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'None')
        # Getting the type of 'self' (line 106)
        self_52796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'self')
        # Setting the type of the member 'libraries' of a type (line 106)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), self_52796, 'libraries', None_52795)
        
        # Call to customize_cmd(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'self' (line 107)
        self_52800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 47), 'self', False)
        # Processing the call keyword arguments (line 107)
        kwargs_52801 = {}
        # Getting the type of 'self' (line 107)
        self_52797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'self', False)
        # Obtaining the member '_f_compiler' of a type (line 107)
        _f_compiler_52798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), self_52797, '_f_compiler')
        # Obtaining the member 'customize_cmd' of a type (line 107)
        customize_cmd_52799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 16), _f_compiler_52798, 'customize_cmd')
        # Calling customize_cmd(args, kwargs) (line 107)
        customize_cmd_call_result_52802 = invoke(stypy.reporting.localization.Localization(__file__, 107, 16), customize_cmd_52799, *[self_52800], **kwargs_52801)
        
        
        # Assigning a Name to a Attribute (line 108):
        
        # Assigning a Name to a Attribute (line 108):
        # Getting the type of 'libraries' (line 108)
        libraries_52803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 33), 'libraries')
        # Getting the type of 'self' (line 108)
        self_52804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'self')
        # Setting the type of the member 'libraries' of a type (line 108)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 16), self_52804, 'libraries', libraries_52803)
        
        # Call to show_customization(...): (line 110)
        # Processing the call keyword arguments (line 110)
        kwargs_52808 = {}
        # Getting the type of 'self' (line 110)
        self_52805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'self', False)
        # Obtaining the member '_f_compiler' of a type (line 110)
        _f_compiler_52806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), self_52805, '_f_compiler')
        # Obtaining the member 'show_customization' of a type (line 110)
        show_customization_52807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), _f_compiler_52806, 'show_customization')
        # Calling show_customization(args, kwargs) (line 110)
        show_customization_call_result_52809 = invoke(stypy.reporting.localization.Localization(__file__, 110, 16), show_customization_52807, *[], **kwargs_52808)
        
        # SSA join for if statement (line 102)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 94)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Attribute (line 112):
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'None' (line 112)
        None_52810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'None')
        # Getting the type of 'self' (line 112)
        self_52811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'self')
        # Setting the type of the member '_f_compiler' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 12), self_52811, '_f_compiler', None_52810)
        # SSA join for if statement (line 94)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to build_libraries(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'self' (line 114)
        self_52814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 29), 'self', False)
        # Obtaining the member 'libraries' of a type (line 114)
        libraries_52815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 29), self_52814, 'libraries')
        # Processing the call keyword arguments (line 114)
        kwargs_52816 = {}
        # Getting the type of 'self' (line 114)
        self_52812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self', False)
        # Obtaining the member 'build_libraries' of a type (line 114)
        build_libraries_52813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_52812, 'build_libraries')
        # Calling build_libraries(args, kwargs) (line 114)
        build_libraries_call_result_52817 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), build_libraries_52813, *[libraries_52815], **kwargs_52816)
        
        
        # Getting the type of 'self' (line 116)
        self_52818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'self')
        # Obtaining the member 'inplace' of a type (line 116)
        inplace_52819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 11), self_52818, 'inplace')
        # Testing the type of an if condition (line 116)
        if_condition_52820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 8), inplace_52819)
        # Assigning a type to the variable 'if_condition_52820' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'if_condition_52820', if_condition_52820)
        # SSA begins for if statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 117)
        self_52821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 22), 'self')
        # Obtaining the member 'distribution' of a type (line 117)
        distribution_52822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 22), self_52821, 'distribution')
        # Obtaining the member 'installed_libraries' of a type (line 117)
        installed_libraries_52823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 22), distribution_52822, 'installed_libraries')
        # Testing the type of a for loop iterable (line 117)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 117, 12), installed_libraries_52823)
        # Getting the type of the for loop variable (line 117)
        for_loop_var_52824 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 117, 12), installed_libraries_52823)
        # Assigning a type to the variable 'l' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'l', for_loop_var_52824)
        # SSA begins for a for statement (line 117)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 118):
        
        # Assigning a Call to a Name (line 118):
        
        # Call to library_filename(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'l' (line 118)
        l_52828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 57), 'l', False)
        # Obtaining the member 'name' of a type (line 118)
        name_52829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 57), l_52828, 'name')
        # Processing the call keyword arguments (line 118)
        kwargs_52830 = {}
        # Getting the type of 'self' (line 118)
        self_52825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 26), 'self', False)
        # Obtaining the member 'compiler' of a type (line 118)
        compiler_52826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 26), self_52825, 'compiler')
        # Obtaining the member 'library_filename' of a type (line 118)
        library_filename_52827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 26), compiler_52826, 'library_filename')
        # Calling library_filename(args, kwargs) (line 118)
        library_filename_call_result_52831 = invoke(stypy.reporting.localization.Localization(__file__, 118, 26), library_filename_52827, *[name_52829], **kwargs_52830)
        
        # Assigning a type to the variable 'libname' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'libname', library_filename_call_result_52831)
        
        # Assigning a Call to a Name (line 119):
        
        # Assigning a Call to a Name (line 119):
        
        # Call to join(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'self' (line 119)
        self_52835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 38), 'self', False)
        # Obtaining the member 'build_clib' of a type (line 119)
        build_clib_52836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 38), self_52835, 'build_clib')
        # Getting the type of 'libname' (line 119)
        libname_52837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 55), 'libname', False)
        # Processing the call keyword arguments (line 119)
        kwargs_52838 = {}
        # Getting the type of 'os' (line 119)
        os_52832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 119)
        path_52833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 25), os_52832, 'path')
        # Obtaining the member 'join' of a type (line 119)
        join_52834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 25), path_52833, 'join')
        # Calling join(args, kwargs) (line 119)
        join_call_result_52839 = invoke(stypy.reporting.localization.Localization(__file__, 119, 25), join_52834, *[build_clib_52836, libname_52837], **kwargs_52838)
        
        # Assigning a type to the variable 'source' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'source', join_call_result_52839)
        
        # Assigning a Call to a Name (line 120):
        
        # Assigning a Call to a Name (line 120):
        
        # Call to join(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'l' (line 120)
        l_52843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 39), 'l', False)
        # Obtaining the member 'target_dir' of a type (line 120)
        target_dir_52844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 39), l_52843, 'target_dir')
        # Getting the type of 'libname' (line 120)
        libname_52845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 53), 'libname', False)
        # Processing the call keyword arguments (line 120)
        kwargs_52846 = {}
        # Getting the type of 'os' (line 120)
        os_52840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 120)
        path_52841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 26), os_52840, 'path')
        # Obtaining the member 'join' of a type (line 120)
        join_52842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 26), path_52841, 'join')
        # Calling join(args, kwargs) (line 120)
        join_call_result_52847 = invoke(stypy.reporting.localization.Localization(__file__, 120, 26), join_52842, *[target_dir_52844, libname_52845], **kwargs_52846)
        
        # Assigning a type to the variable 'target' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'target', join_call_result_52847)
        
        # Call to mkpath(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'l' (line 121)
        l_52850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 28), 'l', False)
        # Obtaining the member 'target_dir' of a type (line 121)
        target_dir_52851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 28), l_52850, 'target_dir')
        # Processing the call keyword arguments (line 121)
        kwargs_52852 = {}
        # Getting the type of 'self' (line 121)
        self_52848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 121)
        mkpath_52849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), self_52848, 'mkpath')
        # Calling mkpath(args, kwargs) (line 121)
        mkpath_call_result_52853 = invoke(stypy.reporting.localization.Localization(__file__, 121, 16), mkpath_52849, *[target_dir_52851], **kwargs_52852)
        
        
        # Call to copy(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'source' (line 122)
        source_52856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'source', False)
        # Getting the type of 'target' (line 122)
        target_52857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 36), 'target', False)
        # Processing the call keyword arguments (line 122)
        kwargs_52858 = {}
        # Getting the type of 'shutil' (line 122)
        shutil_52854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'shutil', False)
        # Obtaining the member 'copy' of a type (line 122)
        copy_52855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 16), shutil_52854, 'copy')
        # Calling copy(args, kwargs) (line 122)
        copy_call_result_52859 = invoke(stypy.reporting.localization.Localization(__file__, 122, 16), copy_52855, *[source_52856, target_52857], **kwargs_52858)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 116)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_52860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52860)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_52860


    @norecursion
    def get_source_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_source_files'
        module_type_store = module_type_store.open_function_context('get_source_files', 124, 4, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_clib.get_source_files.__dict__.__setitem__('stypy_localization', localization)
        build_clib.get_source_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_clib.get_source_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_clib.get_source_files.__dict__.__setitem__('stypy_function_name', 'build_clib.get_source_files')
        build_clib.get_source_files.__dict__.__setitem__('stypy_param_names_list', [])
        build_clib.get_source_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_clib.get_source_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_clib.get_source_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_clib.get_source_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_clib.get_source_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_clib.get_source_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.get_source_files', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to check_library_list(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'self' (line 125)
        self_52863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 32), 'self', False)
        # Obtaining the member 'libraries' of a type (line 125)
        libraries_52864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 32), self_52863, 'libraries')
        # Processing the call keyword arguments (line 125)
        kwargs_52865 = {}
        # Getting the type of 'self' (line 125)
        self_52861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'self', False)
        # Obtaining the member 'check_library_list' of a type (line 125)
        check_library_list_52862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), self_52861, 'check_library_list')
        # Calling check_library_list(args, kwargs) (line 125)
        check_library_list_call_result_52866 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), check_library_list_52862, *[libraries_52864], **kwargs_52865)
        
        
        # Assigning a List to a Name (line 126):
        
        # Assigning a List to a Name (line 126):
        
        # Obtaining an instance of the builtin type 'list' (line 126)
        list_52867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 126)
        
        # Assigning a type to the variable 'filenames' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'filenames', list_52867)
        
        # Getting the type of 'self' (line 127)
        self_52868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'self')
        # Obtaining the member 'libraries' of a type (line 127)
        libraries_52869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 19), self_52868, 'libraries')
        # Testing the type of a for loop iterable (line 127)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 127, 8), libraries_52869)
        # Getting the type of the for loop variable (line 127)
        for_loop_var_52870 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 127, 8), libraries_52869)
        # Assigning a type to the variable 'lib' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'lib', for_loop_var_52870)
        # SSA begins for a for statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to extend(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Call to get_lib_source_files(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'lib' (line 128)
        lib_52874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 50), 'lib', False)
        # Processing the call keyword arguments (line 128)
        kwargs_52875 = {}
        # Getting the type of 'get_lib_source_files' (line 128)
        get_lib_source_files_52873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'get_lib_source_files', False)
        # Calling get_lib_source_files(args, kwargs) (line 128)
        get_lib_source_files_call_result_52876 = invoke(stypy.reporting.localization.Localization(__file__, 128, 29), get_lib_source_files_52873, *[lib_52874], **kwargs_52875)
        
        # Processing the call keyword arguments (line 128)
        kwargs_52877 = {}
        # Getting the type of 'filenames' (line 128)
        filenames_52871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'filenames', False)
        # Obtaining the member 'extend' of a type (line 128)
        extend_52872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), filenames_52871, 'extend')
        # Calling extend(args, kwargs) (line 128)
        extend_call_result_52878 = invoke(stypy.reporting.localization.Localization(__file__, 128, 12), extend_52872, *[get_lib_source_files_call_result_52876], **kwargs_52877)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'filenames' (line 129)
        filenames_52879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'filenames')
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type', filenames_52879)
        
        # ################# End of 'get_source_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_source_files' in the type store
        # Getting the type of 'stypy_return_type' (line 124)
        stypy_return_type_52880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52880)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_source_files'
        return stypy_return_type_52880


    @norecursion
    def build_libraries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_libraries'
        module_type_store = module_type_store.open_function_context('build_libraries', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_clib.build_libraries.__dict__.__setitem__('stypy_localization', localization)
        build_clib.build_libraries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_clib.build_libraries.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_clib.build_libraries.__dict__.__setitem__('stypy_function_name', 'build_clib.build_libraries')
        build_clib.build_libraries.__dict__.__setitem__('stypy_param_names_list', ['libraries'])
        build_clib.build_libraries.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_clib.build_libraries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_clib.build_libraries.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_clib.build_libraries.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_clib.build_libraries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_clib.build_libraries.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.build_libraries', ['libraries'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_libraries', localization, ['libraries'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_libraries(...)' code ##################

        
        # Getting the type of 'libraries' (line 132)
        libraries_52881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 38), 'libraries')
        # Testing the type of a for loop iterable (line 132)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 132, 8), libraries_52881)
        # Getting the type of the for loop variable (line 132)
        for_loop_var_52882 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 132, 8), libraries_52881)
        # Assigning a type to the variable 'lib_name' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'lib_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 8), for_loop_var_52882))
        # Assigning a type to the variable 'build_info' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'build_info', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 8), for_loop_var_52882))
        # SSA begins for a for statement (line 132)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to build_a_library(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'build_info' (line 133)
        build_info_52885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 33), 'build_info', False)
        # Getting the type of 'lib_name' (line 133)
        lib_name_52886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 45), 'lib_name', False)
        # Getting the type of 'libraries' (line 133)
        libraries_52887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 55), 'libraries', False)
        # Processing the call keyword arguments (line 133)
        kwargs_52888 = {}
        # Getting the type of 'self' (line 133)
        self_52883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'self', False)
        # Obtaining the member 'build_a_library' of a type (line 133)
        build_a_library_52884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), self_52883, 'build_a_library')
        # Calling build_a_library(args, kwargs) (line 133)
        build_a_library_call_result_52889 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), build_a_library_52884, *[build_info_52885, lib_name_52886, libraries_52887], **kwargs_52888)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'build_libraries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_libraries' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_52890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52890)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_libraries'
        return stypy_return_type_52890


    @norecursion
    def build_a_library(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_a_library'
        module_type_store = module_type_store.open_function_context('build_a_library', 135, 4, False)
        # Assigning a type to the variable 'self' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_clib.build_a_library.__dict__.__setitem__('stypy_localization', localization)
        build_clib.build_a_library.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_clib.build_a_library.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_clib.build_a_library.__dict__.__setitem__('stypy_function_name', 'build_clib.build_a_library')
        build_clib.build_a_library.__dict__.__setitem__('stypy_param_names_list', ['build_info', 'lib_name', 'libraries'])
        build_clib.build_a_library.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_clib.build_a_library.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_clib.build_a_library.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_clib.build_a_library.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_clib.build_a_library.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_clib.build_a_library.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.build_a_library', ['build_info', 'lib_name', 'libraries'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_a_library', localization, ['build_info', 'lib_name', 'libraries'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_a_library(...)' code ##################

        
        # Assigning a Attribute to a Name (line 137):
        
        # Assigning a Attribute to a Name (line 137):
        # Getting the type of 'self' (line 137)
        self_52891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 19), 'self')
        # Obtaining the member 'compiler' of a type (line 137)
        compiler_52892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 19), self_52891, 'compiler')
        # Assigning a type to the variable 'compiler' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'compiler', compiler_52892)
        
        # Assigning a Attribute to a Name (line 138):
        
        # Assigning a Attribute to a Name (line 138):
        # Getting the type of 'self' (line 138)
        self_52893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 20), 'self')
        # Obtaining the member '_f_compiler' of a type (line 138)
        _f_compiler_52894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 20), self_52893, '_f_compiler')
        # Assigning a type to the variable 'fcompiler' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'fcompiler', _f_compiler_52894)
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to get(...): (line 140)
        # Processing the call arguments (line 140)
        str_52897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 33), 'str', 'sources')
        # Processing the call keyword arguments (line 140)
        kwargs_52898 = {}
        # Getting the type of 'build_info' (line 140)
        build_info_52895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 18), 'build_info', False)
        # Obtaining the member 'get' of a type (line 140)
        get_52896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 18), build_info_52895, 'get')
        # Calling get(args, kwargs) (line 140)
        get_call_result_52899 = invoke(stypy.reporting.localization.Localization(__file__, 140, 18), get_52896, *[str_52897], **kwargs_52898)
        
        # Assigning a type to the variable 'sources' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'sources', get_call_result_52899)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sources' (line 141)
        sources_52900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'sources')
        # Getting the type of 'None' (line 141)
        None_52901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 22), 'None')
        # Applying the binary operator 'is' (line 141)
        result_is__52902 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 11), 'is', sources_52900, None_52901)
        
        
        
        # Call to is_sequence(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'sources' (line 141)
        sources_52904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 46), 'sources', False)
        # Processing the call keyword arguments (line 141)
        kwargs_52905 = {}
        # Getting the type of 'is_sequence' (line 141)
        is_sequence_52903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 34), 'is_sequence', False)
        # Calling is_sequence(args, kwargs) (line 141)
        is_sequence_call_result_52906 = invoke(stypy.reporting.localization.Localization(__file__, 141, 34), is_sequence_52903, *[sources_52904], **kwargs_52905)
        
        # Applying the 'not' unary operator (line 141)
        result_not__52907 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 30), 'not', is_sequence_call_result_52906)
        
        # Applying the binary operator 'or' (line 141)
        result_or_keyword_52908 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 11), 'or', result_is__52902, result_not__52907)
        
        # Testing the type of an if condition (line 141)
        if_condition_52909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 8), result_or_keyword_52908)
        # Assigning a type to the variable 'if_condition_52909' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'if_condition_52909', if_condition_52909)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsSetupError(...): (line 142)
        # Processing the call arguments (line 142)
        str_52911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 39), 'str', "in 'libraries' option (library '%s'), ")
        str_52912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 19), 'str', "'sources' must be present and must be ")
        # Applying the binary operator '+' (line 142)
        result_add_52913 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 39), '+', str_52911, str_52912)
        
        str_52914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 19), 'str', 'a list of source filenames')
        # Applying the binary operator '+' (line 143)
        result_add_52915 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 60), '+', result_add_52913, str_52914)
        
        # Getting the type of 'lib_name' (line 144)
        lib_name_52916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 51), 'lib_name', False)
        # Applying the binary operator '%' (line 142)
        result_mod_52917 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 38), '%', result_add_52915, lib_name_52916)
        
        # Processing the call keyword arguments (line 142)
        kwargs_52918 = {}
        # Getting the type of 'DistutilsSetupError' (line 142)
        DistutilsSetupError_52910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'DistutilsSetupError', False)
        # Calling DistutilsSetupError(args, kwargs) (line 142)
        DistutilsSetupError_call_result_52919 = invoke(stypy.reporting.localization.Localization(__file__, 142, 18), DistutilsSetupError_52910, *[result_mod_52917], **kwargs_52918)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 142, 12), DistutilsSetupError_call_result_52919, 'raise parameter', BaseException)
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 145):
        
        # Assigning a Call to a Name (line 145):
        
        # Call to list(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'sources' (line 145)
        sources_52921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 23), 'sources', False)
        # Processing the call keyword arguments (line 145)
        kwargs_52922 = {}
        # Getting the type of 'list' (line 145)
        list_52920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 18), 'list', False)
        # Calling list(args, kwargs) (line 145)
        list_call_result_52923 = invoke(stypy.reporting.localization.Localization(__file__, 145, 18), list_52920, *[sources_52921], **kwargs_52922)
        
        # Assigning a type to the variable 'sources' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'sources', list_call_result_52923)
        
        # Assigning a Call to a Tuple (line 147):
        
        # Assigning a Call to a Name:
        
        # Call to filter_sources(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'sources' (line 148)
        sources_52925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 36), 'sources', False)
        # Processing the call keyword arguments (line 148)
        kwargs_52926 = {}
        # Getting the type of 'filter_sources' (line 148)
        filter_sources_52924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 21), 'filter_sources', False)
        # Calling filter_sources(args, kwargs) (line 148)
        filter_sources_call_result_52927 = invoke(stypy.reporting.localization.Localization(__file__, 148, 21), filter_sources_52924, *[sources_52925], **kwargs_52926)
        
        # Assigning a type to the variable 'call_assignment_52546' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_52546', filter_sources_call_result_52927)
        
        # Assigning a Call to a Name (line 147):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_52930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 8), 'int')
        # Processing the call keyword arguments
        kwargs_52931 = {}
        # Getting the type of 'call_assignment_52546' (line 147)
        call_assignment_52546_52928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_52546', False)
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___52929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), call_assignment_52546_52928, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_52932 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___52929, *[int_52930], **kwargs_52931)
        
        # Assigning a type to the variable 'call_assignment_52547' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_52547', getitem___call_result_52932)
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'call_assignment_52547' (line 147)
        call_assignment_52547_52933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_52547')
        # Assigning a type to the variable 'c_sources' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'c_sources', call_assignment_52547_52933)
        
        # Assigning a Call to a Name (line 147):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_52936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 8), 'int')
        # Processing the call keyword arguments
        kwargs_52937 = {}
        # Getting the type of 'call_assignment_52546' (line 147)
        call_assignment_52546_52934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_52546', False)
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___52935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), call_assignment_52546_52934, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_52938 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___52935, *[int_52936], **kwargs_52937)
        
        # Assigning a type to the variable 'call_assignment_52548' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_52548', getitem___call_result_52938)
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'call_assignment_52548' (line 147)
        call_assignment_52548_52939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_52548')
        # Assigning a type to the variable 'cxx_sources' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 19), 'cxx_sources', call_assignment_52548_52939)
        
        # Assigning a Call to a Name (line 147):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_52942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 8), 'int')
        # Processing the call keyword arguments
        kwargs_52943 = {}
        # Getting the type of 'call_assignment_52546' (line 147)
        call_assignment_52546_52940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_52546', False)
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___52941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), call_assignment_52546_52940, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_52944 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___52941, *[int_52942], **kwargs_52943)
        
        # Assigning a type to the variable 'call_assignment_52549' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_52549', getitem___call_result_52944)
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'call_assignment_52549' (line 147)
        call_assignment_52549_52945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_52549')
        # Assigning a type to the variable 'f_sources' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 32), 'f_sources', call_assignment_52549_52945)
        
        # Assigning a Call to a Name (line 147):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_52948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 8), 'int')
        # Processing the call keyword arguments
        kwargs_52949 = {}
        # Getting the type of 'call_assignment_52546' (line 147)
        call_assignment_52546_52946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_52546', False)
        # Obtaining the member '__getitem__' of a type (line 147)
        getitem___52947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), call_assignment_52546_52946, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_52950 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___52947, *[int_52948], **kwargs_52949)
        
        # Assigning a type to the variable 'call_assignment_52550' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_52550', getitem___call_result_52950)
        
        # Assigning a Name to a Name (line 147):
        # Getting the type of 'call_assignment_52550' (line 147)
        call_assignment_52550_52951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'call_assignment_52550')
        # Assigning a type to the variable 'fmodule_sources' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 43), 'fmodule_sources', call_assignment_52550_52951)
        
        # Assigning a BoolOp to a Name (line 149):
        
        # Assigning a BoolOp to a Name (line 149):
        
        # Evaluating a boolean operation
        
        
        # Getting the type of 'fmodule_sources' (line 149)
        fmodule_sources_52952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 29), 'fmodule_sources')
        # Applying the 'not' unary operator (line 149)
        result_not__52953 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 25), 'not', fmodule_sources_52952)
        
        # Applying the 'not' unary operator (line 149)
        result_not__52954 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 21), 'not', result_not__52953)
        
        
        
        # Call to get(...): (line 150)
        # Processing the call arguments (line 150)
        str_52957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 36), 'str', 'language')
        str_52958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 48), 'str', 'c')
        # Processing the call keyword arguments (line 150)
        kwargs_52959 = {}
        # Getting the type of 'build_info' (line 150)
        build_info_52955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'build_info', False)
        # Obtaining the member 'get' of a type (line 150)
        get_52956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 21), build_info_52955, 'get')
        # Calling get(args, kwargs) (line 150)
        get_call_result_52960 = invoke(stypy.reporting.localization.Localization(__file__, 150, 21), get_52956, *[str_52957, str_52958], **kwargs_52959)
        
        str_52961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 54), 'str', 'f90')
        # Applying the binary operator '==' (line 150)
        result_eq_52962 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 21), '==', get_call_result_52960, str_52961)
        
        # Applying the binary operator 'or' (line 149)
        result_or_keyword_52963 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 21), 'or', result_not__52954, result_eq_52962)
        
        # Assigning a type to the variable 'requiref90' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'requiref90', result_or_keyword_52963)
        
        # Assigning a List to a Name (line 153):
        
        # Assigning a List to a Name (line 153):
        
        # Obtaining an instance of the builtin type 'list' (line 153)
        list_52964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 153)
        
        # Assigning a type to the variable 'source_languages' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'source_languages', list_52964)
        
        # Getting the type of 'c_sources' (line 154)
        c_sources_52965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 11), 'c_sources')
        # Testing the type of an if condition (line 154)
        if_condition_52966 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 8), c_sources_52965)
        # Assigning a type to the variable 'if_condition_52966' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'if_condition_52966', if_condition_52966)
        # SSA begins for if statement (line 154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 154)
        # Processing the call arguments (line 154)
        str_52969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 46), 'str', 'c')
        # Processing the call keyword arguments (line 154)
        kwargs_52970 = {}
        # Getting the type of 'source_languages' (line 154)
        source_languages_52967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'source_languages', False)
        # Obtaining the member 'append' of a type (line 154)
        append_52968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 22), source_languages_52967, 'append')
        # Calling append(args, kwargs) (line 154)
        append_call_result_52971 = invoke(stypy.reporting.localization.Localization(__file__, 154, 22), append_52968, *[str_52969], **kwargs_52970)
        
        # SSA join for if statement (line 154)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'cxx_sources' (line 155)
        cxx_sources_52972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'cxx_sources')
        # Testing the type of an if condition (line 155)
        if_condition_52973 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 8), cxx_sources_52972)
        # Assigning a type to the variable 'if_condition_52973' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'if_condition_52973', if_condition_52973)
        # SSA begins for if statement (line 155)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 155)
        # Processing the call arguments (line 155)
        str_52976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 48), 'str', 'c++')
        # Processing the call keyword arguments (line 155)
        kwargs_52977 = {}
        # Getting the type of 'source_languages' (line 155)
        source_languages_52974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 24), 'source_languages', False)
        # Obtaining the member 'append' of a type (line 155)
        append_52975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 24), source_languages_52974, 'append')
        # Calling append(args, kwargs) (line 155)
        append_call_result_52978 = invoke(stypy.reporting.localization.Localization(__file__, 155, 24), append_52975, *[str_52976], **kwargs_52977)
        
        # SSA join for if statement (line 155)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'requiref90' (line 156)
        requiref90_52979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 11), 'requiref90')
        # Testing the type of an if condition (line 156)
        if_condition_52980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 8), requiref90_52979)
        # Assigning a type to the variable 'if_condition_52980' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'if_condition_52980', if_condition_52980)
        # SSA begins for if statement (line 156)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 156)
        # Processing the call arguments (line 156)
        str_52983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 47), 'str', 'f90')
        # Processing the call keyword arguments (line 156)
        kwargs_52984 = {}
        # Getting the type of 'source_languages' (line 156)
        source_languages_52981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 23), 'source_languages', False)
        # Obtaining the member 'append' of a type (line 156)
        append_52982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 23), source_languages_52981, 'append')
        # Calling append(args, kwargs) (line 156)
        append_call_result_52985 = invoke(stypy.reporting.localization.Localization(__file__, 156, 23), append_52982, *[str_52983], **kwargs_52984)
        
        # SSA branch for the else part of an if statement (line 156)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'f_sources' (line 157)
        f_sources_52986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 13), 'f_sources')
        # Testing the type of an if condition (line 157)
        if_condition_52987 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 13), f_sources_52986)
        # Assigning a type to the variable 'if_condition_52987' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 13), 'if_condition_52987', if_condition_52987)
        # SSA begins for if statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 157)
        # Processing the call arguments (line 157)
        str_52990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 48), 'str', 'f77')
        # Processing the call keyword arguments (line 157)
        kwargs_52991 = {}
        # Getting the type of 'source_languages' (line 157)
        source_languages_52988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'source_languages', False)
        # Obtaining the member 'append' of a type (line 157)
        append_52989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 24), source_languages_52988, 'append')
        # Calling append(args, kwargs) (line 157)
        append_call_result_52992 = invoke(stypy.reporting.localization.Localization(__file__, 157, 24), append_52989, *[str_52990], **kwargs_52991)
        
        # SSA join for if statement (line 157)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 156)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 158):
        
        # Assigning a Name to a Subscript (line 158):
        # Getting the type of 'source_languages' (line 158)
        source_languages_52993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 41), 'source_languages')
        # Getting the type of 'build_info' (line 158)
        build_info_52994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'build_info')
        str_52995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 19), 'str', 'source_languages')
        # Storing an element on a container (line 158)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 8), build_info_52994, (str_52995, source_languages_52993))
        
        # Assigning a Call to a Name (line 160):
        
        # Assigning a Call to a Name (line 160):
        
        # Call to library_filename(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'lib_name' (line 160)
        lib_name_52998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 45), 'lib_name', False)
        # Processing the call keyword arguments (line 160)
        # Getting the type of 'self' (line 161)
        self_52999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 56), 'self', False)
        # Obtaining the member 'build_clib' of a type (line 161)
        build_clib_53000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 56), self_52999, 'build_clib')
        keyword_53001 = build_clib_53000
        kwargs_53002 = {'output_dir': keyword_53001}
        # Getting the type of 'compiler' (line 160)
        compiler_52996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 19), 'compiler', False)
        # Obtaining the member 'library_filename' of a type (line 160)
        library_filename_52997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 19), compiler_52996, 'library_filename')
        # Calling library_filename(args, kwargs) (line 160)
        library_filename_call_result_53003 = invoke(stypy.reporting.localization.Localization(__file__, 160, 19), library_filename_52997, *[lib_name_52998], **kwargs_53002)
        
        # Assigning a type to the variable 'lib_file' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'lib_file', library_filename_call_result_53003)
        
        # Assigning a BinOp to a Name (line 162):
        
        # Assigning a BinOp to a Name (line 162):
        # Getting the type of 'sources' (line 162)
        sources_53004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 18), 'sources')
        
        # Call to get(...): (line 162)
        # Processing the call arguments (line 162)
        str_53007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 43), 'str', 'depends')
        
        # Obtaining an instance of the builtin type 'list' (line 162)
        list_53008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 54), 'list')
        # Adding type elements to the builtin type 'list' instance (line 162)
        
        # Processing the call keyword arguments (line 162)
        kwargs_53009 = {}
        # Getting the type of 'build_info' (line 162)
        build_info_53005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 28), 'build_info', False)
        # Obtaining the member 'get' of a type (line 162)
        get_53006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 28), build_info_53005, 'get')
        # Calling get(args, kwargs) (line 162)
        get_call_result_53010 = invoke(stypy.reporting.localization.Localization(__file__, 162, 28), get_53006, *[str_53007, list_53008], **kwargs_53009)
        
        # Applying the binary operator '+' (line 162)
        result_add_53011 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 18), '+', sources_53004, get_call_result_53010)
        
        # Assigning a type to the variable 'depends' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'depends', result_add_53011)
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 163)
        self_53012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'self')
        # Obtaining the member 'force' of a type (line 163)
        force_53013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 16), self_53012, 'force')
        
        # Call to newer_group(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'depends' (line 163)
        depends_53015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 42), 'depends', False)
        # Getting the type of 'lib_file' (line 163)
        lib_file_53016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 51), 'lib_file', False)
        str_53017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 61), 'str', 'newer')
        # Processing the call keyword arguments (line 163)
        kwargs_53018 = {}
        # Getting the type of 'newer_group' (line 163)
        newer_group_53014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 30), 'newer_group', False)
        # Calling newer_group(args, kwargs) (line 163)
        newer_group_call_result_53019 = invoke(stypy.reporting.localization.Localization(__file__, 163, 30), newer_group_53014, *[depends_53015, lib_file_53016, str_53017], **kwargs_53018)
        
        # Applying the binary operator 'or' (line 163)
        result_or_keyword_53020 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 16), 'or', force_53013, newer_group_call_result_53019)
        
        # Applying the 'not' unary operator (line 163)
        result_not__53021 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 11), 'not', result_or_keyword_53020)
        
        # Testing the type of an if condition (line 163)
        if_condition_53022 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 8), result_not__53021)
        # Assigning a type to the variable 'if_condition_53022' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'if_condition_53022', if_condition_53022)
        # SSA begins for if statement (line 163)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug(...): (line 164)
        # Processing the call arguments (line 164)
        str_53025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 22), 'str', "skipping '%s' library (up-to-date)")
        # Getting the type of 'lib_name' (line 164)
        lib_name_53026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 60), 'lib_name', False)
        # Processing the call keyword arguments (line 164)
        kwargs_53027 = {}
        # Getting the type of 'log' (line 164)
        log_53023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 164)
        debug_53024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 12), log_53023, 'debug')
        # Calling debug(args, kwargs) (line 164)
        debug_call_result_53028 = invoke(stypy.reporting.localization.Localization(__file__, 164, 12), debug_53024, *[str_53025, lib_name_53026], **kwargs_53027)
        
        # Assigning a type to the variable 'stypy_return_type' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'stypy_return_type', types.NoneType)
        # SSA branch for the else part of an if statement (line 163)
        module_type_store.open_ssa_branch('else')
        
        # Call to info(...): (line 167)
        # Processing the call arguments (line 167)
        str_53031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 21), 'str', "building '%s' library")
        # Getting the type of 'lib_name' (line 167)
        lib_name_53032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 46), 'lib_name', False)
        # Processing the call keyword arguments (line 167)
        kwargs_53033 = {}
        # Getting the type of 'log' (line 167)
        log_53029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 167)
        info_53030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 12), log_53029, 'info')
        # Calling info(args, kwargs) (line 167)
        info_call_result_53034 = invoke(stypy.reporting.localization.Localization(__file__, 167, 12), info_53030, *[str_53031, lib_name_53032], **kwargs_53033)
        
        # SSA join for if statement (line 163)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 169):
        
        # Assigning a Call to a Name (line 169):
        
        # Call to get(...): (line 169)
        # Processing the call arguments (line 169)
        str_53037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 35), 'str', 'config_fc')
        
        # Obtaining an instance of the builtin type 'dict' (line 169)
        dict_53038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 48), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 169)
        
        # Processing the call keyword arguments (line 169)
        kwargs_53039 = {}
        # Getting the type of 'build_info' (line 169)
        build_info_53035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 20), 'build_info', False)
        # Obtaining the member 'get' of a type (line 169)
        get_53036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 20), build_info_53035, 'get')
        # Calling get(args, kwargs) (line 169)
        get_call_result_53040 = invoke(stypy.reporting.localization.Localization(__file__, 169, 20), get_53036, *[str_53037, dict_53038], **kwargs_53039)
        
        # Assigning a type to the variable 'config_fc' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'config_fc', get_call_result_53040)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'fcompiler' (line 170)
        fcompiler_53041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'fcompiler')
        # Getting the type of 'None' (line 170)
        None_53042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 28), 'None')
        # Applying the binary operator 'isnot' (line 170)
        result_is_not_53043 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 11), 'isnot', fcompiler_53041, None_53042)
        
        # Getting the type of 'config_fc' (line 170)
        config_fc_53044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 37), 'config_fc')
        # Applying the binary operator 'and' (line 170)
        result_and_keyword_53045 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 11), 'and', result_is_not_53043, config_fc_53044)
        
        # Testing the type of an if condition (line 170)
        if_condition_53046 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 8), result_and_keyword_53045)
        # Assigning a type to the variable 'if_condition_53046' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'if_condition_53046', if_condition_53046)
        # SSA begins for if statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 171)
        # Processing the call arguments (line 171)
        str_53049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 21), 'str', 'using additional config_fc from setup script for fortran compiler: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 173)
        tuple_53050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 24), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 173)
        # Adding element type (line 173)
        # Getting the type of 'config_fc' (line 173)
        config_fc_53051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 24), 'config_fc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 24), tuple_53050, config_fc_53051)
        
        # Applying the binary operator '%' (line 171)
        result_mod_53052 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 21), '%', str_53049, tuple_53050)
        
        # Processing the call keyword arguments (line 171)
        kwargs_53053 = {}
        # Getting the type of 'log' (line 171)
        log_53047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 171)
        info_53048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 12), log_53047, 'info')
        # Calling info(args, kwargs) (line 171)
        info_call_result_53054 = invoke(stypy.reporting.localization.Localization(__file__, 171, 12), info_53048, *[result_mod_53052], **kwargs_53053)
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 174, 12))
        
        # 'from numpy.distutils.fcompiler import new_fcompiler' statement (line 174)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
        import_53055 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 174, 12), 'numpy.distutils.fcompiler')

        if (type(import_53055) is not StypyTypeError):

            if (import_53055 != 'pyd_module'):
                __import__(import_53055)
                sys_modules_53056 = sys.modules[import_53055]
                import_from_module(stypy.reporting.localization.Localization(__file__, 174, 12), 'numpy.distutils.fcompiler', sys_modules_53056.module_type_store, module_type_store, ['new_fcompiler'])
                nest_module(stypy.reporting.localization.Localization(__file__, 174, 12), __file__, sys_modules_53056, sys_modules_53056.module_type_store, module_type_store)
            else:
                from numpy.distutils.fcompiler import new_fcompiler

                import_from_module(stypy.reporting.localization.Localization(__file__, 174, 12), 'numpy.distutils.fcompiler', None, module_type_store, ['new_fcompiler'], [new_fcompiler])

        else:
            # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 174)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'numpy.distutils.fcompiler', import_53055)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')
        
        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to new_fcompiler(...): (line 175)
        # Processing the call keyword arguments (line 175)
        # Getting the type of 'fcompiler' (line 175)
        fcompiler_53058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 47), 'fcompiler', False)
        # Obtaining the member 'compiler_type' of a type (line 175)
        compiler_type_53059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 47), fcompiler_53058, 'compiler_type')
        keyword_53060 = compiler_type_53059
        # Getting the type of 'self' (line 176)
        self_53061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 46), 'self', False)
        # Obtaining the member 'verbose' of a type (line 176)
        verbose_53062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 46), self_53061, 'verbose')
        keyword_53063 = verbose_53062
        # Getting the type of 'self' (line 177)
        self_53064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 46), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 177)
        dry_run_53065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 46), self_53064, 'dry_run')
        keyword_53066 = dry_run_53065
        # Getting the type of 'self' (line 178)
        self_53067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 44), 'self', False)
        # Obtaining the member 'force' of a type (line 178)
        force_53068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 44), self_53067, 'force')
        keyword_53069 = force_53068
        # Getting the type of 'requiref90' (line 179)
        requiref90_53070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 49), 'requiref90', False)
        keyword_53071 = requiref90_53070
        # Getting the type of 'self' (line 180)
        self_53072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 49), 'self', False)
        # Obtaining the member 'compiler' of a type (line 180)
        compiler_53073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 49), self_53072, 'compiler')
        keyword_53074 = compiler_53073
        kwargs_53075 = {'force': keyword_53069, 'verbose': keyword_53063, 'dry_run': keyword_53066, 'c_compiler': keyword_53074, 'requiref90': keyword_53071, 'compiler': keyword_53060}
        # Getting the type of 'new_fcompiler' (line 175)
        new_fcompiler_53057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 24), 'new_fcompiler', False)
        # Calling new_fcompiler(args, kwargs) (line 175)
        new_fcompiler_call_result_53076 = invoke(stypy.reporting.localization.Localization(__file__, 175, 24), new_fcompiler_53057, *[], **kwargs_53075)
        
        # Assigning a type to the variable 'fcompiler' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'fcompiler', new_fcompiler_call_result_53076)
        
        # Type idiom detected: calculating its left and rigth part (line 181)
        # Getting the type of 'fcompiler' (line 181)
        fcompiler_53077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 12), 'fcompiler')
        # Getting the type of 'None' (line 181)
        None_53078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 32), 'None')
        
        (may_be_53079, more_types_in_union_53080) = may_not_be_none(fcompiler_53077, None_53078)

        if may_be_53079:

            if more_types_in_union_53080:
                # Runtime conditional SSA (line 181)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 182):
            
            # Assigning a Attribute to a Name (line 182):
            # Getting the type of 'self' (line 182)
            self_53081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 23), 'self')
            # Obtaining the member 'distribution' of a type (line 182)
            distribution_53082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 23), self_53081, 'distribution')
            # Assigning a type to the variable 'dist' (line 182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'dist', distribution_53082)
            
            # Assigning a Call to a Name (line 183):
            
            # Assigning a Call to a Name (line 183):
            
            # Call to copy(...): (line 183)
            # Processing the call keyword arguments (line 183)
            kwargs_53089 = {}
            
            # Call to get_option_dict(...): (line 183)
            # Processing the call arguments (line 183)
            str_53085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 54), 'str', 'config_fc')
            # Processing the call keyword arguments (line 183)
            kwargs_53086 = {}
            # Getting the type of 'dist' (line 183)
            dist_53083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 33), 'dist', False)
            # Obtaining the member 'get_option_dict' of a type (line 183)
            get_option_dict_53084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 33), dist_53083, 'get_option_dict')
            # Calling get_option_dict(args, kwargs) (line 183)
            get_option_dict_call_result_53087 = invoke(stypy.reporting.localization.Localization(__file__, 183, 33), get_option_dict_53084, *[str_53085], **kwargs_53086)
            
            # Obtaining the member 'copy' of a type (line 183)
            copy_53088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 33), get_option_dict_call_result_53087, 'copy')
            # Calling copy(args, kwargs) (line 183)
            copy_call_result_53090 = invoke(stypy.reporting.localization.Localization(__file__, 183, 33), copy_53088, *[], **kwargs_53089)
            
            # Assigning a type to the variable 'base_config_fc' (line 183)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'base_config_fc', copy_call_result_53090)
            
            # Call to update(...): (line 184)
            # Processing the call arguments (line 184)
            # Getting the type of 'config_fc' (line 184)
            config_fc_53093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 38), 'config_fc', False)
            # Processing the call keyword arguments (line 184)
            kwargs_53094 = {}
            # Getting the type of 'base_config_fc' (line 184)
            base_config_fc_53091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'base_config_fc', False)
            # Obtaining the member 'update' of a type (line 184)
            update_53092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 16), base_config_fc_53091, 'update')
            # Calling update(args, kwargs) (line 184)
            update_call_result_53095 = invoke(stypy.reporting.localization.Localization(__file__, 184, 16), update_53092, *[config_fc_53093], **kwargs_53094)
            
            
            # Call to customize(...): (line 185)
            # Processing the call arguments (line 185)
            # Getting the type of 'base_config_fc' (line 185)
            base_config_fc_53098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 36), 'base_config_fc', False)
            # Processing the call keyword arguments (line 185)
            kwargs_53099 = {}
            # Getting the type of 'fcompiler' (line 185)
            fcompiler_53096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'fcompiler', False)
            # Obtaining the member 'customize' of a type (line 185)
            customize_53097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 16), fcompiler_53096, 'customize')
            # Calling customize(args, kwargs) (line 185)
            customize_call_result_53100 = invoke(stypy.reporting.localization.Localization(__file__, 185, 16), customize_53097, *[base_config_fc_53098], **kwargs_53099)
            

            if more_types_in_union_53080:
                # SSA join for if statement (line 181)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 170)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'f_sources' (line 188)
        f_sources_53101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'f_sources')
        # Getting the type of 'fmodule_sources' (line 188)
        fmodule_sources_53102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 25), 'fmodule_sources')
        # Applying the binary operator 'or' (line 188)
        result_or_keyword_53103 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 12), 'or', f_sources_53101, fmodule_sources_53102)
        
        
        # Getting the type of 'fcompiler' (line 188)
        fcompiler_53104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 46), 'fcompiler')
        # Getting the type of 'None' (line 188)
        None_53105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 59), 'None')
        # Applying the binary operator 'is' (line 188)
        result_is__53106 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 46), 'is', fcompiler_53104, None_53105)
        
        # Applying the binary operator 'and' (line 188)
        result_and_keyword_53107 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 11), 'and', result_or_keyword_53103, result_is__53106)
        
        # Testing the type of an if condition (line 188)
        if_condition_53108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 8), result_and_keyword_53107)
        # Assigning a type to the variable 'if_condition_53108' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'if_condition_53108', if_condition_53108)
        # SSA begins for if statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsError(...): (line 189)
        # Processing the call arguments (line 189)
        str_53110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 33), 'str', 'library %s has Fortran sources but no Fortran compiler found')
        # Getting the type of 'lib_name' (line 190)
        lib_name_53111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 54), 'lib_name', False)
        # Applying the binary operator '%' (line 189)
        result_mod_53112 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 33), '%', str_53110, lib_name_53111)
        
        # Processing the call keyword arguments (line 189)
        kwargs_53113 = {}
        # Getting the type of 'DistutilsError' (line 189)
        DistutilsError_53109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 18), 'DistutilsError', False)
        # Calling DistutilsError(args, kwargs) (line 189)
        DistutilsError_call_result_53114 = invoke(stypy.reporting.localization.Localization(__file__, 189, 18), DistutilsError_53109, *[result_mod_53112], **kwargs_53113)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 189, 12), DistutilsError_call_result_53114, 'raise parameter', BaseException)
        # SSA join for if statement (line 188)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 192)
        # Getting the type of 'fcompiler' (line 192)
        fcompiler_53115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'fcompiler')
        # Getting the type of 'None' (line 192)
        None_53116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 28), 'None')
        
        (may_be_53117, more_types_in_union_53118) = may_not_be_none(fcompiler_53115, None_53116)

        if may_be_53117:

            if more_types_in_union_53118:
                # Runtime conditional SSA (line 192)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BoolOp to a Attribute (line 193):
            
            # Assigning a BoolOp to a Attribute (line 193):
            
            # Evaluating a boolean operation
            
            # Call to get(...): (line 193)
            # Processing the call arguments (line 193)
            str_53121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 62), 'str', 'extra_f77_compile_args')
            # Processing the call keyword arguments (line 193)
            kwargs_53122 = {}
            # Getting the type of 'build_info' (line 193)
            build_info_53119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 47), 'build_info', False)
            # Obtaining the member 'get' of a type (line 193)
            get_53120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 47), build_info_53119, 'get')
            # Calling get(args, kwargs) (line 193)
            get_call_result_53123 = invoke(stypy.reporting.localization.Localization(__file__, 193, 47), get_53120, *[str_53121], **kwargs_53122)
            
            
            # Obtaining an instance of the builtin type 'list' (line 193)
            list_53124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 91), 'list')
            # Adding type elements to the builtin type 'list' instance (line 193)
            
            # Applying the binary operator 'or' (line 193)
            result_or_keyword_53125 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 47), 'or', get_call_result_53123, list_53124)
            
            # Getting the type of 'fcompiler' (line 193)
            fcompiler_53126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'fcompiler')
            # Setting the type of the member 'extra_f77_compile_args' of a type (line 193)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), fcompiler_53126, 'extra_f77_compile_args', result_or_keyword_53125)
            
            # Assigning a BoolOp to a Attribute (line 194):
            
            # Assigning a BoolOp to a Attribute (line 194):
            
            # Evaluating a boolean operation
            
            # Call to get(...): (line 194)
            # Processing the call arguments (line 194)
            str_53129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 62), 'str', 'extra_f90_compile_args')
            # Processing the call keyword arguments (line 194)
            kwargs_53130 = {}
            # Getting the type of 'build_info' (line 194)
            build_info_53127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 47), 'build_info', False)
            # Obtaining the member 'get' of a type (line 194)
            get_53128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 47), build_info_53127, 'get')
            # Calling get(args, kwargs) (line 194)
            get_call_result_53131 = invoke(stypy.reporting.localization.Localization(__file__, 194, 47), get_53128, *[str_53129], **kwargs_53130)
            
            
            # Obtaining an instance of the builtin type 'list' (line 194)
            list_53132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 91), 'list')
            # Adding type elements to the builtin type 'list' instance (line 194)
            
            # Applying the binary operator 'or' (line 194)
            result_or_keyword_53133 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 47), 'or', get_call_result_53131, list_53132)
            
            # Getting the type of 'fcompiler' (line 194)
            fcompiler_53134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'fcompiler')
            # Setting the type of the member 'extra_f90_compile_args' of a type (line 194)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), fcompiler_53134, 'extra_f90_compile_args', result_or_keyword_53133)

            if more_types_in_union_53118:
                # SSA join for if statement (line 192)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to get(...): (line 196)
        # Processing the call arguments (line 196)
        str_53137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 32), 'str', 'macros')
        # Processing the call keyword arguments (line 196)
        kwargs_53138 = {}
        # Getting the type of 'build_info' (line 196)
        build_info_53135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 17), 'build_info', False)
        # Obtaining the member 'get' of a type (line 196)
        get_53136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 17), build_info_53135, 'get')
        # Calling get(args, kwargs) (line 196)
        get_call_result_53139 = invoke(stypy.reporting.localization.Localization(__file__, 196, 17), get_53136, *[str_53137], **kwargs_53138)
        
        # Assigning a type to the variable 'macros' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'macros', get_call_result_53139)
        
        # Assigning a Call to a Name (line 197):
        
        # Assigning a Call to a Name (line 197):
        
        # Call to get(...): (line 197)
        # Processing the call arguments (line 197)
        str_53142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 38), 'str', 'include_dirs')
        # Processing the call keyword arguments (line 197)
        kwargs_53143 = {}
        # Getting the type of 'build_info' (line 197)
        build_info_53140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 23), 'build_info', False)
        # Obtaining the member 'get' of a type (line 197)
        get_53141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 23), build_info_53140, 'get')
        # Calling get(args, kwargs) (line 197)
        get_call_result_53144 = invoke(stypy.reporting.localization.Localization(__file__, 197, 23), get_53141, *[str_53142], **kwargs_53143)
        
        # Assigning a type to the variable 'include_dirs' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'include_dirs', get_call_result_53144)
        
        # Type idiom detected: calculating its left and rigth part (line 198)
        # Getting the type of 'include_dirs' (line 198)
        include_dirs_53145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 'include_dirs')
        # Getting the type of 'None' (line 198)
        None_53146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 27), 'None')
        
        (may_be_53147, more_types_in_union_53148) = may_be_none(include_dirs_53145, None_53146)

        if may_be_53147:

            if more_types_in_union_53148:
                # Runtime conditional SSA (line 198)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Name (line 199):
            
            # Assigning a List to a Name (line 199):
            
            # Obtaining an instance of the builtin type 'list' (line 199)
            list_53149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 27), 'list')
            # Adding type elements to the builtin type 'list' instance (line 199)
            
            # Assigning a type to the variable 'include_dirs' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'include_dirs', list_53149)

            if more_types_in_union_53148:
                # SSA join for if statement (line 198)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BoolOp to a Name (line 200):
        
        # Assigning a BoolOp to a Name (line 200):
        
        # Evaluating a boolean operation
        
        # Call to get(...): (line 200)
        # Processing the call arguments (line 200)
        str_53152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 40), 'str', 'extra_compiler_args')
        # Processing the call keyword arguments (line 200)
        kwargs_53153 = {}
        # Getting the type of 'build_info' (line 200)
        build_info_53150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 25), 'build_info', False)
        # Obtaining the member 'get' of a type (line 200)
        get_53151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 25), build_info_53150, 'get')
        # Calling get(args, kwargs) (line 200)
        get_call_result_53154 = invoke(stypy.reporting.localization.Localization(__file__, 200, 25), get_53151, *[str_53152], **kwargs_53153)
        
        
        # Obtaining an instance of the builtin type 'list' (line 200)
        list_53155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 66), 'list')
        # Adding type elements to the builtin type 'list' instance (line 200)
        
        # Applying the binary operator 'or' (line 200)
        result_or_keyword_53156 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 25), 'or', get_call_result_53154, list_53155)
        
        # Assigning a type to the variable 'extra_postargs' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'extra_postargs', result_or_keyword_53156)
        
        # Call to extend(...): (line 202)
        # Processing the call arguments (line 202)
        
        # Call to get_numpy_include_dirs(...): (line 202)
        # Processing the call keyword arguments (line 202)
        kwargs_53160 = {}
        # Getting the type of 'get_numpy_include_dirs' (line 202)
        get_numpy_include_dirs_53159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 28), 'get_numpy_include_dirs', False)
        # Calling get_numpy_include_dirs(args, kwargs) (line 202)
        get_numpy_include_dirs_call_result_53161 = invoke(stypy.reporting.localization.Localization(__file__, 202, 28), get_numpy_include_dirs_53159, *[], **kwargs_53160)
        
        # Processing the call keyword arguments (line 202)
        kwargs_53162 = {}
        # Getting the type of 'include_dirs' (line 202)
        include_dirs_53157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'include_dirs', False)
        # Obtaining the member 'extend' of a type (line 202)
        extend_53158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), include_dirs_53157, 'extend')
        # Calling extend(args, kwargs) (line 202)
        extend_call_result_53163 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), extend_53158, *[get_numpy_include_dirs_call_result_53161], **kwargs_53162)
        
        
        # Assigning a BoolOp to a Name (line 204):
        
        # Assigning a BoolOp to a Name (line 204):
        
        # Evaluating a boolean operation
        
        # Call to get(...): (line 204)
        # Processing the call arguments (line 204)
        str_53166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 37), 'str', 'module_dirs')
        # Processing the call keyword arguments (line 204)
        kwargs_53167 = {}
        # Getting the type of 'build_info' (line 204)
        build_info_53164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 22), 'build_info', False)
        # Obtaining the member 'get' of a type (line 204)
        get_53165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 22), build_info_53164, 'get')
        # Calling get(args, kwargs) (line 204)
        get_call_result_53168 = invoke(stypy.reporting.localization.Localization(__file__, 204, 22), get_53165, *[str_53166], **kwargs_53167)
        
        
        # Obtaining an instance of the builtin type 'list' (line 204)
        list_53169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 55), 'list')
        # Adding type elements to the builtin type 'list' instance (line 204)
        
        # Applying the binary operator 'or' (line 204)
        result_or_keyword_53170 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 22), 'or', get_call_result_53168, list_53169)
        
        # Assigning a type to the variable 'module_dirs' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'module_dirs', result_or_keyword_53170)
        
        # Assigning a Call to a Name (line 205):
        
        # Assigning a Call to a Name (line 205):
        
        # Call to dirname(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'lib_file' (line 205)
        lib_file_53174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 43), 'lib_file', False)
        # Processing the call keyword arguments (line 205)
        kwargs_53175 = {}
        # Getting the type of 'os' (line 205)
        os_53171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 205)
        path_53172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 27), os_53171, 'path')
        # Obtaining the member 'dirname' of a type (line 205)
        dirname_53173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 27), path_53172, 'dirname')
        # Calling dirname(args, kwargs) (line 205)
        dirname_call_result_53176 = invoke(stypy.reporting.localization.Localization(__file__, 205, 27), dirname_53173, *[lib_file_53174], **kwargs_53175)
        
        # Assigning a type to the variable 'module_build_dir' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'module_build_dir', dirname_call_result_53176)
        
        # Getting the type of 'requiref90' (line 206)
        requiref90_53177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'requiref90')
        # Testing the type of an if condition (line 206)
        if_condition_53178 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 8), requiref90_53177)
        # Assigning a type to the variable 'if_condition_53178' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'if_condition_53178', if_condition_53178)
        # SSA begins for if statement (line 206)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to mkpath(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'module_build_dir' (line 206)
        module_build_dir_53181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 35), 'module_build_dir', False)
        # Processing the call keyword arguments (line 206)
        kwargs_53182 = {}
        # Getting the type of 'self' (line 206)
        self_53179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 23), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 206)
        mkpath_53180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 23), self_53179, 'mkpath')
        # Calling mkpath(args, kwargs) (line 206)
        mkpath_call_result_53183 = invoke(stypy.reporting.localization.Localization(__file__, 206, 23), mkpath_53180, *[module_build_dir_53181], **kwargs_53182)
        
        # SSA join for if statement (line 206)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'compiler' (line 208)
        compiler_53184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), 'compiler')
        # Obtaining the member 'compiler_type' of a type (line 208)
        compiler_type_53185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 11), compiler_53184, 'compiler_type')
        str_53186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 35), 'str', 'msvc')
        # Applying the binary operator '==' (line 208)
        result_eq_53187 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 11), '==', compiler_type_53185, str_53186)
        
        # Testing the type of an if condition (line 208)
        if_condition_53188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 8), result_eq_53187)
        # Assigning a type to the variable 'if_condition_53188' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'if_condition_53188', if_condition_53188)
        # SSA begins for if statement (line 208)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'c_sources' (line 211)
        c_sources_53189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'c_sources')
        # Getting the type of 'cxx_sources' (line 211)
        cxx_sources_53190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 25), 'cxx_sources')
        # Applying the binary operator '+=' (line 211)
        result_iadd_53191 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 12), '+=', c_sources_53189, cxx_sources_53190)
        # Assigning a type to the variable 'c_sources' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'c_sources', result_iadd_53191)
        
        
        # Assigning a List to a Name (line 212):
        
        # Assigning a List to a Name (line 212):
        
        # Obtaining an instance of the builtin type 'list' (line 212)
        list_53192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 212)
        
        # Assigning a type to the variable 'cxx_sources' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'cxx_sources', list_53192)
        # SSA join for if statement (line 208)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 214):
        
        # Assigning a List to a Name (line 214):
        
        # Obtaining an instance of the builtin type 'list' (line 214)
        list_53193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 214)
        
        # Assigning a type to the variable 'objects' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'objects', list_53193)
        
        # Getting the type of 'c_sources' (line 215)
        c_sources_53194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'c_sources')
        # Testing the type of an if condition (line 215)
        if_condition_53195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 8), c_sources_53194)
        # Assigning a type to the variable 'if_condition_53195' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'if_condition_53195', if_condition_53195)
        # SSA begins for if statement (line 215)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 216)
        # Processing the call arguments (line 216)
        str_53198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 21), 'str', 'compiling C sources')
        # Processing the call keyword arguments (line 216)
        kwargs_53199 = {}
        # Getting the type of 'log' (line 216)
        log_53196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 216)
        info_53197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), log_53196, 'info')
        # Calling info(args, kwargs) (line 216)
        info_call_result_53200 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), info_53197, *[str_53198], **kwargs_53199)
        
        
        # Assigning a Call to a Name (line 217):
        
        # Assigning a Call to a Name (line 217):
        
        # Call to compile(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'c_sources' (line 217)
        c_sources_53203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 39), 'c_sources', False)
        # Processing the call keyword arguments (line 217)
        # Getting the type of 'self' (line 218)
        self_53204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 50), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 218)
        build_temp_53205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 50), self_53204, 'build_temp')
        keyword_53206 = build_temp_53205
        # Getting the type of 'macros' (line 219)
        macros_53207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 46), 'macros', False)
        keyword_53208 = macros_53207
        # Getting the type of 'include_dirs' (line 220)
        include_dirs_53209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 52), 'include_dirs', False)
        keyword_53210 = include_dirs_53209
        # Getting the type of 'self' (line 221)
        self_53211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 45), 'self', False)
        # Obtaining the member 'debug' of a type (line 221)
        debug_53212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 45), self_53211, 'debug')
        keyword_53213 = debug_53212
        # Getting the type of 'extra_postargs' (line 222)
        extra_postargs_53214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 54), 'extra_postargs', False)
        keyword_53215 = extra_postargs_53214
        kwargs_53216 = {'debug': keyword_53213, 'macros': keyword_53208, 'extra_postargs': keyword_53215, 'output_dir': keyword_53206, 'include_dirs': keyword_53210}
        # Getting the type of 'compiler' (line 217)
        compiler_53201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 22), 'compiler', False)
        # Obtaining the member 'compile' of a type (line 217)
        compile_53202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 22), compiler_53201, 'compile')
        # Calling compile(args, kwargs) (line 217)
        compile_call_result_53217 = invoke(stypy.reporting.localization.Localization(__file__, 217, 22), compile_53202, *[c_sources_53203], **kwargs_53216)
        
        # Assigning a type to the variable 'objects' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'objects', compile_call_result_53217)
        # SSA join for if statement (line 215)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'cxx_sources' (line 224)
        cxx_sources_53218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 11), 'cxx_sources')
        # Testing the type of an if condition (line 224)
        if_condition_53219 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 8), cxx_sources_53218)
        # Assigning a type to the variable 'if_condition_53219' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'if_condition_53219', if_condition_53219)
        # SSA begins for if statement (line 224)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 225)
        # Processing the call arguments (line 225)
        str_53222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 21), 'str', 'compiling C++ sources')
        # Processing the call keyword arguments (line 225)
        kwargs_53223 = {}
        # Getting the type of 'log' (line 225)
        log_53220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 225)
        info_53221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 12), log_53220, 'info')
        # Calling info(args, kwargs) (line 225)
        info_call_result_53224 = invoke(stypy.reporting.localization.Localization(__file__, 225, 12), info_53221, *[str_53222], **kwargs_53223)
        
        
        # Assigning a Call to a Name (line 226):
        
        # Assigning a Call to a Name (line 226):
        
        # Call to cxx_compiler(...): (line 226)
        # Processing the call keyword arguments (line 226)
        kwargs_53227 = {}
        # Getting the type of 'compiler' (line 226)
        compiler_53225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 27), 'compiler', False)
        # Obtaining the member 'cxx_compiler' of a type (line 226)
        cxx_compiler_53226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 27), compiler_53225, 'cxx_compiler')
        # Calling cxx_compiler(args, kwargs) (line 226)
        cxx_compiler_call_result_53228 = invoke(stypy.reporting.localization.Localization(__file__, 226, 27), cxx_compiler_53226, *[], **kwargs_53227)
        
        # Assigning a type to the variable 'cxx_compiler' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'cxx_compiler', cxx_compiler_call_result_53228)
        
        # Assigning a Call to a Name (line 227):
        
        # Assigning a Call to a Name (line 227):
        
        # Call to compile(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'cxx_sources' (line 227)
        cxx_sources_53231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 47), 'cxx_sources', False)
        # Processing the call keyword arguments (line 227)
        # Getting the type of 'self' (line 228)
        self_53232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 58), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 228)
        build_temp_53233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 58), self_53232, 'build_temp')
        keyword_53234 = build_temp_53233
        # Getting the type of 'macros' (line 229)
        macros_53235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 54), 'macros', False)
        keyword_53236 = macros_53235
        # Getting the type of 'include_dirs' (line 230)
        include_dirs_53237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 60), 'include_dirs', False)
        keyword_53238 = include_dirs_53237
        # Getting the type of 'self' (line 231)
        self_53239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 53), 'self', False)
        # Obtaining the member 'debug' of a type (line 231)
        debug_53240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 53), self_53239, 'debug')
        keyword_53241 = debug_53240
        # Getting the type of 'extra_postargs' (line 232)
        extra_postargs_53242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 62), 'extra_postargs', False)
        keyword_53243 = extra_postargs_53242
        kwargs_53244 = {'debug': keyword_53241, 'macros': keyword_53236, 'extra_postargs': keyword_53243, 'output_dir': keyword_53234, 'include_dirs': keyword_53238}
        # Getting the type of 'cxx_compiler' (line 227)
        cxx_compiler_53229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 26), 'cxx_compiler', False)
        # Obtaining the member 'compile' of a type (line 227)
        compile_53230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 26), cxx_compiler_53229, 'compile')
        # Calling compile(args, kwargs) (line 227)
        compile_call_result_53245 = invoke(stypy.reporting.localization.Localization(__file__, 227, 26), compile_53230, *[cxx_sources_53231], **kwargs_53244)
        
        # Assigning a type to the variable 'cxx_objects' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'cxx_objects', compile_call_result_53245)
        
        # Call to extend(...): (line 233)
        # Processing the call arguments (line 233)
        # Getting the type of 'cxx_objects' (line 233)
        cxx_objects_53248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 27), 'cxx_objects', False)
        # Processing the call keyword arguments (line 233)
        kwargs_53249 = {}
        # Getting the type of 'objects' (line 233)
        objects_53246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 12), 'objects', False)
        # Obtaining the member 'extend' of a type (line 233)
        extend_53247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 12), objects_53246, 'extend')
        # Calling extend(args, kwargs) (line 233)
        extend_call_result_53250 = invoke(stypy.reporting.localization.Localization(__file__, 233, 12), extend_53247, *[cxx_objects_53248], **kwargs_53249)
        
        # SSA join for if statement (line 224)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'f_sources' (line 235)
        f_sources_53251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 11), 'f_sources')
        # Getting the type of 'fmodule_sources' (line 235)
        fmodule_sources_53252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 24), 'fmodule_sources')
        # Applying the binary operator 'or' (line 235)
        result_or_keyword_53253 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 11), 'or', f_sources_53251, fmodule_sources_53252)
        
        # Testing the type of an if condition (line 235)
        if_condition_53254 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 8), result_or_keyword_53253)
        # Assigning a type to the variable 'if_condition_53254' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'if_condition_53254', if_condition_53254)
        # SSA begins for if statement (line 235)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 236):
        
        # Assigning a List to a Name (line 236):
        
        # Obtaining an instance of the builtin type 'list' (line 236)
        list_53255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 236)
        
        # Assigning a type to the variable 'extra_postargs' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 12), 'extra_postargs', list_53255)
        
        # Assigning a List to a Name (line 237):
        
        # Assigning a List to a Name (line 237):
        
        # Obtaining an instance of the builtin type 'list' (line 237)
        list_53256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 237)
        
        # Assigning a type to the variable 'f_objects' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'f_objects', list_53256)
        
        # Getting the type of 'requiref90' (line 239)
        requiref90_53257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 15), 'requiref90')
        # Testing the type of an if condition (line 239)
        if_condition_53258 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 12), requiref90_53257)
        # Assigning a type to the variable 'if_condition_53258' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'if_condition_53258', if_condition_53258)
        # SSA begins for if statement (line 239)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 240)
        # Getting the type of 'fcompiler' (line 240)
        fcompiler_53259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 19), 'fcompiler')
        # Obtaining the member 'module_dir_switch' of a type (line 240)
        module_dir_switch_53260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 19), fcompiler_53259, 'module_dir_switch')
        # Getting the type of 'None' (line 240)
        None_53261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 50), 'None')
        
        (may_be_53262, more_types_in_union_53263) = may_be_none(module_dir_switch_53260, None_53261)

        if may_be_53262:

            if more_types_in_union_53263:
                # Runtime conditional SSA (line 240)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 241):
            
            # Assigning a Call to a Name (line 241):
            
            # Call to glob(...): (line 241)
            # Processing the call arguments (line 241)
            str_53265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 44), 'str', '*.mod')
            # Processing the call keyword arguments (line 241)
            kwargs_53266 = {}
            # Getting the type of 'glob' (line 241)
            glob_53264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 39), 'glob', False)
            # Calling glob(args, kwargs) (line 241)
            glob_call_result_53267 = invoke(stypy.reporting.localization.Localization(__file__, 241, 39), glob_53264, *[str_53265], **kwargs_53266)
            
            # Assigning a type to the variable 'existing_modules' (line 241)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 20), 'existing_modules', glob_call_result_53267)

            if more_types_in_union_53263:
                # SSA join for if statement (line 240)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'extra_postargs' (line 242)
        extra_postargs_53268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'extra_postargs')
        
        # Call to module_options(...): (line 242)
        # Processing the call arguments (line 242)
        # Getting the type of 'module_dirs' (line 243)
        module_dirs_53271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 20), 'module_dirs', False)
        # Getting the type of 'module_build_dir' (line 243)
        module_build_dir_53272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 33), 'module_build_dir', False)
        # Processing the call keyword arguments (line 242)
        kwargs_53273 = {}
        # Getting the type of 'fcompiler' (line 242)
        fcompiler_53269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 34), 'fcompiler', False)
        # Obtaining the member 'module_options' of a type (line 242)
        module_options_53270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 34), fcompiler_53269, 'module_options')
        # Calling module_options(args, kwargs) (line 242)
        module_options_call_result_53274 = invoke(stypy.reporting.localization.Localization(__file__, 242, 34), module_options_53270, *[module_dirs_53271, module_build_dir_53272], **kwargs_53273)
        
        # Applying the binary operator '+=' (line 242)
        result_iadd_53275 = python_operator(stypy.reporting.localization.Localization(__file__, 242, 16), '+=', extra_postargs_53268, module_options_call_result_53274)
        # Assigning a type to the variable 'extra_postargs' (line 242)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'extra_postargs', result_iadd_53275)
        
        # SSA join for if statement (line 239)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'fmodule_sources' (line 245)
        fmodule_sources_53276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 15), 'fmodule_sources')
        # Testing the type of an if condition (line 245)
        if_condition_53277 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 245, 12), fmodule_sources_53276)
        # Assigning a type to the variable 'if_condition_53277' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'if_condition_53277', if_condition_53277)
        # SSA begins for if statement (line 245)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 246)
        # Processing the call arguments (line 246)
        str_53280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 25), 'str', 'compiling Fortran 90 module sources')
        # Processing the call keyword arguments (line 246)
        kwargs_53281 = {}
        # Getting the type of 'log' (line 246)
        log_53278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'log', False)
        # Obtaining the member 'info' of a type (line 246)
        info_53279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 16), log_53278, 'info')
        # Calling info(args, kwargs) (line 246)
        info_call_result_53282 = invoke(stypy.reporting.localization.Localization(__file__, 246, 16), info_53279, *[str_53280], **kwargs_53281)
        
        
        # Getting the type of 'f_objects' (line 247)
        f_objects_53283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'f_objects')
        
        # Call to compile(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'fmodule_sources' (line 247)
        fmodule_sources_53286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 47), 'fmodule_sources', False)
        # Processing the call keyword arguments (line 247)
        # Getting the type of 'self' (line 248)
        self_53287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 58), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 248)
        build_temp_53288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 58), self_53287, 'build_temp')
        keyword_53289 = build_temp_53288
        # Getting the type of 'macros' (line 249)
        macros_53290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 54), 'macros', False)
        keyword_53291 = macros_53290
        # Getting the type of 'include_dirs' (line 250)
        include_dirs_53292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 60), 'include_dirs', False)
        keyword_53293 = include_dirs_53292
        # Getting the type of 'self' (line 251)
        self_53294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 53), 'self', False)
        # Obtaining the member 'debug' of a type (line 251)
        debug_53295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 53), self_53294, 'debug')
        keyword_53296 = debug_53295
        # Getting the type of 'extra_postargs' (line 252)
        extra_postargs_53297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 62), 'extra_postargs', False)
        keyword_53298 = extra_postargs_53297
        kwargs_53299 = {'debug': keyword_53296, 'macros': keyword_53291, 'extra_postargs': keyword_53298, 'output_dir': keyword_53289, 'include_dirs': keyword_53293}
        # Getting the type of 'fcompiler' (line 247)
        fcompiler_53284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 29), 'fcompiler', False)
        # Obtaining the member 'compile' of a type (line 247)
        compile_53285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 29), fcompiler_53284, 'compile')
        # Calling compile(args, kwargs) (line 247)
        compile_call_result_53300 = invoke(stypy.reporting.localization.Localization(__file__, 247, 29), compile_53285, *[fmodule_sources_53286], **kwargs_53299)
        
        # Applying the binary operator '+=' (line 247)
        result_iadd_53301 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 16), '+=', f_objects_53283, compile_call_result_53300)
        # Assigning a type to the variable 'f_objects' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'f_objects', result_iadd_53301)
        
        # SSA join for if statement (line 245)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'requiref90' (line 254)
        requiref90_53302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 15), 'requiref90')
        
        # Getting the type of 'self' (line 254)
        self_53303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 30), 'self')
        # Obtaining the member '_f_compiler' of a type (line 254)
        _f_compiler_53304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 30), self_53303, '_f_compiler')
        # Obtaining the member 'module_dir_switch' of a type (line 254)
        module_dir_switch_53305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 30), _f_compiler_53304, 'module_dir_switch')
        # Getting the type of 'None' (line 254)
        None_53306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 68), 'None')
        # Applying the binary operator 'is' (line 254)
        result_is__53307 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 30), 'is', module_dir_switch_53305, None_53306)
        
        # Applying the binary operator 'and' (line 254)
        result_and_keyword_53308 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 15), 'and', requiref90_53302, result_is__53307)
        
        # Testing the type of an if condition (line 254)
        if_condition_53309 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 254, 12), result_and_keyword_53308)
        # Assigning a type to the variable 'if_condition_53309' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'if_condition_53309', if_condition_53309)
        # SSA begins for if statement (line 254)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to glob(...): (line 256)
        # Processing the call arguments (line 256)
        str_53311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 30), 'str', '*.mod')
        # Processing the call keyword arguments (line 256)
        kwargs_53312 = {}
        # Getting the type of 'glob' (line 256)
        glob_53310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 25), 'glob', False)
        # Calling glob(args, kwargs) (line 256)
        glob_call_result_53313 = invoke(stypy.reporting.localization.Localization(__file__, 256, 25), glob_53310, *[str_53311], **kwargs_53312)
        
        # Testing the type of a for loop iterable (line 256)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 256, 16), glob_call_result_53313)
        # Getting the type of the for loop variable (line 256)
        for_loop_var_53314 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 256, 16), glob_call_result_53313)
        # Assigning a type to the variable 'f' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 16), 'f', for_loop_var_53314)
        # SSA begins for a for statement (line 256)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'f' (line 257)
        f_53315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 23), 'f')
        # Getting the type of 'existing_modules' (line 257)
        existing_modules_53316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 28), 'existing_modules')
        # Applying the binary operator 'in' (line 257)
        result_contains_53317 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 23), 'in', f_53315, existing_modules_53316)
        
        # Testing the type of an if condition (line 257)
        if_condition_53318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 20), result_contains_53317)
        # Assigning a type to the variable 'if_condition_53318' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 20), 'if_condition_53318', if_condition_53318)
        # SSA begins for if statement (line 257)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 257)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 259):
        
        # Assigning a Call to a Name (line 259):
        
        # Call to join(...): (line 259)
        # Processing the call arguments (line 259)
        # Getting the type of 'module_build_dir' (line 259)
        module_build_dir_53322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 37), 'module_build_dir', False)
        # Getting the type of 'f' (line 259)
        f_53323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 55), 'f', False)
        # Processing the call keyword arguments (line 259)
        kwargs_53324 = {}
        # Getting the type of 'os' (line 259)
        os_53319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 259)
        path_53320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 24), os_53319, 'path')
        # Obtaining the member 'join' of a type (line 259)
        join_53321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 24), path_53320, 'join')
        # Calling join(args, kwargs) (line 259)
        join_call_result_53325 = invoke(stypy.reporting.localization.Localization(__file__, 259, 24), join_53321, *[module_build_dir_53322, f_53323], **kwargs_53324)
        
        # Assigning a type to the variable 't' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 20), 't', join_call_result_53325)
        
        
        
        # Call to abspath(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 'f' (line 260)
        f_53329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 39), 'f', False)
        # Processing the call keyword arguments (line 260)
        kwargs_53330 = {}
        # Getting the type of 'os' (line 260)
        os_53326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 260)
        path_53327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 23), os_53326, 'path')
        # Obtaining the member 'abspath' of a type (line 260)
        abspath_53328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 23), path_53327, 'abspath')
        # Calling abspath(args, kwargs) (line 260)
        abspath_call_result_53331 = invoke(stypy.reporting.localization.Localization(__file__, 260, 23), abspath_53328, *[f_53329], **kwargs_53330)
        
        
        # Call to abspath(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 't' (line 260)
        t_53335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 59), 't', False)
        # Processing the call keyword arguments (line 260)
        kwargs_53336 = {}
        # Getting the type of 'os' (line 260)
        os_53332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 43), 'os', False)
        # Obtaining the member 'path' of a type (line 260)
        path_53333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 43), os_53332, 'path')
        # Obtaining the member 'abspath' of a type (line 260)
        abspath_53334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 43), path_53333, 'abspath')
        # Calling abspath(args, kwargs) (line 260)
        abspath_call_result_53337 = invoke(stypy.reporting.localization.Localization(__file__, 260, 43), abspath_53334, *[t_53335], **kwargs_53336)
        
        # Applying the binary operator '==' (line 260)
        result_eq_53338 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 23), '==', abspath_call_result_53331, abspath_call_result_53337)
        
        # Testing the type of an if condition (line 260)
        if_condition_53339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 20), result_eq_53338)
        # Assigning a type to the variable 'if_condition_53339' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 20), 'if_condition_53339', if_condition_53339)
        # SSA begins for if statement (line 260)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 260)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to isfile(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 't' (line 262)
        t_53343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 38), 't', False)
        # Processing the call keyword arguments (line 262)
        kwargs_53344 = {}
        # Getting the type of 'os' (line 262)
        os_53340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 262)
        path_53341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 23), os_53340, 'path')
        # Obtaining the member 'isfile' of a type (line 262)
        isfile_53342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 23), path_53341, 'isfile')
        # Calling isfile(args, kwargs) (line 262)
        isfile_call_result_53345 = invoke(stypy.reporting.localization.Localization(__file__, 262, 23), isfile_53342, *[t_53343], **kwargs_53344)
        
        # Testing the type of an if condition (line 262)
        if_condition_53346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 20), isfile_call_result_53345)
        # Assigning a type to the variable 'if_condition_53346' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 20), 'if_condition_53346', if_condition_53346)
        # SSA begins for if statement (line 262)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove(...): (line 263)
        # Processing the call arguments (line 263)
        # Getting the type of 't' (line 263)
        t_53349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 34), 't', False)
        # Processing the call keyword arguments (line 263)
        kwargs_53350 = {}
        # Getting the type of 'os' (line 263)
        os_53347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 24), 'os', False)
        # Obtaining the member 'remove' of a type (line 263)
        remove_53348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 24), os_53347, 'remove')
        # Calling remove(args, kwargs) (line 263)
        remove_call_result_53351 = invoke(stypy.reporting.localization.Localization(__file__, 263, 24), remove_53348, *[t_53349], **kwargs_53350)
        
        # SSA join for if statement (line 262)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 264)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to move_file(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'f' (line 265)
        f_53354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 39), 'f', False)
        # Getting the type of 'module_build_dir' (line 265)
        module_build_dir_53355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 42), 'module_build_dir', False)
        # Processing the call keyword arguments (line 265)
        kwargs_53356 = {}
        # Getting the type of 'self' (line 265)
        self_53352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 24), 'self', False)
        # Obtaining the member 'move_file' of a type (line 265)
        move_file_53353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 24), self_53352, 'move_file')
        # Calling move_file(args, kwargs) (line 265)
        move_file_call_result_53357 = invoke(stypy.reporting.localization.Localization(__file__, 265, 24), move_file_53353, *[f_53354, module_build_dir_53355], **kwargs_53356)
        
        # SSA branch for the except part of a try statement (line 264)
        # SSA branch for the except 'DistutilsFileError' branch of a try statement (line 264)
        module_type_store.open_ssa_branch('except')
        
        # Call to warn(...): (line 267)
        # Processing the call arguments (line 267)
        str_53360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 33), 'str', 'failed to move %r to %r')
        
        # Obtaining an instance of the builtin type 'tuple' (line 268)
        tuple_53361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 268)
        # Adding element type (line 268)
        # Getting the type of 'f' (line 268)
        f_53362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 36), 'f', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 36), tuple_53361, f_53362)
        # Adding element type (line 268)
        # Getting the type of 'module_build_dir' (line 268)
        module_build_dir_53363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 39), 'module_build_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 268, 36), tuple_53361, module_build_dir_53363)
        
        # Applying the binary operator '%' (line 267)
        result_mod_53364 = python_operator(stypy.reporting.localization.Localization(__file__, 267, 33), '%', str_53360, tuple_53361)
        
        # Processing the call keyword arguments (line 267)
        kwargs_53365 = {}
        # Getting the type of 'log' (line 267)
        log_53358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 24), 'log', False)
        # Obtaining the member 'warn' of a type (line 267)
        warn_53359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 24), log_53358, 'warn')
        # Calling warn(args, kwargs) (line 267)
        warn_call_result_53366 = invoke(stypy.reporting.localization.Localization(__file__, 267, 24), warn_53359, *[result_mod_53364], **kwargs_53365)
        
        # SSA join for try-except statement (line 264)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 254)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'f_sources' (line 270)
        f_sources_53367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 15), 'f_sources')
        # Testing the type of an if condition (line 270)
        if_condition_53368 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 270, 12), f_sources_53367)
        # Assigning a type to the variable 'if_condition_53368' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'if_condition_53368', if_condition_53368)
        # SSA begins for if statement (line 270)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 271)
        # Processing the call arguments (line 271)
        str_53371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 25), 'str', 'compiling Fortran sources')
        # Processing the call keyword arguments (line 271)
        kwargs_53372 = {}
        # Getting the type of 'log' (line 271)
        log_53369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'log', False)
        # Obtaining the member 'info' of a type (line 271)
        info_53370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 16), log_53369, 'info')
        # Calling info(args, kwargs) (line 271)
        info_call_result_53373 = invoke(stypy.reporting.localization.Localization(__file__, 271, 16), info_53370, *[str_53371], **kwargs_53372)
        
        
        # Getting the type of 'f_objects' (line 272)
        f_objects_53374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'f_objects')
        
        # Call to compile(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'f_sources' (line 272)
        f_sources_53377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 47), 'f_sources', False)
        # Processing the call keyword arguments (line 272)
        # Getting the type of 'self' (line 273)
        self_53378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 58), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 273)
        build_temp_53379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 58), self_53378, 'build_temp')
        keyword_53380 = build_temp_53379
        # Getting the type of 'macros' (line 274)
        macros_53381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 54), 'macros', False)
        keyword_53382 = macros_53381
        # Getting the type of 'include_dirs' (line 275)
        include_dirs_53383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 60), 'include_dirs', False)
        keyword_53384 = include_dirs_53383
        # Getting the type of 'self' (line 276)
        self_53385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 53), 'self', False)
        # Obtaining the member 'debug' of a type (line 276)
        debug_53386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 53), self_53385, 'debug')
        keyword_53387 = debug_53386
        # Getting the type of 'extra_postargs' (line 277)
        extra_postargs_53388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 62), 'extra_postargs', False)
        keyword_53389 = extra_postargs_53388
        kwargs_53390 = {'debug': keyword_53387, 'macros': keyword_53382, 'extra_postargs': keyword_53389, 'output_dir': keyword_53380, 'include_dirs': keyword_53384}
        # Getting the type of 'fcompiler' (line 272)
        fcompiler_53375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 29), 'fcompiler', False)
        # Obtaining the member 'compile' of a type (line 272)
        compile_53376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 29), fcompiler_53375, 'compile')
        # Calling compile(args, kwargs) (line 272)
        compile_call_result_53391 = invoke(stypy.reporting.localization.Localization(__file__, 272, 29), compile_53376, *[f_sources_53377], **kwargs_53390)
        
        # Applying the binary operator '+=' (line 272)
        result_iadd_53392 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 16), '+=', f_objects_53374, compile_call_result_53391)
        # Assigning a type to the variable 'f_objects' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 16), 'f_objects', result_iadd_53392)
        
        # SSA join for if statement (line 270)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 235)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 279):
        
        # Assigning a List to a Name (line 279):
        
        # Obtaining an instance of the builtin type 'list' (line 279)
        list_53393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 279)
        
        # Assigning a type to the variable 'f_objects' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'f_objects', list_53393)
        # SSA join for if statement (line 235)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to extend(...): (line 281)
        # Processing the call arguments (line 281)
        # Getting the type of 'f_objects' (line 281)
        f_objects_53396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 23), 'f_objects', False)
        # Processing the call keyword arguments (line 281)
        kwargs_53397 = {}
        # Getting the type of 'objects' (line 281)
        objects_53394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'objects', False)
        # Obtaining the member 'extend' of a type (line 281)
        extend_53395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 8), objects_53394, 'extend')
        # Calling extend(args, kwargs) (line 281)
        extend_call_result_53398 = invoke(stypy.reporting.localization.Localization(__file__, 281, 8), extend_53395, *[f_objects_53396], **kwargs_53397)
        
        
        # Call to create_static_lib(...): (line 285)
        # Processing the call arguments (line 285)
        # Getting the type of 'objects' (line 285)
        objects_53401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 35), 'objects', False)
        # Getting the type of 'lib_name' (line 285)
        lib_name_53402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 44), 'lib_name', False)
        # Processing the call keyword arguments (line 285)
        # Getting the type of 'self' (line 286)
        self_53403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 46), 'self', False)
        # Obtaining the member 'build_clib' of a type (line 286)
        build_clib_53404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 46), self_53403, 'build_clib')
        keyword_53405 = build_clib_53404
        # Getting the type of 'self' (line 287)
        self_53406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 41), 'self', False)
        # Obtaining the member 'debug' of a type (line 287)
        debug_53407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 41), self_53406, 'debug')
        keyword_53408 = debug_53407
        kwargs_53409 = {'debug': keyword_53408, 'output_dir': keyword_53405}
        # Getting the type of 'compiler' (line 285)
        compiler_53399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'compiler', False)
        # Obtaining the member 'create_static_lib' of a type (line 285)
        create_static_lib_53400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), compiler_53399, 'create_static_lib')
        # Calling create_static_lib(args, kwargs) (line 285)
        create_static_lib_call_result_53410 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), create_static_lib_53400, *[objects_53401, lib_name_53402], **kwargs_53409)
        
        
        # Assigning a Call to a Name (line 290):
        
        # Assigning a Call to a Name (line 290):
        
        # Call to get(...): (line 290)
        # Processing the call arguments (line 290)
        str_53413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 40), 'str', 'libraries')
        
        # Obtaining an instance of the builtin type 'list' (line 290)
        list_53414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 290)
        
        # Processing the call keyword arguments (line 290)
        kwargs_53415 = {}
        # Getting the type of 'build_info' (line 290)
        build_info_53411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 25), 'build_info', False)
        # Obtaining the member 'get' of a type (line 290)
        get_53412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 25), build_info_53411, 'get')
        # Calling get(args, kwargs) (line 290)
        get_call_result_53416 = invoke(stypy.reporting.localization.Localization(__file__, 290, 25), get_53412, *[str_53413, list_53414], **kwargs_53415)
        
        # Assigning a type to the variable 'clib_libraries' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'clib_libraries', get_call_result_53416)
        
        # Getting the type of 'libraries' (line 291)
        libraries_53417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 28), 'libraries')
        # Testing the type of a for loop iterable (line 291)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 291, 8), libraries_53417)
        # Getting the type of the for loop variable (line 291)
        for_loop_var_53418 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 291, 8), libraries_53417)
        # Assigning a type to the variable 'lname' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'lname', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 8), for_loop_var_53418))
        # Assigning a type to the variable 'binfo' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'binfo', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 8), for_loop_var_53418))
        # SSA begins for a for statement (line 291)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'lname' (line 292)
        lname_53419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'lname')
        # Getting the type of 'clib_libraries' (line 292)
        clib_libraries_53420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 24), 'clib_libraries')
        # Applying the binary operator 'in' (line 292)
        result_contains_53421 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 15), 'in', lname_53419, clib_libraries_53420)
        
        # Testing the type of an if condition (line 292)
        if_condition_53422 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 12), result_contains_53421)
        # Assigning a type to the variable 'if_condition_53422' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'if_condition_53422', if_condition_53422)
        # SSA begins for if statement (line 292)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 293)
        # Processing the call arguments (line 293)
        
        # Call to get(...): (line 293)
        # Processing the call arguments (line 293)
        str_53427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 48), 'str', 'libraries')
        
        # Obtaining an instance of the builtin type 'list' (line 293)
        list_53428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 61), 'list')
        # Adding type elements to the builtin type 'list' instance (line 293)
        
        # Processing the call keyword arguments (line 293)
        kwargs_53429 = {}
        # Getting the type of 'binfo' (line 293)
        binfo_53425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 38), 'binfo', False)
        # Obtaining the member 'get' of a type (line 293)
        get_53426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 38), binfo_53425, 'get')
        # Calling get(args, kwargs) (line 293)
        get_call_result_53430 = invoke(stypy.reporting.localization.Localization(__file__, 293, 38), get_53426, *[str_53427, list_53428], **kwargs_53429)
        
        # Processing the call keyword arguments (line 293)
        kwargs_53431 = {}
        # Getting the type of 'clib_libraries' (line 293)
        clib_libraries_53423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 16), 'clib_libraries', False)
        # Obtaining the member 'extend' of a type (line 293)
        extend_53424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 16), clib_libraries_53423, 'extend')
        # Calling extend(args, kwargs) (line 293)
        extend_call_result_53432 = invoke(stypy.reporting.localization.Localization(__file__, 293, 16), extend_53424, *[get_call_result_53430], **kwargs_53431)
        
        # SSA join for if statement (line 292)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'clib_libraries' (line 294)
        clib_libraries_53433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 11), 'clib_libraries')
        # Testing the type of an if condition (line 294)
        if_condition_53434 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 8), clib_libraries_53433)
        # Assigning a type to the variable 'if_condition_53434' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'if_condition_53434', if_condition_53434)
        # SSA begins for if statement (line 294)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 295):
        
        # Assigning a Name to a Subscript (line 295):
        # Getting the type of 'clib_libraries' (line 295)
        clib_libraries_53435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 38), 'clib_libraries')
        # Getting the type of 'build_info' (line 295)
        build_info_53436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'build_info')
        str_53437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 23), 'str', 'libraries')
        # Storing an element on a container (line 295)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 295, 12), build_info_53436, (str_53437, clib_libraries_53435))
        # SSA join for if statement (line 294)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'build_a_library(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_a_library' in the type store
        # Getting the type of 'stypy_return_type' (line 135)
        stypy_return_type_53438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_53438)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_a_library'
        return stypy_return_type_53438


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 25, 0, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'build_clib' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'build_clib', build_clib)

# Assigning a Str to a Name (line 27):
str_53439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 18), 'str', 'build C/C++/F libraries used by Python extensions')
# Getting the type of 'build_clib'
build_clib_53440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_clib')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_clib_53440, 'description', str_53439)

# Assigning a BinOp to a Name (line 29):
# Getting the type of 'old_build_clib' (line 29)
old_build_clib_53441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'old_build_clib')
# Obtaining the member 'user_options' of a type (line 29)
user_options_53442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 19), old_build_clib_53441, 'user_options')

# Obtaining an instance of the builtin type 'list' (line 29)
list_53443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 49), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)
# Adding element type (line 29)

# Obtaining an instance of the builtin type 'tuple' (line 30)
tuple_53444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 30)
# Adding element type (line 30)
str_53445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'str', 'fcompiler=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_53444, str_53445)
# Adding element type (line 30)
# Getting the type of 'None' (line 30)
None_53446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_53444, None_53446)
# Adding element type (line 30)
str_53447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 9), 'str', 'specify the Fortran compiler type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_53444, str_53447)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 49), list_53443, tuple_53444)
# Adding element type (line 29)

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_53448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)
# Adding element type (line 32)
str_53449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 9), 'str', 'inplace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 9), tuple_53448, str_53449)
# Adding element type (line 32)
str_53450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 20), 'str', 'i')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 9), tuple_53448, str_53450)
# Adding element type (line 32)
str_53451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 25), 'str', 'Build in-place')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 9), tuple_53448, str_53451)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 49), list_53443, tuple_53448)
# Adding element type (line 29)

# Obtaining an instance of the builtin type 'tuple' (line 33)
tuple_53452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 33)
# Adding element type (line 33)
str_53453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 9), 'str', 'parallel=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_53452, str_53453)
# Adding element type (line 33)
str_53454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 22), 'str', 'j')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_53452, str_53454)
# Adding element type (line 33)
str_53455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 9), 'str', 'number of parallel jobs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_53452, str_53455)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 49), list_53443, tuple_53452)

# Applying the binary operator '+' (line 29)
result_add_53456 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 19), '+', user_options_53442, list_53443)

# Getting the type of 'build_clib'
build_clib_53457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_clib')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_clib_53457, 'user_options', result_add_53456)

# Assigning a BinOp to a Name (line 37):
# Getting the type of 'old_build_clib' (line 37)
old_build_clib_53458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 22), 'old_build_clib')
# Obtaining the member 'boolean_options' of a type (line 37)
boolean_options_53459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 22), old_build_clib_53458, 'boolean_options')

# Obtaining an instance of the builtin type 'list' (line 37)
list_53460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 55), 'list')
# Adding type elements to the builtin type 'list' instance (line 37)
# Adding element type (line 37)
str_53461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 56), 'str', 'inplace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 55), list_53460, str_53461)

# Applying the binary operator '+' (line 37)
result_add_53462 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 22), '+', boolean_options_53459, list_53460)

# Getting the type of 'build_clib'
build_clib_53463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_clib')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_clib_53463, 'boolean_options', result_add_53462)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
