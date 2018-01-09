
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.build_py
2: 
3: Implements the Distutils 'build_py' command.'''
4: 
5: __revision__ = "$Id$"
6: 
7: import os
8: import sys
9: from glob import glob
10: 
11: from distutils.core import Command
12: from distutils.errors import DistutilsOptionError, DistutilsFileError
13: from distutils.util import convert_path
14: from distutils import log
15: 
16: class build_py(Command):
17: 
18:     description = "\"build\" pure Python modules (copy to build directory)"
19: 
20:     user_options = [
21:         ('build-lib=', 'd', "directory to \"build\" (copy) to"),
22:         ('compile', 'c', "compile .py to .pyc"),
23:         ('no-compile', None, "don't compile .py files [default]"),
24:         ('optimize=', 'O',
25:          "also compile with optimization: -O1 for \"python -O\", "
26:          "-O2 for \"python -OO\", and -O0 to disable [default: -O0]"),
27:         ('force', 'f', "forcibly build everything (ignore file timestamps)"),
28:         ]
29: 
30:     boolean_options = ['compile', 'force']
31:     negative_opt = {'no-compile' : 'compile'}
32: 
33:     def initialize_options(self):
34:         self.build_lib = None
35:         self.py_modules = None
36:         self.package = None
37:         self.package_data = None
38:         self.package_dir = None
39:         self.compile = 0
40:         self.optimize = 0
41:         self.force = None
42: 
43:     def finalize_options(self):
44:         self.set_undefined_options('build',
45:                                    ('build_lib', 'build_lib'),
46:                                    ('force', 'force'))
47: 
48:         # Get the distribution options that are aliases for build_py
49:         # options -- list of packages and list of modules.
50:         self.packages = self.distribution.packages
51:         self.py_modules = self.distribution.py_modules
52:         self.package_data = self.distribution.package_data
53:         self.package_dir = {}
54:         if self.distribution.package_dir:
55:             for name, path in self.distribution.package_dir.items():
56:                 self.package_dir[name] = convert_path(path)
57:         self.data_files = self.get_data_files()
58: 
59:         # Ick, copied straight from install_lib.py (fancy_getopt needs a
60:         # type system!  Hell, *everything* needs a type system!!!)
61:         if not isinstance(self.optimize, int):
62:             try:
63:                 self.optimize = int(self.optimize)
64:                 assert 0 <= self.optimize <= 2
65:             except (ValueError, AssertionError):
66:                 raise DistutilsOptionError("optimize must be 0, 1, or 2")
67: 
68:     def run(self):
69:         # XXX copy_file by default preserves atime and mtime.  IMHO this is
70:         # the right thing to do, but perhaps it should be an option -- in
71:         # particular, a site administrator might want installed files to
72:         # reflect the time of installation rather than the last
73:         # modification time before the installed release.
74: 
75:         # XXX copy_file by default preserves mode, which appears to be the
76:         # wrong thing to do: if a file is read-only in the working
77:         # directory, we want it to be installed read/write so that the next
78:         # installation of the same module distribution can overwrite it
79:         # without problems.  (This might be a Unix-specific issue.)  Thus
80:         # we turn off 'preserve_mode' when copying to the build directory,
81:         # since the build directory is supposed to be exactly what the
82:         # installation will look like (ie. we preserve mode when
83:         # installing).
84: 
85:         # Two options control which modules will be installed: 'packages'
86:         # and 'py_modules'.  The former lets us work with whole packages, not
87:         # specifying individual modules at all; the latter is for
88:         # specifying modules one-at-a-time.
89: 
90:         if self.py_modules:
91:             self.build_modules()
92:         if self.packages:
93:             self.build_packages()
94:             self.build_package_data()
95: 
96:         self.byte_compile(self.get_outputs(include_bytecode=0))
97: 
98:     def get_data_files(self):
99:         '''Generate list of '(package,src_dir,build_dir,filenames)' tuples'''
100:         data = []
101:         if not self.packages:
102:             return data
103:         for package in self.packages:
104:             # Locate package source directory
105:             src_dir = self.get_package_dir(package)
106: 
107:             # Compute package build directory
108:             build_dir = os.path.join(*([self.build_lib] + package.split('.')))
109: 
110:             # Length of path to strip from found files
111:             plen = 0
112:             if src_dir:
113:                 plen = len(src_dir)+1
114: 
115:             # Strip directory from globbed filenames
116:             filenames = [
117:                 file[plen:] for file in self.find_data_files(package, src_dir)
118:                 ]
119:             data.append((package, src_dir, build_dir, filenames))
120:         return data
121: 
122:     def find_data_files(self, package, src_dir):
123:         '''Return filenames for package's data files in 'src_dir''''
124:         globs = (self.package_data.get('', [])
125:                  + self.package_data.get(package, []))
126:         files = []
127:         for pattern in globs:
128:             # Each pattern has to be converted to a platform-specific path
129:             filelist = glob(os.path.join(src_dir, convert_path(pattern)))
130:             # Files that match more than one pattern are only added once
131:             files.extend([fn for fn in filelist if fn not in files
132:                 and os.path.isfile(fn)])
133:         return files
134: 
135:     def build_package_data(self):
136:         '''Copy data files into build directory'''
137:         for package, src_dir, build_dir, filenames in self.data_files:
138:             for filename in filenames:
139:                 target = os.path.join(build_dir, filename)
140:                 self.mkpath(os.path.dirname(target))
141:                 self.copy_file(os.path.join(src_dir, filename), target,
142:                                preserve_mode=False)
143: 
144:     def get_package_dir(self, package):
145:         '''Return the directory, relative to the top of the source
146:            distribution, where package 'package' should be found
147:            (at least according to the 'package_dir' option, if any).'''
148: 
149:         path = package.split('.')
150: 
151:         if not self.package_dir:
152:             if path:
153:                 return os.path.join(*path)
154:             else:
155:                 return ''
156:         else:
157:             tail = []
158:             while path:
159:                 try:
160:                     pdir = self.package_dir['.'.join(path)]
161:                 except KeyError:
162:                     tail.insert(0, path[-1])
163:                     del path[-1]
164:                 else:
165:                     tail.insert(0, pdir)
166:                     return os.path.join(*tail)
167:             else:
168:                 # Oops, got all the way through 'path' without finding a
169:                 # match in package_dir.  If package_dir defines a directory
170:                 # for the root (nameless) package, then fallback on it;
171:                 # otherwise, we might as well have not consulted
172:                 # package_dir at all, as we just use the directory implied
173:                 # by 'tail' (which should be the same as the original value
174:                 # of 'path' at this point).
175:                 pdir = self.package_dir.get('')
176:                 if pdir is not None:
177:                     tail.insert(0, pdir)
178: 
179:                 if tail:
180:                     return os.path.join(*tail)
181:                 else:
182:                     return ''
183: 
184:     def check_package(self, package, package_dir):
185:         # Empty dir name means current directory, which we can probably
186:         # assume exists.  Also, os.path.exists and isdir don't know about
187:         # my "empty string means current dir" convention, so we have to
188:         # circumvent them.
189:         if package_dir != "":
190:             if not os.path.exists(package_dir):
191:                 raise DistutilsFileError(
192:                       "package directory '%s' does not exist" % package_dir)
193:             if not os.path.isdir(package_dir):
194:                 raise DistutilsFileError(
195:                        "supposed package directory '%s' exists, "
196:                        "but is not a directory" % package_dir)
197: 
198:         # Require __init__.py for all but the "root package"
199:         if package:
200:             init_py = os.path.join(package_dir, "__init__.py")
201:             if os.path.isfile(init_py):
202:                 return init_py
203:             else:
204:                 log.warn(("package init file '%s' not found " +
205:                           "(or not a regular file)"), init_py)
206: 
207:         # Either not in a package at all (__init__.py not expected), or
208:         # __init__.py doesn't exist -- so don't return the filename.
209:         return None
210: 
211:     def check_module(self, module, module_file):
212:         if not os.path.isfile(module_file):
213:             log.warn("file %s (for module %s) not found", module_file, module)
214:             return False
215:         else:
216:             return True
217: 
218:     def find_package_modules(self, package, package_dir):
219:         self.check_package(package, package_dir)
220:         module_files = glob(os.path.join(package_dir, "*.py"))
221:         modules = []
222:         setup_script = os.path.abspath(self.distribution.script_name)
223: 
224:         for f in module_files:
225:             abs_f = os.path.abspath(f)
226:             if abs_f != setup_script:
227:                 module = os.path.splitext(os.path.basename(f))[0]
228:                 modules.append((package, module, f))
229:             else:
230:                 self.debug_print("excluding %s" % setup_script)
231:         return modules
232: 
233:     def find_modules(self):
234:         '''Finds individually-specified Python modules, ie. those listed by
235:         module name in 'self.py_modules'.  Returns a list of tuples (package,
236:         module_base, filename): 'package' is a tuple of the path through
237:         package-space to the module; 'module_base' is the bare (no
238:         packages, no dots) module name, and 'filename' is the path to the
239:         ".py" file (relative to the distribution root) that implements the
240:         module.
241:         '''
242:         # Map package names to tuples of useful info about the package:
243:         #    (package_dir, checked)
244:         # package_dir - the directory where we'll find source files for
245:         #   this package
246:         # checked - true if we have checked that the package directory
247:         #   is valid (exists, contains __init__.py, ... ?)
248:         packages = {}
249: 
250:         # List of (package, module, filename) tuples to return
251:         modules = []
252: 
253:         # We treat modules-in-packages almost the same as toplevel modules,
254:         # just the "package" for a toplevel is empty (either an empty
255:         # string or empty list, depending on context).  Differences:
256:         #   - don't check for __init__.py in directory for empty package
257:         for module in self.py_modules:
258:             path = module.split('.')
259:             package = '.'.join(path[0:-1])
260:             module_base = path[-1]
261: 
262:             try:
263:                 (package_dir, checked) = packages[package]
264:             except KeyError:
265:                 package_dir = self.get_package_dir(package)
266:                 checked = 0
267: 
268:             if not checked:
269:                 init_py = self.check_package(package, package_dir)
270:                 packages[package] = (package_dir, 1)
271:                 if init_py:
272:                     modules.append((package, "__init__", init_py))
273: 
274:             # XXX perhaps we should also check for just .pyc files
275:             # (so greedy closed-source bastards can distribute Python
276:             # modules too)
277:             module_file = os.path.join(package_dir, module_base + ".py")
278:             if not self.check_module(module, module_file):
279:                 continue
280: 
281:             modules.append((package, module_base, module_file))
282: 
283:         return modules
284: 
285:     def find_all_modules(self):
286:         '''Compute the list of all modules that will be built, whether
287:         they are specified one-module-at-a-time ('self.py_modules') or
288:         by whole packages ('self.packages').  Return a list of tuples
289:         (package, module, module_file), just like 'find_modules()' and
290:         'find_package_modules()' do.'''
291:         modules = []
292:         if self.py_modules:
293:             modules.extend(self.find_modules())
294:         if self.packages:
295:             for package in self.packages:
296:                 package_dir = self.get_package_dir(package)
297:                 m = self.find_package_modules(package, package_dir)
298:                 modules.extend(m)
299:         return modules
300: 
301:     def get_source_files(self):
302:         return [module[-1] for module in self.find_all_modules()]
303: 
304:     def get_module_outfile(self, build_dir, package, module):
305:         outfile_path = [build_dir] + list(package) + [module + ".py"]
306:         return os.path.join(*outfile_path)
307: 
308:     def get_outputs(self, include_bytecode=1):
309:         modules = self.find_all_modules()
310:         outputs = []
311:         for (package, module, module_file) in modules:
312:             package = package.split('.')
313:             filename = self.get_module_outfile(self.build_lib, package, module)
314:             outputs.append(filename)
315:             if include_bytecode:
316:                 if self.compile:
317:                     outputs.append(filename + "c")
318:                 if self.optimize > 0:
319:                     outputs.append(filename + "o")
320: 
321:         outputs += [
322:             os.path.join(build_dir, filename)
323:             for package, src_dir, build_dir, filenames in self.data_files
324:             for filename in filenames
325:             ]
326: 
327:         return outputs
328: 
329:     def build_module(self, module, module_file, package):
330:         if isinstance(package, str):
331:             package = package.split('.')
332:         elif not isinstance(package, (list, tuple)):
333:             raise TypeError(
334:                   "'package' must be a string (dot-separated), list, or tuple")
335: 
336:         # Now put the module source file into the "build" area -- this is
337:         # easy, we just copy it somewhere under self.build_lib (the build
338:         # directory for Python source).
339:         outfile = self.get_module_outfile(self.build_lib, package, module)
340:         dir = os.path.dirname(outfile)
341:         self.mkpath(dir)
342:         return self.copy_file(module_file, outfile, preserve_mode=0)
343: 
344:     def build_modules(self):
345:         modules = self.find_modules()
346:         for (package, module, module_file) in modules:
347: 
348:             # Now "build" the module -- ie. copy the source file to
349:             # self.build_lib (the build directory for Python source).
350:             # (Actually, it gets copied to the directory for this package
351:             # under self.build_lib.)
352:             self.build_module(module, module_file, package)
353: 
354:     def build_packages(self):
355:         for package in self.packages:
356: 
357:             # Get list of (package, module, module_file) tuples based on
358:             # scanning the package directory.  'package' is only included
359:             # in the tuple so that 'find_modules()' and
360:             # 'find_package_tuples()' have a consistent interface; it's
361:             # ignored here (apart from a sanity check).  Also, 'module' is
362:             # the *unqualified* module name (ie. no dots, no package -- we
363:             # already know its package!), and 'module_file' is the path to
364:             # the .py file, relative to the current directory
365:             # (ie. including 'package_dir').
366:             package_dir = self.get_package_dir(package)
367:             modules = self.find_package_modules(package, package_dir)
368: 
369:             # Now loop over the modules we found, "building" each one (just
370:             # copy it to self.build_lib).
371:             for (package_, module, module_file) in modules:
372:                 assert package == package_
373:                 self.build_module(module, module_file, package)
374: 
375:     def byte_compile(self, files):
376:         if sys.dont_write_bytecode:
377:             self.warn('byte-compiling is disabled, skipping.')
378:             return
379: 
380:         from distutils.util import byte_compile
381:         prefix = self.build_lib
382:         if prefix[-1] != os.sep:
383:             prefix = prefix + os.sep
384: 
385:         # XXX this code is essentially the same as the 'byte_compile()
386:         # method of the "install_lib" command, except for the determination
387:         # of the 'prefix' string.  Hmmm.
388: 
389:         if self.compile:
390:             byte_compile(files, optimize=0,
391:                          force=self.force, prefix=prefix, dry_run=self.dry_run)
392:         if self.optimize > 0:
393:             byte_compile(files, optimize=self.optimize,
394:                          force=self.force, prefix=prefix, dry_run=self.dry_run)
395: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_19701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', "distutils.command.build_py\n\nImplements the Distutils 'build_py' command.")

# Assigning a Str to a Name (line 5):

# Assigning a Str to a Name (line 5):
str_19702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__revision__', str_19702)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import os' statement (line 7)
import os

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import sys' statement (line 8)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from glob import glob' statement (line 9)
try:
    from glob import glob

except:
    glob = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'glob', None, module_type_store, ['glob'], [glob])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.core import Command' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_19703 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core')

if (type(import_19703) is not StypyTypeError):

    if (import_19703 != 'pyd_module'):
        __import__(import_19703)
        sys_modules_19704 = sys.modules[import_19703]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core', sys_modules_19704.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_19704, sys_modules_19704.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core', import_19703)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.errors import DistutilsOptionError, DistutilsFileError' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_19705 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors')

if (type(import_19705) is not StypyTypeError):

    if (import_19705 != 'pyd_module'):
        __import__(import_19705)
        sys_modules_19706 = sys.modules[import_19705]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', sys_modules_19706.module_type_store, module_type_store, ['DistutilsOptionError', 'DistutilsFileError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_19706, sys_modules_19706.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsOptionError, DistutilsFileError

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', None, module_type_store, ['DistutilsOptionError', 'DistutilsFileError'], [DistutilsOptionError, DistutilsFileError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', import_19705)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.util import convert_path' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_19707 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.util')

if (type(import_19707) is not StypyTypeError):

    if (import_19707 != 'pyd_module'):
        __import__(import_19707)
        sys_modules_19708 = sys.modules[import_19707]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.util', sys_modules_19708.module_type_store, module_type_store, ['convert_path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_19708, sys_modules_19708.module_type_store, module_type_store)
    else:
        from distutils.util import convert_path

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.util', None, module_type_store, ['convert_path'], [convert_path])

else:
    # Assigning a type to the variable 'distutils.util' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.util', import_19707)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils import log' statement (line 14)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils', None, module_type_store, ['log'], [log])

# Declaration of the 'build_py' class
# Getting the type of 'Command' (line 16)
Command_19709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'Command')

class build_py(Command_19709, ):
    
    # Assigning a Str to a Name (line 18):
    
    # Assigning a List to a Name (line 20):
    
    # Assigning a List to a Name (line 30):
    
    # Assigning a Dict to a Name (line 31):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        build_py.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.initialize_options.__dict__.__setitem__('stypy_function_name', 'build_py.initialize_options')
        build_py.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build_py.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 34):
        
        # Assigning a Name to a Attribute (line 34):
        # Getting the type of 'None' (line 34)
        None_19710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'None')
        # Getting the type of 'self' (line 34)
        self_19711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'build_lib' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_19711, 'build_lib', None_19710)
        
        # Assigning a Name to a Attribute (line 35):
        
        # Assigning a Name to a Attribute (line 35):
        # Getting the type of 'None' (line 35)
        None_19712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 26), 'None')
        # Getting the type of 'self' (line 35)
        self_19713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self')
        # Setting the type of the member 'py_modules' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_19713, 'py_modules', None_19712)
        
        # Assigning a Name to a Attribute (line 36):
        
        # Assigning a Name to a Attribute (line 36):
        # Getting the type of 'None' (line 36)
        None_19714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'None')
        # Getting the type of 'self' (line 36)
        self_19715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self')
        # Setting the type of the member 'package' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_19715, 'package', None_19714)
        
        # Assigning a Name to a Attribute (line 37):
        
        # Assigning a Name to a Attribute (line 37):
        # Getting the type of 'None' (line 37)
        None_19716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 28), 'None')
        # Getting the type of 'self' (line 37)
        self_19717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self')
        # Setting the type of the member 'package_data' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_19717, 'package_data', None_19716)
        
        # Assigning a Name to a Attribute (line 38):
        
        # Assigning a Name to a Attribute (line 38):
        # Getting the type of 'None' (line 38)
        None_19718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'None')
        # Getting the type of 'self' (line 38)
        self_19719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member 'package_dir' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_19719, 'package_dir', None_19718)
        
        # Assigning a Num to a Attribute (line 39):
        
        # Assigning a Num to a Attribute (line 39):
        int_19720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 23), 'int')
        # Getting the type of 'self' (line 39)
        self_19721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self')
        # Setting the type of the member 'compile' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_19721, 'compile', int_19720)
        
        # Assigning a Num to a Attribute (line 40):
        
        # Assigning a Num to a Attribute (line 40):
        int_19722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 24), 'int')
        # Getting the type of 'self' (line 40)
        self_19723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self')
        # Setting the type of the member 'optimize' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_19723, 'optimize', int_19722)
        
        # Assigning a Name to a Attribute (line 41):
        
        # Assigning a Name to a Attribute (line 41):
        # Getting the type of 'None' (line 41)
        None_19724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 21), 'None')
        # Getting the type of 'self' (line 41)
        self_19725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member 'force' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_19725, 'force', None_19724)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_19726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19726)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_19726


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        build_py.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.finalize_options.__dict__.__setitem__('stypy_function_name', 'build_py.finalize_options')
        build_py.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build_py.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_undefined_options(...): (line 44)
        # Processing the call arguments (line 44)
        str_19729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 35), 'str', 'build')
        
        # Obtaining an instance of the builtin type 'tuple' (line 45)
        tuple_19730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 45)
        # Adding element type (line 45)
        str_19731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 36), 'str', 'build_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 36), tuple_19730, str_19731)
        # Adding element type (line 45)
        str_19732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 49), 'str', 'build_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 36), tuple_19730, str_19732)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 46)
        tuple_19733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 46)
        # Adding element type (line 46)
        str_19734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 36), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 36), tuple_19733, str_19734)
        # Adding element type (line 46)
        str_19735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 45), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 36), tuple_19733, str_19735)
        
        # Processing the call keyword arguments (line 44)
        kwargs_19736 = {}
        # Getting the type of 'self' (line 44)
        self_19727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 44)
        set_undefined_options_19728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), self_19727, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 44)
        set_undefined_options_call_result_19737 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), set_undefined_options_19728, *[str_19729, tuple_19730, tuple_19733], **kwargs_19736)
        
        
        # Assigning a Attribute to a Attribute (line 50):
        
        # Assigning a Attribute to a Attribute (line 50):
        # Getting the type of 'self' (line 50)
        self_19738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'self')
        # Obtaining the member 'distribution' of a type (line 50)
        distribution_19739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 24), self_19738, 'distribution')
        # Obtaining the member 'packages' of a type (line 50)
        packages_19740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 24), distribution_19739, 'packages')
        # Getting the type of 'self' (line 50)
        self_19741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'self')
        # Setting the type of the member 'packages' of a type (line 50)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), self_19741, 'packages', packages_19740)
        
        # Assigning a Attribute to a Attribute (line 51):
        
        # Assigning a Attribute to a Attribute (line 51):
        # Getting the type of 'self' (line 51)
        self_19742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), 'self')
        # Obtaining the member 'distribution' of a type (line 51)
        distribution_19743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 26), self_19742, 'distribution')
        # Obtaining the member 'py_modules' of a type (line 51)
        py_modules_19744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 26), distribution_19743, 'py_modules')
        # Getting the type of 'self' (line 51)
        self_19745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self')
        # Setting the type of the member 'py_modules' of a type (line 51)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_19745, 'py_modules', py_modules_19744)
        
        # Assigning a Attribute to a Attribute (line 52):
        
        # Assigning a Attribute to a Attribute (line 52):
        # Getting the type of 'self' (line 52)
        self_19746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 28), 'self')
        # Obtaining the member 'distribution' of a type (line 52)
        distribution_19747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 28), self_19746, 'distribution')
        # Obtaining the member 'package_data' of a type (line 52)
        package_data_19748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 28), distribution_19747, 'package_data')
        # Getting the type of 'self' (line 52)
        self_19749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self')
        # Setting the type of the member 'package_data' of a type (line 52)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_19749, 'package_data', package_data_19748)
        
        # Assigning a Dict to a Attribute (line 53):
        
        # Assigning a Dict to a Attribute (line 53):
        
        # Obtaining an instance of the builtin type 'dict' (line 53)
        dict_19750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 27), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 53)
        
        # Getting the type of 'self' (line 53)
        self_19751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self')
        # Setting the type of the member 'package_dir' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_19751, 'package_dir', dict_19750)
        
        # Getting the type of 'self' (line 54)
        self_19752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'self')
        # Obtaining the member 'distribution' of a type (line 54)
        distribution_19753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 11), self_19752, 'distribution')
        # Obtaining the member 'package_dir' of a type (line 54)
        package_dir_19754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 11), distribution_19753, 'package_dir')
        # Testing the type of an if condition (line 54)
        if_condition_19755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 8), package_dir_19754)
        # Assigning a type to the variable 'if_condition_19755' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'if_condition_19755', if_condition_19755)
        # SSA begins for if statement (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to items(...): (line 55)
        # Processing the call keyword arguments (line 55)
        kwargs_19760 = {}
        # Getting the type of 'self' (line 55)
        self_19756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 30), 'self', False)
        # Obtaining the member 'distribution' of a type (line 55)
        distribution_19757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 30), self_19756, 'distribution')
        # Obtaining the member 'package_dir' of a type (line 55)
        package_dir_19758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 30), distribution_19757, 'package_dir')
        # Obtaining the member 'items' of a type (line 55)
        items_19759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 30), package_dir_19758, 'items')
        # Calling items(args, kwargs) (line 55)
        items_call_result_19761 = invoke(stypy.reporting.localization.Localization(__file__, 55, 30), items_19759, *[], **kwargs_19760)
        
        # Testing the type of a for loop iterable (line 55)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 55, 12), items_call_result_19761)
        # Getting the type of the for loop variable (line 55)
        for_loop_var_19762 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 55, 12), items_call_result_19761)
        # Assigning a type to the variable 'name' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 12), for_loop_var_19762))
        # Assigning a type to the variable 'path' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'path', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 12), for_loop_var_19762))
        # SSA begins for a for statement (line 55)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Subscript (line 56):
        
        # Assigning a Call to a Subscript (line 56):
        
        # Call to convert_path(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'path' (line 56)
        path_19764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 54), 'path', False)
        # Processing the call keyword arguments (line 56)
        kwargs_19765 = {}
        # Getting the type of 'convert_path' (line 56)
        convert_path_19763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 41), 'convert_path', False)
        # Calling convert_path(args, kwargs) (line 56)
        convert_path_call_result_19766 = invoke(stypy.reporting.localization.Localization(__file__, 56, 41), convert_path_19763, *[path_19764], **kwargs_19765)
        
        # Getting the type of 'self' (line 56)
        self_19767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'self')
        # Obtaining the member 'package_dir' of a type (line 56)
        package_dir_19768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), self_19767, 'package_dir')
        # Getting the type of 'name' (line 56)
        name_19769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'name')
        # Storing an element on a container (line 56)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 16), package_dir_19768, (name_19769, convert_path_call_result_19766))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 54)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 57):
        
        # Assigning a Call to a Attribute (line 57):
        
        # Call to get_data_files(...): (line 57)
        # Processing the call keyword arguments (line 57)
        kwargs_19772 = {}
        # Getting the type of 'self' (line 57)
        self_19770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), 'self', False)
        # Obtaining the member 'get_data_files' of a type (line 57)
        get_data_files_19771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 26), self_19770, 'get_data_files')
        # Calling get_data_files(args, kwargs) (line 57)
        get_data_files_call_result_19773 = invoke(stypy.reporting.localization.Localization(__file__, 57, 26), get_data_files_19771, *[], **kwargs_19772)
        
        # Getting the type of 'self' (line 57)
        self_19774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member 'data_files' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_19774, 'data_files', get_data_files_call_result_19773)
        
        # Type idiom detected: calculating its left and rigth part (line 61)
        # Getting the type of 'int' (line 61)
        int_19775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 41), 'int')
        # Getting the type of 'self' (line 61)
        self_19776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'self')
        # Obtaining the member 'optimize' of a type (line 61)
        optimize_19777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 26), self_19776, 'optimize')
        
        (may_be_19778, more_types_in_union_19779) = may_not_be_subtype(int_19775, optimize_19777)

        if may_be_19778:

            if more_types_in_union_19779:
                # Runtime conditional SSA (line 61)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 61)
            self_19780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
            # Obtaining the member 'optimize' of a type (line 61)
            optimize_19781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_19780, 'optimize')
            # Setting the type of the member 'optimize' of a type (line 61)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_19780, 'optimize', remove_subtype_from_union(optimize_19777, int))
            
            
            # SSA begins for try-except statement (line 62)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Attribute (line 63):
            
            # Assigning a Call to a Attribute (line 63):
            
            # Call to int(...): (line 63)
            # Processing the call arguments (line 63)
            # Getting the type of 'self' (line 63)
            self_19783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 36), 'self', False)
            # Obtaining the member 'optimize' of a type (line 63)
            optimize_19784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 36), self_19783, 'optimize')
            # Processing the call keyword arguments (line 63)
            kwargs_19785 = {}
            # Getting the type of 'int' (line 63)
            int_19782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 32), 'int', False)
            # Calling int(args, kwargs) (line 63)
            int_call_result_19786 = invoke(stypy.reporting.localization.Localization(__file__, 63, 32), int_19782, *[optimize_19784], **kwargs_19785)
            
            # Getting the type of 'self' (line 63)
            self_19787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'self')
            # Setting the type of the member 'optimize' of a type (line 63)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 16), self_19787, 'optimize', int_call_result_19786)
            # Evaluating assert statement condition
            
            int_19788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 23), 'int')
            # Getting the type of 'self' (line 64)
            self_19789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 28), 'self')
            # Obtaining the member 'optimize' of a type (line 64)
            optimize_19790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 28), self_19789, 'optimize')
            # Applying the binary operator '<=' (line 64)
            result_le_19791 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 23), '<=', int_19788, optimize_19790)
            int_19792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 45), 'int')
            # Applying the binary operator '<=' (line 64)
            result_le_19793 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 23), '<=', optimize_19790, int_19792)
            # Applying the binary operator '&' (line 64)
            result_and__19794 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 23), '&', result_le_19791, result_le_19793)
            
            # SSA branch for the except part of a try statement (line 62)
            # SSA branch for the except 'Tuple' branch of a try statement (line 62)
            module_type_store.open_ssa_branch('except')
            
            # Call to DistutilsOptionError(...): (line 66)
            # Processing the call arguments (line 66)
            str_19796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 43), 'str', 'optimize must be 0, 1, or 2')
            # Processing the call keyword arguments (line 66)
            kwargs_19797 = {}
            # Getting the type of 'DistutilsOptionError' (line 66)
            DistutilsOptionError_19795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'DistutilsOptionError', False)
            # Calling DistutilsOptionError(args, kwargs) (line 66)
            DistutilsOptionError_call_result_19798 = invoke(stypy.reporting.localization.Localization(__file__, 66, 22), DistutilsOptionError_19795, *[str_19796], **kwargs_19797)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 66, 16), DistutilsOptionError_call_result_19798, 'raise parameter', BaseException)
            # SSA join for try-except statement (line 62)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_19779:
                # SSA join for if statement (line 61)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_19799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19799)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_19799


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.run.__dict__.__setitem__('stypy_localization', localization)
        build_py.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.run.__dict__.__setitem__('stypy_function_name', 'build_py.run')
        build_py.run.__dict__.__setitem__('stypy_param_names_list', [])
        build_py.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.run', [], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'self' (line 90)
        self_19800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 11), 'self')
        # Obtaining the member 'py_modules' of a type (line 90)
        py_modules_19801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 11), self_19800, 'py_modules')
        # Testing the type of an if condition (line 90)
        if_condition_19802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 8), py_modules_19801)
        # Assigning a type to the variable 'if_condition_19802' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'if_condition_19802', if_condition_19802)
        # SSA begins for if statement (line 90)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to build_modules(...): (line 91)
        # Processing the call keyword arguments (line 91)
        kwargs_19805 = {}
        # Getting the type of 'self' (line 91)
        self_19803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'self', False)
        # Obtaining the member 'build_modules' of a type (line 91)
        build_modules_19804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), self_19803, 'build_modules')
        # Calling build_modules(args, kwargs) (line 91)
        build_modules_call_result_19806 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), build_modules_19804, *[], **kwargs_19805)
        
        # SSA join for if statement (line 90)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 92)
        self_19807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'self')
        # Obtaining the member 'packages' of a type (line 92)
        packages_19808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 11), self_19807, 'packages')
        # Testing the type of an if condition (line 92)
        if_condition_19809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 8), packages_19808)
        # Assigning a type to the variable 'if_condition_19809' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'if_condition_19809', if_condition_19809)
        # SSA begins for if statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to build_packages(...): (line 93)
        # Processing the call keyword arguments (line 93)
        kwargs_19812 = {}
        # Getting the type of 'self' (line 93)
        self_19810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'self', False)
        # Obtaining the member 'build_packages' of a type (line 93)
        build_packages_19811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), self_19810, 'build_packages')
        # Calling build_packages(args, kwargs) (line 93)
        build_packages_call_result_19813 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), build_packages_19811, *[], **kwargs_19812)
        
        
        # Call to build_package_data(...): (line 94)
        # Processing the call keyword arguments (line 94)
        kwargs_19816 = {}
        # Getting the type of 'self' (line 94)
        self_19814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'self', False)
        # Obtaining the member 'build_package_data' of a type (line 94)
        build_package_data_19815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), self_19814, 'build_package_data')
        # Calling build_package_data(args, kwargs) (line 94)
        build_package_data_call_result_19817 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), build_package_data_19815, *[], **kwargs_19816)
        
        # SSA join for if statement (line 92)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to byte_compile(...): (line 96)
        # Processing the call arguments (line 96)
        
        # Call to get_outputs(...): (line 96)
        # Processing the call keyword arguments (line 96)
        int_19822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 60), 'int')
        keyword_19823 = int_19822
        kwargs_19824 = {'include_bytecode': keyword_19823}
        # Getting the type of 'self' (line 96)
        self_19820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'self', False)
        # Obtaining the member 'get_outputs' of a type (line 96)
        get_outputs_19821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 26), self_19820, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 96)
        get_outputs_call_result_19825 = invoke(stypy.reporting.localization.Localization(__file__, 96, 26), get_outputs_19821, *[], **kwargs_19824)
        
        # Processing the call keyword arguments (line 96)
        kwargs_19826 = {}
        # Getting the type of 'self' (line 96)
        self_19818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self', False)
        # Obtaining the member 'byte_compile' of a type (line 96)
        byte_compile_19819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_19818, 'byte_compile')
        # Calling byte_compile(args, kwargs) (line 96)
        byte_compile_call_result_19827 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), byte_compile_19819, *[get_outputs_call_result_19825], **kwargs_19826)
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_19828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19828)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_19828


    @norecursion
    def get_data_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_data_files'
        module_type_store = module_type_store.open_function_context('get_data_files', 98, 4, False)
        # Assigning a type to the variable 'self' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.get_data_files.__dict__.__setitem__('stypy_localization', localization)
        build_py.get_data_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.get_data_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.get_data_files.__dict__.__setitem__('stypy_function_name', 'build_py.get_data_files')
        build_py.get_data_files.__dict__.__setitem__('stypy_param_names_list', [])
        build_py.get_data_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.get_data_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.get_data_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.get_data_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.get_data_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.get_data_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.get_data_files', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_data_files', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_data_files(...)' code ##################

        str_19829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 8), 'str', "Generate list of '(package,src_dir,build_dir,filenames)' tuples")
        
        # Assigning a List to a Name (line 100):
        
        # Assigning a List to a Name (line 100):
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_19830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        
        # Assigning a type to the variable 'data' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'data', list_19830)
        
        
        # Getting the type of 'self' (line 101)
        self_19831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'self')
        # Obtaining the member 'packages' of a type (line 101)
        packages_19832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 15), self_19831, 'packages')
        # Applying the 'not' unary operator (line 101)
        result_not__19833 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 11), 'not', packages_19832)
        
        # Testing the type of an if condition (line 101)
        if_condition_19834 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 8), result_not__19833)
        # Assigning a type to the variable 'if_condition_19834' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'if_condition_19834', if_condition_19834)
        # SSA begins for if statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'data' (line 102)
        data_19835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 19), 'data')
        # Assigning a type to the variable 'stypy_return_type' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'stypy_return_type', data_19835)
        # SSA join for if statement (line 101)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 103)
        self_19836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 23), 'self')
        # Obtaining the member 'packages' of a type (line 103)
        packages_19837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 23), self_19836, 'packages')
        # Testing the type of a for loop iterable (line 103)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 103, 8), packages_19837)
        # Getting the type of the for loop variable (line 103)
        for_loop_var_19838 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 103, 8), packages_19837)
        # Assigning a type to the variable 'package' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'package', for_loop_var_19838)
        # SSA begins for a for statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to get_package_dir(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'package' (line 105)
        package_19841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 43), 'package', False)
        # Processing the call keyword arguments (line 105)
        kwargs_19842 = {}
        # Getting the type of 'self' (line 105)
        self_19839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 22), 'self', False)
        # Obtaining the member 'get_package_dir' of a type (line 105)
        get_package_dir_19840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 22), self_19839, 'get_package_dir')
        # Calling get_package_dir(args, kwargs) (line 105)
        get_package_dir_call_result_19843 = invoke(stypy.reporting.localization.Localization(__file__, 105, 22), get_package_dir_19840, *[package_19841], **kwargs_19842)
        
        # Assigning a type to the variable 'src_dir' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'src_dir', get_package_dir_call_result_19843)
        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Call to join(...): (line 108)
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_19847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        # Getting the type of 'self' (line 108)
        self_19848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), 'self', False)
        # Obtaining the member 'build_lib' of a type (line 108)
        build_lib_19849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 40), self_19848, 'build_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 39), list_19847, build_lib_19849)
        
        
        # Call to split(...): (line 108)
        # Processing the call arguments (line 108)
        str_19852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 72), 'str', '.')
        # Processing the call keyword arguments (line 108)
        kwargs_19853 = {}
        # Getting the type of 'package' (line 108)
        package_19850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 58), 'package', False)
        # Obtaining the member 'split' of a type (line 108)
        split_19851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 58), package_19850, 'split')
        # Calling split(args, kwargs) (line 108)
        split_call_result_19854 = invoke(stypy.reporting.localization.Localization(__file__, 108, 58), split_19851, *[str_19852], **kwargs_19853)
        
        # Applying the binary operator '+' (line 108)
        result_add_19855 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 39), '+', list_19847, split_call_result_19854)
        
        # Processing the call keyword arguments (line 108)
        kwargs_19856 = {}
        # Getting the type of 'os' (line 108)
        os_19844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 108)
        path_19845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 24), os_19844, 'path')
        # Obtaining the member 'join' of a type (line 108)
        join_19846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 24), path_19845, 'join')
        # Calling join(args, kwargs) (line 108)
        join_call_result_19857 = invoke(stypy.reporting.localization.Localization(__file__, 108, 24), join_19846, *[result_add_19855], **kwargs_19856)
        
        # Assigning a type to the variable 'build_dir' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'build_dir', join_call_result_19857)
        
        # Assigning a Num to a Name (line 111):
        
        # Assigning a Num to a Name (line 111):
        int_19858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 19), 'int')
        # Assigning a type to the variable 'plen' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'plen', int_19858)
        
        # Getting the type of 'src_dir' (line 112)
        src_dir_19859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'src_dir')
        # Testing the type of an if condition (line 112)
        if_condition_19860 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 12), src_dir_19859)
        # Assigning a type to the variable 'if_condition_19860' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'if_condition_19860', if_condition_19860)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 113):
        
        # Assigning a BinOp to a Name (line 113):
        
        # Call to len(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'src_dir' (line 113)
        src_dir_19862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'src_dir', False)
        # Processing the call keyword arguments (line 113)
        kwargs_19863 = {}
        # Getting the type of 'len' (line 113)
        len_19861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 23), 'len', False)
        # Calling len(args, kwargs) (line 113)
        len_call_result_19864 = invoke(stypy.reporting.localization.Localization(__file__, 113, 23), len_19861, *[src_dir_19862], **kwargs_19863)
        
        int_19865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 36), 'int')
        # Applying the binary operator '+' (line 113)
        result_add_19866 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 23), '+', len_call_result_19864, int_19865)
        
        # Assigning a type to the variable 'plen' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'plen', result_add_19866)
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a ListComp to a Name (line 116):
        
        # Assigning a ListComp to a Name (line 116):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to find_data_files(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'package' (line 117)
        package_19874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 61), 'package', False)
        # Getting the type of 'src_dir' (line 117)
        src_dir_19875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 70), 'src_dir', False)
        # Processing the call keyword arguments (line 117)
        kwargs_19876 = {}
        # Getting the type of 'self' (line 117)
        self_19872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 40), 'self', False)
        # Obtaining the member 'find_data_files' of a type (line 117)
        find_data_files_19873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 40), self_19872, 'find_data_files')
        # Calling find_data_files(args, kwargs) (line 117)
        find_data_files_call_result_19877 = invoke(stypy.reporting.localization.Localization(__file__, 117, 40), find_data_files_19873, *[package_19874, src_dir_19875], **kwargs_19876)
        
        comprehension_19878 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 16), find_data_files_call_result_19877)
        # Assigning a type to the variable 'file' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'file', comprehension_19878)
        
        # Obtaining the type of the subscript
        # Getting the type of 'plen' (line 117)
        plen_19867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 21), 'plen')
        slice_19868 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 117, 16), plen_19867, None, None)
        # Getting the type of 'file' (line 117)
        file_19869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'file')
        # Obtaining the member '__getitem__' of a type (line 117)
        getitem___19870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 16), file_19869, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 117)
        subscript_call_result_19871 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), getitem___19870, slice_19868)
        
        list_19879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 16), list_19879, subscript_call_result_19871)
        # Assigning a type to the variable 'filenames' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'filenames', list_19879)
        
        # Call to append(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Obtaining an instance of the builtin type 'tuple' (line 119)
        tuple_19882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 119)
        # Adding element type (line 119)
        # Getting the type of 'package' (line 119)
        package_19883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'package', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 25), tuple_19882, package_19883)
        # Adding element type (line 119)
        # Getting the type of 'src_dir' (line 119)
        src_dir_19884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 34), 'src_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 25), tuple_19882, src_dir_19884)
        # Adding element type (line 119)
        # Getting the type of 'build_dir' (line 119)
        build_dir_19885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 43), 'build_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 25), tuple_19882, build_dir_19885)
        # Adding element type (line 119)
        # Getting the type of 'filenames' (line 119)
        filenames_19886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 54), 'filenames', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 25), tuple_19882, filenames_19886)
        
        # Processing the call keyword arguments (line 119)
        kwargs_19887 = {}
        # Getting the type of 'data' (line 119)
        data_19880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'data', False)
        # Obtaining the member 'append' of a type (line 119)
        append_19881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), data_19880, 'append')
        # Calling append(args, kwargs) (line 119)
        append_call_result_19888 = invoke(stypy.reporting.localization.Localization(__file__, 119, 12), append_19881, *[tuple_19882], **kwargs_19887)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'data' (line 120)
        data_19889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'data')
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', data_19889)
        
        # ################# End of 'get_data_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_data_files' in the type store
        # Getting the type of 'stypy_return_type' (line 98)
        stypy_return_type_19890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19890)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_data_files'
        return stypy_return_type_19890


    @norecursion
    def find_data_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find_data_files'
        module_type_store = module_type_store.open_function_context('find_data_files', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.find_data_files.__dict__.__setitem__('stypy_localization', localization)
        build_py.find_data_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.find_data_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.find_data_files.__dict__.__setitem__('stypy_function_name', 'build_py.find_data_files')
        build_py.find_data_files.__dict__.__setitem__('stypy_param_names_list', ['package', 'src_dir'])
        build_py.find_data_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.find_data_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.find_data_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.find_data_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.find_data_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.find_data_files.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.find_data_files', ['package', 'src_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_data_files', localization, ['package', 'src_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_data_files(...)' code ##################

        str_19891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 8), 'str', "Return filenames for package's data files in 'src_dir'")
        
        # Assigning a BinOp to a Name (line 124):
        
        # Assigning a BinOp to a Name (line 124):
        
        # Call to get(...): (line 124)
        # Processing the call arguments (line 124)
        str_19895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 39), 'str', '')
        
        # Obtaining an instance of the builtin type 'list' (line 124)
        list_19896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 43), 'list')
        # Adding type elements to the builtin type 'list' instance (line 124)
        
        # Processing the call keyword arguments (line 124)
        kwargs_19897 = {}
        # Getting the type of 'self' (line 124)
        self_19892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 17), 'self', False)
        # Obtaining the member 'package_data' of a type (line 124)
        package_data_19893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 17), self_19892, 'package_data')
        # Obtaining the member 'get' of a type (line 124)
        get_19894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 17), package_data_19893, 'get')
        # Calling get(args, kwargs) (line 124)
        get_call_result_19898 = invoke(stypy.reporting.localization.Localization(__file__, 124, 17), get_19894, *[str_19895, list_19896], **kwargs_19897)
        
        
        # Call to get(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'package' (line 125)
        package_19902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 41), 'package', False)
        
        # Obtaining an instance of the builtin type 'list' (line 125)
        list_19903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 125)
        
        # Processing the call keyword arguments (line 125)
        kwargs_19904 = {}
        # Getting the type of 'self' (line 125)
        self_19899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'self', False)
        # Obtaining the member 'package_data' of a type (line 125)
        package_data_19900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 19), self_19899, 'package_data')
        # Obtaining the member 'get' of a type (line 125)
        get_19901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 19), package_data_19900, 'get')
        # Calling get(args, kwargs) (line 125)
        get_call_result_19905 = invoke(stypy.reporting.localization.Localization(__file__, 125, 19), get_19901, *[package_19902, list_19903], **kwargs_19904)
        
        # Applying the binary operator '+' (line 124)
        result_add_19906 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 17), '+', get_call_result_19898, get_call_result_19905)
        
        # Assigning a type to the variable 'globs' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'globs', result_add_19906)
        
        # Assigning a List to a Name (line 126):
        
        # Assigning a List to a Name (line 126):
        
        # Obtaining an instance of the builtin type 'list' (line 126)
        list_19907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 126)
        
        # Assigning a type to the variable 'files' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'files', list_19907)
        
        # Getting the type of 'globs' (line 127)
        globs_19908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'globs')
        # Testing the type of a for loop iterable (line 127)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 127, 8), globs_19908)
        # Getting the type of the for loop variable (line 127)
        for_loop_var_19909 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 127, 8), globs_19908)
        # Assigning a type to the variable 'pattern' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'pattern', for_loop_var_19909)
        # SSA begins for a for statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 129):
        
        # Assigning a Call to a Name (line 129):
        
        # Call to glob(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Call to join(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'src_dir' (line 129)
        src_dir_19914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 41), 'src_dir', False)
        
        # Call to convert_path(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'pattern' (line 129)
        pattern_19916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 63), 'pattern', False)
        # Processing the call keyword arguments (line 129)
        kwargs_19917 = {}
        # Getting the type of 'convert_path' (line 129)
        convert_path_19915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 50), 'convert_path', False)
        # Calling convert_path(args, kwargs) (line 129)
        convert_path_call_result_19918 = invoke(stypy.reporting.localization.Localization(__file__, 129, 50), convert_path_19915, *[pattern_19916], **kwargs_19917)
        
        # Processing the call keyword arguments (line 129)
        kwargs_19919 = {}
        # Getting the type of 'os' (line 129)
        os_19911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 129)
        path_19912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 28), os_19911, 'path')
        # Obtaining the member 'join' of a type (line 129)
        join_19913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 28), path_19912, 'join')
        # Calling join(args, kwargs) (line 129)
        join_call_result_19920 = invoke(stypy.reporting.localization.Localization(__file__, 129, 28), join_19913, *[src_dir_19914, convert_path_call_result_19918], **kwargs_19919)
        
        # Processing the call keyword arguments (line 129)
        kwargs_19921 = {}
        # Getting the type of 'glob' (line 129)
        glob_19910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'glob', False)
        # Calling glob(args, kwargs) (line 129)
        glob_call_result_19922 = invoke(stypy.reporting.localization.Localization(__file__, 129, 23), glob_19910, *[join_call_result_19920], **kwargs_19921)
        
        # Assigning a type to the variable 'filelist' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'filelist', glob_call_result_19922)
        
        # Call to extend(...): (line 131)
        # Processing the call arguments (line 131)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'filelist' (line 131)
        filelist_19936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 39), 'filelist', False)
        comprehension_19937 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 26), filelist_19936)
        # Assigning a type to the variable 'fn' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 26), 'fn', comprehension_19937)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'fn' (line 131)
        fn_19926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 51), 'fn', False)
        # Getting the type of 'files' (line 131)
        files_19927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 61), 'files', False)
        # Applying the binary operator 'notin' (line 131)
        result_contains_19928 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 51), 'notin', fn_19926, files_19927)
        
        
        # Call to isfile(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'fn' (line 132)
        fn_19932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 35), 'fn', False)
        # Processing the call keyword arguments (line 132)
        kwargs_19933 = {}
        # Getting the type of 'os' (line 132)
        os_19929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 132)
        path_19930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 20), os_19929, 'path')
        # Obtaining the member 'isfile' of a type (line 132)
        isfile_19931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 20), path_19930, 'isfile')
        # Calling isfile(args, kwargs) (line 132)
        isfile_call_result_19934 = invoke(stypy.reporting.localization.Localization(__file__, 132, 20), isfile_19931, *[fn_19932], **kwargs_19933)
        
        # Applying the binary operator 'and' (line 131)
        result_and_keyword_19935 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 51), 'and', result_contains_19928, isfile_call_result_19934)
        
        # Getting the type of 'fn' (line 131)
        fn_19925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 26), 'fn', False)
        list_19938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 26), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 26), list_19938, fn_19925)
        # Processing the call keyword arguments (line 131)
        kwargs_19939 = {}
        # Getting the type of 'files' (line 131)
        files_19923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'files', False)
        # Obtaining the member 'extend' of a type (line 131)
        extend_19924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), files_19923, 'extend')
        # Calling extend(args, kwargs) (line 131)
        extend_call_result_19940 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), extend_19924, *[list_19938], **kwargs_19939)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'files' (line 133)
        files_19941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'files')
        # Assigning a type to the variable 'stypy_return_type' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'stypy_return_type', files_19941)
        
        # ################# End of 'find_data_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_data_files' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_19942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19942)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_data_files'
        return stypy_return_type_19942


    @norecursion
    def build_package_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_package_data'
        module_type_store = module_type_store.open_function_context('build_package_data', 135, 4, False)
        # Assigning a type to the variable 'self' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.build_package_data.__dict__.__setitem__('stypy_localization', localization)
        build_py.build_package_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.build_package_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.build_package_data.__dict__.__setitem__('stypy_function_name', 'build_py.build_package_data')
        build_py.build_package_data.__dict__.__setitem__('stypy_param_names_list', [])
        build_py.build_package_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.build_package_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.build_package_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.build_package_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.build_package_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.build_package_data.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.build_package_data', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_package_data', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_package_data(...)' code ##################

        str_19943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 8), 'str', 'Copy data files into build directory')
        
        # Getting the type of 'self' (line 137)
        self_19944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 54), 'self')
        # Obtaining the member 'data_files' of a type (line 137)
        data_files_19945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 54), self_19944, 'data_files')
        # Testing the type of a for loop iterable (line 137)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 137, 8), data_files_19945)
        # Getting the type of the for loop variable (line 137)
        for_loop_var_19946 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 137, 8), data_files_19945)
        # Assigning a type to the variable 'package' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'package', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 8), for_loop_var_19946))
        # Assigning a type to the variable 'src_dir' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'src_dir', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 8), for_loop_var_19946))
        # Assigning a type to the variable 'build_dir' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'build_dir', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 8), for_loop_var_19946))
        # Assigning a type to the variable 'filenames' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'filenames', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 8), for_loop_var_19946))
        # SSA begins for a for statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'filenames' (line 138)
        filenames_19947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), 'filenames')
        # Testing the type of a for loop iterable (line 138)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 138, 12), filenames_19947)
        # Getting the type of the for loop variable (line 138)
        for_loop_var_19948 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 138, 12), filenames_19947)
        # Assigning a type to the variable 'filename' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'filename', for_loop_var_19948)
        # SSA begins for a for statement (line 138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 139):
        
        # Assigning a Call to a Name (line 139):
        
        # Call to join(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'build_dir' (line 139)
        build_dir_19952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 38), 'build_dir', False)
        # Getting the type of 'filename' (line 139)
        filename_19953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 49), 'filename', False)
        # Processing the call keyword arguments (line 139)
        kwargs_19954 = {}
        # Getting the type of 'os' (line 139)
        os_19949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 139)
        path_19950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 25), os_19949, 'path')
        # Obtaining the member 'join' of a type (line 139)
        join_19951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 25), path_19950, 'join')
        # Calling join(args, kwargs) (line 139)
        join_call_result_19955 = invoke(stypy.reporting.localization.Localization(__file__, 139, 25), join_19951, *[build_dir_19952, filename_19953], **kwargs_19954)
        
        # Assigning a type to the variable 'target' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'target', join_call_result_19955)
        
        # Call to mkpath(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Call to dirname(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'target' (line 140)
        target_19961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 44), 'target', False)
        # Processing the call keyword arguments (line 140)
        kwargs_19962 = {}
        # Getting the type of 'os' (line 140)
        os_19958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 140)
        path_19959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 28), os_19958, 'path')
        # Obtaining the member 'dirname' of a type (line 140)
        dirname_19960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 28), path_19959, 'dirname')
        # Calling dirname(args, kwargs) (line 140)
        dirname_call_result_19963 = invoke(stypy.reporting.localization.Localization(__file__, 140, 28), dirname_19960, *[target_19961], **kwargs_19962)
        
        # Processing the call keyword arguments (line 140)
        kwargs_19964 = {}
        # Getting the type of 'self' (line 140)
        self_19956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 140)
        mkpath_19957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 16), self_19956, 'mkpath')
        # Calling mkpath(args, kwargs) (line 140)
        mkpath_call_result_19965 = invoke(stypy.reporting.localization.Localization(__file__, 140, 16), mkpath_19957, *[dirname_call_result_19963], **kwargs_19964)
        
        
        # Call to copy_file(...): (line 141)
        # Processing the call arguments (line 141)
        
        # Call to join(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'src_dir' (line 141)
        src_dir_19971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 44), 'src_dir', False)
        # Getting the type of 'filename' (line 141)
        filename_19972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 53), 'filename', False)
        # Processing the call keyword arguments (line 141)
        kwargs_19973 = {}
        # Getting the type of 'os' (line 141)
        os_19968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 141)
        path_19969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 31), os_19968, 'path')
        # Obtaining the member 'join' of a type (line 141)
        join_19970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 31), path_19969, 'join')
        # Calling join(args, kwargs) (line 141)
        join_call_result_19974 = invoke(stypy.reporting.localization.Localization(__file__, 141, 31), join_19970, *[src_dir_19971, filename_19972], **kwargs_19973)
        
        # Getting the type of 'target' (line 141)
        target_19975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 64), 'target', False)
        # Processing the call keyword arguments (line 141)
        # Getting the type of 'False' (line 142)
        False_19976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 45), 'False', False)
        keyword_19977 = False_19976
        kwargs_19978 = {'preserve_mode': keyword_19977}
        # Getting the type of 'self' (line 141)
        self_19966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 16), 'self', False)
        # Obtaining the member 'copy_file' of a type (line 141)
        copy_file_19967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 16), self_19966, 'copy_file')
        # Calling copy_file(args, kwargs) (line 141)
        copy_file_call_result_19979 = invoke(stypy.reporting.localization.Localization(__file__, 141, 16), copy_file_19967, *[join_call_result_19974, target_19975], **kwargs_19978)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'build_package_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_package_data' in the type store
        # Getting the type of 'stypy_return_type' (line 135)
        stypy_return_type_19980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19980)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_package_data'
        return stypy_return_type_19980


    @norecursion
    def get_package_dir(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_package_dir'
        module_type_store = module_type_store.open_function_context('get_package_dir', 144, 4, False)
        # Assigning a type to the variable 'self' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.get_package_dir.__dict__.__setitem__('stypy_localization', localization)
        build_py.get_package_dir.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.get_package_dir.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.get_package_dir.__dict__.__setitem__('stypy_function_name', 'build_py.get_package_dir')
        build_py.get_package_dir.__dict__.__setitem__('stypy_param_names_list', ['package'])
        build_py.get_package_dir.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.get_package_dir.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.get_package_dir.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.get_package_dir.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.get_package_dir.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.get_package_dir.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.get_package_dir', ['package'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_package_dir', localization, ['package'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_package_dir(...)' code ##################

        str_19981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, (-1)), 'str', "Return the directory, relative to the top of the source\n           distribution, where package 'package' should be found\n           (at least according to the 'package_dir' option, if any).")
        
        # Assigning a Call to a Name (line 149):
        
        # Assigning a Call to a Name (line 149):
        
        # Call to split(...): (line 149)
        # Processing the call arguments (line 149)
        str_19984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 29), 'str', '.')
        # Processing the call keyword arguments (line 149)
        kwargs_19985 = {}
        # Getting the type of 'package' (line 149)
        package_19982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'package', False)
        # Obtaining the member 'split' of a type (line 149)
        split_19983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 15), package_19982, 'split')
        # Calling split(args, kwargs) (line 149)
        split_call_result_19986 = invoke(stypy.reporting.localization.Localization(__file__, 149, 15), split_19983, *[str_19984], **kwargs_19985)
        
        # Assigning a type to the variable 'path' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'path', split_call_result_19986)
        
        
        # Getting the type of 'self' (line 151)
        self_19987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'self')
        # Obtaining the member 'package_dir' of a type (line 151)
        package_dir_19988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 15), self_19987, 'package_dir')
        # Applying the 'not' unary operator (line 151)
        result_not__19989 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 11), 'not', package_dir_19988)
        
        # Testing the type of an if condition (line 151)
        if_condition_19990 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 8), result_not__19989)
        # Assigning a type to the variable 'if_condition_19990' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'if_condition_19990', if_condition_19990)
        # SSA begins for if statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'path' (line 152)
        path_19991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'path')
        # Testing the type of an if condition (line 152)
        if_condition_19992 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 12), path_19991)
        # Assigning a type to the variable 'if_condition_19992' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'if_condition_19992', if_condition_19992)
        # SSA begins for if statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to join(...): (line 153)
        # Getting the type of 'path' (line 153)
        path_19996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 37), 'path', False)
        # Processing the call keyword arguments (line 153)
        kwargs_19997 = {}
        # Getting the type of 'os' (line 153)
        os_19993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 153)
        path_19994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 23), os_19993, 'path')
        # Obtaining the member 'join' of a type (line 153)
        join_19995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 23), path_19994, 'join')
        # Calling join(args, kwargs) (line 153)
        join_call_result_19998 = invoke(stypy.reporting.localization.Localization(__file__, 153, 23), join_19995, *[path_19996], **kwargs_19997)
        
        # Assigning a type to the variable 'stypy_return_type' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'stypy_return_type', join_call_result_19998)
        # SSA branch for the else part of an if statement (line 152)
        module_type_store.open_ssa_branch('else')
        str_19999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 23), 'str', '')
        # Assigning a type to the variable 'stypy_return_type' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'stypy_return_type', str_19999)
        # SSA join for if statement (line 152)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 151)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 157):
        
        # Assigning a List to a Name (line 157):
        
        # Obtaining an instance of the builtin type 'list' (line 157)
        list_20000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 157)
        
        # Assigning a type to the variable 'tail' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'tail', list_20000)
        
        # Getting the type of 'path' (line 158)
        path_20001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 18), 'path')
        # Testing the type of an if condition (line 158)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 12), path_20001)
        # SSA begins for while statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        # SSA begins for try-except statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 160):
        
        # Assigning a Subscript to a Name (line 160):
        
        # Obtaining the type of the subscript
        
        # Call to join(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'path' (line 160)
        path_20004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 53), 'path', False)
        # Processing the call keyword arguments (line 160)
        kwargs_20005 = {}
        str_20002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 44), 'str', '.')
        # Obtaining the member 'join' of a type (line 160)
        join_20003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 44), str_20002, 'join')
        # Calling join(args, kwargs) (line 160)
        join_call_result_20006 = invoke(stypy.reporting.localization.Localization(__file__, 160, 44), join_20003, *[path_20004], **kwargs_20005)
        
        # Getting the type of 'self' (line 160)
        self_20007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 27), 'self')
        # Obtaining the member 'package_dir' of a type (line 160)
        package_dir_20008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 27), self_20007, 'package_dir')
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___20009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 27), package_dir_20008, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_20010 = invoke(stypy.reporting.localization.Localization(__file__, 160, 27), getitem___20009, join_call_result_20006)
        
        # Assigning a type to the variable 'pdir' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'pdir', subscript_call_result_20010)
        # SSA branch for the except part of a try statement (line 159)
        # SSA branch for the except 'KeyError' branch of a try statement (line 159)
        module_type_store.open_ssa_branch('except')
        
        # Call to insert(...): (line 162)
        # Processing the call arguments (line 162)
        int_20013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 32), 'int')
        
        # Obtaining the type of the subscript
        int_20014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 40), 'int')
        # Getting the type of 'path' (line 162)
        path_20015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 35), 'path', False)
        # Obtaining the member '__getitem__' of a type (line 162)
        getitem___20016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 35), path_20015, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 162)
        subscript_call_result_20017 = invoke(stypy.reporting.localization.Localization(__file__, 162, 35), getitem___20016, int_20014)
        
        # Processing the call keyword arguments (line 162)
        kwargs_20018 = {}
        # Getting the type of 'tail' (line 162)
        tail_20011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 20), 'tail', False)
        # Obtaining the member 'insert' of a type (line 162)
        insert_20012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 20), tail_20011, 'insert')
        # Calling insert(args, kwargs) (line 162)
        insert_call_result_20019 = invoke(stypy.reporting.localization.Localization(__file__, 162, 20), insert_20012, *[int_20013, subscript_call_result_20017], **kwargs_20018)
        
        # Deleting a member
        # Getting the type of 'path' (line 163)
        path_20020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 24), 'path')
        
        # Obtaining the type of the subscript
        int_20021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 29), 'int')
        # Getting the type of 'path' (line 163)
        path_20022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 24), 'path')
        # Obtaining the member '__getitem__' of a type (line 163)
        getitem___20023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 24), path_20022, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 163)
        subscript_call_result_20024 = invoke(stypy.reporting.localization.Localization(__file__, 163, 24), getitem___20023, int_20021)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 20), path_20020, subscript_call_result_20024)
        # SSA branch for the else branch of a try statement (line 159)
        module_type_store.open_ssa_branch('except else')
        
        # Call to insert(...): (line 165)
        # Processing the call arguments (line 165)
        int_20027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 32), 'int')
        # Getting the type of 'pdir' (line 165)
        pdir_20028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 35), 'pdir', False)
        # Processing the call keyword arguments (line 165)
        kwargs_20029 = {}
        # Getting the type of 'tail' (line 165)
        tail_20025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 20), 'tail', False)
        # Obtaining the member 'insert' of a type (line 165)
        insert_20026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 20), tail_20025, 'insert')
        # Calling insert(args, kwargs) (line 165)
        insert_call_result_20030 = invoke(stypy.reporting.localization.Localization(__file__, 165, 20), insert_20026, *[int_20027, pdir_20028], **kwargs_20029)
        
        
        # Call to join(...): (line 166)
        # Getting the type of 'tail' (line 166)
        tail_20034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 41), 'tail', False)
        # Processing the call keyword arguments (line 166)
        kwargs_20035 = {}
        # Getting the type of 'os' (line 166)
        os_20031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 166)
        path_20032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 27), os_20031, 'path')
        # Obtaining the member 'join' of a type (line 166)
        join_20033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 27), path_20032, 'join')
        # Calling join(args, kwargs) (line 166)
        join_call_result_20036 = invoke(stypy.reporting.localization.Localization(__file__, 166, 27), join_20033, *[tail_20034], **kwargs_20035)
        
        # Assigning a type to the variable 'stypy_return_type' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'stypy_return_type', join_call_result_20036)
        # SSA join for try-except statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a while statement (line 158)
        module_type_store.open_ssa_branch('while loop else')
        
        # Assigning a Call to a Name (line 175):
        
        # Assigning a Call to a Name (line 175):
        
        # Call to get(...): (line 175)
        # Processing the call arguments (line 175)
        str_20040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 44), 'str', '')
        # Processing the call keyword arguments (line 175)
        kwargs_20041 = {}
        # Getting the type of 'self' (line 175)
        self_20037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 23), 'self', False)
        # Obtaining the member 'package_dir' of a type (line 175)
        package_dir_20038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 23), self_20037, 'package_dir')
        # Obtaining the member 'get' of a type (line 175)
        get_20039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 23), package_dir_20038, 'get')
        # Calling get(args, kwargs) (line 175)
        get_call_result_20042 = invoke(stypy.reporting.localization.Localization(__file__, 175, 23), get_20039, *[str_20040], **kwargs_20041)
        
        # Assigning a type to the variable 'pdir' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 16), 'pdir', get_call_result_20042)
        
        # Type idiom detected: calculating its left and rigth part (line 176)
        # Getting the type of 'pdir' (line 176)
        pdir_20043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'pdir')
        # Getting the type of 'None' (line 176)
        None_20044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 31), 'None')
        
        (may_be_20045, more_types_in_union_20046) = may_not_be_none(pdir_20043, None_20044)

        if may_be_20045:

            if more_types_in_union_20046:
                # Runtime conditional SSA (line 176)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to insert(...): (line 177)
            # Processing the call arguments (line 177)
            int_20049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 32), 'int')
            # Getting the type of 'pdir' (line 177)
            pdir_20050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 35), 'pdir', False)
            # Processing the call keyword arguments (line 177)
            kwargs_20051 = {}
            # Getting the type of 'tail' (line 177)
            tail_20047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'tail', False)
            # Obtaining the member 'insert' of a type (line 177)
            insert_20048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 20), tail_20047, 'insert')
            # Calling insert(args, kwargs) (line 177)
            insert_call_result_20052 = invoke(stypy.reporting.localization.Localization(__file__, 177, 20), insert_20048, *[int_20049, pdir_20050], **kwargs_20051)
            

            if more_types_in_union_20046:
                # SSA join for if statement (line 176)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'tail' (line 179)
        tail_20053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 19), 'tail')
        # Testing the type of an if condition (line 179)
        if_condition_20054 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 16), tail_20053)
        # Assigning a type to the variable 'if_condition_20054' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'if_condition_20054', if_condition_20054)
        # SSA begins for if statement (line 179)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to join(...): (line 180)
        # Getting the type of 'tail' (line 180)
        tail_20058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 41), 'tail', False)
        # Processing the call keyword arguments (line 180)
        kwargs_20059 = {}
        # Getting the type of 'os' (line 180)
        os_20055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 180)
        path_20056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 27), os_20055, 'path')
        # Obtaining the member 'join' of a type (line 180)
        join_20057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 27), path_20056, 'join')
        # Calling join(args, kwargs) (line 180)
        join_call_result_20060 = invoke(stypy.reporting.localization.Localization(__file__, 180, 27), join_20057, *[tail_20058], **kwargs_20059)
        
        # Assigning a type to the variable 'stypy_return_type' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 20), 'stypy_return_type', join_call_result_20060)
        # SSA branch for the else part of an if statement (line 179)
        module_type_store.open_ssa_branch('else')
        str_20061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 27), 'str', '')
        # Assigning a type to the variable 'stypy_return_type' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 20), 'stypy_return_type', str_20061)
        # SSA join for if statement (line 179)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 158)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 151)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_package_dir(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_package_dir' in the type store
        # Getting the type of 'stypy_return_type' (line 144)
        stypy_return_type_20062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20062)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_package_dir'
        return stypy_return_type_20062


    @norecursion
    def check_package(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_package'
        module_type_store = module_type_store.open_function_context('check_package', 184, 4, False)
        # Assigning a type to the variable 'self' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.check_package.__dict__.__setitem__('stypy_localization', localization)
        build_py.check_package.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.check_package.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.check_package.__dict__.__setitem__('stypy_function_name', 'build_py.check_package')
        build_py.check_package.__dict__.__setitem__('stypy_param_names_list', ['package', 'package_dir'])
        build_py.check_package.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.check_package.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.check_package.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.check_package.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.check_package.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.check_package.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.check_package', ['package', 'package_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_package', localization, ['package', 'package_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_package(...)' code ##################

        
        
        # Getting the type of 'package_dir' (line 189)
        package_dir_20063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 11), 'package_dir')
        str_20064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 26), 'str', '')
        # Applying the binary operator '!=' (line 189)
        result_ne_20065 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 11), '!=', package_dir_20063, str_20064)
        
        # Testing the type of an if condition (line 189)
        if_condition_20066 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 8), result_ne_20065)
        # Assigning a type to the variable 'if_condition_20066' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'if_condition_20066', if_condition_20066)
        # SSA begins for if statement (line 189)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to exists(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'package_dir' (line 190)
        package_dir_20070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 34), 'package_dir', False)
        # Processing the call keyword arguments (line 190)
        kwargs_20071 = {}
        # Getting the type of 'os' (line 190)
        os_20067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 190)
        path_20068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 19), os_20067, 'path')
        # Obtaining the member 'exists' of a type (line 190)
        exists_20069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 19), path_20068, 'exists')
        # Calling exists(args, kwargs) (line 190)
        exists_call_result_20072 = invoke(stypy.reporting.localization.Localization(__file__, 190, 19), exists_20069, *[package_dir_20070], **kwargs_20071)
        
        # Applying the 'not' unary operator (line 190)
        result_not__20073 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 15), 'not', exists_call_result_20072)
        
        # Testing the type of an if condition (line 190)
        if_condition_20074 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 12), result_not__20073)
        # Assigning a type to the variable 'if_condition_20074' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'if_condition_20074', if_condition_20074)
        # SSA begins for if statement (line 190)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsFileError(...): (line 191)
        # Processing the call arguments (line 191)
        str_20076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 22), 'str', "package directory '%s' does not exist")
        # Getting the type of 'package_dir' (line 192)
        package_dir_20077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 64), 'package_dir', False)
        # Applying the binary operator '%' (line 192)
        result_mod_20078 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 22), '%', str_20076, package_dir_20077)
        
        # Processing the call keyword arguments (line 191)
        kwargs_20079 = {}
        # Getting the type of 'DistutilsFileError' (line 191)
        DistutilsFileError_20075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 22), 'DistutilsFileError', False)
        # Calling DistutilsFileError(args, kwargs) (line 191)
        DistutilsFileError_call_result_20080 = invoke(stypy.reporting.localization.Localization(__file__, 191, 22), DistutilsFileError_20075, *[result_mod_20078], **kwargs_20079)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 191, 16), DistutilsFileError_call_result_20080, 'raise parameter', BaseException)
        # SSA join for if statement (line 190)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to isdir(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'package_dir' (line 193)
        package_dir_20084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 33), 'package_dir', False)
        # Processing the call keyword arguments (line 193)
        kwargs_20085 = {}
        # Getting the type of 'os' (line 193)
        os_20081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 193)
        path_20082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 19), os_20081, 'path')
        # Obtaining the member 'isdir' of a type (line 193)
        isdir_20083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 19), path_20082, 'isdir')
        # Calling isdir(args, kwargs) (line 193)
        isdir_call_result_20086 = invoke(stypy.reporting.localization.Localization(__file__, 193, 19), isdir_20083, *[package_dir_20084], **kwargs_20085)
        
        # Applying the 'not' unary operator (line 193)
        result_not__20087 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 15), 'not', isdir_call_result_20086)
        
        # Testing the type of an if condition (line 193)
        if_condition_20088 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 12), result_not__20087)
        # Assigning a type to the variable 'if_condition_20088' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'if_condition_20088', if_condition_20088)
        # SSA begins for if statement (line 193)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsFileError(...): (line 194)
        # Processing the call arguments (line 194)
        str_20090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 23), 'str', "supposed package directory '%s' exists, but is not a directory")
        # Getting the type of 'package_dir' (line 196)
        package_dir_20091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 50), 'package_dir', False)
        # Applying the binary operator '%' (line 195)
        result_mod_20092 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 23), '%', str_20090, package_dir_20091)
        
        # Processing the call keyword arguments (line 194)
        kwargs_20093 = {}
        # Getting the type of 'DistutilsFileError' (line 194)
        DistutilsFileError_20089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 22), 'DistutilsFileError', False)
        # Calling DistutilsFileError(args, kwargs) (line 194)
        DistutilsFileError_call_result_20094 = invoke(stypy.reporting.localization.Localization(__file__, 194, 22), DistutilsFileError_20089, *[result_mod_20092], **kwargs_20093)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 194, 16), DistutilsFileError_call_result_20094, 'raise parameter', BaseException)
        # SSA join for if statement (line 193)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 189)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'package' (line 199)
        package_20095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'package')
        # Testing the type of an if condition (line 199)
        if_condition_20096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 8), package_20095)
        # Assigning a type to the variable 'if_condition_20096' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'if_condition_20096', if_condition_20096)
        # SSA begins for if statement (line 199)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 200):
        
        # Assigning a Call to a Name (line 200):
        
        # Call to join(...): (line 200)
        # Processing the call arguments (line 200)
        # Getting the type of 'package_dir' (line 200)
        package_dir_20100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 35), 'package_dir', False)
        str_20101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 48), 'str', '__init__.py')
        # Processing the call keyword arguments (line 200)
        kwargs_20102 = {}
        # Getting the type of 'os' (line 200)
        os_20097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 200)
        path_20098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 22), os_20097, 'path')
        # Obtaining the member 'join' of a type (line 200)
        join_20099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 22), path_20098, 'join')
        # Calling join(args, kwargs) (line 200)
        join_call_result_20103 = invoke(stypy.reporting.localization.Localization(__file__, 200, 22), join_20099, *[package_dir_20100, str_20101], **kwargs_20102)
        
        # Assigning a type to the variable 'init_py' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'init_py', join_call_result_20103)
        
        
        # Call to isfile(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'init_py' (line 201)
        init_py_20107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 30), 'init_py', False)
        # Processing the call keyword arguments (line 201)
        kwargs_20108 = {}
        # Getting the type of 'os' (line 201)
        os_20104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 201)
        path_20105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 15), os_20104, 'path')
        # Obtaining the member 'isfile' of a type (line 201)
        isfile_20106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 15), path_20105, 'isfile')
        # Calling isfile(args, kwargs) (line 201)
        isfile_call_result_20109 = invoke(stypy.reporting.localization.Localization(__file__, 201, 15), isfile_20106, *[init_py_20107], **kwargs_20108)
        
        # Testing the type of an if condition (line 201)
        if_condition_20110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 12), isfile_call_result_20109)
        # Assigning a type to the variable 'if_condition_20110' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'if_condition_20110', if_condition_20110)
        # SSA begins for if statement (line 201)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'init_py' (line 202)
        init_py_20111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 23), 'init_py')
        # Assigning a type to the variable 'stypy_return_type' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'stypy_return_type', init_py_20111)
        # SSA branch for the else part of an if statement (line 201)
        module_type_store.open_ssa_branch('else')
        
        # Call to warn(...): (line 204)
        # Processing the call arguments (line 204)
        str_20114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 26), 'str', "package init file '%s' not found ")
        str_20115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 26), 'str', '(or not a regular file)')
        # Applying the binary operator '+' (line 204)
        result_add_20116 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 26), '+', str_20114, str_20115)
        
        # Getting the type of 'init_py' (line 205)
        init_py_20117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 54), 'init_py', False)
        # Processing the call keyword arguments (line 204)
        kwargs_20118 = {}
        # Getting the type of 'log' (line 204)
        log_20112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'log', False)
        # Obtaining the member 'warn' of a type (line 204)
        warn_20113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 16), log_20112, 'warn')
        # Calling warn(args, kwargs) (line 204)
        warn_call_result_20119 = invoke(stypy.reporting.localization.Localization(__file__, 204, 16), warn_20113, *[result_add_20116, init_py_20117], **kwargs_20118)
        
        # SSA join for if statement (line 201)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 199)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'None' (line 209)
        None_20120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'stypy_return_type', None_20120)
        
        # ################# End of 'check_package(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_package' in the type store
        # Getting the type of 'stypy_return_type' (line 184)
        stypy_return_type_20121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20121)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_package'
        return stypy_return_type_20121


    @norecursion
    def check_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_module'
        module_type_store = module_type_store.open_function_context('check_module', 211, 4, False)
        # Assigning a type to the variable 'self' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.check_module.__dict__.__setitem__('stypy_localization', localization)
        build_py.check_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.check_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.check_module.__dict__.__setitem__('stypy_function_name', 'build_py.check_module')
        build_py.check_module.__dict__.__setitem__('stypy_param_names_list', ['module', 'module_file'])
        build_py.check_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.check_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.check_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.check_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.check_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.check_module.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.check_module', ['module', 'module_file'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_module', localization, ['module', 'module_file'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_module(...)' code ##################

        
        
        
        # Call to isfile(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'module_file' (line 212)
        module_file_20125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 30), 'module_file', False)
        # Processing the call keyword arguments (line 212)
        kwargs_20126 = {}
        # Getting the type of 'os' (line 212)
        os_20122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 212)
        path_20123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 15), os_20122, 'path')
        # Obtaining the member 'isfile' of a type (line 212)
        isfile_20124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 15), path_20123, 'isfile')
        # Calling isfile(args, kwargs) (line 212)
        isfile_call_result_20127 = invoke(stypy.reporting.localization.Localization(__file__, 212, 15), isfile_20124, *[module_file_20125], **kwargs_20126)
        
        # Applying the 'not' unary operator (line 212)
        result_not__20128 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 11), 'not', isfile_call_result_20127)
        
        # Testing the type of an if condition (line 212)
        if_condition_20129 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 8), result_not__20128)
        # Assigning a type to the variable 'if_condition_20129' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'if_condition_20129', if_condition_20129)
        # SSA begins for if statement (line 212)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 213)
        # Processing the call arguments (line 213)
        str_20132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 21), 'str', 'file %s (for module %s) not found')
        # Getting the type of 'module_file' (line 213)
        module_file_20133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 58), 'module_file', False)
        # Getting the type of 'module' (line 213)
        module_20134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 71), 'module', False)
        # Processing the call keyword arguments (line 213)
        kwargs_20135 = {}
        # Getting the type of 'log' (line 213)
        log_20130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'log', False)
        # Obtaining the member 'warn' of a type (line 213)
        warn_20131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 12), log_20130, 'warn')
        # Calling warn(args, kwargs) (line 213)
        warn_call_result_20136 = invoke(stypy.reporting.localization.Localization(__file__, 213, 12), warn_20131, *[str_20132, module_file_20133, module_20134], **kwargs_20135)
        
        # Getting the type of 'False' (line 214)
        False_20137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'stypy_return_type', False_20137)
        # SSA branch for the else part of an if statement (line 212)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'True' (line 216)
        True_20138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'stypy_return_type', True_20138)
        # SSA join for if statement (line 212)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_module' in the type store
        # Getting the type of 'stypy_return_type' (line 211)
        stypy_return_type_20139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20139)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_module'
        return stypy_return_type_20139


    @norecursion
    def find_package_modules(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find_package_modules'
        module_type_store = module_type_store.open_function_context('find_package_modules', 218, 4, False)
        # Assigning a type to the variable 'self' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.find_package_modules.__dict__.__setitem__('stypy_localization', localization)
        build_py.find_package_modules.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.find_package_modules.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.find_package_modules.__dict__.__setitem__('stypy_function_name', 'build_py.find_package_modules')
        build_py.find_package_modules.__dict__.__setitem__('stypy_param_names_list', ['package', 'package_dir'])
        build_py.find_package_modules.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.find_package_modules.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.find_package_modules.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.find_package_modules.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.find_package_modules.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.find_package_modules.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.find_package_modules', ['package', 'package_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_package_modules', localization, ['package', 'package_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_package_modules(...)' code ##################

        
        # Call to check_package(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'package' (line 219)
        package_20142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 27), 'package', False)
        # Getting the type of 'package_dir' (line 219)
        package_dir_20143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 36), 'package_dir', False)
        # Processing the call keyword arguments (line 219)
        kwargs_20144 = {}
        # Getting the type of 'self' (line 219)
        self_20140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'self', False)
        # Obtaining the member 'check_package' of a type (line 219)
        check_package_20141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), self_20140, 'check_package')
        # Calling check_package(args, kwargs) (line 219)
        check_package_call_result_20145 = invoke(stypy.reporting.localization.Localization(__file__, 219, 8), check_package_20141, *[package_20142, package_dir_20143], **kwargs_20144)
        
        
        # Assigning a Call to a Name (line 220):
        
        # Assigning a Call to a Name (line 220):
        
        # Call to glob(...): (line 220)
        # Processing the call arguments (line 220)
        
        # Call to join(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'package_dir' (line 220)
        package_dir_20150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 41), 'package_dir', False)
        str_20151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 54), 'str', '*.py')
        # Processing the call keyword arguments (line 220)
        kwargs_20152 = {}
        # Getting the type of 'os' (line 220)
        os_20147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 220)
        path_20148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 28), os_20147, 'path')
        # Obtaining the member 'join' of a type (line 220)
        join_20149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 28), path_20148, 'join')
        # Calling join(args, kwargs) (line 220)
        join_call_result_20153 = invoke(stypy.reporting.localization.Localization(__file__, 220, 28), join_20149, *[package_dir_20150, str_20151], **kwargs_20152)
        
        # Processing the call keyword arguments (line 220)
        kwargs_20154 = {}
        # Getting the type of 'glob' (line 220)
        glob_20146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 23), 'glob', False)
        # Calling glob(args, kwargs) (line 220)
        glob_call_result_20155 = invoke(stypy.reporting.localization.Localization(__file__, 220, 23), glob_20146, *[join_call_result_20153], **kwargs_20154)
        
        # Assigning a type to the variable 'module_files' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'module_files', glob_call_result_20155)
        
        # Assigning a List to a Name (line 221):
        
        # Assigning a List to a Name (line 221):
        
        # Obtaining an instance of the builtin type 'list' (line 221)
        list_20156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 221)
        
        # Assigning a type to the variable 'modules' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'modules', list_20156)
        
        # Assigning a Call to a Name (line 222):
        
        # Assigning a Call to a Name (line 222):
        
        # Call to abspath(...): (line 222)
        # Processing the call arguments (line 222)
        # Getting the type of 'self' (line 222)
        self_20160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 39), 'self', False)
        # Obtaining the member 'distribution' of a type (line 222)
        distribution_20161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 39), self_20160, 'distribution')
        # Obtaining the member 'script_name' of a type (line 222)
        script_name_20162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 39), distribution_20161, 'script_name')
        # Processing the call keyword arguments (line 222)
        kwargs_20163 = {}
        # Getting the type of 'os' (line 222)
        os_20157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 222)
        path_20158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 23), os_20157, 'path')
        # Obtaining the member 'abspath' of a type (line 222)
        abspath_20159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 23), path_20158, 'abspath')
        # Calling abspath(args, kwargs) (line 222)
        abspath_call_result_20164 = invoke(stypy.reporting.localization.Localization(__file__, 222, 23), abspath_20159, *[script_name_20162], **kwargs_20163)
        
        # Assigning a type to the variable 'setup_script' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'setup_script', abspath_call_result_20164)
        
        # Getting the type of 'module_files' (line 224)
        module_files_20165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 17), 'module_files')
        # Testing the type of a for loop iterable (line 224)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 224, 8), module_files_20165)
        # Getting the type of the for loop variable (line 224)
        for_loop_var_20166 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 224, 8), module_files_20165)
        # Assigning a type to the variable 'f' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'f', for_loop_var_20166)
        # SSA begins for a for statement (line 224)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 225):
        
        # Assigning a Call to a Name (line 225):
        
        # Call to abspath(...): (line 225)
        # Processing the call arguments (line 225)
        # Getting the type of 'f' (line 225)
        f_20170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 36), 'f', False)
        # Processing the call keyword arguments (line 225)
        kwargs_20171 = {}
        # Getting the type of 'os' (line 225)
        os_20167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 225)
        path_20168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 20), os_20167, 'path')
        # Obtaining the member 'abspath' of a type (line 225)
        abspath_20169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 20), path_20168, 'abspath')
        # Calling abspath(args, kwargs) (line 225)
        abspath_call_result_20172 = invoke(stypy.reporting.localization.Localization(__file__, 225, 20), abspath_20169, *[f_20170], **kwargs_20171)
        
        # Assigning a type to the variable 'abs_f' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'abs_f', abspath_call_result_20172)
        
        
        # Getting the type of 'abs_f' (line 226)
        abs_f_20173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'abs_f')
        # Getting the type of 'setup_script' (line 226)
        setup_script_20174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'setup_script')
        # Applying the binary operator '!=' (line 226)
        result_ne_20175 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 15), '!=', abs_f_20173, setup_script_20174)
        
        # Testing the type of an if condition (line 226)
        if_condition_20176 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 12), result_ne_20175)
        # Assigning a type to the variable 'if_condition_20176' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'if_condition_20176', if_condition_20176)
        # SSA begins for if statement (line 226)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 227):
        
        # Assigning a Subscript to a Name (line 227):
        
        # Obtaining the type of the subscript
        int_20177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 63), 'int')
        
        # Call to splitext(...): (line 227)
        # Processing the call arguments (line 227)
        
        # Call to basename(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'f' (line 227)
        f_20184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 59), 'f', False)
        # Processing the call keyword arguments (line 227)
        kwargs_20185 = {}
        # Getting the type of 'os' (line 227)
        os_20181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 42), 'os', False)
        # Obtaining the member 'path' of a type (line 227)
        path_20182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 42), os_20181, 'path')
        # Obtaining the member 'basename' of a type (line 227)
        basename_20183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 42), path_20182, 'basename')
        # Calling basename(args, kwargs) (line 227)
        basename_call_result_20186 = invoke(stypy.reporting.localization.Localization(__file__, 227, 42), basename_20183, *[f_20184], **kwargs_20185)
        
        # Processing the call keyword arguments (line 227)
        kwargs_20187 = {}
        # Getting the type of 'os' (line 227)
        os_20178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 227)
        path_20179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 25), os_20178, 'path')
        # Obtaining the member 'splitext' of a type (line 227)
        splitext_20180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 25), path_20179, 'splitext')
        # Calling splitext(args, kwargs) (line 227)
        splitext_call_result_20188 = invoke(stypy.reporting.localization.Localization(__file__, 227, 25), splitext_20180, *[basename_call_result_20186], **kwargs_20187)
        
        # Obtaining the member '__getitem__' of a type (line 227)
        getitem___20189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 25), splitext_call_result_20188, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 227)
        subscript_call_result_20190 = invoke(stypy.reporting.localization.Localization(__file__, 227, 25), getitem___20189, int_20177)
        
        # Assigning a type to the variable 'module' (line 227)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'module', subscript_call_result_20190)
        
        # Call to append(...): (line 228)
        # Processing the call arguments (line 228)
        
        # Obtaining an instance of the builtin type 'tuple' (line 228)
        tuple_20193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 228)
        # Adding element type (line 228)
        # Getting the type of 'package' (line 228)
        package_20194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 32), 'package', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 32), tuple_20193, package_20194)
        # Adding element type (line 228)
        # Getting the type of 'module' (line 228)
        module_20195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 41), 'module', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 32), tuple_20193, module_20195)
        # Adding element type (line 228)
        # Getting the type of 'f' (line 228)
        f_20196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 49), 'f', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 32), tuple_20193, f_20196)
        
        # Processing the call keyword arguments (line 228)
        kwargs_20197 = {}
        # Getting the type of 'modules' (line 228)
        modules_20191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'modules', False)
        # Obtaining the member 'append' of a type (line 228)
        append_20192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 16), modules_20191, 'append')
        # Calling append(args, kwargs) (line 228)
        append_call_result_20198 = invoke(stypy.reporting.localization.Localization(__file__, 228, 16), append_20192, *[tuple_20193], **kwargs_20197)
        
        # SSA branch for the else part of an if statement (line 226)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug_print(...): (line 230)
        # Processing the call arguments (line 230)
        str_20201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 33), 'str', 'excluding %s')
        # Getting the type of 'setup_script' (line 230)
        setup_script_20202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 50), 'setup_script', False)
        # Applying the binary operator '%' (line 230)
        result_mod_20203 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 33), '%', str_20201, setup_script_20202)
        
        # Processing the call keyword arguments (line 230)
        kwargs_20204 = {}
        # Getting the type of 'self' (line 230)
        self_20199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'self', False)
        # Obtaining the member 'debug_print' of a type (line 230)
        debug_print_20200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 16), self_20199, 'debug_print')
        # Calling debug_print(args, kwargs) (line 230)
        debug_print_call_result_20205 = invoke(stypy.reporting.localization.Localization(__file__, 230, 16), debug_print_20200, *[result_mod_20203], **kwargs_20204)
        
        # SSA join for if statement (line 226)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'modules' (line 231)
        modules_20206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'modules')
        # Assigning a type to the variable 'stypy_return_type' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'stypy_return_type', modules_20206)
        
        # ################# End of 'find_package_modules(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_package_modules' in the type store
        # Getting the type of 'stypy_return_type' (line 218)
        stypy_return_type_20207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20207)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_package_modules'
        return stypy_return_type_20207


    @norecursion
    def find_modules(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find_modules'
        module_type_store = module_type_store.open_function_context('find_modules', 233, 4, False)
        # Assigning a type to the variable 'self' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.find_modules.__dict__.__setitem__('stypy_localization', localization)
        build_py.find_modules.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.find_modules.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.find_modules.__dict__.__setitem__('stypy_function_name', 'build_py.find_modules')
        build_py.find_modules.__dict__.__setitem__('stypy_param_names_list', [])
        build_py.find_modules.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.find_modules.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.find_modules.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.find_modules.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.find_modules.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.find_modules.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.find_modules', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_modules', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_modules(...)' code ##################

        str_20208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, (-1)), 'str', 'Finds individually-specified Python modules, ie. those listed by\n        module name in \'self.py_modules\'.  Returns a list of tuples (package,\n        module_base, filename): \'package\' is a tuple of the path through\n        package-space to the module; \'module_base\' is the bare (no\n        packages, no dots) module name, and \'filename\' is the path to the\n        ".py" file (relative to the distribution root) that implements the\n        module.\n        ')
        
        # Assigning a Dict to a Name (line 248):
        
        # Assigning a Dict to a Name (line 248):
        
        # Obtaining an instance of the builtin type 'dict' (line 248)
        dict_20209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 248)
        
        # Assigning a type to the variable 'packages' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'packages', dict_20209)
        
        # Assigning a List to a Name (line 251):
        
        # Assigning a List to a Name (line 251):
        
        # Obtaining an instance of the builtin type 'list' (line 251)
        list_20210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 251)
        
        # Assigning a type to the variable 'modules' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'modules', list_20210)
        
        # Getting the type of 'self' (line 257)
        self_20211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 22), 'self')
        # Obtaining the member 'py_modules' of a type (line 257)
        py_modules_20212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 22), self_20211, 'py_modules')
        # Testing the type of a for loop iterable (line 257)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 257, 8), py_modules_20212)
        # Getting the type of the for loop variable (line 257)
        for_loop_var_20213 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 257, 8), py_modules_20212)
        # Assigning a type to the variable 'module' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'module', for_loop_var_20213)
        # SSA begins for a for statement (line 257)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 258):
        
        # Assigning a Call to a Name (line 258):
        
        # Call to split(...): (line 258)
        # Processing the call arguments (line 258)
        str_20216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 32), 'str', '.')
        # Processing the call keyword arguments (line 258)
        kwargs_20217 = {}
        # Getting the type of 'module' (line 258)
        module_20214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 19), 'module', False)
        # Obtaining the member 'split' of a type (line 258)
        split_20215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 19), module_20214, 'split')
        # Calling split(args, kwargs) (line 258)
        split_call_result_20218 = invoke(stypy.reporting.localization.Localization(__file__, 258, 19), split_20215, *[str_20216], **kwargs_20217)
        
        # Assigning a type to the variable 'path' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'path', split_call_result_20218)
        
        # Assigning a Call to a Name (line 259):
        
        # Assigning a Call to a Name (line 259):
        
        # Call to join(...): (line 259)
        # Processing the call arguments (line 259)
        
        # Obtaining the type of the subscript
        int_20221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 36), 'int')
        int_20222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 38), 'int')
        slice_20223 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 259, 31), int_20221, int_20222, None)
        # Getting the type of 'path' (line 259)
        path_20224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 31), 'path', False)
        # Obtaining the member '__getitem__' of a type (line 259)
        getitem___20225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 31), path_20224, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 259)
        subscript_call_result_20226 = invoke(stypy.reporting.localization.Localization(__file__, 259, 31), getitem___20225, slice_20223)
        
        # Processing the call keyword arguments (line 259)
        kwargs_20227 = {}
        str_20219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 22), 'str', '.')
        # Obtaining the member 'join' of a type (line 259)
        join_20220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 22), str_20219, 'join')
        # Calling join(args, kwargs) (line 259)
        join_call_result_20228 = invoke(stypy.reporting.localization.Localization(__file__, 259, 22), join_20220, *[subscript_call_result_20226], **kwargs_20227)
        
        # Assigning a type to the variable 'package' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'package', join_call_result_20228)
        
        # Assigning a Subscript to a Name (line 260):
        
        # Assigning a Subscript to a Name (line 260):
        
        # Obtaining the type of the subscript
        int_20229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 31), 'int')
        # Getting the type of 'path' (line 260)
        path_20230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 26), 'path')
        # Obtaining the member '__getitem__' of a type (line 260)
        getitem___20231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 26), path_20230, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 260)
        subscript_call_result_20232 = invoke(stypy.reporting.localization.Localization(__file__, 260, 26), getitem___20231, int_20229)
        
        # Assigning a type to the variable 'module_base' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'module_base', subscript_call_result_20232)
        
        
        # SSA begins for try-except statement (line 262)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Tuple (line 263):
        
        # Assigning a Subscript to a Name (line 263):
        
        # Obtaining the type of the subscript
        int_20233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 16), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'package' (line 263)
        package_20234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 50), 'package')
        # Getting the type of 'packages' (line 263)
        packages_20235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 41), 'packages')
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___20236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 41), packages_20235, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_20237 = invoke(stypy.reporting.localization.Localization(__file__, 263, 41), getitem___20236, package_20234)
        
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___20238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 16), subscript_call_result_20237, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_20239 = invoke(stypy.reporting.localization.Localization(__file__, 263, 16), getitem___20238, int_20233)
        
        # Assigning a type to the variable 'tuple_var_assignment_19699' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), 'tuple_var_assignment_19699', subscript_call_result_20239)
        
        # Assigning a Subscript to a Name (line 263):
        
        # Obtaining the type of the subscript
        int_20240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, 16), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'package' (line 263)
        package_20241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 50), 'package')
        # Getting the type of 'packages' (line 263)
        packages_20242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 41), 'packages')
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___20243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 41), packages_20242, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_20244 = invoke(stypy.reporting.localization.Localization(__file__, 263, 41), getitem___20243, package_20241)
        
        # Obtaining the member '__getitem__' of a type (line 263)
        getitem___20245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 16), subscript_call_result_20244, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 263)
        subscript_call_result_20246 = invoke(stypy.reporting.localization.Localization(__file__, 263, 16), getitem___20245, int_20240)
        
        # Assigning a type to the variable 'tuple_var_assignment_19700' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), 'tuple_var_assignment_19700', subscript_call_result_20246)
        
        # Assigning a Name to a Name (line 263):
        # Getting the type of 'tuple_var_assignment_19699' (line 263)
        tuple_var_assignment_19699_20247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), 'tuple_var_assignment_19699')
        # Assigning a type to the variable 'package_dir' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 17), 'package_dir', tuple_var_assignment_19699_20247)
        
        # Assigning a Name to a Name (line 263):
        # Getting the type of 'tuple_var_assignment_19700' (line 263)
        tuple_var_assignment_19700_20248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 16), 'tuple_var_assignment_19700')
        # Assigning a type to the variable 'checked' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 30), 'checked', tuple_var_assignment_19700_20248)
        # SSA branch for the except part of a try statement (line 262)
        # SSA branch for the except 'KeyError' branch of a try statement (line 262)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 265):
        
        # Assigning a Call to a Name (line 265):
        
        # Call to get_package_dir(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'package' (line 265)
        package_20251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 51), 'package', False)
        # Processing the call keyword arguments (line 265)
        kwargs_20252 = {}
        # Getting the type of 'self' (line 265)
        self_20249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 30), 'self', False)
        # Obtaining the member 'get_package_dir' of a type (line 265)
        get_package_dir_20250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 30), self_20249, 'get_package_dir')
        # Calling get_package_dir(args, kwargs) (line 265)
        get_package_dir_call_result_20253 = invoke(stypy.reporting.localization.Localization(__file__, 265, 30), get_package_dir_20250, *[package_20251], **kwargs_20252)
        
        # Assigning a type to the variable 'package_dir' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 16), 'package_dir', get_package_dir_call_result_20253)
        
        # Assigning a Num to a Name (line 266):
        
        # Assigning a Num to a Name (line 266):
        int_20254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 26), 'int')
        # Assigning a type to the variable 'checked' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'checked', int_20254)
        # SSA join for try-except statement (line 262)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'checked' (line 268)
        checked_20255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 19), 'checked')
        # Applying the 'not' unary operator (line 268)
        result_not__20256 = python_operator(stypy.reporting.localization.Localization(__file__, 268, 15), 'not', checked_20255)
        
        # Testing the type of an if condition (line 268)
        if_condition_20257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 12), result_not__20256)
        # Assigning a type to the variable 'if_condition_20257' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'if_condition_20257', if_condition_20257)
        # SSA begins for if statement (line 268)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 269):
        
        # Assigning a Call to a Name (line 269):
        
        # Call to check_package(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'package' (line 269)
        package_20260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 45), 'package', False)
        # Getting the type of 'package_dir' (line 269)
        package_dir_20261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 54), 'package_dir', False)
        # Processing the call keyword arguments (line 269)
        kwargs_20262 = {}
        # Getting the type of 'self' (line 269)
        self_20258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 26), 'self', False)
        # Obtaining the member 'check_package' of a type (line 269)
        check_package_20259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 26), self_20258, 'check_package')
        # Calling check_package(args, kwargs) (line 269)
        check_package_call_result_20263 = invoke(stypy.reporting.localization.Localization(__file__, 269, 26), check_package_20259, *[package_20260, package_dir_20261], **kwargs_20262)
        
        # Assigning a type to the variable 'init_py' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 16), 'init_py', check_package_call_result_20263)
        
        # Assigning a Tuple to a Subscript (line 270):
        
        # Assigning a Tuple to a Subscript (line 270):
        
        # Obtaining an instance of the builtin type 'tuple' (line 270)
        tuple_20264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 270)
        # Adding element type (line 270)
        # Getting the type of 'package_dir' (line 270)
        package_dir_20265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 37), 'package_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 37), tuple_20264, package_dir_20265)
        # Adding element type (line 270)
        int_20266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 50), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 37), tuple_20264, int_20266)
        
        # Getting the type of 'packages' (line 270)
        packages_20267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'packages')
        # Getting the type of 'package' (line 270)
        package_20268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 25), 'package')
        # Storing an element on a container (line 270)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 270, 16), packages_20267, (package_20268, tuple_20264))
        
        # Getting the type of 'init_py' (line 271)
        init_py_20269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 19), 'init_py')
        # Testing the type of an if condition (line 271)
        if_condition_20270 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 271, 16), init_py_20269)
        # Assigning a type to the variable 'if_condition_20270' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'if_condition_20270', if_condition_20270)
        # SSA begins for if statement (line 271)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 272)
        # Processing the call arguments (line 272)
        
        # Obtaining an instance of the builtin type 'tuple' (line 272)
        tuple_20273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 272)
        # Adding element type (line 272)
        # Getting the type of 'package' (line 272)
        package_20274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 36), 'package', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 36), tuple_20273, package_20274)
        # Adding element type (line 272)
        str_20275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 45), 'str', '__init__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 36), tuple_20273, str_20275)
        # Adding element type (line 272)
        # Getting the type of 'init_py' (line 272)
        init_py_20276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 57), 'init_py', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 272, 36), tuple_20273, init_py_20276)
        
        # Processing the call keyword arguments (line 272)
        kwargs_20277 = {}
        # Getting the type of 'modules' (line 272)
        modules_20271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 20), 'modules', False)
        # Obtaining the member 'append' of a type (line 272)
        append_20272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 20), modules_20271, 'append')
        # Calling append(args, kwargs) (line 272)
        append_call_result_20278 = invoke(stypy.reporting.localization.Localization(__file__, 272, 20), append_20272, *[tuple_20273], **kwargs_20277)
        
        # SSA join for if statement (line 271)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 268)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 277):
        
        # Assigning a Call to a Name (line 277):
        
        # Call to join(...): (line 277)
        # Processing the call arguments (line 277)
        # Getting the type of 'package_dir' (line 277)
        package_dir_20282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 39), 'package_dir', False)
        # Getting the type of 'module_base' (line 277)
        module_base_20283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 52), 'module_base', False)
        str_20284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 66), 'str', '.py')
        # Applying the binary operator '+' (line 277)
        result_add_20285 = python_operator(stypy.reporting.localization.Localization(__file__, 277, 52), '+', module_base_20283, str_20284)
        
        # Processing the call keyword arguments (line 277)
        kwargs_20286 = {}
        # Getting the type of 'os' (line 277)
        os_20279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 26), 'os', False)
        # Obtaining the member 'path' of a type (line 277)
        path_20280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 26), os_20279, 'path')
        # Obtaining the member 'join' of a type (line 277)
        join_20281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 26), path_20280, 'join')
        # Calling join(args, kwargs) (line 277)
        join_call_result_20287 = invoke(stypy.reporting.localization.Localization(__file__, 277, 26), join_20281, *[package_dir_20282, result_add_20285], **kwargs_20286)
        
        # Assigning a type to the variable 'module_file' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'module_file', join_call_result_20287)
        
        
        
        # Call to check_module(...): (line 278)
        # Processing the call arguments (line 278)
        # Getting the type of 'module' (line 278)
        module_20290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 37), 'module', False)
        # Getting the type of 'module_file' (line 278)
        module_file_20291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 45), 'module_file', False)
        # Processing the call keyword arguments (line 278)
        kwargs_20292 = {}
        # Getting the type of 'self' (line 278)
        self_20288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'self', False)
        # Obtaining the member 'check_module' of a type (line 278)
        check_module_20289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 19), self_20288, 'check_module')
        # Calling check_module(args, kwargs) (line 278)
        check_module_call_result_20293 = invoke(stypy.reporting.localization.Localization(__file__, 278, 19), check_module_20289, *[module_20290, module_file_20291], **kwargs_20292)
        
        # Applying the 'not' unary operator (line 278)
        result_not__20294 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 15), 'not', check_module_call_result_20293)
        
        # Testing the type of an if condition (line 278)
        if_condition_20295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 12), result_not__20294)
        # Assigning a type to the variable 'if_condition_20295' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'if_condition_20295', if_condition_20295)
        # SSA begins for if statement (line 278)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 278)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 281)
        # Processing the call arguments (line 281)
        
        # Obtaining an instance of the builtin type 'tuple' (line 281)
        tuple_20298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 281)
        # Adding element type (line 281)
        # Getting the type of 'package' (line 281)
        package_20299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 28), 'package', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 28), tuple_20298, package_20299)
        # Adding element type (line 281)
        # Getting the type of 'module_base' (line 281)
        module_base_20300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 37), 'module_base', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 28), tuple_20298, module_base_20300)
        # Adding element type (line 281)
        # Getting the type of 'module_file' (line 281)
        module_file_20301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 50), 'module_file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 28), tuple_20298, module_file_20301)
        
        # Processing the call keyword arguments (line 281)
        kwargs_20302 = {}
        # Getting the type of 'modules' (line 281)
        modules_20296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'modules', False)
        # Obtaining the member 'append' of a type (line 281)
        append_20297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 12), modules_20296, 'append')
        # Calling append(args, kwargs) (line 281)
        append_call_result_20303 = invoke(stypy.reporting.localization.Localization(__file__, 281, 12), append_20297, *[tuple_20298], **kwargs_20302)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'modules' (line 283)
        modules_20304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 15), 'modules')
        # Assigning a type to the variable 'stypy_return_type' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'stypy_return_type', modules_20304)
        
        # ################# End of 'find_modules(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_modules' in the type store
        # Getting the type of 'stypy_return_type' (line 233)
        stypy_return_type_20305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20305)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_modules'
        return stypy_return_type_20305


    @norecursion
    def find_all_modules(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find_all_modules'
        module_type_store = module_type_store.open_function_context('find_all_modules', 285, 4, False)
        # Assigning a type to the variable 'self' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.find_all_modules.__dict__.__setitem__('stypy_localization', localization)
        build_py.find_all_modules.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.find_all_modules.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.find_all_modules.__dict__.__setitem__('stypy_function_name', 'build_py.find_all_modules')
        build_py.find_all_modules.__dict__.__setitem__('stypy_param_names_list', [])
        build_py.find_all_modules.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.find_all_modules.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.find_all_modules.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.find_all_modules.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.find_all_modules.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.find_all_modules.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.find_all_modules', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_all_modules', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_all_modules(...)' code ##################

        str_20306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, (-1)), 'str', "Compute the list of all modules that will be built, whether\n        they are specified one-module-at-a-time ('self.py_modules') or\n        by whole packages ('self.packages').  Return a list of tuples\n        (package, module, module_file), just like 'find_modules()' and\n        'find_package_modules()' do.")
        
        # Assigning a List to a Name (line 291):
        
        # Assigning a List to a Name (line 291):
        
        # Obtaining an instance of the builtin type 'list' (line 291)
        list_20307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 291)
        
        # Assigning a type to the variable 'modules' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'modules', list_20307)
        
        # Getting the type of 'self' (line 292)
        self_20308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 11), 'self')
        # Obtaining the member 'py_modules' of a type (line 292)
        py_modules_20309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 11), self_20308, 'py_modules')
        # Testing the type of an if condition (line 292)
        if_condition_20310 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 8), py_modules_20309)
        # Assigning a type to the variable 'if_condition_20310' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'if_condition_20310', if_condition_20310)
        # SSA begins for if statement (line 292)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 293)
        # Processing the call arguments (line 293)
        
        # Call to find_modules(...): (line 293)
        # Processing the call keyword arguments (line 293)
        kwargs_20315 = {}
        # Getting the type of 'self' (line 293)
        self_20313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 27), 'self', False)
        # Obtaining the member 'find_modules' of a type (line 293)
        find_modules_20314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 27), self_20313, 'find_modules')
        # Calling find_modules(args, kwargs) (line 293)
        find_modules_call_result_20316 = invoke(stypy.reporting.localization.Localization(__file__, 293, 27), find_modules_20314, *[], **kwargs_20315)
        
        # Processing the call keyword arguments (line 293)
        kwargs_20317 = {}
        # Getting the type of 'modules' (line 293)
        modules_20311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'modules', False)
        # Obtaining the member 'extend' of a type (line 293)
        extend_20312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 12), modules_20311, 'extend')
        # Calling extend(args, kwargs) (line 293)
        extend_call_result_20318 = invoke(stypy.reporting.localization.Localization(__file__, 293, 12), extend_20312, *[find_modules_call_result_20316], **kwargs_20317)
        
        # SSA join for if statement (line 292)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 294)
        self_20319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 11), 'self')
        # Obtaining the member 'packages' of a type (line 294)
        packages_20320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 11), self_20319, 'packages')
        # Testing the type of an if condition (line 294)
        if_condition_20321 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 8), packages_20320)
        # Assigning a type to the variable 'if_condition_20321' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'if_condition_20321', if_condition_20321)
        # SSA begins for if statement (line 294)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 295)
        self_20322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 27), 'self')
        # Obtaining the member 'packages' of a type (line 295)
        packages_20323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 27), self_20322, 'packages')
        # Testing the type of a for loop iterable (line 295)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 295, 12), packages_20323)
        # Getting the type of the for loop variable (line 295)
        for_loop_var_20324 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 295, 12), packages_20323)
        # Assigning a type to the variable 'package' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'package', for_loop_var_20324)
        # SSA begins for a for statement (line 295)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 296):
        
        # Assigning a Call to a Name (line 296):
        
        # Call to get_package_dir(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 'package' (line 296)
        package_20327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 51), 'package', False)
        # Processing the call keyword arguments (line 296)
        kwargs_20328 = {}
        # Getting the type of 'self' (line 296)
        self_20325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 30), 'self', False)
        # Obtaining the member 'get_package_dir' of a type (line 296)
        get_package_dir_20326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 30), self_20325, 'get_package_dir')
        # Calling get_package_dir(args, kwargs) (line 296)
        get_package_dir_call_result_20329 = invoke(stypy.reporting.localization.Localization(__file__, 296, 30), get_package_dir_20326, *[package_20327], **kwargs_20328)
        
        # Assigning a type to the variable 'package_dir' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'package_dir', get_package_dir_call_result_20329)
        
        # Assigning a Call to a Name (line 297):
        
        # Assigning a Call to a Name (line 297):
        
        # Call to find_package_modules(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 'package' (line 297)
        package_20332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 46), 'package', False)
        # Getting the type of 'package_dir' (line 297)
        package_dir_20333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 55), 'package_dir', False)
        # Processing the call keyword arguments (line 297)
        kwargs_20334 = {}
        # Getting the type of 'self' (line 297)
        self_20330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 20), 'self', False)
        # Obtaining the member 'find_package_modules' of a type (line 297)
        find_package_modules_20331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 20), self_20330, 'find_package_modules')
        # Calling find_package_modules(args, kwargs) (line 297)
        find_package_modules_call_result_20335 = invoke(stypy.reporting.localization.Localization(__file__, 297, 20), find_package_modules_20331, *[package_20332, package_dir_20333], **kwargs_20334)
        
        # Assigning a type to the variable 'm' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'm', find_package_modules_call_result_20335)
        
        # Call to extend(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'm' (line 298)
        m_20338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 31), 'm', False)
        # Processing the call keyword arguments (line 298)
        kwargs_20339 = {}
        # Getting the type of 'modules' (line 298)
        modules_20336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'modules', False)
        # Obtaining the member 'extend' of a type (line 298)
        extend_20337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 16), modules_20336, 'extend')
        # Calling extend(args, kwargs) (line 298)
        extend_call_result_20340 = invoke(stypy.reporting.localization.Localization(__file__, 298, 16), extend_20337, *[m_20338], **kwargs_20339)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 294)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'modules' (line 299)
        modules_20341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 15), 'modules')
        # Assigning a type to the variable 'stypy_return_type' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'stypy_return_type', modules_20341)
        
        # ################# End of 'find_all_modules(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_all_modules' in the type store
        # Getting the type of 'stypy_return_type' (line 285)
        stypy_return_type_20342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20342)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_all_modules'
        return stypy_return_type_20342


    @norecursion
    def get_source_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_source_files'
        module_type_store = module_type_store.open_function_context('get_source_files', 301, 4, False)
        # Assigning a type to the variable 'self' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.get_source_files.__dict__.__setitem__('stypy_localization', localization)
        build_py.get_source_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.get_source_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.get_source_files.__dict__.__setitem__('stypy_function_name', 'build_py.get_source_files')
        build_py.get_source_files.__dict__.__setitem__('stypy_param_names_list', [])
        build_py.get_source_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.get_source_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.get_source_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.get_source_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.get_source_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.get_source_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.get_source_files', [], None, None, defaults, varargs, kwargs)

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

        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to find_all_modules(...): (line 302)
        # Processing the call keyword arguments (line 302)
        kwargs_20349 = {}
        # Getting the type of 'self' (line 302)
        self_20347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 41), 'self', False)
        # Obtaining the member 'find_all_modules' of a type (line 302)
        find_all_modules_20348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 41), self_20347, 'find_all_modules')
        # Calling find_all_modules(args, kwargs) (line 302)
        find_all_modules_call_result_20350 = invoke(stypy.reporting.localization.Localization(__file__, 302, 41), find_all_modules_20348, *[], **kwargs_20349)
        
        comprehension_20351 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 16), find_all_modules_call_result_20350)
        # Assigning a type to the variable 'module' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'module', comprehension_20351)
        
        # Obtaining the type of the subscript
        int_20343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 23), 'int')
        # Getting the type of 'module' (line 302)
        module_20344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'module')
        # Obtaining the member '__getitem__' of a type (line 302)
        getitem___20345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 16), module_20344, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 302)
        subscript_call_result_20346 = invoke(stypy.reporting.localization.Localization(__file__, 302, 16), getitem___20345, int_20343)
        
        list_20352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 302, 16), list_20352, subscript_call_result_20346)
        # Assigning a type to the variable 'stypy_return_type' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'stypy_return_type', list_20352)
        
        # ################# End of 'get_source_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_source_files' in the type store
        # Getting the type of 'stypy_return_type' (line 301)
        stypy_return_type_20353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20353)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_source_files'
        return stypy_return_type_20353


    @norecursion
    def get_module_outfile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_module_outfile'
        module_type_store = module_type_store.open_function_context('get_module_outfile', 304, 4, False)
        # Assigning a type to the variable 'self' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.get_module_outfile.__dict__.__setitem__('stypy_localization', localization)
        build_py.get_module_outfile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.get_module_outfile.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.get_module_outfile.__dict__.__setitem__('stypy_function_name', 'build_py.get_module_outfile')
        build_py.get_module_outfile.__dict__.__setitem__('stypy_param_names_list', ['build_dir', 'package', 'module'])
        build_py.get_module_outfile.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.get_module_outfile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.get_module_outfile.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.get_module_outfile.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.get_module_outfile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.get_module_outfile.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.get_module_outfile', ['build_dir', 'package', 'module'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_module_outfile', localization, ['build_dir', 'package', 'module'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_module_outfile(...)' code ##################

        
        # Assigning a BinOp to a Name (line 305):
        
        # Assigning a BinOp to a Name (line 305):
        
        # Obtaining an instance of the builtin type 'list' (line 305)
        list_20354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 305)
        # Adding element type (line 305)
        # Getting the type of 'build_dir' (line 305)
        build_dir_20355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 24), 'build_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 23), list_20354, build_dir_20355)
        
        
        # Call to list(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'package' (line 305)
        package_20357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 42), 'package', False)
        # Processing the call keyword arguments (line 305)
        kwargs_20358 = {}
        # Getting the type of 'list' (line 305)
        list_20356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 37), 'list', False)
        # Calling list(args, kwargs) (line 305)
        list_call_result_20359 = invoke(stypy.reporting.localization.Localization(__file__, 305, 37), list_20356, *[package_20357], **kwargs_20358)
        
        # Applying the binary operator '+' (line 305)
        result_add_20360 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 23), '+', list_20354, list_call_result_20359)
        
        
        # Obtaining an instance of the builtin type 'list' (line 305)
        list_20361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 305)
        # Adding element type (line 305)
        # Getting the type of 'module' (line 305)
        module_20362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 54), 'module')
        str_20363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 63), 'str', '.py')
        # Applying the binary operator '+' (line 305)
        result_add_20364 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 54), '+', module_20362, str_20363)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 305, 53), list_20361, result_add_20364)
        
        # Applying the binary operator '+' (line 305)
        result_add_20365 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 51), '+', result_add_20360, list_20361)
        
        # Assigning a type to the variable 'outfile_path' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'outfile_path', result_add_20365)
        
        # Call to join(...): (line 306)
        # Getting the type of 'outfile_path' (line 306)
        outfile_path_20369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 29), 'outfile_path', False)
        # Processing the call keyword arguments (line 306)
        kwargs_20370 = {}
        # Getting the type of 'os' (line 306)
        os_20366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 306)
        path_20367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 15), os_20366, 'path')
        # Obtaining the member 'join' of a type (line 306)
        join_20368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 15), path_20367, 'join')
        # Calling join(args, kwargs) (line 306)
        join_call_result_20371 = invoke(stypy.reporting.localization.Localization(__file__, 306, 15), join_20368, *[outfile_path_20369], **kwargs_20370)
        
        # Assigning a type to the variable 'stypy_return_type' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'stypy_return_type', join_call_result_20371)
        
        # ################# End of 'get_module_outfile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_module_outfile' in the type store
        # Getting the type of 'stypy_return_type' (line 304)
        stypy_return_type_20372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20372)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_module_outfile'
        return stypy_return_type_20372


    @norecursion
    def get_outputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_20373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 43), 'int')
        defaults = [int_20373]
        # Create a new context for function 'get_outputs'
        module_type_store = module_type_store.open_function_context('get_outputs', 308, 4, False)
        # Assigning a type to the variable 'self' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.get_outputs.__dict__.__setitem__('stypy_localization', localization)
        build_py.get_outputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.get_outputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.get_outputs.__dict__.__setitem__('stypy_function_name', 'build_py.get_outputs')
        build_py.get_outputs.__dict__.__setitem__('stypy_param_names_list', ['include_bytecode'])
        build_py.get_outputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.get_outputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.get_outputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.get_outputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.get_outputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.get_outputs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.get_outputs', ['include_bytecode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_outputs', localization, ['include_bytecode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_outputs(...)' code ##################

        
        # Assigning a Call to a Name (line 309):
        
        # Assigning a Call to a Name (line 309):
        
        # Call to find_all_modules(...): (line 309)
        # Processing the call keyword arguments (line 309)
        kwargs_20376 = {}
        # Getting the type of 'self' (line 309)
        self_20374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 18), 'self', False)
        # Obtaining the member 'find_all_modules' of a type (line 309)
        find_all_modules_20375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 18), self_20374, 'find_all_modules')
        # Calling find_all_modules(args, kwargs) (line 309)
        find_all_modules_call_result_20377 = invoke(stypy.reporting.localization.Localization(__file__, 309, 18), find_all_modules_20375, *[], **kwargs_20376)
        
        # Assigning a type to the variable 'modules' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'modules', find_all_modules_call_result_20377)
        
        # Assigning a List to a Name (line 310):
        
        # Assigning a List to a Name (line 310):
        
        # Obtaining an instance of the builtin type 'list' (line 310)
        list_20378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 310)
        
        # Assigning a type to the variable 'outputs' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'outputs', list_20378)
        
        # Getting the type of 'modules' (line 311)
        modules_20379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 46), 'modules')
        # Testing the type of a for loop iterable (line 311)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 311, 8), modules_20379)
        # Getting the type of the for loop variable (line 311)
        for_loop_var_20380 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 311, 8), modules_20379)
        # Assigning a type to the variable 'package' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'package', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 8), for_loop_var_20380))
        # Assigning a type to the variable 'module' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'module', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 8), for_loop_var_20380))
        # Assigning a type to the variable 'module_file' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'module_file', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 8), for_loop_var_20380))
        # SSA begins for a for statement (line 311)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 312):
        
        # Assigning a Call to a Name (line 312):
        
        # Call to split(...): (line 312)
        # Processing the call arguments (line 312)
        str_20383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 36), 'str', '.')
        # Processing the call keyword arguments (line 312)
        kwargs_20384 = {}
        # Getting the type of 'package' (line 312)
        package_20381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 22), 'package', False)
        # Obtaining the member 'split' of a type (line 312)
        split_20382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 22), package_20381, 'split')
        # Calling split(args, kwargs) (line 312)
        split_call_result_20385 = invoke(stypy.reporting.localization.Localization(__file__, 312, 22), split_20382, *[str_20383], **kwargs_20384)
        
        # Assigning a type to the variable 'package' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'package', split_call_result_20385)
        
        # Assigning a Call to a Name (line 313):
        
        # Assigning a Call to a Name (line 313):
        
        # Call to get_module_outfile(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'self' (line 313)
        self_20388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 47), 'self', False)
        # Obtaining the member 'build_lib' of a type (line 313)
        build_lib_20389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 47), self_20388, 'build_lib')
        # Getting the type of 'package' (line 313)
        package_20390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 63), 'package', False)
        # Getting the type of 'module' (line 313)
        module_20391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 72), 'module', False)
        # Processing the call keyword arguments (line 313)
        kwargs_20392 = {}
        # Getting the type of 'self' (line 313)
        self_20386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 23), 'self', False)
        # Obtaining the member 'get_module_outfile' of a type (line 313)
        get_module_outfile_20387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 23), self_20386, 'get_module_outfile')
        # Calling get_module_outfile(args, kwargs) (line 313)
        get_module_outfile_call_result_20393 = invoke(stypy.reporting.localization.Localization(__file__, 313, 23), get_module_outfile_20387, *[build_lib_20389, package_20390, module_20391], **kwargs_20392)
        
        # Assigning a type to the variable 'filename' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'filename', get_module_outfile_call_result_20393)
        
        # Call to append(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'filename' (line 314)
        filename_20396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 27), 'filename', False)
        # Processing the call keyword arguments (line 314)
        kwargs_20397 = {}
        # Getting the type of 'outputs' (line 314)
        outputs_20394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'outputs', False)
        # Obtaining the member 'append' of a type (line 314)
        append_20395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 12), outputs_20394, 'append')
        # Calling append(args, kwargs) (line 314)
        append_call_result_20398 = invoke(stypy.reporting.localization.Localization(__file__, 314, 12), append_20395, *[filename_20396], **kwargs_20397)
        
        
        # Getting the type of 'include_bytecode' (line 315)
        include_bytecode_20399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 15), 'include_bytecode')
        # Testing the type of an if condition (line 315)
        if_condition_20400 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 315, 12), include_bytecode_20399)
        # Assigning a type to the variable 'if_condition_20400' (line 315)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'if_condition_20400', if_condition_20400)
        # SSA begins for if statement (line 315)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 316)
        self_20401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 19), 'self')
        # Obtaining the member 'compile' of a type (line 316)
        compile_20402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 19), self_20401, 'compile')
        # Testing the type of an if condition (line 316)
        if_condition_20403 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 316, 16), compile_20402)
        # Assigning a type to the variable 'if_condition_20403' (line 316)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 16), 'if_condition_20403', if_condition_20403)
        # SSA begins for if statement (line 316)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 317)
        # Processing the call arguments (line 317)
        # Getting the type of 'filename' (line 317)
        filename_20406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 35), 'filename', False)
        str_20407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 46), 'str', 'c')
        # Applying the binary operator '+' (line 317)
        result_add_20408 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 35), '+', filename_20406, str_20407)
        
        # Processing the call keyword arguments (line 317)
        kwargs_20409 = {}
        # Getting the type of 'outputs' (line 317)
        outputs_20404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 20), 'outputs', False)
        # Obtaining the member 'append' of a type (line 317)
        append_20405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 20), outputs_20404, 'append')
        # Calling append(args, kwargs) (line 317)
        append_call_result_20410 = invoke(stypy.reporting.localization.Localization(__file__, 317, 20), append_20405, *[result_add_20408], **kwargs_20409)
        
        # SSA join for if statement (line 316)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 318)
        self_20411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 19), 'self')
        # Obtaining the member 'optimize' of a type (line 318)
        optimize_20412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 19), self_20411, 'optimize')
        int_20413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 35), 'int')
        # Applying the binary operator '>' (line 318)
        result_gt_20414 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 19), '>', optimize_20412, int_20413)
        
        # Testing the type of an if condition (line 318)
        if_condition_20415 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 16), result_gt_20414)
        # Assigning a type to the variable 'if_condition_20415' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 16), 'if_condition_20415', if_condition_20415)
        # SSA begins for if statement (line 318)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 319)
        # Processing the call arguments (line 319)
        # Getting the type of 'filename' (line 319)
        filename_20418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 35), 'filename', False)
        str_20419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 46), 'str', 'o')
        # Applying the binary operator '+' (line 319)
        result_add_20420 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 35), '+', filename_20418, str_20419)
        
        # Processing the call keyword arguments (line 319)
        kwargs_20421 = {}
        # Getting the type of 'outputs' (line 319)
        outputs_20416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 20), 'outputs', False)
        # Obtaining the member 'append' of a type (line 319)
        append_20417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 20), outputs_20416, 'append')
        # Calling append(args, kwargs) (line 319)
        append_call_result_20422 = invoke(stypy.reporting.localization.Localization(__file__, 319, 20), append_20417, *[result_add_20420], **kwargs_20421)
        
        # SSA join for if statement (line 318)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 315)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'outputs' (line 321)
        outputs_20423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'outputs')
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 323)
        self_20431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 58), 'self')
        # Obtaining the member 'data_files' of a type (line 323)
        data_files_20432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 58), self_20431, 'data_files')
        comprehension_20433 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 12), data_files_20432)
        # Assigning a type to the variable 'package' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'package', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 12), comprehension_20433))
        # Assigning a type to the variable 'src_dir' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'src_dir', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 12), comprehension_20433))
        # Assigning a type to the variable 'build_dir' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'build_dir', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 12), comprehension_20433))
        # Assigning a type to the variable 'filenames' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'filenames', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 12), comprehension_20433))
        # Calculating comprehension expression
        # Getting the type of 'filenames' (line 324)
        filenames_20434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 28), 'filenames')
        comprehension_20435 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 12), filenames_20434)
        # Assigning a type to the variable 'filename' (line 322)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'filename', comprehension_20435)
        
        # Call to join(...): (line 322)
        # Processing the call arguments (line 322)
        # Getting the type of 'build_dir' (line 322)
        build_dir_20427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 25), 'build_dir', False)
        # Getting the type of 'filename' (line 322)
        filename_20428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 36), 'filename', False)
        # Processing the call keyword arguments (line 322)
        kwargs_20429 = {}
        # Getting the type of 'os' (line 322)
        os_20424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'os', False)
        # Obtaining the member 'path' of a type (line 322)
        path_20425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 12), os_20424, 'path')
        # Obtaining the member 'join' of a type (line 322)
        join_20426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 12), path_20425, 'join')
        # Calling join(args, kwargs) (line 322)
        join_call_result_20430 = invoke(stypy.reporting.localization.Localization(__file__, 322, 12), join_20426, *[build_dir_20427, filename_20428], **kwargs_20429)
        
        list_20436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 12), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 322, 12), list_20436, join_call_result_20430)
        # Applying the binary operator '+=' (line 321)
        result_iadd_20437 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 8), '+=', outputs_20423, list_20436)
        # Assigning a type to the variable 'outputs' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'outputs', result_iadd_20437)
        
        # Getting the type of 'outputs' (line 327)
        outputs_20438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 15), 'outputs')
        # Assigning a type to the variable 'stypy_return_type' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'stypy_return_type', outputs_20438)
        
        # ################# End of 'get_outputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_outputs' in the type store
        # Getting the type of 'stypy_return_type' (line 308)
        stypy_return_type_20439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20439)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_outputs'
        return stypy_return_type_20439


    @norecursion
    def build_module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_module'
        module_type_store = module_type_store.open_function_context('build_module', 329, 4, False)
        # Assigning a type to the variable 'self' (line 330)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.build_module.__dict__.__setitem__('stypy_localization', localization)
        build_py.build_module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.build_module.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.build_module.__dict__.__setitem__('stypy_function_name', 'build_py.build_module')
        build_py.build_module.__dict__.__setitem__('stypy_param_names_list', ['module', 'module_file', 'package'])
        build_py.build_module.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.build_module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.build_module.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.build_module.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.build_module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.build_module.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.build_module', ['module', 'module_file', 'package'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_module', localization, ['module', 'module_file', 'package'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_module(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 330)
        # Getting the type of 'str' (line 330)
        str_20440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 31), 'str')
        # Getting the type of 'package' (line 330)
        package_20441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 22), 'package')
        
        (may_be_20442, more_types_in_union_20443) = may_be_subtype(str_20440, package_20441)

        if may_be_20442:

            if more_types_in_union_20443:
                # Runtime conditional SSA (line 330)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'package' (line 330)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'package', remove_not_subtype_from_union(package_20441, str))
            
            # Assigning a Call to a Name (line 331):
            
            # Assigning a Call to a Name (line 331):
            
            # Call to split(...): (line 331)
            # Processing the call arguments (line 331)
            str_20446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 36), 'str', '.')
            # Processing the call keyword arguments (line 331)
            kwargs_20447 = {}
            # Getting the type of 'package' (line 331)
            package_20444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 22), 'package', False)
            # Obtaining the member 'split' of a type (line 331)
            split_20445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 22), package_20444, 'split')
            # Calling split(args, kwargs) (line 331)
            split_call_result_20448 = invoke(stypy.reporting.localization.Localization(__file__, 331, 22), split_20445, *[str_20446], **kwargs_20447)
            
            # Assigning a type to the variable 'package' (line 331)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'package', split_call_result_20448)

            if more_types_in_union_20443:
                # Runtime conditional SSA for else branch (line 330)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_20442) or more_types_in_union_20443):
            # Assigning a type to the variable 'package' (line 330)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'package', remove_subtype_from_union(package_20441, str))
            
            
            
            # Call to isinstance(...): (line 332)
            # Processing the call arguments (line 332)
            # Getting the type of 'package' (line 332)
            package_20450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 28), 'package', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 332)
            tuple_20451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 38), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 332)
            # Adding element type (line 332)
            # Getting the type of 'list' (line 332)
            list_20452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 38), 'list', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 38), tuple_20451, list_20452)
            # Adding element type (line 332)
            # Getting the type of 'tuple' (line 332)
            tuple_20453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 44), 'tuple', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 332, 38), tuple_20451, tuple_20453)
            
            # Processing the call keyword arguments (line 332)
            kwargs_20454 = {}
            # Getting the type of 'isinstance' (line 332)
            isinstance_20449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 17), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 332)
            isinstance_call_result_20455 = invoke(stypy.reporting.localization.Localization(__file__, 332, 17), isinstance_20449, *[package_20450, tuple_20451], **kwargs_20454)
            
            # Applying the 'not' unary operator (line 332)
            result_not__20456 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 13), 'not', isinstance_call_result_20455)
            
            # Testing the type of an if condition (line 332)
            if_condition_20457 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 13), result_not__20456)
            # Assigning a type to the variable 'if_condition_20457' (line 332)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 13), 'if_condition_20457', if_condition_20457)
            # SSA begins for if statement (line 332)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 333)
            # Processing the call arguments (line 333)
            str_20459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 18), 'str', "'package' must be a string (dot-separated), list, or tuple")
            # Processing the call keyword arguments (line 333)
            kwargs_20460 = {}
            # Getting the type of 'TypeError' (line 333)
            TypeError_20458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 18), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 333)
            TypeError_call_result_20461 = invoke(stypy.reporting.localization.Localization(__file__, 333, 18), TypeError_20458, *[str_20459], **kwargs_20460)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 333, 12), TypeError_call_result_20461, 'raise parameter', BaseException)
            # SSA join for if statement (line 332)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_20442 and more_types_in_union_20443):
                # SSA join for if statement (line 330)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 339):
        
        # Assigning a Call to a Name (line 339):
        
        # Call to get_module_outfile(...): (line 339)
        # Processing the call arguments (line 339)
        # Getting the type of 'self' (line 339)
        self_20464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 42), 'self', False)
        # Obtaining the member 'build_lib' of a type (line 339)
        build_lib_20465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 42), self_20464, 'build_lib')
        # Getting the type of 'package' (line 339)
        package_20466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 58), 'package', False)
        # Getting the type of 'module' (line 339)
        module_20467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 67), 'module', False)
        # Processing the call keyword arguments (line 339)
        kwargs_20468 = {}
        # Getting the type of 'self' (line 339)
        self_20462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 18), 'self', False)
        # Obtaining the member 'get_module_outfile' of a type (line 339)
        get_module_outfile_20463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 18), self_20462, 'get_module_outfile')
        # Calling get_module_outfile(args, kwargs) (line 339)
        get_module_outfile_call_result_20469 = invoke(stypy.reporting.localization.Localization(__file__, 339, 18), get_module_outfile_20463, *[build_lib_20465, package_20466, module_20467], **kwargs_20468)
        
        # Assigning a type to the variable 'outfile' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'outfile', get_module_outfile_call_result_20469)
        
        # Assigning a Call to a Name (line 340):
        
        # Assigning a Call to a Name (line 340):
        
        # Call to dirname(...): (line 340)
        # Processing the call arguments (line 340)
        # Getting the type of 'outfile' (line 340)
        outfile_20473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 30), 'outfile', False)
        # Processing the call keyword arguments (line 340)
        kwargs_20474 = {}
        # Getting the type of 'os' (line 340)
        os_20470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 14), 'os', False)
        # Obtaining the member 'path' of a type (line 340)
        path_20471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 14), os_20470, 'path')
        # Obtaining the member 'dirname' of a type (line 340)
        dirname_20472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 14), path_20471, 'dirname')
        # Calling dirname(args, kwargs) (line 340)
        dirname_call_result_20475 = invoke(stypy.reporting.localization.Localization(__file__, 340, 14), dirname_20472, *[outfile_20473], **kwargs_20474)
        
        # Assigning a type to the variable 'dir' (line 340)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'dir', dirname_call_result_20475)
        
        # Call to mkpath(...): (line 341)
        # Processing the call arguments (line 341)
        # Getting the type of 'dir' (line 341)
        dir_20478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 20), 'dir', False)
        # Processing the call keyword arguments (line 341)
        kwargs_20479 = {}
        # Getting the type of 'self' (line 341)
        self_20476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 341)
        mkpath_20477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), self_20476, 'mkpath')
        # Calling mkpath(args, kwargs) (line 341)
        mkpath_call_result_20480 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), mkpath_20477, *[dir_20478], **kwargs_20479)
        
        
        # Call to copy_file(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 'module_file' (line 342)
        module_file_20483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 30), 'module_file', False)
        # Getting the type of 'outfile' (line 342)
        outfile_20484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 43), 'outfile', False)
        # Processing the call keyword arguments (line 342)
        int_20485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 66), 'int')
        keyword_20486 = int_20485
        kwargs_20487 = {'preserve_mode': keyword_20486}
        # Getting the type of 'self' (line 342)
        self_20481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 15), 'self', False)
        # Obtaining the member 'copy_file' of a type (line 342)
        copy_file_20482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 15), self_20481, 'copy_file')
        # Calling copy_file(args, kwargs) (line 342)
        copy_file_call_result_20488 = invoke(stypy.reporting.localization.Localization(__file__, 342, 15), copy_file_20482, *[module_file_20483, outfile_20484], **kwargs_20487)
        
        # Assigning a type to the variable 'stypy_return_type' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'stypy_return_type', copy_file_call_result_20488)
        
        # ################# End of 'build_module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_module' in the type store
        # Getting the type of 'stypy_return_type' (line 329)
        stypy_return_type_20489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20489)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_module'
        return stypy_return_type_20489


    @norecursion
    def build_modules(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_modules'
        module_type_store = module_type_store.open_function_context('build_modules', 344, 4, False)
        # Assigning a type to the variable 'self' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.build_modules.__dict__.__setitem__('stypy_localization', localization)
        build_py.build_modules.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.build_modules.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.build_modules.__dict__.__setitem__('stypy_function_name', 'build_py.build_modules')
        build_py.build_modules.__dict__.__setitem__('stypy_param_names_list', [])
        build_py.build_modules.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.build_modules.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.build_modules.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.build_modules.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.build_modules.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.build_modules.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.build_modules', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_modules', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_modules(...)' code ##################

        
        # Assigning a Call to a Name (line 345):
        
        # Assigning a Call to a Name (line 345):
        
        # Call to find_modules(...): (line 345)
        # Processing the call keyword arguments (line 345)
        kwargs_20492 = {}
        # Getting the type of 'self' (line 345)
        self_20490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 18), 'self', False)
        # Obtaining the member 'find_modules' of a type (line 345)
        find_modules_20491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 18), self_20490, 'find_modules')
        # Calling find_modules(args, kwargs) (line 345)
        find_modules_call_result_20493 = invoke(stypy.reporting.localization.Localization(__file__, 345, 18), find_modules_20491, *[], **kwargs_20492)
        
        # Assigning a type to the variable 'modules' (line 345)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'modules', find_modules_call_result_20493)
        
        # Getting the type of 'modules' (line 346)
        modules_20494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 46), 'modules')
        # Testing the type of a for loop iterable (line 346)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 346, 8), modules_20494)
        # Getting the type of the for loop variable (line 346)
        for_loop_var_20495 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 346, 8), modules_20494)
        # Assigning a type to the variable 'package' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'package', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 8), for_loop_var_20495))
        # Assigning a type to the variable 'module' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'module', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 8), for_loop_var_20495))
        # Assigning a type to the variable 'module_file' (line 346)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'module_file', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 346, 8), for_loop_var_20495))
        # SSA begins for a for statement (line 346)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to build_module(...): (line 352)
        # Processing the call arguments (line 352)
        # Getting the type of 'module' (line 352)
        module_20498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 30), 'module', False)
        # Getting the type of 'module_file' (line 352)
        module_file_20499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 38), 'module_file', False)
        # Getting the type of 'package' (line 352)
        package_20500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 51), 'package', False)
        # Processing the call keyword arguments (line 352)
        kwargs_20501 = {}
        # Getting the type of 'self' (line 352)
        self_20496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'self', False)
        # Obtaining the member 'build_module' of a type (line 352)
        build_module_20497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 12), self_20496, 'build_module')
        # Calling build_module(args, kwargs) (line 352)
        build_module_call_result_20502 = invoke(stypy.reporting.localization.Localization(__file__, 352, 12), build_module_20497, *[module_20498, module_file_20499, package_20500], **kwargs_20501)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'build_modules(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_modules' in the type store
        # Getting the type of 'stypy_return_type' (line 344)
        stypy_return_type_20503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20503)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_modules'
        return stypy_return_type_20503


    @norecursion
    def build_packages(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_packages'
        module_type_store = module_type_store.open_function_context('build_packages', 354, 4, False)
        # Assigning a type to the variable 'self' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.build_packages.__dict__.__setitem__('stypy_localization', localization)
        build_py.build_packages.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.build_packages.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.build_packages.__dict__.__setitem__('stypy_function_name', 'build_py.build_packages')
        build_py.build_packages.__dict__.__setitem__('stypy_param_names_list', [])
        build_py.build_packages.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.build_packages.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.build_packages.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.build_packages.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.build_packages.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.build_packages.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.build_packages', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_packages', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_packages(...)' code ##################

        
        # Getting the type of 'self' (line 355)
        self_20504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 23), 'self')
        # Obtaining the member 'packages' of a type (line 355)
        packages_20505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 23), self_20504, 'packages')
        # Testing the type of a for loop iterable (line 355)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 355, 8), packages_20505)
        # Getting the type of the for loop variable (line 355)
        for_loop_var_20506 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 355, 8), packages_20505)
        # Assigning a type to the variable 'package' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'package', for_loop_var_20506)
        # SSA begins for a for statement (line 355)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 366):
        
        # Assigning a Call to a Name (line 366):
        
        # Call to get_package_dir(...): (line 366)
        # Processing the call arguments (line 366)
        # Getting the type of 'package' (line 366)
        package_20509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 47), 'package', False)
        # Processing the call keyword arguments (line 366)
        kwargs_20510 = {}
        # Getting the type of 'self' (line 366)
        self_20507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 26), 'self', False)
        # Obtaining the member 'get_package_dir' of a type (line 366)
        get_package_dir_20508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 26), self_20507, 'get_package_dir')
        # Calling get_package_dir(args, kwargs) (line 366)
        get_package_dir_call_result_20511 = invoke(stypy.reporting.localization.Localization(__file__, 366, 26), get_package_dir_20508, *[package_20509], **kwargs_20510)
        
        # Assigning a type to the variable 'package_dir' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'package_dir', get_package_dir_call_result_20511)
        
        # Assigning a Call to a Name (line 367):
        
        # Assigning a Call to a Name (line 367):
        
        # Call to find_package_modules(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'package' (line 367)
        package_20514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 48), 'package', False)
        # Getting the type of 'package_dir' (line 367)
        package_dir_20515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 57), 'package_dir', False)
        # Processing the call keyword arguments (line 367)
        kwargs_20516 = {}
        # Getting the type of 'self' (line 367)
        self_20512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 22), 'self', False)
        # Obtaining the member 'find_package_modules' of a type (line 367)
        find_package_modules_20513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 22), self_20512, 'find_package_modules')
        # Calling find_package_modules(args, kwargs) (line 367)
        find_package_modules_call_result_20517 = invoke(stypy.reporting.localization.Localization(__file__, 367, 22), find_package_modules_20513, *[package_20514, package_dir_20515], **kwargs_20516)
        
        # Assigning a type to the variable 'modules' (line 367)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'modules', find_package_modules_call_result_20517)
        
        # Getting the type of 'modules' (line 371)
        modules_20518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 51), 'modules')
        # Testing the type of a for loop iterable (line 371)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 371, 12), modules_20518)
        # Getting the type of the for loop variable (line 371)
        for_loop_var_20519 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 371, 12), modules_20518)
        # Assigning a type to the variable 'package_' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'package_', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 12), for_loop_var_20519))
        # Assigning a type to the variable 'module' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'module', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 12), for_loop_var_20519))
        # Assigning a type to the variable 'module_file' (line 371)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 12), 'module_file', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 371, 12), for_loop_var_20519))
        # SSA begins for a for statement (line 371)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Evaluating assert statement condition
        
        # Getting the type of 'package' (line 372)
        package_20520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 23), 'package')
        # Getting the type of 'package_' (line 372)
        package__20521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 34), 'package_')
        # Applying the binary operator '==' (line 372)
        result_eq_20522 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 23), '==', package_20520, package__20521)
        
        
        # Call to build_module(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 'module' (line 373)
        module_20525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 34), 'module', False)
        # Getting the type of 'module_file' (line 373)
        module_file_20526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 42), 'module_file', False)
        # Getting the type of 'package' (line 373)
        package_20527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 55), 'package', False)
        # Processing the call keyword arguments (line 373)
        kwargs_20528 = {}
        # Getting the type of 'self' (line 373)
        self_20523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 16), 'self', False)
        # Obtaining the member 'build_module' of a type (line 373)
        build_module_20524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 16), self_20523, 'build_module')
        # Calling build_module(args, kwargs) (line 373)
        build_module_call_result_20529 = invoke(stypy.reporting.localization.Localization(__file__, 373, 16), build_module_20524, *[module_20525, module_file_20526, package_20527], **kwargs_20528)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'build_packages(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_packages' in the type store
        # Getting the type of 'stypy_return_type' (line 354)
        stypy_return_type_20530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20530)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_packages'
        return stypy_return_type_20530


    @norecursion
    def byte_compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'byte_compile'
        module_type_store = module_type_store.open_function_context('byte_compile', 375, 4, False)
        # Assigning a type to the variable 'self' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.byte_compile.__dict__.__setitem__('stypy_localization', localization)
        build_py.byte_compile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.byte_compile.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.byte_compile.__dict__.__setitem__('stypy_function_name', 'build_py.byte_compile')
        build_py.byte_compile.__dict__.__setitem__('stypy_param_names_list', ['files'])
        build_py.byte_compile.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.byte_compile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.byte_compile.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.byte_compile.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.byte_compile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.byte_compile.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.byte_compile', ['files'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'byte_compile', localization, ['files'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'byte_compile(...)' code ##################

        
        # Getting the type of 'sys' (line 376)
        sys_20531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 11), 'sys')
        # Obtaining the member 'dont_write_bytecode' of a type (line 376)
        dont_write_bytecode_20532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 11), sys_20531, 'dont_write_bytecode')
        # Testing the type of an if condition (line 376)
        if_condition_20533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 376, 8), dont_write_bytecode_20532)
        # Assigning a type to the variable 'if_condition_20533' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 8), 'if_condition_20533', if_condition_20533)
        # SSA begins for if statement (line 376)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 377)
        # Processing the call arguments (line 377)
        str_20536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 22), 'str', 'byte-compiling is disabled, skipping.')
        # Processing the call keyword arguments (line 377)
        kwargs_20537 = {}
        # Getting the type of 'self' (line 377)
        self_20534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 12), 'self', False)
        # Obtaining the member 'warn' of a type (line 377)
        warn_20535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 12), self_20534, 'warn')
        # Calling warn(args, kwargs) (line 377)
        warn_call_result_20538 = invoke(stypy.reporting.localization.Localization(__file__, 377, 12), warn_20535, *[str_20536], **kwargs_20537)
        
        # Assigning a type to the variable 'stypy_return_type' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 376)
        module_type_store = module_type_store.join_ssa_context()
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 380, 8))
        
        # 'from distutils.util import byte_compile' statement (line 380)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
        import_20539 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 380, 8), 'distutils.util')

        if (type(import_20539) is not StypyTypeError):

            if (import_20539 != 'pyd_module'):
                __import__(import_20539)
                sys_modules_20540 = sys.modules[import_20539]
                import_from_module(stypy.reporting.localization.Localization(__file__, 380, 8), 'distutils.util', sys_modules_20540.module_type_store, module_type_store, ['byte_compile'])
                nest_module(stypy.reporting.localization.Localization(__file__, 380, 8), __file__, sys_modules_20540, sys_modules_20540.module_type_store, module_type_store)
            else:
                from distutils.util import byte_compile

                import_from_module(stypy.reporting.localization.Localization(__file__, 380, 8), 'distutils.util', None, module_type_store, ['byte_compile'], [byte_compile])

        else:
            # Assigning a type to the variable 'distutils.util' (line 380)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'distutils.util', import_20539)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
        
        
        # Assigning a Attribute to a Name (line 381):
        
        # Assigning a Attribute to a Name (line 381):
        # Getting the type of 'self' (line 381)
        self_20541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 17), 'self')
        # Obtaining the member 'build_lib' of a type (line 381)
        build_lib_20542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 17), self_20541, 'build_lib')
        # Assigning a type to the variable 'prefix' (line 381)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'prefix', build_lib_20542)
        
        
        
        # Obtaining the type of the subscript
        int_20543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 18), 'int')
        # Getting the type of 'prefix' (line 382)
        prefix_20544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 11), 'prefix')
        # Obtaining the member '__getitem__' of a type (line 382)
        getitem___20545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 11), prefix_20544, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 382)
        subscript_call_result_20546 = invoke(stypy.reporting.localization.Localization(__file__, 382, 11), getitem___20545, int_20543)
        
        # Getting the type of 'os' (line 382)
        os_20547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 25), 'os')
        # Obtaining the member 'sep' of a type (line 382)
        sep_20548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 25), os_20547, 'sep')
        # Applying the binary operator '!=' (line 382)
        result_ne_20549 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 11), '!=', subscript_call_result_20546, sep_20548)
        
        # Testing the type of an if condition (line 382)
        if_condition_20550 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 382, 8), result_ne_20549)
        # Assigning a type to the variable 'if_condition_20550' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'if_condition_20550', if_condition_20550)
        # SSA begins for if statement (line 382)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 383):
        
        # Assigning a BinOp to a Name (line 383):
        # Getting the type of 'prefix' (line 383)
        prefix_20551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 21), 'prefix')
        # Getting the type of 'os' (line 383)
        os_20552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 30), 'os')
        # Obtaining the member 'sep' of a type (line 383)
        sep_20553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 30), os_20552, 'sep')
        # Applying the binary operator '+' (line 383)
        result_add_20554 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 21), '+', prefix_20551, sep_20553)
        
        # Assigning a type to the variable 'prefix' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 12), 'prefix', result_add_20554)
        # SSA join for if statement (line 382)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 389)
        self_20555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 11), 'self')
        # Obtaining the member 'compile' of a type (line 389)
        compile_20556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 11), self_20555, 'compile')
        # Testing the type of an if condition (line 389)
        if_condition_20557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 8), compile_20556)
        # Assigning a type to the variable 'if_condition_20557' (line 389)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'if_condition_20557', if_condition_20557)
        # SSA begins for if statement (line 389)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to byte_compile(...): (line 390)
        # Processing the call arguments (line 390)
        # Getting the type of 'files' (line 390)
        files_20559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 25), 'files', False)
        # Processing the call keyword arguments (line 390)
        int_20560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 41), 'int')
        keyword_20561 = int_20560
        # Getting the type of 'self' (line 391)
        self_20562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 31), 'self', False)
        # Obtaining the member 'force' of a type (line 391)
        force_20563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 31), self_20562, 'force')
        keyword_20564 = force_20563
        # Getting the type of 'prefix' (line 391)
        prefix_20565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 50), 'prefix', False)
        keyword_20566 = prefix_20565
        # Getting the type of 'self' (line 391)
        self_20567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 66), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 391)
        dry_run_20568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 66), self_20567, 'dry_run')
        keyword_20569 = dry_run_20568
        kwargs_20570 = {'prefix': keyword_20566, 'force': keyword_20564, 'optimize': keyword_20561, 'dry_run': keyword_20569}
        # Getting the type of 'byte_compile' (line 390)
        byte_compile_20558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'byte_compile', False)
        # Calling byte_compile(args, kwargs) (line 390)
        byte_compile_call_result_20571 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), byte_compile_20558, *[files_20559], **kwargs_20570)
        
        # SSA join for if statement (line 389)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 392)
        self_20572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 11), 'self')
        # Obtaining the member 'optimize' of a type (line 392)
        optimize_20573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 11), self_20572, 'optimize')
        int_20574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 27), 'int')
        # Applying the binary operator '>' (line 392)
        result_gt_20575 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 11), '>', optimize_20573, int_20574)
        
        # Testing the type of an if condition (line 392)
        if_condition_20576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 392, 8), result_gt_20575)
        # Assigning a type to the variable 'if_condition_20576' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'if_condition_20576', if_condition_20576)
        # SSA begins for if statement (line 392)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to byte_compile(...): (line 393)
        # Processing the call arguments (line 393)
        # Getting the type of 'files' (line 393)
        files_20578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 25), 'files', False)
        # Processing the call keyword arguments (line 393)
        # Getting the type of 'self' (line 393)
        self_20579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 41), 'self', False)
        # Obtaining the member 'optimize' of a type (line 393)
        optimize_20580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 41), self_20579, 'optimize')
        keyword_20581 = optimize_20580
        # Getting the type of 'self' (line 394)
        self_20582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 31), 'self', False)
        # Obtaining the member 'force' of a type (line 394)
        force_20583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 31), self_20582, 'force')
        keyword_20584 = force_20583
        # Getting the type of 'prefix' (line 394)
        prefix_20585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 50), 'prefix', False)
        keyword_20586 = prefix_20585
        # Getting the type of 'self' (line 394)
        self_20587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 66), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 394)
        dry_run_20588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 66), self_20587, 'dry_run')
        keyword_20589 = dry_run_20588
        kwargs_20590 = {'prefix': keyword_20586, 'force': keyword_20584, 'optimize': keyword_20581, 'dry_run': keyword_20589}
        # Getting the type of 'byte_compile' (line 393)
        byte_compile_20577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 12), 'byte_compile', False)
        # Calling byte_compile(args, kwargs) (line 393)
        byte_compile_call_result_20591 = invoke(stypy.reporting.localization.Localization(__file__, 393, 12), byte_compile_20577, *[files_20578], **kwargs_20590)
        
        # SSA join for if statement (line 392)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'byte_compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'byte_compile' in the type store
        # Getting the type of 'stypy_return_type' (line 375)
        stypy_return_type_20592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20592)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'byte_compile'
        return stypy_return_type_20592


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 16, 0, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'build_py' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'build_py', build_py)

# Assigning a Str to a Name (line 18):
str_20593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 18), 'str', '"build" pure Python modules (copy to build directory)')
# Getting the type of 'build_py'
build_py_20594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_py')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_py_20594, 'description', str_20593)

# Assigning a List to a Name (line 20):

# Obtaining an instance of the builtin type 'list' (line 20)
list_20595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 21)
tuple_20596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 21)
# Adding element type (line 21)
str_20597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 9), 'str', 'build-lib=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 9), tuple_20596, str_20597)
# Adding element type (line 21)
str_20598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 9), tuple_20596, str_20598)
# Adding element type (line 21)
str_20599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'str', 'directory to "build" (copy) to')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 9), tuple_20596, str_20599)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 19), list_20595, tuple_20596)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 22)
tuple_20600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 22)
# Adding element type (line 22)
str_20601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'str', 'compile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_20600, str_20601)
# Adding element type (line 22)
str_20602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 20), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_20600, str_20602)
# Adding element type (line 22)
str_20603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'str', 'compile .py to .pyc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_20600, str_20603)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 19), list_20595, tuple_20600)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_20604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
str_20605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'str', 'no-compile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_20604, str_20605)
# Adding element type (line 23)
# Getting the type of 'None' (line 23)
None_20606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_20604, None_20606)
# Adding element type (line 23)
str_20607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 29), 'str', "don't compile .py files [default]")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_20604, str_20607)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 19), list_20595, tuple_20604)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_20608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
str_20609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'str', 'optimize=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_20608, str_20609)
# Adding element type (line 24)
str_20610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 22), 'str', 'O')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_20608, str_20610)
# Adding element type (line 24)
str_20611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'str', 'also compile with optimization: -O1 for "python -O", -O2 for "python -OO", and -O0 to disable [default: -O0]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_20608, str_20611)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 19), list_20595, tuple_20608)
# Adding element type (line 20)

# Obtaining an instance of the builtin type 'tuple' (line 27)
tuple_20612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 27)
# Adding element type (line 27)
str_20613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 9), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 9), tuple_20612, str_20613)
# Adding element type (line 27)
str_20614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 18), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 9), tuple_20612, str_20614)
# Adding element type (line 27)
str_20615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'str', 'forcibly build everything (ignore file timestamps)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 9), tuple_20612, str_20615)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 19), list_20595, tuple_20612)

# Getting the type of 'build_py'
build_py_20616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_py')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_py_20616, 'user_options', list_20595)

# Assigning a List to a Name (line 30):

# Obtaining an instance of the builtin type 'list' (line 30)
list_20617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 30)
# Adding element type (line 30)
str_20618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 23), 'str', 'compile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 22), list_20617, str_20618)
# Adding element type (line 30)
str_20619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 34), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 22), list_20617, str_20619)

# Getting the type of 'build_py'
build_py_20620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_py')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_py_20620, 'boolean_options', list_20617)

# Assigning a Dict to a Name (line 31):

# Obtaining an instance of the builtin type 'dict' (line 31)
dict_20621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 31)
# Adding element type (key, value) (line 31)
str_20622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'str', 'no-compile')
str_20623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 35), 'str', 'compile')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), dict_20621, (str_20622, str_20623))

# Getting the type of 'build_py'
build_py_20624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_py')
# Setting the type of the member 'negative_opt' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_py_20624, 'negative_opt', dict_20621)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
