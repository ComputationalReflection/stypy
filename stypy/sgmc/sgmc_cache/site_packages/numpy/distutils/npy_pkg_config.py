
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import sys
4: import re
5: import os
6: 
7: if sys.version_info[0] < 3:
8:     from ConfigParser import SafeConfigParser, NoOptionError
9: else:
10:     from configparser import ConfigParser, SafeConfigParser, NoOptionError
11: 
12: __all__ = ['FormatError', 'PkgNotFound', 'LibraryInfo', 'VariableSet',
13:         'read_config', 'parse_flags']
14: 
15: _VAR = re.compile('\$\{([a-zA-Z0-9_-]+)\}')
16: 
17: class FormatError(IOError):
18:     '''
19:     Exception thrown when there is a problem parsing a configuration file.
20: 
21:     '''
22:     def __init__(self, msg):
23:         self.msg = msg
24: 
25:     def __str__(self):
26:         return self.msg
27: 
28: class PkgNotFound(IOError):
29:     '''Exception raised when a package can not be located.'''
30:     def __init__(self, msg):
31:         self.msg = msg
32: 
33:     def __str__(self):
34:         return self.msg
35: 
36: def parse_flags(line):
37:     '''
38:     Parse a line from a config file containing compile flags.
39: 
40:     Parameters
41:     ----------
42:     line : str
43:         A single line containing one or more compile flags.
44: 
45:     Returns
46:     -------
47:     d : dict
48:         Dictionary of parsed flags, split into relevant categories.
49:         These categories are the keys of `d`:
50: 
51:         * 'include_dirs'
52:         * 'library_dirs'
53:         * 'libraries'
54:         * 'macros'
55:         * 'ignored'
56: 
57:     '''
58:     d = {'include_dirs': [], 'library_dirs': [], 'libraries': [],
59:          'macros': [], 'ignored': []}
60: 
61:     flags = (' ' + line).split(' -')
62:     for flag in flags:
63:         flag = '-' + flag
64:         if len(flag) > 0:
65:             if flag.startswith('-I'):
66:                 d['include_dirs'].append(flag[2:].strip())
67:             elif flag.startswith('-L'):
68:                 d['library_dirs'].append(flag[2:].strip())
69:             elif flag.startswith('-l'):
70:                 d['libraries'].append(flag[2:].strip())
71:             elif flag.startswith('-D'):
72:                 d['macros'].append(flag[2:].strip())
73:             else:
74:                 d['ignored'].append(flag)
75: 
76:     return d
77: 
78: def _escape_backslash(val):
79:     return val.replace('\\', '\\\\')
80: 
81: class LibraryInfo(object):
82:     '''
83:     Object containing build information about a library.
84: 
85:     Parameters
86:     ----------
87:     name : str
88:         The library name.
89:     description : str
90:         Description of the library.
91:     version : str
92:         Version string.
93:     sections : dict
94:         The sections of the configuration file for the library. The keys are
95:         the section headers, the values the text under each header.
96:     vars : class instance
97:         A `VariableSet` instance, which contains ``(name, value)`` pairs for
98:         variables defined in the configuration file for the library.
99:     requires : sequence, optional
100:         The required libraries for the library to be installed.
101: 
102:     Notes
103:     -----
104:     All input parameters (except "sections" which is a method) are available as
105:     attributes of the same name.
106: 
107:     '''
108:     def __init__(self, name, description, version, sections, vars, requires=None):
109:         self.name = name
110:         self.description = description
111:         if requires:
112:             self.requires = requires
113:         else:
114:             self.requires = []
115:         self.version = version
116:         self._sections = sections
117:         self.vars = vars
118: 
119:     def sections(self):
120:         '''
121:         Return the section headers of the config file.
122: 
123:         Parameters
124:         ----------
125:         None
126: 
127:         Returns
128:         -------
129:         keys : list of str
130:             The list of section headers.
131: 
132:         '''
133:         return list(self._sections.keys())
134: 
135:     def cflags(self, section="default"):
136:         val = self.vars.interpolate(self._sections[section]['cflags'])
137:         return _escape_backslash(val)
138: 
139:     def libs(self, section="default"):
140:         val = self.vars.interpolate(self._sections[section]['libs'])
141:         return _escape_backslash(val)
142: 
143:     def __str__(self):
144:         m = ['Name: %s' % self.name, 'Description: %s' % self.description]
145:         if self.requires:
146:             m.append('Requires:')
147:         else:
148:             m.append('Requires: %s' % ",".join(self.requires))
149:         m.append('Version: %s' % self.version)
150: 
151:         return "\n".join(m)
152: 
153: class VariableSet(object):
154:     '''
155:     Container object for the variables defined in a config file.
156: 
157:     `VariableSet` can be used as a plain dictionary, with the variable names
158:     as keys.
159: 
160:     Parameters
161:     ----------
162:     d : dict
163:         Dict of items in the "variables" section of the configuration file.
164: 
165:     '''
166:     def __init__(self, d):
167:         self._raw_data = dict([(k, v) for k, v in d.items()])
168: 
169:         self._re = {}
170:         self._re_sub = {}
171: 
172:         self._init_parse()
173: 
174:     def _init_parse(self):
175:         for k, v in self._raw_data.items():
176:             self._init_parse_var(k, v)
177: 
178:     def _init_parse_var(self, name, value):
179:         self._re[name] = re.compile(r'\$\{%s\}' % name)
180:         self._re_sub[name] = value
181: 
182:     def interpolate(self, value):
183:         # Brute force: we keep interpolating until there is no '${var}' anymore
184:         # or until interpolated string is equal to input string
185:         def _interpolate(value):
186:             for k in self._re.keys():
187:                 value = self._re[k].sub(self._re_sub[k], value)
188:             return value
189:         while _VAR.search(value):
190:             nvalue = _interpolate(value)
191:             if nvalue == value:
192:                 break
193:             value = nvalue
194: 
195:         return value
196: 
197:     def variables(self):
198:         '''
199:         Return the list of variable names.
200: 
201:         Parameters
202:         ----------
203:         None
204: 
205:         Returns
206:         -------
207:         names : list of str
208:             The names of all variables in the `VariableSet` instance.
209: 
210:         '''
211:         return list(self._raw_data.keys())
212: 
213:     # Emulate a dict to set/get variables values
214:     def __getitem__(self, name):
215:         return self._raw_data[name]
216: 
217:     def __setitem__(self, name, value):
218:         self._raw_data[name] = value
219:         self._init_parse_var(name, value)
220: 
221: def parse_meta(config):
222:     if not config.has_section('meta'):
223:         raise FormatError("No meta section found !")
224: 
225:     d = {}
226:     for name, value in config.items('meta'):
227:         d[name] = value
228: 
229:     for k in ['name', 'description', 'version']:
230:         if not k in d:
231:             raise FormatError("Option %s (section [meta]) is mandatory, "
232:                 "but not found" % k)
233: 
234:     if not 'requires' in d:
235:         d['requires'] = []
236: 
237:     return d
238: 
239: def parse_variables(config):
240:     if not config.has_section('variables'):
241:         raise FormatError("No variables section found !")
242: 
243:     d = {}
244: 
245:     for name, value in config.items("variables"):
246:         d[name] = value
247: 
248:     return VariableSet(d)
249: 
250: def parse_sections(config):
251:     return meta_d, r
252: 
253: def pkg_to_filename(pkg_name):
254:     return "%s.ini" % pkg_name
255: 
256: def parse_config(filename, dirs=None):
257:     if dirs:
258:         filenames = [os.path.join(d, filename) for d in dirs]
259:     else:
260:         filenames = [filename]
261: 
262:     if sys.version[:3] > '3.1':
263:         # SafeConfigParser is deprecated in py-3.2 and renamed to ConfigParser
264:         config = ConfigParser()
265:     else:
266:         config = SafeConfigParser()
267: 
268:     n = config.read(filenames)
269:     if not len(n) >= 1:
270:         raise PkgNotFound("Could not find file(s) %s" % str(filenames))
271: 
272:     # Parse meta and variables sections
273:     meta = parse_meta(config)
274: 
275:     vars = {}
276:     if config.has_section('variables'):
277:         for name, value in config.items("variables"):
278:             vars[name] = _escape_backslash(value)
279: 
280:     # Parse "normal" sections
281:     secs = [s for s in config.sections() if not s in ['meta', 'variables']]
282:     sections = {}
283: 
284:     requires = {}
285:     for s in secs:
286:         d = {}
287:         if config.has_option(s, "requires"):
288:             requires[s] = config.get(s, 'requires')
289: 
290:         for name, value in config.items(s):
291:             d[name] = value
292:         sections[s] = d
293: 
294:     return meta, vars, sections, requires
295: 
296: def _read_config_imp(filenames, dirs=None):
297:     def _read_config(f):
298:         meta, vars, sections, reqs = parse_config(f, dirs)
299:         # recursively add sections and variables of required libraries
300:         for rname, rvalue in reqs.items():
301:             nmeta, nvars, nsections, nreqs = _read_config(pkg_to_filename(rvalue))
302: 
303:             # Update var dict for variables not in 'top' config file
304:             for k, v in nvars.items():
305:                 if not k in vars:
306:                     vars[k] = v
307: 
308:             # Update sec dict
309:             for oname, ovalue in nsections[rname].items():
310:                 if ovalue:
311:                     sections[rname][oname] += ' %s' % ovalue
312: 
313:         return meta, vars, sections, reqs
314: 
315:     meta, vars, sections, reqs = _read_config(filenames)
316: 
317:     # FIXME: document this. If pkgname is defined in the variables section, and
318:     # there is no pkgdir variable defined, pkgdir is automatically defined to
319:     # the path of pkgname. This requires the package to be imported to work
320:     if not 'pkgdir' in vars and "pkgname" in vars:
321:         pkgname = vars["pkgname"]
322:         if not pkgname in sys.modules:
323:             raise ValueError("You should import %s to get information on %s" %
324:                              (pkgname, meta["name"]))
325: 
326:         mod = sys.modules[pkgname]
327:         vars["pkgdir"] = _escape_backslash(os.path.dirname(mod.__file__))
328: 
329:     return LibraryInfo(name=meta["name"], description=meta["description"],
330:             version=meta["version"], sections=sections, vars=VariableSet(vars))
331: 
332: # Trivial cache to cache LibraryInfo instances creation. To be really
333: # efficient, the cache should be handled in read_config, since a same file can
334: # be parsed many time outside LibraryInfo creation, but I doubt this will be a
335: # problem in practice
336: _CACHE = {}
337: def read_config(pkgname, dirs=None):
338:     '''
339:     Return library info for a package from its configuration file.
340: 
341:     Parameters
342:     ----------
343:     pkgname : str
344:         Name of the package (should match the name of the .ini file, without
345:         the extension, e.g. foo for the file foo.ini).
346:     dirs : sequence, optional
347:         If given, should be a sequence of directories - usually including
348:         the NumPy base directory - where to look for npy-pkg-config files.
349: 
350:     Returns
351:     -------
352:     pkginfo : class instance
353:         The `LibraryInfo` instance containing the build information.
354: 
355:     Raises
356:     ------
357:     PkgNotFound
358:         If the package is not found.
359: 
360:     See Also
361:     --------
362:     misc_util.get_info, misc_util.get_pkg_info
363: 
364:     Examples
365:     --------
366:     >>> npymath_info = np.distutils.npy_pkg_config.read_config('npymath')
367:     >>> type(npymath_info)
368:     <class 'numpy.distutils.npy_pkg_config.LibraryInfo'>
369:     >>> print(npymath_info)
370:     Name: npymath
371:     Description: Portable, core math library implementing C99 standard
372:     Requires:
373:     Version: 0.1  #random
374: 
375:     '''
376:     try:
377:         return _CACHE[pkgname]
378:     except KeyError:
379:         v = _read_config_imp(pkg_to_filename(pkgname), dirs)
380:         _CACHE[pkgname] = v
381:         return v
382: 
383: # TODO:
384: #   - implements version comparison (modversion + atleast)
385: 
386: # pkg-config simple emulator - useful for debugging, and maybe later to query
387: # the system
388: if __name__ == '__main__':
389:     import sys
390:     from optparse import OptionParser
391:     import glob
392: 
393:     parser = OptionParser()
394:     parser.add_option("--cflags", dest="cflags", action="store_true",
395:                       help="output all preprocessor and compiler flags")
396:     parser.add_option("--libs", dest="libs", action="store_true",
397:                       help="output all linker flags")
398:     parser.add_option("--use-section", dest="section",
399:                       help="use this section instead of default for options")
400:     parser.add_option("--version", dest="version", action="store_true",
401:                       help="output version")
402:     parser.add_option("--atleast-version", dest="min_version",
403:                       help="Minimal version")
404:     parser.add_option("--list-all", dest="list_all", action="store_true",
405:                       help="Minimal version")
406:     parser.add_option("--define-variable", dest="define_variable",
407:                       help="Replace variable with the given value")
408: 
409:     (options, args) = parser.parse_args(sys.argv)
410: 
411:     if len(args) < 2:
412:         raise ValueError("Expect package name on the command line:")
413: 
414:     if options.list_all:
415:         files = glob.glob("*.ini")
416:         for f in files:
417:             info = read_config(f)
418:             print("%s\t%s - %s" % (info.name, info.name, info.description))
419: 
420:     pkg_name = args[1]
421:     import os
422:     d = os.environ.get('NPY_PKG_CONFIG_PATH')
423:     if d:
424:         info = read_config(pkg_name, ['numpy/core/lib/npy-pkg-config', '.', d])
425:     else:
426:         info = read_config(pkg_name, ['numpy/core/lib/npy-pkg-config', '.'])
427: 
428:     if options.section:
429:         section = options.section
430:     else:
431:         section = "default"
432: 
433:     if options.define_variable:
434:         m = re.search('([\S]+)=([\S]+)', options.define_variable)
435:         if not m:
436:             raise ValueError("--define-variable option should be of " \
437:                              "the form --define-variable=foo=bar")
438:         else:
439:             name = m.group(1)
440:             value = m.group(2)
441:         info.vars[name] = value
442: 
443:     if options.cflags:
444:         print(info.cflags(section))
445:     if options.libs:
446:         print(info.libs(section))
447:     if options.version:
448:         print(info.version)
449:     if options.min_version:
450:         print(info.version >= options.min_version)
451: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import re' statement (line 4)
import re

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import os' statement (line 5)
import os

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os', os, module_type_store)




# Obtaining the type of the subscript
int_44154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 20), 'int')
# Getting the type of 'sys' (line 7)
sys_44155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 7)
version_info_44156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 3), sys_44155, 'version_info')
# Obtaining the member '__getitem__' of a type (line 7)
getitem___44157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 3), version_info_44156, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 7)
subscript_call_result_44158 = invoke(stypy.reporting.localization.Localization(__file__, 7, 3), getitem___44157, int_44154)

int_44159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'int')
# Applying the binary operator '<' (line 7)
result_lt_44160 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 3), '<', subscript_call_result_44158, int_44159)

# Testing the type of an if condition (line 7)
if_condition_44161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 7, 0), result_lt_44160)
# Assigning a type to the variable 'if_condition_44161' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'if_condition_44161', if_condition_44161)
# SSA begins for if statement (line 7)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))

# 'from ConfigParser import SafeConfigParser, NoOptionError' statement (line 8)
from ConfigParser import SafeConfigParser, NoOptionError

import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'ConfigParser', None, module_type_store, ['SafeConfigParser', 'NoOptionError'], [SafeConfigParser, NoOptionError])

# SSA branch for the else part of an if statement (line 7)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 4))

# 'from configparser import ConfigParser, SafeConfigParser, NoOptionError' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_44162 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'configparser')

if (type(import_44162) is not StypyTypeError):

    if (import_44162 != 'pyd_module'):
        __import__(import_44162)
        sys_modules_44163 = sys.modules[import_44162]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'configparser', sys_modules_44163.module_type_store, module_type_store, ['ConfigParser', 'SafeConfigParser', 'NoOptionError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 4), __file__, sys_modules_44163, sys_modules_44163.module_type_store, module_type_store)
    else:
        from configparser import ConfigParser, SafeConfigParser, NoOptionError

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'configparser', None, module_type_store, ['ConfigParser', 'SafeConfigParser', 'NoOptionError'], [ConfigParser, SafeConfigParser, NoOptionError])

else:
    # Assigning a type to the variable 'configparser' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'configparser', import_44162)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

# SSA join for if statement (line 7)
module_type_store = module_type_store.join_ssa_context()


# Assigning a List to a Name (line 12):

# Assigning a List to a Name (line 12):
__all__ = ['FormatError', 'PkgNotFound', 'LibraryInfo', 'VariableSet', 'read_config', 'parse_flags']
module_type_store.set_exportable_members(['FormatError', 'PkgNotFound', 'LibraryInfo', 'VariableSet', 'read_config', 'parse_flags'])

# Obtaining an instance of the builtin type 'list' (line 12)
list_44164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_44165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'FormatError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_44164, str_44165)
# Adding element type (line 12)
str_44166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 26), 'str', 'PkgNotFound')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_44164, str_44166)
# Adding element type (line 12)
str_44167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 41), 'str', 'LibraryInfo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_44164, str_44167)
# Adding element type (line 12)
str_44168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 56), 'str', 'VariableSet')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_44164, str_44168)
# Adding element type (line 12)
str_44169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 8), 'str', 'read_config')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_44164, str_44169)
# Adding element type (line 12)
str_44170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 23), 'str', 'parse_flags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_44164, str_44170)

# Assigning a type to the variable '__all__' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '__all__', list_44164)

# Assigning a Call to a Name (line 15):

# Assigning a Call to a Name (line 15):

# Call to compile(...): (line 15)
# Processing the call arguments (line 15)
str_44173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 18), 'str', '\\$\\{([a-zA-Z0-9_-]+)\\}')
# Processing the call keyword arguments (line 15)
kwargs_44174 = {}
# Getting the type of 're' (line 15)
re_44171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 7), 're', False)
# Obtaining the member 'compile' of a type (line 15)
compile_44172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 7), re_44171, 'compile')
# Calling compile(args, kwargs) (line 15)
compile_call_result_44175 = invoke(stypy.reporting.localization.Localization(__file__, 15, 7), compile_44172, *[str_44173], **kwargs_44174)

# Assigning a type to the variable '_VAR' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), '_VAR', compile_call_result_44175)
# Declaration of the 'FormatError' class
# Getting the type of 'IOError' (line 17)
IOError_44176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 18), 'IOError')

class FormatError(IOError_44176, ):
    str_44177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, (-1)), 'str', '\n    Exception thrown when there is a problem parsing a configuration file.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormatError.__init__', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 23):
        
        # Assigning a Name to a Attribute (line 23):
        # Getting the type of 'msg' (line 23)
        msg_44178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'msg')
        # Getting the type of 'self' (line 23)
        self_44179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Setting the type of the member 'msg' of a type (line 23)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_44179, 'msg', msg_44178)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 25, 4, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FormatError.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        FormatError.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FormatError.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FormatError.stypy__str__.__dict__.__setitem__('stypy_function_name', 'FormatError.__str__')
        FormatError.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        FormatError.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FormatError.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FormatError.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FormatError.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FormatError.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FormatError.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FormatError.__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        # Getting the type of 'self' (line 26)
        self_44180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'self')
        # Obtaining the member 'msg' of a type (line 26)
        msg_44181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 15), self_44180, 'msg')
        # Assigning a type to the variable 'stypy_return_type' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'stypy_return_type', msg_44181)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_44182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44182)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_44182


# Assigning a type to the variable 'FormatError' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'FormatError', FormatError)
# Declaration of the 'PkgNotFound' class
# Getting the type of 'IOError' (line 28)
IOError_44183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 18), 'IOError')

class PkgNotFound(IOError_44183, ):
    str_44184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'str', 'Exception raised when a package can not be located.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PkgNotFound.__init__', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 31):
        
        # Assigning a Name to a Attribute (line 31):
        # Getting the type of 'msg' (line 31)
        msg_44185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'msg')
        # Getting the type of 'self' (line 31)
        self_44186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'msg' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_44186, 'msg', msg_44185)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PkgNotFound.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        PkgNotFound.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PkgNotFound.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        PkgNotFound.stypy__str__.__dict__.__setitem__('stypy_function_name', 'PkgNotFound.__str__')
        PkgNotFound.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        PkgNotFound.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        PkgNotFound.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PkgNotFound.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        PkgNotFound.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        PkgNotFound.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PkgNotFound.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PkgNotFound.__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        # Getting the type of 'self' (line 34)
        self_44187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'self')
        # Obtaining the member 'msg' of a type (line 34)
        msg_44188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 15), self_44187, 'msg')
        # Assigning a type to the variable 'stypy_return_type' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', msg_44188)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_44189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44189)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_44189


# Assigning a type to the variable 'PkgNotFound' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'PkgNotFound', PkgNotFound)

@norecursion
def parse_flags(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'parse_flags'
    module_type_store = module_type_store.open_function_context('parse_flags', 36, 0, False)
    
    # Passed parameters checking function
    parse_flags.stypy_localization = localization
    parse_flags.stypy_type_of_self = None
    parse_flags.stypy_type_store = module_type_store
    parse_flags.stypy_function_name = 'parse_flags'
    parse_flags.stypy_param_names_list = ['line']
    parse_flags.stypy_varargs_param_name = None
    parse_flags.stypy_kwargs_param_name = None
    parse_flags.stypy_call_defaults = defaults
    parse_flags.stypy_call_varargs = varargs
    parse_flags.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_flags', ['line'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_flags', localization, ['line'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_flags(...)' code ##################

    str_44190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, (-1)), 'str', "\n    Parse a line from a config file containing compile flags.\n\n    Parameters\n    ----------\n    line : str\n        A single line containing one or more compile flags.\n\n    Returns\n    -------\n    d : dict\n        Dictionary of parsed flags, split into relevant categories.\n        These categories are the keys of `d`:\n\n        * 'include_dirs'\n        * 'library_dirs'\n        * 'libraries'\n        * 'macros'\n        * 'ignored'\n\n    ")
    
    # Assigning a Dict to a Name (line 58):
    
    # Assigning a Dict to a Name (line 58):
    
    # Obtaining an instance of the builtin type 'dict' (line 58)
    dict_44191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 58)
    # Adding element type (key, value) (line 58)
    str_44192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 9), 'str', 'include_dirs')
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_44193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 8), dict_44191, (str_44192, list_44193))
    # Adding element type (key, value) (line 58)
    str_44194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 29), 'str', 'library_dirs')
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_44195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 45), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 8), dict_44191, (str_44194, list_44195))
    # Adding element type (key, value) (line 58)
    str_44196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 49), 'str', 'libraries')
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_44197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 62), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 8), dict_44191, (str_44196, list_44197))
    # Adding element type (key, value) (line 58)
    str_44198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 9), 'str', 'macros')
    
    # Obtaining an instance of the builtin type 'list' (line 59)
    list_44199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 59)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 8), dict_44191, (str_44198, list_44199))
    # Adding element type (key, value) (line 58)
    str_44200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 23), 'str', 'ignored')
    
    # Obtaining an instance of the builtin type 'list' (line 59)
    list_44201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 59)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 8), dict_44191, (str_44200, list_44201))
    
    # Assigning a type to the variable 'd' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'd', dict_44191)
    
    # Assigning a Call to a Name (line 61):
    
    # Assigning a Call to a Name (line 61):
    
    # Call to split(...): (line 61)
    # Processing the call arguments (line 61)
    str_44206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 31), 'str', ' -')
    # Processing the call keyword arguments (line 61)
    kwargs_44207 = {}
    str_44202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 13), 'str', ' ')
    # Getting the type of 'line' (line 61)
    line_44203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'line', False)
    # Applying the binary operator '+' (line 61)
    result_add_44204 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 13), '+', str_44202, line_44203)
    
    # Obtaining the member 'split' of a type (line 61)
    split_44205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 13), result_add_44204, 'split')
    # Calling split(args, kwargs) (line 61)
    split_call_result_44208 = invoke(stypy.reporting.localization.Localization(__file__, 61, 13), split_44205, *[str_44206], **kwargs_44207)
    
    # Assigning a type to the variable 'flags' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'flags', split_call_result_44208)
    
    # Getting the type of 'flags' (line 62)
    flags_44209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'flags')
    # Testing the type of a for loop iterable (line 62)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 62, 4), flags_44209)
    # Getting the type of the for loop variable (line 62)
    for_loop_var_44210 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 62, 4), flags_44209)
    # Assigning a type to the variable 'flag' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'flag', for_loop_var_44210)
    # SSA begins for a for statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 63):
    
    # Assigning a BinOp to a Name (line 63):
    str_44211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 15), 'str', '-')
    # Getting the type of 'flag' (line 63)
    flag_44212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 21), 'flag')
    # Applying the binary operator '+' (line 63)
    result_add_44213 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 15), '+', str_44211, flag_44212)
    
    # Assigning a type to the variable 'flag' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'flag', result_add_44213)
    
    
    
    # Call to len(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'flag' (line 64)
    flag_44215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'flag', False)
    # Processing the call keyword arguments (line 64)
    kwargs_44216 = {}
    # Getting the type of 'len' (line 64)
    len_44214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'len', False)
    # Calling len(args, kwargs) (line 64)
    len_call_result_44217 = invoke(stypy.reporting.localization.Localization(__file__, 64, 11), len_44214, *[flag_44215], **kwargs_44216)
    
    int_44218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 23), 'int')
    # Applying the binary operator '>' (line 64)
    result_gt_44219 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 11), '>', len_call_result_44217, int_44218)
    
    # Testing the type of an if condition (line 64)
    if_condition_44220 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), result_gt_44219)
    # Assigning a type to the variable 'if_condition_44220' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_44220', if_condition_44220)
    # SSA begins for if statement (line 64)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to startswith(...): (line 65)
    # Processing the call arguments (line 65)
    str_44223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 31), 'str', '-I')
    # Processing the call keyword arguments (line 65)
    kwargs_44224 = {}
    # Getting the type of 'flag' (line 65)
    flag_44221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'flag', False)
    # Obtaining the member 'startswith' of a type (line 65)
    startswith_44222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 15), flag_44221, 'startswith')
    # Calling startswith(args, kwargs) (line 65)
    startswith_call_result_44225 = invoke(stypy.reporting.localization.Localization(__file__, 65, 15), startswith_44222, *[str_44223], **kwargs_44224)
    
    # Testing the type of an if condition (line 65)
    if_condition_44226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 12), startswith_call_result_44225)
    # Assigning a type to the variable 'if_condition_44226' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'if_condition_44226', if_condition_44226)
    # SSA begins for if statement (line 65)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 66)
    # Processing the call arguments (line 66)
    
    # Call to strip(...): (line 66)
    # Processing the call keyword arguments (line 66)
    kwargs_44238 = {}
    
    # Obtaining the type of the subscript
    int_44232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 46), 'int')
    slice_44233 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 66, 41), int_44232, None, None)
    # Getting the type of 'flag' (line 66)
    flag_44234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 41), 'flag', False)
    # Obtaining the member '__getitem__' of a type (line 66)
    getitem___44235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 41), flag_44234, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
    subscript_call_result_44236 = invoke(stypy.reporting.localization.Localization(__file__, 66, 41), getitem___44235, slice_44233)
    
    # Obtaining the member 'strip' of a type (line 66)
    strip_44237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 41), subscript_call_result_44236, 'strip')
    # Calling strip(args, kwargs) (line 66)
    strip_call_result_44239 = invoke(stypy.reporting.localization.Localization(__file__, 66, 41), strip_44237, *[], **kwargs_44238)
    
    # Processing the call keyword arguments (line 66)
    kwargs_44240 = {}
    
    # Obtaining the type of the subscript
    str_44227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 18), 'str', 'include_dirs')
    # Getting the type of 'd' (line 66)
    d_44228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'd', False)
    # Obtaining the member '__getitem__' of a type (line 66)
    getitem___44229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 16), d_44228, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
    subscript_call_result_44230 = invoke(stypy.reporting.localization.Localization(__file__, 66, 16), getitem___44229, str_44227)
    
    # Obtaining the member 'append' of a type (line 66)
    append_44231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 16), subscript_call_result_44230, 'append')
    # Calling append(args, kwargs) (line 66)
    append_call_result_44241 = invoke(stypy.reporting.localization.Localization(__file__, 66, 16), append_44231, *[strip_call_result_44239], **kwargs_44240)
    
    # SSA branch for the else part of an if statement (line 65)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to startswith(...): (line 67)
    # Processing the call arguments (line 67)
    str_44244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 33), 'str', '-L')
    # Processing the call keyword arguments (line 67)
    kwargs_44245 = {}
    # Getting the type of 'flag' (line 67)
    flag_44242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'flag', False)
    # Obtaining the member 'startswith' of a type (line 67)
    startswith_44243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 17), flag_44242, 'startswith')
    # Calling startswith(args, kwargs) (line 67)
    startswith_call_result_44246 = invoke(stypy.reporting.localization.Localization(__file__, 67, 17), startswith_44243, *[str_44244], **kwargs_44245)
    
    # Testing the type of an if condition (line 67)
    if_condition_44247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 17), startswith_call_result_44246)
    # Assigning a type to the variable 'if_condition_44247' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'if_condition_44247', if_condition_44247)
    # SSA begins for if statement (line 67)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Call to strip(...): (line 68)
    # Processing the call keyword arguments (line 68)
    kwargs_44259 = {}
    
    # Obtaining the type of the subscript
    int_44253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 46), 'int')
    slice_44254 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 68, 41), int_44253, None, None)
    # Getting the type of 'flag' (line 68)
    flag_44255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 41), 'flag', False)
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___44256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 41), flag_44255, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_44257 = invoke(stypy.reporting.localization.Localization(__file__, 68, 41), getitem___44256, slice_44254)
    
    # Obtaining the member 'strip' of a type (line 68)
    strip_44258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 41), subscript_call_result_44257, 'strip')
    # Calling strip(args, kwargs) (line 68)
    strip_call_result_44260 = invoke(stypy.reporting.localization.Localization(__file__, 68, 41), strip_44258, *[], **kwargs_44259)
    
    # Processing the call keyword arguments (line 68)
    kwargs_44261 = {}
    
    # Obtaining the type of the subscript
    str_44248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 18), 'str', 'library_dirs')
    # Getting the type of 'd' (line 68)
    d_44249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'd', False)
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___44250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), d_44249, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_44251 = invoke(stypy.reporting.localization.Localization(__file__, 68, 16), getitem___44250, str_44248)
    
    # Obtaining the member 'append' of a type (line 68)
    append_44252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), subscript_call_result_44251, 'append')
    # Calling append(args, kwargs) (line 68)
    append_call_result_44262 = invoke(stypy.reporting.localization.Localization(__file__, 68, 16), append_44252, *[strip_call_result_44260], **kwargs_44261)
    
    # SSA branch for the else part of an if statement (line 67)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to startswith(...): (line 69)
    # Processing the call arguments (line 69)
    str_44265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 33), 'str', '-l')
    # Processing the call keyword arguments (line 69)
    kwargs_44266 = {}
    # Getting the type of 'flag' (line 69)
    flag_44263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'flag', False)
    # Obtaining the member 'startswith' of a type (line 69)
    startswith_44264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 17), flag_44263, 'startswith')
    # Calling startswith(args, kwargs) (line 69)
    startswith_call_result_44267 = invoke(stypy.reporting.localization.Localization(__file__, 69, 17), startswith_44264, *[str_44265], **kwargs_44266)
    
    # Testing the type of an if condition (line 69)
    if_condition_44268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 17), startswith_call_result_44267)
    # Assigning a type to the variable 'if_condition_44268' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'if_condition_44268', if_condition_44268)
    # SSA begins for if statement (line 69)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 70)
    # Processing the call arguments (line 70)
    
    # Call to strip(...): (line 70)
    # Processing the call keyword arguments (line 70)
    kwargs_44280 = {}
    
    # Obtaining the type of the subscript
    int_44274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 43), 'int')
    slice_44275 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 70, 38), int_44274, None, None)
    # Getting the type of 'flag' (line 70)
    flag_44276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 38), 'flag', False)
    # Obtaining the member '__getitem__' of a type (line 70)
    getitem___44277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 38), flag_44276, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
    subscript_call_result_44278 = invoke(stypy.reporting.localization.Localization(__file__, 70, 38), getitem___44277, slice_44275)
    
    # Obtaining the member 'strip' of a type (line 70)
    strip_44279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 38), subscript_call_result_44278, 'strip')
    # Calling strip(args, kwargs) (line 70)
    strip_call_result_44281 = invoke(stypy.reporting.localization.Localization(__file__, 70, 38), strip_44279, *[], **kwargs_44280)
    
    # Processing the call keyword arguments (line 70)
    kwargs_44282 = {}
    
    # Obtaining the type of the subscript
    str_44269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 18), 'str', 'libraries')
    # Getting the type of 'd' (line 70)
    d_44270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'd', False)
    # Obtaining the member '__getitem__' of a type (line 70)
    getitem___44271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), d_44270, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
    subscript_call_result_44272 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), getitem___44271, str_44269)
    
    # Obtaining the member 'append' of a type (line 70)
    append_44273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), subscript_call_result_44272, 'append')
    # Calling append(args, kwargs) (line 70)
    append_call_result_44283 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), append_44273, *[strip_call_result_44281], **kwargs_44282)
    
    # SSA branch for the else part of an if statement (line 69)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to startswith(...): (line 71)
    # Processing the call arguments (line 71)
    str_44286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 33), 'str', '-D')
    # Processing the call keyword arguments (line 71)
    kwargs_44287 = {}
    # Getting the type of 'flag' (line 71)
    flag_44284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'flag', False)
    # Obtaining the member 'startswith' of a type (line 71)
    startswith_44285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 17), flag_44284, 'startswith')
    # Calling startswith(args, kwargs) (line 71)
    startswith_call_result_44288 = invoke(stypy.reporting.localization.Localization(__file__, 71, 17), startswith_44285, *[str_44286], **kwargs_44287)
    
    # Testing the type of an if condition (line 71)
    if_condition_44289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 17), startswith_call_result_44288)
    # Assigning a type to the variable 'if_condition_44289' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 17), 'if_condition_44289', if_condition_44289)
    # SSA begins for if statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 72)
    # Processing the call arguments (line 72)
    
    # Call to strip(...): (line 72)
    # Processing the call keyword arguments (line 72)
    kwargs_44301 = {}
    
    # Obtaining the type of the subscript
    int_44295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 40), 'int')
    slice_44296 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 72, 35), int_44295, None, None)
    # Getting the type of 'flag' (line 72)
    flag_44297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 35), 'flag', False)
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___44298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 35), flag_44297, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_44299 = invoke(stypy.reporting.localization.Localization(__file__, 72, 35), getitem___44298, slice_44296)
    
    # Obtaining the member 'strip' of a type (line 72)
    strip_44300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 35), subscript_call_result_44299, 'strip')
    # Calling strip(args, kwargs) (line 72)
    strip_call_result_44302 = invoke(stypy.reporting.localization.Localization(__file__, 72, 35), strip_44300, *[], **kwargs_44301)
    
    # Processing the call keyword arguments (line 72)
    kwargs_44303 = {}
    
    # Obtaining the type of the subscript
    str_44290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 18), 'str', 'macros')
    # Getting the type of 'd' (line 72)
    d_44291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'd', False)
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___44292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 16), d_44291, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_44293 = invoke(stypy.reporting.localization.Localization(__file__, 72, 16), getitem___44292, str_44290)
    
    # Obtaining the member 'append' of a type (line 72)
    append_44294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 16), subscript_call_result_44293, 'append')
    # Calling append(args, kwargs) (line 72)
    append_call_result_44304 = invoke(stypy.reporting.localization.Localization(__file__, 72, 16), append_44294, *[strip_call_result_44302], **kwargs_44303)
    
    # SSA branch for the else part of an if statement (line 71)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'flag' (line 74)
    flag_44310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 36), 'flag', False)
    # Processing the call keyword arguments (line 74)
    kwargs_44311 = {}
    
    # Obtaining the type of the subscript
    str_44305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 18), 'str', 'ignored')
    # Getting the type of 'd' (line 74)
    d_44306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'd', False)
    # Obtaining the member '__getitem__' of a type (line 74)
    getitem___44307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 16), d_44306, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 74)
    subscript_call_result_44308 = invoke(stypy.reporting.localization.Localization(__file__, 74, 16), getitem___44307, str_44305)
    
    # Obtaining the member 'append' of a type (line 74)
    append_44309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 16), subscript_call_result_44308, 'append')
    # Calling append(args, kwargs) (line 74)
    append_call_result_44312 = invoke(stypy.reporting.localization.Localization(__file__, 74, 16), append_44309, *[flag_44310], **kwargs_44311)
    
    # SSA join for if statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 69)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 67)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 65)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 64)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'd' (line 76)
    d_44313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'd')
    # Assigning a type to the variable 'stypy_return_type' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type', d_44313)
    
    # ################# End of 'parse_flags(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_flags' in the type store
    # Getting the type of 'stypy_return_type' (line 36)
    stypy_return_type_44314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44314)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_flags'
    return stypy_return_type_44314

# Assigning a type to the variable 'parse_flags' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'parse_flags', parse_flags)

@norecursion
def _escape_backslash(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_escape_backslash'
    module_type_store = module_type_store.open_function_context('_escape_backslash', 78, 0, False)
    
    # Passed parameters checking function
    _escape_backslash.stypy_localization = localization
    _escape_backslash.stypy_type_of_self = None
    _escape_backslash.stypy_type_store = module_type_store
    _escape_backslash.stypy_function_name = '_escape_backslash'
    _escape_backslash.stypy_param_names_list = ['val']
    _escape_backslash.stypy_varargs_param_name = None
    _escape_backslash.stypy_kwargs_param_name = None
    _escape_backslash.stypy_call_defaults = defaults
    _escape_backslash.stypy_call_varargs = varargs
    _escape_backslash.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_escape_backslash', ['val'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_escape_backslash', localization, ['val'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_escape_backslash(...)' code ##################

    
    # Call to replace(...): (line 79)
    # Processing the call arguments (line 79)
    str_44317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 23), 'str', '\\')
    str_44318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 29), 'str', '\\\\')
    # Processing the call keyword arguments (line 79)
    kwargs_44319 = {}
    # Getting the type of 'val' (line 79)
    val_44315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'val', False)
    # Obtaining the member 'replace' of a type (line 79)
    replace_44316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 11), val_44315, 'replace')
    # Calling replace(args, kwargs) (line 79)
    replace_call_result_44320 = invoke(stypy.reporting.localization.Localization(__file__, 79, 11), replace_44316, *[str_44317, str_44318], **kwargs_44319)
    
    # Assigning a type to the variable 'stypy_return_type' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type', replace_call_result_44320)
    
    # ################# End of '_escape_backslash(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_escape_backslash' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_44321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44321)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_escape_backslash'
    return stypy_return_type_44321

# Assigning a type to the variable '_escape_backslash' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), '_escape_backslash', _escape_backslash)
# Declaration of the 'LibraryInfo' class

class LibraryInfo(object, ):
    str_44322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, (-1)), 'str', '\n    Object containing build information about a library.\n\n    Parameters\n    ----------\n    name : str\n        The library name.\n    description : str\n        Description of the library.\n    version : str\n        Version string.\n    sections : dict\n        The sections of the configuration file for the library. The keys are\n        the section headers, the values the text under each header.\n    vars : class instance\n        A `VariableSet` instance, which contains ``(name, value)`` pairs for\n        variables defined in the configuration file for the library.\n    requires : sequence, optional\n        The required libraries for the library to be installed.\n\n    Notes\n    -----\n    All input parameters (except "sections" which is a method) are available as\n    attributes of the same name.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 108)
        None_44323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 76), 'None')
        defaults = [None_44323]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LibraryInfo.__init__', ['name', 'description', 'version', 'sections', 'vars', 'requires'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name', 'description', 'version', 'sections', 'vars', 'requires'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 109):
        
        # Assigning a Name to a Attribute (line 109):
        # Getting the type of 'name' (line 109)
        name_44324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 20), 'name')
        # Getting the type of 'self' (line 109)
        self_44325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self')
        # Setting the type of the member 'name' of a type (line 109)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_44325, 'name', name_44324)
        
        # Assigning a Name to a Attribute (line 110):
        
        # Assigning a Name to a Attribute (line 110):
        # Getting the type of 'description' (line 110)
        description_44326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'description')
        # Getting the type of 'self' (line 110)
        self_44327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self')
        # Setting the type of the member 'description' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_44327, 'description', description_44326)
        
        # Getting the type of 'requires' (line 111)
        requires_44328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'requires')
        # Testing the type of an if condition (line 111)
        if_condition_44329 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 8), requires_44328)
        # Assigning a type to the variable 'if_condition_44329' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'if_condition_44329', if_condition_44329)
        # SSA begins for if statement (line 111)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 112):
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'requires' (line 112)
        requires_44330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'requires')
        # Getting the type of 'self' (line 112)
        self_44331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'self')
        # Setting the type of the member 'requires' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 12), self_44331, 'requires', requires_44330)
        # SSA branch for the else part of an if statement (line 111)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Attribute (line 114):
        
        # Assigning a List to a Attribute (line 114):
        
        # Obtaining an instance of the builtin type 'list' (line 114)
        list_44332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 114)
        
        # Getting the type of 'self' (line 114)
        self_44333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'self')
        # Setting the type of the member 'requires' of a type (line 114)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), self_44333, 'requires', list_44332)
        # SSA join for if statement (line 111)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 115):
        
        # Assigning a Name to a Attribute (line 115):
        # Getting the type of 'version' (line 115)
        version_44334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 23), 'version')
        # Getting the type of 'self' (line 115)
        self_44335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'self')
        # Setting the type of the member 'version' of a type (line 115)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 8), self_44335, 'version', version_44334)
        
        # Assigning a Name to a Attribute (line 116):
        
        # Assigning a Name to a Attribute (line 116):
        # Getting the type of 'sections' (line 116)
        sections_44336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 25), 'sections')
        # Getting the type of 'self' (line 116)
        self_44337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'self')
        # Setting the type of the member '_sections' of a type (line 116)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), self_44337, '_sections', sections_44336)
        
        # Assigning a Name to a Attribute (line 117):
        
        # Assigning a Name to a Attribute (line 117):
        # Getting the type of 'vars' (line 117)
        vars_44338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 20), 'vars')
        # Getting the type of 'self' (line 117)
        self_44339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self')
        # Setting the type of the member 'vars' of a type (line 117)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), self_44339, 'vars', vars_44338)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def sections(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sections'
        module_type_store = module_type_store.open_function_context('sections', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LibraryInfo.sections.__dict__.__setitem__('stypy_localization', localization)
        LibraryInfo.sections.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LibraryInfo.sections.__dict__.__setitem__('stypy_type_store', module_type_store)
        LibraryInfo.sections.__dict__.__setitem__('stypy_function_name', 'LibraryInfo.sections')
        LibraryInfo.sections.__dict__.__setitem__('stypy_param_names_list', [])
        LibraryInfo.sections.__dict__.__setitem__('stypy_varargs_param_name', None)
        LibraryInfo.sections.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LibraryInfo.sections.__dict__.__setitem__('stypy_call_defaults', defaults)
        LibraryInfo.sections.__dict__.__setitem__('stypy_call_varargs', varargs)
        LibraryInfo.sections.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LibraryInfo.sections.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LibraryInfo.sections', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sections', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sections(...)' code ##################

        str_44340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, (-1)), 'str', '\n        Return the section headers of the config file.\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        keys : list of str\n            The list of section headers.\n\n        ')
        
        # Call to list(...): (line 133)
        # Processing the call arguments (line 133)
        
        # Call to keys(...): (line 133)
        # Processing the call keyword arguments (line 133)
        kwargs_44345 = {}
        # Getting the type of 'self' (line 133)
        self_44342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 20), 'self', False)
        # Obtaining the member '_sections' of a type (line 133)
        _sections_44343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 20), self_44342, '_sections')
        # Obtaining the member 'keys' of a type (line 133)
        keys_44344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 20), _sections_44343, 'keys')
        # Calling keys(args, kwargs) (line 133)
        keys_call_result_44346 = invoke(stypy.reporting.localization.Localization(__file__, 133, 20), keys_44344, *[], **kwargs_44345)
        
        # Processing the call keyword arguments (line 133)
        kwargs_44347 = {}
        # Getting the type of 'list' (line 133)
        list_44341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'list', False)
        # Calling list(args, kwargs) (line 133)
        list_call_result_44348 = invoke(stypy.reporting.localization.Localization(__file__, 133, 15), list_44341, *[keys_call_result_44346], **kwargs_44347)
        
        # Assigning a type to the variable 'stypy_return_type' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'stypy_return_type', list_call_result_44348)
        
        # ################# End of 'sections(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sections' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_44349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44349)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sections'
        return stypy_return_type_44349


    @norecursion
    def cflags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_44350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 29), 'str', 'default')
        defaults = [str_44350]
        # Create a new context for function 'cflags'
        module_type_store = module_type_store.open_function_context('cflags', 135, 4, False)
        # Assigning a type to the variable 'self' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LibraryInfo.cflags.__dict__.__setitem__('stypy_localization', localization)
        LibraryInfo.cflags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LibraryInfo.cflags.__dict__.__setitem__('stypy_type_store', module_type_store)
        LibraryInfo.cflags.__dict__.__setitem__('stypy_function_name', 'LibraryInfo.cflags')
        LibraryInfo.cflags.__dict__.__setitem__('stypy_param_names_list', ['section'])
        LibraryInfo.cflags.__dict__.__setitem__('stypy_varargs_param_name', None)
        LibraryInfo.cflags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LibraryInfo.cflags.__dict__.__setitem__('stypy_call_defaults', defaults)
        LibraryInfo.cflags.__dict__.__setitem__('stypy_call_varargs', varargs)
        LibraryInfo.cflags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LibraryInfo.cflags.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LibraryInfo.cflags', ['section'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cflags', localization, ['section'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cflags(...)' code ##################

        
        # Assigning a Call to a Name (line 136):
        
        # Assigning a Call to a Name (line 136):
        
        # Call to interpolate(...): (line 136)
        # Processing the call arguments (line 136)
        
        # Obtaining the type of the subscript
        str_44354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 60), 'str', 'cflags')
        
        # Obtaining the type of the subscript
        # Getting the type of 'section' (line 136)
        section_44355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 51), 'section', False)
        # Getting the type of 'self' (line 136)
        self_44356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 36), 'self', False)
        # Obtaining the member '_sections' of a type (line 136)
        _sections_44357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 36), self_44356, '_sections')
        # Obtaining the member '__getitem__' of a type (line 136)
        getitem___44358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 36), _sections_44357, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 136)
        subscript_call_result_44359 = invoke(stypy.reporting.localization.Localization(__file__, 136, 36), getitem___44358, section_44355)
        
        # Obtaining the member '__getitem__' of a type (line 136)
        getitem___44360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 36), subscript_call_result_44359, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 136)
        subscript_call_result_44361 = invoke(stypy.reporting.localization.Localization(__file__, 136, 36), getitem___44360, str_44354)
        
        # Processing the call keyword arguments (line 136)
        kwargs_44362 = {}
        # Getting the type of 'self' (line 136)
        self_44351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 14), 'self', False)
        # Obtaining the member 'vars' of a type (line 136)
        vars_44352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 14), self_44351, 'vars')
        # Obtaining the member 'interpolate' of a type (line 136)
        interpolate_44353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 14), vars_44352, 'interpolate')
        # Calling interpolate(args, kwargs) (line 136)
        interpolate_call_result_44363 = invoke(stypy.reporting.localization.Localization(__file__, 136, 14), interpolate_44353, *[subscript_call_result_44361], **kwargs_44362)
        
        # Assigning a type to the variable 'val' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'val', interpolate_call_result_44363)
        
        # Call to _escape_backslash(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'val' (line 137)
        val_44365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 33), 'val', False)
        # Processing the call keyword arguments (line 137)
        kwargs_44366 = {}
        # Getting the type of '_escape_backslash' (line 137)
        _escape_backslash_44364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), '_escape_backslash', False)
        # Calling _escape_backslash(args, kwargs) (line 137)
        _escape_backslash_call_result_44367 = invoke(stypy.reporting.localization.Localization(__file__, 137, 15), _escape_backslash_44364, *[val_44365], **kwargs_44366)
        
        # Assigning a type to the variable 'stypy_return_type' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'stypy_return_type', _escape_backslash_call_result_44367)
        
        # ################# End of 'cflags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cflags' in the type store
        # Getting the type of 'stypy_return_type' (line 135)
        stypy_return_type_44368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44368)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cflags'
        return stypy_return_type_44368


    @norecursion
    def libs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_44369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 27), 'str', 'default')
        defaults = [str_44369]
        # Create a new context for function 'libs'
        module_type_store = module_type_store.open_function_context('libs', 139, 4, False)
        # Assigning a type to the variable 'self' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LibraryInfo.libs.__dict__.__setitem__('stypy_localization', localization)
        LibraryInfo.libs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LibraryInfo.libs.__dict__.__setitem__('stypy_type_store', module_type_store)
        LibraryInfo.libs.__dict__.__setitem__('stypy_function_name', 'LibraryInfo.libs')
        LibraryInfo.libs.__dict__.__setitem__('stypy_param_names_list', ['section'])
        LibraryInfo.libs.__dict__.__setitem__('stypy_varargs_param_name', None)
        LibraryInfo.libs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LibraryInfo.libs.__dict__.__setitem__('stypy_call_defaults', defaults)
        LibraryInfo.libs.__dict__.__setitem__('stypy_call_varargs', varargs)
        LibraryInfo.libs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LibraryInfo.libs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LibraryInfo.libs', ['section'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'libs', localization, ['section'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'libs(...)' code ##################

        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to interpolate(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Obtaining the type of the subscript
        str_44373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 60), 'str', 'libs')
        
        # Obtaining the type of the subscript
        # Getting the type of 'section' (line 140)
        section_44374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 51), 'section', False)
        # Getting the type of 'self' (line 140)
        self_44375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 36), 'self', False)
        # Obtaining the member '_sections' of a type (line 140)
        _sections_44376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 36), self_44375, '_sections')
        # Obtaining the member '__getitem__' of a type (line 140)
        getitem___44377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 36), _sections_44376, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 140)
        subscript_call_result_44378 = invoke(stypy.reporting.localization.Localization(__file__, 140, 36), getitem___44377, section_44374)
        
        # Obtaining the member '__getitem__' of a type (line 140)
        getitem___44379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 36), subscript_call_result_44378, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 140)
        subscript_call_result_44380 = invoke(stypy.reporting.localization.Localization(__file__, 140, 36), getitem___44379, str_44373)
        
        # Processing the call keyword arguments (line 140)
        kwargs_44381 = {}
        # Getting the type of 'self' (line 140)
        self_44370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 14), 'self', False)
        # Obtaining the member 'vars' of a type (line 140)
        vars_44371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 14), self_44370, 'vars')
        # Obtaining the member 'interpolate' of a type (line 140)
        interpolate_44372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 14), vars_44371, 'interpolate')
        # Calling interpolate(args, kwargs) (line 140)
        interpolate_call_result_44382 = invoke(stypy.reporting.localization.Localization(__file__, 140, 14), interpolate_44372, *[subscript_call_result_44380], **kwargs_44381)
        
        # Assigning a type to the variable 'val' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'val', interpolate_call_result_44382)
        
        # Call to _escape_backslash(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'val' (line 141)
        val_44384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 33), 'val', False)
        # Processing the call keyword arguments (line 141)
        kwargs_44385 = {}
        # Getting the type of '_escape_backslash' (line 141)
        _escape_backslash_44383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), '_escape_backslash', False)
        # Calling _escape_backslash(args, kwargs) (line 141)
        _escape_backslash_call_result_44386 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), _escape_backslash_44383, *[val_44384], **kwargs_44385)
        
        # Assigning a type to the variable 'stypy_return_type' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'stypy_return_type', _escape_backslash_call_result_44386)
        
        # ################# End of 'libs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'libs' in the type store
        # Getting the type of 'stypy_return_type' (line 139)
        stypy_return_type_44387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44387)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'libs'
        return stypy_return_type_44387


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LibraryInfo.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        LibraryInfo.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LibraryInfo.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LibraryInfo.stypy__str__.__dict__.__setitem__('stypy_function_name', 'LibraryInfo.__str__')
        LibraryInfo.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        LibraryInfo.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LibraryInfo.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LibraryInfo.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LibraryInfo.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LibraryInfo.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LibraryInfo.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LibraryInfo.__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        
        # Assigning a List to a Name (line 144):
        
        # Assigning a List to a Name (line 144):
        
        # Obtaining an instance of the builtin type 'list' (line 144)
        list_44388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 144)
        # Adding element type (line 144)
        str_44389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 13), 'str', 'Name: %s')
        # Getting the type of 'self' (line 144)
        self_44390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 26), 'self')
        # Obtaining the member 'name' of a type (line 144)
        name_44391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 26), self_44390, 'name')
        # Applying the binary operator '%' (line 144)
        result_mod_44392 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 13), '%', str_44389, name_44391)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 12), list_44388, result_mod_44392)
        # Adding element type (line 144)
        str_44393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 37), 'str', 'Description: %s')
        # Getting the type of 'self' (line 144)
        self_44394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 57), 'self')
        # Obtaining the member 'description' of a type (line 144)
        description_44395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 57), self_44394, 'description')
        # Applying the binary operator '%' (line 144)
        result_mod_44396 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 37), '%', str_44393, description_44395)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 12), list_44388, result_mod_44396)
        
        # Assigning a type to the variable 'm' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'm', list_44388)
        
        # Getting the type of 'self' (line 145)
        self_44397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), 'self')
        # Obtaining the member 'requires' of a type (line 145)
        requires_44398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 11), self_44397, 'requires')
        # Testing the type of an if condition (line 145)
        if_condition_44399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 8), requires_44398)
        # Assigning a type to the variable 'if_condition_44399' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'if_condition_44399', if_condition_44399)
        # SSA begins for if statement (line 145)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 146)
        # Processing the call arguments (line 146)
        str_44402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 21), 'str', 'Requires:')
        # Processing the call keyword arguments (line 146)
        kwargs_44403 = {}
        # Getting the type of 'm' (line 146)
        m_44400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'm', False)
        # Obtaining the member 'append' of a type (line 146)
        append_44401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), m_44400, 'append')
        # Calling append(args, kwargs) (line 146)
        append_call_result_44404 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), append_44401, *[str_44402], **kwargs_44403)
        
        # SSA branch for the else part of an if statement (line 145)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 148)
        # Processing the call arguments (line 148)
        str_44407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 21), 'str', 'Requires: %s')
        
        # Call to join(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'self' (line 148)
        self_44410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 47), 'self', False)
        # Obtaining the member 'requires' of a type (line 148)
        requires_44411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 47), self_44410, 'requires')
        # Processing the call keyword arguments (line 148)
        kwargs_44412 = {}
        str_44408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 38), 'str', ',')
        # Obtaining the member 'join' of a type (line 148)
        join_44409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 38), str_44408, 'join')
        # Calling join(args, kwargs) (line 148)
        join_call_result_44413 = invoke(stypy.reporting.localization.Localization(__file__, 148, 38), join_44409, *[requires_44411], **kwargs_44412)
        
        # Applying the binary operator '%' (line 148)
        result_mod_44414 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 21), '%', str_44407, join_call_result_44413)
        
        # Processing the call keyword arguments (line 148)
        kwargs_44415 = {}
        # Getting the type of 'm' (line 148)
        m_44405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'm', False)
        # Obtaining the member 'append' of a type (line 148)
        append_44406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), m_44405, 'append')
        # Calling append(args, kwargs) (line 148)
        append_call_result_44416 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), append_44406, *[result_mod_44414], **kwargs_44415)
        
        # SSA join for if statement (line 145)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 149)
        # Processing the call arguments (line 149)
        str_44419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 17), 'str', 'Version: %s')
        # Getting the type of 'self' (line 149)
        self_44420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 33), 'self', False)
        # Obtaining the member 'version' of a type (line 149)
        version_44421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 33), self_44420, 'version')
        # Applying the binary operator '%' (line 149)
        result_mod_44422 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 17), '%', str_44419, version_44421)
        
        # Processing the call keyword arguments (line 149)
        kwargs_44423 = {}
        # Getting the type of 'm' (line 149)
        m_44417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'm', False)
        # Obtaining the member 'append' of a type (line 149)
        append_44418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), m_44417, 'append')
        # Calling append(args, kwargs) (line 149)
        append_call_result_44424 = invoke(stypy.reporting.localization.Localization(__file__, 149, 8), append_44418, *[result_mod_44422], **kwargs_44423)
        
        
        # Call to join(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'm' (line 151)
        m_44427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 25), 'm', False)
        # Processing the call keyword arguments (line 151)
        kwargs_44428 = {}
        str_44425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 15), 'str', '\n')
        # Obtaining the member 'join' of a type (line 151)
        join_44426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 15), str_44425, 'join')
        # Calling join(args, kwargs) (line 151)
        join_call_result_44429 = invoke(stypy.reporting.localization.Localization(__file__, 151, 15), join_44426, *[m_44427], **kwargs_44428)
        
        # Assigning a type to the variable 'stypy_return_type' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'stypy_return_type', join_call_result_44429)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_44430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44430)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_44430


# Assigning a type to the variable 'LibraryInfo' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'LibraryInfo', LibraryInfo)
# Declaration of the 'VariableSet' class

class VariableSet(object, ):
    str_44431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, (-1)), 'str', '\n    Container object for the variables defined in a config file.\n\n    `VariableSet` can be used as a plain dictionary, with the variable names\n    as keys.\n\n    Parameters\n    ----------\n    d : dict\n        Dict of items in the "variables" section of the configuration file.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 166, 4, False)
        # Assigning a type to the variable 'self' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VariableSet.__init__', ['d'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['d'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 167):
        
        # Assigning a Call to a Attribute (line 167):
        
        # Call to dict(...): (line 167)
        # Processing the call arguments (line 167)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to items(...): (line 167)
        # Processing the call keyword arguments (line 167)
        kwargs_44438 = {}
        # Getting the type of 'd' (line 167)
        d_44436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 50), 'd', False)
        # Obtaining the member 'items' of a type (line 167)
        items_44437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 50), d_44436, 'items')
        # Calling items(args, kwargs) (line 167)
        items_call_result_44439 = invoke(stypy.reporting.localization.Localization(__file__, 167, 50), items_44437, *[], **kwargs_44438)
        
        comprehension_44440 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 31), items_call_result_44439)
        # Assigning a type to the variable 'k' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 31), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 31), comprehension_44440))
        # Assigning a type to the variable 'v' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 31), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 31), comprehension_44440))
        
        # Obtaining an instance of the builtin type 'tuple' (line 167)
        tuple_44433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 167)
        # Adding element type (line 167)
        # Getting the type of 'k' (line 167)
        k_44434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 32), 'k', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 32), tuple_44433, k_44434)
        # Adding element type (line 167)
        # Getting the type of 'v' (line 167)
        v_44435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 35), 'v', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 32), tuple_44433, v_44435)
        
        list_44441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 31), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 31), list_44441, tuple_44433)
        # Processing the call keyword arguments (line 167)
        kwargs_44442 = {}
        # Getting the type of 'dict' (line 167)
        dict_44432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), 'dict', False)
        # Calling dict(args, kwargs) (line 167)
        dict_call_result_44443 = invoke(stypy.reporting.localization.Localization(__file__, 167, 25), dict_44432, *[list_44441], **kwargs_44442)
        
        # Getting the type of 'self' (line 167)
        self_44444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'self')
        # Setting the type of the member '_raw_data' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), self_44444, '_raw_data', dict_call_result_44443)
        
        # Assigning a Dict to a Attribute (line 169):
        
        # Assigning a Dict to a Attribute (line 169):
        
        # Obtaining an instance of the builtin type 'dict' (line 169)
        dict_44445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 169)
        
        # Getting the type of 'self' (line 169)
        self_44446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'self')
        # Setting the type of the member '_re' of a type (line 169)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), self_44446, '_re', dict_44445)
        
        # Assigning a Dict to a Attribute (line 170):
        
        # Assigning a Dict to a Attribute (line 170):
        
        # Obtaining an instance of the builtin type 'dict' (line 170)
        dict_44447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 23), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 170)
        
        # Getting the type of 'self' (line 170)
        self_44448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'self')
        # Setting the type of the member '_re_sub' of a type (line 170)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), self_44448, '_re_sub', dict_44447)
        
        # Call to _init_parse(...): (line 172)
        # Processing the call keyword arguments (line 172)
        kwargs_44451 = {}
        # Getting the type of 'self' (line 172)
        self_44449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'self', False)
        # Obtaining the member '_init_parse' of a type (line 172)
        _init_parse_44450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), self_44449, '_init_parse')
        # Calling _init_parse(args, kwargs) (line 172)
        _init_parse_call_result_44452 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), _init_parse_44450, *[], **kwargs_44451)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _init_parse(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_init_parse'
        module_type_store = module_type_store.open_function_context('_init_parse', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VariableSet._init_parse.__dict__.__setitem__('stypy_localization', localization)
        VariableSet._init_parse.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VariableSet._init_parse.__dict__.__setitem__('stypy_type_store', module_type_store)
        VariableSet._init_parse.__dict__.__setitem__('stypy_function_name', 'VariableSet._init_parse')
        VariableSet._init_parse.__dict__.__setitem__('stypy_param_names_list', [])
        VariableSet._init_parse.__dict__.__setitem__('stypy_varargs_param_name', None)
        VariableSet._init_parse.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VariableSet._init_parse.__dict__.__setitem__('stypy_call_defaults', defaults)
        VariableSet._init_parse.__dict__.__setitem__('stypy_call_varargs', varargs)
        VariableSet._init_parse.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VariableSet._init_parse.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VariableSet._init_parse', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_init_parse', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_init_parse(...)' code ##################

        
        
        # Call to items(...): (line 175)
        # Processing the call keyword arguments (line 175)
        kwargs_44456 = {}
        # Getting the type of 'self' (line 175)
        self_44453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 20), 'self', False)
        # Obtaining the member '_raw_data' of a type (line 175)
        _raw_data_44454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 20), self_44453, '_raw_data')
        # Obtaining the member 'items' of a type (line 175)
        items_44455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 20), _raw_data_44454, 'items')
        # Calling items(args, kwargs) (line 175)
        items_call_result_44457 = invoke(stypy.reporting.localization.Localization(__file__, 175, 20), items_44455, *[], **kwargs_44456)
        
        # Testing the type of a for loop iterable (line 175)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 175, 8), items_call_result_44457)
        # Getting the type of the for loop variable (line 175)
        for_loop_var_44458 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 175, 8), items_call_result_44457)
        # Assigning a type to the variable 'k' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 8), for_loop_var_44458))
        # Assigning a type to the variable 'v' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 8), for_loop_var_44458))
        # SSA begins for a for statement (line 175)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to _init_parse_var(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'k' (line 176)
        k_44461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 33), 'k', False)
        # Getting the type of 'v' (line 176)
        v_44462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'v', False)
        # Processing the call keyword arguments (line 176)
        kwargs_44463 = {}
        # Getting the type of 'self' (line 176)
        self_44459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'self', False)
        # Obtaining the member '_init_parse_var' of a type (line 176)
        _init_parse_var_44460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), self_44459, '_init_parse_var')
        # Calling _init_parse_var(args, kwargs) (line 176)
        _init_parse_var_call_result_44464 = invoke(stypy.reporting.localization.Localization(__file__, 176, 12), _init_parse_var_44460, *[k_44461, v_44462], **kwargs_44463)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_init_parse(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_init_parse' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_44465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44465)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_init_parse'
        return stypy_return_type_44465


    @norecursion
    def _init_parse_var(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_init_parse_var'
        module_type_store = module_type_store.open_function_context('_init_parse_var', 178, 4, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VariableSet._init_parse_var.__dict__.__setitem__('stypy_localization', localization)
        VariableSet._init_parse_var.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VariableSet._init_parse_var.__dict__.__setitem__('stypy_type_store', module_type_store)
        VariableSet._init_parse_var.__dict__.__setitem__('stypy_function_name', 'VariableSet._init_parse_var')
        VariableSet._init_parse_var.__dict__.__setitem__('stypy_param_names_list', ['name', 'value'])
        VariableSet._init_parse_var.__dict__.__setitem__('stypy_varargs_param_name', None)
        VariableSet._init_parse_var.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VariableSet._init_parse_var.__dict__.__setitem__('stypy_call_defaults', defaults)
        VariableSet._init_parse_var.__dict__.__setitem__('stypy_call_varargs', varargs)
        VariableSet._init_parse_var.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VariableSet._init_parse_var.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VariableSet._init_parse_var', ['name', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_init_parse_var', localization, ['name', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_init_parse_var(...)' code ##################

        
        # Assigning a Call to a Subscript (line 179):
        
        # Assigning a Call to a Subscript (line 179):
        
        # Call to compile(...): (line 179)
        # Processing the call arguments (line 179)
        str_44468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 36), 'str', '\\$\\{%s\\}')
        # Getting the type of 'name' (line 179)
        name_44469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 50), 'name', False)
        # Applying the binary operator '%' (line 179)
        result_mod_44470 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 36), '%', str_44468, name_44469)
        
        # Processing the call keyword arguments (line 179)
        kwargs_44471 = {}
        # Getting the type of 're' (line 179)
        re_44466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 25), 're', False)
        # Obtaining the member 'compile' of a type (line 179)
        compile_44467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 25), re_44466, 'compile')
        # Calling compile(args, kwargs) (line 179)
        compile_call_result_44472 = invoke(stypy.reporting.localization.Localization(__file__, 179, 25), compile_44467, *[result_mod_44470], **kwargs_44471)
        
        # Getting the type of 'self' (line 179)
        self_44473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'self')
        # Obtaining the member '_re' of a type (line 179)
        _re_44474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), self_44473, '_re')
        # Getting the type of 'name' (line 179)
        name_44475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 17), 'name')
        # Storing an element on a container (line 179)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 8), _re_44474, (name_44475, compile_call_result_44472))
        
        # Assigning a Name to a Subscript (line 180):
        
        # Assigning a Name to a Subscript (line 180):
        # Getting the type of 'value' (line 180)
        value_44476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 29), 'value')
        # Getting the type of 'self' (line 180)
        self_44477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'self')
        # Obtaining the member '_re_sub' of a type (line 180)
        _re_sub_44478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), self_44477, '_re_sub')
        # Getting the type of 'name' (line 180)
        name_44479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 21), 'name')
        # Storing an element on a container (line 180)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 8), _re_sub_44478, (name_44479, value_44476))
        
        # ################# End of '_init_parse_var(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_init_parse_var' in the type store
        # Getting the type of 'stypy_return_type' (line 178)
        stypy_return_type_44480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44480)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_init_parse_var'
        return stypy_return_type_44480


    @norecursion
    def interpolate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'interpolate'
        module_type_store = module_type_store.open_function_context('interpolate', 182, 4, False)
        # Assigning a type to the variable 'self' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VariableSet.interpolate.__dict__.__setitem__('stypy_localization', localization)
        VariableSet.interpolate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VariableSet.interpolate.__dict__.__setitem__('stypy_type_store', module_type_store)
        VariableSet.interpolate.__dict__.__setitem__('stypy_function_name', 'VariableSet.interpolate')
        VariableSet.interpolate.__dict__.__setitem__('stypy_param_names_list', ['value'])
        VariableSet.interpolate.__dict__.__setitem__('stypy_varargs_param_name', None)
        VariableSet.interpolate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VariableSet.interpolate.__dict__.__setitem__('stypy_call_defaults', defaults)
        VariableSet.interpolate.__dict__.__setitem__('stypy_call_varargs', varargs)
        VariableSet.interpolate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VariableSet.interpolate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VariableSet.interpolate', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'interpolate', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'interpolate(...)' code ##################


        @norecursion
        def _interpolate(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_interpolate'
            module_type_store = module_type_store.open_function_context('_interpolate', 185, 8, False)
            
            # Passed parameters checking function
            _interpolate.stypy_localization = localization
            _interpolate.stypy_type_of_self = None
            _interpolate.stypy_type_store = module_type_store
            _interpolate.stypy_function_name = '_interpolate'
            _interpolate.stypy_param_names_list = ['value']
            _interpolate.stypy_varargs_param_name = None
            _interpolate.stypy_kwargs_param_name = None
            _interpolate.stypy_call_defaults = defaults
            _interpolate.stypy_call_varargs = varargs
            _interpolate.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_interpolate', ['value'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_interpolate', localization, ['value'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_interpolate(...)' code ##################

            
            
            # Call to keys(...): (line 186)
            # Processing the call keyword arguments (line 186)
            kwargs_44484 = {}
            # Getting the type of 'self' (line 186)
            self_44481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 21), 'self', False)
            # Obtaining the member '_re' of a type (line 186)
            _re_44482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 21), self_44481, '_re')
            # Obtaining the member 'keys' of a type (line 186)
            keys_44483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 21), _re_44482, 'keys')
            # Calling keys(args, kwargs) (line 186)
            keys_call_result_44485 = invoke(stypy.reporting.localization.Localization(__file__, 186, 21), keys_44483, *[], **kwargs_44484)
            
            # Testing the type of a for loop iterable (line 186)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 186, 12), keys_call_result_44485)
            # Getting the type of the for loop variable (line 186)
            for_loop_var_44486 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 186, 12), keys_call_result_44485)
            # Assigning a type to the variable 'k' (line 186)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'k', for_loop_var_44486)
            # SSA begins for a for statement (line 186)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 187):
            
            # Assigning a Call to a Name (line 187):
            
            # Call to sub(...): (line 187)
            # Processing the call arguments (line 187)
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 187)
            k_44493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 53), 'k', False)
            # Getting the type of 'self' (line 187)
            self_44494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 40), 'self', False)
            # Obtaining the member '_re_sub' of a type (line 187)
            _re_sub_44495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 40), self_44494, '_re_sub')
            # Obtaining the member '__getitem__' of a type (line 187)
            getitem___44496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 40), _re_sub_44495, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 187)
            subscript_call_result_44497 = invoke(stypy.reporting.localization.Localization(__file__, 187, 40), getitem___44496, k_44493)
            
            # Getting the type of 'value' (line 187)
            value_44498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 57), 'value', False)
            # Processing the call keyword arguments (line 187)
            kwargs_44499 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 187)
            k_44487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 33), 'k', False)
            # Getting the type of 'self' (line 187)
            self_44488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 24), 'self', False)
            # Obtaining the member '_re' of a type (line 187)
            _re_44489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 24), self_44488, '_re')
            # Obtaining the member '__getitem__' of a type (line 187)
            getitem___44490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 24), _re_44489, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 187)
            subscript_call_result_44491 = invoke(stypy.reporting.localization.Localization(__file__, 187, 24), getitem___44490, k_44487)
            
            # Obtaining the member 'sub' of a type (line 187)
            sub_44492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 24), subscript_call_result_44491, 'sub')
            # Calling sub(args, kwargs) (line 187)
            sub_call_result_44500 = invoke(stypy.reporting.localization.Localization(__file__, 187, 24), sub_44492, *[subscript_call_result_44497, value_44498], **kwargs_44499)
            
            # Assigning a type to the variable 'value' (line 187)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'value', sub_call_result_44500)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'value' (line 188)
            value_44501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 19), 'value')
            # Assigning a type to the variable 'stypy_return_type' (line 188)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'stypy_return_type', value_44501)
            
            # ################# End of '_interpolate(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_interpolate' in the type store
            # Getting the type of 'stypy_return_type' (line 185)
            stypy_return_type_44502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_44502)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_interpolate'
            return stypy_return_type_44502

        # Assigning a type to the variable '_interpolate' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), '_interpolate', _interpolate)
        
        
        # Call to search(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'value' (line 189)
        value_44505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 26), 'value', False)
        # Processing the call keyword arguments (line 189)
        kwargs_44506 = {}
        # Getting the type of '_VAR' (line 189)
        _VAR_44503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 14), '_VAR', False)
        # Obtaining the member 'search' of a type (line 189)
        search_44504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 14), _VAR_44503, 'search')
        # Calling search(args, kwargs) (line 189)
        search_call_result_44507 = invoke(stypy.reporting.localization.Localization(__file__, 189, 14), search_44504, *[value_44505], **kwargs_44506)
        
        # Testing the type of an if condition (line 189)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 8), search_call_result_44507)
        # SSA begins for while statement (line 189)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 190):
        
        # Assigning a Call to a Name (line 190):
        
        # Call to _interpolate(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'value' (line 190)
        value_44509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 34), 'value', False)
        # Processing the call keyword arguments (line 190)
        kwargs_44510 = {}
        # Getting the type of '_interpolate' (line 190)
        _interpolate_44508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 21), '_interpolate', False)
        # Calling _interpolate(args, kwargs) (line 190)
        _interpolate_call_result_44511 = invoke(stypy.reporting.localization.Localization(__file__, 190, 21), _interpolate_44508, *[value_44509], **kwargs_44510)
        
        # Assigning a type to the variable 'nvalue' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'nvalue', _interpolate_call_result_44511)
        
        
        # Getting the type of 'nvalue' (line 191)
        nvalue_44512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 15), 'nvalue')
        # Getting the type of 'value' (line 191)
        value_44513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 25), 'value')
        # Applying the binary operator '==' (line 191)
        result_eq_44514 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 15), '==', nvalue_44512, value_44513)
        
        # Testing the type of an if condition (line 191)
        if_condition_44515 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 12), result_eq_44514)
        # Assigning a type to the variable 'if_condition_44515' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'if_condition_44515', if_condition_44515)
        # SSA begins for if statement (line 191)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 191)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 193):
        
        # Assigning a Name to a Name (line 193):
        # Getting the type of 'nvalue' (line 193)
        nvalue_44516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 20), 'nvalue')
        # Assigning a type to the variable 'value' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'value', nvalue_44516)
        # SSA join for while statement (line 189)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'value' (line 195)
        value_44517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'stypy_return_type', value_44517)
        
        # ################# End of 'interpolate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'interpolate' in the type store
        # Getting the type of 'stypy_return_type' (line 182)
        stypy_return_type_44518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44518)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'interpolate'
        return stypy_return_type_44518


    @norecursion
    def variables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'variables'
        module_type_store = module_type_store.open_function_context('variables', 197, 4, False)
        # Assigning a type to the variable 'self' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VariableSet.variables.__dict__.__setitem__('stypy_localization', localization)
        VariableSet.variables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VariableSet.variables.__dict__.__setitem__('stypy_type_store', module_type_store)
        VariableSet.variables.__dict__.__setitem__('stypy_function_name', 'VariableSet.variables')
        VariableSet.variables.__dict__.__setitem__('stypy_param_names_list', [])
        VariableSet.variables.__dict__.__setitem__('stypy_varargs_param_name', None)
        VariableSet.variables.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VariableSet.variables.__dict__.__setitem__('stypy_call_defaults', defaults)
        VariableSet.variables.__dict__.__setitem__('stypy_call_varargs', varargs)
        VariableSet.variables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VariableSet.variables.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VariableSet.variables', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'variables', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'variables(...)' code ##################

        str_44519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, (-1)), 'str', '\n        Return the list of variable names.\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        names : list of str\n            The names of all variables in the `VariableSet` instance.\n\n        ')
        
        # Call to list(...): (line 211)
        # Processing the call arguments (line 211)
        
        # Call to keys(...): (line 211)
        # Processing the call keyword arguments (line 211)
        kwargs_44524 = {}
        # Getting the type of 'self' (line 211)
        self_44521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'self', False)
        # Obtaining the member '_raw_data' of a type (line 211)
        _raw_data_44522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 20), self_44521, '_raw_data')
        # Obtaining the member 'keys' of a type (line 211)
        keys_44523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 20), _raw_data_44522, 'keys')
        # Calling keys(args, kwargs) (line 211)
        keys_call_result_44525 = invoke(stypy.reporting.localization.Localization(__file__, 211, 20), keys_44523, *[], **kwargs_44524)
        
        # Processing the call keyword arguments (line 211)
        kwargs_44526 = {}
        # Getting the type of 'list' (line 211)
        list_44520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 15), 'list', False)
        # Calling list(args, kwargs) (line 211)
        list_call_result_44527 = invoke(stypy.reporting.localization.Localization(__file__, 211, 15), list_44520, *[keys_call_result_44525], **kwargs_44526)
        
        # Assigning a type to the variable 'stypy_return_type' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'stypy_return_type', list_call_result_44527)
        
        # ################# End of 'variables(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'variables' in the type store
        # Getting the type of 'stypy_return_type' (line 197)
        stypy_return_type_44528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44528)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'variables'
        return stypy_return_type_44528


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 214, 4, False)
        # Assigning a type to the variable 'self' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VariableSet.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        VariableSet.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VariableSet.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        VariableSet.__getitem__.__dict__.__setitem__('stypy_function_name', 'VariableSet.__getitem__')
        VariableSet.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['name'])
        VariableSet.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        VariableSet.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VariableSet.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        VariableSet.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        VariableSet.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VariableSet.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VariableSet.__getitem__', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 215)
        name_44529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 30), 'name')
        # Getting the type of 'self' (line 215)
        self_44530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 15), 'self')
        # Obtaining the member '_raw_data' of a type (line 215)
        _raw_data_44531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 15), self_44530, '_raw_data')
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___44532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 15), _raw_data_44531, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_44533 = invoke(stypy.reporting.localization.Localization(__file__, 215, 15), getitem___44532, name_44529)
        
        # Assigning a type to the variable 'stypy_return_type' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'stypy_return_type', subscript_call_result_44533)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 214)
        stypy_return_type_44534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44534)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_44534


    @norecursion
    def __setitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setitem__'
        module_type_store = module_type_store.open_function_context('__setitem__', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VariableSet.__setitem__.__dict__.__setitem__('stypy_localization', localization)
        VariableSet.__setitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VariableSet.__setitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        VariableSet.__setitem__.__dict__.__setitem__('stypy_function_name', 'VariableSet.__setitem__')
        VariableSet.__setitem__.__dict__.__setitem__('stypy_param_names_list', ['name', 'value'])
        VariableSet.__setitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        VariableSet.__setitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VariableSet.__setitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        VariableSet.__setitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        VariableSet.__setitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VariableSet.__setitem__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VariableSet.__setitem__', ['name', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setitem__', localization, ['name', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setitem__(...)' code ##################

        
        # Assigning a Name to a Subscript (line 218):
        
        # Assigning a Name to a Subscript (line 218):
        # Getting the type of 'value' (line 218)
        value_44535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 31), 'value')
        # Getting the type of 'self' (line 218)
        self_44536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'self')
        # Obtaining the member '_raw_data' of a type (line 218)
        _raw_data_44537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 8), self_44536, '_raw_data')
        # Getting the type of 'name' (line 218)
        name_44538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 23), 'name')
        # Storing an element on a container (line 218)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 8), _raw_data_44537, (name_44538, value_44535))
        
        # Call to _init_parse_var(...): (line 219)
        # Processing the call arguments (line 219)
        # Getting the type of 'name' (line 219)
        name_44541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 29), 'name', False)
        # Getting the type of 'value' (line 219)
        value_44542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 35), 'value', False)
        # Processing the call keyword arguments (line 219)
        kwargs_44543 = {}
        # Getting the type of 'self' (line 219)
        self_44539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'self', False)
        # Obtaining the member '_init_parse_var' of a type (line 219)
        _init_parse_var_44540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), self_44539, '_init_parse_var')
        # Calling _init_parse_var(args, kwargs) (line 219)
        _init_parse_var_call_result_44544 = invoke(stypy.reporting.localization.Localization(__file__, 219, 8), _init_parse_var_44540, *[name_44541, value_44542], **kwargs_44543)
        
        
        # ################# End of '__setitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 217)
        stypy_return_type_44545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44545)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setitem__'
        return stypy_return_type_44545


# Assigning a type to the variable 'VariableSet' (line 153)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 0), 'VariableSet', VariableSet)

@norecursion
def parse_meta(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'parse_meta'
    module_type_store = module_type_store.open_function_context('parse_meta', 221, 0, False)
    
    # Passed parameters checking function
    parse_meta.stypy_localization = localization
    parse_meta.stypy_type_of_self = None
    parse_meta.stypy_type_store = module_type_store
    parse_meta.stypy_function_name = 'parse_meta'
    parse_meta.stypy_param_names_list = ['config']
    parse_meta.stypy_varargs_param_name = None
    parse_meta.stypy_kwargs_param_name = None
    parse_meta.stypy_call_defaults = defaults
    parse_meta.stypy_call_varargs = varargs
    parse_meta.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_meta', ['config'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_meta', localization, ['config'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_meta(...)' code ##################

    
    
    
    # Call to has_section(...): (line 222)
    # Processing the call arguments (line 222)
    str_44548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 30), 'str', 'meta')
    # Processing the call keyword arguments (line 222)
    kwargs_44549 = {}
    # Getting the type of 'config' (line 222)
    config_44546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 11), 'config', False)
    # Obtaining the member 'has_section' of a type (line 222)
    has_section_44547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 11), config_44546, 'has_section')
    # Calling has_section(args, kwargs) (line 222)
    has_section_call_result_44550 = invoke(stypy.reporting.localization.Localization(__file__, 222, 11), has_section_44547, *[str_44548], **kwargs_44549)
    
    # Applying the 'not' unary operator (line 222)
    result_not__44551 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 7), 'not', has_section_call_result_44550)
    
    # Testing the type of an if condition (line 222)
    if_condition_44552 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 222, 4), result_not__44551)
    # Assigning a type to the variable 'if_condition_44552' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'if_condition_44552', if_condition_44552)
    # SSA begins for if statement (line 222)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to FormatError(...): (line 223)
    # Processing the call arguments (line 223)
    str_44554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 26), 'str', 'No meta section found !')
    # Processing the call keyword arguments (line 223)
    kwargs_44555 = {}
    # Getting the type of 'FormatError' (line 223)
    FormatError_44553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 14), 'FormatError', False)
    # Calling FormatError(args, kwargs) (line 223)
    FormatError_call_result_44556 = invoke(stypy.reporting.localization.Localization(__file__, 223, 14), FormatError_44553, *[str_44554], **kwargs_44555)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 223, 8), FormatError_call_result_44556, 'raise parameter', BaseException)
    # SSA join for if statement (line 222)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 225):
    
    # Assigning a Dict to a Name (line 225):
    
    # Obtaining an instance of the builtin type 'dict' (line 225)
    dict_44557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 225)
    
    # Assigning a type to the variable 'd' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'd', dict_44557)
    
    
    # Call to items(...): (line 226)
    # Processing the call arguments (line 226)
    str_44560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 36), 'str', 'meta')
    # Processing the call keyword arguments (line 226)
    kwargs_44561 = {}
    # Getting the type of 'config' (line 226)
    config_44558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), 'config', False)
    # Obtaining the member 'items' of a type (line 226)
    items_44559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 23), config_44558, 'items')
    # Calling items(args, kwargs) (line 226)
    items_call_result_44562 = invoke(stypy.reporting.localization.Localization(__file__, 226, 23), items_44559, *[str_44560], **kwargs_44561)
    
    # Testing the type of a for loop iterable (line 226)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 226, 4), items_call_result_44562)
    # Getting the type of the for loop variable (line 226)
    for_loop_var_44563 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 226, 4), items_call_result_44562)
    # Assigning a type to the variable 'name' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 4), for_loop_var_44563))
    # Assigning a type to the variable 'value' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 4), for_loop_var_44563))
    # SSA begins for a for statement (line 226)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Subscript (line 227):
    
    # Assigning a Name to a Subscript (line 227):
    # Getting the type of 'value' (line 227)
    value_44564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 18), 'value')
    # Getting the type of 'd' (line 227)
    d_44565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'd')
    # Getting the type of 'name' (line 227)
    name_44566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 10), 'name')
    # Storing an element on a container (line 227)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 8), d_44565, (name_44566, value_44564))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining an instance of the builtin type 'list' (line 229)
    list_44567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 229)
    # Adding element type (line 229)
    str_44568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 14), 'str', 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 13), list_44567, str_44568)
    # Adding element type (line 229)
    str_44569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 22), 'str', 'description')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 13), list_44567, str_44569)
    # Adding element type (line 229)
    str_44570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 37), 'str', 'version')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 13), list_44567, str_44570)
    
    # Testing the type of a for loop iterable (line 229)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 229, 4), list_44567)
    # Getting the type of the for loop variable (line 229)
    for_loop_var_44571 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 229, 4), list_44567)
    # Assigning a type to the variable 'k' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'k', for_loop_var_44571)
    # SSA begins for a for statement (line 229)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Getting the type of 'k' (line 230)
    k_44572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'k')
    # Getting the type of 'd' (line 230)
    d_44573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'd')
    # Applying the binary operator 'in' (line 230)
    result_contains_44574 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 15), 'in', k_44572, d_44573)
    
    # Applying the 'not' unary operator (line 230)
    result_not__44575 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 11), 'not', result_contains_44574)
    
    # Testing the type of an if condition (line 230)
    if_condition_44576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 8), result_not__44575)
    # Assigning a type to the variable 'if_condition_44576' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'if_condition_44576', if_condition_44576)
    # SSA begins for if statement (line 230)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to FormatError(...): (line 231)
    # Processing the call arguments (line 231)
    str_44578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 30), 'str', 'Option %s (section [meta]) is mandatory, but not found')
    # Getting the type of 'k' (line 232)
    k_44579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 34), 'k', False)
    # Applying the binary operator '%' (line 231)
    result_mod_44580 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 30), '%', str_44578, k_44579)
    
    # Processing the call keyword arguments (line 231)
    kwargs_44581 = {}
    # Getting the type of 'FormatError' (line 231)
    FormatError_44577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 18), 'FormatError', False)
    # Calling FormatError(args, kwargs) (line 231)
    FormatError_call_result_44582 = invoke(stypy.reporting.localization.Localization(__file__, 231, 18), FormatError_44577, *[result_mod_44580], **kwargs_44581)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 231, 12), FormatError_call_result_44582, 'raise parameter', BaseException)
    # SSA join for if statement (line 230)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    str_44583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 11), 'str', 'requires')
    # Getting the type of 'd' (line 234)
    d_44584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 25), 'd')
    # Applying the binary operator 'in' (line 234)
    result_contains_44585 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 11), 'in', str_44583, d_44584)
    
    # Applying the 'not' unary operator (line 234)
    result_not__44586 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 7), 'not', result_contains_44585)
    
    # Testing the type of an if condition (line 234)
    if_condition_44587 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 234, 4), result_not__44586)
    # Assigning a type to the variable 'if_condition_44587' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'if_condition_44587', if_condition_44587)
    # SSA begins for if statement (line 234)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Subscript (line 235):
    
    # Assigning a List to a Subscript (line 235):
    
    # Obtaining an instance of the builtin type 'list' (line 235)
    list_44588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 235)
    
    # Getting the type of 'd' (line 235)
    d_44589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'd')
    str_44590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 10), 'str', 'requires')
    # Storing an element on a container (line 235)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 235, 8), d_44589, (str_44590, list_44588))
    # SSA join for if statement (line 234)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'd' (line 237)
    d_44591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 11), 'd')
    # Assigning a type to the variable 'stypy_return_type' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'stypy_return_type', d_44591)
    
    # ################# End of 'parse_meta(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_meta' in the type store
    # Getting the type of 'stypy_return_type' (line 221)
    stypy_return_type_44592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44592)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_meta'
    return stypy_return_type_44592

# Assigning a type to the variable 'parse_meta' (line 221)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 0), 'parse_meta', parse_meta)

@norecursion
def parse_variables(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'parse_variables'
    module_type_store = module_type_store.open_function_context('parse_variables', 239, 0, False)
    
    # Passed parameters checking function
    parse_variables.stypy_localization = localization
    parse_variables.stypy_type_of_self = None
    parse_variables.stypy_type_store = module_type_store
    parse_variables.stypy_function_name = 'parse_variables'
    parse_variables.stypy_param_names_list = ['config']
    parse_variables.stypy_varargs_param_name = None
    parse_variables.stypy_kwargs_param_name = None
    parse_variables.stypy_call_defaults = defaults
    parse_variables.stypy_call_varargs = varargs
    parse_variables.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_variables', ['config'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_variables', localization, ['config'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_variables(...)' code ##################

    
    
    
    # Call to has_section(...): (line 240)
    # Processing the call arguments (line 240)
    str_44595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 30), 'str', 'variables')
    # Processing the call keyword arguments (line 240)
    kwargs_44596 = {}
    # Getting the type of 'config' (line 240)
    config_44593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'config', False)
    # Obtaining the member 'has_section' of a type (line 240)
    has_section_44594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 11), config_44593, 'has_section')
    # Calling has_section(args, kwargs) (line 240)
    has_section_call_result_44597 = invoke(stypy.reporting.localization.Localization(__file__, 240, 11), has_section_44594, *[str_44595], **kwargs_44596)
    
    # Applying the 'not' unary operator (line 240)
    result_not__44598 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 7), 'not', has_section_call_result_44597)
    
    # Testing the type of an if condition (line 240)
    if_condition_44599 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 4), result_not__44598)
    # Assigning a type to the variable 'if_condition_44599' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'if_condition_44599', if_condition_44599)
    # SSA begins for if statement (line 240)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to FormatError(...): (line 241)
    # Processing the call arguments (line 241)
    str_44601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 26), 'str', 'No variables section found !')
    # Processing the call keyword arguments (line 241)
    kwargs_44602 = {}
    # Getting the type of 'FormatError' (line 241)
    FormatError_44600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 14), 'FormatError', False)
    # Calling FormatError(args, kwargs) (line 241)
    FormatError_call_result_44603 = invoke(stypy.reporting.localization.Localization(__file__, 241, 14), FormatError_44600, *[str_44601], **kwargs_44602)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 241, 8), FormatError_call_result_44603, 'raise parameter', BaseException)
    # SSA join for if statement (line 240)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 243):
    
    # Assigning a Dict to a Name (line 243):
    
    # Obtaining an instance of the builtin type 'dict' (line 243)
    dict_44604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 243)
    
    # Assigning a type to the variable 'd' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'd', dict_44604)
    
    
    # Call to items(...): (line 245)
    # Processing the call arguments (line 245)
    str_44607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 36), 'str', 'variables')
    # Processing the call keyword arguments (line 245)
    kwargs_44608 = {}
    # Getting the type of 'config' (line 245)
    config_44605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 23), 'config', False)
    # Obtaining the member 'items' of a type (line 245)
    items_44606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 23), config_44605, 'items')
    # Calling items(args, kwargs) (line 245)
    items_call_result_44609 = invoke(stypy.reporting.localization.Localization(__file__, 245, 23), items_44606, *[str_44607], **kwargs_44608)
    
    # Testing the type of a for loop iterable (line 245)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 245, 4), items_call_result_44609)
    # Getting the type of the for loop variable (line 245)
    for_loop_var_44610 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 245, 4), items_call_result_44609)
    # Assigning a type to the variable 'name' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 4), for_loop_var_44610))
    # Assigning a type to the variable 'value' (line 245)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 4), for_loop_var_44610))
    # SSA begins for a for statement (line 245)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Subscript (line 246):
    
    # Assigning a Name to a Subscript (line 246):
    # Getting the type of 'value' (line 246)
    value_44611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 18), 'value')
    # Getting the type of 'd' (line 246)
    d_44612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'd')
    # Getting the type of 'name' (line 246)
    name_44613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 10), 'name')
    # Storing an element on a container (line 246)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 8), d_44612, (name_44613, value_44611))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to VariableSet(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'd' (line 248)
    d_44615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 23), 'd', False)
    # Processing the call keyword arguments (line 248)
    kwargs_44616 = {}
    # Getting the type of 'VariableSet' (line 248)
    VariableSet_44614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 11), 'VariableSet', False)
    # Calling VariableSet(args, kwargs) (line 248)
    VariableSet_call_result_44617 = invoke(stypy.reporting.localization.Localization(__file__, 248, 11), VariableSet_44614, *[d_44615], **kwargs_44616)
    
    # Assigning a type to the variable 'stypy_return_type' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'stypy_return_type', VariableSet_call_result_44617)
    
    # ################# End of 'parse_variables(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_variables' in the type store
    # Getting the type of 'stypy_return_type' (line 239)
    stypy_return_type_44618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44618)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_variables'
    return stypy_return_type_44618

# Assigning a type to the variable 'parse_variables' (line 239)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 0), 'parse_variables', parse_variables)

@norecursion
def parse_sections(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'parse_sections'
    module_type_store = module_type_store.open_function_context('parse_sections', 250, 0, False)
    
    # Passed parameters checking function
    parse_sections.stypy_localization = localization
    parse_sections.stypy_type_of_self = None
    parse_sections.stypy_type_store = module_type_store
    parse_sections.stypy_function_name = 'parse_sections'
    parse_sections.stypy_param_names_list = ['config']
    parse_sections.stypy_varargs_param_name = None
    parse_sections.stypy_kwargs_param_name = None
    parse_sections.stypy_call_defaults = defaults
    parse_sections.stypy_call_varargs = varargs
    parse_sections.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_sections', ['config'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_sections', localization, ['config'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_sections(...)' code ##################

    
    # Obtaining an instance of the builtin type 'tuple' (line 251)
    tuple_44619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 251)
    # Adding element type (line 251)
    # Getting the type of 'meta_d' (line 251)
    meta_d_44620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 'meta_d')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 11), tuple_44619, meta_d_44620)
    # Adding element type (line 251)
    # Getting the type of 'r' (line 251)
    r_44621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 19), 'r')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 11), tuple_44619, r_44621)
    
    # Assigning a type to the variable 'stypy_return_type' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'stypy_return_type', tuple_44619)
    
    # ################# End of 'parse_sections(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_sections' in the type store
    # Getting the type of 'stypy_return_type' (line 250)
    stypy_return_type_44622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44622)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_sections'
    return stypy_return_type_44622

# Assigning a type to the variable 'parse_sections' (line 250)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 0), 'parse_sections', parse_sections)

@norecursion
def pkg_to_filename(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pkg_to_filename'
    module_type_store = module_type_store.open_function_context('pkg_to_filename', 253, 0, False)
    
    # Passed parameters checking function
    pkg_to_filename.stypy_localization = localization
    pkg_to_filename.stypy_type_of_self = None
    pkg_to_filename.stypy_type_store = module_type_store
    pkg_to_filename.stypy_function_name = 'pkg_to_filename'
    pkg_to_filename.stypy_param_names_list = ['pkg_name']
    pkg_to_filename.stypy_varargs_param_name = None
    pkg_to_filename.stypy_kwargs_param_name = None
    pkg_to_filename.stypy_call_defaults = defaults
    pkg_to_filename.stypy_call_varargs = varargs
    pkg_to_filename.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pkg_to_filename', ['pkg_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pkg_to_filename', localization, ['pkg_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pkg_to_filename(...)' code ##################

    str_44623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 11), 'str', '%s.ini')
    # Getting the type of 'pkg_name' (line 254)
    pkg_name_44624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 22), 'pkg_name')
    # Applying the binary operator '%' (line 254)
    result_mod_44625 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 11), '%', str_44623, pkg_name_44624)
    
    # Assigning a type to the variable 'stypy_return_type' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'stypy_return_type', result_mod_44625)
    
    # ################# End of 'pkg_to_filename(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pkg_to_filename' in the type store
    # Getting the type of 'stypy_return_type' (line 253)
    stypy_return_type_44626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44626)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pkg_to_filename'
    return stypy_return_type_44626

# Assigning a type to the variable 'pkg_to_filename' (line 253)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 0), 'pkg_to_filename', pkg_to_filename)

@norecursion
def parse_config(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 256)
    None_44627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 32), 'None')
    defaults = [None_44627]
    # Create a new context for function 'parse_config'
    module_type_store = module_type_store.open_function_context('parse_config', 256, 0, False)
    
    # Passed parameters checking function
    parse_config.stypy_localization = localization
    parse_config.stypy_type_of_self = None
    parse_config.stypy_type_store = module_type_store
    parse_config.stypy_function_name = 'parse_config'
    parse_config.stypy_param_names_list = ['filename', 'dirs']
    parse_config.stypy_varargs_param_name = None
    parse_config.stypy_kwargs_param_name = None
    parse_config.stypy_call_defaults = defaults
    parse_config.stypy_call_varargs = varargs
    parse_config.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_config', ['filename', 'dirs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_config', localization, ['filename', 'dirs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_config(...)' code ##################

    
    # Getting the type of 'dirs' (line 257)
    dirs_44628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 7), 'dirs')
    # Testing the type of an if condition (line 257)
    if_condition_44629 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 4), dirs_44628)
    # Assigning a type to the variable 'if_condition_44629' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'if_condition_44629', if_condition_44629)
    # SSA begins for if statement (line 257)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a ListComp to a Name (line 258):
    
    # Assigning a ListComp to a Name (line 258):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'dirs' (line 258)
    dirs_44637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 56), 'dirs')
    comprehension_44638 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 21), dirs_44637)
    # Assigning a type to the variable 'd' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 21), 'd', comprehension_44638)
    
    # Call to join(...): (line 258)
    # Processing the call arguments (line 258)
    # Getting the type of 'd' (line 258)
    d_44633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 34), 'd', False)
    # Getting the type of 'filename' (line 258)
    filename_44634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 37), 'filename', False)
    # Processing the call keyword arguments (line 258)
    kwargs_44635 = {}
    # Getting the type of 'os' (line 258)
    os_44630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 21), 'os', False)
    # Obtaining the member 'path' of a type (line 258)
    path_44631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 21), os_44630, 'path')
    # Obtaining the member 'join' of a type (line 258)
    join_44632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 21), path_44631, 'join')
    # Calling join(args, kwargs) (line 258)
    join_call_result_44636 = invoke(stypy.reporting.localization.Localization(__file__, 258, 21), join_44632, *[d_44633, filename_44634], **kwargs_44635)
    
    list_44639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 21), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 21), list_44639, join_call_result_44636)
    # Assigning a type to the variable 'filenames' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'filenames', list_44639)
    # SSA branch for the else part of an if statement (line 257)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a List to a Name (line 260):
    
    # Assigning a List to a Name (line 260):
    
    # Obtaining an instance of the builtin type 'list' (line 260)
    list_44640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 260)
    # Adding element type (line 260)
    # Getting the type of 'filename' (line 260)
    filename_44641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 21), 'filename')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 260, 20), list_44640, filename_44641)
    
    # Assigning a type to the variable 'filenames' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'filenames', list_44640)
    # SSA join for if statement (line 257)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_44642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 20), 'int')
    slice_44643 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 262, 7), None, int_44642, None)
    # Getting the type of 'sys' (line 262)
    sys_44644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 7), 'sys')
    # Obtaining the member 'version' of a type (line 262)
    version_44645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 7), sys_44644, 'version')
    # Obtaining the member '__getitem__' of a type (line 262)
    getitem___44646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 7), version_44645, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 262)
    subscript_call_result_44647 = invoke(stypy.reporting.localization.Localization(__file__, 262, 7), getitem___44646, slice_44643)
    
    str_44648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 25), 'str', '3.1')
    # Applying the binary operator '>' (line 262)
    result_gt_44649 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 7), '>', subscript_call_result_44647, str_44648)
    
    # Testing the type of an if condition (line 262)
    if_condition_44650 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 4), result_gt_44649)
    # Assigning a type to the variable 'if_condition_44650' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'if_condition_44650', if_condition_44650)
    # SSA begins for if statement (line 262)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 264):
    
    # Assigning a Call to a Name (line 264):
    
    # Call to ConfigParser(...): (line 264)
    # Processing the call keyword arguments (line 264)
    kwargs_44652 = {}
    # Getting the type of 'ConfigParser' (line 264)
    ConfigParser_44651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 17), 'ConfigParser', False)
    # Calling ConfigParser(args, kwargs) (line 264)
    ConfigParser_call_result_44653 = invoke(stypy.reporting.localization.Localization(__file__, 264, 17), ConfigParser_44651, *[], **kwargs_44652)
    
    # Assigning a type to the variable 'config' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'config', ConfigParser_call_result_44653)
    # SSA branch for the else part of an if statement (line 262)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 266):
    
    # Assigning a Call to a Name (line 266):
    
    # Call to SafeConfigParser(...): (line 266)
    # Processing the call keyword arguments (line 266)
    kwargs_44655 = {}
    # Getting the type of 'SafeConfigParser' (line 266)
    SafeConfigParser_44654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 17), 'SafeConfigParser', False)
    # Calling SafeConfigParser(args, kwargs) (line 266)
    SafeConfigParser_call_result_44656 = invoke(stypy.reporting.localization.Localization(__file__, 266, 17), SafeConfigParser_44654, *[], **kwargs_44655)
    
    # Assigning a type to the variable 'config' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'config', SafeConfigParser_call_result_44656)
    # SSA join for if statement (line 262)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 268):
    
    # Assigning a Call to a Name (line 268):
    
    # Call to read(...): (line 268)
    # Processing the call arguments (line 268)
    # Getting the type of 'filenames' (line 268)
    filenames_44659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 20), 'filenames', False)
    # Processing the call keyword arguments (line 268)
    kwargs_44660 = {}
    # Getting the type of 'config' (line 268)
    config_44657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'config', False)
    # Obtaining the member 'read' of a type (line 268)
    read_44658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), config_44657, 'read')
    # Calling read(args, kwargs) (line 268)
    read_call_result_44661 = invoke(stypy.reporting.localization.Localization(__file__, 268, 8), read_44658, *[filenames_44659], **kwargs_44660)
    
    # Assigning a type to the variable 'n' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'n', read_call_result_44661)
    
    
    
    
    # Call to len(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'n' (line 269)
    n_44663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 15), 'n', False)
    # Processing the call keyword arguments (line 269)
    kwargs_44664 = {}
    # Getting the type of 'len' (line 269)
    len_44662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 11), 'len', False)
    # Calling len(args, kwargs) (line 269)
    len_call_result_44665 = invoke(stypy.reporting.localization.Localization(__file__, 269, 11), len_44662, *[n_44663], **kwargs_44664)
    
    int_44666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 21), 'int')
    # Applying the binary operator '>=' (line 269)
    result_ge_44667 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 11), '>=', len_call_result_44665, int_44666)
    
    # Applying the 'not' unary operator (line 269)
    result_not__44668 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 7), 'not', result_ge_44667)
    
    # Testing the type of an if condition (line 269)
    if_condition_44669 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 4), result_not__44668)
    # Assigning a type to the variable 'if_condition_44669' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'if_condition_44669', if_condition_44669)
    # SSA begins for if statement (line 269)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to PkgNotFound(...): (line 270)
    # Processing the call arguments (line 270)
    str_44671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 26), 'str', 'Could not find file(s) %s')
    
    # Call to str(...): (line 270)
    # Processing the call arguments (line 270)
    # Getting the type of 'filenames' (line 270)
    filenames_44673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 60), 'filenames', False)
    # Processing the call keyword arguments (line 270)
    kwargs_44674 = {}
    # Getting the type of 'str' (line 270)
    str_44672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 56), 'str', False)
    # Calling str(args, kwargs) (line 270)
    str_call_result_44675 = invoke(stypy.reporting.localization.Localization(__file__, 270, 56), str_44672, *[filenames_44673], **kwargs_44674)
    
    # Applying the binary operator '%' (line 270)
    result_mod_44676 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 26), '%', str_44671, str_call_result_44675)
    
    # Processing the call keyword arguments (line 270)
    kwargs_44677 = {}
    # Getting the type of 'PkgNotFound' (line 270)
    PkgNotFound_44670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 14), 'PkgNotFound', False)
    # Calling PkgNotFound(args, kwargs) (line 270)
    PkgNotFound_call_result_44678 = invoke(stypy.reporting.localization.Localization(__file__, 270, 14), PkgNotFound_44670, *[result_mod_44676], **kwargs_44677)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 270, 8), PkgNotFound_call_result_44678, 'raise parameter', BaseException)
    # SSA join for if statement (line 269)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 273):
    
    # Assigning a Call to a Name (line 273):
    
    # Call to parse_meta(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'config' (line 273)
    config_44680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 22), 'config', False)
    # Processing the call keyword arguments (line 273)
    kwargs_44681 = {}
    # Getting the type of 'parse_meta' (line 273)
    parse_meta_44679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 11), 'parse_meta', False)
    # Calling parse_meta(args, kwargs) (line 273)
    parse_meta_call_result_44682 = invoke(stypy.reporting.localization.Localization(__file__, 273, 11), parse_meta_44679, *[config_44680], **kwargs_44681)
    
    # Assigning a type to the variable 'meta' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'meta', parse_meta_call_result_44682)
    
    # Assigning a Dict to a Name (line 275):
    
    # Assigning a Dict to a Name (line 275):
    
    # Obtaining an instance of the builtin type 'dict' (line 275)
    dict_44683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 275)
    
    # Assigning a type to the variable 'vars' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'vars', dict_44683)
    
    
    # Call to has_section(...): (line 276)
    # Processing the call arguments (line 276)
    str_44686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 26), 'str', 'variables')
    # Processing the call keyword arguments (line 276)
    kwargs_44687 = {}
    # Getting the type of 'config' (line 276)
    config_44684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 7), 'config', False)
    # Obtaining the member 'has_section' of a type (line 276)
    has_section_44685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 7), config_44684, 'has_section')
    # Calling has_section(args, kwargs) (line 276)
    has_section_call_result_44688 = invoke(stypy.reporting.localization.Localization(__file__, 276, 7), has_section_44685, *[str_44686], **kwargs_44687)
    
    # Testing the type of an if condition (line 276)
    if_condition_44689 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 276, 4), has_section_call_result_44688)
    # Assigning a type to the variable 'if_condition_44689' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'if_condition_44689', if_condition_44689)
    # SSA begins for if statement (line 276)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to items(...): (line 277)
    # Processing the call arguments (line 277)
    str_44692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 40), 'str', 'variables')
    # Processing the call keyword arguments (line 277)
    kwargs_44693 = {}
    # Getting the type of 'config' (line 277)
    config_44690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 27), 'config', False)
    # Obtaining the member 'items' of a type (line 277)
    items_44691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 27), config_44690, 'items')
    # Calling items(args, kwargs) (line 277)
    items_call_result_44694 = invoke(stypy.reporting.localization.Localization(__file__, 277, 27), items_44691, *[str_44692], **kwargs_44693)
    
    # Testing the type of a for loop iterable (line 277)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 277, 8), items_call_result_44694)
    # Getting the type of the for loop variable (line 277)
    for_loop_var_44695 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 277, 8), items_call_result_44694)
    # Assigning a type to the variable 'name' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 8), for_loop_var_44695))
    # Assigning a type to the variable 'value' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 277, 8), for_loop_var_44695))
    # SSA begins for a for statement (line 277)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 278):
    
    # Assigning a Call to a Subscript (line 278):
    
    # Call to _escape_backslash(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'value' (line 278)
    value_44697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 43), 'value', False)
    # Processing the call keyword arguments (line 278)
    kwargs_44698 = {}
    # Getting the type of '_escape_backslash' (line 278)
    _escape_backslash_44696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 25), '_escape_backslash', False)
    # Calling _escape_backslash(args, kwargs) (line 278)
    _escape_backslash_call_result_44699 = invoke(stypy.reporting.localization.Localization(__file__, 278, 25), _escape_backslash_44696, *[value_44697], **kwargs_44698)
    
    # Getting the type of 'vars' (line 278)
    vars_44700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'vars')
    # Getting the type of 'name' (line 278)
    name_44701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 17), 'name')
    # Storing an element on a container (line 278)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 12), vars_44700, (name_44701, _escape_backslash_call_result_44699))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 276)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Name (line 281):
    
    # Assigning a ListComp to a Name (line 281):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to sections(...): (line 281)
    # Processing the call keyword arguments (line 281)
    kwargs_44711 = {}
    # Getting the type of 'config' (line 281)
    config_44709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 23), 'config', False)
    # Obtaining the member 'sections' of a type (line 281)
    sections_44710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 23), config_44709, 'sections')
    # Calling sections(args, kwargs) (line 281)
    sections_call_result_44712 = invoke(stypy.reporting.localization.Localization(__file__, 281, 23), sections_44710, *[], **kwargs_44711)
    
    comprehension_44713 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 12), sections_call_result_44712)
    # Assigning a type to the variable 's' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 's', comprehension_44713)
    
    
    # Getting the type of 's' (line 281)
    s_44703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 48), 's')
    
    # Obtaining an instance of the builtin type 'list' (line 281)
    list_44704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 53), 'list')
    # Adding type elements to the builtin type 'list' instance (line 281)
    # Adding element type (line 281)
    str_44705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 54), 'str', 'meta')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 53), list_44704, str_44705)
    # Adding element type (line 281)
    str_44706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 62), 'str', 'variables')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 53), list_44704, str_44706)
    
    # Applying the binary operator 'in' (line 281)
    result_contains_44707 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 48), 'in', s_44703, list_44704)
    
    # Applying the 'not' unary operator (line 281)
    result_not__44708 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 44), 'not', result_contains_44707)
    
    # Getting the type of 's' (line 281)
    s_44702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 's')
    list_44714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 281, 12), list_44714, s_44702)
    # Assigning a type to the variable 'secs' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'secs', list_44714)
    
    # Assigning a Dict to a Name (line 282):
    
    # Assigning a Dict to a Name (line 282):
    
    # Obtaining an instance of the builtin type 'dict' (line 282)
    dict_44715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 282)
    
    # Assigning a type to the variable 'sections' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'sections', dict_44715)
    
    # Assigning a Dict to a Name (line 284):
    
    # Assigning a Dict to a Name (line 284):
    
    # Obtaining an instance of the builtin type 'dict' (line 284)
    dict_44716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 284)
    
    # Assigning a type to the variable 'requires' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'requires', dict_44716)
    
    # Getting the type of 'secs' (line 285)
    secs_44717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 13), 'secs')
    # Testing the type of a for loop iterable (line 285)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 285, 4), secs_44717)
    # Getting the type of the for loop variable (line 285)
    for_loop_var_44718 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 285, 4), secs_44717)
    # Assigning a type to the variable 's' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 's', for_loop_var_44718)
    # SSA begins for a for statement (line 285)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Dict to a Name (line 286):
    
    # Assigning a Dict to a Name (line 286):
    
    # Obtaining an instance of the builtin type 'dict' (line 286)
    dict_44719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 12), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 286)
    
    # Assigning a type to the variable 'd' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'd', dict_44719)
    
    
    # Call to has_option(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 's' (line 287)
    s_44722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 29), 's', False)
    str_44723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 32), 'str', 'requires')
    # Processing the call keyword arguments (line 287)
    kwargs_44724 = {}
    # Getting the type of 'config' (line 287)
    config_44720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 11), 'config', False)
    # Obtaining the member 'has_option' of a type (line 287)
    has_option_44721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 11), config_44720, 'has_option')
    # Calling has_option(args, kwargs) (line 287)
    has_option_call_result_44725 = invoke(stypy.reporting.localization.Localization(__file__, 287, 11), has_option_44721, *[s_44722, str_44723], **kwargs_44724)
    
    # Testing the type of an if condition (line 287)
    if_condition_44726 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 287, 8), has_option_call_result_44725)
    # Assigning a type to the variable 'if_condition_44726' (line 287)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'if_condition_44726', if_condition_44726)
    # SSA begins for if statement (line 287)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 288):
    
    # Assigning a Call to a Subscript (line 288):
    
    # Call to get(...): (line 288)
    # Processing the call arguments (line 288)
    # Getting the type of 's' (line 288)
    s_44729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 37), 's', False)
    str_44730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 40), 'str', 'requires')
    # Processing the call keyword arguments (line 288)
    kwargs_44731 = {}
    # Getting the type of 'config' (line 288)
    config_44727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 26), 'config', False)
    # Obtaining the member 'get' of a type (line 288)
    get_44728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 26), config_44727, 'get')
    # Calling get(args, kwargs) (line 288)
    get_call_result_44732 = invoke(stypy.reporting.localization.Localization(__file__, 288, 26), get_44728, *[s_44729, str_44730], **kwargs_44731)
    
    # Getting the type of 'requires' (line 288)
    requires_44733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'requires')
    # Getting the type of 's' (line 288)
    s_44734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 21), 's')
    # Storing an element on a container (line 288)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 12), requires_44733, (s_44734, get_call_result_44732))
    # SSA join for if statement (line 287)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to items(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 's' (line 290)
    s_44737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 40), 's', False)
    # Processing the call keyword arguments (line 290)
    kwargs_44738 = {}
    # Getting the type of 'config' (line 290)
    config_44735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 27), 'config', False)
    # Obtaining the member 'items' of a type (line 290)
    items_44736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 27), config_44735, 'items')
    # Calling items(args, kwargs) (line 290)
    items_call_result_44739 = invoke(stypy.reporting.localization.Localization(__file__, 290, 27), items_44736, *[s_44737], **kwargs_44738)
    
    # Testing the type of a for loop iterable (line 290)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 290, 8), items_call_result_44739)
    # Getting the type of the for loop variable (line 290)
    for_loop_var_44740 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 290, 8), items_call_result_44739)
    # Assigning a type to the variable 'name' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 8), for_loop_var_44740))
    # Assigning a type to the variable 'value' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 8), for_loop_var_44740))
    # SSA begins for a for statement (line 290)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Subscript (line 291):
    
    # Assigning a Name to a Subscript (line 291):
    # Getting the type of 'value' (line 291)
    value_44741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 22), 'value')
    # Getting the type of 'd' (line 291)
    d_44742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'd')
    # Getting the type of 'name' (line 291)
    name_44743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 14), 'name')
    # Storing an element on a container (line 291)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 291, 12), d_44742, (name_44743, value_44741))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 292):
    
    # Assigning a Name to a Subscript (line 292):
    # Getting the type of 'd' (line 292)
    d_44744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 22), 'd')
    # Getting the type of 'sections' (line 292)
    sections_44745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'sections')
    # Getting the type of 's' (line 292)
    s_44746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 17), 's')
    # Storing an element on a container (line 292)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 292, 8), sections_44745, (s_44746, d_44744))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 294)
    tuple_44747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 294)
    # Adding element type (line 294)
    # Getting the type of 'meta' (line 294)
    meta_44748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 11), 'meta')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 11), tuple_44747, meta_44748)
    # Adding element type (line 294)
    # Getting the type of 'vars' (line 294)
    vars_44749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 17), 'vars')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 11), tuple_44747, vars_44749)
    # Adding element type (line 294)
    # Getting the type of 'sections' (line 294)
    sections_44750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 23), 'sections')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 11), tuple_44747, sections_44750)
    # Adding element type (line 294)
    # Getting the type of 'requires' (line 294)
    requires_44751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 33), 'requires')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 294, 11), tuple_44747, requires_44751)
    
    # Assigning a type to the variable 'stypy_return_type' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'stypy_return_type', tuple_44747)
    
    # ################# End of 'parse_config(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_config' in the type store
    # Getting the type of 'stypy_return_type' (line 256)
    stypy_return_type_44752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44752)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_config'
    return stypy_return_type_44752

# Assigning a type to the variable 'parse_config' (line 256)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 0), 'parse_config', parse_config)

@norecursion
def _read_config_imp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 296)
    None_44753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 37), 'None')
    defaults = [None_44753]
    # Create a new context for function '_read_config_imp'
    module_type_store = module_type_store.open_function_context('_read_config_imp', 296, 0, False)
    
    # Passed parameters checking function
    _read_config_imp.stypy_localization = localization
    _read_config_imp.stypy_type_of_self = None
    _read_config_imp.stypy_type_store = module_type_store
    _read_config_imp.stypy_function_name = '_read_config_imp'
    _read_config_imp.stypy_param_names_list = ['filenames', 'dirs']
    _read_config_imp.stypy_varargs_param_name = None
    _read_config_imp.stypy_kwargs_param_name = None
    _read_config_imp.stypy_call_defaults = defaults
    _read_config_imp.stypy_call_varargs = varargs
    _read_config_imp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_read_config_imp', ['filenames', 'dirs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_read_config_imp', localization, ['filenames', 'dirs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_read_config_imp(...)' code ##################


    @norecursion
    def _read_config(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_read_config'
        module_type_store = module_type_store.open_function_context('_read_config', 297, 4, False)
        
        # Passed parameters checking function
        _read_config.stypy_localization = localization
        _read_config.stypy_type_of_self = None
        _read_config.stypy_type_store = module_type_store
        _read_config.stypy_function_name = '_read_config'
        _read_config.stypy_param_names_list = ['f']
        _read_config.stypy_varargs_param_name = None
        _read_config.stypy_kwargs_param_name = None
        _read_config.stypy_call_defaults = defaults
        _read_config.stypy_call_varargs = varargs
        _read_config.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_read_config', ['f'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_read_config', localization, ['f'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_read_config(...)' code ##################

        
        # Assigning a Call to a Tuple (line 298):
        
        # Assigning a Call to a Name:
        
        # Call to parse_config(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'f' (line 298)
        f_44755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 50), 'f', False)
        # Getting the type of 'dirs' (line 298)
        dirs_44756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 53), 'dirs', False)
        # Processing the call keyword arguments (line 298)
        kwargs_44757 = {}
        # Getting the type of 'parse_config' (line 298)
        parse_config_44754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 37), 'parse_config', False)
        # Calling parse_config(args, kwargs) (line 298)
        parse_config_call_result_44758 = invoke(stypy.reporting.localization.Localization(__file__, 298, 37), parse_config_44754, *[f_44755, dirs_44756], **kwargs_44757)
        
        # Assigning a type to the variable 'call_assignment_44136' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'call_assignment_44136', parse_config_call_result_44758)
        
        # Assigning a Call to a Name (line 298):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_44761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 8), 'int')
        # Processing the call keyword arguments
        kwargs_44762 = {}
        # Getting the type of 'call_assignment_44136' (line 298)
        call_assignment_44136_44759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'call_assignment_44136', False)
        # Obtaining the member '__getitem__' of a type (line 298)
        getitem___44760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), call_assignment_44136_44759, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_44763 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___44760, *[int_44761], **kwargs_44762)
        
        # Assigning a type to the variable 'call_assignment_44137' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'call_assignment_44137', getitem___call_result_44763)
        
        # Assigning a Name to a Name (line 298):
        # Getting the type of 'call_assignment_44137' (line 298)
        call_assignment_44137_44764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'call_assignment_44137')
        # Assigning a type to the variable 'meta' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'meta', call_assignment_44137_44764)
        
        # Assigning a Call to a Name (line 298):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_44767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 8), 'int')
        # Processing the call keyword arguments
        kwargs_44768 = {}
        # Getting the type of 'call_assignment_44136' (line 298)
        call_assignment_44136_44765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'call_assignment_44136', False)
        # Obtaining the member '__getitem__' of a type (line 298)
        getitem___44766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), call_assignment_44136_44765, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_44769 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___44766, *[int_44767], **kwargs_44768)
        
        # Assigning a type to the variable 'call_assignment_44138' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'call_assignment_44138', getitem___call_result_44769)
        
        # Assigning a Name to a Name (line 298):
        # Getting the type of 'call_assignment_44138' (line 298)
        call_assignment_44138_44770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'call_assignment_44138')
        # Assigning a type to the variable 'vars' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 14), 'vars', call_assignment_44138_44770)
        
        # Assigning a Call to a Name (line 298):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_44773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 8), 'int')
        # Processing the call keyword arguments
        kwargs_44774 = {}
        # Getting the type of 'call_assignment_44136' (line 298)
        call_assignment_44136_44771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'call_assignment_44136', False)
        # Obtaining the member '__getitem__' of a type (line 298)
        getitem___44772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), call_assignment_44136_44771, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_44775 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___44772, *[int_44773], **kwargs_44774)
        
        # Assigning a type to the variable 'call_assignment_44139' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'call_assignment_44139', getitem___call_result_44775)
        
        # Assigning a Name to a Name (line 298):
        # Getting the type of 'call_assignment_44139' (line 298)
        call_assignment_44139_44776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'call_assignment_44139')
        # Assigning a type to the variable 'sections' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 20), 'sections', call_assignment_44139_44776)
        
        # Assigning a Call to a Name (line 298):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_44779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 8), 'int')
        # Processing the call keyword arguments
        kwargs_44780 = {}
        # Getting the type of 'call_assignment_44136' (line 298)
        call_assignment_44136_44777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'call_assignment_44136', False)
        # Obtaining the member '__getitem__' of a type (line 298)
        getitem___44778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), call_assignment_44136_44777, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_44781 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___44778, *[int_44779], **kwargs_44780)
        
        # Assigning a type to the variable 'call_assignment_44140' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'call_assignment_44140', getitem___call_result_44781)
        
        # Assigning a Name to a Name (line 298):
        # Getting the type of 'call_assignment_44140' (line 298)
        call_assignment_44140_44782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'call_assignment_44140')
        # Assigning a type to the variable 'reqs' (line 298)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 30), 'reqs', call_assignment_44140_44782)
        
        
        # Call to items(...): (line 300)
        # Processing the call keyword arguments (line 300)
        kwargs_44785 = {}
        # Getting the type of 'reqs' (line 300)
        reqs_44783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 29), 'reqs', False)
        # Obtaining the member 'items' of a type (line 300)
        items_44784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 29), reqs_44783, 'items')
        # Calling items(args, kwargs) (line 300)
        items_call_result_44786 = invoke(stypy.reporting.localization.Localization(__file__, 300, 29), items_44784, *[], **kwargs_44785)
        
        # Testing the type of a for loop iterable (line 300)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 300, 8), items_call_result_44786)
        # Getting the type of the for loop variable (line 300)
        for_loop_var_44787 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 300, 8), items_call_result_44786)
        # Assigning a type to the variable 'rname' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'rname', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 8), for_loop_var_44787))
        # Assigning a type to the variable 'rvalue' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'rvalue', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 8), for_loop_var_44787))
        # SSA begins for a for statement (line 300)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 301):
        
        # Assigning a Call to a Name:
        
        # Call to _read_config(...): (line 301)
        # Processing the call arguments (line 301)
        
        # Call to pkg_to_filename(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 'rvalue' (line 301)
        rvalue_44790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 74), 'rvalue', False)
        # Processing the call keyword arguments (line 301)
        kwargs_44791 = {}
        # Getting the type of 'pkg_to_filename' (line 301)
        pkg_to_filename_44789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 58), 'pkg_to_filename', False)
        # Calling pkg_to_filename(args, kwargs) (line 301)
        pkg_to_filename_call_result_44792 = invoke(stypy.reporting.localization.Localization(__file__, 301, 58), pkg_to_filename_44789, *[rvalue_44790], **kwargs_44791)
        
        # Processing the call keyword arguments (line 301)
        kwargs_44793 = {}
        # Getting the type of '_read_config' (line 301)
        _read_config_44788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 45), '_read_config', False)
        # Calling _read_config(args, kwargs) (line 301)
        _read_config_call_result_44794 = invoke(stypy.reporting.localization.Localization(__file__, 301, 45), _read_config_44788, *[pkg_to_filename_call_result_44792], **kwargs_44793)
        
        # Assigning a type to the variable 'call_assignment_44141' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'call_assignment_44141', _read_config_call_result_44794)
        
        # Assigning a Call to a Name (line 301):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_44797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 12), 'int')
        # Processing the call keyword arguments
        kwargs_44798 = {}
        # Getting the type of 'call_assignment_44141' (line 301)
        call_assignment_44141_44795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'call_assignment_44141', False)
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___44796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 12), call_assignment_44141_44795, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_44799 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___44796, *[int_44797], **kwargs_44798)
        
        # Assigning a type to the variable 'call_assignment_44142' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'call_assignment_44142', getitem___call_result_44799)
        
        # Assigning a Name to a Name (line 301):
        # Getting the type of 'call_assignment_44142' (line 301)
        call_assignment_44142_44800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'call_assignment_44142')
        # Assigning a type to the variable 'nmeta' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'nmeta', call_assignment_44142_44800)
        
        # Assigning a Call to a Name (line 301):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_44803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 12), 'int')
        # Processing the call keyword arguments
        kwargs_44804 = {}
        # Getting the type of 'call_assignment_44141' (line 301)
        call_assignment_44141_44801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'call_assignment_44141', False)
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___44802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 12), call_assignment_44141_44801, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_44805 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___44802, *[int_44803], **kwargs_44804)
        
        # Assigning a type to the variable 'call_assignment_44143' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'call_assignment_44143', getitem___call_result_44805)
        
        # Assigning a Name to a Name (line 301):
        # Getting the type of 'call_assignment_44143' (line 301)
        call_assignment_44143_44806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'call_assignment_44143')
        # Assigning a type to the variable 'nvars' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 19), 'nvars', call_assignment_44143_44806)
        
        # Assigning a Call to a Name (line 301):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_44809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 12), 'int')
        # Processing the call keyword arguments
        kwargs_44810 = {}
        # Getting the type of 'call_assignment_44141' (line 301)
        call_assignment_44141_44807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'call_assignment_44141', False)
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___44808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 12), call_assignment_44141_44807, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_44811 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___44808, *[int_44809], **kwargs_44810)
        
        # Assigning a type to the variable 'call_assignment_44144' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'call_assignment_44144', getitem___call_result_44811)
        
        # Assigning a Name to a Name (line 301):
        # Getting the type of 'call_assignment_44144' (line 301)
        call_assignment_44144_44812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'call_assignment_44144')
        # Assigning a type to the variable 'nsections' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 26), 'nsections', call_assignment_44144_44812)
        
        # Assigning a Call to a Name (line 301):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_44815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 12), 'int')
        # Processing the call keyword arguments
        kwargs_44816 = {}
        # Getting the type of 'call_assignment_44141' (line 301)
        call_assignment_44141_44813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'call_assignment_44141', False)
        # Obtaining the member '__getitem__' of a type (line 301)
        getitem___44814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 12), call_assignment_44141_44813, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_44817 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___44814, *[int_44815], **kwargs_44816)
        
        # Assigning a type to the variable 'call_assignment_44145' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'call_assignment_44145', getitem___call_result_44817)
        
        # Assigning a Name to a Name (line 301):
        # Getting the type of 'call_assignment_44145' (line 301)
        call_assignment_44145_44818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'call_assignment_44145')
        # Assigning a type to the variable 'nreqs' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 37), 'nreqs', call_assignment_44145_44818)
        
        
        # Call to items(...): (line 304)
        # Processing the call keyword arguments (line 304)
        kwargs_44821 = {}
        # Getting the type of 'nvars' (line 304)
        nvars_44819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 24), 'nvars', False)
        # Obtaining the member 'items' of a type (line 304)
        items_44820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 24), nvars_44819, 'items')
        # Calling items(args, kwargs) (line 304)
        items_call_result_44822 = invoke(stypy.reporting.localization.Localization(__file__, 304, 24), items_44820, *[], **kwargs_44821)
        
        # Testing the type of a for loop iterable (line 304)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 304, 12), items_call_result_44822)
        # Getting the type of the for loop variable (line 304)
        for_loop_var_44823 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 304, 12), items_call_result_44822)
        # Assigning a type to the variable 'k' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), for_loop_var_44823))
        # Assigning a type to the variable 'v' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 12), for_loop_var_44823))
        # SSA begins for a for statement (line 304)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Getting the type of 'k' (line 305)
        k_44824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 23), 'k')
        # Getting the type of 'vars' (line 305)
        vars_44825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 28), 'vars')
        # Applying the binary operator 'in' (line 305)
        result_contains_44826 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 23), 'in', k_44824, vars_44825)
        
        # Applying the 'not' unary operator (line 305)
        result_not__44827 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 19), 'not', result_contains_44826)
        
        # Testing the type of an if condition (line 305)
        if_condition_44828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 16), result_not__44827)
        # Assigning a type to the variable 'if_condition_44828' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 16), 'if_condition_44828', if_condition_44828)
        # SSA begins for if statement (line 305)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 306):
        
        # Assigning a Name to a Subscript (line 306):
        # Getting the type of 'v' (line 306)
        v_44829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 30), 'v')
        # Getting the type of 'vars' (line 306)
        vars_44830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 20), 'vars')
        # Getting the type of 'k' (line 306)
        k_44831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 25), 'k')
        # Storing an element on a container (line 306)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 306, 20), vars_44830, (k_44831, v_44829))
        # SSA join for if statement (line 305)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to items(...): (line 309)
        # Processing the call keyword arguments (line 309)
        kwargs_44837 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'rname' (line 309)
        rname_44832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 43), 'rname', False)
        # Getting the type of 'nsections' (line 309)
        nsections_44833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 33), 'nsections', False)
        # Obtaining the member '__getitem__' of a type (line 309)
        getitem___44834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 33), nsections_44833, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 309)
        subscript_call_result_44835 = invoke(stypy.reporting.localization.Localization(__file__, 309, 33), getitem___44834, rname_44832)
        
        # Obtaining the member 'items' of a type (line 309)
        items_44836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 33), subscript_call_result_44835, 'items')
        # Calling items(args, kwargs) (line 309)
        items_call_result_44838 = invoke(stypy.reporting.localization.Localization(__file__, 309, 33), items_44836, *[], **kwargs_44837)
        
        # Testing the type of a for loop iterable (line 309)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 309, 12), items_call_result_44838)
        # Getting the type of the for loop variable (line 309)
        for_loop_var_44839 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 309, 12), items_call_result_44838)
        # Assigning a type to the variable 'oname' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'oname', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 12), for_loop_var_44839))
        # Assigning a type to the variable 'ovalue' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'ovalue', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 309, 12), for_loop_var_44839))
        # SSA begins for a for statement (line 309)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'ovalue' (line 310)
        ovalue_44840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 19), 'ovalue')
        # Testing the type of an if condition (line 310)
        if_condition_44841 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 310, 16), ovalue_44840)
        # Assigning a type to the variable 'if_condition_44841' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 16), 'if_condition_44841', if_condition_44841)
        # SSA begins for if statement (line 310)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'rname' (line 311)
        rname_44842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 29), 'rname')
        # Getting the type of 'sections' (line 311)
        sections_44843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 20), 'sections')
        # Obtaining the member '__getitem__' of a type (line 311)
        getitem___44844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 20), sections_44843, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 311)
        subscript_call_result_44845 = invoke(stypy.reporting.localization.Localization(__file__, 311, 20), getitem___44844, rname_44842)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'oname' (line 311)
        oname_44846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 36), 'oname')
        
        # Obtaining the type of the subscript
        # Getting the type of 'rname' (line 311)
        rname_44847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 29), 'rname')
        # Getting the type of 'sections' (line 311)
        sections_44848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 20), 'sections')
        # Obtaining the member '__getitem__' of a type (line 311)
        getitem___44849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 20), sections_44848, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 311)
        subscript_call_result_44850 = invoke(stypy.reporting.localization.Localization(__file__, 311, 20), getitem___44849, rname_44847)
        
        # Obtaining the member '__getitem__' of a type (line 311)
        getitem___44851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 20), subscript_call_result_44850, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 311)
        subscript_call_result_44852 = invoke(stypy.reporting.localization.Localization(__file__, 311, 20), getitem___44851, oname_44846)
        
        str_44853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 46), 'str', ' %s')
        # Getting the type of 'ovalue' (line 311)
        ovalue_44854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 54), 'ovalue')
        # Applying the binary operator '%' (line 311)
        result_mod_44855 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 46), '%', str_44853, ovalue_44854)
        
        # Applying the binary operator '+=' (line 311)
        result_iadd_44856 = python_operator(stypy.reporting.localization.Localization(__file__, 311, 20), '+=', subscript_call_result_44852, result_mod_44855)
        
        # Obtaining the type of the subscript
        # Getting the type of 'rname' (line 311)
        rname_44857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 29), 'rname')
        # Getting the type of 'sections' (line 311)
        sections_44858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 20), 'sections')
        # Obtaining the member '__getitem__' of a type (line 311)
        getitem___44859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 20), sections_44858, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 311)
        subscript_call_result_44860 = invoke(stypy.reporting.localization.Localization(__file__, 311, 20), getitem___44859, rname_44857)
        
        # Getting the type of 'oname' (line 311)
        oname_44861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 36), 'oname')
        # Storing an element on a container (line 311)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 20), subscript_call_result_44860, (oname_44861, result_iadd_44856))
        
        # SSA join for if statement (line 310)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 313)
        tuple_44862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 313)
        # Adding element type (line 313)
        # Getting the type of 'meta' (line 313)
        meta_44863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 15), 'meta')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 15), tuple_44862, meta_44863)
        # Adding element type (line 313)
        # Getting the type of 'vars' (line 313)
        vars_44864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 21), 'vars')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 15), tuple_44862, vars_44864)
        # Adding element type (line 313)
        # Getting the type of 'sections' (line 313)
        sections_44865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 27), 'sections')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 15), tuple_44862, sections_44865)
        # Adding element type (line 313)
        # Getting the type of 'reqs' (line 313)
        reqs_44866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 37), 'reqs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 15), tuple_44862, reqs_44866)
        
        # Assigning a type to the variable 'stypy_return_type' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'stypy_return_type', tuple_44862)
        
        # ################# End of '_read_config(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_read_config' in the type store
        # Getting the type of 'stypy_return_type' (line 297)
        stypy_return_type_44867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44867)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_read_config'
        return stypy_return_type_44867

    # Assigning a type to the variable '_read_config' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), '_read_config', _read_config)
    
    # Assigning a Call to a Tuple (line 315):
    
    # Assigning a Call to a Name:
    
    # Call to _read_config(...): (line 315)
    # Processing the call arguments (line 315)
    # Getting the type of 'filenames' (line 315)
    filenames_44869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 46), 'filenames', False)
    # Processing the call keyword arguments (line 315)
    kwargs_44870 = {}
    # Getting the type of '_read_config' (line 315)
    _read_config_44868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 33), '_read_config', False)
    # Calling _read_config(args, kwargs) (line 315)
    _read_config_call_result_44871 = invoke(stypy.reporting.localization.Localization(__file__, 315, 33), _read_config_44868, *[filenames_44869], **kwargs_44870)
    
    # Assigning a type to the variable 'call_assignment_44146' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'call_assignment_44146', _read_config_call_result_44871)
    
    # Assigning a Call to a Name (line 315):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_44874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 4), 'int')
    # Processing the call keyword arguments
    kwargs_44875 = {}
    # Getting the type of 'call_assignment_44146' (line 315)
    call_assignment_44146_44872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'call_assignment_44146', False)
    # Obtaining the member '__getitem__' of a type (line 315)
    getitem___44873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 4), call_assignment_44146_44872, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_44876 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___44873, *[int_44874], **kwargs_44875)
    
    # Assigning a type to the variable 'call_assignment_44147' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'call_assignment_44147', getitem___call_result_44876)
    
    # Assigning a Name to a Name (line 315):
    # Getting the type of 'call_assignment_44147' (line 315)
    call_assignment_44147_44877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'call_assignment_44147')
    # Assigning a type to the variable 'meta' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'meta', call_assignment_44147_44877)
    
    # Assigning a Call to a Name (line 315):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_44880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 4), 'int')
    # Processing the call keyword arguments
    kwargs_44881 = {}
    # Getting the type of 'call_assignment_44146' (line 315)
    call_assignment_44146_44878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'call_assignment_44146', False)
    # Obtaining the member '__getitem__' of a type (line 315)
    getitem___44879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 4), call_assignment_44146_44878, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_44882 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___44879, *[int_44880], **kwargs_44881)
    
    # Assigning a type to the variable 'call_assignment_44148' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'call_assignment_44148', getitem___call_result_44882)
    
    # Assigning a Name to a Name (line 315):
    # Getting the type of 'call_assignment_44148' (line 315)
    call_assignment_44148_44883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'call_assignment_44148')
    # Assigning a type to the variable 'vars' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 10), 'vars', call_assignment_44148_44883)
    
    # Assigning a Call to a Name (line 315):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_44886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 4), 'int')
    # Processing the call keyword arguments
    kwargs_44887 = {}
    # Getting the type of 'call_assignment_44146' (line 315)
    call_assignment_44146_44884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'call_assignment_44146', False)
    # Obtaining the member '__getitem__' of a type (line 315)
    getitem___44885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 4), call_assignment_44146_44884, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_44888 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___44885, *[int_44886], **kwargs_44887)
    
    # Assigning a type to the variable 'call_assignment_44149' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'call_assignment_44149', getitem___call_result_44888)
    
    # Assigning a Name to a Name (line 315):
    # Getting the type of 'call_assignment_44149' (line 315)
    call_assignment_44149_44889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'call_assignment_44149')
    # Assigning a type to the variable 'sections' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'sections', call_assignment_44149_44889)
    
    # Assigning a Call to a Name (line 315):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_44892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 4), 'int')
    # Processing the call keyword arguments
    kwargs_44893 = {}
    # Getting the type of 'call_assignment_44146' (line 315)
    call_assignment_44146_44890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'call_assignment_44146', False)
    # Obtaining the member '__getitem__' of a type (line 315)
    getitem___44891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 4), call_assignment_44146_44890, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_44894 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___44891, *[int_44892], **kwargs_44893)
    
    # Assigning a type to the variable 'call_assignment_44150' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'call_assignment_44150', getitem___call_result_44894)
    
    # Assigning a Name to a Name (line 315):
    # Getting the type of 'call_assignment_44150' (line 315)
    call_assignment_44150_44895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 4), 'call_assignment_44150')
    # Assigning a type to the variable 'reqs' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 26), 'reqs', call_assignment_44150_44895)
    
    
    # Evaluating a boolean operation
    
    
    str_44896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 11), 'str', 'pkgdir')
    # Getting the type of 'vars' (line 320)
    vars_44897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 23), 'vars')
    # Applying the binary operator 'in' (line 320)
    result_contains_44898 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 11), 'in', str_44896, vars_44897)
    
    # Applying the 'not' unary operator (line 320)
    result_not__44899 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 7), 'not', result_contains_44898)
    
    
    str_44900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 32), 'str', 'pkgname')
    # Getting the type of 'vars' (line 320)
    vars_44901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 45), 'vars')
    # Applying the binary operator 'in' (line 320)
    result_contains_44902 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 32), 'in', str_44900, vars_44901)
    
    # Applying the binary operator 'and' (line 320)
    result_and_keyword_44903 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 7), 'and', result_not__44899, result_contains_44902)
    
    # Testing the type of an if condition (line 320)
    if_condition_44904 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 320, 4), result_and_keyword_44903)
    # Assigning a type to the variable 'if_condition_44904' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'if_condition_44904', if_condition_44904)
    # SSA begins for if statement (line 320)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 321):
    
    # Assigning a Subscript to a Name (line 321):
    
    # Obtaining the type of the subscript
    str_44905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 23), 'str', 'pkgname')
    # Getting the type of 'vars' (line 321)
    vars_44906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 18), 'vars')
    # Obtaining the member '__getitem__' of a type (line 321)
    getitem___44907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 18), vars_44906, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 321)
    subscript_call_result_44908 = invoke(stypy.reporting.localization.Localization(__file__, 321, 18), getitem___44907, str_44905)
    
    # Assigning a type to the variable 'pkgname' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'pkgname', subscript_call_result_44908)
    
    
    
    # Getting the type of 'pkgname' (line 322)
    pkgname_44909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 15), 'pkgname')
    # Getting the type of 'sys' (line 322)
    sys_44910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 26), 'sys')
    # Obtaining the member 'modules' of a type (line 322)
    modules_44911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 26), sys_44910, 'modules')
    # Applying the binary operator 'in' (line 322)
    result_contains_44912 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 15), 'in', pkgname_44909, modules_44911)
    
    # Applying the 'not' unary operator (line 322)
    result_not__44913 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 11), 'not', result_contains_44912)
    
    # Testing the type of an if condition (line 322)
    if_condition_44914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 8), result_not__44913)
    # Assigning a type to the variable 'if_condition_44914' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'if_condition_44914', if_condition_44914)
    # SSA begins for if statement (line 322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 323)
    # Processing the call arguments (line 323)
    str_44916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 29), 'str', 'You should import %s to get information on %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 324)
    tuple_44917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 324)
    # Adding element type (line 324)
    # Getting the type of 'pkgname' (line 324)
    pkgname_44918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 30), 'pkgname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 30), tuple_44917, pkgname_44918)
    # Adding element type (line 324)
    
    # Obtaining the type of the subscript
    str_44919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 44), 'str', 'name')
    # Getting the type of 'meta' (line 324)
    meta_44920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 39), 'meta', False)
    # Obtaining the member '__getitem__' of a type (line 324)
    getitem___44921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 39), meta_44920, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 324)
    subscript_call_result_44922 = invoke(stypy.reporting.localization.Localization(__file__, 324, 39), getitem___44921, str_44919)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 324, 30), tuple_44917, subscript_call_result_44922)
    
    # Applying the binary operator '%' (line 323)
    result_mod_44923 = python_operator(stypy.reporting.localization.Localization(__file__, 323, 29), '%', str_44916, tuple_44917)
    
    # Processing the call keyword arguments (line 323)
    kwargs_44924 = {}
    # Getting the type of 'ValueError' (line 323)
    ValueError_44915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 323)
    ValueError_call_result_44925 = invoke(stypy.reporting.localization.Localization(__file__, 323, 18), ValueError_44915, *[result_mod_44923], **kwargs_44924)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 323, 12), ValueError_call_result_44925, 'raise parameter', BaseException)
    # SSA join for if statement (line 322)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 326):
    
    # Assigning a Subscript to a Name (line 326):
    
    # Obtaining the type of the subscript
    # Getting the type of 'pkgname' (line 326)
    pkgname_44926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 26), 'pkgname')
    # Getting the type of 'sys' (line 326)
    sys_44927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 14), 'sys')
    # Obtaining the member 'modules' of a type (line 326)
    modules_44928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 14), sys_44927, 'modules')
    # Obtaining the member '__getitem__' of a type (line 326)
    getitem___44929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 14), modules_44928, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 326)
    subscript_call_result_44930 = invoke(stypy.reporting.localization.Localization(__file__, 326, 14), getitem___44929, pkgname_44926)
    
    # Assigning a type to the variable 'mod' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'mod', subscript_call_result_44930)
    
    # Assigning a Call to a Subscript (line 327):
    
    # Assigning a Call to a Subscript (line 327):
    
    # Call to _escape_backslash(...): (line 327)
    # Processing the call arguments (line 327)
    
    # Call to dirname(...): (line 327)
    # Processing the call arguments (line 327)
    # Getting the type of 'mod' (line 327)
    mod_44935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 59), 'mod', False)
    # Obtaining the member '__file__' of a type (line 327)
    file___44936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 59), mod_44935, '__file__')
    # Processing the call keyword arguments (line 327)
    kwargs_44937 = {}
    # Getting the type of 'os' (line 327)
    os_44932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 43), 'os', False)
    # Obtaining the member 'path' of a type (line 327)
    path_44933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 43), os_44932, 'path')
    # Obtaining the member 'dirname' of a type (line 327)
    dirname_44934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 43), path_44933, 'dirname')
    # Calling dirname(args, kwargs) (line 327)
    dirname_call_result_44938 = invoke(stypy.reporting.localization.Localization(__file__, 327, 43), dirname_44934, *[file___44936], **kwargs_44937)
    
    # Processing the call keyword arguments (line 327)
    kwargs_44939 = {}
    # Getting the type of '_escape_backslash' (line 327)
    _escape_backslash_44931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 25), '_escape_backslash', False)
    # Calling _escape_backslash(args, kwargs) (line 327)
    _escape_backslash_call_result_44940 = invoke(stypy.reporting.localization.Localization(__file__, 327, 25), _escape_backslash_44931, *[dirname_call_result_44938], **kwargs_44939)
    
    # Getting the type of 'vars' (line 327)
    vars_44941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'vars')
    str_44942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 13), 'str', 'pkgdir')
    # Storing an element on a container (line 327)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 327, 8), vars_44941, (str_44942, _escape_backslash_call_result_44940))
    # SSA join for if statement (line 320)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to LibraryInfo(...): (line 329)
    # Processing the call keyword arguments (line 329)
    
    # Obtaining the type of the subscript
    str_44944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 33), 'str', 'name')
    # Getting the type of 'meta' (line 329)
    meta_44945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 28), 'meta', False)
    # Obtaining the member '__getitem__' of a type (line 329)
    getitem___44946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 28), meta_44945, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 329)
    subscript_call_result_44947 = invoke(stypy.reporting.localization.Localization(__file__, 329, 28), getitem___44946, str_44944)
    
    keyword_44948 = subscript_call_result_44947
    
    # Obtaining the type of the subscript
    str_44949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 59), 'str', 'description')
    # Getting the type of 'meta' (line 329)
    meta_44950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 54), 'meta', False)
    # Obtaining the member '__getitem__' of a type (line 329)
    getitem___44951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 54), meta_44950, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 329)
    subscript_call_result_44952 = invoke(stypy.reporting.localization.Localization(__file__, 329, 54), getitem___44951, str_44949)
    
    keyword_44953 = subscript_call_result_44952
    
    # Obtaining the type of the subscript
    str_44954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 330, 25), 'str', 'version')
    # Getting the type of 'meta' (line 330)
    meta_44955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 20), 'meta', False)
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___44956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 20), meta_44955, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_44957 = invoke(stypy.reporting.localization.Localization(__file__, 330, 20), getitem___44956, str_44954)
    
    keyword_44958 = subscript_call_result_44957
    # Getting the type of 'sections' (line 330)
    sections_44959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 46), 'sections', False)
    keyword_44960 = sections_44959
    
    # Call to VariableSet(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'vars' (line 330)
    vars_44962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 73), 'vars', False)
    # Processing the call keyword arguments (line 330)
    kwargs_44963 = {}
    # Getting the type of 'VariableSet' (line 330)
    VariableSet_44961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 61), 'VariableSet', False)
    # Calling VariableSet(args, kwargs) (line 330)
    VariableSet_call_result_44964 = invoke(stypy.reporting.localization.Localization(__file__, 330, 61), VariableSet_44961, *[vars_44962], **kwargs_44963)
    
    keyword_44965 = VariableSet_call_result_44964
    kwargs_44966 = {'version': keyword_44958, 'sections': keyword_44960, 'name': keyword_44948, 'vars': keyword_44965, 'description': keyword_44953}
    # Getting the type of 'LibraryInfo' (line 329)
    LibraryInfo_44943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 11), 'LibraryInfo', False)
    # Calling LibraryInfo(args, kwargs) (line 329)
    LibraryInfo_call_result_44967 = invoke(stypy.reporting.localization.Localization(__file__, 329, 11), LibraryInfo_44943, *[], **kwargs_44966)
    
    # Assigning a type to the variable 'stypy_return_type' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'stypy_return_type', LibraryInfo_call_result_44967)
    
    # ################# End of '_read_config_imp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_read_config_imp' in the type store
    # Getting the type of 'stypy_return_type' (line 296)
    stypy_return_type_44968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44968)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_read_config_imp'
    return stypy_return_type_44968

# Assigning a type to the variable '_read_config_imp' (line 296)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 0), '_read_config_imp', _read_config_imp)

# Assigning a Dict to a Name (line 336):

# Assigning a Dict to a Name (line 336):

# Obtaining an instance of the builtin type 'dict' (line 336)
dict_44969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 336)

# Assigning a type to the variable '_CACHE' (line 336)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 0), '_CACHE', dict_44969)

@norecursion
def read_config(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 337)
    None_44970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 30), 'None')
    defaults = [None_44970]
    # Create a new context for function 'read_config'
    module_type_store = module_type_store.open_function_context('read_config', 337, 0, False)
    
    # Passed parameters checking function
    read_config.stypy_localization = localization
    read_config.stypy_type_of_self = None
    read_config.stypy_type_store = module_type_store
    read_config.stypy_function_name = 'read_config'
    read_config.stypy_param_names_list = ['pkgname', 'dirs']
    read_config.stypy_varargs_param_name = None
    read_config.stypy_kwargs_param_name = None
    read_config.stypy_call_defaults = defaults
    read_config.stypy_call_varargs = varargs
    read_config.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'read_config', ['pkgname', 'dirs'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'read_config', localization, ['pkgname', 'dirs'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'read_config(...)' code ##################

    str_44971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, (-1)), 'str', "\n    Return library info for a package from its configuration file.\n\n    Parameters\n    ----------\n    pkgname : str\n        Name of the package (should match the name of the .ini file, without\n        the extension, e.g. foo for the file foo.ini).\n    dirs : sequence, optional\n        If given, should be a sequence of directories - usually including\n        the NumPy base directory - where to look for npy-pkg-config files.\n\n    Returns\n    -------\n    pkginfo : class instance\n        The `LibraryInfo` instance containing the build information.\n\n    Raises\n    ------\n    PkgNotFound\n        If the package is not found.\n\n    See Also\n    --------\n    misc_util.get_info, misc_util.get_pkg_info\n\n    Examples\n    --------\n    >>> npymath_info = np.distutils.npy_pkg_config.read_config('npymath')\n    >>> type(npymath_info)\n    <class 'numpy.distutils.npy_pkg_config.LibraryInfo'>\n    >>> print(npymath_info)\n    Name: npymath\n    Description: Portable, core math library implementing C99 standard\n    Requires:\n    Version: 0.1  #random\n\n    ")
    
    
    # SSA begins for try-except statement (line 376)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    # Getting the type of 'pkgname' (line 377)
    pkgname_44972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 22), 'pkgname')
    # Getting the type of '_CACHE' (line 377)
    _CACHE_44973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 15), '_CACHE')
    # Obtaining the member '__getitem__' of a type (line 377)
    getitem___44974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 15), _CACHE_44973, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 377)
    subscript_call_result_44975 = invoke(stypy.reporting.localization.Localization(__file__, 377, 15), getitem___44974, pkgname_44972)
    
    # Assigning a type to the variable 'stypy_return_type' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'stypy_return_type', subscript_call_result_44975)
    # SSA branch for the except part of a try statement (line 376)
    # SSA branch for the except 'KeyError' branch of a try statement (line 376)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 379):
    
    # Assigning a Call to a Name (line 379):
    
    # Call to _read_config_imp(...): (line 379)
    # Processing the call arguments (line 379)
    
    # Call to pkg_to_filename(...): (line 379)
    # Processing the call arguments (line 379)
    # Getting the type of 'pkgname' (line 379)
    pkgname_44978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 45), 'pkgname', False)
    # Processing the call keyword arguments (line 379)
    kwargs_44979 = {}
    # Getting the type of 'pkg_to_filename' (line 379)
    pkg_to_filename_44977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 29), 'pkg_to_filename', False)
    # Calling pkg_to_filename(args, kwargs) (line 379)
    pkg_to_filename_call_result_44980 = invoke(stypy.reporting.localization.Localization(__file__, 379, 29), pkg_to_filename_44977, *[pkgname_44978], **kwargs_44979)
    
    # Getting the type of 'dirs' (line 379)
    dirs_44981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 55), 'dirs', False)
    # Processing the call keyword arguments (line 379)
    kwargs_44982 = {}
    # Getting the type of '_read_config_imp' (line 379)
    _read_config_imp_44976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), '_read_config_imp', False)
    # Calling _read_config_imp(args, kwargs) (line 379)
    _read_config_imp_call_result_44983 = invoke(stypy.reporting.localization.Localization(__file__, 379, 12), _read_config_imp_44976, *[pkg_to_filename_call_result_44980, dirs_44981], **kwargs_44982)
    
    # Assigning a type to the variable 'v' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'v', _read_config_imp_call_result_44983)
    
    # Assigning a Name to a Subscript (line 380):
    
    # Assigning a Name to a Subscript (line 380):
    # Getting the type of 'v' (line 380)
    v_44984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 26), 'v')
    # Getting the type of '_CACHE' (line 380)
    _CACHE_44985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), '_CACHE')
    # Getting the type of 'pkgname' (line 380)
    pkgname_44986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 15), 'pkgname')
    # Storing an element on a container (line 380)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 380, 8), _CACHE_44985, (pkgname_44986, v_44984))
    # Getting the type of 'v' (line 381)
    v_44987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 15), 'v')
    # Assigning a type to the variable 'stypy_return_type' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 8), 'stypy_return_type', v_44987)
    # SSA join for try-except statement (line 376)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'read_config(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'read_config' in the type store
    # Getting the type of 'stypy_return_type' (line 337)
    stypy_return_type_44988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_44988)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'read_config'
    return stypy_return_type_44988

# Assigning a type to the variable 'read_config' (line 337)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 0), 'read_config', read_config)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 389, 4))
    
    # 'import sys' statement (line 389)
    import sys

    import_module(stypy.reporting.localization.Localization(__file__, 389, 4), 'sys', sys, module_type_store)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 390, 4))
    
    # 'from optparse import OptionParser' statement (line 390)
    from optparse import OptionParser

    import_from_module(stypy.reporting.localization.Localization(__file__, 390, 4), 'optparse', None, module_type_store, ['OptionParser'], [OptionParser])
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 391, 4))
    
    # 'import glob' statement (line 391)
    import glob

    import_module(stypy.reporting.localization.Localization(__file__, 391, 4), 'glob', glob, module_type_store)
    
    
    # Assigning a Call to a Name (line 393):
    
    # Assigning a Call to a Name (line 393):
    
    # Call to OptionParser(...): (line 393)
    # Processing the call keyword arguments (line 393)
    kwargs_44990 = {}
    # Getting the type of 'OptionParser' (line 393)
    OptionParser_44989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 13), 'OptionParser', False)
    # Calling OptionParser(args, kwargs) (line 393)
    OptionParser_call_result_44991 = invoke(stypy.reporting.localization.Localization(__file__, 393, 13), OptionParser_44989, *[], **kwargs_44990)
    
    # Assigning a type to the variable 'parser' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'parser', OptionParser_call_result_44991)
    
    # Call to add_option(...): (line 394)
    # Processing the call arguments (line 394)
    str_44994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 22), 'str', '--cflags')
    # Processing the call keyword arguments (line 394)
    str_44995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 39), 'str', 'cflags')
    keyword_44996 = str_44995
    str_44997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 56), 'str', 'store_true')
    keyword_44998 = str_44997
    str_44999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 27), 'str', 'output all preprocessor and compiler flags')
    keyword_45000 = str_44999
    kwargs_45001 = {'dest': keyword_44996, 'action': keyword_44998, 'help': keyword_45000}
    # Getting the type of 'parser' (line 394)
    parser_44992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'parser', False)
    # Obtaining the member 'add_option' of a type (line 394)
    add_option_44993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 4), parser_44992, 'add_option')
    # Calling add_option(args, kwargs) (line 394)
    add_option_call_result_45002 = invoke(stypy.reporting.localization.Localization(__file__, 394, 4), add_option_44993, *[str_44994], **kwargs_45001)
    
    
    # Call to add_option(...): (line 396)
    # Processing the call arguments (line 396)
    str_45005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 22), 'str', '--libs')
    # Processing the call keyword arguments (line 396)
    str_45006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 37), 'str', 'libs')
    keyword_45007 = str_45006
    str_45008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 52), 'str', 'store_true')
    keyword_45009 = str_45008
    str_45010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 27), 'str', 'output all linker flags')
    keyword_45011 = str_45010
    kwargs_45012 = {'dest': keyword_45007, 'action': keyword_45009, 'help': keyword_45011}
    # Getting the type of 'parser' (line 396)
    parser_45003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'parser', False)
    # Obtaining the member 'add_option' of a type (line 396)
    add_option_45004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 4), parser_45003, 'add_option')
    # Calling add_option(args, kwargs) (line 396)
    add_option_call_result_45013 = invoke(stypy.reporting.localization.Localization(__file__, 396, 4), add_option_45004, *[str_45005], **kwargs_45012)
    
    
    # Call to add_option(...): (line 398)
    # Processing the call arguments (line 398)
    str_45016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 22), 'str', '--use-section')
    # Processing the call keyword arguments (line 398)
    str_45017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 44), 'str', 'section')
    keyword_45018 = str_45017
    str_45019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 27), 'str', 'use this section instead of default for options')
    keyword_45020 = str_45019
    kwargs_45021 = {'dest': keyword_45018, 'help': keyword_45020}
    # Getting the type of 'parser' (line 398)
    parser_45014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'parser', False)
    # Obtaining the member 'add_option' of a type (line 398)
    add_option_45015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 4), parser_45014, 'add_option')
    # Calling add_option(args, kwargs) (line 398)
    add_option_call_result_45022 = invoke(stypy.reporting.localization.Localization(__file__, 398, 4), add_option_45015, *[str_45016], **kwargs_45021)
    
    
    # Call to add_option(...): (line 400)
    # Processing the call arguments (line 400)
    str_45025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 22), 'str', '--version')
    # Processing the call keyword arguments (line 400)
    str_45026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 40), 'str', 'version')
    keyword_45027 = str_45026
    str_45028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 58), 'str', 'store_true')
    keyword_45029 = str_45028
    str_45030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 27), 'str', 'output version')
    keyword_45031 = str_45030
    kwargs_45032 = {'dest': keyword_45027, 'action': keyword_45029, 'help': keyword_45031}
    # Getting the type of 'parser' (line 400)
    parser_45023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'parser', False)
    # Obtaining the member 'add_option' of a type (line 400)
    add_option_45024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 4), parser_45023, 'add_option')
    # Calling add_option(args, kwargs) (line 400)
    add_option_call_result_45033 = invoke(stypy.reporting.localization.Localization(__file__, 400, 4), add_option_45024, *[str_45025], **kwargs_45032)
    
    
    # Call to add_option(...): (line 402)
    # Processing the call arguments (line 402)
    str_45036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 22), 'str', '--atleast-version')
    # Processing the call keyword arguments (line 402)
    str_45037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 48), 'str', 'min_version')
    keyword_45038 = str_45037
    str_45039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 27), 'str', 'Minimal version')
    keyword_45040 = str_45039
    kwargs_45041 = {'dest': keyword_45038, 'help': keyword_45040}
    # Getting the type of 'parser' (line 402)
    parser_45034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'parser', False)
    # Obtaining the member 'add_option' of a type (line 402)
    add_option_45035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 4), parser_45034, 'add_option')
    # Calling add_option(args, kwargs) (line 402)
    add_option_call_result_45042 = invoke(stypy.reporting.localization.Localization(__file__, 402, 4), add_option_45035, *[str_45036], **kwargs_45041)
    
    
    # Call to add_option(...): (line 404)
    # Processing the call arguments (line 404)
    str_45045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 22), 'str', '--list-all')
    # Processing the call keyword arguments (line 404)
    str_45046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 41), 'str', 'list_all')
    keyword_45047 = str_45046
    str_45048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 60), 'str', 'store_true')
    keyword_45049 = str_45048
    str_45050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 27), 'str', 'Minimal version')
    keyword_45051 = str_45050
    kwargs_45052 = {'dest': keyword_45047, 'action': keyword_45049, 'help': keyword_45051}
    # Getting the type of 'parser' (line 404)
    parser_45043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'parser', False)
    # Obtaining the member 'add_option' of a type (line 404)
    add_option_45044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 4), parser_45043, 'add_option')
    # Calling add_option(args, kwargs) (line 404)
    add_option_call_result_45053 = invoke(stypy.reporting.localization.Localization(__file__, 404, 4), add_option_45044, *[str_45045], **kwargs_45052)
    
    
    # Call to add_option(...): (line 406)
    # Processing the call arguments (line 406)
    str_45056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 22), 'str', '--define-variable')
    # Processing the call keyword arguments (line 406)
    str_45057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 48), 'str', 'define_variable')
    keyword_45058 = str_45057
    str_45059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 27), 'str', 'Replace variable with the given value')
    keyword_45060 = str_45059
    kwargs_45061 = {'dest': keyword_45058, 'help': keyword_45060}
    # Getting the type of 'parser' (line 406)
    parser_45054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'parser', False)
    # Obtaining the member 'add_option' of a type (line 406)
    add_option_45055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 4), parser_45054, 'add_option')
    # Calling add_option(args, kwargs) (line 406)
    add_option_call_result_45062 = invoke(stypy.reporting.localization.Localization(__file__, 406, 4), add_option_45055, *[str_45056], **kwargs_45061)
    
    
    # Assigning a Call to a Tuple (line 409):
    
    # Assigning a Call to a Name:
    
    # Call to parse_args(...): (line 409)
    # Processing the call arguments (line 409)
    # Getting the type of 'sys' (line 409)
    sys_45065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 40), 'sys', False)
    # Obtaining the member 'argv' of a type (line 409)
    argv_45066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 40), sys_45065, 'argv')
    # Processing the call keyword arguments (line 409)
    kwargs_45067 = {}
    # Getting the type of 'parser' (line 409)
    parser_45063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 22), 'parser', False)
    # Obtaining the member 'parse_args' of a type (line 409)
    parse_args_45064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 22), parser_45063, 'parse_args')
    # Calling parse_args(args, kwargs) (line 409)
    parse_args_call_result_45068 = invoke(stypy.reporting.localization.Localization(__file__, 409, 22), parse_args_45064, *[argv_45066], **kwargs_45067)
    
    # Assigning a type to the variable 'call_assignment_44151' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'call_assignment_44151', parse_args_call_result_45068)
    
    # Assigning a Call to a Name (line 409):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_45071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 4), 'int')
    # Processing the call keyword arguments
    kwargs_45072 = {}
    # Getting the type of 'call_assignment_44151' (line 409)
    call_assignment_44151_45069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'call_assignment_44151', False)
    # Obtaining the member '__getitem__' of a type (line 409)
    getitem___45070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 4), call_assignment_44151_45069, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_45073 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___45070, *[int_45071], **kwargs_45072)
    
    # Assigning a type to the variable 'call_assignment_44152' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'call_assignment_44152', getitem___call_result_45073)
    
    # Assigning a Name to a Name (line 409):
    # Getting the type of 'call_assignment_44152' (line 409)
    call_assignment_44152_45074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'call_assignment_44152')
    # Assigning a type to the variable 'options' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 5), 'options', call_assignment_44152_45074)
    
    # Assigning a Call to a Name (line 409):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_45077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 4), 'int')
    # Processing the call keyword arguments
    kwargs_45078 = {}
    # Getting the type of 'call_assignment_44151' (line 409)
    call_assignment_44151_45075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'call_assignment_44151', False)
    # Obtaining the member '__getitem__' of a type (line 409)
    getitem___45076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 4), call_assignment_44151_45075, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_45079 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___45076, *[int_45077], **kwargs_45078)
    
    # Assigning a type to the variable 'call_assignment_44153' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'call_assignment_44153', getitem___call_result_45079)
    
    # Assigning a Name to a Name (line 409):
    # Getting the type of 'call_assignment_44153' (line 409)
    call_assignment_44153_45080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'call_assignment_44153')
    # Assigning a type to the variable 'args' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 14), 'args', call_assignment_44153_45080)
    
    
    
    # Call to len(...): (line 411)
    # Processing the call arguments (line 411)
    # Getting the type of 'args' (line 411)
    args_45082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 11), 'args', False)
    # Processing the call keyword arguments (line 411)
    kwargs_45083 = {}
    # Getting the type of 'len' (line 411)
    len_45081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 7), 'len', False)
    # Calling len(args, kwargs) (line 411)
    len_call_result_45084 = invoke(stypy.reporting.localization.Localization(__file__, 411, 7), len_45081, *[args_45082], **kwargs_45083)
    
    int_45085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 19), 'int')
    # Applying the binary operator '<' (line 411)
    result_lt_45086 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 7), '<', len_call_result_45084, int_45085)
    
    # Testing the type of an if condition (line 411)
    if_condition_45087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 411, 4), result_lt_45086)
    # Assigning a type to the variable 'if_condition_45087' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'if_condition_45087', if_condition_45087)
    # SSA begins for if statement (line 411)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 412)
    # Processing the call arguments (line 412)
    str_45089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 25), 'str', 'Expect package name on the command line:')
    # Processing the call keyword arguments (line 412)
    kwargs_45090 = {}
    # Getting the type of 'ValueError' (line 412)
    ValueError_45088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 412)
    ValueError_call_result_45091 = invoke(stypy.reporting.localization.Localization(__file__, 412, 14), ValueError_45088, *[str_45089], **kwargs_45090)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 412, 8), ValueError_call_result_45091, 'raise parameter', BaseException)
    # SSA join for if statement (line 411)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'options' (line 414)
    options_45092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 7), 'options')
    # Obtaining the member 'list_all' of a type (line 414)
    list_all_45093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 7), options_45092, 'list_all')
    # Testing the type of an if condition (line 414)
    if_condition_45094 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 414, 4), list_all_45093)
    # Assigning a type to the variable 'if_condition_45094' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'if_condition_45094', if_condition_45094)
    # SSA begins for if statement (line 414)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 415):
    
    # Assigning a Call to a Name (line 415):
    
    # Call to glob(...): (line 415)
    # Processing the call arguments (line 415)
    str_45097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 26), 'str', '*.ini')
    # Processing the call keyword arguments (line 415)
    kwargs_45098 = {}
    # Getting the type of 'glob' (line 415)
    glob_45095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 16), 'glob', False)
    # Obtaining the member 'glob' of a type (line 415)
    glob_45096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 16), glob_45095, 'glob')
    # Calling glob(args, kwargs) (line 415)
    glob_call_result_45099 = invoke(stypy.reporting.localization.Localization(__file__, 415, 16), glob_45096, *[str_45097], **kwargs_45098)
    
    # Assigning a type to the variable 'files' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'files', glob_call_result_45099)
    
    # Getting the type of 'files' (line 416)
    files_45100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 17), 'files')
    # Testing the type of a for loop iterable (line 416)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 416, 8), files_45100)
    # Getting the type of the for loop variable (line 416)
    for_loop_var_45101 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 416, 8), files_45100)
    # Assigning a type to the variable 'f' (line 416)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'f', for_loop_var_45101)
    # SSA begins for a for statement (line 416)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 417):
    
    # Assigning a Call to a Name (line 417):
    
    # Call to read_config(...): (line 417)
    # Processing the call arguments (line 417)
    # Getting the type of 'f' (line 417)
    f_45103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 31), 'f', False)
    # Processing the call keyword arguments (line 417)
    kwargs_45104 = {}
    # Getting the type of 'read_config' (line 417)
    read_config_45102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 19), 'read_config', False)
    # Calling read_config(args, kwargs) (line 417)
    read_config_call_result_45105 = invoke(stypy.reporting.localization.Localization(__file__, 417, 19), read_config_45102, *[f_45103], **kwargs_45104)
    
    # Assigning a type to the variable 'info' (line 417)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 12), 'info', read_config_call_result_45105)
    
    # Call to print(...): (line 418)
    # Processing the call arguments (line 418)
    str_45107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 18), 'str', '%s\t%s - %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 418)
    tuple_45108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 418)
    # Adding element type (line 418)
    # Getting the type of 'info' (line 418)
    info_45109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 35), 'info', False)
    # Obtaining the member 'name' of a type (line 418)
    name_45110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 35), info_45109, 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 35), tuple_45108, name_45110)
    # Adding element type (line 418)
    # Getting the type of 'info' (line 418)
    info_45111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 46), 'info', False)
    # Obtaining the member 'name' of a type (line 418)
    name_45112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 46), info_45111, 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 35), tuple_45108, name_45112)
    # Adding element type (line 418)
    # Getting the type of 'info' (line 418)
    info_45113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 57), 'info', False)
    # Obtaining the member 'description' of a type (line 418)
    description_45114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 57), info_45113, 'description')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 418, 35), tuple_45108, description_45114)
    
    # Applying the binary operator '%' (line 418)
    result_mod_45115 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 18), '%', str_45107, tuple_45108)
    
    # Processing the call keyword arguments (line 418)
    kwargs_45116 = {}
    # Getting the type of 'print' (line 418)
    print_45106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 12), 'print', False)
    # Calling print(args, kwargs) (line 418)
    print_call_result_45117 = invoke(stypy.reporting.localization.Localization(__file__, 418, 12), print_45106, *[result_mod_45115], **kwargs_45116)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 414)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 420):
    
    # Assigning a Subscript to a Name (line 420):
    
    # Obtaining the type of the subscript
    int_45118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 20), 'int')
    # Getting the type of 'args' (line 420)
    args_45119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 15), 'args')
    # Obtaining the member '__getitem__' of a type (line 420)
    getitem___45120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 15), args_45119, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 420)
    subscript_call_result_45121 = invoke(stypy.reporting.localization.Localization(__file__, 420, 15), getitem___45120, int_45118)
    
    # Assigning a type to the variable 'pkg_name' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'pkg_name', subscript_call_result_45121)
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 421, 4))
    
    # 'import os' statement (line 421)
    import os

    import_module(stypy.reporting.localization.Localization(__file__, 421, 4), 'os', os, module_type_store)
    
    
    # Assigning a Call to a Name (line 422):
    
    # Assigning a Call to a Name (line 422):
    
    # Call to get(...): (line 422)
    # Processing the call arguments (line 422)
    str_45125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 23), 'str', 'NPY_PKG_CONFIG_PATH')
    # Processing the call keyword arguments (line 422)
    kwargs_45126 = {}
    # Getting the type of 'os' (line 422)
    os_45122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'os', False)
    # Obtaining the member 'environ' of a type (line 422)
    environ_45123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 8), os_45122, 'environ')
    # Obtaining the member 'get' of a type (line 422)
    get_45124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 8), environ_45123, 'get')
    # Calling get(args, kwargs) (line 422)
    get_call_result_45127 = invoke(stypy.reporting.localization.Localization(__file__, 422, 8), get_45124, *[str_45125], **kwargs_45126)
    
    # Assigning a type to the variable 'd' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'd', get_call_result_45127)
    
    # Getting the type of 'd' (line 423)
    d_45128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 7), 'd')
    # Testing the type of an if condition (line 423)
    if_condition_45129 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 423, 4), d_45128)
    # Assigning a type to the variable 'if_condition_45129' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'if_condition_45129', if_condition_45129)
    # SSA begins for if statement (line 423)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 424):
    
    # Assigning a Call to a Name (line 424):
    
    # Call to read_config(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'pkg_name' (line 424)
    pkg_name_45131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 27), 'pkg_name', False)
    
    # Obtaining an instance of the builtin type 'list' (line 424)
    list_45132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 424)
    # Adding element type (line 424)
    str_45133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 38), 'str', 'numpy/core/lib/npy-pkg-config')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 37), list_45132, str_45133)
    # Adding element type (line 424)
    str_45134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 71), 'str', '.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 37), list_45132, str_45134)
    # Adding element type (line 424)
    # Getting the type of 'd' (line 424)
    d_45135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 76), 'd', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 37), list_45132, d_45135)
    
    # Processing the call keyword arguments (line 424)
    kwargs_45136 = {}
    # Getting the type of 'read_config' (line 424)
    read_config_45130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 15), 'read_config', False)
    # Calling read_config(args, kwargs) (line 424)
    read_config_call_result_45137 = invoke(stypy.reporting.localization.Localization(__file__, 424, 15), read_config_45130, *[pkg_name_45131, list_45132], **kwargs_45136)
    
    # Assigning a type to the variable 'info' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'info', read_config_call_result_45137)
    # SSA branch for the else part of an if statement (line 423)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 426):
    
    # Assigning a Call to a Name (line 426):
    
    # Call to read_config(...): (line 426)
    # Processing the call arguments (line 426)
    # Getting the type of 'pkg_name' (line 426)
    pkg_name_45139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 27), 'pkg_name', False)
    
    # Obtaining an instance of the builtin type 'list' (line 426)
    list_45140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 426)
    # Adding element type (line 426)
    str_45141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 38), 'str', 'numpy/core/lib/npy-pkg-config')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 37), list_45140, str_45141)
    # Adding element type (line 426)
    str_45142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 71), 'str', '.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 37), list_45140, str_45142)
    
    # Processing the call keyword arguments (line 426)
    kwargs_45143 = {}
    # Getting the type of 'read_config' (line 426)
    read_config_45138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 15), 'read_config', False)
    # Calling read_config(args, kwargs) (line 426)
    read_config_call_result_45144 = invoke(stypy.reporting.localization.Localization(__file__, 426, 15), read_config_45138, *[pkg_name_45139, list_45140], **kwargs_45143)
    
    # Assigning a type to the variable 'info' (line 426)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'info', read_config_call_result_45144)
    # SSA join for if statement (line 423)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'options' (line 428)
    options_45145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 7), 'options')
    # Obtaining the member 'section' of a type (line 428)
    section_45146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 7), options_45145, 'section')
    # Testing the type of an if condition (line 428)
    if_condition_45147 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 428, 4), section_45146)
    # Assigning a type to the variable 'if_condition_45147' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'if_condition_45147', if_condition_45147)
    # SSA begins for if statement (line 428)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 429):
    
    # Assigning a Attribute to a Name (line 429):
    # Getting the type of 'options' (line 429)
    options_45148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 18), 'options')
    # Obtaining the member 'section' of a type (line 429)
    section_45149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 18), options_45148, 'section')
    # Assigning a type to the variable 'section' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'section', section_45149)
    # SSA branch for the else part of an if statement (line 428)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 431):
    
    # Assigning a Str to a Name (line 431):
    str_45150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 18), 'str', 'default')
    # Assigning a type to the variable 'section' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'section', str_45150)
    # SSA join for if statement (line 428)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'options' (line 433)
    options_45151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 7), 'options')
    # Obtaining the member 'define_variable' of a type (line 433)
    define_variable_45152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 7), options_45151, 'define_variable')
    # Testing the type of an if condition (line 433)
    if_condition_45153 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 433, 4), define_variable_45152)
    # Assigning a type to the variable 'if_condition_45153' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'if_condition_45153', if_condition_45153)
    # SSA begins for if statement (line 433)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 434):
    
    # Assigning a Call to a Name (line 434):
    
    # Call to search(...): (line 434)
    # Processing the call arguments (line 434)
    str_45156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 22), 'str', '([\\S]+)=([\\S]+)')
    # Getting the type of 'options' (line 434)
    options_45157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 41), 'options', False)
    # Obtaining the member 'define_variable' of a type (line 434)
    define_variable_45158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 41), options_45157, 'define_variable')
    # Processing the call keyword arguments (line 434)
    kwargs_45159 = {}
    # Getting the type of 're' (line 434)
    re_45154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 12), 're', False)
    # Obtaining the member 'search' of a type (line 434)
    search_45155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 12), re_45154, 'search')
    # Calling search(args, kwargs) (line 434)
    search_call_result_45160 = invoke(stypy.reporting.localization.Localization(__file__, 434, 12), search_45155, *[str_45156, define_variable_45158], **kwargs_45159)
    
    # Assigning a type to the variable 'm' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'm', search_call_result_45160)
    
    
    # Getting the type of 'm' (line 435)
    m_45161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 15), 'm')
    # Applying the 'not' unary operator (line 435)
    result_not__45162 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 11), 'not', m_45161)
    
    # Testing the type of an if condition (line 435)
    if_condition_45163 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 435, 8), result_not__45162)
    # Assigning a type to the variable 'if_condition_45163' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'if_condition_45163', if_condition_45163)
    # SSA begins for if statement (line 435)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 436)
    # Processing the call arguments (line 436)
    str_45165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 29), 'str', '--define-variable option should be of the form --define-variable=foo=bar')
    # Processing the call keyword arguments (line 436)
    kwargs_45166 = {}
    # Getting the type of 'ValueError' (line 436)
    ValueError_45164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 436)
    ValueError_call_result_45167 = invoke(stypy.reporting.localization.Localization(__file__, 436, 18), ValueError_45164, *[str_45165], **kwargs_45166)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 436, 12), ValueError_call_result_45167, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 435)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 439):
    
    # Assigning a Call to a Name (line 439):
    
    # Call to group(...): (line 439)
    # Processing the call arguments (line 439)
    int_45170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 27), 'int')
    # Processing the call keyword arguments (line 439)
    kwargs_45171 = {}
    # Getting the type of 'm' (line 439)
    m_45168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 19), 'm', False)
    # Obtaining the member 'group' of a type (line 439)
    group_45169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 19), m_45168, 'group')
    # Calling group(args, kwargs) (line 439)
    group_call_result_45172 = invoke(stypy.reporting.localization.Localization(__file__, 439, 19), group_45169, *[int_45170], **kwargs_45171)
    
    # Assigning a type to the variable 'name' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 12), 'name', group_call_result_45172)
    
    # Assigning a Call to a Name (line 440):
    
    # Assigning a Call to a Name (line 440):
    
    # Call to group(...): (line 440)
    # Processing the call arguments (line 440)
    int_45175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 440, 28), 'int')
    # Processing the call keyword arguments (line 440)
    kwargs_45176 = {}
    # Getting the type of 'm' (line 440)
    m_45173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 20), 'm', False)
    # Obtaining the member 'group' of a type (line 440)
    group_45174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 440, 20), m_45173, 'group')
    # Calling group(args, kwargs) (line 440)
    group_call_result_45177 = invoke(stypy.reporting.localization.Localization(__file__, 440, 20), group_45174, *[int_45175], **kwargs_45176)
    
    # Assigning a type to the variable 'value' (line 440)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 440, 12), 'value', group_call_result_45177)
    # SSA join for if statement (line 435)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 441):
    
    # Assigning a Name to a Subscript (line 441):
    # Getting the type of 'value' (line 441)
    value_45178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 26), 'value')
    # Getting the type of 'info' (line 441)
    info_45179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'info')
    # Obtaining the member 'vars' of a type (line 441)
    vars_45180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 8), info_45179, 'vars')
    # Getting the type of 'name' (line 441)
    name_45181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 18), 'name')
    # Storing an element on a container (line 441)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 441, 8), vars_45180, (name_45181, value_45178))
    # SSA join for if statement (line 433)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'options' (line 443)
    options_45182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 7), 'options')
    # Obtaining the member 'cflags' of a type (line 443)
    cflags_45183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 7), options_45182, 'cflags')
    # Testing the type of an if condition (line 443)
    if_condition_45184 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 443, 4), cflags_45183)
    # Assigning a type to the variable 'if_condition_45184' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'if_condition_45184', if_condition_45184)
    # SSA begins for if statement (line 443)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 444)
    # Processing the call arguments (line 444)
    
    # Call to cflags(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'section' (line 444)
    section_45188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 26), 'section', False)
    # Processing the call keyword arguments (line 444)
    kwargs_45189 = {}
    # Getting the type of 'info' (line 444)
    info_45186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 14), 'info', False)
    # Obtaining the member 'cflags' of a type (line 444)
    cflags_45187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 14), info_45186, 'cflags')
    # Calling cflags(args, kwargs) (line 444)
    cflags_call_result_45190 = invoke(stypy.reporting.localization.Localization(__file__, 444, 14), cflags_45187, *[section_45188], **kwargs_45189)
    
    # Processing the call keyword arguments (line 444)
    kwargs_45191 = {}
    # Getting the type of 'print' (line 444)
    print_45185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'print', False)
    # Calling print(args, kwargs) (line 444)
    print_call_result_45192 = invoke(stypy.reporting.localization.Localization(__file__, 444, 8), print_45185, *[cflags_call_result_45190], **kwargs_45191)
    
    # SSA join for if statement (line 443)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'options' (line 445)
    options_45193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 7), 'options')
    # Obtaining the member 'libs' of a type (line 445)
    libs_45194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 7), options_45193, 'libs')
    # Testing the type of an if condition (line 445)
    if_condition_45195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 445, 4), libs_45194)
    # Assigning a type to the variable 'if_condition_45195' (line 445)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 4), 'if_condition_45195', if_condition_45195)
    # SSA begins for if statement (line 445)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 446)
    # Processing the call arguments (line 446)
    
    # Call to libs(...): (line 446)
    # Processing the call arguments (line 446)
    # Getting the type of 'section' (line 446)
    section_45199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 24), 'section', False)
    # Processing the call keyword arguments (line 446)
    kwargs_45200 = {}
    # Getting the type of 'info' (line 446)
    info_45197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 14), 'info', False)
    # Obtaining the member 'libs' of a type (line 446)
    libs_45198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 14), info_45197, 'libs')
    # Calling libs(args, kwargs) (line 446)
    libs_call_result_45201 = invoke(stypy.reporting.localization.Localization(__file__, 446, 14), libs_45198, *[section_45199], **kwargs_45200)
    
    # Processing the call keyword arguments (line 446)
    kwargs_45202 = {}
    # Getting the type of 'print' (line 446)
    print_45196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'print', False)
    # Calling print(args, kwargs) (line 446)
    print_call_result_45203 = invoke(stypy.reporting.localization.Localization(__file__, 446, 8), print_45196, *[libs_call_result_45201], **kwargs_45202)
    
    # SSA join for if statement (line 445)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'options' (line 447)
    options_45204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 7), 'options')
    # Obtaining the member 'version' of a type (line 447)
    version_45205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 7), options_45204, 'version')
    # Testing the type of an if condition (line 447)
    if_condition_45206 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 447, 4), version_45205)
    # Assigning a type to the variable 'if_condition_45206' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'if_condition_45206', if_condition_45206)
    # SSA begins for if statement (line 447)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 448)
    # Processing the call arguments (line 448)
    # Getting the type of 'info' (line 448)
    info_45208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 14), 'info', False)
    # Obtaining the member 'version' of a type (line 448)
    version_45209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 14), info_45208, 'version')
    # Processing the call keyword arguments (line 448)
    kwargs_45210 = {}
    # Getting the type of 'print' (line 448)
    print_45207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 8), 'print', False)
    # Calling print(args, kwargs) (line 448)
    print_call_result_45211 = invoke(stypy.reporting.localization.Localization(__file__, 448, 8), print_45207, *[version_45209], **kwargs_45210)
    
    # SSA join for if statement (line 447)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'options' (line 449)
    options_45212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 7), 'options')
    # Obtaining the member 'min_version' of a type (line 449)
    min_version_45213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 7), options_45212, 'min_version')
    # Testing the type of an if condition (line 449)
    if_condition_45214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 449, 4), min_version_45213)
    # Assigning a type to the variable 'if_condition_45214' (line 449)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'if_condition_45214', if_condition_45214)
    # SSA begins for if statement (line 449)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 450)
    # Processing the call arguments (line 450)
    
    # Getting the type of 'info' (line 450)
    info_45216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 14), 'info', False)
    # Obtaining the member 'version' of a type (line 450)
    version_45217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 14), info_45216, 'version')
    # Getting the type of 'options' (line 450)
    options_45218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 30), 'options', False)
    # Obtaining the member 'min_version' of a type (line 450)
    min_version_45219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 30), options_45218, 'min_version')
    # Applying the binary operator '>=' (line 450)
    result_ge_45220 = python_operator(stypy.reporting.localization.Localization(__file__, 450, 14), '>=', version_45217, min_version_45219)
    
    # Processing the call keyword arguments (line 450)
    kwargs_45221 = {}
    # Getting the type of 'print' (line 450)
    print_45215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'print', False)
    # Calling print(args, kwargs) (line 450)
    print_call_result_45222 = invoke(stypy.reporting.localization.Localization(__file__, 450, 8), print_45215, *[result_ge_45220], **kwargs_45221)
    
    # SSA join for if statement (line 449)
    module_type_store = module_type_store.join_ssa_context()
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
