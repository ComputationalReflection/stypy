
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.bdist_wininst
2: 
3: Implements the Distutils 'bdist_wininst' command: create a windows installer
4: exe-program.'''
5: 
6: __revision__ = "$Id$"
7: 
8: import sys
9: import os
10: import string
11: 
12: from sysconfig import get_python_version
13: 
14: from distutils.core import Command
15: from distutils.dir_util import remove_tree
16: from distutils.errors import DistutilsOptionError, DistutilsPlatformError
17: from distutils import log
18: from distutils.util import get_platform
19: 
20: class bdist_wininst (Command):
21: 
22:     description = "create an executable installer for MS Windows"
23: 
24:     user_options = [('bdist-dir=', None,
25:                      "temporary directory for creating the distribution"),
26:                     ('plat-name=', 'p',
27:                      "platform name to embed in generated filenames "
28:                      "(default: %s)" % get_platform()),
29:                     ('keep-temp', 'k',
30:                      "keep the pseudo-installation tree around after " +
31:                      "creating the distribution archive"),
32:                     ('target-version=', None,
33:                      "require a specific python version" +
34:                      " on the target system"),
35:                     ('no-target-compile', 'c',
36:                      "do not compile .py to .pyc on the target system"),
37:                     ('no-target-optimize', 'o',
38:                      "do not compile .py to .pyo (optimized)"
39:                      "on the target system"),
40:                     ('dist-dir=', 'd',
41:                      "directory to put final built distributions in"),
42:                     ('bitmap=', 'b',
43:                      "bitmap to use for the installer instead of python-powered logo"),
44:                     ('title=', 't',
45:                      "title to display on the installer background instead of default"),
46:                     ('skip-build', None,
47:                      "skip rebuilding everything (for testing/debugging)"),
48:                     ('install-script=', None,
49:                      "basename of installation script to be run after"
50:                      "installation or before deinstallation"),
51:                     ('pre-install-script=', None,
52:                      "Fully qualified filename of a script to be run before "
53:                      "any files are installed.  This script need not be in the "
54:                      "distribution"),
55:                     ('user-access-control=', None,
56:                      "specify Vista's UAC handling - 'none'/default=no "
57:                      "handling, 'auto'=use UAC if target Python installed for "
58:                      "all users, 'force'=always use UAC"),
59:                    ]
60: 
61:     boolean_options = ['keep-temp', 'no-target-compile', 'no-target-optimize',
62:                        'skip-build']
63: 
64:     def initialize_options (self):
65:         self.bdist_dir = None
66:         self.plat_name = None
67:         self.keep_temp = 0
68:         self.no_target_compile = 0
69:         self.no_target_optimize = 0
70:         self.target_version = None
71:         self.dist_dir = None
72:         self.bitmap = None
73:         self.title = None
74:         self.skip_build = None
75:         self.install_script = None
76:         self.pre_install_script = None
77:         self.user_access_control = None
78: 
79:     # initialize_options()
80: 
81: 
82:     def finalize_options (self):
83:         self.set_undefined_options('bdist', ('skip_build', 'skip_build'))
84: 
85:         if self.bdist_dir is None:
86:             if self.skip_build and self.plat_name:
87:                 # If build is skipped and plat_name is overridden, bdist will
88:                 # not see the correct 'plat_name' - so set that up manually.
89:                 bdist = self.distribution.get_command_obj('bdist')
90:                 bdist.plat_name = self.plat_name
91:                 # next the command will be initialized using that name
92:             bdist_base = self.get_finalized_command('bdist').bdist_base
93:             self.bdist_dir = os.path.join(bdist_base, 'wininst')
94: 
95:         if not self.target_version:
96:             self.target_version = ""
97: 
98:         if not self.skip_build and self.distribution.has_ext_modules():
99:             short_version = get_python_version()
100:             if self.target_version and self.target_version != short_version:
101:                 raise DistutilsOptionError, \
102:                       "target version can only be %s, or the '--skip-build'" \
103:                       " option must be specified" % (short_version,)
104:             self.target_version = short_version
105: 
106:         self.set_undefined_options('bdist',
107:                                    ('dist_dir', 'dist_dir'),
108:                                    ('plat_name', 'plat_name'),
109:                                   )
110: 
111:         if self.install_script:
112:             for script in self.distribution.scripts:
113:                 if self.install_script == os.path.basename(script):
114:                     break
115:             else:
116:                 raise DistutilsOptionError, \
117:                       "install_script '%s' not found in scripts" % \
118:                       self.install_script
119:     # finalize_options()
120: 
121: 
122:     def run (self):
123:         if (sys.platform != "win32" and
124:             (self.distribution.has_ext_modules() or
125:              self.distribution.has_c_libraries())):
126:             raise DistutilsPlatformError \
127:                   ("distribution contains extensions and/or C libraries; "
128:                    "must be compiled on a Windows 32 platform")
129: 
130:         if not self.skip_build:
131:             self.run_command('build')
132: 
133:         install = self.reinitialize_command('install', reinit_subcommands=1)
134:         install.root = self.bdist_dir
135:         install.skip_build = self.skip_build
136:         install.warn_dir = 0
137:         install.plat_name = self.plat_name
138: 
139:         install_lib = self.reinitialize_command('install_lib')
140:         # we do not want to include pyc or pyo files
141:         install_lib.compile = 0
142:         install_lib.optimize = 0
143: 
144:         if self.distribution.has_ext_modules():
145:             # If we are building an installer for a Python version other
146:             # than the one we are currently running, then we need to ensure
147:             # our build_lib reflects the other Python version rather than ours.
148:             # Note that for target_version!=sys.version, we must have skipped the
149:             # build step, so there is no issue with enforcing the build of this
150:             # version.
151:             target_version = self.target_version
152:             if not target_version:
153:                 assert self.skip_build, "Should have already checked this"
154:                 target_version = sys.version[0:3]
155:             plat_specifier = ".%s-%s" % (self.plat_name, target_version)
156:             build = self.get_finalized_command('build')
157:             build.build_lib = os.path.join(build.build_base,
158:                                            'lib' + plat_specifier)
159: 
160:         # Use a custom scheme for the zip-file, because we have to decide
161:         # at installation time which scheme to use.
162:         for key in ('purelib', 'platlib', 'headers', 'scripts', 'data'):
163:             value = string.upper(key)
164:             if key == 'headers':
165:                 value = value + '/Include/$dist_name'
166:             setattr(install,
167:                     'install_' + key,
168:                     value)
169: 
170:         log.info("installing to %s", self.bdist_dir)
171:         install.ensure_finalized()
172: 
173:         # avoid warning of 'install_lib' about installing
174:         # into a directory not in sys.path
175:         sys.path.insert(0, os.path.join(self.bdist_dir, 'PURELIB'))
176: 
177:         install.run()
178: 
179:         del sys.path[0]
180: 
181:         # And make an archive relative to the root of the
182:         # pseudo-installation tree.
183:         from tempfile import mktemp
184:         archive_basename = mktemp()
185:         fullname = self.distribution.get_fullname()
186:         arcname = self.make_archive(archive_basename, "zip",
187:                                     root_dir=self.bdist_dir)
188:         # create an exe containing the zip-file
189:         self.create_exe(arcname, fullname, self.bitmap)
190:         if self.distribution.has_ext_modules():
191:             pyversion = get_python_version()
192:         else:
193:             pyversion = 'any'
194:         self.distribution.dist_files.append(('bdist_wininst', pyversion,
195:                                              self.get_installer_filename(fullname)))
196:         # remove the zip-file again
197:         log.debug("removing temporary file '%s'", arcname)
198:         os.remove(arcname)
199: 
200:         if not self.keep_temp:
201:             remove_tree(self.bdist_dir, dry_run=self.dry_run)
202: 
203:     # run()
204: 
205:     def get_inidata (self):
206:         # Return data describing the installation.
207: 
208:         lines = []
209:         metadata = self.distribution.metadata
210: 
211:         # Write the [metadata] section.
212:         lines.append("[metadata]")
213: 
214:         # 'info' will be displayed in the installer's dialog box,
215:         # describing the items to be installed.
216:         info = (metadata.long_description or '') + '\n'
217: 
218:         # Escape newline characters
219:         def escape(s):
220:             return string.replace(s, "\n", "\\n")
221: 
222:         for name in ["author", "author_email", "description", "maintainer",
223:                      "maintainer_email", "name", "url", "version"]:
224:             data = getattr(metadata, name, "")
225:             if data:
226:                 info = info + ("\n    %s: %s" % \
227:                                (string.capitalize(name), escape(data)))
228:                 lines.append("%s=%s" % (name, escape(data)))
229: 
230:         # The [setup] section contains entries controlling
231:         # the installer runtime.
232:         lines.append("\n[Setup]")
233:         if self.install_script:
234:             lines.append("install_script=%s" % self.install_script)
235:         lines.append("info=%s" % escape(info))
236:         lines.append("target_compile=%d" % (not self.no_target_compile))
237:         lines.append("target_optimize=%d" % (not self.no_target_optimize))
238:         if self.target_version:
239:             lines.append("target_version=%s" % self.target_version)
240:         if self.user_access_control:
241:             lines.append("user_access_control=%s" % self.user_access_control)
242: 
243:         title = self.title or self.distribution.get_fullname()
244:         lines.append("title=%s" % escape(title))
245:         import time
246:         import distutils
247:         build_info = "Built %s with distutils-%s" % \
248:                      (time.ctime(time.time()), distutils.__version__)
249:         lines.append("build_info=%s" % build_info)
250:         return string.join(lines, "\n")
251: 
252:     # get_inidata()
253: 
254:     def create_exe (self, arcname, fullname, bitmap=None):
255:         import struct
256: 
257:         self.mkpath(self.dist_dir)
258: 
259:         cfgdata = self.get_inidata()
260: 
261:         installer_name = self.get_installer_filename(fullname)
262:         self.announce("creating %s" % installer_name)
263: 
264:         if bitmap:
265:             bitmapdata = open(bitmap, "rb").read()
266:             bitmaplen = len(bitmapdata)
267:         else:
268:             bitmaplen = 0
269: 
270:         file = open(installer_name, "wb")
271:         file.write(self.get_exe_bytes())
272:         if bitmap:
273:             file.write(bitmapdata)
274: 
275:         # Convert cfgdata from unicode to ascii, mbcs encoded
276:         try:
277:             unicode
278:         except NameError:
279:             pass
280:         else:
281:             if isinstance(cfgdata, unicode):
282:                 cfgdata = cfgdata.encode("mbcs")
283: 
284:         # Append the pre-install script
285:         cfgdata = cfgdata + "\0"
286:         if self.pre_install_script:
287:             script_data = open(self.pre_install_script, "r").read()
288:             cfgdata = cfgdata + script_data + "\n\0"
289:         else:
290:             # empty pre-install script
291:             cfgdata = cfgdata + "\0"
292:         file.write(cfgdata)
293: 
294:         # The 'magic number' 0x1234567B is used to make sure that the
295:         # binary layout of 'cfgdata' is what the wininst.exe binary
296:         # expects.  If the layout changes, increment that number, make
297:         # the corresponding changes to the wininst.exe sources, and
298:         # recompile them.
299:         header = struct.pack("<iii",
300:                              0x1234567B,       # tag
301:                              len(cfgdata),     # length
302:                              bitmaplen,        # number of bytes in bitmap
303:                              )
304:         file.write(header)
305:         file.write(open(arcname, "rb").read())
306: 
307:     # create_exe()
308: 
309:     def get_installer_filename(self, fullname):
310:         # Factored out to allow overriding in subclasses
311:         if self.target_version:
312:             # if we create an installer for a specific python version,
313:             # it's better to include this in the name
314:             installer_name = os.path.join(self.dist_dir,
315:                                           "%s.%s-py%s.exe" %
316:                                            (fullname, self.plat_name, self.target_version))
317:         else:
318:             installer_name = os.path.join(self.dist_dir,
319:                                           "%s.%s.exe" % (fullname, self.plat_name))
320:         return installer_name
321:     # get_installer_filename()
322: 
323:     def get_exe_bytes (self):
324:         from distutils.msvccompiler import get_build_version
325:         # If a target-version other than the current version has been
326:         # specified, then using the MSVC version from *this* build is no good.
327:         # Without actually finding and executing the target version and parsing
328:         # its sys.version, we just hard-code our knowledge of old versions.
329:         # NOTE: Possible alternative is to allow "--target-version" to
330:         # specify a Python executable rather than a simple version string.
331:         # We can then execute this program to obtain any info we need, such
332:         # as the real sys.version string for the build.
333:         cur_version = get_python_version()
334:         if self.target_version and self.target_version != cur_version:
335:             # If the target version is *later* than us, then we assume they
336:             # use what we use
337:             # string compares seem wrong, but are what sysconfig.py itself uses
338:             if self.target_version > cur_version:
339:                 bv = get_build_version()
340:             else:
341:                 if self.target_version < "2.4":
342:                     bv = 6.0
343:                 else:
344:                     bv = 7.1
345:         else:
346:             # for current version - use authoritative check.
347:             bv = get_build_version()
348: 
349:         # wininst-x.y.exe is in the same directory as this file
350:         directory = os.path.dirname(__file__)
351:         # we must use a wininst-x.y.exe built with the same C compiler
352:         # used for python.  XXX What about mingw, borland, and so on?
353: 
354:         # if plat_name starts with "win" but is not "win32"
355:         # we want to strip "win" and leave the rest (e.g. -amd64)
356:         # for all other cases, we don't want any suffix
357:         if self.plat_name != 'win32' and self.plat_name[:3] == 'win':
358:             sfix = self.plat_name[3:]
359:         else:
360:             sfix = ''
361: 
362:         filename = os.path.join(directory, "wininst-%.1f%s.exe" % (bv, sfix))
363:         f = open(filename, "rb")
364:         try:
365:             return f.read()
366:         finally:
367:             f.close()
368: # class bdist_wininst
369: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_16319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', "distutils.command.bdist_wininst\n\nImplements the Distutils 'bdist_wininst' command: create a windows installer\nexe-program.")

# Assigning a Str to a Name (line 6):
str_16320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__revision__', str_16320)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import sys' statement (line 8)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import os' statement (line 9)
import os

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import string' statement (line 10)
import string

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'string', string, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from sysconfig import get_python_version' statement (line 12)
try:
    from sysconfig import get_python_version

except:
    get_python_version = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'sysconfig', None, module_type_store, ['get_python_version'], [get_python_version])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.core import Command' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_16321 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.core')

if (type(import_16321) is not StypyTypeError):

    if (import_16321 != 'pyd_module'):
        __import__(import_16321)
        sys_modules_16322 = sys.modules[import_16321]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.core', sys_modules_16322.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_16322, sys_modules_16322.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.core', import_16321)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.dir_util import remove_tree' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_16323 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.dir_util')

if (type(import_16323) is not StypyTypeError):

    if (import_16323 != 'pyd_module'):
        __import__(import_16323)
        sys_modules_16324 = sys.modules[import_16323]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.dir_util', sys_modules_16324.module_type_store, module_type_store, ['remove_tree'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_16324, sys_modules_16324.module_type_store, module_type_store)
    else:
        from distutils.dir_util import remove_tree

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.dir_util', None, module_type_store, ['remove_tree'], [remove_tree])

else:
    # Assigning a type to the variable 'distutils.dir_util' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.dir_util', import_16323)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils.errors import DistutilsOptionError, DistutilsPlatformError' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_16325 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors')

if (type(import_16325) is not StypyTypeError):

    if (import_16325 != 'pyd_module'):
        __import__(import_16325)
        sys_modules_16326 = sys.modules[import_16325]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', sys_modules_16326.module_type_store, module_type_store, ['DistutilsOptionError', 'DistutilsPlatformError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_16326, sys_modules_16326.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsOptionError, DistutilsPlatformError

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', None, module_type_store, ['DistutilsOptionError', 'DistutilsPlatformError'], [DistutilsOptionError, DistutilsPlatformError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', import_16325)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from distutils import log' statement (line 17)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils', None, module_type_store, ['log'], [log])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'from distutils.util import get_platform' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_16327 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.util')

if (type(import_16327) is not StypyTypeError):

    if (import_16327 != 'pyd_module'):
        __import__(import_16327)
        sys_modules_16328 = sys.modules[import_16327]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.util', sys_modules_16328.module_type_store, module_type_store, ['get_platform'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 0), __file__, sys_modules_16328, sys_modules_16328.module_type_store, module_type_store)
    else:
        from distutils.util import get_platform

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.util', None, module_type_store, ['get_platform'], [get_platform])

else:
    # Assigning a type to the variable 'distutils.util' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'distutils.util', import_16327)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

# Declaration of the 'bdist_wininst' class
# Getting the type of 'Command' (line 20)
Command_16329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'Command')

class bdist_wininst(Command_16329, ):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_wininst.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        bdist_wininst.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_wininst.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_wininst.initialize_options.__dict__.__setitem__('stypy_function_name', 'bdist_wininst.initialize_options')
        bdist_wininst.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_wininst.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_wininst.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_wininst.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_wininst.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_wininst.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_wininst.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_wininst.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 65):
        # Getting the type of 'None' (line 65)
        None_16330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'None')
        # Getting the type of 'self' (line 65)
        self_16331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self')
        # Setting the type of the member 'bdist_dir' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_16331, 'bdist_dir', None_16330)
        
        # Assigning a Name to a Attribute (line 66):
        # Getting the type of 'None' (line 66)
        None_16332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'None')
        # Getting the type of 'self' (line 66)
        self_16333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self')
        # Setting the type of the member 'plat_name' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_16333, 'plat_name', None_16332)
        
        # Assigning a Num to a Attribute (line 67):
        int_16334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 25), 'int')
        # Getting the type of 'self' (line 67)
        self_16335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self')
        # Setting the type of the member 'keep_temp' of a type (line 67)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_16335, 'keep_temp', int_16334)
        
        # Assigning a Num to a Attribute (line 68):
        int_16336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 33), 'int')
        # Getting the type of 'self' (line 68)
        self_16337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self')
        # Setting the type of the member 'no_target_compile' of a type (line 68)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_16337, 'no_target_compile', int_16336)
        
        # Assigning a Num to a Attribute (line 69):
        int_16338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 34), 'int')
        # Getting the type of 'self' (line 69)
        self_16339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self')
        # Setting the type of the member 'no_target_optimize' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_16339, 'no_target_optimize', int_16338)
        
        # Assigning a Name to a Attribute (line 70):
        # Getting the type of 'None' (line 70)
        None_16340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 30), 'None')
        # Getting the type of 'self' (line 70)
        self_16341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self')
        # Setting the type of the member 'target_version' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_16341, 'target_version', None_16340)
        
        # Assigning a Name to a Attribute (line 71):
        # Getting the type of 'None' (line 71)
        None_16342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'None')
        # Getting the type of 'self' (line 71)
        self_16343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self')
        # Setting the type of the member 'dist_dir' of a type (line 71)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_16343, 'dist_dir', None_16342)
        
        # Assigning a Name to a Attribute (line 72):
        # Getting the type of 'None' (line 72)
        None_16344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'None')
        # Getting the type of 'self' (line 72)
        self_16345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self')
        # Setting the type of the member 'bitmap' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_16345, 'bitmap', None_16344)
        
        # Assigning a Name to a Attribute (line 73):
        # Getting the type of 'None' (line 73)
        None_16346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 21), 'None')
        # Getting the type of 'self' (line 73)
        self_16347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self')
        # Setting the type of the member 'title' of a type (line 73)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_16347, 'title', None_16346)
        
        # Assigning a Name to a Attribute (line 74):
        # Getting the type of 'None' (line 74)
        None_16348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'None')
        # Getting the type of 'self' (line 74)
        self_16349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member 'skip_build' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_16349, 'skip_build', None_16348)
        
        # Assigning a Name to a Attribute (line 75):
        # Getting the type of 'None' (line 75)
        None_16350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 30), 'None')
        # Getting the type of 'self' (line 75)
        self_16351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self')
        # Setting the type of the member 'install_script' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_16351, 'install_script', None_16350)
        
        # Assigning a Name to a Attribute (line 76):
        # Getting the type of 'None' (line 76)
        None_16352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 34), 'None')
        # Getting the type of 'self' (line 76)
        self_16353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self')
        # Setting the type of the member 'pre_install_script' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_16353, 'pre_install_script', None_16352)
        
        # Assigning a Name to a Attribute (line 77):
        # Getting the type of 'None' (line 77)
        None_16354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 35), 'None')
        # Getting the type of 'self' (line 77)
        self_16355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self')
        # Setting the type of the member 'user_access_control' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_16355, 'user_access_control', None_16354)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_16356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16356)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_16356


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_wininst.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        bdist_wininst.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_wininst.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_wininst.finalize_options.__dict__.__setitem__('stypy_function_name', 'bdist_wininst.finalize_options')
        bdist_wininst.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_wininst.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_wininst.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_wininst.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_wininst.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_wininst.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_wininst.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_wininst.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_undefined_options(...): (line 83)
        # Processing the call arguments (line 83)
        str_16359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 35), 'str', 'bdist')
        
        # Obtaining an instance of the builtin type 'tuple' (line 83)
        tuple_16360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 83)
        # Adding element type (line 83)
        str_16361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 45), 'str', 'skip_build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 45), tuple_16360, str_16361)
        # Adding element type (line 83)
        str_16362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 59), 'str', 'skip_build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 45), tuple_16360, str_16362)
        
        # Processing the call keyword arguments (line 83)
        kwargs_16363 = {}
        # Getting the type of 'self' (line 83)
        self_16357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 83)
        set_undefined_options_16358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_16357, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 83)
        set_undefined_options_call_result_16364 = invoke(stypy.reporting.localization.Localization(__file__, 83, 8), set_undefined_options_16358, *[str_16359, tuple_16360], **kwargs_16363)
        
        
        # Type idiom detected: calculating its left and rigth part (line 85)
        # Getting the type of 'self' (line 85)
        self_16365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'self')
        # Obtaining the member 'bdist_dir' of a type (line 85)
        bdist_dir_16366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 11), self_16365, 'bdist_dir')
        # Getting the type of 'None' (line 85)
        None_16367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'None')
        
        (may_be_16368, more_types_in_union_16369) = may_be_none(bdist_dir_16366, None_16367)

        if may_be_16368:

            if more_types_in_union_16369:
                # Runtime conditional SSA (line 85)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Evaluating a boolean operation
            # Getting the type of 'self' (line 86)
            self_16370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'self')
            # Obtaining the member 'skip_build' of a type (line 86)
            skip_build_16371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), self_16370, 'skip_build')
            # Getting the type of 'self' (line 86)
            self_16372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 35), 'self')
            # Obtaining the member 'plat_name' of a type (line 86)
            plat_name_16373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 35), self_16372, 'plat_name')
            # Applying the binary operator 'and' (line 86)
            result_and_keyword_16374 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 15), 'and', skip_build_16371, plat_name_16373)
            
            # Testing the type of an if condition (line 86)
            if_condition_16375 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 12), result_and_keyword_16374)
            # Assigning a type to the variable 'if_condition_16375' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'if_condition_16375', if_condition_16375)
            # SSA begins for if statement (line 86)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 89):
            
            # Call to get_command_obj(...): (line 89)
            # Processing the call arguments (line 89)
            str_16379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 58), 'str', 'bdist')
            # Processing the call keyword arguments (line 89)
            kwargs_16380 = {}
            # Getting the type of 'self' (line 89)
            self_16376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'self', False)
            # Obtaining the member 'distribution' of a type (line 89)
            distribution_16377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 24), self_16376, 'distribution')
            # Obtaining the member 'get_command_obj' of a type (line 89)
            get_command_obj_16378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 24), distribution_16377, 'get_command_obj')
            # Calling get_command_obj(args, kwargs) (line 89)
            get_command_obj_call_result_16381 = invoke(stypy.reporting.localization.Localization(__file__, 89, 24), get_command_obj_16378, *[str_16379], **kwargs_16380)
            
            # Assigning a type to the variable 'bdist' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'bdist', get_command_obj_call_result_16381)
            
            # Assigning a Attribute to a Attribute (line 90):
            # Getting the type of 'self' (line 90)
            self_16382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 34), 'self')
            # Obtaining the member 'plat_name' of a type (line 90)
            plat_name_16383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 34), self_16382, 'plat_name')
            # Getting the type of 'bdist' (line 90)
            bdist_16384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'bdist')
            # Setting the type of the member 'plat_name' of a type (line 90)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 16), bdist_16384, 'plat_name', plat_name_16383)
            # SSA join for if statement (line 86)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Attribute to a Name (line 92):
            
            # Call to get_finalized_command(...): (line 92)
            # Processing the call arguments (line 92)
            str_16387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 52), 'str', 'bdist')
            # Processing the call keyword arguments (line 92)
            kwargs_16388 = {}
            # Getting the type of 'self' (line 92)
            self_16385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'self', False)
            # Obtaining the member 'get_finalized_command' of a type (line 92)
            get_finalized_command_16386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 25), self_16385, 'get_finalized_command')
            # Calling get_finalized_command(args, kwargs) (line 92)
            get_finalized_command_call_result_16389 = invoke(stypy.reporting.localization.Localization(__file__, 92, 25), get_finalized_command_16386, *[str_16387], **kwargs_16388)
            
            # Obtaining the member 'bdist_base' of a type (line 92)
            bdist_base_16390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 25), get_finalized_command_call_result_16389, 'bdist_base')
            # Assigning a type to the variable 'bdist_base' (line 92)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'bdist_base', bdist_base_16390)
            
            # Assigning a Call to a Attribute (line 93):
            
            # Call to join(...): (line 93)
            # Processing the call arguments (line 93)
            # Getting the type of 'bdist_base' (line 93)
            bdist_base_16394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 42), 'bdist_base', False)
            str_16395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 54), 'str', 'wininst')
            # Processing the call keyword arguments (line 93)
            kwargs_16396 = {}
            # Getting the type of 'os' (line 93)
            os_16391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 29), 'os', False)
            # Obtaining the member 'path' of a type (line 93)
            path_16392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 29), os_16391, 'path')
            # Obtaining the member 'join' of a type (line 93)
            join_16393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 29), path_16392, 'join')
            # Calling join(args, kwargs) (line 93)
            join_call_result_16397 = invoke(stypy.reporting.localization.Localization(__file__, 93, 29), join_16393, *[bdist_base_16394, str_16395], **kwargs_16396)
            
            # Getting the type of 'self' (line 93)
            self_16398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'self')
            # Setting the type of the member 'bdist_dir' of a type (line 93)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), self_16398, 'bdist_dir', join_call_result_16397)

            if more_types_in_union_16369:
                # SSA join for if statement (line 85)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'self' (line 95)
        self_16399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'self')
        # Obtaining the member 'target_version' of a type (line 95)
        target_version_16400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 15), self_16399, 'target_version')
        # Applying the 'not' unary operator (line 95)
        result_not__16401 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 11), 'not', target_version_16400)
        
        # Testing the type of an if condition (line 95)
        if_condition_16402 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 8), result_not__16401)
        # Assigning a type to the variable 'if_condition_16402' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'if_condition_16402', if_condition_16402)
        # SSA begins for if statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Attribute (line 96):
        str_16403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 34), 'str', '')
        # Getting the type of 'self' (line 96)
        self_16404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'self')
        # Setting the type of the member 'target_version' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), self_16404, 'target_version', str_16403)
        # SSA join for if statement (line 95)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 98)
        self_16405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'self')
        # Obtaining the member 'skip_build' of a type (line 98)
        skip_build_16406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 15), self_16405, 'skip_build')
        # Applying the 'not' unary operator (line 98)
        result_not__16407 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 11), 'not', skip_build_16406)
        
        
        # Call to has_ext_modules(...): (line 98)
        # Processing the call keyword arguments (line 98)
        kwargs_16411 = {}
        # Getting the type of 'self' (line 98)
        self_16408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 35), 'self', False)
        # Obtaining the member 'distribution' of a type (line 98)
        distribution_16409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 35), self_16408, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 98)
        has_ext_modules_16410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 35), distribution_16409, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 98)
        has_ext_modules_call_result_16412 = invoke(stypy.reporting.localization.Localization(__file__, 98, 35), has_ext_modules_16410, *[], **kwargs_16411)
        
        # Applying the binary operator 'and' (line 98)
        result_and_keyword_16413 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 11), 'and', result_not__16407, has_ext_modules_call_result_16412)
        
        # Testing the type of an if condition (line 98)
        if_condition_16414 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 8), result_and_keyword_16413)
        # Assigning a type to the variable 'if_condition_16414' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'if_condition_16414', if_condition_16414)
        # SSA begins for if statement (line 98)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 99):
        
        # Call to get_python_version(...): (line 99)
        # Processing the call keyword arguments (line 99)
        kwargs_16416 = {}
        # Getting the type of 'get_python_version' (line 99)
        get_python_version_16415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 28), 'get_python_version', False)
        # Calling get_python_version(args, kwargs) (line 99)
        get_python_version_call_result_16417 = invoke(stypy.reporting.localization.Localization(__file__, 99, 28), get_python_version_16415, *[], **kwargs_16416)
        
        # Assigning a type to the variable 'short_version' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'short_version', get_python_version_call_result_16417)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 100)
        self_16418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'self')
        # Obtaining the member 'target_version' of a type (line 100)
        target_version_16419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), self_16418, 'target_version')
        
        # Getting the type of 'self' (line 100)
        self_16420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 39), 'self')
        # Obtaining the member 'target_version' of a type (line 100)
        target_version_16421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 39), self_16420, 'target_version')
        # Getting the type of 'short_version' (line 100)
        short_version_16422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 62), 'short_version')
        # Applying the binary operator '!=' (line 100)
        result_ne_16423 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 39), '!=', target_version_16421, short_version_16422)
        
        # Applying the binary operator 'and' (line 100)
        result_and_keyword_16424 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), 'and', target_version_16419, result_ne_16423)
        
        # Testing the type of an if condition (line 100)
        if_condition_16425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 12), result_and_keyword_16424)
        # Assigning a type to the variable 'if_condition_16425' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'if_condition_16425', if_condition_16425)
        # SSA begins for if statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsOptionError' (line 101)
        DistutilsOptionError_16426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'DistutilsOptionError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 101, 16), DistutilsOptionError_16426, 'raise parameter', BaseException)
        # SSA join for if statement (line 100)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 104):
        # Getting the type of 'short_version' (line 104)
        short_version_16427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'short_version')
        # Getting the type of 'self' (line 104)
        self_16428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'self')
        # Setting the type of the member 'target_version' of a type (line 104)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 12), self_16428, 'target_version', short_version_16427)
        # SSA join for if statement (line 98)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to set_undefined_options(...): (line 106)
        # Processing the call arguments (line 106)
        str_16431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 35), 'str', 'bdist')
        
        # Obtaining an instance of the builtin type 'tuple' (line 107)
        tuple_16432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 107)
        # Adding element type (line 107)
        str_16433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 36), 'str', 'dist_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 36), tuple_16432, str_16433)
        # Adding element type (line 107)
        str_16434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 48), 'str', 'dist_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 36), tuple_16432, str_16434)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 108)
        tuple_16435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 108)
        # Adding element type (line 108)
        str_16436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 36), 'str', 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 36), tuple_16435, str_16436)
        # Adding element type (line 108)
        str_16437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 49), 'str', 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 36), tuple_16435, str_16437)
        
        # Processing the call keyword arguments (line 106)
        kwargs_16438 = {}
        # Getting the type of 'self' (line 106)
        self_16429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 106)
        set_undefined_options_16430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), self_16429, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 106)
        set_undefined_options_call_result_16439 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), set_undefined_options_16430, *[str_16431, tuple_16432, tuple_16435], **kwargs_16438)
        
        
        # Getting the type of 'self' (line 111)
        self_16440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'self')
        # Obtaining the member 'install_script' of a type (line 111)
        install_script_16441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 11), self_16440, 'install_script')
        # Testing the type of an if condition (line 111)
        if_condition_16442 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 8), install_script_16441)
        # Assigning a type to the variable 'if_condition_16442' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'if_condition_16442', if_condition_16442)
        # SSA begins for if statement (line 111)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 112)
        self_16443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 26), 'self')
        # Obtaining the member 'distribution' of a type (line 112)
        distribution_16444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 26), self_16443, 'distribution')
        # Obtaining the member 'scripts' of a type (line 112)
        scripts_16445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 26), distribution_16444, 'scripts')
        # Testing the type of a for loop iterable (line 112)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 112, 12), scripts_16445)
        # Getting the type of the for loop variable (line 112)
        for_loop_var_16446 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 112, 12), scripts_16445)
        # Assigning a type to the variable 'script' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'script', for_loop_var_16446)
        # SSA begins for a for statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'self' (line 113)
        self_16447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'self')
        # Obtaining the member 'install_script' of a type (line 113)
        install_script_16448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 19), self_16447, 'install_script')
        
        # Call to basename(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'script' (line 113)
        script_16452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 59), 'script', False)
        # Processing the call keyword arguments (line 113)
        kwargs_16453 = {}
        # Getting the type of 'os' (line 113)
        os_16449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 42), 'os', False)
        # Obtaining the member 'path' of a type (line 113)
        path_16450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 42), os_16449, 'path')
        # Obtaining the member 'basename' of a type (line 113)
        basename_16451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 42), path_16450, 'basename')
        # Calling basename(args, kwargs) (line 113)
        basename_call_result_16454 = invoke(stypy.reporting.localization.Localization(__file__, 113, 42), basename_16451, *[script_16452], **kwargs_16453)
        
        # Applying the binary operator '==' (line 113)
        result_eq_16455 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 19), '==', install_script_16448, basename_call_result_16454)
        
        # Testing the type of an if condition (line 113)
        if_condition_16456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 16), result_eq_16455)
        # Assigning a type to the variable 'if_condition_16456' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'if_condition_16456', if_condition_16456)
        # SSA begins for if statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 113)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of a for statement (line 112)
        module_type_store.open_ssa_branch('for loop else')
        # Getting the type of 'DistutilsOptionError' (line 116)
        DistutilsOptionError_16457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 22), 'DistutilsOptionError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 116, 16), DistutilsOptionError_16457, 'raise parameter', BaseException)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 111)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_16458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16458)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_16458


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_wininst.run.__dict__.__setitem__('stypy_localization', localization)
        bdist_wininst.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_wininst.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_wininst.run.__dict__.__setitem__('stypy_function_name', 'bdist_wininst.run')
        bdist_wininst.run.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_wininst.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_wininst.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_wininst.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_wininst.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_wininst.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_wininst.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_wininst.run', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sys' (line 123)
        sys_16459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'sys')
        # Obtaining the member 'platform' of a type (line 123)
        platform_16460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), sys_16459, 'platform')
        str_16461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 28), 'str', 'win32')
        # Applying the binary operator '!=' (line 123)
        result_ne_16462 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 12), '!=', platform_16460, str_16461)
        
        
        # Evaluating a boolean operation
        
        # Call to has_ext_modules(...): (line 124)
        # Processing the call keyword arguments (line 124)
        kwargs_16466 = {}
        # Getting the type of 'self' (line 124)
        self_16463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 13), 'self', False)
        # Obtaining the member 'distribution' of a type (line 124)
        distribution_16464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 13), self_16463, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 124)
        has_ext_modules_16465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 13), distribution_16464, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 124)
        has_ext_modules_call_result_16467 = invoke(stypy.reporting.localization.Localization(__file__, 124, 13), has_ext_modules_16465, *[], **kwargs_16466)
        
        
        # Call to has_c_libraries(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_16471 = {}
        # Getting the type of 'self' (line 125)
        self_16468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'self', False)
        # Obtaining the member 'distribution' of a type (line 125)
        distribution_16469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), self_16468, 'distribution')
        # Obtaining the member 'has_c_libraries' of a type (line 125)
        has_c_libraries_16470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), distribution_16469, 'has_c_libraries')
        # Calling has_c_libraries(args, kwargs) (line 125)
        has_c_libraries_call_result_16472 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), has_c_libraries_16470, *[], **kwargs_16471)
        
        # Applying the binary operator 'or' (line 124)
        result_or_keyword_16473 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 13), 'or', has_ext_modules_call_result_16467, has_c_libraries_call_result_16472)
        
        # Applying the binary operator 'and' (line 123)
        result_and_keyword_16474 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 12), 'and', result_ne_16462, result_or_keyword_16473)
        
        # Testing the type of an if condition (line 123)
        if_condition_16475 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 8), result_and_keyword_16474)
        # Assigning a type to the variable 'if_condition_16475' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'if_condition_16475', if_condition_16475)
        # SSA begins for if statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsPlatformError(...): (line 126)
        # Processing the call arguments (line 126)
        str_16477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 19), 'str', 'distribution contains extensions and/or C libraries; must be compiled on a Windows 32 platform')
        # Processing the call keyword arguments (line 126)
        kwargs_16478 = {}
        # Getting the type of 'DistutilsPlatformError' (line 126)
        DistutilsPlatformError_16476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 18), 'DistutilsPlatformError', False)
        # Calling DistutilsPlatformError(args, kwargs) (line 126)
        DistutilsPlatformError_call_result_16479 = invoke(stypy.reporting.localization.Localization(__file__, 126, 18), DistutilsPlatformError_16476, *[str_16477], **kwargs_16478)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 126, 12), DistutilsPlatformError_call_result_16479, 'raise parameter', BaseException)
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 130)
        self_16480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'self')
        # Obtaining the member 'skip_build' of a type (line 130)
        skip_build_16481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 15), self_16480, 'skip_build')
        # Applying the 'not' unary operator (line 130)
        result_not__16482 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 11), 'not', skip_build_16481)
        
        # Testing the type of an if condition (line 130)
        if_condition_16483 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), result_not__16482)
        # Assigning a type to the variable 'if_condition_16483' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_16483', if_condition_16483)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to run_command(...): (line 131)
        # Processing the call arguments (line 131)
        str_16486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 29), 'str', 'build')
        # Processing the call keyword arguments (line 131)
        kwargs_16487 = {}
        # Getting the type of 'self' (line 131)
        self_16484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'self', False)
        # Obtaining the member 'run_command' of a type (line 131)
        run_command_16485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), self_16484, 'run_command')
        # Calling run_command(args, kwargs) (line 131)
        run_command_call_result_16488 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), run_command_16485, *[str_16486], **kwargs_16487)
        
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 133):
        
        # Call to reinitialize_command(...): (line 133)
        # Processing the call arguments (line 133)
        str_16491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 44), 'str', 'install')
        # Processing the call keyword arguments (line 133)
        int_16492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 74), 'int')
        keyword_16493 = int_16492
        kwargs_16494 = {'reinit_subcommands': keyword_16493}
        # Getting the type of 'self' (line 133)
        self_16489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 18), 'self', False)
        # Obtaining the member 'reinitialize_command' of a type (line 133)
        reinitialize_command_16490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 18), self_16489, 'reinitialize_command')
        # Calling reinitialize_command(args, kwargs) (line 133)
        reinitialize_command_call_result_16495 = invoke(stypy.reporting.localization.Localization(__file__, 133, 18), reinitialize_command_16490, *[str_16491], **kwargs_16494)
        
        # Assigning a type to the variable 'install' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'install', reinitialize_command_call_result_16495)
        
        # Assigning a Attribute to a Attribute (line 134):
        # Getting the type of 'self' (line 134)
        self_16496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 23), 'self')
        # Obtaining the member 'bdist_dir' of a type (line 134)
        bdist_dir_16497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 23), self_16496, 'bdist_dir')
        # Getting the type of 'install' (line 134)
        install_16498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'install')
        # Setting the type of the member 'root' of a type (line 134)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), install_16498, 'root', bdist_dir_16497)
        
        # Assigning a Attribute to a Attribute (line 135):
        # Getting the type of 'self' (line 135)
        self_16499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 29), 'self')
        # Obtaining the member 'skip_build' of a type (line 135)
        skip_build_16500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 29), self_16499, 'skip_build')
        # Getting the type of 'install' (line 135)
        install_16501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'install')
        # Setting the type of the member 'skip_build' of a type (line 135)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), install_16501, 'skip_build', skip_build_16500)
        
        # Assigning a Num to a Attribute (line 136):
        int_16502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 27), 'int')
        # Getting the type of 'install' (line 136)
        install_16503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'install')
        # Setting the type of the member 'warn_dir' of a type (line 136)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), install_16503, 'warn_dir', int_16502)
        
        # Assigning a Attribute to a Attribute (line 137):
        # Getting the type of 'self' (line 137)
        self_16504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 28), 'self')
        # Obtaining the member 'plat_name' of a type (line 137)
        plat_name_16505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 28), self_16504, 'plat_name')
        # Getting the type of 'install' (line 137)
        install_16506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'install')
        # Setting the type of the member 'plat_name' of a type (line 137)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), install_16506, 'plat_name', plat_name_16505)
        
        # Assigning a Call to a Name (line 139):
        
        # Call to reinitialize_command(...): (line 139)
        # Processing the call arguments (line 139)
        str_16509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 48), 'str', 'install_lib')
        # Processing the call keyword arguments (line 139)
        kwargs_16510 = {}
        # Getting the type of 'self' (line 139)
        self_16507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 22), 'self', False)
        # Obtaining the member 'reinitialize_command' of a type (line 139)
        reinitialize_command_16508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 22), self_16507, 'reinitialize_command')
        # Calling reinitialize_command(args, kwargs) (line 139)
        reinitialize_command_call_result_16511 = invoke(stypy.reporting.localization.Localization(__file__, 139, 22), reinitialize_command_16508, *[str_16509], **kwargs_16510)
        
        # Assigning a type to the variable 'install_lib' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'install_lib', reinitialize_command_call_result_16511)
        
        # Assigning a Num to a Attribute (line 141):
        int_16512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 30), 'int')
        # Getting the type of 'install_lib' (line 141)
        install_lib_16513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'install_lib')
        # Setting the type of the member 'compile' of a type (line 141)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), install_lib_16513, 'compile', int_16512)
        
        # Assigning a Num to a Attribute (line 142):
        int_16514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 31), 'int')
        # Getting the type of 'install_lib' (line 142)
        install_lib_16515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'install_lib')
        # Setting the type of the member 'optimize' of a type (line 142)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), install_lib_16515, 'optimize', int_16514)
        
        
        # Call to has_ext_modules(...): (line 144)
        # Processing the call keyword arguments (line 144)
        kwargs_16519 = {}
        # Getting the type of 'self' (line 144)
        self_16516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'self', False)
        # Obtaining the member 'distribution' of a type (line 144)
        distribution_16517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 11), self_16516, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 144)
        has_ext_modules_16518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 11), distribution_16517, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 144)
        has_ext_modules_call_result_16520 = invoke(stypy.reporting.localization.Localization(__file__, 144, 11), has_ext_modules_16518, *[], **kwargs_16519)
        
        # Testing the type of an if condition (line 144)
        if_condition_16521 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 8), has_ext_modules_call_result_16520)
        # Assigning a type to the variable 'if_condition_16521' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'if_condition_16521', if_condition_16521)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 151):
        # Getting the type of 'self' (line 151)
        self_16522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 29), 'self')
        # Obtaining the member 'target_version' of a type (line 151)
        target_version_16523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 29), self_16522, 'target_version')
        # Assigning a type to the variable 'target_version' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'target_version', target_version_16523)
        
        
        # Getting the type of 'target_version' (line 152)
        target_version_16524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 19), 'target_version')
        # Applying the 'not' unary operator (line 152)
        result_not__16525 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 15), 'not', target_version_16524)
        
        # Testing the type of an if condition (line 152)
        if_condition_16526 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 12), result_not__16525)
        # Assigning a type to the variable 'if_condition_16526' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'if_condition_16526', if_condition_16526)
        # SSA begins for if statement (line 152)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Evaluating assert statement condition
        # Getting the type of 'self' (line 153)
        self_16527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 23), 'self')
        # Obtaining the member 'skip_build' of a type (line 153)
        skip_build_16528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 23), self_16527, 'skip_build')
        
        # Assigning a Subscript to a Name (line 154):
        
        # Obtaining the type of the subscript
        int_16529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 45), 'int')
        int_16530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 47), 'int')
        slice_16531 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 154, 33), int_16529, int_16530, None)
        # Getting the type of 'sys' (line 154)
        sys_16532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 33), 'sys')
        # Obtaining the member 'version' of a type (line 154)
        version_16533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 33), sys_16532, 'version')
        # Obtaining the member '__getitem__' of a type (line 154)
        getitem___16534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 33), version_16533, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 154)
        subscript_call_result_16535 = invoke(stypy.reporting.localization.Localization(__file__, 154, 33), getitem___16534, slice_16531)
        
        # Assigning a type to the variable 'target_version' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'target_version', subscript_call_result_16535)
        # SSA join for if statement (line 152)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 155):
        str_16536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 29), 'str', '.%s-%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 155)
        tuple_16537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 155)
        # Adding element type (line 155)
        # Getting the type of 'self' (line 155)
        self_16538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 41), 'self')
        # Obtaining the member 'plat_name' of a type (line 155)
        plat_name_16539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 41), self_16538, 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 41), tuple_16537, plat_name_16539)
        # Adding element type (line 155)
        # Getting the type of 'target_version' (line 155)
        target_version_16540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 57), 'target_version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 41), tuple_16537, target_version_16540)
        
        # Applying the binary operator '%' (line 155)
        result_mod_16541 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 29), '%', str_16536, tuple_16537)
        
        # Assigning a type to the variable 'plat_specifier' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'plat_specifier', result_mod_16541)
        
        # Assigning a Call to a Name (line 156):
        
        # Call to get_finalized_command(...): (line 156)
        # Processing the call arguments (line 156)
        str_16544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 47), 'str', 'build')
        # Processing the call keyword arguments (line 156)
        kwargs_16545 = {}
        # Getting the type of 'self' (line 156)
        self_16542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 20), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 156)
        get_finalized_command_16543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 20), self_16542, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 156)
        get_finalized_command_call_result_16546 = invoke(stypy.reporting.localization.Localization(__file__, 156, 20), get_finalized_command_16543, *[str_16544], **kwargs_16545)
        
        # Assigning a type to the variable 'build' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'build', get_finalized_command_call_result_16546)
        
        # Assigning a Call to a Attribute (line 157):
        
        # Call to join(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'build' (line 157)
        build_16550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 43), 'build', False)
        # Obtaining the member 'build_base' of a type (line 157)
        build_base_16551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 43), build_16550, 'build_base')
        str_16552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 43), 'str', 'lib')
        # Getting the type of 'plat_specifier' (line 158)
        plat_specifier_16553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 51), 'plat_specifier', False)
        # Applying the binary operator '+' (line 158)
        result_add_16554 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 43), '+', str_16552, plat_specifier_16553)
        
        # Processing the call keyword arguments (line 157)
        kwargs_16555 = {}
        # Getting the type of 'os' (line 157)
        os_16547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 30), 'os', False)
        # Obtaining the member 'path' of a type (line 157)
        path_16548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 30), os_16547, 'path')
        # Obtaining the member 'join' of a type (line 157)
        join_16549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 30), path_16548, 'join')
        # Calling join(args, kwargs) (line 157)
        join_call_result_16556 = invoke(stypy.reporting.localization.Localization(__file__, 157, 30), join_16549, *[build_base_16551, result_add_16554], **kwargs_16555)
        
        # Getting the type of 'build' (line 157)
        build_16557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'build')
        # Setting the type of the member 'build_lib' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), build_16557, 'build_lib', join_call_result_16556)
        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 162)
        tuple_16558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 162)
        # Adding element type (line 162)
        str_16559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 20), 'str', 'purelib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 20), tuple_16558, str_16559)
        # Adding element type (line 162)
        str_16560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 31), 'str', 'platlib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 20), tuple_16558, str_16560)
        # Adding element type (line 162)
        str_16561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 42), 'str', 'headers')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 20), tuple_16558, str_16561)
        # Adding element type (line 162)
        str_16562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 53), 'str', 'scripts')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 20), tuple_16558, str_16562)
        # Adding element type (line 162)
        str_16563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 64), 'str', 'data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 20), tuple_16558, str_16563)
        
        # Testing the type of a for loop iterable (line 162)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 162, 8), tuple_16558)
        # Getting the type of the for loop variable (line 162)
        for_loop_var_16564 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 162, 8), tuple_16558)
        # Assigning a type to the variable 'key' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'key', for_loop_var_16564)
        # SSA begins for a for statement (line 162)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 163):
        
        # Call to upper(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'key' (line 163)
        key_16567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 33), 'key', False)
        # Processing the call keyword arguments (line 163)
        kwargs_16568 = {}
        # Getting the type of 'string' (line 163)
        string_16565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'string', False)
        # Obtaining the member 'upper' of a type (line 163)
        upper_16566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 20), string_16565, 'upper')
        # Calling upper(args, kwargs) (line 163)
        upper_call_result_16569 = invoke(stypy.reporting.localization.Localization(__file__, 163, 20), upper_16566, *[key_16567], **kwargs_16568)
        
        # Assigning a type to the variable 'value' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'value', upper_call_result_16569)
        
        
        # Getting the type of 'key' (line 164)
        key_16570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'key')
        str_16571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 22), 'str', 'headers')
        # Applying the binary operator '==' (line 164)
        result_eq_16572 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 15), '==', key_16570, str_16571)
        
        # Testing the type of an if condition (line 164)
        if_condition_16573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 12), result_eq_16572)
        # Assigning a type to the variable 'if_condition_16573' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'if_condition_16573', if_condition_16573)
        # SSA begins for if statement (line 164)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 165):
        # Getting the type of 'value' (line 165)
        value_16574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 24), 'value')
        str_16575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 32), 'str', '/Include/$dist_name')
        # Applying the binary operator '+' (line 165)
        result_add_16576 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 24), '+', value_16574, str_16575)
        
        # Assigning a type to the variable 'value' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'value', result_add_16576)
        # SSA join for if statement (line 164)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to setattr(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'install' (line 166)
        install_16578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 20), 'install', False)
        str_16579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 20), 'str', 'install_')
        # Getting the type of 'key' (line 167)
        key_16580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 33), 'key', False)
        # Applying the binary operator '+' (line 167)
        result_add_16581 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 20), '+', str_16579, key_16580)
        
        # Getting the type of 'value' (line 168)
        value_16582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'value', False)
        # Processing the call keyword arguments (line 166)
        kwargs_16583 = {}
        # Getting the type of 'setattr' (line 166)
        setattr_16577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 166)
        setattr_call_result_16584 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), setattr_16577, *[install_16578, result_add_16581, value_16582], **kwargs_16583)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to info(...): (line 170)
        # Processing the call arguments (line 170)
        str_16587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 17), 'str', 'installing to %s')
        # Getting the type of 'self' (line 170)
        self_16588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 37), 'self', False)
        # Obtaining the member 'bdist_dir' of a type (line 170)
        bdist_dir_16589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 37), self_16588, 'bdist_dir')
        # Processing the call keyword arguments (line 170)
        kwargs_16590 = {}
        # Getting the type of 'log' (line 170)
        log_16585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 170)
        info_16586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), log_16585, 'info')
        # Calling info(args, kwargs) (line 170)
        info_call_result_16591 = invoke(stypy.reporting.localization.Localization(__file__, 170, 8), info_16586, *[str_16587, bdist_dir_16589], **kwargs_16590)
        
        
        # Call to ensure_finalized(...): (line 171)
        # Processing the call keyword arguments (line 171)
        kwargs_16594 = {}
        # Getting the type of 'install' (line 171)
        install_16592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'install', False)
        # Obtaining the member 'ensure_finalized' of a type (line 171)
        ensure_finalized_16593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), install_16592, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 171)
        ensure_finalized_call_result_16595 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), ensure_finalized_16593, *[], **kwargs_16594)
        
        
        # Call to insert(...): (line 175)
        # Processing the call arguments (line 175)
        int_16599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 24), 'int')
        
        # Call to join(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'self' (line 175)
        self_16603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 40), 'self', False)
        # Obtaining the member 'bdist_dir' of a type (line 175)
        bdist_dir_16604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 40), self_16603, 'bdist_dir')
        str_16605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 56), 'str', 'PURELIB')
        # Processing the call keyword arguments (line 175)
        kwargs_16606 = {}
        # Getting the type of 'os' (line 175)
        os_16600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 175)
        path_16601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 27), os_16600, 'path')
        # Obtaining the member 'join' of a type (line 175)
        join_16602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 27), path_16601, 'join')
        # Calling join(args, kwargs) (line 175)
        join_call_result_16607 = invoke(stypy.reporting.localization.Localization(__file__, 175, 27), join_16602, *[bdist_dir_16604, str_16605], **kwargs_16606)
        
        # Processing the call keyword arguments (line 175)
        kwargs_16608 = {}
        # Getting the type of 'sys' (line 175)
        sys_16596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'sys', False)
        # Obtaining the member 'path' of a type (line 175)
        path_16597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), sys_16596, 'path')
        # Obtaining the member 'insert' of a type (line 175)
        insert_16598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), path_16597, 'insert')
        # Calling insert(args, kwargs) (line 175)
        insert_call_result_16609 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), insert_16598, *[int_16599, join_call_result_16607], **kwargs_16608)
        
        
        # Call to run(...): (line 177)
        # Processing the call keyword arguments (line 177)
        kwargs_16612 = {}
        # Getting the type of 'install' (line 177)
        install_16610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'install', False)
        # Obtaining the member 'run' of a type (line 177)
        run_16611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), install_16610, 'run')
        # Calling run(args, kwargs) (line 177)
        run_call_result_16613 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), run_16611, *[], **kwargs_16612)
        
        # Deleting a member
        # Getting the type of 'sys' (line 179)
        sys_16614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'sys')
        # Obtaining the member 'path' of a type (line 179)
        path_16615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 12), sys_16614, 'path')
        
        # Obtaining the type of the subscript
        int_16616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 21), 'int')
        # Getting the type of 'sys' (line 179)
        sys_16617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'sys')
        # Obtaining the member 'path' of a type (line 179)
        path_16618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 12), sys_16617, 'path')
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___16619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 12), path_16618, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_16620 = invoke(stypy.reporting.localization.Localization(__file__, 179, 12), getitem___16619, int_16616)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 8), path_16615, subscript_call_result_16620)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 183, 8))
        
        # 'from tempfile import mktemp' statement (line 183)
        try:
            from tempfile import mktemp

        except:
            mktemp = UndefinedType
        import_from_module(stypy.reporting.localization.Localization(__file__, 183, 8), 'tempfile', None, module_type_store, ['mktemp'], [mktemp])
        
        
        # Assigning a Call to a Name (line 184):
        
        # Call to mktemp(...): (line 184)
        # Processing the call keyword arguments (line 184)
        kwargs_16622 = {}
        # Getting the type of 'mktemp' (line 184)
        mktemp_16621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 27), 'mktemp', False)
        # Calling mktemp(args, kwargs) (line 184)
        mktemp_call_result_16623 = invoke(stypy.reporting.localization.Localization(__file__, 184, 27), mktemp_16621, *[], **kwargs_16622)
        
        # Assigning a type to the variable 'archive_basename' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'archive_basename', mktemp_call_result_16623)
        
        # Assigning a Call to a Name (line 185):
        
        # Call to get_fullname(...): (line 185)
        # Processing the call keyword arguments (line 185)
        kwargs_16627 = {}
        # Getting the type of 'self' (line 185)
        self_16624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 19), 'self', False)
        # Obtaining the member 'distribution' of a type (line 185)
        distribution_16625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 19), self_16624, 'distribution')
        # Obtaining the member 'get_fullname' of a type (line 185)
        get_fullname_16626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 19), distribution_16625, 'get_fullname')
        # Calling get_fullname(args, kwargs) (line 185)
        get_fullname_call_result_16628 = invoke(stypy.reporting.localization.Localization(__file__, 185, 19), get_fullname_16626, *[], **kwargs_16627)
        
        # Assigning a type to the variable 'fullname' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'fullname', get_fullname_call_result_16628)
        
        # Assigning a Call to a Name (line 186):
        
        # Call to make_archive(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'archive_basename' (line 186)
        archive_basename_16631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 36), 'archive_basename', False)
        str_16632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 54), 'str', 'zip')
        # Processing the call keyword arguments (line 186)
        # Getting the type of 'self' (line 187)
        self_16633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 45), 'self', False)
        # Obtaining the member 'bdist_dir' of a type (line 187)
        bdist_dir_16634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 45), self_16633, 'bdist_dir')
        keyword_16635 = bdist_dir_16634
        kwargs_16636 = {'root_dir': keyword_16635}
        # Getting the type of 'self' (line 186)
        self_16629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 18), 'self', False)
        # Obtaining the member 'make_archive' of a type (line 186)
        make_archive_16630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 18), self_16629, 'make_archive')
        # Calling make_archive(args, kwargs) (line 186)
        make_archive_call_result_16637 = invoke(stypy.reporting.localization.Localization(__file__, 186, 18), make_archive_16630, *[archive_basename_16631, str_16632], **kwargs_16636)
        
        # Assigning a type to the variable 'arcname' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'arcname', make_archive_call_result_16637)
        
        # Call to create_exe(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'arcname' (line 189)
        arcname_16640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 24), 'arcname', False)
        # Getting the type of 'fullname' (line 189)
        fullname_16641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 33), 'fullname', False)
        # Getting the type of 'self' (line 189)
        self_16642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 43), 'self', False)
        # Obtaining the member 'bitmap' of a type (line 189)
        bitmap_16643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 43), self_16642, 'bitmap')
        # Processing the call keyword arguments (line 189)
        kwargs_16644 = {}
        # Getting the type of 'self' (line 189)
        self_16638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'self', False)
        # Obtaining the member 'create_exe' of a type (line 189)
        create_exe_16639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), self_16638, 'create_exe')
        # Calling create_exe(args, kwargs) (line 189)
        create_exe_call_result_16645 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), create_exe_16639, *[arcname_16640, fullname_16641, bitmap_16643], **kwargs_16644)
        
        
        
        # Call to has_ext_modules(...): (line 190)
        # Processing the call keyword arguments (line 190)
        kwargs_16649 = {}
        # Getting the type of 'self' (line 190)
        self_16646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 11), 'self', False)
        # Obtaining the member 'distribution' of a type (line 190)
        distribution_16647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 11), self_16646, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 190)
        has_ext_modules_16648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 11), distribution_16647, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 190)
        has_ext_modules_call_result_16650 = invoke(stypy.reporting.localization.Localization(__file__, 190, 11), has_ext_modules_16648, *[], **kwargs_16649)
        
        # Testing the type of an if condition (line 190)
        if_condition_16651 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 8), has_ext_modules_call_result_16650)
        # Assigning a type to the variable 'if_condition_16651' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'if_condition_16651', if_condition_16651)
        # SSA begins for if statement (line 190)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 191):
        
        # Call to get_python_version(...): (line 191)
        # Processing the call keyword arguments (line 191)
        kwargs_16653 = {}
        # Getting the type of 'get_python_version' (line 191)
        get_python_version_16652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 24), 'get_python_version', False)
        # Calling get_python_version(args, kwargs) (line 191)
        get_python_version_call_result_16654 = invoke(stypy.reporting.localization.Localization(__file__, 191, 24), get_python_version_16652, *[], **kwargs_16653)
        
        # Assigning a type to the variable 'pyversion' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'pyversion', get_python_version_call_result_16654)
        # SSA branch for the else part of an if statement (line 190)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 193):
        str_16655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 24), 'str', 'any')
        # Assigning a type to the variable 'pyversion' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'pyversion', str_16655)
        # SSA join for if statement (line 190)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 194)
        # Processing the call arguments (line 194)
        
        # Obtaining an instance of the builtin type 'tuple' (line 194)
        tuple_16660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 194)
        # Adding element type (line 194)
        str_16661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 45), 'str', 'bdist_wininst')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 45), tuple_16660, str_16661)
        # Adding element type (line 194)
        # Getting the type of 'pyversion' (line 194)
        pyversion_16662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 62), 'pyversion', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 45), tuple_16660, pyversion_16662)
        # Adding element type (line 194)
        
        # Call to get_installer_filename(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'fullname' (line 195)
        fullname_16665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 73), 'fullname', False)
        # Processing the call keyword arguments (line 195)
        kwargs_16666 = {}
        # Getting the type of 'self' (line 195)
        self_16663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 45), 'self', False)
        # Obtaining the member 'get_installer_filename' of a type (line 195)
        get_installer_filename_16664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 45), self_16663, 'get_installer_filename')
        # Calling get_installer_filename(args, kwargs) (line 195)
        get_installer_filename_call_result_16667 = invoke(stypy.reporting.localization.Localization(__file__, 195, 45), get_installer_filename_16664, *[fullname_16665], **kwargs_16666)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 45), tuple_16660, get_installer_filename_call_result_16667)
        
        # Processing the call keyword arguments (line 194)
        kwargs_16668 = {}
        # Getting the type of 'self' (line 194)
        self_16656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'self', False)
        # Obtaining the member 'distribution' of a type (line 194)
        distribution_16657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), self_16656, 'distribution')
        # Obtaining the member 'dist_files' of a type (line 194)
        dist_files_16658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), distribution_16657, 'dist_files')
        # Obtaining the member 'append' of a type (line 194)
        append_16659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), dist_files_16658, 'append')
        # Calling append(args, kwargs) (line 194)
        append_call_result_16669 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), append_16659, *[tuple_16660], **kwargs_16668)
        
        
        # Call to debug(...): (line 197)
        # Processing the call arguments (line 197)
        str_16672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 18), 'str', "removing temporary file '%s'")
        # Getting the type of 'arcname' (line 197)
        arcname_16673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 50), 'arcname', False)
        # Processing the call keyword arguments (line 197)
        kwargs_16674 = {}
        # Getting the type of 'log' (line 197)
        log_16670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'log', False)
        # Obtaining the member 'debug' of a type (line 197)
        debug_16671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), log_16670, 'debug')
        # Calling debug(args, kwargs) (line 197)
        debug_call_result_16675 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), debug_16671, *[str_16672, arcname_16673], **kwargs_16674)
        
        
        # Call to remove(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'arcname' (line 198)
        arcname_16678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 18), 'arcname', False)
        # Processing the call keyword arguments (line 198)
        kwargs_16679 = {}
        # Getting the type of 'os' (line 198)
        os_16676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'os', False)
        # Obtaining the member 'remove' of a type (line 198)
        remove_16677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), os_16676, 'remove')
        # Calling remove(args, kwargs) (line 198)
        remove_call_result_16680 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), remove_16677, *[arcname_16678], **kwargs_16679)
        
        
        
        # Getting the type of 'self' (line 200)
        self_16681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'self')
        # Obtaining the member 'keep_temp' of a type (line 200)
        keep_temp_16682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 15), self_16681, 'keep_temp')
        # Applying the 'not' unary operator (line 200)
        result_not__16683 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 11), 'not', keep_temp_16682)
        
        # Testing the type of an if condition (line 200)
        if_condition_16684 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 8), result_not__16683)
        # Assigning a type to the variable 'if_condition_16684' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'if_condition_16684', if_condition_16684)
        # SSA begins for if statement (line 200)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove_tree(...): (line 201)
        # Processing the call arguments (line 201)
        # Getting the type of 'self' (line 201)
        self_16686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 24), 'self', False)
        # Obtaining the member 'bdist_dir' of a type (line 201)
        bdist_dir_16687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 24), self_16686, 'bdist_dir')
        # Processing the call keyword arguments (line 201)
        # Getting the type of 'self' (line 201)
        self_16688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 48), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 201)
        dry_run_16689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 48), self_16688, 'dry_run')
        keyword_16690 = dry_run_16689
        kwargs_16691 = {'dry_run': keyword_16690}
        # Getting the type of 'remove_tree' (line 201)
        remove_tree_16685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'remove_tree', False)
        # Calling remove_tree(args, kwargs) (line 201)
        remove_tree_call_result_16692 = invoke(stypy.reporting.localization.Localization(__file__, 201, 12), remove_tree_16685, *[bdist_dir_16687], **kwargs_16691)
        
        # SSA join for if statement (line 200)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_16693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16693)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_16693


    @norecursion
    def get_inidata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_inidata'
        module_type_store = module_type_store.open_function_context('get_inidata', 205, 4, False)
        # Assigning a type to the variable 'self' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_wininst.get_inidata.__dict__.__setitem__('stypy_localization', localization)
        bdist_wininst.get_inidata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_wininst.get_inidata.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_wininst.get_inidata.__dict__.__setitem__('stypy_function_name', 'bdist_wininst.get_inidata')
        bdist_wininst.get_inidata.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_wininst.get_inidata.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_wininst.get_inidata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_wininst.get_inidata.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_wininst.get_inidata.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_wininst.get_inidata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_wininst.get_inidata.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_wininst.get_inidata', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_inidata', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_inidata(...)' code ##################

        
        # Assigning a List to a Name (line 208):
        
        # Obtaining an instance of the builtin type 'list' (line 208)
        list_16694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 208)
        
        # Assigning a type to the variable 'lines' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'lines', list_16694)
        
        # Assigning a Attribute to a Name (line 209):
        # Getting the type of 'self' (line 209)
        self_16695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 19), 'self')
        # Obtaining the member 'distribution' of a type (line 209)
        distribution_16696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 19), self_16695, 'distribution')
        # Obtaining the member 'metadata' of a type (line 209)
        metadata_16697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 19), distribution_16696, 'metadata')
        # Assigning a type to the variable 'metadata' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'metadata', metadata_16697)
        
        # Call to append(...): (line 212)
        # Processing the call arguments (line 212)
        str_16700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 21), 'str', '[metadata]')
        # Processing the call keyword arguments (line 212)
        kwargs_16701 = {}
        # Getting the type of 'lines' (line 212)
        lines_16698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'lines', False)
        # Obtaining the member 'append' of a type (line 212)
        append_16699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), lines_16698, 'append')
        # Calling append(args, kwargs) (line 212)
        append_call_result_16702 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), append_16699, *[str_16700], **kwargs_16701)
        
        
        # Assigning a BinOp to a Name (line 216):
        
        # Evaluating a boolean operation
        # Getting the type of 'metadata' (line 216)
        metadata_16703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'metadata')
        # Obtaining the member 'long_description' of a type (line 216)
        long_description_16704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 16), metadata_16703, 'long_description')
        str_16705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 45), 'str', '')
        # Applying the binary operator 'or' (line 216)
        result_or_keyword_16706 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 16), 'or', long_description_16704, str_16705)
        
        str_16707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 51), 'str', '\n')
        # Applying the binary operator '+' (line 216)
        result_add_16708 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 15), '+', result_or_keyword_16706, str_16707)
        
        # Assigning a type to the variable 'info' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'info', result_add_16708)

        @norecursion
        def escape(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'escape'
            module_type_store = module_type_store.open_function_context('escape', 219, 8, False)
            
            # Passed parameters checking function
            escape.stypy_localization = localization
            escape.stypy_type_of_self = None
            escape.stypy_type_store = module_type_store
            escape.stypy_function_name = 'escape'
            escape.stypy_param_names_list = ['s']
            escape.stypy_varargs_param_name = None
            escape.stypy_kwargs_param_name = None
            escape.stypy_call_defaults = defaults
            escape.stypy_call_varargs = varargs
            escape.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'escape', ['s'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'escape', localization, ['s'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'escape(...)' code ##################

            
            # Call to replace(...): (line 220)
            # Processing the call arguments (line 220)
            # Getting the type of 's' (line 220)
            s_16711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 34), 's', False)
            str_16712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 37), 'str', '\n')
            str_16713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 43), 'str', '\\n')
            # Processing the call keyword arguments (line 220)
            kwargs_16714 = {}
            # Getting the type of 'string' (line 220)
            string_16709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'string', False)
            # Obtaining the member 'replace' of a type (line 220)
            replace_16710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 19), string_16709, 'replace')
            # Calling replace(args, kwargs) (line 220)
            replace_call_result_16715 = invoke(stypy.reporting.localization.Localization(__file__, 220, 19), replace_16710, *[s_16711, str_16712, str_16713], **kwargs_16714)
            
            # Assigning a type to the variable 'stypy_return_type' (line 220)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'stypy_return_type', replace_call_result_16715)
            
            # ################# End of 'escape(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'escape' in the type store
            # Getting the type of 'stypy_return_type' (line 219)
            stypy_return_type_16716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_16716)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'escape'
            return stypy_return_type_16716

        # Assigning a type to the variable 'escape' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'escape', escape)
        
        
        # Obtaining an instance of the builtin type 'list' (line 222)
        list_16717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 222)
        # Adding element type (line 222)
        str_16718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 21), 'str', 'author')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 20), list_16717, str_16718)
        # Adding element type (line 222)
        str_16719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 31), 'str', 'author_email')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 20), list_16717, str_16719)
        # Adding element type (line 222)
        str_16720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 47), 'str', 'description')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 20), list_16717, str_16720)
        # Adding element type (line 222)
        str_16721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 62), 'str', 'maintainer')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 20), list_16717, str_16721)
        # Adding element type (line 222)
        str_16722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 21), 'str', 'maintainer_email')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 20), list_16717, str_16722)
        # Adding element type (line 222)
        str_16723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 41), 'str', 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 20), list_16717, str_16723)
        # Adding element type (line 222)
        str_16724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 49), 'str', 'url')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 20), list_16717, str_16724)
        # Adding element type (line 222)
        str_16725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 56), 'str', 'version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 20), list_16717, str_16725)
        
        # Testing the type of a for loop iterable (line 222)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 222, 8), list_16717)
        # Getting the type of the for loop variable (line 222)
        for_loop_var_16726 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 222, 8), list_16717)
        # Assigning a type to the variable 'name' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'name', for_loop_var_16726)
        # SSA begins for a for statement (line 222)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 224):
        
        # Call to getattr(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'metadata' (line 224)
        metadata_16728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 27), 'metadata', False)
        # Getting the type of 'name' (line 224)
        name_16729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 37), 'name', False)
        str_16730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 43), 'str', '')
        # Processing the call keyword arguments (line 224)
        kwargs_16731 = {}
        # Getting the type of 'getattr' (line 224)
        getattr_16727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 19), 'getattr', False)
        # Calling getattr(args, kwargs) (line 224)
        getattr_call_result_16732 = invoke(stypy.reporting.localization.Localization(__file__, 224, 19), getattr_16727, *[metadata_16728, name_16729, str_16730], **kwargs_16731)
        
        # Assigning a type to the variable 'data' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'data', getattr_call_result_16732)
        
        # Getting the type of 'data' (line 225)
        data_16733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 15), 'data')
        # Testing the type of an if condition (line 225)
        if_condition_16734 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 12), data_16733)
        # Assigning a type to the variable 'if_condition_16734' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'if_condition_16734', if_condition_16734)
        # SSA begins for if statement (line 225)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 226):
        # Getting the type of 'info' (line 226)
        info_16735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), 'info')
        str_16736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 31), 'str', '\n    %s: %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 227)
        tuple_16737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 227)
        # Adding element type (line 227)
        
        # Call to capitalize(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'name' (line 227)
        name_16740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 50), 'name', False)
        # Processing the call keyword arguments (line 227)
        kwargs_16741 = {}
        # Getting the type of 'string' (line 227)
        string_16738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 32), 'string', False)
        # Obtaining the member 'capitalize' of a type (line 227)
        capitalize_16739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 32), string_16738, 'capitalize')
        # Calling capitalize(args, kwargs) (line 227)
        capitalize_call_result_16742 = invoke(stypy.reporting.localization.Localization(__file__, 227, 32), capitalize_16739, *[name_16740], **kwargs_16741)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 32), tuple_16737, capitalize_call_result_16742)
        # Adding element type (line 227)
        
        # Call to escape(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'data' (line 227)
        data_16744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 64), 'data', False)
        # Processing the call keyword arguments (line 227)
        kwargs_16745 = {}
        # Getting the type of 'escape' (line 227)
        escape_16743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 57), 'escape', False)
        # Calling escape(args, kwargs) (line 227)
        escape_call_result_16746 = invoke(stypy.reporting.localization.Localization(__file__, 227, 57), escape_16743, *[data_16744], **kwargs_16745)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 227, 32), tuple_16737, escape_call_result_16746)
        
        # Applying the binary operator '%' (line 226)
        result_mod_16747 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 31), '%', str_16736, tuple_16737)
        
        # Applying the binary operator '+' (line 226)
        result_add_16748 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 23), '+', info_16735, result_mod_16747)
        
        # Assigning a type to the variable 'info' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'info', result_add_16748)
        
        # Call to append(...): (line 228)
        # Processing the call arguments (line 228)
        str_16751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 29), 'str', '%s=%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 228)
        tuple_16752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 228)
        # Adding element type (line 228)
        # Getting the type of 'name' (line 228)
        name_16753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 40), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 40), tuple_16752, name_16753)
        # Adding element type (line 228)
        
        # Call to escape(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'data' (line 228)
        data_16755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 53), 'data', False)
        # Processing the call keyword arguments (line 228)
        kwargs_16756 = {}
        # Getting the type of 'escape' (line 228)
        escape_16754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 46), 'escape', False)
        # Calling escape(args, kwargs) (line 228)
        escape_call_result_16757 = invoke(stypy.reporting.localization.Localization(__file__, 228, 46), escape_16754, *[data_16755], **kwargs_16756)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 40), tuple_16752, escape_call_result_16757)
        
        # Applying the binary operator '%' (line 228)
        result_mod_16758 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 29), '%', str_16751, tuple_16752)
        
        # Processing the call keyword arguments (line 228)
        kwargs_16759 = {}
        # Getting the type of 'lines' (line 228)
        lines_16749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'lines', False)
        # Obtaining the member 'append' of a type (line 228)
        append_16750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 16), lines_16749, 'append')
        # Calling append(args, kwargs) (line 228)
        append_call_result_16760 = invoke(stypy.reporting.localization.Localization(__file__, 228, 16), append_16750, *[result_mod_16758], **kwargs_16759)
        
        # SSA join for if statement (line 225)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 232)
        # Processing the call arguments (line 232)
        str_16763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 21), 'str', '\n[Setup]')
        # Processing the call keyword arguments (line 232)
        kwargs_16764 = {}
        # Getting the type of 'lines' (line 232)
        lines_16761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'lines', False)
        # Obtaining the member 'append' of a type (line 232)
        append_16762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), lines_16761, 'append')
        # Calling append(args, kwargs) (line 232)
        append_call_result_16765 = invoke(stypy.reporting.localization.Localization(__file__, 232, 8), append_16762, *[str_16763], **kwargs_16764)
        
        
        # Getting the type of 'self' (line 233)
        self_16766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 11), 'self')
        # Obtaining the member 'install_script' of a type (line 233)
        install_script_16767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 11), self_16766, 'install_script')
        # Testing the type of an if condition (line 233)
        if_condition_16768 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 233, 8), install_script_16767)
        # Assigning a type to the variable 'if_condition_16768' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'if_condition_16768', if_condition_16768)
        # SSA begins for if statement (line 233)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 234)
        # Processing the call arguments (line 234)
        str_16771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 25), 'str', 'install_script=%s')
        # Getting the type of 'self' (line 234)
        self_16772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 47), 'self', False)
        # Obtaining the member 'install_script' of a type (line 234)
        install_script_16773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 47), self_16772, 'install_script')
        # Applying the binary operator '%' (line 234)
        result_mod_16774 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 25), '%', str_16771, install_script_16773)
        
        # Processing the call keyword arguments (line 234)
        kwargs_16775 = {}
        # Getting the type of 'lines' (line 234)
        lines_16769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'lines', False)
        # Obtaining the member 'append' of a type (line 234)
        append_16770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 12), lines_16769, 'append')
        # Calling append(args, kwargs) (line 234)
        append_call_result_16776 = invoke(stypy.reporting.localization.Localization(__file__, 234, 12), append_16770, *[result_mod_16774], **kwargs_16775)
        
        # SSA join for if statement (line 233)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 235)
        # Processing the call arguments (line 235)
        str_16779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 21), 'str', 'info=%s')
        
        # Call to escape(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 'info' (line 235)
        info_16781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 40), 'info', False)
        # Processing the call keyword arguments (line 235)
        kwargs_16782 = {}
        # Getting the type of 'escape' (line 235)
        escape_16780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 33), 'escape', False)
        # Calling escape(args, kwargs) (line 235)
        escape_call_result_16783 = invoke(stypy.reporting.localization.Localization(__file__, 235, 33), escape_16780, *[info_16781], **kwargs_16782)
        
        # Applying the binary operator '%' (line 235)
        result_mod_16784 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 21), '%', str_16779, escape_call_result_16783)
        
        # Processing the call keyword arguments (line 235)
        kwargs_16785 = {}
        # Getting the type of 'lines' (line 235)
        lines_16777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'lines', False)
        # Obtaining the member 'append' of a type (line 235)
        append_16778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), lines_16777, 'append')
        # Calling append(args, kwargs) (line 235)
        append_call_result_16786 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), append_16778, *[result_mod_16784], **kwargs_16785)
        
        
        # Call to append(...): (line 236)
        # Processing the call arguments (line 236)
        str_16789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 21), 'str', 'target_compile=%d')
        
        # Getting the type of 'self' (line 236)
        self_16790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 48), 'self', False)
        # Obtaining the member 'no_target_compile' of a type (line 236)
        no_target_compile_16791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 48), self_16790, 'no_target_compile')
        # Applying the 'not' unary operator (line 236)
        result_not__16792 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 44), 'not', no_target_compile_16791)
        
        # Applying the binary operator '%' (line 236)
        result_mod_16793 = python_operator(stypy.reporting.localization.Localization(__file__, 236, 21), '%', str_16789, result_not__16792)
        
        # Processing the call keyword arguments (line 236)
        kwargs_16794 = {}
        # Getting the type of 'lines' (line 236)
        lines_16787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'lines', False)
        # Obtaining the member 'append' of a type (line 236)
        append_16788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), lines_16787, 'append')
        # Calling append(args, kwargs) (line 236)
        append_call_result_16795 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), append_16788, *[result_mod_16793], **kwargs_16794)
        
        
        # Call to append(...): (line 237)
        # Processing the call arguments (line 237)
        str_16798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 21), 'str', 'target_optimize=%d')
        
        # Getting the type of 'self' (line 237)
        self_16799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 49), 'self', False)
        # Obtaining the member 'no_target_optimize' of a type (line 237)
        no_target_optimize_16800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 49), self_16799, 'no_target_optimize')
        # Applying the 'not' unary operator (line 237)
        result_not__16801 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 45), 'not', no_target_optimize_16800)
        
        # Applying the binary operator '%' (line 237)
        result_mod_16802 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 21), '%', str_16798, result_not__16801)
        
        # Processing the call keyword arguments (line 237)
        kwargs_16803 = {}
        # Getting the type of 'lines' (line 237)
        lines_16796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'lines', False)
        # Obtaining the member 'append' of a type (line 237)
        append_16797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), lines_16796, 'append')
        # Calling append(args, kwargs) (line 237)
        append_call_result_16804 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), append_16797, *[result_mod_16802], **kwargs_16803)
        
        
        # Getting the type of 'self' (line 238)
        self_16805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 11), 'self')
        # Obtaining the member 'target_version' of a type (line 238)
        target_version_16806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 11), self_16805, 'target_version')
        # Testing the type of an if condition (line 238)
        if_condition_16807 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 8), target_version_16806)
        # Assigning a type to the variable 'if_condition_16807' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'if_condition_16807', if_condition_16807)
        # SSA begins for if statement (line 238)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 239)
        # Processing the call arguments (line 239)
        str_16810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 25), 'str', 'target_version=%s')
        # Getting the type of 'self' (line 239)
        self_16811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 47), 'self', False)
        # Obtaining the member 'target_version' of a type (line 239)
        target_version_16812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 47), self_16811, 'target_version')
        # Applying the binary operator '%' (line 239)
        result_mod_16813 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 25), '%', str_16810, target_version_16812)
        
        # Processing the call keyword arguments (line 239)
        kwargs_16814 = {}
        # Getting the type of 'lines' (line 239)
        lines_16808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'lines', False)
        # Obtaining the member 'append' of a type (line 239)
        append_16809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), lines_16808, 'append')
        # Calling append(args, kwargs) (line 239)
        append_call_result_16815 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), append_16809, *[result_mod_16813], **kwargs_16814)
        
        # SSA join for if statement (line 238)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 240)
        self_16816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'self')
        # Obtaining the member 'user_access_control' of a type (line 240)
        user_access_control_16817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 11), self_16816, 'user_access_control')
        # Testing the type of an if condition (line 240)
        if_condition_16818 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 8), user_access_control_16817)
        # Assigning a type to the variable 'if_condition_16818' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'if_condition_16818', if_condition_16818)
        # SSA begins for if statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 241)
        # Processing the call arguments (line 241)
        str_16821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 25), 'str', 'user_access_control=%s')
        # Getting the type of 'self' (line 241)
        self_16822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 52), 'self', False)
        # Obtaining the member 'user_access_control' of a type (line 241)
        user_access_control_16823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 52), self_16822, 'user_access_control')
        # Applying the binary operator '%' (line 241)
        result_mod_16824 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 25), '%', str_16821, user_access_control_16823)
        
        # Processing the call keyword arguments (line 241)
        kwargs_16825 = {}
        # Getting the type of 'lines' (line 241)
        lines_16819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'lines', False)
        # Obtaining the member 'append' of a type (line 241)
        append_16820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), lines_16819, 'append')
        # Calling append(args, kwargs) (line 241)
        append_call_result_16826 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), append_16820, *[result_mod_16824], **kwargs_16825)
        
        # SSA join for if statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BoolOp to a Name (line 243):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 243)
        self_16827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'self')
        # Obtaining the member 'title' of a type (line 243)
        title_16828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 16), self_16827, 'title')
        
        # Call to get_fullname(...): (line 243)
        # Processing the call keyword arguments (line 243)
        kwargs_16832 = {}
        # Getting the type of 'self' (line 243)
        self_16829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 30), 'self', False)
        # Obtaining the member 'distribution' of a type (line 243)
        distribution_16830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 30), self_16829, 'distribution')
        # Obtaining the member 'get_fullname' of a type (line 243)
        get_fullname_16831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 30), distribution_16830, 'get_fullname')
        # Calling get_fullname(args, kwargs) (line 243)
        get_fullname_call_result_16833 = invoke(stypy.reporting.localization.Localization(__file__, 243, 30), get_fullname_16831, *[], **kwargs_16832)
        
        # Applying the binary operator 'or' (line 243)
        result_or_keyword_16834 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 16), 'or', title_16828, get_fullname_call_result_16833)
        
        # Assigning a type to the variable 'title' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'title', result_or_keyword_16834)
        
        # Call to append(...): (line 244)
        # Processing the call arguments (line 244)
        str_16837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 21), 'str', 'title=%s')
        
        # Call to escape(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'title' (line 244)
        title_16839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 41), 'title', False)
        # Processing the call keyword arguments (line 244)
        kwargs_16840 = {}
        # Getting the type of 'escape' (line 244)
        escape_16838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 34), 'escape', False)
        # Calling escape(args, kwargs) (line 244)
        escape_call_result_16841 = invoke(stypy.reporting.localization.Localization(__file__, 244, 34), escape_16838, *[title_16839], **kwargs_16840)
        
        # Applying the binary operator '%' (line 244)
        result_mod_16842 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 21), '%', str_16837, escape_call_result_16841)
        
        # Processing the call keyword arguments (line 244)
        kwargs_16843 = {}
        # Getting the type of 'lines' (line 244)
        lines_16835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'lines', False)
        # Obtaining the member 'append' of a type (line 244)
        append_16836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), lines_16835, 'append')
        # Calling append(args, kwargs) (line 244)
        append_call_result_16844 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), append_16836, *[result_mod_16842], **kwargs_16843)
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 245, 8))
        
        # 'import time' statement (line 245)
        import time

        import_module(stypy.reporting.localization.Localization(__file__, 245, 8), 'time', time, module_type_store)
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 246, 8))
        
        # 'import distutils' statement (line 246)
        import distutils

        import_module(stypy.reporting.localization.Localization(__file__, 246, 8), 'distutils', distutils, module_type_store)
        
        
        # Assigning a BinOp to a Name (line 247):
        str_16845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 21), 'str', 'Built %s with distutils-%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 248)
        tuple_16846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 248)
        # Adding element type (line 248)
        
        # Call to ctime(...): (line 248)
        # Processing the call arguments (line 248)
        
        # Call to time(...): (line 248)
        # Processing the call keyword arguments (line 248)
        kwargs_16851 = {}
        # Getting the type of 'time' (line 248)
        time_16849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 33), 'time', False)
        # Obtaining the member 'time' of a type (line 248)
        time_16850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 33), time_16849, 'time')
        # Calling time(args, kwargs) (line 248)
        time_call_result_16852 = invoke(stypy.reporting.localization.Localization(__file__, 248, 33), time_16850, *[], **kwargs_16851)
        
        # Processing the call keyword arguments (line 248)
        kwargs_16853 = {}
        # Getting the type of 'time' (line 248)
        time_16847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 22), 'time', False)
        # Obtaining the member 'ctime' of a type (line 248)
        ctime_16848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 22), time_16847, 'ctime')
        # Calling ctime(args, kwargs) (line 248)
        ctime_call_result_16854 = invoke(stypy.reporting.localization.Localization(__file__, 248, 22), ctime_16848, *[time_call_result_16852], **kwargs_16853)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 22), tuple_16846, ctime_call_result_16854)
        # Adding element type (line 248)
        # Getting the type of 'distutils' (line 248)
        distutils_16855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 47), 'distutils')
        # Obtaining the member '__version__' of a type (line 248)
        version___16856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 47), distutils_16855, '__version__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 22), tuple_16846, version___16856)
        
        # Applying the binary operator '%' (line 247)
        result_mod_16857 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 21), '%', str_16845, tuple_16846)
        
        # Assigning a type to the variable 'build_info' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'build_info', result_mod_16857)
        
        # Call to append(...): (line 249)
        # Processing the call arguments (line 249)
        str_16860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 21), 'str', 'build_info=%s')
        # Getting the type of 'build_info' (line 249)
        build_info_16861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 39), 'build_info', False)
        # Applying the binary operator '%' (line 249)
        result_mod_16862 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 21), '%', str_16860, build_info_16861)
        
        # Processing the call keyword arguments (line 249)
        kwargs_16863 = {}
        # Getting the type of 'lines' (line 249)
        lines_16858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'lines', False)
        # Obtaining the member 'append' of a type (line 249)
        append_16859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), lines_16858, 'append')
        # Calling append(args, kwargs) (line 249)
        append_call_result_16864 = invoke(stypy.reporting.localization.Localization(__file__, 249, 8), append_16859, *[result_mod_16862], **kwargs_16863)
        
        
        # Call to join(...): (line 250)
        # Processing the call arguments (line 250)
        # Getting the type of 'lines' (line 250)
        lines_16867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 27), 'lines', False)
        str_16868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 34), 'str', '\n')
        # Processing the call keyword arguments (line 250)
        kwargs_16869 = {}
        # Getting the type of 'string' (line 250)
        string_16865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 15), 'string', False)
        # Obtaining the member 'join' of a type (line 250)
        join_16866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 15), string_16865, 'join')
        # Calling join(args, kwargs) (line 250)
        join_call_result_16870 = invoke(stypy.reporting.localization.Localization(__file__, 250, 15), join_16866, *[lines_16867, str_16868], **kwargs_16869)
        
        # Assigning a type to the variable 'stypy_return_type' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'stypy_return_type', join_call_result_16870)
        
        # ################# End of 'get_inidata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_inidata' in the type store
        # Getting the type of 'stypy_return_type' (line 205)
        stypy_return_type_16871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16871)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_inidata'
        return stypy_return_type_16871


    @norecursion
    def create_exe(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 254)
        None_16872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 52), 'None')
        defaults = [None_16872]
        # Create a new context for function 'create_exe'
        module_type_store = module_type_store.open_function_context('create_exe', 254, 4, False)
        # Assigning a type to the variable 'self' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_wininst.create_exe.__dict__.__setitem__('stypy_localization', localization)
        bdist_wininst.create_exe.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_wininst.create_exe.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_wininst.create_exe.__dict__.__setitem__('stypy_function_name', 'bdist_wininst.create_exe')
        bdist_wininst.create_exe.__dict__.__setitem__('stypy_param_names_list', ['arcname', 'fullname', 'bitmap'])
        bdist_wininst.create_exe.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_wininst.create_exe.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_wininst.create_exe.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_wininst.create_exe.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_wininst.create_exe.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_wininst.create_exe.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_wininst.create_exe', ['arcname', 'fullname', 'bitmap'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_exe', localization, ['arcname', 'fullname', 'bitmap'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_exe(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 255, 8))
        
        # 'import struct' statement (line 255)
        import struct

        import_module(stypy.reporting.localization.Localization(__file__, 255, 8), 'struct', struct, module_type_store)
        
        
        # Call to mkpath(...): (line 257)
        # Processing the call arguments (line 257)
        # Getting the type of 'self' (line 257)
        self_16875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 20), 'self', False)
        # Obtaining the member 'dist_dir' of a type (line 257)
        dist_dir_16876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 20), self_16875, 'dist_dir')
        # Processing the call keyword arguments (line 257)
        kwargs_16877 = {}
        # Getting the type of 'self' (line 257)
        self_16873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 257)
        mkpath_16874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), self_16873, 'mkpath')
        # Calling mkpath(args, kwargs) (line 257)
        mkpath_call_result_16878 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), mkpath_16874, *[dist_dir_16876], **kwargs_16877)
        
        
        # Assigning a Call to a Name (line 259):
        
        # Call to get_inidata(...): (line 259)
        # Processing the call keyword arguments (line 259)
        kwargs_16881 = {}
        # Getting the type of 'self' (line 259)
        self_16879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 18), 'self', False)
        # Obtaining the member 'get_inidata' of a type (line 259)
        get_inidata_16880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 18), self_16879, 'get_inidata')
        # Calling get_inidata(args, kwargs) (line 259)
        get_inidata_call_result_16882 = invoke(stypy.reporting.localization.Localization(__file__, 259, 18), get_inidata_16880, *[], **kwargs_16881)
        
        # Assigning a type to the variable 'cfgdata' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'cfgdata', get_inidata_call_result_16882)
        
        # Assigning a Call to a Name (line 261):
        
        # Call to get_installer_filename(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'fullname' (line 261)
        fullname_16885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 53), 'fullname', False)
        # Processing the call keyword arguments (line 261)
        kwargs_16886 = {}
        # Getting the type of 'self' (line 261)
        self_16883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 25), 'self', False)
        # Obtaining the member 'get_installer_filename' of a type (line 261)
        get_installer_filename_16884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 25), self_16883, 'get_installer_filename')
        # Calling get_installer_filename(args, kwargs) (line 261)
        get_installer_filename_call_result_16887 = invoke(stypy.reporting.localization.Localization(__file__, 261, 25), get_installer_filename_16884, *[fullname_16885], **kwargs_16886)
        
        # Assigning a type to the variable 'installer_name' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'installer_name', get_installer_filename_call_result_16887)
        
        # Call to announce(...): (line 262)
        # Processing the call arguments (line 262)
        str_16890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 22), 'str', 'creating %s')
        # Getting the type of 'installer_name' (line 262)
        installer_name_16891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 38), 'installer_name', False)
        # Applying the binary operator '%' (line 262)
        result_mod_16892 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 22), '%', str_16890, installer_name_16891)
        
        # Processing the call keyword arguments (line 262)
        kwargs_16893 = {}
        # Getting the type of 'self' (line 262)
        self_16888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'self', False)
        # Obtaining the member 'announce' of a type (line 262)
        announce_16889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), self_16888, 'announce')
        # Calling announce(args, kwargs) (line 262)
        announce_call_result_16894 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), announce_16889, *[result_mod_16892], **kwargs_16893)
        
        
        # Getting the type of 'bitmap' (line 264)
        bitmap_16895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 11), 'bitmap')
        # Testing the type of an if condition (line 264)
        if_condition_16896 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 264, 8), bitmap_16895)
        # Assigning a type to the variable 'if_condition_16896' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'if_condition_16896', if_condition_16896)
        # SSA begins for if statement (line 264)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 265):
        
        # Call to read(...): (line 265)
        # Processing the call keyword arguments (line 265)
        kwargs_16903 = {}
        
        # Call to open(...): (line 265)
        # Processing the call arguments (line 265)
        # Getting the type of 'bitmap' (line 265)
        bitmap_16898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 30), 'bitmap', False)
        str_16899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 38), 'str', 'rb')
        # Processing the call keyword arguments (line 265)
        kwargs_16900 = {}
        # Getting the type of 'open' (line 265)
        open_16897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 25), 'open', False)
        # Calling open(args, kwargs) (line 265)
        open_call_result_16901 = invoke(stypy.reporting.localization.Localization(__file__, 265, 25), open_16897, *[bitmap_16898, str_16899], **kwargs_16900)
        
        # Obtaining the member 'read' of a type (line 265)
        read_16902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 25), open_call_result_16901, 'read')
        # Calling read(args, kwargs) (line 265)
        read_call_result_16904 = invoke(stypy.reporting.localization.Localization(__file__, 265, 25), read_16902, *[], **kwargs_16903)
        
        # Assigning a type to the variable 'bitmapdata' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'bitmapdata', read_call_result_16904)
        
        # Assigning a Call to a Name (line 266):
        
        # Call to len(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'bitmapdata' (line 266)
        bitmapdata_16906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 28), 'bitmapdata', False)
        # Processing the call keyword arguments (line 266)
        kwargs_16907 = {}
        # Getting the type of 'len' (line 266)
        len_16905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 24), 'len', False)
        # Calling len(args, kwargs) (line 266)
        len_call_result_16908 = invoke(stypy.reporting.localization.Localization(__file__, 266, 24), len_16905, *[bitmapdata_16906], **kwargs_16907)
        
        # Assigning a type to the variable 'bitmaplen' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'bitmaplen', len_call_result_16908)
        # SSA branch for the else part of an if statement (line 264)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 268):
        int_16909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 24), 'int')
        # Assigning a type to the variable 'bitmaplen' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'bitmaplen', int_16909)
        # SSA join for if statement (line 264)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 270):
        
        # Call to open(...): (line 270)
        # Processing the call arguments (line 270)
        # Getting the type of 'installer_name' (line 270)
        installer_name_16911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'installer_name', False)
        str_16912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 36), 'str', 'wb')
        # Processing the call keyword arguments (line 270)
        kwargs_16913 = {}
        # Getting the type of 'open' (line 270)
        open_16910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 15), 'open', False)
        # Calling open(args, kwargs) (line 270)
        open_call_result_16914 = invoke(stypy.reporting.localization.Localization(__file__, 270, 15), open_16910, *[installer_name_16911, str_16912], **kwargs_16913)
        
        # Assigning a type to the variable 'file' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'file', open_call_result_16914)
        
        # Call to write(...): (line 271)
        # Processing the call arguments (line 271)
        
        # Call to get_exe_bytes(...): (line 271)
        # Processing the call keyword arguments (line 271)
        kwargs_16919 = {}
        # Getting the type of 'self' (line 271)
        self_16917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 19), 'self', False)
        # Obtaining the member 'get_exe_bytes' of a type (line 271)
        get_exe_bytes_16918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 19), self_16917, 'get_exe_bytes')
        # Calling get_exe_bytes(args, kwargs) (line 271)
        get_exe_bytes_call_result_16920 = invoke(stypy.reporting.localization.Localization(__file__, 271, 19), get_exe_bytes_16918, *[], **kwargs_16919)
        
        # Processing the call keyword arguments (line 271)
        kwargs_16921 = {}
        # Getting the type of 'file' (line 271)
        file_16915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'file', False)
        # Obtaining the member 'write' of a type (line 271)
        write_16916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 8), file_16915, 'write')
        # Calling write(args, kwargs) (line 271)
        write_call_result_16922 = invoke(stypy.reporting.localization.Localization(__file__, 271, 8), write_16916, *[get_exe_bytes_call_result_16920], **kwargs_16921)
        
        
        # Getting the type of 'bitmap' (line 272)
        bitmap_16923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 11), 'bitmap')
        # Testing the type of an if condition (line 272)
        if_condition_16924 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 8), bitmap_16923)
        # Assigning a type to the variable 'if_condition_16924' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'if_condition_16924', if_condition_16924)
        # SSA begins for if statement (line 272)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 273)
        # Processing the call arguments (line 273)
        # Getting the type of 'bitmapdata' (line 273)
        bitmapdata_16927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 23), 'bitmapdata', False)
        # Processing the call keyword arguments (line 273)
        kwargs_16928 = {}
        # Getting the type of 'file' (line 273)
        file_16925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'file', False)
        # Obtaining the member 'write' of a type (line 273)
        write_16926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 12), file_16925, 'write')
        # Calling write(args, kwargs) (line 273)
        write_call_result_16929 = invoke(stypy.reporting.localization.Localization(__file__, 273, 12), write_16926, *[bitmapdata_16927], **kwargs_16928)
        
        # SSA join for if statement (line 272)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 276)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        # Getting the type of 'unicode' (line 277)
        unicode_16930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 12), 'unicode')
        # SSA branch for the except part of a try statement (line 276)
        # SSA branch for the except 'NameError' branch of a try statement (line 276)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA branch for the else branch of a try statement (line 276)
        module_type_store.open_ssa_branch('except else')
        
        # Type idiom detected: calculating its left and rigth part (line 281)
        # Getting the type of 'unicode' (line 281)
        unicode_16931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 35), 'unicode')
        # Getting the type of 'cfgdata' (line 281)
        cfgdata_16932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 26), 'cfgdata')
        
        (may_be_16933, more_types_in_union_16934) = may_be_subtype(unicode_16931, cfgdata_16932)

        if may_be_16933:

            if more_types_in_union_16934:
                # Runtime conditional SSA (line 281)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'cfgdata' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'cfgdata', remove_not_subtype_from_union(cfgdata_16932, unicode))
            
            # Assigning a Call to a Name (line 282):
            
            # Call to encode(...): (line 282)
            # Processing the call arguments (line 282)
            str_16937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 41), 'str', 'mbcs')
            # Processing the call keyword arguments (line 282)
            kwargs_16938 = {}
            # Getting the type of 'cfgdata' (line 282)
            cfgdata_16935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 26), 'cfgdata', False)
            # Obtaining the member 'encode' of a type (line 282)
            encode_16936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 26), cfgdata_16935, 'encode')
            # Calling encode(args, kwargs) (line 282)
            encode_call_result_16939 = invoke(stypy.reporting.localization.Localization(__file__, 282, 26), encode_16936, *[str_16937], **kwargs_16938)
            
            # Assigning a type to the variable 'cfgdata' (line 282)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 16), 'cfgdata', encode_call_result_16939)

            if more_types_in_union_16934:
                # SSA join for if statement (line 281)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for try-except statement (line 276)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 285):
        # Getting the type of 'cfgdata' (line 285)
        cfgdata_16940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 18), 'cfgdata')
        str_16941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 28), 'str', '\x00')
        # Applying the binary operator '+' (line 285)
        result_add_16942 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 18), '+', cfgdata_16940, str_16941)
        
        # Assigning a type to the variable 'cfgdata' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'cfgdata', result_add_16942)
        
        # Getting the type of 'self' (line 286)
        self_16943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 11), 'self')
        # Obtaining the member 'pre_install_script' of a type (line 286)
        pre_install_script_16944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 11), self_16943, 'pre_install_script')
        # Testing the type of an if condition (line 286)
        if_condition_16945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 8), pre_install_script_16944)
        # Assigning a type to the variable 'if_condition_16945' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'if_condition_16945', if_condition_16945)
        # SSA begins for if statement (line 286)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 287):
        
        # Call to read(...): (line 287)
        # Processing the call keyword arguments (line 287)
        kwargs_16953 = {}
        
        # Call to open(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'self' (line 287)
        self_16947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), 'self', False)
        # Obtaining the member 'pre_install_script' of a type (line 287)
        pre_install_script_16948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 31), self_16947, 'pre_install_script')
        str_16949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 56), 'str', 'r')
        # Processing the call keyword arguments (line 287)
        kwargs_16950 = {}
        # Getting the type of 'open' (line 287)
        open_16946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 26), 'open', False)
        # Calling open(args, kwargs) (line 287)
        open_call_result_16951 = invoke(stypy.reporting.localization.Localization(__file__, 287, 26), open_16946, *[pre_install_script_16948, str_16949], **kwargs_16950)
        
        # Obtaining the member 'read' of a type (line 287)
        read_16952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 26), open_call_result_16951, 'read')
        # Calling read(args, kwargs) (line 287)
        read_call_result_16954 = invoke(stypy.reporting.localization.Localization(__file__, 287, 26), read_16952, *[], **kwargs_16953)
        
        # Assigning a type to the variable 'script_data' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 12), 'script_data', read_call_result_16954)
        
        # Assigning a BinOp to a Name (line 288):
        # Getting the type of 'cfgdata' (line 288)
        cfgdata_16955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 22), 'cfgdata')
        # Getting the type of 'script_data' (line 288)
        script_data_16956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 32), 'script_data')
        # Applying the binary operator '+' (line 288)
        result_add_16957 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 22), '+', cfgdata_16955, script_data_16956)
        
        str_16958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 46), 'str', '\n\x00')
        # Applying the binary operator '+' (line 288)
        result_add_16959 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 44), '+', result_add_16957, str_16958)
        
        # Assigning a type to the variable 'cfgdata' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'cfgdata', result_add_16959)
        # SSA branch for the else part of an if statement (line 286)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 291):
        # Getting the type of 'cfgdata' (line 291)
        cfgdata_16960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 22), 'cfgdata')
        str_16961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 32), 'str', '\x00')
        # Applying the binary operator '+' (line 291)
        result_add_16962 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 22), '+', cfgdata_16960, str_16961)
        
        # Assigning a type to the variable 'cfgdata' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 12), 'cfgdata', result_add_16962)
        # SSA join for if statement (line 286)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to write(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 'cfgdata' (line 292)
        cfgdata_16965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 19), 'cfgdata', False)
        # Processing the call keyword arguments (line 292)
        kwargs_16966 = {}
        # Getting the type of 'file' (line 292)
        file_16963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'file', False)
        # Obtaining the member 'write' of a type (line 292)
        write_16964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), file_16963, 'write')
        # Calling write(args, kwargs) (line 292)
        write_call_result_16967 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), write_16964, *[cfgdata_16965], **kwargs_16966)
        
        
        # Assigning a Call to a Name (line 299):
        
        # Call to pack(...): (line 299)
        # Processing the call arguments (line 299)
        str_16970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 29), 'str', '<iii')
        int_16971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 29), 'int')
        
        # Call to len(...): (line 301)
        # Processing the call arguments (line 301)
        # Getting the type of 'cfgdata' (line 301)
        cfgdata_16973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 33), 'cfgdata', False)
        # Processing the call keyword arguments (line 301)
        kwargs_16974 = {}
        # Getting the type of 'len' (line 301)
        len_16972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 29), 'len', False)
        # Calling len(args, kwargs) (line 301)
        len_call_result_16975 = invoke(stypy.reporting.localization.Localization(__file__, 301, 29), len_16972, *[cfgdata_16973], **kwargs_16974)
        
        # Getting the type of 'bitmaplen' (line 302)
        bitmaplen_16976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 29), 'bitmaplen', False)
        # Processing the call keyword arguments (line 299)
        kwargs_16977 = {}
        # Getting the type of 'struct' (line 299)
        struct_16968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 17), 'struct', False)
        # Obtaining the member 'pack' of a type (line 299)
        pack_16969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 17), struct_16968, 'pack')
        # Calling pack(args, kwargs) (line 299)
        pack_call_result_16978 = invoke(stypy.reporting.localization.Localization(__file__, 299, 17), pack_16969, *[str_16970, int_16971, len_call_result_16975, bitmaplen_16976], **kwargs_16977)
        
        # Assigning a type to the variable 'header' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'header', pack_call_result_16978)
        
        # Call to write(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'header' (line 304)
        header_16981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 19), 'header', False)
        # Processing the call keyword arguments (line 304)
        kwargs_16982 = {}
        # Getting the type of 'file' (line 304)
        file_16979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'file', False)
        # Obtaining the member 'write' of a type (line 304)
        write_16980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), file_16979, 'write')
        # Calling write(args, kwargs) (line 304)
        write_call_result_16983 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), write_16980, *[header_16981], **kwargs_16982)
        
        
        # Call to write(...): (line 305)
        # Processing the call arguments (line 305)
        
        # Call to read(...): (line 305)
        # Processing the call keyword arguments (line 305)
        kwargs_16992 = {}
        
        # Call to open(...): (line 305)
        # Processing the call arguments (line 305)
        # Getting the type of 'arcname' (line 305)
        arcname_16987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 24), 'arcname', False)
        str_16988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 33), 'str', 'rb')
        # Processing the call keyword arguments (line 305)
        kwargs_16989 = {}
        # Getting the type of 'open' (line 305)
        open_16986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 19), 'open', False)
        # Calling open(args, kwargs) (line 305)
        open_call_result_16990 = invoke(stypy.reporting.localization.Localization(__file__, 305, 19), open_16986, *[arcname_16987, str_16988], **kwargs_16989)
        
        # Obtaining the member 'read' of a type (line 305)
        read_16991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 19), open_call_result_16990, 'read')
        # Calling read(args, kwargs) (line 305)
        read_call_result_16993 = invoke(stypy.reporting.localization.Localization(__file__, 305, 19), read_16991, *[], **kwargs_16992)
        
        # Processing the call keyword arguments (line 305)
        kwargs_16994 = {}
        # Getting the type of 'file' (line 305)
        file_16984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'file', False)
        # Obtaining the member 'write' of a type (line 305)
        write_16985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 8), file_16984, 'write')
        # Calling write(args, kwargs) (line 305)
        write_call_result_16995 = invoke(stypy.reporting.localization.Localization(__file__, 305, 8), write_16985, *[read_call_result_16993], **kwargs_16994)
        
        
        # ################# End of 'create_exe(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_exe' in the type store
        # Getting the type of 'stypy_return_type' (line 254)
        stypy_return_type_16996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16996)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_exe'
        return stypy_return_type_16996


    @norecursion
    def get_installer_filename(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_installer_filename'
        module_type_store = module_type_store.open_function_context('get_installer_filename', 309, 4, False)
        # Assigning a type to the variable 'self' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_wininst.get_installer_filename.__dict__.__setitem__('stypy_localization', localization)
        bdist_wininst.get_installer_filename.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_wininst.get_installer_filename.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_wininst.get_installer_filename.__dict__.__setitem__('stypy_function_name', 'bdist_wininst.get_installer_filename')
        bdist_wininst.get_installer_filename.__dict__.__setitem__('stypy_param_names_list', ['fullname'])
        bdist_wininst.get_installer_filename.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_wininst.get_installer_filename.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_wininst.get_installer_filename.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_wininst.get_installer_filename.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_wininst.get_installer_filename.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_wininst.get_installer_filename.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_wininst.get_installer_filename', ['fullname'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_installer_filename', localization, ['fullname'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_installer_filename(...)' code ##################

        
        # Getting the type of 'self' (line 311)
        self_16997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 11), 'self')
        # Obtaining the member 'target_version' of a type (line 311)
        target_version_16998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 11), self_16997, 'target_version')
        # Testing the type of an if condition (line 311)
        if_condition_16999 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 311, 8), target_version_16998)
        # Assigning a type to the variable 'if_condition_16999' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'if_condition_16999', if_condition_16999)
        # SSA begins for if statement (line 311)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 314):
        
        # Call to join(...): (line 314)
        # Processing the call arguments (line 314)
        # Getting the type of 'self' (line 314)
        self_17003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 42), 'self', False)
        # Obtaining the member 'dist_dir' of a type (line 314)
        dist_dir_17004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 42), self_17003, 'dist_dir')
        str_17005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 42), 'str', '%s.%s-py%s.exe')
        
        # Obtaining an instance of the builtin type 'tuple' (line 316)
        tuple_17006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 44), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 316)
        # Adding element type (line 316)
        # Getting the type of 'fullname' (line 316)
        fullname_17007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 44), 'fullname', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 44), tuple_17006, fullname_17007)
        # Adding element type (line 316)
        # Getting the type of 'self' (line 316)
        self_17008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 54), 'self', False)
        # Obtaining the member 'plat_name' of a type (line 316)
        plat_name_17009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 54), self_17008, 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 44), tuple_17006, plat_name_17009)
        # Adding element type (line 316)
        # Getting the type of 'self' (line 316)
        self_17010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 70), 'self', False)
        # Obtaining the member 'target_version' of a type (line 316)
        target_version_17011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 70), self_17010, 'target_version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 316, 44), tuple_17006, target_version_17011)
        
        # Applying the binary operator '%' (line 315)
        result_mod_17012 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 42), '%', str_17005, tuple_17006)
        
        # Processing the call keyword arguments (line 314)
        kwargs_17013 = {}
        # Getting the type of 'os' (line 314)
        os_17000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 29), 'os', False)
        # Obtaining the member 'path' of a type (line 314)
        path_17001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 29), os_17000, 'path')
        # Obtaining the member 'join' of a type (line 314)
        join_17002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 29), path_17001, 'join')
        # Calling join(args, kwargs) (line 314)
        join_call_result_17014 = invoke(stypy.reporting.localization.Localization(__file__, 314, 29), join_17002, *[dist_dir_17004, result_mod_17012], **kwargs_17013)
        
        # Assigning a type to the variable 'installer_name' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'installer_name', join_call_result_17014)
        # SSA branch for the else part of an if statement (line 311)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 318):
        
        # Call to join(...): (line 318)
        # Processing the call arguments (line 318)
        # Getting the type of 'self' (line 318)
        self_17018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 42), 'self', False)
        # Obtaining the member 'dist_dir' of a type (line 318)
        dist_dir_17019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 42), self_17018, 'dist_dir')
        str_17020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 42), 'str', '%s.%s.exe')
        
        # Obtaining an instance of the builtin type 'tuple' (line 319)
        tuple_17021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 319)
        # Adding element type (line 319)
        # Getting the type of 'fullname' (line 319)
        fullname_17022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 57), 'fullname', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 57), tuple_17021, fullname_17022)
        # Adding element type (line 319)
        # Getting the type of 'self' (line 319)
        self_17023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 67), 'self', False)
        # Obtaining the member 'plat_name' of a type (line 319)
        plat_name_17024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 67), self_17023, 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 319, 57), tuple_17021, plat_name_17024)
        
        # Applying the binary operator '%' (line 319)
        result_mod_17025 = python_operator(stypy.reporting.localization.Localization(__file__, 319, 42), '%', str_17020, tuple_17021)
        
        # Processing the call keyword arguments (line 318)
        kwargs_17026 = {}
        # Getting the type of 'os' (line 318)
        os_17015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 29), 'os', False)
        # Obtaining the member 'path' of a type (line 318)
        path_17016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 29), os_17015, 'path')
        # Obtaining the member 'join' of a type (line 318)
        join_17017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 29), path_17016, 'join')
        # Calling join(args, kwargs) (line 318)
        join_call_result_17027 = invoke(stypy.reporting.localization.Localization(__file__, 318, 29), join_17017, *[dist_dir_17019, result_mod_17025], **kwargs_17026)
        
        # Assigning a type to the variable 'installer_name' (line 318)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'installer_name', join_call_result_17027)
        # SSA join for if statement (line 311)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'installer_name' (line 320)
        installer_name_17028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 15), 'installer_name')
        # Assigning a type to the variable 'stypy_return_type' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'stypy_return_type', installer_name_17028)
        
        # ################# End of 'get_installer_filename(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_installer_filename' in the type store
        # Getting the type of 'stypy_return_type' (line 309)
        stypy_return_type_17029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17029)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_installer_filename'
        return stypy_return_type_17029


    @norecursion
    def get_exe_bytes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_exe_bytes'
        module_type_store = module_type_store.open_function_context('get_exe_bytes', 323, 4, False)
        # Assigning a type to the variable 'self' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_wininst.get_exe_bytes.__dict__.__setitem__('stypy_localization', localization)
        bdist_wininst.get_exe_bytes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_wininst.get_exe_bytes.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_wininst.get_exe_bytes.__dict__.__setitem__('stypy_function_name', 'bdist_wininst.get_exe_bytes')
        bdist_wininst.get_exe_bytes.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_wininst.get_exe_bytes.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_wininst.get_exe_bytes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_wininst.get_exe_bytes.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_wininst.get_exe_bytes.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_wininst.get_exe_bytes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_wininst.get_exe_bytes.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_wininst.get_exe_bytes', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_exe_bytes', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_exe_bytes(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 324, 8))
        
        # 'from distutils.msvccompiler import get_build_version' statement (line 324)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
        import_17030 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 324, 8), 'distutils.msvccompiler')

        if (type(import_17030) is not StypyTypeError):

            if (import_17030 != 'pyd_module'):
                __import__(import_17030)
                sys_modules_17031 = sys.modules[import_17030]
                import_from_module(stypy.reporting.localization.Localization(__file__, 324, 8), 'distutils.msvccompiler', sys_modules_17031.module_type_store, module_type_store, ['get_build_version'])
                nest_module(stypy.reporting.localization.Localization(__file__, 324, 8), __file__, sys_modules_17031, sys_modules_17031.module_type_store, module_type_store)
            else:
                from distutils.msvccompiler import get_build_version

                import_from_module(stypy.reporting.localization.Localization(__file__, 324, 8), 'distutils.msvccompiler', None, module_type_store, ['get_build_version'], [get_build_version])

        else:
            # Assigning a type to the variable 'distutils.msvccompiler' (line 324)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'distutils.msvccompiler', import_17030)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
        
        
        # Assigning a Call to a Name (line 333):
        
        # Call to get_python_version(...): (line 333)
        # Processing the call keyword arguments (line 333)
        kwargs_17033 = {}
        # Getting the type of 'get_python_version' (line 333)
        get_python_version_17032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 22), 'get_python_version', False)
        # Calling get_python_version(args, kwargs) (line 333)
        get_python_version_call_result_17034 = invoke(stypy.reporting.localization.Localization(__file__, 333, 22), get_python_version_17032, *[], **kwargs_17033)
        
        # Assigning a type to the variable 'cur_version' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'cur_version', get_python_version_call_result_17034)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 334)
        self_17035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 11), 'self')
        # Obtaining the member 'target_version' of a type (line 334)
        target_version_17036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 11), self_17035, 'target_version')
        
        # Getting the type of 'self' (line 334)
        self_17037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 35), 'self')
        # Obtaining the member 'target_version' of a type (line 334)
        target_version_17038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 35), self_17037, 'target_version')
        # Getting the type of 'cur_version' (line 334)
        cur_version_17039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 58), 'cur_version')
        # Applying the binary operator '!=' (line 334)
        result_ne_17040 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 35), '!=', target_version_17038, cur_version_17039)
        
        # Applying the binary operator 'and' (line 334)
        result_and_keyword_17041 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 11), 'and', target_version_17036, result_ne_17040)
        
        # Testing the type of an if condition (line 334)
        if_condition_17042 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 8), result_and_keyword_17041)
        # Assigning a type to the variable 'if_condition_17042' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'if_condition_17042', if_condition_17042)
        # SSA begins for if statement (line 334)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 338)
        self_17043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 15), 'self')
        # Obtaining the member 'target_version' of a type (line 338)
        target_version_17044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 15), self_17043, 'target_version')
        # Getting the type of 'cur_version' (line 338)
        cur_version_17045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 37), 'cur_version')
        # Applying the binary operator '>' (line 338)
        result_gt_17046 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 15), '>', target_version_17044, cur_version_17045)
        
        # Testing the type of an if condition (line 338)
        if_condition_17047 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 12), result_gt_17046)
        # Assigning a type to the variable 'if_condition_17047' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'if_condition_17047', if_condition_17047)
        # SSA begins for if statement (line 338)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 339):
        
        # Call to get_build_version(...): (line 339)
        # Processing the call keyword arguments (line 339)
        kwargs_17049 = {}
        # Getting the type of 'get_build_version' (line 339)
        get_build_version_17048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 21), 'get_build_version', False)
        # Calling get_build_version(args, kwargs) (line 339)
        get_build_version_call_result_17050 = invoke(stypy.reporting.localization.Localization(__file__, 339, 21), get_build_version_17048, *[], **kwargs_17049)
        
        # Assigning a type to the variable 'bv' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 16), 'bv', get_build_version_call_result_17050)
        # SSA branch for the else part of an if statement (line 338)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 341)
        self_17051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 19), 'self')
        # Obtaining the member 'target_version' of a type (line 341)
        target_version_17052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 19), self_17051, 'target_version')
        str_17053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 41), 'str', '2.4')
        # Applying the binary operator '<' (line 341)
        result_lt_17054 = python_operator(stypy.reporting.localization.Localization(__file__, 341, 19), '<', target_version_17052, str_17053)
        
        # Testing the type of an if condition (line 341)
        if_condition_17055 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 16), result_lt_17054)
        # Assigning a type to the variable 'if_condition_17055' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 16), 'if_condition_17055', if_condition_17055)
        # SSA begins for if statement (line 341)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 342):
        float_17056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 25), 'float')
        # Assigning a type to the variable 'bv' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 20), 'bv', float_17056)
        # SSA branch for the else part of an if statement (line 341)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 344):
        float_17057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 25), 'float')
        # Assigning a type to the variable 'bv' (line 344)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 20), 'bv', float_17057)
        # SSA join for if statement (line 341)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 338)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 334)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 347):
        
        # Call to get_build_version(...): (line 347)
        # Processing the call keyword arguments (line 347)
        kwargs_17059 = {}
        # Getting the type of 'get_build_version' (line 347)
        get_build_version_17058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 17), 'get_build_version', False)
        # Calling get_build_version(args, kwargs) (line 347)
        get_build_version_call_result_17060 = invoke(stypy.reporting.localization.Localization(__file__, 347, 17), get_build_version_17058, *[], **kwargs_17059)
        
        # Assigning a type to the variable 'bv' (line 347)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'bv', get_build_version_call_result_17060)
        # SSA join for if statement (line 334)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 350):
        
        # Call to dirname(...): (line 350)
        # Processing the call arguments (line 350)
        # Getting the type of '__file__' (line 350)
        file___17064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 36), '__file__', False)
        # Processing the call keyword arguments (line 350)
        kwargs_17065 = {}
        # Getting the type of 'os' (line 350)
        os_17061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 350)
        path_17062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 20), os_17061, 'path')
        # Obtaining the member 'dirname' of a type (line 350)
        dirname_17063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 20), path_17062, 'dirname')
        # Calling dirname(args, kwargs) (line 350)
        dirname_call_result_17066 = invoke(stypy.reporting.localization.Localization(__file__, 350, 20), dirname_17063, *[file___17064], **kwargs_17065)
        
        # Assigning a type to the variable 'directory' (line 350)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 8), 'directory', dirname_call_result_17066)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 357)
        self_17067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 11), 'self')
        # Obtaining the member 'plat_name' of a type (line 357)
        plat_name_17068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 11), self_17067, 'plat_name')
        str_17069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 29), 'str', 'win32')
        # Applying the binary operator '!=' (line 357)
        result_ne_17070 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 11), '!=', plat_name_17068, str_17069)
        
        
        
        # Obtaining the type of the subscript
        int_17071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 57), 'int')
        slice_17072 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 357, 41), None, int_17071, None)
        # Getting the type of 'self' (line 357)
        self_17073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 41), 'self')
        # Obtaining the member 'plat_name' of a type (line 357)
        plat_name_17074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 41), self_17073, 'plat_name')
        # Obtaining the member '__getitem__' of a type (line 357)
        getitem___17075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 41), plat_name_17074, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 357)
        subscript_call_result_17076 = invoke(stypy.reporting.localization.Localization(__file__, 357, 41), getitem___17075, slice_17072)
        
        str_17077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 63), 'str', 'win')
        # Applying the binary operator '==' (line 357)
        result_eq_17078 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 41), '==', subscript_call_result_17076, str_17077)
        
        # Applying the binary operator 'and' (line 357)
        result_and_keyword_17079 = python_operator(stypy.reporting.localization.Localization(__file__, 357, 11), 'and', result_ne_17070, result_eq_17078)
        
        # Testing the type of an if condition (line 357)
        if_condition_17080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 8), result_and_keyword_17079)
        # Assigning a type to the variable 'if_condition_17080' (line 357)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'if_condition_17080', if_condition_17080)
        # SSA begins for if statement (line 357)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 358):
        
        # Obtaining the type of the subscript
        int_17081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 34), 'int')
        slice_17082 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 358, 19), int_17081, None, None)
        # Getting the type of 'self' (line 358)
        self_17083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 19), 'self')
        # Obtaining the member 'plat_name' of a type (line 358)
        plat_name_17084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 19), self_17083, 'plat_name')
        # Obtaining the member '__getitem__' of a type (line 358)
        getitem___17085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 19), plat_name_17084, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 358)
        subscript_call_result_17086 = invoke(stypy.reporting.localization.Localization(__file__, 358, 19), getitem___17085, slice_17082)
        
        # Assigning a type to the variable 'sfix' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'sfix', subscript_call_result_17086)
        # SSA branch for the else part of an if statement (line 357)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 360):
        str_17087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 19), 'str', '')
        # Assigning a type to the variable 'sfix' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'sfix', str_17087)
        # SSA join for if statement (line 357)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 362):
        
        # Call to join(...): (line 362)
        # Processing the call arguments (line 362)
        # Getting the type of 'directory' (line 362)
        directory_17091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 32), 'directory', False)
        str_17092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 43), 'str', 'wininst-%.1f%s.exe')
        
        # Obtaining an instance of the builtin type 'tuple' (line 362)
        tuple_17093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 67), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 362)
        # Adding element type (line 362)
        # Getting the type of 'bv' (line 362)
        bv_17094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 67), 'bv', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 67), tuple_17093, bv_17094)
        # Adding element type (line 362)
        # Getting the type of 'sfix' (line 362)
        sfix_17095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 71), 'sfix', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 362, 67), tuple_17093, sfix_17095)
        
        # Applying the binary operator '%' (line 362)
        result_mod_17096 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 43), '%', str_17092, tuple_17093)
        
        # Processing the call keyword arguments (line 362)
        kwargs_17097 = {}
        # Getting the type of 'os' (line 362)
        os_17088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 362)
        path_17089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 19), os_17088, 'path')
        # Obtaining the member 'join' of a type (line 362)
        join_17090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 19), path_17089, 'join')
        # Calling join(args, kwargs) (line 362)
        join_call_result_17098 = invoke(stypy.reporting.localization.Localization(__file__, 362, 19), join_17090, *[directory_17091, result_mod_17096], **kwargs_17097)
        
        # Assigning a type to the variable 'filename' (line 362)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'filename', join_call_result_17098)
        
        # Assigning a Call to a Name (line 363):
        
        # Call to open(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'filename' (line 363)
        filename_17100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 17), 'filename', False)
        str_17101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 27), 'str', 'rb')
        # Processing the call keyword arguments (line 363)
        kwargs_17102 = {}
        # Getting the type of 'open' (line 363)
        open_17099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'open', False)
        # Calling open(args, kwargs) (line 363)
        open_call_result_17103 = invoke(stypy.reporting.localization.Localization(__file__, 363, 12), open_17099, *[filename_17100, str_17101], **kwargs_17102)
        
        # Assigning a type to the variable 'f' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'f', open_call_result_17103)
        
        # Try-finally block (line 364)
        
        # Call to read(...): (line 365)
        # Processing the call keyword arguments (line 365)
        kwargs_17106 = {}
        # Getting the type of 'f' (line 365)
        f_17104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 19), 'f', False)
        # Obtaining the member 'read' of a type (line 365)
        read_17105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 19), f_17104, 'read')
        # Calling read(args, kwargs) (line 365)
        read_call_result_17107 = invoke(stypy.reporting.localization.Localization(__file__, 365, 19), read_17105, *[], **kwargs_17106)
        
        # Assigning a type to the variable 'stypy_return_type' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 12), 'stypy_return_type', read_call_result_17107)
        
        # finally branch of the try-finally block (line 364)
        
        # Call to close(...): (line 367)
        # Processing the call keyword arguments (line 367)
        kwargs_17110 = {}
        # Getting the type of 'f' (line 367)
        f_17108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 367)
        close_17109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 367, 12), f_17108, 'close')
        # Calling close(args, kwargs) (line 367)
        close_call_result_17111 = invoke(stypy.reporting.localization.Localization(__file__, 367, 12), close_17109, *[], **kwargs_17110)
        
        
        
        # ################# End of 'get_exe_bytes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_exe_bytes' in the type store
        # Getting the type of 'stypy_return_type' (line 323)
        stypy_return_type_17112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17112)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_exe_bytes'
        return stypy_return_type_17112


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 20, 0, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_wininst.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'bdist_wininst' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'bdist_wininst', bdist_wininst)

# Assigning a Str to a Name (line 22):
str_17113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'str', 'create an executable installer for MS Windows')
# Getting the type of 'bdist_wininst'
bdist_wininst_17114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist_wininst')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_wininst_17114, 'description', str_17113)

# Assigning a List to a Name (line 24):

# Obtaining an instance of the builtin type 'list' (line 24)
list_17115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_17116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
str_17117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 21), 'str', 'bdist-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 21), tuple_17116, str_17117)
# Adding element type (line 24)
# Getting the type of 'None' (line 24)
None_17118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 35), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 21), tuple_17116, None_17118)
# Adding element type (line 24)
str_17119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'str', 'temporary directory for creating the distribution')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 21), tuple_17116, str_17119)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_17115, tuple_17116)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 26)
tuple_17120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 26)
# Adding element type (line 26)
str_17121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'str', 'plat-name=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), tuple_17120, str_17121)
# Adding element type (line 26)
str_17122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 35), 'str', 'p')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), tuple_17120, str_17122)
# Adding element type (line 26)
str_17123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 21), 'str', 'platform name to embed in generated filenames (default: %s)')

# Call to get_platform(...): (line 28)
# Processing the call keyword arguments (line 28)
kwargs_17125 = {}
# Getting the type of 'get_platform' (line 28)
get_platform_17124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 39), 'get_platform', False)
# Calling get_platform(args, kwargs) (line 28)
get_platform_call_result_17126 = invoke(stypy.reporting.localization.Localization(__file__, 28, 39), get_platform_17124, *[], **kwargs_17125)

# Applying the binary operator '%' (line 27)
result_mod_17127 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 21), '%', str_17123, get_platform_call_result_17126)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 21), tuple_17120, result_mod_17127)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_17115, tuple_17120)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 29)
tuple_17128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 29)
# Adding element type (line 29)
str_17129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 21), 'str', 'keep-temp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 21), tuple_17128, str_17129)
# Adding element type (line 29)
str_17130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 34), 'str', 'k')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 21), tuple_17128, str_17130)
# Adding element type (line 29)
str_17131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 21), 'str', 'keep the pseudo-installation tree around after ')
str_17132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 21), 'str', 'creating the distribution archive')
# Applying the binary operator '+' (line 30)
result_add_17133 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 21), '+', str_17131, str_17132)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 21), tuple_17128, result_add_17133)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_17115, tuple_17128)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_17134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)
# Adding element type (line 32)
str_17135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'str', 'target-version=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 21), tuple_17134, str_17135)
# Adding element type (line 32)
# Getting the type of 'None' (line 32)
None_17136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 21), tuple_17134, None_17136)
# Adding element type (line 32)
str_17137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'str', 'require a specific python version')
str_17138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 21), 'str', ' on the target system')
# Applying the binary operator '+' (line 33)
result_add_17139 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 21), '+', str_17137, str_17138)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 21), tuple_17134, result_add_17139)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_17115, tuple_17134)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_17140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
str_17141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 21), 'str', 'no-target-compile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 21), tuple_17140, str_17141)
# Adding element type (line 35)
str_17142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 42), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 21), tuple_17140, str_17142)
# Adding element type (line 35)
str_17143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 21), 'str', 'do not compile .py to .pyc on the target system')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 21), tuple_17140, str_17143)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_17115, tuple_17140)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 37)
tuple_17144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 37)
# Adding element type (line 37)
str_17145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'str', 'no-target-optimize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 21), tuple_17144, str_17145)
# Adding element type (line 37)
str_17146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 43), 'str', 'o')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 21), tuple_17144, str_17146)
# Adding element type (line 37)
str_17147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'str', 'do not compile .py to .pyo (optimized)on the target system')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 21), tuple_17144, str_17147)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_17115, tuple_17144)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 40)
tuple_17148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 40)
# Adding element type (line 40)
str_17149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 21), 'str', 'dist-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 21), tuple_17148, str_17149)
# Adding element type (line 40)
str_17150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 34), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 21), tuple_17148, str_17150)
# Adding element type (line 40)
str_17151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 21), 'str', 'directory to put final built distributions in')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 21), tuple_17148, str_17151)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_17115, tuple_17148)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 42)
tuple_17152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 42)
# Adding element type (line 42)
str_17153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 21), 'str', 'bitmap=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 21), tuple_17152, str_17153)
# Adding element type (line 42)
str_17154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 32), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 21), tuple_17152, str_17154)
# Adding element type (line 42)
str_17155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'str', 'bitmap to use for the installer instead of python-powered logo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 21), tuple_17152, str_17155)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_17115, tuple_17152)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 44)
tuple_17156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 44)
# Adding element type (line 44)
str_17157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 21), 'str', 'title=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 21), tuple_17156, str_17157)
# Adding element type (line 44)
str_17158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 31), 'str', 't')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 21), tuple_17156, str_17158)
# Adding element type (line 44)
str_17159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 21), 'str', 'title to display on the installer background instead of default')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 21), tuple_17156, str_17159)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_17115, tuple_17156)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 46)
tuple_17160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 46)
# Adding element type (line 46)
str_17161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 21), 'str', 'skip-build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 21), tuple_17160, str_17161)
# Adding element type (line 46)
# Getting the type of 'None' (line 46)
None_17162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 35), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 21), tuple_17160, None_17162)
# Adding element type (line 46)
str_17163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 21), 'str', 'skip rebuilding everything (for testing/debugging)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 21), tuple_17160, str_17163)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_17115, tuple_17160)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 48)
tuple_17164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 48)
# Adding element type (line 48)
str_17165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 21), 'str', 'install-script=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 21), tuple_17164, str_17165)
# Adding element type (line 48)
# Getting the type of 'None' (line 48)
None_17166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 40), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 21), tuple_17164, None_17166)
# Adding element type (line 48)
str_17167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 21), 'str', 'basename of installation script to be run afterinstallation or before deinstallation')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 21), tuple_17164, str_17167)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_17115, tuple_17164)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 51)
tuple_17168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 51)
# Adding element type (line 51)
str_17169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 21), 'str', 'pre-install-script=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 21), tuple_17168, str_17169)
# Adding element type (line 51)
# Getting the type of 'None' (line 51)
None_17170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 44), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 21), tuple_17168, None_17170)
# Adding element type (line 51)
str_17171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 21), 'str', 'Fully qualified filename of a script to be run before any files are installed.  This script need not be in the distribution')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 21), tuple_17168, str_17171)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_17115, tuple_17168)
# Adding element type (line 24)

# Obtaining an instance of the builtin type 'tuple' (line 55)
tuple_17172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 55)
# Adding element type (line 55)
str_17173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 21), 'str', 'user-access-control=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 21), tuple_17172, str_17173)
# Adding element type (line 55)
# Getting the type of 'None' (line 55)
None_17174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 45), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 21), tuple_17172, None_17174)
# Adding element type (line 55)
str_17175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 21), 'str', "specify Vista's UAC handling - 'none'/default=no handling, 'auto'=use UAC if target Python installed for all users, 'force'=always use UAC")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 21), tuple_17172, str_17175)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 19), list_17115, tuple_17172)

# Getting the type of 'bdist_wininst'
bdist_wininst_17176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist_wininst')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_wininst_17176, 'user_options', list_17115)

# Assigning a List to a Name (line 61):

# Obtaining an instance of the builtin type 'list' (line 61)
list_17177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 61)
# Adding element type (line 61)
str_17178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 23), 'str', 'keep-temp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 22), list_17177, str_17178)
# Adding element type (line 61)
str_17179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 36), 'str', 'no-target-compile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 22), list_17177, str_17179)
# Adding element type (line 61)
str_17180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 57), 'str', 'no-target-optimize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 22), list_17177, str_17180)
# Adding element type (line 61)
str_17181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 23), 'str', 'skip-build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 22), list_17177, str_17181)

# Getting the type of 'bdist_wininst'
bdist_wininst_17182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist_wininst')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_wininst_17182, 'boolean_options', list_17177)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
