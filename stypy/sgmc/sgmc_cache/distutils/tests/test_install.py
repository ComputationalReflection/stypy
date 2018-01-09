
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Tests for distutils.command.install.'''
2: 
3: import os
4: import sys
5: import unittest
6: import site
7: 
8: from test.test_support import captured_stdout, run_unittest
9: 
10: from distutils import sysconfig
11: from distutils.command.install import install
12: from distutils.command import install as install_module
13: from distutils.command.build_ext import build_ext
14: from distutils.command.install import INSTALL_SCHEMES
15: from distutils.core import Distribution
16: from distutils.errors import DistutilsOptionError
17: from distutils.extension import Extension
18: 
19: from distutils.tests import support
20: 
21: 
22: def _make_ext_name(modname):
23:     if os.name == 'nt' and sys.executable.endswith('_d.exe'):
24:         modname += '_d'
25:     return modname + sysconfig.get_config_var('SO')
26: 
27: 
28: class InstallTestCase(support.TempdirManager,
29:                       support.LoggingSilencer,
30:                       unittest.TestCase):
31: 
32:     def test_home_installation_scheme(self):
33:         # This ensure two things:
34:         # - that --home generates the desired set of directory names
35:         # - test --home is supported on all platforms
36:         builddir = self.mkdtemp()
37:         destination = os.path.join(builddir, "installation")
38: 
39:         dist = Distribution({"name": "foopkg"})
40:         # script_name need not exist, it just need to be initialized
41:         dist.script_name = os.path.join(builddir, "setup.py")
42:         dist.command_obj["build"] = support.DummyCommand(
43:             build_base=builddir,
44:             build_lib=os.path.join(builddir, "lib"),
45:             )
46: 
47:         cmd = install(dist)
48:         cmd.home = destination
49:         cmd.ensure_finalized()
50: 
51:         self.assertEqual(cmd.install_base, destination)
52:         self.assertEqual(cmd.install_platbase, destination)
53: 
54:         def check_path(got, expected):
55:             got = os.path.normpath(got)
56:             expected = os.path.normpath(expected)
57:             self.assertEqual(got, expected)
58: 
59:         libdir = os.path.join(destination, "lib", "python")
60:         check_path(cmd.install_lib, libdir)
61:         check_path(cmd.install_platlib, libdir)
62:         check_path(cmd.install_purelib, libdir)
63:         check_path(cmd.install_headers,
64:                    os.path.join(destination, "include", "python", "foopkg"))
65:         check_path(cmd.install_scripts, os.path.join(destination, "bin"))
66:         check_path(cmd.install_data, destination)
67: 
68:     @unittest.skipIf(sys.version < '2.6',
69:                      'site.USER_SITE was introduced in 2.6')
70:     def test_user_site(self):
71:         # preparing the environment for the test
72:         self.old_user_base = site.USER_BASE
73:         self.old_user_site = site.USER_SITE
74:         self.tmpdir = self.mkdtemp()
75:         self.user_base = os.path.join(self.tmpdir, 'B')
76:         self.user_site = os.path.join(self.tmpdir, 'S')
77:         site.USER_BASE = self.user_base
78:         site.USER_SITE = self.user_site
79:         install_module.USER_BASE = self.user_base
80:         install_module.USER_SITE = self.user_site
81: 
82:         def _expanduser(path):
83:             return self.tmpdir
84:         self.old_expand = os.path.expanduser
85:         os.path.expanduser = _expanduser
86: 
87:         def cleanup():
88:             site.USER_BASE = self.old_user_base
89:             site.USER_SITE = self.old_user_site
90:             install_module.USER_BASE = self.old_user_base
91:             install_module.USER_SITE = self.old_user_site
92:             os.path.expanduser = self.old_expand
93: 
94:         self.addCleanup(cleanup)
95: 
96:         for key in ('nt_user', 'unix_user', 'os2_home'):
97:             self.assertIn(key, INSTALL_SCHEMES)
98: 
99:         dist = Distribution({'name': 'xx'})
100:         cmd = install(dist)
101: 
102:         # making sure the user option is there
103:         options = [name for name, short, lable in
104:                    cmd.user_options]
105:         self.assertIn('user', options)
106: 
107:         # setting a value
108:         cmd.user = 1
109: 
110:         # user base and site shouldn't be created yet
111:         self.assertFalse(os.path.exists(self.user_base))
112:         self.assertFalse(os.path.exists(self.user_site))
113: 
114:         # let's run finalize
115:         cmd.ensure_finalized()
116: 
117:         # now they should
118:         self.assertTrue(os.path.exists(self.user_base))
119:         self.assertTrue(os.path.exists(self.user_site))
120: 
121:         self.assertIn('userbase', cmd.config_vars)
122:         self.assertIn('usersite', cmd.config_vars)
123: 
124:     def test_handle_extra_path(self):
125:         dist = Distribution({'name': 'xx', 'extra_path': 'path,dirs'})
126:         cmd = install(dist)
127: 
128:         # two elements
129:         cmd.handle_extra_path()
130:         self.assertEqual(cmd.extra_path, ['path', 'dirs'])
131:         self.assertEqual(cmd.extra_dirs, 'dirs')
132:         self.assertEqual(cmd.path_file, 'path')
133: 
134:         # one element
135:         cmd.extra_path = ['path']
136:         cmd.handle_extra_path()
137:         self.assertEqual(cmd.extra_path, ['path'])
138:         self.assertEqual(cmd.extra_dirs, 'path')
139:         self.assertEqual(cmd.path_file, 'path')
140: 
141:         # none
142:         dist.extra_path = cmd.extra_path = None
143:         cmd.handle_extra_path()
144:         self.assertEqual(cmd.extra_path, None)
145:         self.assertEqual(cmd.extra_dirs, '')
146:         self.assertEqual(cmd.path_file, None)
147: 
148:         # three elements (no way !)
149:         cmd.extra_path = 'path,dirs,again'
150:         self.assertRaises(DistutilsOptionError, cmd.handle_extra_path)
151: 
152:     def test_finalize_options(self):
153:         dist = Distribution({'name': 'xx'})
154:         cmd = install(dist)
155: 
156:         # must supply either prefix/exec-prefix/home or
157:         # install-base/install-platbase -- not both
158:         cmd.prefix = 'prefix'
159:         cmd.install_base = 'base'
160:         self.assertRaises(DistutilsOptionError, cmd.finalize_options)
161: 
162:         # must supply either home or prefix/exec-prefix -- not both
163:         cmd.install_base = None
164:         cmd.home = 'home'
165:         self.assertRaises(DistutilsOptionError, cmd.finalize_options)
166: 
167:         # can't combine user with prefix/exec_prefix/home or
168:         # install_(plat)base
169:         cmd.prefix = None
170:         cmd.user = 'user'
171:         self.assertRaises(DistutilsOptionError, cmd.finalize_options)
172: 
173:     def test_record(self):
174:         install_dir = self.mkdtemp()
175:         project_dir, dist = self.create_dist(py_modules=['hello'],
176:                                              scripts=['sayhi'])
177:         os.chdir(project_dir)
178:         self.write_file('hello.py', "def main(): print 'o hai'")
179:         self.write_file('sayhi', 'from hello import main; main()')
180: 
181:         cmd = install(dist)
182:         dist.command_obj['install'] = cmd
183:         cmd.root = install_dir
184:         cmd.record = os.path.join(project_dir, 'filelist')
185:         cmd.ensure_finalized()
186:         cmd.run()
187: 
188:         f = open(cmd.record)
189:         try:
190:             content = f.read()
191:         finally:
192:             f.close()
193: 
194:         found = [os.path.basename(line) for line in content.splitlines()]
195:         expected = ['hello.py', 'hello.pyc', 'sayhi',
196:                     'UNKNOWN-0.0.0-py%s.%s.egg-info' % sys.version_info[:2]]
197:         self.assertEqual(found, expected)
198: 
199:     def test_record_extensions(self):
200:         install_dir = self.mkdtemp()
201:         project_dir, dist = self.create_dist(ext_modules=[
202:             Extension('xx', ['xxmodule.c'])])
203:         os.chdir(project_dir)
204:         support.copy_xxmodule_c(project_dir)
205: 
206:         buildextcmd = build_ext(dist)
207:         support.fixup_build_ext(buildextcmd)
208:         buildextcmd.ensure_finalized()
209: 
210:         cmd = install(dist)
211:         dist.command_obj['install'] = cmd
212:         dist.command_obj['build_ext'] = buildextcmd
213:         cmd.root = install_dir
214:         cmd.record = os.path.join(project_dir, 'filelist')
215:         cmd.ensure_finalized()
216:         cmd.run()
217: 
218:         f = open(cmd.record)
219:         try:
220:             content = f.read()
221:         finally:
222:             f.close()
223: 
224:         found = [os.path.basename(line) for line in content.splitlines()]
225:         expected = [_make_ext_name('xx'),
226:                     'UNKNOWN-0.0.0-py%s.%s.egg-info' % sys.version_info[:2]]
227:         self.assertEqual(found, expected)
228: 
229:     def test_debug_mode(self):
230:         # this covers the code called when DEBUG is set
231:         old_logs_len = len(self.logs)
232:         install_module.DEBUG = True
233:         try:
234:             with captured_stdout():
235:                 self.test_record()
236:         finally:
237:             install_module.DEBUG = False
238:         self.assertGreater(len(self.logs), old_logs_len)
239: 
240: 
241: def test_suite():
242:     return unittest.makeSuite(InstallTestCase)
243: 
244: if __name__ == "__main__":
245:     run_unittest(test_suite())
246: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_39664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Tests for distutils.command.install.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import unittest' statement (line 5)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import site' statement (line 6)
import site

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'site', site, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from test.test_support import captured_stdout, run_unittest' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_39665 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'test.test_support')

if (type(import_39665) is not StypyTypeError):

    if (import_39665 != 'pyd_module'):
        __import__(import_39665)
        sys_modules_39666 = sys.modules[import_39665]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'test.test_support', sys_modules_39666.module_type_store, module_type_store, ['captured_stdout', 'run_unittest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_39666, sys_modules_39666.module_type_store, module_type_store)
    else:
        from test.test_support import captured_stdout, run_unittest

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'test.test_support', None, module_type_store, ['captured_stdout', 'run_unittest'], [captured_stdout, run_unittest])

else:
    # Assigning a type to the variable 'test.test_support' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'test.test_support', import_39665)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils import sysconfig' statement (line 10)
try:
    from distutils import sysconfig

except:
    sysconfig = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils', None, module_type_store, ['sysconfig'], [sysconfig])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.command.install import install' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_39667 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command.install')

if (type(import_39667) is not StypyTypeError):

    if (import_39667 != 'pyd_module'):
        __import__(import_39667)
        sys_modules_39668 = sys.modules[import_39667]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command.install', sys_modules_39668.module_type_store, module_type_store, ['install'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_39668, sys_modules_39668.module_type_store, module_type_store)
    else:
        from distutils.command.install import install

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command.install', None, module_type_store, ['install'], [install])

else:
    # Assigning a type to the variable 'distutils.command.install' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.command.install', import_39667)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.command import install_module' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_39669 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.command')

if (type(import_39669) is not StypyTypeError):

    if (import_39669 != 'pyd_module'):
        __import__(import_39669)
        sys_modules_39670 = sys.modules[import_39669]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.command', sys_modules_39670.module_type_store, module_type_store, ['install'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_39670, sys_modules_39670.module_type_store, module_type_store)
    else:
        from distutils.command import install as install_module

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.command', None, module_type_store, ['install'], [install_module])

else:
    # Assigning a type to the variable 'distutils.command' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.command', import_39669)

# Adding an alias
module_type_store.add_alias('install_module', 'install')
remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.command.build_ext import build_ext' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_39671 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.command.build_ext')

if (type(import_39671) is not StypyTypeError):

    if (import_39671 != 'pyd_module'):
        __import__(import_39671)
        sys_modules_39672 = sys.modules[import_39671]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.command.build_ext', sys_modules_39672.module_type_store, module_type_store, ['build_ext'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_39672, sys_modules_39672.module_type_store, module_type_store)
    else:
        from distutils.command.build_ext import build_ext

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.command.build_ext', None, module_type_store, ['build_ext'], [build_ext])

else:
    # Assigning a type to the variable 'distutils.command.build_ext' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.command.build_ext', import_39671)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.command.install import INSTALL_SCHEMES' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_39673 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.command.install')

if (type(import_39673) is not StypyTypeError):

    if (import_39673 != 'pyd_module'):
        __import__(import_39673)
        sys_modules_39674 = sys.modules[import_39673]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.command.install', sys_modules_39674.module_type_store, module_type_store, ['INSTALL_SCHEMES'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_39674, sys_modules_39674.module_type_store, module_type_store)
    else:
        from distutils.command.install import INSTALL_SCHEMES

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.command.install', None, module_type_store, ['INSTALL_SCHEMES'], [INSTALL_SCHEMES])

else:
    # Assigning a type to the variable 'distutils.command.install' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.command.install', import_39673)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.core import Distribution' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_39675 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.core')

if (type(import_39675) is not StypyTypeError):

    if (import_39675 != 'pyd_module'):
        __import__(import_39675)
        sys_modules_39676 = sys.modules[import_39675]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.core', sys_modules_39676.module_type_store, module_type_store, ['Distribution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_39676, sys_modules_39676.module_type_store, module_type_store)
    else:
        from distutils.core import Distribution

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.core', None, module_type_store, ['Distribution'], [Distribution])

else:
    # Assigning a type to the variable 'distutils.core' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.core', import_39675)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils.errors import DistutilsOptionError' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_39677 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors')

if (type(import_39677) is not StypyTypeError):

    if (import_39677 != 'pyd_module'):
        __import__(import_39677)
        sys_modules_39678 = sys.modules[import_39677]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', sys_modules_39678.module_type_store, module_type_store, ['DistutilsOptionError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_39678, sys_modules_39678.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsOptionError

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', None, module_type_store, ['DistutilsOptionError'], [DistutilsOptionError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', import_39677)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from distutils.extension import Extension' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_39679 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.extension')

if (type(import_39679) is not StypyTypeError):

    if (import_39679 != 'pyd_module'):
        __import__(import_39679)
        sys_modules_39680 = sys.modules[import_39679]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.extension', sys_modules_39680.module_type_store, module_type_store, ['Extension'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 0), __file__, sys_modules_39680, sys_modules_39680.module_type_store, module_type_store)
    else:
        from distutils.extension import Extension

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.extension', None, module_type_store, ['Extension'], [Extension])

else:
    # Assigning a type to the variable 'distutils.extension' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils.extension', import_39679)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from distutils.tests import support' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_39681 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.tests')

if (type(import_39681) is not StypyTypeError):

    if (import_39681 != 'pyd_module'):
        __import__(import_39681)
        sys_modules_39682 = sys.modules[import_39681]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.tests', sys_modules_39682.module_type_store, module_type_store, ['support'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_39682, sys_modules_39682.module_type_store, module_type_store)
    else:
        from distutils.tests import support

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.tests', None, module_type_store, ['support'], [support])

else:
    # Assigning a type to the variable 'distutils.tests' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.tests', import_39681)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


@norecursion
def _make_ext_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_make_ext_name'
    module_type_store = module_type_store.open_function_context('_make_ext_name', 22, 0, False)
    
    # Passed parameters checking function
    _make_ext_name.stypy_localization = localization
    _make_ext_name.stypy_type_of_self = None
    _make_ext_name.stypy_type_store = module_type_store
    _make_ext_name.stypy_function_name = '_make_ext_name'
    _make_ext_name.stypy_param_names_list = ['modname']
    _make_ext_name.stypy_varargs_param_name = None
    _make_ext_name.stypy_kwargs_param_name = None
    _make_ext_name.stypy_call_defaults = defaults
    _make_ext_name.stypy_call_varargs = varargs
    _make_ext_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_make_ext_name', ['modname'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_make_ext_name', localization, ['modname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_make_ext_name(...)' code ##################

    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'os' (line 23)
    os_39683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 7), 'os')
    # Obtaining the member 'name' of a type (line 23)
    name_39684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 7), os_39683, 'name')
    str_39685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 18), 'str', 'nt')
    # Applying the binary operator '==' (line 23)
    result_eq_39686 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 7), '==', name_39684, str_39685)
    
    
    # Call to endswith(...): (line 23)
    # Processing the call arguments (line 23)
    str_39690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 51), 'str', '_d.exe')
    # Processing the call keyword arguments (line 23)
    kwargs_39691 = {}
    # Getting the type of 'sys' (line 23)
    sys_39687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 27), 'sys', False)
    # Obtaining the member 'executable' of a type (line 23)
    executable_39688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 27), sys_39687, 'executable')
    # Obtaining the member 'endswith' of a type (line 23)
    endswith_39689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 27), executable_39688, 'endswith')
    # Calling endswith(args, kwargs) (line 23)
    endswith_call_result_39692 = invoke(stypy.reporting.localization.Localization(__file__, 23, 27), endswith_39689, *[str_39690], **kwargs_39691)
    
    # Applying the binary operator 'and' (line 23)
    result_and_keyword_39693 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 7), 'and', result_eq_39686, endswith_call_result_39692)
    
    # Testing the type of an if condition (line 23)
    if_condition_39694 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 4), result_and_keyword_39693)
    # Assigning a type to the variable 'if_condition_39694' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'if_condition_39694', if_condition_39694)
    # SSA begins for if statement (line 23)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'modname' (line 24)
    modname_39695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'modname')
    str_39696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'str', '_d')
    # Applying the binary operator '+=' (line 24)
    result_iadd_39697 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 8), '+=', modname_39695, str_39696)
    # Assigning a type to the variable 'modname' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'modname', result_iadd_39697)
    
    # SSA join for if statement (line 23)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'modname' (line 25)
    modname_39698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'modname')
    
    # Call to get_config_var(...): (line 25)
    # Processing the call arguments (line 25)
    str_39701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 46), 'str', 'SO')
    # Processing the call keyword arguments (line 25)
    kwargs_39702 = {}
    # Getting the type of 'sysconfig' (line 25)
    sysconfig_39699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 21), 'sysconfig', False)
    # Obtaining the member 'get_config_var' of a type (line 25)
    get_config_var_39700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 21), sysconfig_39699, 'get_config_var')
    # Calling get_config_var(args, kwargs) (line 25)
    get_config_var_call_result_39703 = invoke(stypy.reporting.localization.Localization(__file__, 25, 21), get_config_var_39700, *[str_39701], **kwargs_39702)
    
    # Applying the binary operator '+' (line 25)
    result_add_39704 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 11), '+', modname_39698, get_config_var_call_result_39703)
    
    # Assigning a type to the variable 'stypy_return_type' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type', result_add_39704)
    
    # ################# End of '_make_ext_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_make_ext_name' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_39705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39705)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_make_ext_name'
    return stypy_return_type_39705

# Assigning a type to the variable '_make_ext_name' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), '_make_ext_name', _make_ext_name)
# Declaration of the 'InstallTestCase' class
# Getting the type of 'support' (line 28)
support_39706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'support')
# Obtaining the member 'TempdirManager' of a type (line 28)
TempdirManager_39707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 22), support_39706, 'TempdirManager')
# Getting the type of 'support' (line 29)
support_39708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'support')
# Obtaining the member 'LoggingSilencer' of a type (line 29)
LoggingSilencer_39709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 22), support_39708, 'LoggingSilencer')
# Getting the type of 'unittest' (line 30)
unittest_39710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'unittest')
# Obtaining the member 'TestCase' of a type (line 30)
TestCase_39711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 22), unittest_39710, 'TestCase')

class InstallTestCase(TempdirManager_39707, LoggingSilencer_39709, TestCase_39711, ):

    @norecursion
    def test_home_installation_scheme(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_home_installation_scheme'
        module_type_store = module_type_store.open_function_context('test_home_installation_scheme', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstallTestCase.test_home_installation_scheme.__dict__.__setitem__('stypy_localization', localization)
        InstallTestCase.test_home_installation_scheme.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallTestCase.test_home_installation_scheme.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallTestCase.test_home_installation_scheme.__dict__.__setitem__('stypy_function_name', 'InstallTestCase.test_home_installation_scheme')
        InstallTestCase.test_home_installation_scheme.__dict__.__setitem__('stypy_param_names_list', [])
        InstallTestCase.test_home_installation_scheme.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallTestCase.test_home_installation_scheme.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallTestCase.test_home_installation_scheme.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallTestCase.test_home_installation_scheme.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallTestCase.test_home_installation_scheme.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallTestCase.test_home_installation_scheme.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallTestCase.test_home_installation_scheme', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_home_installation_scheme', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_home_installation_scheme(...)' code ##################

        
        # Assigning a Call to a Name (line 36):
        
        # Assigning a Call to a Name (line 36):
        
        # Call to mkdtemp(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_39714 = {}
        # Getting the type of 'self' (line 36)
        self_39712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 36)
        mkdtemp_39713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 19), self_39712, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 36)
        mkdtemp_call_result_39715 = invoke(stypy.reporting.localization.Localization(__file__, 36, 19), mkdtemp_39713, *[], **kwargs_39714)
        
        # Assigning a type to the variable 'builddir' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'builddir', mkdtemp_call_result_39715)
        
        # Assigning a Call to a Name (line 37):
        
        # Assigning a Call to a Name (line 37):
        
        # Call to join(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'builddir' (line 37)
        builddir_39719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 35), 'builddir', False)
        str_39720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 45), 'str', 'installation')
        # Processing the call keyword arguments (line 37)
        kwargs_39721 = {}
        # Getting the type of 'os' (line 37)
        os_39716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 37)
        path_39717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 22), os_39716, 'path')
        # Obtaining the member 'join' of a type (line 37)
        join_39718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 22), path_39717, 'join')
        # Calling join(args, kwargs) (line 37)
        join_call_result_39722 = invoke(stypy.reporting.localization.Localization(__file__, 37, 22), join_39718, *[builddir_39719, str_39720], **kwargs_39721)
        
        # Assigning a type to the variable 'destination' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'destination', join_call_result_39722)
        
        # Assigning a Call to a Name (line 39):
        
        # Assigning a Call to a Name (line 39):
        
        # Call to Distribution(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Obtaining an instance of the builtin type 'dict' (line 39)
        dict_39724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 39)
        # Adding element type (key, value) (line 39)
        str_39725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 29), 'str', 'name')
        str_39726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 37), 'str', 'foopkg')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 28), dict_39724, (str_39725, str_39726))
        
        # Processing the call keyword arguments (line 39)
        kwargs_39727 = {}
        # Getting the type of 'Distribution' (line 39)
        Distribution_39723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 39)
        Distribution_call_result_39728 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), Distribution_39723, *[dict_39724], **kwargs_39727)
        
        # Assigning a type to the variable 'dist' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'dist', Distribution_call_result_39728)
        
        # Assigning a Call to a Attribute (line 41):
        
        # Assigning a Call to a Attribute (line 41):
        
        # Call to join(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'builddir' (line 41)
        builddir_39732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 40), 'builddir', False)
        str_39733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 50), 'str', 'setup.py')
        # Processing the call keyword arguments (line 41)
        kwargs_39734 = {}
        # Getting the type of 'os' (line 41)
        os_39729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 41)
        path_39730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 27), os_39729, 'path')
        # Obtaining the member 'join' of a type (line 41)
        join_39731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 27), path_39730, 'join')
        # Calling join(args, kwargs) (line 41)
        join_call_result_39735 = invoke(stypy.reporting.localization.Localization(__file__, 41, 27), join_39731, *[builddir_39732, str_39733], **kwargs_39734)
        
        # Getting the type of 'dist' (line 41)
        dist_39736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'dist')
        # Setting the type of the member 'script_name' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), dist_39736, 'script_name', join_call_result_39735)
        
        # Assigning a Call to a Subscript (line 42):
        
        # Assigning a Call to a Subscript (line 42):
        
        # Call to DummyCommand(...): (line 42)
        # Processing the call keyword arguments (line 42)
        # Getting the type of 'builddir' (line 43)
        builddir_39739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'builddir', False)
        keyword_39740 = builddir_39739
        
        # Call to join(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'builddir' (line 44)
        builddir_39744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'builddir', False)
        str_39745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 45), 'str', 'lib')
        # Processing the call keyword arguments (line 44)
        kwargs_39746 = {}
        # Getting the type of 'os' (line 44)
        os_39741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 44)
        path_39742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 22), os_39741, 'path')
        # Obtaining the member 'join' of a type (line 44)
        join_39743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 22), path_39742, 'join')
        # Calling join(args, kwargs) (line 44)
        join_call_result_39747 = invoke(stypy.reporting.localization.Localization(__file__, 44, 22), join_39743, *[builddir_39744, str_39745], **kwargs_39746)
        
        keyword_39748 = join_call_result_39747
        kwargs_39749 = {'build_base': keyword_39740, 'build_lib': keyword_39748}
        # Getting the type of 'support' (line 42)
        support_39737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 36), 'support', False)
        # Obtaining the member 'DummyCommand' of a type (line 42)
        DummyCommand_39738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 36), support_39737, 'DummyCommand')
        # Calling DummyCommand(args, kwargs) (line 42)
        DummyCommand_call_result_39750 = invoke(stypy.reporting.localization.Localization(__file__, 42, 36), DummyCommand_39738, *[], **kwargs_39749)
        
        # Getting the type of 'dist' (line 42)
        dist_39751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'dist')
        # Obtaining the member 'command_obj' of a type (line 42)
        command_obj_39752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), dist_39751, 'command_obj')
        str_39753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'str', 'build')
        # Storing an element on a container (line 42)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 8), command_obj_39752, (str_39753, DummyCommand_call_result_39750))
        
        # Assigning a Call to a Name (line 47):
        
        # Assigning a Call to a Name (line 47):
        
        # Call to install(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'dist' (line 47)
        dist_39755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 22), 'dist', False)
        # Processing the call keyword arguments (line 47)
        kwargs_39756 = {}
        # Getting the type of 'install' (line 47)
        install_39754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 14), 'install', False)
        # Calling install(args, kwargs) (line 47)
        install_call_result_39757 = invoke(stypy.reporting.localization.Localization(__file__, 47, 14), install_39754, *[dist_39755], **kwargs_39756)
        
        # Assigning a type to the variable 'cmd' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'cmd', install_call_result_39757)
        
        # Assigning a Name to a Attribute (line 48):
        
        # Assigning a Name to a Attribute (line 48):
        # Getting the type of 'destination' (line 48)
        destination_39758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'destination')
        # Getting the type of 'cmd' (line 48)
        cmd_39759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'cmd')
        # Setting the type of the member 'home' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), cmd_39759, 'home', destination_39758)
        
        # Call to ensure_finalized(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_39762 = {}
        # Getting the type of 'cmd' (line 49)
        cmd_39760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 49)
        ensure_finalized_39761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), cmd_39760, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 49)
        ensure_finalized_call_result_39763 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), ensure_finalized_39761, *[], **kwargs_39762)
        
        
        # Call to assertEqual(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'cmd' (line 51)
        cmd_39766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'cmd', False)
        # Obtaining the member 'install_base' of a type (line 51)
        install_base_39767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 25), cmd_39766, 'install_base')
        # Getting the type of 'destination' (line 51)
        destination_39768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 43), 'destination', False)
        # Processing the call keyword arguments (line 51)
        kwargs_39769 = {}
        # Getting the type of 'self' (line 51)
        self_39764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 51)
        assertEqual_39765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_39764, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 51)
        assertEqual_call_result_39770 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), assertEqual_39765, *[install_base_39767, destination_39768], **kwargs_39769)
        
        
        # Call to assertEqual(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'cmd' (line 52)
        cmd_39773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 25), 'cmd', False)
        # Obtaining the member 'install_platbase' of a type (line 52)
        install_platbase_39774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 25), cmd_39773, 'install_platbase')
        # Getting the type of 'destination' (line 52)
        destination_39775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 47), 'destination', False)
        # Processing the call keyword arguments (line 52)
        kwargs_39776 = {}
        # Getting the type of 'self' (line 52)
        self_39771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 52)
        assertEqual_39772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_39771, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 52)
        assertEqual_call_result_39777 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), assertEqual_39772, *[install_platbase_39774, destination_39775], **kwargs_39776)
        

        @norecursion
        def check_path(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'check_path'
            module_type_store = module_type_store.open_function_context('check_path', 54, 8, False)
            
            # Passed parameters checking function
            check_path.stypy_localization = localization
            check_path.stypy_type_of_self = None
            check_path.stypy_type_store = module_type_store
            check_path.stypy_function_name = 'check_path'
            check_path.stypy_param_names_list = ['got', 'expected']
            check_path.stypy_varargs_param_name = None
            check_path.stypy_kwargs_param_name = None
            check_path.stypy_call_defaults = defaults
            check_path.stypy_call_varargs = varargs
            check_path.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'check_path', ['got', 'expected'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'check_path', localization, ['got', 'expected'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'check_path(...)' code ##################

            
            # Assigning a Call to a Name (line 55):
            
            # Assigning a Call to a Name (line 55):
            
            # Call to normpath(...): (line 55)
            # Processing the call arguments (line 55)
            # Getting the type of 'got' (line 55)
            got_39781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'got', False)
            # Processing the call keyword arguments (line 55)
            kwargs_39782 = {}
            # Getting the type of 'os' (line 55)
            os_39778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 18), 'os', False)
            # Obtaining the member 'path' of a type (line 55)
            path_39779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 18), os_39778, 'path')
            # Obtaining the member 'normpath' of a type (line 55)
            normpath_39780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 18), path_39779, 'normpath')
            # Calling normpath(args, kwargs) (line 55)
            normpath_call_result_39783 = invoke(stypy.reporting.localization.Localization(__file__, 55, 18), normpath_39780, *[got_39781], **kwargs_39782)
            
            # Assigning a type to the variable 'got' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'got', normpath_call_result_39783)
            
            # Assigning a Call to a Name (line 56):
            
            # Assigning a Call to a Name (line 56):
            
            # Call to normpath(...): (line 56)
            # Processing the call arguments (line 56)
            # Getting the type of 'expected' (line 56)
            expected_39787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 40), 'expected', False)
            # Processing the call keyword arguments (line 56)
            kwargs_39788 = {}
            # Getting the type of 'os' (line 56)
            os_39784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 23), 'os', False)
            # Obtaining the member 'path' of a type (line 56)
            path_39785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 23), os_39784, 'path')
            # Obtaining the member 'normpath' of a type (line 56)
            normpath_39786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 23), path_39785, 'normpath')
            # Calling normpath(args, kwargs) (line 56)
            normpath_call_result_39789 = invoke(stypy.reporting.localization.Localization(__file__, 56, 23), normpath_39786, *[expected_39787], **kwargs_39788)
            
            # Assigning a type to the variable 'expected' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'expected', normpath_call_result_39789)
            
            # Call to assertEqual(...): (line 57)
            # Processing the call arguments (line 57)
            # Getting the type of 'got' (line 57)
            got_39792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 29), 'got', False)
            # Getting the type of 'expected' (line 57)
            expected_39793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 34), 'expected', False)
            # Processing the call keyword arguments (line 57)
            kwargs_39794 = {}
            # Getting the type of 'self' (line 57)
            self_39790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'self', False)
            # Obtaining the member 'assertEqual' of a type (line 57)
            assertEqual_39791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), self_39790, 'assertEqual')
            # Calling assertEqual(args, kwargs) (line 57)
            assertEqual_call_result_39795 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), assertEqual_39791, *[got_39792, expected_39793], **kwargs_39794)
            
            
            # ################# End of 'check_path(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'check_path' in the type store
            # Getting the type of 'stypy_return_type' (line 54)
            stypy_return_type_39796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_39796)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'check_path'
            return stypy_return_type_39796

        # Assigning a type to the variable 'check_path' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'check_path', check_path)
        
        # Assigning a Call to a Name (line 59):
        
        # Assigning a Call to a Name (line 59):
        
        # Call to join(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'destination' (line 59)
        destination_39800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 30), 'destination', False)
        str_39801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 43), 'str', 'lib')
        str_39802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 50), 'str', 'python')
        # Processing the call keyword arguments (line 59)
        kwargs_39803 = {}
        # Getting the type of 'os' (line 59)
        os_39797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 59)
        path_39798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 17), os_39797, 'path')
        # Obtaining the member 'join' of a type (line 59)
        join_39799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 17), path_39798, 'join')
        # Calling join(args, kwargs) (line 59)
        join_call_result_39804 = invoke(stypy.reporting.localization.Localization(__file__, 59, 17), join_39799, *[destination_39800, str_39801, str_39802], **kwargs_39803)
        
        # Assigning a type to the variable 'libdir' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'libdir', join_call_result_39804)
        
        # Call to check_path(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'cmd' (line 60)
        cmd_39806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), 'cmd', False)
        # Obtaining the member 'install_lib' of a type (line 60)
        install_lib_39807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 19), cmd_39806, 'install_lib')
        # Getting the type of 'libdir' (line 60)
        libdir_39808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 36), 'libdir', False)
        # Processing the call keyword arguments (line 60)
        kwargs_39809 = {}
        # Getting the type of 'check_path' (line 60)
        check_path_39805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'check_path', False)
        # Calling check_path(args, kwargs) (line 60)
        check_path_call_result_39810 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), check_path_39805, *[install_lib_39807, libdir_39808], **kwargs_39809)
        
        
        # Call to check_path(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'cmd' (line 61)
        cmd_39812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'cmd', False)
        # Obtaining the member 'install_platlib' of a type (line 61)
        install_platlib_39813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 19), cmd_39812, 'install_platlib')
        # Getting the type of 'libdir' (line 61)
        libdir_39814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 40), 'libdir', False)
        # Processing the call keyword arguments (line 61)
        kwargs_39815 = {}
        # Getting the type of 'check_path' (line 61)
        check_path_39811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'check_path', False)
        # Calling check_path(args, kwargs) (line 61)
        check_path_call_result_39816 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), check_path_39811, *[install_platlib_39813, libdir_39814], **kwargs_39815)
        
        
        # Call to check_path(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'cmd' (line 62)
        cmd_39818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 19), 'cmd', False)
        # Obtaining the member 'install_purelib' of a type (line 62)
        install_purelib_39819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 19), cmd_39818, 'install_purelib')
        # Getting the type of 'libdir' (line 62)
        libdir_39820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 40), 'libdir', False)
        # Processing the call keyword arguments (line 62)
        kwargs_39821 = {}
        # Getting the type of 'check_path' (line 62)
        check_path_39817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'check_path', False)
        # Calling check_path(args, kwargs) (line 62)
        check_path_call_result_39822 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), check_path_39817, *[install_purelib_39819, libdir_39820], **kwargs_39821)
        
        
        # Call to check_path(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'cmd' (line 63)
        cmd_39824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'cmd', False)
        # Obtaining the member 'install_headers' of a type (line 63)
        install_headers_39825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 19), cmd_39824, 'install_headers')
        
        # Call to join(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'destination' (line 64)
        destination_39829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 32), 'destination', False)
        str_39830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 45), 'str', 'include')
        str_39831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 56), 'str', 'python')
        str_39832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 66), 'str', 'foopkg')
        # Processing the call keyword arguments (line 64)
        kwargs_39833 = {}
        # Getting the type of 'os' (line 64)
        os_39826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 64)
        path_39827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 19), os_39826, 'path')
        # Obtaining the member 'join' of a type (line 64)
        join_39828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 19), path_39827, 'join')
        # Calling join(args, kwargs) (line 64)
        join_call_result_39834 = invoke(stypy.reporting.localization.Localization(__file__, 64, 19), join_39828, *[destination_39829, str_39830, str_39831, str_39832], **kwargs_39833)
        
        # Processing the call keyword arguments (line 63)
        kwargs_39835 = {}
        # Getting the type of 'check_path' (line 63)
        check_path_39823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'check_path', False)
        # Calling check_path(args, kwargs) (line 63)
        check_path_call_result_39836 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), check_path_39823, *[install_headers_39825, join_call_result_39834], **kwargs_39835)
        
        
        # Call to check_path(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'cmd' (line 65)
        cmd_39838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'cmd', False)
        # Obtaining the member 'install_scripts' of a type (line 65)
        install_scripts_39839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), cmd_39838, 'install_scripts')
        
        # Call to join(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'destination' (line 65)
        destination_39843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 53), 'destination', False)
        str_39844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 66), 'str', 'bin')
        # Processing the call keyword arguments (line 65)
        kwargs_39845 = {}
        # Getting the type of 'os' (line 65)
        os_39840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 40), 'os', False)
        # Obtaining the member 'path' of a type (line 65)
        path_39841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 40), os_39840, 'path')
        # Obtaining the member 'join' of a type (line 65)
        join_39842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 40), path_39841, 'join')
        # Calling join(args, kwargs) (line 65)
        join_call_result_39846 = invoke(stypy.reporting.localization.Localization(__file__, 65, 40), join_39842, *[destination_39843, str_39844], **kwargs_39845)
        
        # Processing the call keyword arguments (line 65)
        kwargs_39847 = {}
        # Getting the type of 'check_path' (line 65)
        check_path_39837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'check_path', False)
        # Calling check_path(args, kwargs) (line 65)
        check_path_call_result_39848 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), check_path_39837, *[install_scripts_39839, join_call_result_39846], **kwargs_39847)
        
        
        # Call to check_path(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'cmd' (line 66)
        cmd_39850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 'cmd', False)
        # Obtaining the member 'install_data' of a type (line 66)
        install_data_39851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 19), cmd_39850, 'install_data')
        # Getting the type of 'destination' (line 66)
        destination_39852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 37), 'destination', False)
        # Processing the call keyword arguments (line 66)
        kwargs_39853 = {}
        # Getting the type of 'check_path' (line 66)
        check_path_39849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'check_path', False)
        # Calling check_path(args, kwargs) (line 66)
        check_path_call_result_39854 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), check_path_39849, *[install_data_39851, destination_39852], **kwargs_39853)
        
        
        # ################# End of 'test_home_installation_scheme(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_home_installation_scheme' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_39855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_39855)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_home_installation_scheme'
        return stypy_return_type_39855


    @norecursion
    def test_user_site(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_user_site'
        module_type_store = module_type_store.open_function_context('test_user_site', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstallTestCase.test_user_site.__dict__.__setitem__('stypy_localization', localization)
        InstallTestCase.test_user_site.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallTestCase.test_user_site.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallTestCase.test_user_site.__dict__.__setitem__('stypy_function_name', 'InstallTestCase.test_user_site')
        InstallTestCase.test_user_site.__dict__.__setitem__('stypy_param_names_list', [])
        InstallTestCase.test_user_site.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallTestCase.test_user_site.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallTestCase.test_user_site.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallTestCase.test_user_site.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallTestCase.test_user_site.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallTestCase.test_user_site.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallTestCase.test_user_site', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_user_site', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_user_site(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 72):
        
        # Assigning a Attribute to a Attribute (line 72):
        # Getting the type of 'site' (line 72)
        site_39856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 29), 'site')
        # Obtaining the member 'USER_BASE' of a type (line 72)
        USER_BASE_39857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 29), site_39856, 'USER_BASE')
        # Getting the type of 'self' (line 72)
        self_39858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'self')
        # Setting the type of the member 'old_user_base' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), self_39858, 'old_user_base', USER_BASE_39857)
        
        # Assigning a Attribute to a Attribute (line 73):
        
        # Assigning a Attribute to a Attribute (line 73):
        # Getting the type of 'site' (line 73)
        site_39859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 29), 'site')
        # Obtaining the member 'USER_SITE' of a type (line 73)
        USER_SITE_39860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 29), site_39859, 'USER_SITE')
        # Getting the type of 'self' (line 73)
        self_39861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self')
        # Setting the type of the member 'old_user_site' of a type (line 73)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_39861, 'old_user_site', USER_SITE_39860)
        
        # Assigning a Call to a Attribute (line 74):
        
        # Assigning a Call to a Attribute (line 74):
        
        # Call to mkdtemp(...): (line 74)
        # Processing the call keyword arguments (line 74)
        kwargs_39864 = {}
        # Getting the type of 'self' (line 74)
        self_39862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 22), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 74)
        mkdtemp_39863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 22), self_39862, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 74)
        mkdtemp_call_result_39865 = invoke(stypy.reporting.localization.Localization(__file__, 74, 22), mkdtemp_39863, *[], **kwargs_39864)
        
        # Getting the type of 'self' (line 74)
        self_39866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member 'tmpdir' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_39866, 'tmpdir', mkdtemp_call_result_39865)
        
        # Assigning a Call to a Attribute (line 75):
        
        # Assigning a Call to a Attribute (line 75):
        
        # Call to join(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'self' (line 75)
        self_39870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 38), 'self', False)
        # Obtaining the member 'tmpdir' of a type (line 75)
        tmpdir_39871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 38), self_39870, 'tmpdir')
        str_39872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 51), 'str', 'B')
        # Processing the call keyword arguments (line 75)
        kwargs_39873 = {}
        # Getting the type of 'os' (line 75)
        os_39867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 75)
        path_39868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 25), os_39867, 'path')
        # Obtaining the member 'join' of a type (line 75)
        join_39869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 25), path_39868, 'join')
        # Calling join(args, kwargs) (line 75)
        join_call_result_39874 = invoke(stypy.reporting.localization.Localization(__file__, 75, 25), join_39869, *[tmpdir_39871, str_39872], **kwargs_39873)
        
        # Getting the type of 'self' (line 75)
        self_39875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self')
        # Setting the type of the member 'user_base' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_39875, 'user_base', join_call_result_39874)
        
        # Assigning a Call to a Attribute (line 76):
        
        # Assigning a Call to a Attribute (line 76):
        
        # Call to join(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'self' (line 76)
        self_39879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 38), 'self', False)
        # Obtaining the member 'tmpdir' of a type (line 76)
        tmpdir_39880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 38), self_39879, 'tmpdir')
        str_39881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 51), 'str', 'S')
        # Processing the call keyword arguments (line 76)
        kwargs_39882 = {}
        # Getting the type of 'os' (line 76)
        os_39876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 76)
        path_39877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 25), os_39876, 'path')
        # Obtaining the member 'join' of a type (line 76)
        join_39878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 25), path_39877, 'join')
        # Calling join(args, kwargs) (line 76)
        join_call_result_39883 = invoke(stypy.reporting.localization.Localization(__file__, 76, 25), join_39878, *[tmpdir_39880, str_39881], **kwargs_39882)
        
        # Getting the type of 'self' (line 76)
        self_39884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self')
        # Setting the type of the member 'user_site' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_39884, 'user_site', join_call_result_39883)
        
        # Assigning a Attribute to a Attribute (line 77):
        
        # Assigning a Attribute to a Attribute (line 77):
        # Getting the type of 'self' (line 77)
        self_39885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'self')
        # Obtaining the member 'user_base' of a type (line 77)
        user_base_39886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 25), self_39885, 'user_base')
        # Getting the type of 'site' (line 77)
        site_39887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'site')
        # Setting the type of the member 'USER_BASE' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), site_39887, 'USER_BASE', user_base_39886)
        
        # Assigning a Attribute to a Attribute (line 78):
        
        # Assigning a Attribute to a Attribute (line 78):
        # Getting the type of 'self' (line 78)
        self_39888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 25), 'self')
        # Obtaining the member 'user_site' of a type (line 78)
        user_site_39889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 25), self_39888, 'user_site')
        # Getting the type of 'site' (line 78)
        site_39890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'site')
        # Setting the type of the member 'USER_SITE' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), site_39890, 'USER_SITE', user_site_39889)
        
        # Assigning a Attribute to a Attribute (line 79):
        
        # Assigning a Attribute to a Attribute (line 79):
        # Getting the type of 'self' (line 79)
        self_39891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 35), 'self')
        # Obtaining the member 'user_base' of a type (line 79)
        user_base_39892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 35), self_39891, 'user_base')
        # Getting the type of 'install_module' (line 79)
        install_module_39893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'install_module')
        # Setting the type of the member 'USER_BASE' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), install_module_39893, 'USER_BASE', user_base_39892)
        
        # Assigning a Attribute to a Attribute (line 80):
        
        # Assigning a Attribute to a Attribute (line 80):
        # Getting the type of 'self' (line 80)
        self_39894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 35), 'self')
        # Obtaining the member 'user_site' of a type (line 80)
        user_site_39895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 35), self_39894, 'user_site')
        # Getting the type of 'install_module' (line 80)
        install_module_39896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'install_module')
        # Setting the type of the member 'USER_SITE' of a type (line 80)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), install_module_39896, 'USER_SITE', user_site_39895)

        @norecursion
        def _expanduser(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_expanduser'
            module_type_store = module_type_store.open_function_context('_expanduser', 82, 8, False)
            
            # Passed parameters checking function
            _expanduser.stypy_localization = localization
            _expanduser.stypy_type_of_self = None
            _expanduser.stypy_type_store = module_type_store
            _expanduser.stypy_function_name = '_expanduser'
            _expanduser.stypy_param_names_list = ['path']
            _expanduser.stypy_varargs_param_name = None
            _expanduser.stypy_kwargs_param_name = None
            _expanduser.stypy_call_defaults = defaults
            _expanduser.stypy_call_varargs = varargs
            _expanduser.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_expanduser', ['path'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_expanduser', localization, ['path'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_expanduser(...)' code ##################

            # Getting the type of 'self' (line 83)
            self_39897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'self')
            # Obtaining the member 'tmpdir' of a type (line 83)
            tmpdir_39898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 19), self_39897, 'tmpdir')
            # Assigning a type to the variable 'stypy_return_type' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'stypy_return_type', tmpdir_39898)
            
            # ################# End of '_expanduser(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_expanduser' in the type store
            # Getting the type of 'stypy_return_type' (line 82)
            stypy_return_type_39899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_39899)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_expanduser'
            return stypy_return_type_39899

        # Assigning a type to the variable '_expanduser' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), '_expanduser', _expanduser)
        
        # Assigning a Attribute to a Attribute (line 84):
        
        # Assigning a Attribute to a Attribute (line 84):
        # Getting the type of 'os' (line 84)
        os_39900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 26), 'os')
        # Obtaining the member 'path' of a type (line 84)
        path_39901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 26), os_39900, 'path')
        # Obtaining the member 'expanduser' of a type (line 84)
        expanduser_39902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 26), path_39901, 'expanduser')
        # Getting the type of 'self' (line 84)
        self_39903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'self')
        # Setting the type of the member 'old_expand' of a type (line 84)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), self_39903, 'old_expand', expanduser_39902)
        
        # Assigning a Name to a Attribute (line 85):
        
        # Assigning a Name to a Attribute (line 85):
        # Getting the type of '_expanduser' (line 85)
        _expanduser_39904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), '_expanduser')
        # Getting the type of 'os' (line 85)
        os_39905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'os')
        # Obtaining the member 'path' of a type (line 85)
        path_39906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), os_39905, 'path')
        # Setting the type of the member 'expanduser' of a type (line 85)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), path_39906, 'expanduser', _expanduser_39904)

        @norecursion
        def cleanup(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cleanup'
            module_type_store = module_type_store.open_function_context('cleanup', 87, 8, False)
            
            # Passed parameters checking function
            cleanup.stypy_localization = localization
            cleanup.stypy_type_of_self = None
            cleanup.stypy_type_store = module_type_store
            cleanup.stypy_function_name = 'cleanup'
            cleanup.stypy_param_names_list = []
            cleanup.stypy_varargs_param_name = None
            cleanup.stypy_kwargs_param_name = None
            cleanup.stypy_call_defaults = defaults
            cleanup.stypy_call_varargs = varargs
            cleanup.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'cleanup', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'cleanup', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'cleanup(...)' code ##################

            
            # Assigning a Attribute to a Attribute (line 88):
            
            # Assigning a Attribute to a Attribute (line 88):
            # Getting the type of 'self' (line 88)
            self_39907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 29), 'self')
            # Obtaining the member 'old_user_base' of a type (line 88)
            old_user_base_39908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 29), self_39907, 'old_user_base')
            # Getting the type of 'site' (line 88)
            site_39909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'site')
            # Setting the type of the member 'USER_BASE' of a type (line 88)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), site_39909, 'USER_BASE', old_user_base_39908)
            
            # Assigning a Attribute to a Attribute (line 89):
            
            # Assigning a Attribute to a Attribute (line 89):
            # Getting the type of 'self' (line 89)
            self_39910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 29), 'self')
            # Obtaining the member 'old_user_site' of a type (line 89)
            old_user_site_39911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 29), self_39910, 'old_user_site')
            # Getting the type of 'site' (line 89)
            site_39912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'site')
            # Setting the type of the member 'USER_SITE' of a type (line 89)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), site_39912, 'USER_SITE', old_user_site_39911)
            
            # Assigning a Attribute to a Attribute (line 90):
            
            # Assigning a Attribute to a Attribute (line 90):
            # Getting the type of 'self' (line 90)
            self_39913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 39), 'self')
            # Obtaining the member 'old_user_base' of a type (line 90)
            old_user_base_39914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 39), self_39913, 'old_user_base')
            # Getting the type of 'install_module' (line 90)
            install_module_39915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'install_module')
            # Setting the type of the member 'USER_BASE' of a type (line 90)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), install_module_39915, 'USER_BASE', old_user_base_39914)
            
            # Assigning a Attribute to a Attribute (line 91):
            
            # Assigning a Attribute to a Attribute (line 91):
            # Getting the type of 'self' (line 91)
            self_39916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 39), 'self')
            # Obtaining the member 'old_user_site' of a type (line 91)
            old_user_site_39917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 39), self_39916, 'old_user_site')
            # Getting the type of 'install_module' (line 91)
            install_module_39918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'install_module')
            # Setting the type of the member 'USER_SITE' of a type (line 91)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), install_module_39918, 'USER_SITE', old_user_site_39917)
            
            # Assigning a Attribute to a Attribute (line 92):
            
            # Assigning a Attribute to a Attribute (line 92):
            # Getting the type of 'self' (line 92)
            self_39919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 33), 'self')
            # Obtaining the member 'old_expand' of a type (line 92)
            old_expand_39920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 33), self_39919, 'old_expand')
            # Getting the type of 'os' (line 92)
            os_39921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'os')
            # Obtaining the member 'path' of a type (line 92)
            path_39922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), os_39921, 'path')
            # Setting the type of the member 'expanduser' of a type (line 92)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), path_39922, 'expanduser', old_expand_39920)
            
            # ################# End of 'cleanup(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cleanup' in the type store
            # Getting the type of 'stypy_return_type' (line 87)
            stypy_return_type_39923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_39923)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cleanup'
            return stypy_return_type_39923

        # Assigning a type to the variable 'cleanup' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'cleanup', cleanup)
        
        # Call to addCleanup(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'cleanup' (line 94)
        cleanup_39926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 24), 'cleanup', False)
        # Processing the call keyword arguments (line 94)
        kwargs_39927 = {}
        # Getting the type of 'self' (line 94)
        self_39924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self', False)
        # Obtaining the member 'addCleanup' of a type (line 94)
        addCleanup_39925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_39924, 'addCleanup')
        # Calling addCleanup(args, kwargs) (line 94)
        addCleanup_call_result_39928 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), addCleanup_39925, *[cleanup_39926], **kwargs_39927)
        
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 96)
        tuple_39929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 96)
        # Adding element type (line 96)
        str_39930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 20), 'str', 'nt_user')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 20), tuple_39929, str_39930)
        # Adding element type (line 96)
        str_39931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 31), 'str', 'unix_user')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 20), tuple_39929, str_39931)
        # Adding element type (line 96)
        str_39932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 44), 'str', 'os2_home')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 20), tuple_39929, str_39932)
        
        # Testing the type of a for loop iterable (line 96)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 96, 8), tuple_39929)
        # Getting the type of the for loop variable (line 96)
        for_loop_var_39933 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 96, 8), tuple_39929)
        # Assigning a type to the variable 'key' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'key', for_loop_var_39933)
        # SSA begins for a for statement (line 96)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assertIn(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'key' (line 97)
        key_39936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 26), 'key', False)
        # Getting the type of 'INSTALL_SCHEMES' (line 97)
        INSTALL_SCHEMES_39937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 31), 'INSTALL_SCHEMES', False)
        # Processing the call keyword arguments (line 97)
        kwargs_39938 = {}
        # Getting the type of 'self' (line 97)
        self_39934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 97)
        assertIn_39935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 12), self_39934, 'assertIn')
        # Calling assertIn(args, kwargs) (line 97)
        assertIn_call_result_39939 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), assertIn_39935, *[key_39936, INSTALL_SCHEMES_39937], **kwargs_39938)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to Distribution(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Obtaining an instance of the builtin type 'dict' (line 99)
        dict_39941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 99)
        # Adding element type (key, value) (line 99)
        str_39942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 29), 'str', 'name')
        str_39943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 37), 'str', 'xx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 28), dict_39941, (str_39942, str_39943))
        
        # Processing the call keyword arguments (line 99)
        kwargs_39944 = {}
        # Getting the type of 'Distribution' (line 99)
        Distribution_39940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 99)
        Distribution_call_result_39945 = invoke(stypy.reporting.localization.Localization(__file__, 99, 15), Distribution_39940, *[dict_39941], **kwargs_39944)
        
        # Assigning a type to the variable 'dist' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'dist', Distribution_call_result_39945)
        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to install(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'dist' (line 100)
        dist_39947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'dist', False)
        # Processing the call keyword arguments (line 100)
        kwargs_39948 = {}
        # Getting the type of 'install' (line 100)
        install_39946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), 'install', False)
        # Calling install(args, kwargs) (line 100)
        install_call_result_39949 = invoke(stypy.reporting.localization.Localization(__file__, 100, 14), install_39946, *[dist_39947], **kwargs_39948)
        
        # Assigning a type to the variable 'cmd' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'cmd', install_call_result_39949)
        
        # Assigning a ListComp to a Name (line 103):
        
        # Assigning a ListComp to a Name (line 103):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'cmd' (line 104)
        cmd_39951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'cmd')
        # Obtaining the member 'user_options' of a type (line 104)
        user_options_39952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 19), cmd_39951, 'user_options')
        comprehension_39953 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 19), user_options_39952)
        # Assigning a type to the variable 'name' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 19), comprehension_39953))
        # Assigning a type to the variable 'short' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'short', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 19), comprehension_39953))
        # Assigning a type to the variable 'lable' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'lable', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 19), comprehension_39953))
        # Getting the type of 'name' (line 103)
        name_39950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'name')
        list_39954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 19), list_39954, name_39950)
        # Assigning a type to the variable 'options' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'options', list_39954)
        
        # Call to assertIn(...): (line 105)
        # Processing the call arguments (line 105)
        str_39957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 22), 'str', 'user')
        # Getting the type of 'options' (line 105)
        options_39958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 30), 'options', False)
        # Processing the call keyword arguments (line 105)
        kwargs_39959 = {}
        # Getting the type of 'self' (line 105)
        self_39955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 105)
        assertIn_39956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 8), self_39955, 'assertIn')
        # Calling assertIn(args, kwargs) (line 105)
        assertIn_call_result_39960 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), assertIn_39956, *[str_39957, options_39958], **kwargs_39959)
        
        
        # Assigning a Num to a Attribute (line 108):
        
        # Assigning a Num to a Attribute (line 108):
        int_39961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 19), 'int')
        # Getting the type of 'cmd' (line 108)
        cmd_39962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'cmd')
        # Setting the type of the member 'user' of a type (line 108)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), cmd_39962, 'user', int_39961)
        
        # Call to assertFalse(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Call to exists(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'self' (line 111)
        self_39968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 40), 'self', False)
        # Obtaining the member 'user_base' of a type (line 111)
        user_base_39969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 40), self_39968, 'user_base')
        # Processing the call keyword arguments (line 111)
        kwargs_39970 = {}
        # Getting the type of 'os' (line 111)
        os_39965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 111)
        path_39966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 25), os_39965, 'path')
        # Obtaining the member 'exists' of a type (line 111)
        exists_39967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 25), path_39966, 'exists')
        # Calling exists(args, kwargs) (line 111)
        exists_call_result_39971 = invoke(stypy.reporting.localization.Localization(__file__, 111, 25), exists_39967, *[user_base_39969], **kwargs_39970)
        
        # Processing the call keyword arguments (line 111)
        kwargs_39972 = {}
        # Getting the type of 'self' (line 111)
        self_39963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 111)
        assertFalse_39964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_39963, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 111)
        assertFalse_call_result_39973 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), assertFalse_39964, *[exists_call_result_39971], **kwargs_39972)
        
        
        # Call to assertFalse(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Call to exists(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'self' (line 112)
        self_39979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 40), 'self', False)
        # Obtaining the member 'user_site' of a type (line 112)
        user_site_39980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 40), self_39979, 'user_site')
        # Processing the call keyword arguments (line 112)
        kwargs_39981 = {}
        # Getting the type of 'os' (line 112)
        os_39976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 112)
        path_39977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 25), os_39976, 'path')
        # Obtaining the member 'exists' of a type (line 112)
        exists_39978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 25), path_39977, 'exists')
        # Calling exists(args, kwargs) (line 112)
        exists_call_result_39982 = invoke(stypy.reporting.localization.Localization(__file__, 112, 25), exists_39978, *[user_site_39980], **kwargs_39981)
        
        # Processing the call keyword arguments (line 112)
        kwargs_39983 = {}
        # Getting the type of 'self' (line 112)
        self_39974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'self', False)
        # Obtaining the member 'assertFalse' of a type (line 112)
        assertFalse_39975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), self_39974, 'assertFalse')
        # Calling assertFalse(args, kwargs) (line 112)
        assertFalse_call_result_39984 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), assertFalse_39975, *[exists_call_result_39982], **kwargs_39983)
        
        
        # Call to ensure_finalized(...): (line 115)
        # Processing the call keyword arguments (line 115)
        kwargs_39987 = {}
        # Getting the type of 'cmd' (line 115)
        cmd_39985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 115)
        ensure_finalized_39986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 8), cmd_39985, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 115)
        ensure_finalized_call_result_39988 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), ensure_finalized_39986, *[], **kwargs_39987)
        
        
        # Call to assertTrue(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Call to exists(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'self' (line 118)
        self_39994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 39), 'self', False)
        # Obtaining the member 'user_base' of a type (line 118)
        user_base_39995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 39), self_39994, 'user_base')
        # Processing the call keyword arguments (line 118)
        kwargs_39996 = {}
        # Getting the type of 'os' (line 118)
        os_39991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 118)
        path_39992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 24), os_39991, 'path')
        # Obtaining the member 'exists' of a type (line 118)
        exists_39993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 24), path_39992, 'exists')
        # Calling exists(args, kwargs) (line 118)
        exists_call_result_39997 = invoke(stypy.reporting.localization.Localization(__file__, 118, 24), exists_39993, *[user_base_39995], **kwargs_39996)
        
        # Processing the call keyword arguments (line 118)
        kwargs_39998 = {}
        # Getting the type of 'self' (line 118)
        self_39989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 118)
        assertTrue_39990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), self_39989, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 118)
        assertTrue_call_result_39999 = invoke(stypy.reporting.localization.Localization(__file__, 118, 8), assertTrue_39990, *[exists_call_result_39997], **kwargs_39998)
        
        
        # Call to assertTrue(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Call to exists(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'self' (line 119)
        self_40005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 39), 'self', False)
        # Obtaining the member 'user_site' of a type (line 119)
        user_site_40006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 39), self_40005, 'user_site')
        # Processing the call keyword arguments (line 119)
        kwargs_40007 = {}
        # Getting the type of 'os' (line 119)
        os_40002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 119)
        path_40003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 24), os_40002, 'path')
        # Obtaining the member 'exists' of a type (line 119)
        exists_40004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 24), path_40003, 'exists')
        # Calling exists(args, kwargs) (line 119)
        exists_call_result_40008 = invoke(stypy.reporting.localization.Localization(__file__, 119, 24), exists_40004, *[user_site_40006], **kwargs_40007)
        
        # Processing the call keyword arguments (line 119)
        kwargs_40009 = {}
        # Getting the type of 'self' (line 119)
        self_40000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self', False)
        # Obtaining the member 'assertTrue' of a type (line 119)
        assertTrue_40001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_40000, 'assertTrue')
        # Calling assertTrue(args, kwargs) (line 119)
        assertTrue_call_result_40010 = invoke(stypy.reporting.localization.Localization(__file__, 119, 8), assertTrue_40001, *[exists_call_result_40008], **kwargs_40009)
        
        
        # Call to assertIn(...): (line 121)
        # Processing the call arguments (line 121)
        str_40013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 22), 'str', 'userbase')
        # Getting the type of 'cmd' (line 121)
        cmd_40014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 34), 'cmd', False)
        # Obtaining the member 'config_vars' of a type (line 121)
        config_vars_40015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 34), cmd_40014, 'config_vars')
        # Processing the call keyword arguments (line 121)
        kwargs_40016 = {}
        # Getting the type of 'self' (line 121)
        self_40011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 121)
        assertIn_40012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_40011, 'assertIn')
        # Calling assertIn(args, kwargs) (line 121)
        assertIn_call_result_40017 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), assertIn_40012, *[str_40013, config_vars_40015], **kwargs_40016)
        
        
        # Call to assertIn(...): (line 122)
        # Processing the call arguments (line 122)
        str_40020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 22), 'str', 'usersite')
        # Getting the type of 'cmd' (line 122)
        cmd_40021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 34), 'cmd', False)
        # Obtaining the member 'config_vars' of a type (line 122)
        config_vars_40022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 34), cmd_40021, 'config_vars')
        # Processing the call keyword arguments (line 122)
        kwargs_40023 = {}
        # Getting the type of 'self' (line 122)
        self_40018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self', False)
        # Obtaining the member 'assertIn' of a type (line 122)
        assertIn_40019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_40018, 'assertIn')
        # Calling assertIn(args, kwargs) (line 122)
        assertIn_call_result_40024 = invoke(stypy.reporting.localization.Localization(__file__, 122, 8), assertIn_40019, *[str_40020, config_vars_40022], **kwargs_40023)
        
        
        # ################# End of 'test_user_site(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_user_site' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_40025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40025)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_user_site'
        return stypy_return_type_40025


    @norecursion
    def test_handle_extra_path(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_handle_extra_path'
        module_type_store = module_type_store.open_function_context('test_handle_extra_path', 124, 4, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstallTestCase.test_handle_extra_path.__dict__.__setitem__('stypy_localization', localization)
        InstallTestCase.test_handle_extra_path.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallTestCase.test_handle_extra_path.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallTestCase.test_handle_extra_path.__dict__.__setitem__('stypy_function_name', 'InstallTestCase.test_handle_extra_path')
        InstallTestCase.test_handle_extra_path.__dict__.__setitem__('stypy_param_names_list', [])
        InstallTestCase.test_handle_extra_path.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallTestCase.test_handle_extra_path.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallTestCase.test_handle_extra_path.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallTestCase.test_handle_extra_path.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallTestCase.test_handle_extra_path.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallTestCase.test_handle_extra_path.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallTestCase.test_handle_extra_path', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_handle_extra_path', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_handle_extra_path(...)' code ##################

        
        # Assigning a Call to a Name (line 125):
        
        # Assigning a Call to a Name (line 125):
        
        # Call to Distribution(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Obtaining an instance of the builtin type 'dict' (line 125)
        dict_40027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 125)
        # Adding element type (key, value) (line 125)
        str_40028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 29), 'str', 'name')
        str_40029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 37), 'str', 'xx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 28), dict_40027, (str_40028, str_40029))
        # Adding element type (key, value) (line 125)
        str_40030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 43), 'str', 'extra_path')
        str_40031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 57), 'str', 'path,dirs')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 28), dict_40027, (str_40030, str_40031))
        
        # Processing the call keyword arguments (line 125)
        kwargs_40032 = {}
        # Getting the type of 'Distribution' (line 125)
        Distribution_40026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 125)
        Distribution_call_result_40033 = invoke(stypy.reporting.localization.Localization(__file__, 125, 15), Distribution_40026, *[dict_40027], **kwargs_40032)
        
        # Assigning a type to the variable 'dist' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'dist', Distribution_call_result_40033)
        
        # Assigning a Call to a Name (line 126):
        
        # Assigning a Call to a Name (line 126):
        
        # Call to install(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'dist' (line 126)
        dist_40035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'dist', False)
        # Processing the call keyword arguments (line 126)
        kwargs_40036 = {}
        # Getting the type of 'install' (line 126)
        install_40034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 14), 'install', False)
        # Calling install(args, kwargs) (line 126)
        install_call_result_40037 = invoke(stypy.reporting.localization.Localization(__file__, 126, 14), install_40034, *[dist_40035], **kwargs_40036)
        
        # Assigning a type to the variable 'cmd' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'cmd', install_call_result_40037)
        
        # Call to handle_extra_path(...): (line 129)
        # Processing the call keyword arguments (line 129)
        kwargs_40040 = {}
        # Getting the type of 'cmd' (line 129)
        cmd_40038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'cmd', False)
        # Obtaining the member 'handle_extra_path' of a type (line 129)
        handle_extra_path_40039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), cmd_40038, 'handle_extra_path')
        # Calling handle_extra_path(args, kwargs) (line 129)
        handle_extra_path_call_result_40041 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), handle_extra_path_40039, *[], **kwargs_40040)
        
        
        # Call to assertEqual(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'cmd' (line 130)
        cmd_40044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 25), 'cmd', False)
        # Obtaining the member 'extra_path' of a type (line 130)
        extra_path_40045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 25), cmd_40044, 'extra_path')
        
        # Obtaining an instance of the builtin type 'list' (line 130)
        list_40046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 130)
        # Adding element type (line 130)
        str_40047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 42), 'str', 'path')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 41), list_40046, str_40047)
        # Adding element type (line 130)
        str_40048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 50), 'str', 'dirs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 41), list_40046, str_40048)
        
        # Processing the call keyword arguments (line 130)
        kwargs_40049 = {}
        # Getting the type of 'self' (line 130)
        self_40042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 130)
        assertEqual_40043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), self_40042, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 130)
        assertEqual_call_result_40050 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), assertEqual_40043, *[extra_path_40045, list_40046], **kwargs_40049)
        
        
        # Call to assertEqual(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'cmd' (line 131)
        cmd_40053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'cmd', False)
        # Obtaining the member 'extra_dirs' of a type (line 131)
        extra_dirs_40054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 25), cmd_40053, 'extra_dirs')
        str_40055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 41), 'str', 'dirs')
        # Processing the call keyword arguments (line 131)
        kwargs_40056 = {}
        # Getting the type of 'self' (line 131)
        self_40051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 131)
        assertEqual_40052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), self_40051, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 131)
        assertEqual_call_result_40057 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), assertEqual_40052, *[extra_dirs_40054, str_40055], **kwargs_40056)
        
        
        # Call to assertEqual(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'cmd' (line 132)
        cmd_40060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 25), 'cmd', False)
        # Obtaining the member 'path_file' of a type (line 132)
        path_file_40061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 25), cmd_40060, 'path_file')
        str_40062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 40), 'str', 'path')
        # Processing the call keyword arguments (line 132)
        kwargs_40063 = {}
        # Getting the type of 'self' (line 132)
        self_40058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 132)
        assertEqual_40059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), self_40058, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 132)
        assertEqual_call_result_40064 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), assertEqual_40059, *[path_file_40061, str_40062], **kwargs_40063)
        
        
        # Assigning a List to a Attribute (line 135):
        
        # Assigning a List to a Attribute (line 135):
        
        # Obtaining an instance of the builtin type 'list' (line 135)
        list_40065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 135)
        # Adding element type (line 135)
        str_40066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 26), 'str', 'path')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 25), list_40065, str_40066)
        
        # Getting the type of 'cmd' (line 135)
        cmd_40067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'cmd')
        # Setting the type of the member 'extra_path' of a type (line 135)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), cmd_40067, 'extra_path', list_40065)
        
        # Call to handle_extra_path(...): (line 136)
        # Processing the call keyword arguments (line 136)
        kwargs_40070 = {}
        # Getting the type of 'cmd' (line 136)
        cmd_40068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'cmd', False)
        # Obtaining the member 'handle_extra_path' of a type (line 136)
        handle_extra_path_40069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), cmd_40068, 'handle_extra_path')
        # Calling handle_extra_path(args, kwargs) (line 136)
        handle_extra_path_call_result_40071 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), handle_extra_path_40069, *[], **kwargs_40070)
        
        
        # Call to assertEqual(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'cmd' (line 137)
        cmd_40074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 25), 'cmd', False)
        # Obtaining the member 'extra_path' of a type (line 137)
        extra_path_40075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 25), cmd_40074, 'extra_path')
        
        # Obtaining an instance of the builtin type 'list' (line 137)
        list_40076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 41), 'list')
        # Adding type elements to the builtin type 'list' instance (line 137)
        # Adding element type (line 137)
        str_40077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 42), 'str', 'path')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 41), list_40076, str_40077)
        
        # Processing the call keyword arguments (line 137)
        kwargs_40078 = {}
        # Getting the type of 'self' (line 137)
        self_40072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 137)
        assertEqual_40073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), self_40072, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 137)
        assertEqual_call_result_40079 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), assertEqual_40073, *[extra_path_40075, list_40076], **kwargs_40078)
        
        
        # Call to assertEqual(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'cmd' (line 138)
        cmd_40082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 25), 'cmd', False)
        # Obtaining the member 'extra_dirs' of a type (line 138)
        extra_dirs_40083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 25), cmd_40082, 'extra_dirs')
        str_40084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 41), 'str', 'path')
        # Processing the call keyword arguments (line 138)
        kwargs_40085 = {}
        # Getting the type of 'self' (line 138)
        self_40080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 138)
        assertEqual_40081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), self_40080, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 138)
        assertEqual_call_result_40086 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), assertEqual_40081, *[extra_dirs_40083, str_40084], **kwargs_40085)
        
        
        # Call to assertEqual(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'cmd' (line 139)
        cmd_40089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 25), 'cmd', False)
        # Obtaining the member 'path_file' of a type (line 139)
        path_file_40090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 25), cmd_40089, 'path_file')
        str_40091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 40), 'str', 'path')
        # Processing the call keyword arguments (line 139)
        kwargs_40092 = {}
        # Getting the type of 'self' (line 139)
        self_40087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 139)
        assertEqual_40088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), self_40087, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 139)
        assertEqual_call_result_40093 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), assertEqual_40088, *[path_file_40090, str_40091], **kwargs_40092)
        
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Name to a Attribute (line 142):
        # Getting the type of 'None' (line 142)
        None_40094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 43), 'None')
        # Getting the type of 'cmd' (line 142)
        cmd_40095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 26), 'cmd')
        # Setting the type of the member 'extra_path' of a type (line 142)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 26), cmd_40095, 'extra_path', None_40094)
        
        # Assigning a Attribute to a Attribute (line 142):
        # Getting the type of 'cmd' (line 142)
        cmd_40096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 26), 'cmd')
        # Obtaining the member 'extra_path' of a type (line 142)
        extra_path_40097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 26), cmd_40096, 'extra_path')
        # Getting the type of 'dist' (line 142)
        dist_40098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'dist')
        # Setting the type of the member 'extra_path' of a type (line 142)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), dist_40098, 'extra_path', extra_path_40097)
        
        # Call to handle_extra_path(...): (line 143)
        # Processing the call keyword arguments (line 143)
        kwargs_40101 = {}
        # Getting the type of 'cmd' (line 143)
        cmd_40099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'cmd', False)
        # Obtaining the member 'handle_extra_path' of a type (line 143)
        handle_extra_path_40100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), cmd_40099, 'handle_extra_path')
        # Calling handle_extra_path(args, kwargs) (line 143)
        handle_extra_path_call_result_40102 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), handle_extra_path_40100, *[], **kwargs_40101)
        
        
        # Call to assertEqual(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'cmd' (line 144)
        cmd_40105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 25), 'cmd', False)
        # Obtaining the member 'extra_path' of a type (line 144)
        extra_path_40106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 25), cmd_40105, 'extra_path')
        # Getting the type of 'None' (line 144)
        None_40107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 41), 'None', False)
        # Processing the call keyword arguments (line 144)
        kwargs_40108 = {}
        # Getting the type of 'self' (line 144)
        self_40103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 144)
        assertEqual_40104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), self_40103, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 144)
        assertEqual_call_result_40109 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), assertEqual_40104, *[extra_path_40106, None_40107], **kwargs_40108)
        
        
        # Call to assertEqual(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'cmd' (line 145)
        cmd_40112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'cmd', False)
        # Obtaining the member 'extra_dirs' of a type (line 145)
        extra_dirs_40113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 25), cmd_40112, 'extra_dirs')
        str_40114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 41), 'str', '')
        # Processing the call keyword arguments (line 145)
        kwargs_40115 = {}
        # Getting the type of 'self' (line 145)
        self_40110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 145)
        assertEqual_40111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), self_40110, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 145)
        assertEqual_call_result_40116 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), assertEqual_40111, *[extra_dirs_40113, str_40114], **kwargs_40115)
        
        
        # Call to assertEqual(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'cmd' (line 146)
        cmd_40119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), 'cmd', False)
        # Obtaining the member 'path_file' of a type (line 146)
        path_file_40120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 25), cmd_40119, 'path_file')
        # Getting the type of 'None' (line 146)
        None_40121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 40), 'None', False)
        # Processing the call keyword arguments (line 146)
        kwargs_40122 = {}
        # Getting the type of 'self' (line 146)
        self_40117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 146)
        assertEqual_40118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_40117, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 146)
        assertEqual_call_result_40123 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), assertEqual_40118, *[path_file_40120, None_40121], **kwargs_40122)
        
        
        # Assigning a Str to a Attribute (line 149):
        
        # Assigning a Str to a Attribute (line 149):
        str_40124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 25), 'str', 'path,dirs,again')
        # Getting the type of 'cmd' (line 149)
        cmd_40125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'cmd')
        # Setting the type of the member 'extra_path' of a type (line 149)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), cmd_40125, 'extra_path', str_40124)
        
        # Call to assertRaises(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'DistutilsOptionError' (line 150)
        DistutilsOptionError_40128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 26), 'DistutilsOptionError', False)
        # Getting the type of 'cmd' (line 150)
        cmd_40129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 48), 'cmd', False)
        # Obtaining the member 'handle_extra_path' of a type (line 150)
        handle_extra_path_40130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 48), cmd_40129, 'handle_extra_path')
        # Processing the call keyword arguments (line 150)
        kwargs_40131 = {}
        # Getting the type of 'self' (line 150)
        self_40126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 150)
        assertRaises_40127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), self_40126, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 150)
        assertRaises_call_result_40132 = invoke(stypy.reporting.localization.Localization(__file__, 150, 8), assertRaises_40127, *[DistutilsOptionError_40128, handle_extra_path_40130], **kwargs_40131)
        
        
        # ################# End of 'test_handle_extra_path(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_handle_extra_path' in the type store
        # Getting the type of 'stypy_return_type' (line 124)
        stypy_return_type_40133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40133)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_handle_extra_path'
        return stypy_return_type_40133


    @norecursion
    def test_finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_finalize_options'
        module_type_store = module_type_store.open_function_context('test_finalize_options', 152, 4, False)
        # Assigning a type to the variable 'self' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstallTestCase.test_finalize_options.__dict__.__setitem__('stypy_localization', localization)
        InstallTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallTestCase.test_finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallTestCase.test_finalize_options.__dict__.__setitem__('stypy_function_name', 'InstallTestCase.test_finalize_options')
        InstallTestCase.test_finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        InstallTestCase.test_finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallTestCase.test_finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallTestCase.test_finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallTestCase.test_finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallTestCase.test_finalize_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_finalize_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_finalize_options(...)' code ##################

        
        # Assigning a Call to a Name (line 153):
        
        # Assigning a Call to a Name (line 153):
        
        # Call to Distribution(...): (line 153)
        # Processing the call arguments (line 153)
        
        # Obtaining an instance of the builtin type 'dict' (line 153)
        dict_40135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 28), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 153)
        # Adding element type (key, value) (line 153)
        str_40136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 29), 'str', 'name')
        str_40137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 37), 'str', 'xx')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 28), dict_40135, (str_40136, str_40137))
        
        # Processing the call keyword arguments (line 153)
        kwargs_40138 = {}
        # Getting the type of 'Distribution' (line 153)
        Distribution_40134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 153)
        Distribution_call_result_40139 = invoke(stypy.reporting.localization.Localization(__file__, 153, 15), Distribution_40134, *[dict_40135], **kwargs_40138)
        
        # Assigning a type to the variable 'dist' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'dist', Distribution_call_result_40139)
        
        # Assigning a Call to a Name (line 154):
        
        # Assigning a Call to a Name (line 154):
        
        # Call to install(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'dist' (line 154)
        dist_40141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'dist', False)
        # Processing the call keyword arguments (line 154)
        kwargs_40142 = {}
        # Getting the type of 'install' (line 154)
        install_40140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 14), 'install', False)
        # Calling install(args, kwargs) (line 154)
        install_call_result_40143 = invoke(stypy.reporting.localization.Localization(__file__, 154, 14), install_40140, *[dist_40141], **kwargs_40142)
        
        # Assigning a type to the variable 'cmd' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'cmd', install_call_result_40143)
        
        # Assigning a Str to a Attribute (line 158):
        
        # Assigning a Str to a Attribute (line 158):
        str_40144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 21), 'str', 'prefix')
        # Getting the type of 'cmd' (line 158)
        cmd_40145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'cmd')
        # Setting the type of the member 'prefix' of a type (line 158)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), cmd_40145, 'prefix', str_40144)
        
        # Assigning a Str to a Attribute (line 159):
        
        # Assigning a Str to a Attribute (line 159):
        str_40146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 27), 'str', 'base')
        # Getting the type of 'cmd' (line 159)
        cmd_40147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'cmd')
        # Setting the type of the member 'install_base' of a type (line 159)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), cmd_40147, 'install_base', str_40146)
        
        # Call to assertRaises(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'DistutilsOptionError' (line 160)
        DistutilsOptionError_40150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 26), 'DistutilsOptionError', False)
        # Getting the type of 'cmd' (line 160)
        cmd_40151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 48), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 160)
        finalize_options_40152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 48), cmd_40151, 'finalize_options')
        # Processing the call keyword arguments (line 160)
        kwargs_40153 = {}
        # Getting the type of 'self' (line 160)
        self_40148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 160)
        assertRaises_40149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), self_40148, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 160)
        assertRaises_call_result_40154 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), assertRaises_40149, *[DistutilsOptionError_40150, finalize_options_40152], **kwargs_40153)
        
        
        # Assigning a Name to a Attribute (line 163):
        
        # Assigning a Name to a Attribute (line 163):
        # Getting the type of 'None' (line 163)
        None_40155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 27), 'None')
        # Getting the type of 'cmd' (line 163)
        cmd_40156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'cmd')
        # Setting the type of the member 'install_base' of a type (line 163)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), cmd_40156, 'install_base', None_40155)
        
        # Assigning a Str to a Attribute (line 164):
        
        # Assigning a Str to a Attribute (line 164):
        str_40157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 19), 'str', 'home')
        # Getting the type of 'cmd' (line 164)
        cmd_40158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'cmd')
        # Setting the type of the member 'home' of a type (line 164)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), cmd_40158, 'home', str_40157)
        
        # Call to assertRaises(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'DistutilsOptionError' (line 165)
        DistutilsOptionError_40161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 26), 'DistutilsOptionError', False)
        # Getting the type of 'cmd' (line 165)
        cmd_40162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 48), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 165)
        finalize_options_40163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 48), cmd_40162, 'finalize_options')
        # Processing the call keyword arguments (line 165)
        kwargs_40164 = {}
        # Getting the type of 'self' (line 165)
        self_40159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 165)
        assertRaises_40160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), self_40159, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 165)
        assertRaises_call_result_40165 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), assertRaises_40160, *[DistutilsOptionError_40161, finalize_options_40163], **kwargs_40164)
        
        
        # Assigning a Name to a Attribute (line 169):
        
        # Assigning a Name to a Attribute (line 169):
        # Getting the type of 'None' (line 169)
        None_40166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 21), 'None')
        # Getting the type of 'cmd' (line 169)
        cmd_40167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'cmd')
        # Setting the type of the member 'prefix' of a type (line 169)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), cmd_40167, 'prefix', None_40166)
        
        # Assigning a Str to a Attribute (line 170):
        
        # Assigning a Str to a Attribute (line 170):
        str_40168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 19), 'str', 'user')
        # Getting the type of 'cmd' (line 170)
        cmd_40169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'cmd')
        # Setting the type of the member 'user' of a type (line 170)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), cmd_40169, 'user', str_40168)
        
        # Call to assertRaises(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'DistutilsOptionError' (line 171)
        DistutilsOptionError_40172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 26), 'DistutilsOptionError', False)
        # Getting the type of 'cmd' (line 171)
        cmd_40173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 48), 'cmd', False)
        # Obtaining the member 'finalize_options' of a type (line 171)
        finalize_options_40174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 48), cmd_40173, 'finalize_options')
        # Processing the call keyword arguments (line 171)
        kwargs_40175 = {}
        # Getting the type of 'self' (line 171)
        self_40170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'self', False)
        # Obtaining the member 'assertRaises' of a type (line 171)
        assertRaises_40171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), self_40170, 'assertRaises')
        # Calling assertRaises(args, kwargs) (line 171)
        assertRaises_call_result_40176 = invoke(stypy.reporting.localization.Localization(__file__, 171, 8), assertRaises_40171, *[DistutilsOptionError_40172, finalize_options_40174], **kwargs_40175)
        
        
        # ################# End of 'test_finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 152)
        stypy_return_type_40177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40177)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_finalize_options'
        return stypy_return_type_40177


    @norecursion
    def test_record(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_record'
        module_type_store = module_type_store.open_function_context('test_record', 173, 4, False)
        # Assigning a type to the variable 'self' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstallTestCase.test_record.__dict__.__setitem__('stypy_localization', localization)
        InstallTestCase.test_record.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallTestCase.test_record.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallTestCase.test_record.__dict__.__setitem__('stypy_function_name', 'InstallTestCase.test_record')
        InstallTestCase.test_record.__dict__.__setitem__('stypy_param_names_list', [])
        InstallTestCase.test_record.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallTestCase.test_record.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallTestCase.test_record.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallTestCase.test_record.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallTestCase.test_record.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallTestCase.test_record.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallTestCase.test_record', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_record', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_record(...)' code ##################

        
        # Assigning a Call to a Name (line 174):
        
        # Assigning a Call to a Name (line 174):
        
        # Call to mkdtemp(...): (line 174)
        # Processing the call keyword arguments (line 174)
        kwargs_40180 = {}
        # Getting the type of 'self' (line 174)
        self_40178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 22), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 174)
        mkdtemp_40179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 22), self_40178, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 174)
        mkdtemp_call_result_40181 = invoke(stypy.reporting.localization.Localization(__file__, 174, 22), mkdtemp_40179, *[], **kwargs_40180)
        
        # Assigning a type to the variable 'install_dir' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'install_dir', mkdtemp_call_result_40181)
        
        # Assigning a Call to a Tuple (line 175):
        
        # Assigning a Subscript to a Name (line 175):
        
        # Obtaining the type of the subscript
        int_40182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 8), 'int')
        
        # Call to create_dist(...): (line 175)
        # Processing the call keyword arguments (line 175)
        
        # Obtaining an instance of the builtin type 'list' (line 175)
        list_40185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 175)
        # Adding element type (line 175)
        str_40186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 57), 'str', 'hello')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 56), list_40185, str_40186)
        
        keyword_40187 = list_40185
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_40188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        str_40189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 54), 'str', 'sayhi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 53), list_40188, str_40189)
        
        keyword_40190 = list_40188
        kwargs_40191 = {'py_modules': keyword_40187, 'scripts': keyword_40190}
        # Getting the type of 'self' (line 175)
        self_40183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 28), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 175)
        create_dist_40184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 28), self_40183, 'create_dist')
        # Calling create_dist(args, kwargs) (line 175)
        create_dist_call_result_40192 = invoke(stypy.reporting.localization.Localization(__file__, 175, 28), create_dist_40184, *[], **kwargs_40191)
        
        # Obtaining the member '__getitem__' of a type (line 175)
        getitem___40193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), create_dist_call_result_40192, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 175)
        subscript_call_result_40194 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), getitem___40193, int_40182)
        
        # Assigning a type to the variable 'tuple_var_assignment_39660' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'tuple_var_assignment_39660', subscript_call_result_40194)
        
        # Assigning a Subscript to a Name (line 175):
        
        # Obtaining the type of the subscript
        int_40195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 8), 'int')
        
        # Call to create_dist(...): (line 175)
        # Processing the call keyword arguments (line 175)
        
        # Obtaining an instance of the builtin type 'list' (line 175)
        list_40198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 175)
        # Adding element type (line 175)
        str_40199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 57), 'str', 'hello')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 56), list_40198, str_40199)
        
        keyword_40200 = list_40198
        
        # Obtaining an instance of the builtin type 'list' (line 176)
        list_40201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 53), 'list')
        # Adding type elements to the builtin type 'list' instance (line 176)
        # Adding element type (line 176)
        str_40202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 54), 'str', 'sayhi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 53), list_40201, str_40202)
        
        keyword_40203 = list_40201
        kwargs_40204 = {'py_modules': keyword_40200, 'scripts': keyword_40203}
        # Getting the type of 'self' (line 175)
        self_40196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 28), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 175)
        create_dist_40197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 28), self_40196, 'create_dist')
        # Calling create_dist(args, kwargs) (line 175)
        create_dist_call_result_40205 = invoke(stypy.reporting.localization.Localization(__file__, 175, 28), create_dist_40197, *[], **kwargs_40204)
        
        # Obtaining the member '__getitem__' of a type (line 175)
        getitem___40206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), create_dist_call_result_40205, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 175)
        subscript_call_result_40207 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), getitem___40206, int_40195)
        
        # Assigning a type to the variable 'tuple_var_assignment_39661' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'tuple_var_assignment_39661', subscript_call_result_40207)
        
        # Assigning a Name to a Name (line 175):
        # Getting the type of 'tuple_var_assignment_39660' (line 175)
        tuple_var_assignment_39660_40208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'tuple_var_assignment_39660')
        # Assigning a type to the variable 'project_dir' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'project_dir', tuple_var_assignment_39660_40208)
        
        # Assigning a Name to a Name (line 175):
        # Getting the type of 'tuple_var_assignment_39661' (line 175)
        tuple_var_assignment_39661_40209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'tuple_var_assignment_39661')
        # Assigning a type to the variable 'dist' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 21), 'dist', tuple_var_assignment_39661_40209)
        
        # Call to chdir(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'project_dir' (line 177)
        project_dir_40212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 17), 'project_dir', False)
        # Processing the call keyword arguments (line 177)
        kwargs_40213 = {}
        # Getting the type of 'os' (line 177)
        os_40210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 177)
        chdir_40211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), os_40210, 'chdir')
        # Calling chdir(args, kwargs) (line 177)
        chdir_call_result_40214 = invoke(stypy.reporting.localization.Localization(__file__, 177, 8), chdir_40211, *[project_dir_40212], **kwargs_40213)
        
        
        # Call to write_file(...): (line 178)
        # Processing the call arguments (line 178)
        str_40217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 24), 'str', 'hello.py')
        str_40218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 36), 'str', "def main(): print 'o hai'")
        # Processing the call keyword arguments (line 178)
        kwargs_40219 = {}
        # Getting the type of 'self' (line 178)
        self_40215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 178)
        write_file_40216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), self_40215, 'write_file')
        # Calling write_file(args, kwargs) (line 178)
        write_file_call_result_40220 = invoke(stypy.reporting.localization.Localization(__file__, 178, 8), write_file_40216, *[str_40217, str_40218], **kwargs_40219)
        
        
        # Call to write_file(...): (line 179)
        # Processing the call arguments (line 179)
        str_40223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 24), 'str', 'sayhi')
        str_40224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 33), 'str', 'from hello import main; main()')
        # Processing the call keyword arguments (line 179)
        kwargs_40225 = {}
        # Getting the type of 'self' (line 179)
        self_40221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'self', False)
        # Obtaining the member 'write_file' of a type (line 179)
        write_file_40222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), self_40221, 'write_file')
        # Calling write_file(args, kwargs) (line 179)
        write_file_call_result_40226 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), write_file_40222, *[str_40223, str_40224], **kwargs_40225)
        
        
        # Assigning a Call to a Name (line 181):
        
        # Assigning a Call to a Name (line 181):
        
        # Call to install(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'dist' (line 181)
        dist_40228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 22), 'dist', False)
        # Processing the call keyword arguments (line 181)
        kwargs_40229 = {}
        # Getting the type of 'install' (line 181)
        install_40227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 14), 'install', False)
        # Calling install(args, kwargs) (line 181)
        install_call_result_40230 = invoke(stypy.reporting.localization.Localization(__file__, 181, 14), install_40227, *[dist_40228], **kwargs_40229)
        
        # Assigning a type to the variable 'cmd' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'cmd', install_call_result_40230)
        
        # Assigning a Name to a Subscript (line 182):
        
        # Assigning a Name to a Subscript (line 182):
        # Getting the type of 'cmd' (line 182)
        cmd_40231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 38), 'cmd')
        # Getting the type of 'dist' (line 182)
        dist_40232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'dist')
        # Obtaining the member 'command_obj' of a type (line 182)
        command_obj_40233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), dist_40232, 'command_obj')
        str_40234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 25), 'str', 'install')
        # Storing an element on a container (line 182)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 8), command_obj_40233, (str_40234, cmd_40231))
        
        # Assigning a Name to a Attribute (line 183):
        
        # Assigning a Name to a Attribute (line 183):
        # Getting the type of 'install_dir' (line 183)
        install_dir_40235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 19), 'install_dir')
        # Getting the type of 'cmd' (line 183)
        cmd_40236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'cmd')
        # Setting the type of the member 'root' of a type (line 183)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), cmd_40236, 'root', install_dir_40235)
        
        # Assigning a Call to a Attribute (line 184):
        
        # Assigning a Call to a Attribute (line 184):
        
        # Call to join(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'project_dir' (line 184)
        project_dir_40240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 34), 'project_dir', False)
        str_40241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 47), 'str', 'filelist')
        # Processing the call keyword arguments (line 184)
        kwargs_40242 = {}
        # Getting the type of 'os' (line 184)
        os_40237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 184)
        path_40238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 21), os_40237, 'path')
        # Obtaining the member 'join' of a type (line 184)
        join_40239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 21), path_40238, 'join')
        # Calling join(args, kwargs) (line 184)
        join_call_result_40243 = invoke(stypy.reporting.localization.Localization(__file__, 184, 21), join_40239, *[project_dir_40240, str_40241], **kwargs_40242)
        
        # Getting the type of 'cmd' (line 184)
        cmd_40244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'cmd')
        # Setting the type of the member 'record' of a type (line 184)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), cmd_40244, 'record', join_call_result_40243)
        
        # Call to ensure_finalized(...): (line 185)
        # Processing the call keyword arguments (line 185)
        kwargs_40247 = {}
        # Getting the type of 'cmd' (line 185)
        cmd_40245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 185)
        ensure_finalized_40246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), cmd_40245, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 185)
        ensure_finalized_call_result_40248 = invoke(stypy.reporting.localization.Localization(__file__, 185, 8), ensure_finalized_40246, *[], **kwargs_40247)
        
        
        # Call to run(...): (line 186)
        # Processing the call keyword arguments (line 186)
        kwargs_40251 = {}
        # Getting the type of 'cmd' (line 186)
        cmd_40249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 186)
        run_40250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), cmd_40249, 'run')
        # Calling run(args, kwargs) (line 186)
        run_call_result_40252 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), run_40250, *[], **kwargs_40251)
        
        
        # Assigning a Call to a Name (line 188):
        
        # Assigning a Call to a Name (line 188):
        
        # Call to open(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'cmd' (line 188)
        cmd_40254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 17), 'cmd', False)
        # Obtaining the member 'record' of a type (line 188)
        record_40255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 17), cmd_40254, 'record')
        # Processing the call keyword arguments (line 188)
        kwargs_40256 = {}
        # Getting the type of 'open' (line 188)
        open_40253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'open', False)
        # Calling open(args, kwargs) (line 188)
        open_call_result_40257 = invoke(stypy.reporting.localization.Localization(__file__, 188, 12), open_40253, *[record_40255], **kwargs_40256)
        
        # Assigning a type to the variable 'f' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'f', open_call_result_40257)
        
        # Try-finally block (line 189)
        
        # Assigning a Call to a Name (line 190):
        
        # Assigning a Call to a Name (line 190):
        
        # Call to read(...): (line 190)
        # Processing the call keyword arguments (line 190)
        kwargs_40260 = {}
        # Getting the type of 'f' (line 190)
        f_40258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 22), 'f', False)
        # Obtaining the member 'read' of a type (line 190)
        read_40259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 22), f_40258, 'read')
        # Calling read(args, kwargs) (line 190)
        read_call_result_40261 = invoke(stypy.reporting.localization.Localization(__file__, 190, 22), read_40259, *[], **kwargs_40260)
        
        # Assigning a type to the variable 'content' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'content', read_call_result_40261)
        
        # finally branch of the try-finally block (line 189)
        
        # Call to close(...): (line 192)
        # Processing the call keyword arguments (line 192)
        kwargs_40264 = {}
        # Getting the type of 'f' (line 192)
        f_40262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 192)
        close_40263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), f_40262, 'close')
        # Calling close(args, kwargs) (line 192)
        close_call_result_40265 = invoke(stypy.reporting.localization.Localization(__file__, 192, 12), close_40263, *[], **kwargs_40264)
        
        
        
        # Assigning a ListComp to a Name (line 194):
        
        # Assigning a ListComp to a Name (line 194):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to splitlines(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_40274 = {}
        # Getting the type of 'content' (line 194)
        content_40272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 52), 'content', False)
        # Obtaining the member 'splitlines' of a type (line 194)
        splitlines_40273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 52), content_40272, 'splitlines')
        # Calling splitlines(args, kwargs) (line 194)
        splitlines_call_result_40275 = invoke(stypy.reporting.localization.Localization(__file__, 194, 52), splitlines_40273, *[], **kwargs_40274)
        
        comprehension_40276 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 17), splitlines_call_result_40275)
        # Assigning a type to the variable 'line' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'line', comprehension_40276)
        
        # Call to basename(...): (line 194)
        # Processing the call arguments (line 194)
        # Getting the type of 'line' (line 194)
        line_40269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 34), 'line', False)
        # Processing the call keyword arguments (line 194)
        kwargs_40270 = {}
        # Getting the type of 'os' (line 194)
        os_40266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 194)
        path_40267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 17), os_40266, 'path')
        # Obtaining the member 'basename' of a type (line 194)
        basename_40268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 17), path_40267, 'basename')
        # Calling basename(args, kwargs) (line 194)
        basename_call_result_40271 = invoke(stypy.reporting.localization.Localization(__file__, 194, 17), basename_40268, *[line_40269], **kwargs_40270)
        
        list_40277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 17), list_40277, basename_call_result_40271)
        # Assigning a type to the variable 'found' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'found', list_40277)
        
        # Assigning a List to a Name (line 195):
        
        # Assigning a List to a Name (line 195):
        
        # Obtaining an instance of the builtin type 'list' (line 195)
        list_40278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 195)
        # Adding element type (line 195)
        str_40279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 20), 'str', 'hello.py')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 19), list_40278, str_40279)
        # Adding element type (line 195)
        str_40280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 32), 'str', 'hello.pyc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 19), list_40278, str_40280)
        # Adding element type (line 195)
        str_40281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 45), 'str', 'sayhi')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 19), list_40278, str_40281)
        # Adding element type (line 195)
        str_40282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 20), 'str', 'UNKNOWN-0.0.0-py%s.%s.egg-info')
        
        # Obtaining the type of the subscript
        int_40283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 73), 'int')
        slice_40284 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 196, 55), None, int_40283, None)
        # Getting the type of 'sys' (line 196)
        sys_40285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 55), 'sys')
        # Obtaining the member 'version_info' of a type (line 196)
        version_info_40286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 55), sys_40285, 'version_info')
        # Obtaining the member '__getitem__' of a type (line 196)
        getitem___40287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 55), version_info_40286, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 196)
        subscript_call_result_40288 = invoke(stypy.reporting.localization.Localization(__file__, 196, 55), getitem___40287, slice_40284)
        
        # Applying the binary operator '%' (line 196)
        result_mod_40289 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 20), '%', str_40282, subscript_call_result_40288)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 19), list_40278, result_mod_40289)
        
        # Assigning a type to the variable 'expected' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'expected', list_40278)
        
        # Call to assertEqual(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'found' (line 197)
        found_40292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 25), 'found', False)
        # Getting the type of 'expected' (line 197)
        expected_40293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 32), 'expected', False)
        # Processing the call keyword arguments (line 197)
        kwargs_40294 = {}
        # Getting the type of 'self' (line 197)
        self_40290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 197)
        assertEqual_40291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), self_40290, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 197)
        assertEqual_call_result_40295 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), assertEqual_40291, *[found_40292, expected_40293], **kwargs_40294)
        
        
        # ################# End of 'test_record(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_record' in the type store
        # Getting the type of 'stypy_return_type' (line 173)
        stypy_return_type_40296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40296)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_record'
        return stypy_return_type_40296


    @norecursion
    def test_record_extensions(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_record_extensions'
        module_type_store = module_type_store.open_function_context('test_record_extensions', 199, 4, False)
        # Assigning a type to the variable 'self' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstallTestCase.test_record_extensions.__dict__.__setitem__('stypy_localization', localization)
        InstallTestCase.test_record_extensions.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallTestCase.test_record_extensions.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallTestCase.test_record_extensions.__dict__.__setitem__('stypy_function_name', 'InstallTestCase.test_record_extensions')
        InstallTestCase.test_record_extensions.__dict__.__setitem__('stypy_param_names_list', [])
        InstallTestCase.test_record_extensions.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallTestCase.test_record_extensions.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallTestCase.test_record_extensions.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallTestCase.test_record_extensions.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallTestCase.test_record_extensions.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallTestCase.test_record_extensions.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallTestCase.test_record_extensions', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_record_extensions', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_record_extensions(...)' code ##################

        
        # Assigning a Call to a Name (line 200):
        
        # Assigning a Call to a Name (line 200):
        
        # Call to mkdtemp(...): (line 200)
        # Processing the call keyword arguments (line 200)
        kwargs_40299 = {}
        # Getting the type of 'self' (line 200)
        self_40297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 200)
        mkdtemp_40298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 22), self_40297, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 200)
        mkdtemp_call_result_40300 = invoke(stypy.reporting.localization.Localization(__file__, 200, 22), mkdtemp_40298, *[], **kwargs_40299)
        
        # Assigning a type to the variable 'install_dir' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'install_dir', mkdtemp_call_result_40300)
        
        # Assigning a Call to a Tuple (line 201):
        
        # Assigning a Subscript to a Name (line 201):
        
        # Obtaining the type of the subscript
        int_40301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 8), 'int')
        
        # Call to create_dist(...): (line 201)
        # Processing the call keyword arguments (line 201)
        
        # Obtaining an instance of the builtin type 'list' (line 201)
        list_40304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 201)
        # Adding element type (line 201)
        
        # Call to Extension(...): (line 202)
        # Processing the call arguments (line 202)
        str_40306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 22), 'str', 'xx')
        
        # Obtaining an instance of the builtin type 'list' (line 202)
        list_40307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 202)
        # Adding element type (line 202)
        str_40308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 29), 'str', 'xxmodule.c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 28), list_40307, str_40308)
        
        # Processing the call keyword arguments (line 202)
        kwargs_40309 = {}
        # Getting the type of 'Extension' (line 202)
        Extension_40305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'Extension', False)
        # Calling Extension(args, kwargs) (line 202)
        Extension_call_result_40310 = invoke(stypy.reporting.localization.Localization(__file__, 202, 12), Extension_40305, *[str_40306, list_40307], **kwargs_40309)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 57), list_40304, Extension_call_result_40310)
        
        keyword_40311 = list_40304
        kwargs_40312 = {'ext_modules': keyword_40311}
        # Getting the type of 'self' (line 201)
        self_40302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 28), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 201)
        create_dist_40303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 28), self_40302, 'create_dist')
        # Calling create_dist(args, kwargs) (line 201)
        create_dist_call_result_40313 = invoke(stypy.reporting.localization.Localization(__file__, 201, 28), create_dist_40303, *[], **kwargs_40312)
        
        # Obtaining the member '__getitem__' of a type (line 201)
        getitem___40314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), create_dist_call_result_40313, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 201)
        subscript_call_result_40315 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), getitem___40314, int_40301)
        
        # Assigning a type to the variable 'tuple_var_assignment_39662' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'tuple_var_assignment_39662', subscript_call_result_40315)
        
        # Assigning a Subscript to a Name (line 201):
        
        # Obtaining the type of the subscript
        int_40316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 8), 'int')
        
        # Call to create_dist(...): (line 201)
        # Processing the call keyword arguments (line 201)
        
        # Obtaining an instance of the builtin type 'list' (line 201)
        list_40319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 201)
        # Adding element type (line 201)
        
        # Call to Extension(...): (line 202)
        # Processing the call arguments (line 202)
        str_40321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 22), 'str', 'xx')
        
        # Obtaining an instance of the builtin type 'list' (line 202)
        list_40322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 202)
        # Adding element type (line 202)
        str_40323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 29), 'str', 'xxmodule.c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 28), list_40322, str_40323)
        
        # Processing the call keyword arguments (line 202)
        kwargs_40324 = {}
        # Getting the type of 'Extension' (line 202)
        Extension_40320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'Extension', False)
        # Calling Extension(args, kwargs) (line 202)
        Extension_call_result_40325 = invoke(stypy.reporting.localization.Localization(__file__, 202, 12), Extension_40320, *[str_40321, list_40322], **kwargs_40324)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 57), list_40319, Extension_call_result_40325)
        
        keyword_40326 = list_40319
        kwargs_40327 = {'ext_modules': keyword_40326}
        # Getting the type of 'self' (line 201)
        self_40317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 28), 'self', False)
        # Obtaining the member 'create_dist' of a type (line 201)
        create_dist_40318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 28), self_40317, 'create_dist')
        # Calling create_dist(args, kwargs) (line 201)
        create_dist_call_result_40328 = invoke(stypy.reporting.localization.Localization(__file__, 201, 28), create_dist_40318, *[], **kwargs_40327)
        
        # Obtaining the member '__getitem__' of a type (line 201)
        getitem___40329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 8), create_dist_call_result_40328, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 201)
        subscript_call_result_40330 = invoke(stypy.reporting.localization.Localization(__file__, 201, 8), getitem___40329, int_40316)
        
        # Assigning a type to the variable 'tuple_var_assignment_39663' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'tuple_var_assignment_39663', subscript_call_result_40330)
        
        # Assigning a Name to a Name (line 201):
        # Getting the type of 'tuple_var_assignment_39662' (line 201)
        tuple_var_assignment_39662_40331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'tuple_var_assignment_39662')
        # Assigning a type to the variable 'project_dir' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'project_dir', tuple_var_assignment_39662_40331)
        
        # Assigning a Name to a Name (line 201):
        # Getting the type of 'tuple_var_assignment_39663' (line 201)
        tuple_var_assignment_39663_40332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'tuple_var_assignment_39663')
        # Assigning a type to the variable 'dist' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 21), 'dist', tuple_var_assignment_39663_40332)
        
        # Call to chdir(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'project_dir' (line 203)
        project_dir_40335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 17), 'project_dir', False)
        # Processing the call keyword arguments (line 203)
        kwargs_40336 = {}
        # Getting the type of 'os' (line 203)
        os_40333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 203)
        chdir_40334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), os_40333, 'chdir')
        # Calling chdir(args, kwargs) (line 203)
        chdir_call_result_40337 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), chdir_40334, *[project_dir_40335], **kwargs_40336)
        
        
        # Call to copy_xxmodule_c(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'project_dir' (line 204)
        project_dir_40340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 32), 'project_dir', False)
        # Processing the call keyword arguments (line 204)
        kwargs_40341 = {}
        # Getting the type of 'support' (line 204)
        support_40338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'support', False)
        # Obtaining the member 'copy_xxmodule_c' of a type (line 204)
        copy_xxmodule_c_40339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), support_40338, 'copy_xxmodule_c')
        # Calling copy_xxmodule_c(args, kwargs) (line 204)
        copy_xxmodule_c_call_result_40342 = invoke(stypy.reporting.localization.Localization(__file__, 204, 8), copy_xxmodule_c_40339, *[project_dir_40340], **kwargs_40341)
        
        
        # Assigning a Call to a Name (line 206):
        
        # Assigning a Call to a Name (line 206):
        
        # Call to build_ext(...): (line 206)
        # Processing the call arguments (line 206)
        # Getting the type of 'dist' (line 206)
        dist_40344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 32), 'dist', False)
        # Processing the call keyword arguments (line 206)
        kwargs_40345 = {}
        # Getting the type of 'build_ext' (line 206)
        build_ext_40343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 22), 'build_ext', False)
        # Calling build_ext(args, kwargs) (line 206)
        build_ext_call_result_40346 = invoke(stypy.reporting.localization.Localization(__file__, 206, 22), build_ext_40343, *[dist_40344], **kwargs_40345)
        
        # Assigning a type to the variable 'buildextcmd' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'buildextcmd', build_ext_call_result_40346)
        
        # Call to fixup_build_ext(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'buildextcmd' (line 207)
        buildextcmd_40349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 32), 'buildextcmd', False)
        # Processing the call keyword arguments (line 207)
        kwargs_40350 = {}
        # Getting the type of 'support' (line 207)
        support_40347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'support', False)
        # Obtaining the member 'fixup_build_ext' of a type (line 207)
        fixup_build_ext_40348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), support_40347, 'fixup_build_ext')
        # Calling fixup_build_ext(args, kwargs) (line 207)
        fixup_build_ext_call_result_40351 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), fixup_build_ext_40348, *[buildextcmd_40349], **kwargs_40350)
        
        
        # Call to ensure_finalized(...): (line 208)
        # Processing the call keyword arguments (line 208)
        kwargs_40354 = {}
        # Getting the type of 'buildextcmd' (line 208)
        buildextcmd_40352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'buildextcmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 208)
        ensure_finalized_40353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), buildextcmd_40352, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 208)
        ensure_finalized_call_result_40355 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), ensure_finalized_40353, *[], **kwargs_40354)
        
        
        # Assigning a Call to a Name (line 210):
        
        # Assigning a Call to a Name (line 210):
        
        # Call to install(...): (line 210)
        # Processing the call arguments (line 210)
        # Getting the type of 'dist' (line 210)
        dist_40357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 22), 'dist', False)
        # Processing the call keyword arguments (line 210)
        kwargs_40358 = {}
        # Getting the type of 'install' (line 210)
        install_40356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 14), 'install', False)
        # Calling install(args, kwargs) (line 210)
        install_call_result_40359 = invoke(stypy.reporting.localization.Localization(__file__, 210, 14), install_40356, *[dist_40357], **kwargs_40358)
        
        # Assigning a type to the variable 'cmd' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'cmd', install_call_result_40359)
        
        # Assigning a Name to a Subscript (line 211):
        
        # Assigning a Name to a Subscript (line 211):
        # Getting the type of 'cmd' (line 211)
        cmd_40360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 38), 'cmd')
        # Getting the type of 'dist' (line 211)
        dist_40361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'dist')
        # Obtaining the member 'command_obj' of a type (line 211)
        command_obj_40362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), dist_40361, 'command_obj')
        str_40363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 25), 'str', 'install')
        # Storing an element on a container (line 211)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 8), command_obj_40362, (str_40363, cmd_40360))
        
        # Assigning a Name to a Subscript (line 212):
        
        # Assigning a Name to a Subscript (line 212):
        # Getting the type of 'buildextcmd' (line 212)
        buildextcmd_40364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 40), 'buildextcmd')
        # Getting the type of 'dist' (line 212)
        dist_40365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'dist')
        # Obtaining the member 'command_obj' of a type (line 212)
        command_obj_40366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), dist_40365, 'command_obj')
        str_40367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 25), 'str', 'build_ext')
        # Storing an element on a container (line 212)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 8), command_obj_40366, (str_40367, buildextcmd_40364))
        
        # Assigning a Name to a Attribute (line 213):
        
        # Assigning a Name to a Attribute (line 213):
        # Getting the type of 'install_dir' (line 213)
        install_dir_40368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'install_dir')
        # Getting the type of 'cmd' (line 213)
        cmd_40369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'cmd')
        # Setting the type of the member 'root' of a type (line 213)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), cmd_40369, 'root', install_dir_40368)
        
        # Assigning a Call to a Attribute (line 214):
        
        # Assigning a Call to a Attribute (line 214):
        
        # Call to join(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'project_dir' (line 214)
        project_dir_40373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 34), 'project_dir', False)
        str_40374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 47), 'str', 'filelist')
        # Processing the call keyword arguments (line 214)
        kwargs_40375 = {}
        # Getting the type of 'os' (line 214)
        os_40370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 214)
        path_40371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 21), os_40370, 'path')
        # Obtaining the member 'join' of a type (line 214)
        join_40372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 21), path_40371, 'join')
        # Calling join(args, kwargs) (line 214)
        join_call_result_40376 = invoke(stypy.reporting.localization.Localization(__file__, 214, 21), join_40372, *[project_dir_40373, str_40374], **kwargs_40375)
        
        # Getting the type of 'cmd' (line 214)
        cmd_40377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'cmd')
        # Setting the type of the member 'record' of a type (line 214)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), cmd_40377, 'record', join_call_result_40376)
        
        # Call to ensure_finalized(...): (line 215)
        # Processing the call keyword arguments (line 215)
        kwargs_40380 = {}
        # Getting the type of 'cmd' (line 215)
        cmd_40378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'cmd', False)
        # Obtaining the member 'ensure_finalized' of a type (line 215)
        ensure_finalized_40379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), cmd_40378, 'ensure_finalized')
        # Calling ensure_finalized(args, kwargs) (line 215)
        ensure_finalized_call_result_40381 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), ensure_finalized_40379, *[], **kwargs_40380)
        
        
        # Call to run(...): (line 216)
        # Processing the call keyword arguments (line 216)
        kwargs_40384 = {}
        # Getting the type of 'cmd' (line 216)
        cmd_40382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'cmd', False)
        # Obtaining the member 'run' of a type (line 216)
        run_40383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), cmd_40382, 'run')
        # Calling run(args, kwargs) (line 216)
        run_call_result_40385 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), run_40383, *[], **kwargs_40384)
        
        
        # Assigning a Call to a Name (line 218):
        
        # Assigning a Call to a Name (line 218):
        
        # Call to open(...): (line 218)
        # Processing the call arguments (line 218)
        # Getting the type of 'cmd' (line 218)
        cmd_40387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 17), 'cmd', False)
        # Obtaining the member 'record' of a type (line 218)
        record_40388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 17), cmd_40387, 'record')
        # Processing the call keyword arguments (line 218)
        kwargs_40389 = {}
        # Getting the type of 'open' (line 218)
        open_40386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'open', False)
        # Calling open(args, kwargs) (line 218)
        open_call_result_40390 = invoke(stypy.reporting.localization.Localization(__file__, 218, 12), open_40386, *[record_40388], **kwargs_40389)
        
        # Assigning a type to the variable 'f' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'f', open_call_result_40390)
        
        # Try-finally block (line 219)
        
        # Assigning a Call to a Name (line 220):
        
        # Assigning a Call to a Name (line 220):
        
        # Call to read(...): (line 220)
        # Processing the call keyword arguments (line 220)
        kwargs_40393 = {}
        # Getting the type of 'f' (line 220)
        f_40391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 22), 'f', False)
        # Obtaining the member 'read' of a type (line 220)
        read_40392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 22), f_40391, 'read')
        # Calling read(args, kwargs) (line 220)
        read_call_result_40394 = invoke(stypy.reporting.localization.Localization(__file__, 220, 22), read_40392, *[], **kwargs_40393)
        
        # Assigning a type to the variable 'content' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'content', read_call_result_40394)
        
        # finally branch of the try-finally block (line 219)
        
        # Call to close(...): (line 222)
        # Processing the call keyword arguments (line 222)
        kwargs_40397 = {}
        # Getting the type of 'f' (line 222)
        f_40395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 222)
        close_40396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 12), f_40395, 'close')
        # Calling close(args, kwargs) (line 222)
        close_call_result_40398 = invoke(stypy.reporting.localization.Localization(__file__, 222, 12), close_40396, *[], **kwargs_40397)
        
        
        
        # Assigning a ListComp to a Name (line 224):
        
        # Assigning a ListComp to a Name (line 224):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to splitlines(...): (line 224)
        # Processing the call keyword arguments (line 224)
        kwargs_40407 = {}
        # Getting the type of 'content' (line 224)
        content_40405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 52), 'content', False)
        # Obtaining the member 'splitlines' of a type (line 224)
        splitlines_40406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 52), content_40405, 'splitlines')
        # Calling splitlines(args, kwargs) (line 224)
        splitlines_call_result_40408 = invoke(stypy.reporting.localization.Localization(__file__, 224, 52), splitlines_40406, *[], **kwargs_40407)
        
        comprehension_40409 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 17), splitlines_call_result_40408)
        # Assigning a type to the variable 'line' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 17), 'line', comprehension_40409)
        
        # Call to basename(...): (line 224)
        # Processing the call arguments (line 224)
        # Getting the type of 'line' (line 224)
        line_40402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 34), 'line', False)
        # Processing the call keyword arguments (line 224)
        kwargs_40403 = {}
        # Getting the type of 'os' (line 224)
        os_40399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 224)
        path_40400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 17), os_40399, 'path')
        # Obtaining the member 'basename' of a type (line 224)
        basename_40401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 17), path_40400, 'basename')
        # Calling basename(args, kwargs) (line 224)
        basename_call_result_40404 = invoke(stypy.reporting.localization.Localization(__file__, 224, 17), basename_40401, *[line_40402], **kwargs_40403)
        
        list_40410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 17), list_40410, basename_call_result_40404)
        # Assigning a type to the variable 'found' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'found', list_40410)
        
        # Assigning a List to a Name (line 225):
        
        # Assigning a List to a Name (line 225):
        
        # Obtaining an instance of the builtin type 'list' (line 225)
        list_40411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 225)
        # Adding element type (line 225)
        
        # Call to _make_ext_name(...): (line 225)
        # Processing the call arguments (line 225)
        str_40413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 35), 'str', 'xx')
        # Processing the call keyword arguments (line 225)
        kwargs_40414 = {}
        # Getting the type of '_make_ext_name' (line 225)
        _make_ext_name_40412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 20), '_make_ext_name', False)
        # Calling _make_ext_name(args, kwargs) (line 225)
        _make_ext_name_call_result_40415 = invoke(stypy.reporting.localization.Localization(__file__, 225, 20), _make_ext_name_40412, *[str_40413], **kwargs_40414)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 19), list_40411, _make_ext_name_call_result_40415)
        # Adding element type (line 225)
        str_40416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 20), 'str', 'UNKNOWN-0.0.0-py%s.%s.egg-info')
        
        # Obtaining the type of the subscript
        int_40417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 73), 'int')
        slice_40418 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 226, 55), None, int_40417, None)
        # Getting the type of 'sys' (line 226)
        sys_40419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 55), 'sys')
        # Obtaining the member 'version_info' of a type (line 226)
        version_info_40420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 55), sys_40419, 'version_info')
        # Obtaining the member '__getitem__' of a type (line 226)
        getitem___40421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 55), version_info_40420, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 226)
        subscript_call_result_40422 = invoke(stypy.reporting.localization.Localization(__file__, 226, 55), getitem___40421, slice_40418)
        
        # Applying the binary operator '%' (line 226)
        result_mod_40423 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 20), '%', str_40416, subscript_call_result_40422)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 225, 19), list_40411, result_mod_40423)
        
        # Assigning a type to the variable 'expected' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'expected', list_40411)
        
        # Call to assertEqual(...): (line 227)
        # Processing the call arguments (line 227)
        # Getting the type of 'found' (line 227)
        found_40426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 25), 'found', False)
        # Getting the type of 'expected' (line 227)
        expected_40427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 32), 'expected', False)
        # Processing the call keyword arguments (line 227)
        kwargs_40428 = {}
        # Getting the type of 'self' (line 227)
        self_40424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'self', False)
        # Obtaining the member 'assertEqual' of a type (line 227)
        assertEqual_40425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), self_40424, 'assertEqual')
        # Calling assertEqual(args, kwargs) (line 227)
        assertEqual_call_result_40429 = invoke(stypy.reporting.localization.Localization(__file__, 227, 8), assertEqual_40425, *[found_40426, expected_40427], **kwargs_40428)
        
        
        # ################# End of 'test_record_extensions(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_record_extensions' in the type store
        # Getting the type of 'stypy_return_type' (line 199)
        stypy_return_type_40430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40430)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_record_extensions'
        return stypy_return_type_40430


    @norecursion
    def test_debug_mode(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_debug_mode'
        module_type_store = module_type_store.open_function_context('test_debug_mode', 229, 4, False)
        # Assigning a type to the variable 'self' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        InstallTestCase.test_debug_mode.__dict__.__setitem__('stypy_localization', localization)
        InstallTestCase.test_debug_mode.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        InstallTestCase.test_debug_mode.__dict__.__setitem__('stypy_type_store', module_type_store)
        InstallTestCase.test_debug_mode.__dict__.__setitem__('stypy_function_name', 'InstallTestCase.test_debug_mode')
        InstallTestCase.test_debug_mode.__dict__.__setitem__('stypy_param_names_list', [])
        InstallTestCase.test_debug_mode.__dict__.__setitem__('stypy_varargs_param_name', None)
        InstallTestCase.test_debug_mode.__dict__.__setitem__('stypy_kwargs_param_name', None)
        InstallTestCase.test_debug_mode.__dict__.__setitem__('stypy_call_defaults', defaults)
        InstallTestCase.test_debug_mode.__dict__.__setitem__('stypy_call_varargs', varargs)
        InstallTestCase.test_debug_mode.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        InstallTestCase.test_debug_mode.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallTestCase.test_debug_mode', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_debug_mode', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_debug_mode(...)' code ##################

        
        # Assigning a Call to a Name (line 231):
        
        # Assigning a Call to a Name (line 231):
        
        # Call to len(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'self' (line 231)
        self_40432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 27), 'self', False)
        # Obtaining the member 'logs' of a type (line 231)
        logs_40433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 27), self_40432, 'logs')
        # Processing the call keyword arguments (line 231)
        kwargs_40434 = {}
        # Getting the type of 'len' (line 231)
        len_40431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 23), 'len', False)
        # Calling len(args, kwargs) (line 231)
        len_call_result_40435 = invoke(stypy.reporting.localization.Localization(__file__, 231, 23), len_40431, *[logs_40433], **kwargs_40434)
        
        # Assigning a type to the variable 'old_logs_len' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'old_logs_len', len_call_result_40435)
        
        # Assigning a Name to a Attribute (line 232):
        
        # Assigning a Name to a Attribute (line 232):
        # Getting the type of 'True' (line 232)
        True_40436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 31), 'True')
        # Getting the type of 'install_module' (line 232)
        install_module_40437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'install_module')
        # Setting the type of the member 'DEBUG' of a type (line 232)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 8), install_module_40437, 'DEBUG', True_40436)
        
        # Try-finally block (line 233)
        
        # Call to captured_stdout(...): (line 234)
        # Processing the call keyword arguments (line 234)
        kwargs_40439 = {}
        # Getting the type of 'captured_stdout' (line 234)
        captured_stdout_40438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 17), 'captured_stdout', False)
        # Calling captured_stdout(args, kwargs) (line 234)
        captured_stdout_call_result_40440 = invoke(stypy.reporting.localization.Localization(__file__, 234, 17), captured_stdout_40438, *[], **kwargs_40439)
        
        with_40441 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 234, 17), captured_stdout_call_result_40440, 'with parameter', '__enter__', '__exit__')

        if with_40441:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 234)
            enter___40442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 17), captured_stdout_call_result_40440, '__enter__')
            with_enter_40443 = invoke(stypy.reporting.localization.Localization(__file__, 234, 17), enter___40442)
            
            # Call to test_record(...): (line 235)
            # Processing the call keyword arguments (line 235)
            kwargs_40446 = {}
            # Getting the type of 'self' (line 235)
            self_40444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'self', False)
            # Obtaining the member 'test_record' of a type (line 235)
            test_record_40445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 16), self_40444, 'test_record')
            # Calling test_record(args, kwargs) (line 235)
            test_record_call_result_40447 = invoke(stypy.reporting.localization.Localization(__file__, 235, 16), test_record_40445, *[], **kwargs_40446)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 234)
            exit___40448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 17), captured_stdout_call_result_40440, '__exit__')
            with_exit_40449 = invoke(stypy.reporting.localization.Localization(__file__, 234, 17), exit___40448, None, None, None)

        
        # finally branch of the try-finally block (line 233)
        
        # Assigning a Name to a Attribute (line 237):
        
        # Assigning a Name to a Attribute (line 237):
        # Getting the type of 'False' (line 237)
        False_40450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 35), 'False')
        # Getting the type of 'install_module' (line 237)
        install_module_40451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'install_module')
        # Setting the type of the member 'DEBUG' of a type (line 237)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), install_module_40451, 'DEBUG', False_40450)
        
        
        # Call to assertGreater(...): (line 238)
        # Processing the call arguments (line 238)
        
        # Call to len(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'self' (line 238)
        self_40455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 31), 'self', False)
        # Obtaining the member 'logs' of a type (line 238)
        logs_40456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 31), self_40455, 'logs')
        # Processing the call keyword arguments (line 238)
        kwargs_40457 = {}
        # Getting the type of 'len' (line 238)
        len_40454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 27), 'len', False)
        # Calling len(args, kwargs) (line 238)
        len_call_result_40458 = invoke(stypy.reporting.localization.Localization(__file__, 238, 27), len_40454, *[logs_40456], **kwargs_40457)
        
        # Getting the type of 'old_logs_len' (line 238)
        old_logs_len_40459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 43), 'old_logs_len', False)
        # Processing the call keyword arguments (line 238)
        kwargs_40460 = {}
        # Getting the type of 'self' (line 238)
        self_40452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'self', False)
        # Obtaining the member 'assertGreater' of a type (line 238)
        assertGreater_40453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), self_40452, 'assertGreater')
        # Calling assertGreater(args, kwargs) (line 238)
        assertGreater_call_result_40461 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), assertGreater_40453, *[len_call_result_40458, old_logs_len_40459], **kwargs_40460)
        
        
        # ################# End of 'test_debug_mode(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_debug_mode' in the type store
        # Getting the type of 'stypy_return_type' (line 229)
        stypy_return_type_40462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40462)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_debug_mode'
        return stypy_return_type_40462


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 28, 0, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'InstallTestCase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'InstallTestCase' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'InstallTestCase', InstallTestCase)

@norecursion
def test_suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_suite'
    module_type_store = module_type_store.open_function_context('test_suite', 241, 0, False)
    
    # Passed parameters checking function
    test_suite.stypy_localization = localization
    test_suite.stypy_type_of_self = None
    test_suite.stypy_type_store = module_type_store
    test_suite.stypy_function_name = 'test_suite'
    test_suite.stypy_param_names_list = []
    test_suite.stypy_varargs_param_name = None
    test_suite.stypy_kwargs_param_name = None
    test_suite.stypy_call_defaults = defaults
    test_suite.stypy_call_varargs = varargs
    test_suite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_suite', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_suite', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_suite(...)' code ##################

    
    # Call to makeSuite(...): (line 242)
    # Processing the call arguments (line 242)
    # Getting the type of 'InstallTestCase' (line 242)
    InstallTestCase_40465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 30), 'InstallTestCase', False)
    # Processing the call keyword arguments (line 242)
    kwargs_40466 = {}
    # Getting the type of 'unittest' (line 242)
    unittest_40463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'unittest', False)
    # Obtaining the member 'makeSuite' of a type (line 242)
    makeSuite_40464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 11), unittest_40463, 'makeSuite')
    # Calling makeSuite(args, kwargs) (line 242)
    makeSuite_call_result_40467 = invoke(stypy.reporting.localization.Localization(__file__, 242, 11), makeSuite_40464, *[InstallTestCase_40465], **kwargs_40466)
    
    # Assigning a type to the variable 'stypy_return_type' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'stypy_return_type', makeSuite_call_result_40467)
    
    # ################# End of 'test_suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_suite' in the type store
    # Getting the type of 'stypy_return_type' (line 241)
    stypy_return_type_40468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_40468)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_suite'
    return stypy_return_type_40468

# Assigning a type to the variable 'test_suite' (line 241)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 0), 'test_suite', test_suite)

if (__name__ == '__main__'):
    
    # Call to run_unittest(...): (line 245)
    # Processing the call arguments (line 245)
    
    # Call to test_suite(...): (line 245)
    # Processing the call keyword arguments (line 245)
    kwargs_40471 = {}
    # Getting the type of 'test_suite' (line 245)
    test_suite_40470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 17), 'test_suite', False)
    # Calling test_suite(args, kwargs) (line 245)
    test_suite_call_result_40472 = invoke(stypy.reporting.localization.Localization(__file__, 245, 17), test_suite_40470, *[], **kwargs_40471)
    
    # Processing the call keyword arguments (line 245)
    kwargs_40473 = {}
    # Getting the type of 'run_unittest' (line 245)
    run_unittest_40469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 4), 'run_unittest', False)
    # Calling run_unittest(args, kwargs) (line 245)
    run_unittest_call_result_40474 = invoke(stypy.reporting.localization.Localization(__file__, 245, 4), run_unittest_40469, *[test_suite_call_result_40472], **kwargs_40473)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
