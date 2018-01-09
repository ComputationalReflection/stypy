
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import sys
4: from distutils.core import *
5: 
6: if 'setuptools' in sys.modules:
7:     have_setuptools = True
8:     from setuptools import setup as old_setup
9:     # easy_install imports math, it may be picked up from cwd
10:     from setuptools.command import easy_install
11:     try:
12:         # very old versions of setuptools don't have this
13:         from setuptools.command import bdist_egg
14:     except ImportError:
15:         have_setuptools = False
16: else:
17:     from distutils.core import setup as old_setup
18:     have_setuptools = False
19: 
20: import warnings
21: import distutils.core
22: import distutils.dist
23: 
24: from numpy.distutils.extension import Extension
25: from numpy.distutils.numpy_distribution import NumpyDistribution
26: from numpy.distutils.command import config, config_compiler, \
27:      build, build_py, build_ext, build_clib, build_src, build_scripts, \
28:      sdist, install_data, install_headers, install, bdist_rpm, \
29:      install_clib
30: from numpy.distutils.misc_util import get_data_files, is_sequence, is_string
31: 
32: numpy_cmdclass = {'build':            build.build,
33:                   'build_src':        build_src.build_src,
34:                   'build_scripts':    build_scripts.build_scripts,
35:                   'config_cc':        config_compiler.config_cc,
36:                   'config_fc':        config_compiler.config_fc,
37:                   'config':           config.config,
38:                   'build_ext':        build_ext.build_ext,
39:                   'build_py':         build_py.build_py,
40:                   'build_clib':       build_clib.build_clib,
41:                   'sdist':            sdist.sdist,
42:                   'install_data':     install_data.install_data,
43:                   'install_headers':  install_headers.install_headers,
44:                   'install_clib':     install_clib.install_clib,
45:                   'install':          install.install,
46:                   'bdist_rpm':        bdist_rpm.bdist_rpm,
47:                   }
48: if have_setuptools:
49:     # Use our own versions of develop and egg_info to ensure that build_src is
50:     # handled appropriately.
51:     from numpy.distutils.command import develop, egg_info
52:     numpy_cmdclass['bdist_egg'] = bdist_egg.bdist_egg
53:     numpy_cmdclass['develop'] = develop.develop
54:     numpy_cmdclass['easy_install'] = easy_install.easy_install
55:     numpy_cmdclass['egg_info'] = egg_info.egg_info
56: 
57: def _dict_append(d, **kws):
58:     for k, v in kws.items():
59:         if k not in d:
60:             d[k] = v
61:             continue
62:         dv = d[k]
63:         if isinstance(dv, tuple):
64:             d[k] = dv + tuple(v)
65:         elif isinstance(dv, list):
66:             d[k] = dv + list(v)
67:         elif isinstance(dv, dict):
68:             _dict_append(dv, **v)
69:         elif is_string(dv):
70:             d[k] = dv + v
71:         else:
72:             raise TypeError(repr(type(dv)))
73: 
74: def _command_line_ok(_cache=[]):
75:     ''' Return True if command line does not contain any
76:     help or display requests.
77:     '''
78:     if _cache:
79:         return _cache[0]
80:     ok = True
81:     display_opts = ['--'+n for n in Distribution.display_option_names]
82:     for o in Distribution.display_options:
83:         if o[1]:
84:             display_opts.append('-'+o[1])
85:     for arg in sys.argv:
86:         if arg.startswith('--help') or arg=='-h' or arg in display_opts:
87:             ok = False
88:             break
89:     _cache.append(ok)
90:     return ok
91: 
92: def get_distribution(always=False):
93:     dist = distutils.core._setup_distribution
94:     # XXX Hack to get numpy installable with easy_install.
95:     # The problem is easy_install runs it's own setup(), which
96:     # sets up distutils.core._setup_distribution. However,
97:     # when our setup() runs, that gets overwritten and lost.
98:     # We can't use isinstance, as the DistributionWithoutHelpCommands
99:     # class is local to a function in setuptools.command.easy_install
100:     if dist is not None and \
101:             'DistributionWithoutHelpCommands' in repr(dist):
102:         dist = None
103:     if always and dist is None:
104:         dist = NumpyDistribution()
105:     return dist
106: 
107: def setup(**attr):
108: 
109:     cmdclass = numpy_cmdclass.copy()
110: 
111:     new_attr = attr.copy()
112:     if 'cmdclass' in new_attr:
113:         cmdclass.update(new_attr['cmdclass'])
114:     new_attr['cmdclass'] = cmdclass
115: 
116:     if 'configuration' in new_attr:
117:         # To avoid calling configuration if there are any errors
118:         # or help request in command in the line.
119:         configuration = new_attr.pop('configuration')
120: 
121:         old_dist = distutils.core._setup_distribution
122:         old_stop = distutils.core._setup_stop_after
123:         distutils.core._setup_distribution = None
124:         distutils.core._setup_stop_after = "commandline"
125:         try:
126:             dist = setup(**new_attr)
127:         finally:
128:             distutils.core._setup_distribution = old_dist
129:             distutils.core._setup_stop_after = old_stop
130:         if dist.help or not _command_line_ok():
131:             # probably displayed help, skip running any commands
132:             return dist
133: 
134:         # create setup dictionary and append to new_attr
135:         config = configuration()
136:         if hasattr(config, 'todict'):
137:             config = config.todict()
138:         _dict_append(new_attr, **config)
139: 
140:     # Move extension source libraries to libraries
141:     libraries = []
142:     for ext in new_attr.get('ext_modules', []):
143:         new_libraries = []
144:         for item in ext.libraries:
145:             if is_sequence(item):
146:                 lib_name, build_info = item
147:                 _check_append_ext_library(libraries, lib_name, build_info)
148:                 new_libraries.append(lib_name)
149:             elif is_string(item):
150:                 new_libraries.append(item)
151:             else:
152:                 raise TypeError("invalid description of extension module "
153:                                 "library %r" % (item,))
154:         ext.libraries = new_libraries
155:     if libraries:
156:         if 'libraries' not in new_attr:
157:             new_attr['libraries'] = []
158:         for item in libraries:
159:             _check_append_library(new_attr['libraries'], item)
160: 
161:     # sources in ext_modules or libraries may contain header files
162:     if ('ext_modules' in new_attr or 'libraries' in new_attr) \
163:        and 'headers' not in new_attr:
164:         new_attr['headers'] = []
165: 
166:     # Use our custom NumpyDistribution class instead of distutils' one
167:     new_attr['distclass'] = NumpyDistribution
168: 
169:     return old_setup(**new_attr)
170: 
171: def _check_append_library(libraries, item):
172:     for libitem in libraries:
173:         if is_sequence(libitem):
174:             if is_sequence(item):
175:                 if item[0]==libitem[0]:
176:                     if item[1] is libitem[1]:
177:                         return
178:                     warnings.warn("[0] libraries list contains %r with"
179:                                   " different build_info" % (item[0],))
180:                     break
181:             else:
182:                 if item==libitem[0]:
183:                     warnings.warn("[1] libraries list contains %r with"
184:                                   " no build_info" % (item[0],))
185:                     break
186:         else:
187:             if is_sequence(item):
188:                 if item[0]==libitem:
189:                     warnings.warn("[2] libraries list contains %r with"
190:                                   " no build_info" % (item[0],))
191:                     break
192:             else:
193:                 if item==libitem:
194:                     return
195:     libraries.append(item)
196: 
197: def _check_append_ext_library(libraries, lib_name, build_info):
198:     for item in libraries:
199:         if is_sequence(item):
200:             if item[0]==lib_name:
201:                 if item[1] is build_info:
202:                     return
203:                 warnings.warn("[3] libraries list contains %r with"
204:                               " different build_info" % (lib_name,))
205:                 break
206:         elif item==lib_name:
207:             warnings.warn("[4] libraries list contains %r with"
208:                           " no build_info" % (lib_name,))
209:             break
210:     libraries.append((lib_name, build_info))
211: 

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

# 'from distutils.core import ' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_28866 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.core')

if (type(import_28866) is not StypyTypeError):

    if (import_28866 != 'pyd_module'):
        __import__(import_28866)
        sys_modules_28867 = sys.modules[import_28866]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.core', sys_modules_28867.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_28867, sys_modules_28867.module_type_store, module_type_store)
    else:
        from distutils.core import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.core', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'distutils.core' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.core', import_28866)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')



str_28868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 3), 'str', 'setuptools')
# Getting the type of 'sys' (line 6)
sys_28869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 19), 'sys')
# Obtaining the member 'modules' of a type (line 6)
modules_28870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 19), sys_28869, 'modules')
# Applying the binary operator 'in' (line 6)
result_contains_28871 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 3), 'in', str_28868, modules_28870)

# Testing the type of an if condition (line 6)
if_condition_28872 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 6, 0), result_contains_28871)
# Assigning a type to the variable 'if_condition_28872' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'if_condition_28872', if_condition_28872)
# SSA begins for if statement (line 6)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Name (line 7):

# Assigning a Name to a Name (line 7):
# Getting the type of 'True' (line 7)
True_28873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 22), 'True')
# Assigning a type to the variable 'have_setuptools' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'have_setuptools', True_28873)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))

# 'from setuptools import old_setup' statement (line 8)
from setuptools import setup as old_setup

import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'setuptools', None, module_type_store, ['setup'], [old_setup])
# Adding an alias
module_type_store.add_alias('old_setup', 'setup')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 4))

# 'from setuptools.command import easy_install' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_28874 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'setuptools.command')

if (type(import_28874) is not StypyTypeError):

    if (import_28874 != 'pyd_module'):
        __import__(import_28874)
        sys_modules_28875 = sys.modules[import_28874]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'setuptools.command', sys_modules_28875.module_type_store, module_type_store, ['easy_install'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 4), __file__, sys_modules_28875, sys_modules_28875.module_type_store, module_type_store)
    else:
        from setuptools.command import easy_install

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'setuptools.command', None, module_type_store, ['easy_install'], [easy_install])

else:
    # Assigning a type to the variable 'setuptools.command' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'setuptools.command', import_28874)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')



# SSA begins for try-except statement (line 11)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 8))

# 'from setuptools.command import bdist_egg' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_28876 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 8), 'setuptools.command')

if (type(import_28876) is not StypyTypeError):

    if (import_28876 != 'pyd_module'):
        __import__(import_28876)
        sys_modules_28877 = sys.modules[import_28876]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 8), 'setuptools.command', sys_modules_28877.module_type_store, module_type_store, ['bdist_egg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 8), __file__, sys_modules_28877, sys_modules_28877.module_type_store, module_type_store)
    else:
        from setuptools.command import bdist_egg

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 8), 'setuptools.command', None, module_type_store, ['bdist_egg'], [bdist_egg])

else:
    # Assigning a type to the variable 'setuptools.command' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'setuptools.command', import_28876)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

# SSA branch for the except part of a try statement (line 11)
# SSA branch for the except 'ImportError' branch of a try statement (line 11)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 15):

# Assigning a Name to a Name (line 15):
# Getting the type of 'False' (line 15)
False_28878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 26), 'False')
# Assigning a type to the variable 'have_setuptools' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'have_setuptools', False_28878)
# SSA join for try-except statement (line 11)
module_type_store = module_type_store.join_ssa_context()

# SSA branch for the else part of an if statement (line 6)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 4))

# 'from distutils.core import old_setup' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_28879 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 4), 'distutils.core')

if (type(import_28879) is not StypyTypeError):

    if (import_28879 != 'pyd_module'):
        __import__(import_28879)
        sys_modules_28880 = sys.modules[import_28879]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 4), 'distutils.core', sys_modules_28880.module_type_store, module_type_store, ['setup'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 4), __file__, sys_modules_28880, sys_modules_28880.module_type_store, module_type_store)
    else:
        from distutils.core import setup as old_setup

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 4), 'distutils.core', None, module_type_store, ['setup'], [old_setup])

else:
    # Assigning a type to the variable 'distutils.core' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'distutils.core', import_28879)

# Adding an alias
module_type_store.add_alias('old_setup', 'setup')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')


# Assigning a Name to a Name (line 18):

# Assigning a Name to a Name (line 18):
# Getting the type of 'False' (line 18)
False_28881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 22), 'False')
# Assigning a type to the variable 'have_setuptools' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'have_setuptools', False_28881)
# SSA join for if statement (line 6)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import warnings' statement (line 20)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import distutils.core' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_28882 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.core')

if (type(import_28882) is not StypyTypeError):

    if (import_28882 != 'pyd_module'):
        __import__(import_28882)
        sys_modules_28883 = sys.modules[import_28882]
        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.core', sys_modules_28883.module_type_store, module_type_store)
    else:
        import distutils.core

        import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.core', distutils.core, module_type_store)

else:
    # Assigning a type to the variable 'distutils.core' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.core', import_28882)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'import distutils.dist' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_28884 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.dist')

if (type(import_28884) is not StypyTypeError):

    if (import_28884 != 'pyd_module'):
        __import__(import_28884)
        sys_modules_28885 = sys.modules[import_28884]
        import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.dist', sys_modules_28885.module_type_store, module_type_store)
    else:
        import distutils.dist

        import_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.dist', distutils.dist, module_type_store)

else:
    # Assigning a type to the variable 'distutils.dist' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.dist', import_28884)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from numpy.distutils.extension import Extension' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_28886 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.distutils.extension')

if (type(import_28886) is not StypyTypeError):

    if (import_28886 != 'pyd_module'):
        __import__(import_28886)
        sys_modules_28887 = sys.modules[import_28886]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.distutils.extension', sys_modules_28887.module_type_store, module_type_store, ['Extension'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_28887, sys_modules_28887.module_type_store, module_type_store)
    else:
        from numpy.distutils.extension import Extension

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.distutils.extension', None, module_type_store, ['Extension'], [Extension])

else:
    # Assigning a type to the variable 'numpy.distutils.extension' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.distutils.extension', import_28886)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from numpy.distutils.numpy_distribution import NumpyDistribution' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_28888 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.distutils.numpy_distribution')

if (type(import_28888) is not StypyTypeError):

    if (import_28888 != 'pyd_module'):
        __import__(import_28888)
        sys_modules_28889 = sys.modules[import_28888]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.distutils.numpy_distribution', sys_modules_28889.module_type_store, module_type_store, ['NumpyDistribution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_28889, sys_modules_28889.module_type_store, module_type_store)
    else:
        from numpy.distutils.numpy_distribution import NumpyDistribution

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.distutils.numpy_distribution', None, module_type_store, ['NumpyDistribution'], [NumpyDistribution])

else:
    # Assigning a type to the variable 'numpy.distutils.numpy_distribution' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.distutils.numpy_distribution', import_28888)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from numpy.distutils.command import config, config_compiler, build, build_py, build_ext, build_clib, build_src, build_scripts, sdist, install_data, install_headers, install, bdist_rpm, install_clib' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_28890 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.distutils.command')

if (type(import_28890) is not StypyTypeError):

    if (import_28890 != 'pyd_module'):
        __import__(import_28890)
        sys_modules_28891 = sys.modules[import_28890]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.distutils.command', sys_modules_28891.module_type_store, module_type_store, ['config', 'config_compiler', 'build', 'build_py', 'build_ext', 'build_clib', 'build_src', 'build_scripts', 'sdist', 'install_data', 'install_headers', 'install', 'bdist_rpm', 'install_clib'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_28891, sys_modules_28891.module_type_store, module_type_store)
    else:
        from numpy.distutils.command import config, config_compiler, build, build_py, build_ext, build_clib, build_src, build_scripts, sdist, install_data, install_headers, install, bdist_rpm, install_clib

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.distutils.command', None, module_type_store, ['config', 'config_compiler', 'build', 'build_py', 'build_ext', 'build_clib', 'build_src', 'build_scripts', 'sdist', 'install_data', 'install_headers', 'install', 'bdist_rpm', 'install_clib'], [config, config_compiler, build, build_py, build_ext, build_clib, build_src, build_scripts, sdist, install_data, install_headers, install, bdist_rpm, install_clib])

else:
    # Assigning a type to the variable 'numpy.distutils.command' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.distutils.command', import_28890)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'from numpy.distutils.misc_util import get_data_files, is_sequence, is_string' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_28892 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy.distutils.misc_util')

if (type(import_28892) is not StypyTypeError):

    if (import_28892 != 'pyd_module'):
        __import__(import_28892)
        sys_modules_28893 = sys.modules[import_28892]
        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy.distutils.misc_util', sys_modules_28893.module_type_store, module_type_store, ['get_data_files', 'is_sequence', 'is_string'])
        nest_module(stypy.reporting.localization.Localization(__file__, 30, 0), __file__, sys_modules_28893, sys_modules_28893.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import get_data_files, is_sequence, is_string

        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy.distutils.misc_util', None, module_type_store, ['get_data_files', 'is_sequence', 'is_string'], [get_data_files, is_sequence, is_string])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'numpy.distutils.misc_util', import_28892)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')


# Assigning a Dict to a Name (line 32):

# Assigning a Dict to a Name (line 32):

# Obtaining an instance of the builtin type 'dict' (line 32)
dict_28894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 17), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 32)
# Adding element type (key, value) (line 32)
str_28895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 18), 'str', 'build')
# Getting the type of 'build' (line 32)
build_28896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 38), 'build')
# Obtaining the member 'build' of a type (line 32)
build_28897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 38), build_28896, 'build')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 17), dict_28894, (str_28895, build_28897))
# Adding element type (key, value) (line 32)
str_28898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 18), 'str', 'build_src')
# Getting the type of 'build_src' (line 33)
build_src_28899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 38), 'build_src')
# Obtaining the member 'build_src' of a type (line 33)
build_src_28900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 38), build_src_28899, 'build_src')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 17), dict_28894, (str_28898, build_src_28900))
# Adding element type (key, value) (line 32)
str_28901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 18), 'str', 'build_scripts')
# Getting the type of 'build_scripts' (line 34)
build_scripts_28902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 38), 'build_scripts')
# Obtaining the member 'build_scripts' of a type (line 34)
build_scripts_28903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 38), build_scripts_28902, 'build_scripts')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 17), dict_28894, (str_28901, build_scripts_28903))
# Adding element type (key, value) (line 32)
str_28904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 18), 'str', 'config_cc')
# Getting the type of 'config_compiler' (line 35)
config_compiler_28905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 38), 'config_compiler')
# Obtaining the member 'config_cc' of a type (line 35)
config_cc_28906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 38), config_compiler_28905, 'config_cc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 17), dict_28894, (str_28904, config_cc_28906))
# Adding element type (key, value) (line 32)
str_28907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 18), 'str', 'config_fc')
# Getting the type of 'config_compiler' (line 36)
config_compiler_28908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 38), 'config_compiler')
# Obtaining the member 'config_fc' of a type (line 36)
config_fc_28909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 38), config_compiler_28908, 'config_fc')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 17), dict_28894, (str_28907, config_fc_28909))
# Adding element type (key, value) (line 32)
str_28910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 18), 'str', 'config')
# Getting the type of 'config' (line 37)
config_28911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 38), 'config')
# Obtaining the member 'config' of a type (line 37)
config_28912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 38), config_28911, 'config')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 17), dict_28894, (str_28910, config_28912))
# Adding element type (key, value) (line 32)
str_28913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 18), 'str', 'build_ext')
# Getting the type of 'build_ext' (line 38)
build_ext_28914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 38), 'build_ext')
# Obtaining the member 'build_ext' of a type (line 38)
build_ext_28915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 38), build_ext_28914, 'build_ext')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 17), dict_28894, (str_28913, build_ext_28915))
# Adding element type (key, value) (line 32)
str_28916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 18), 'str', 'build_py')
# Getting the type of 'build_py' (line 39)
build_py_28917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 38), 'build_py')
# Obtaining the member 'build_py' of a type (line 39)
build_py_28918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 38), build_py_28917, 'build_py')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 17), dict_28894, (str_28916, build_py_28918))
# Adding element type (key, value) (line 32)
str_28919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 18), 'str', 'build_clib')
# Getting the type of 'build_clib' (line 40)
build_clib_28920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 38), 'build_clib')
# Obtaining the member 'build_clib' of a type (line 40)
build_clib_28921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 38), build_clib_28920, 'build_clib')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 17), dict_28894, (str_28919, build_clib_28921))
# Adding element type (key, value) (line 32)
str_28922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 18), 'str', 'sdist')
# Getting the type of 'sdist' (line 41)
sdist_28923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 38), 'sdist')
# Obtaining the member 'sdist' of a type (line 41)
sdist_28924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 38), sdist_28923, 'sdist')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 17), dict_28894, (str_28922, sdist_28924))
# Adding element type (key, value) (line 32)
str_28925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 18), 'str', 'install_data')
# Getting the type of 'install_data' (line 42)
install_data_28926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 38), 'install_data')
# Obtaining the member 'install_data' of a type (line 42)
install_data_28927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 38), install_data_28926, 'install_data')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 17), dict_28894, (str_28925, install_data_28927))
# Adding element type (key, value) (line 32)
str_28928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 18), 'str', 'install_headers')
# Getting the type of 'install_headers' (line 43)
install_headers_28929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 38), 'install_headers')
# Obtaining the member 'install_headers' of a type (line 43)
install_headers_28930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 38), install_headers_28929, 'install_headers')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 17), dict_28894, (str_28928, install_headers_28930))
# Adding element type (key, value) (line 32)
str_28931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 18), 'str', 'install_clib')
# Getting the type of 'install_clib' (line 44)
install_clib_28932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 38), 'install_clib')
# Obtaining the member 'install_clib' of a type (line 44)
install_clib_28933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 38), install_clib_28932, 'install_clib')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 17), dict_28894, (str_28931, install_clib_28933))
# Adding element type (key, value) (line 32)
str_28934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 18), 'str', 'install')
# Getting the type of 'install' (line 45)
install_28935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 38), 'install')
# Obtaining the member 'install' of a type (line 45)
install_28936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 38), install_28935, 'install')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 17), dict_28894, (str_28934, install_28936))
# Adding element type (key, value) (line 32)
str_28937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 18), 'str', 'bdist_rpm')
# Getting the type of 'bdist_rpm' (line 46)
bdist_rpm_28938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 38), 'bdist_rpm')
# Obtaining the member 'bdist_rpm' of a type (line 46)
bdist_rpm_28939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 38), bdist_rpm_28938, 'bdist_rpm')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 17), dict_28894, (str_28937, bdist_rpm_28939))

# Assigning a type to the variable 'numpy_cmdclass' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'numpy_cmdclass', dict_28894)

# Getting the type of 'have_setuptools' (line 48)
have_setuptools_28940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 3), 'have_setuptools')
# Testing the type of an if condition (line 48)
if_condition_28941 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 0), have_setuptools_28940)
# Assigning a type to the variable 'if_condition_28941' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'if_condition_28941', if_condition_28941)
# SSA begins for if statement (line 48)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 51, 4))

# 'from numpy.distutils.command import develop, egg_info' statement (line 51)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_28942 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 51, 4), 'numpy.distutils.command')

if (type(import_28942) is not StypyTypeError):

    if (import_28942 != 'pyd_module'):
        __import__(import_28942)
        sys_modules_28943 = sys.modules[import_28942]
        import_from_module(stypy.reporting.localization.Localization(__file__, 51, 4), 'numpy.distutils.command', sys_modules_28943.module_type_store, module_type_store, ['develop', 'egg_info'])
        nest_module(stypy.reporting.localization.Localization(__file__, 51, 4), __file__, sys_modules_28943, sys_modules_28943.module_type_store, module_type_store)
    else:
        from numpy.distutils.command import develop, egg_info

        import_from_module(stypy.reporting.localization.Localization(__file__, 51, 4), 'numpy.distutils.command', None, module_type_store, ['develop', 'egg_info'], [develop, egg_info])

else:
    # Assigning a type to the variable 'numpy.distutils.command' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'numpy.distutils.command', import_28942)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')


# Assigning a Attribute to a Subscript (line 52):

# Assigning a Attribute to a Subscript (line 52):
# Getting the type of 'bdist_egg' (line 52)
bdist_egg_28944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 34), 'bdist_egg')
# Obtaining the member 'bdist_egg' of a type (line 52)
bdist_egg_28945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 34), bdist_egg_28944, 'bdist_egg')
# Getting the type of 'numpy_cmdclass' (line 52)
numpy_cmdclass_28946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'numpy_cmdclass')
str_28947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 19), 'str', 'bdist_egg')
# Storing an element on a container (line 52)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 4), numpy_cmdclass_28946, (str_28947, bdist_egg_28945))

# Assigning a Attribute to a Subscript (line 53):

# Assigning a Attribute to a Subscript (line 53):
# Getting the type of 'develop' (line 53)
develop_28948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 32), 'develop')
# Obtaining the member 'develop' of a type (line 53)
develop_28949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 32), develop_28948, 'develop')
# Getting the type of 'numpy_cmdclass' (line 53)
numpy_cmdclass_28950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'numpy_cmdclass')
str_28951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 19), 'str', 'develop')
# Storing an element on a container (line 53)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 4), numpy_cmdclass_28950, (str_28951, develop_28949))

# Assigning a Attribute to a Subscript (line 54):

# Assigning a Attribute to a Subscript (line 54):
# Getting the type of 'easy_install' (line 54)
easy_install_28952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 37), 'easy_install')
# Obtaining the member 'easy_install' of a type (line 54)
easy_install_28953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 37), easy_install_28952, 'easy_install')
# Getting the type of 'numpy_cmdclass' (line 54)
numpy_cmdclass_28954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'numpy_cmdclass')
str_28955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 19), 'str', 'easy_install')
# Storing an element on a container (line 54)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 4), numpy_cmdclass_28954, (str_28955, easy_install_28953))

# Assigning a Attribute to a Subscript (line 55):

# Assigning a Attribute to a Subscript (line 55):
# Getting the type of 'egg_info' (line 55)
egg_info_28956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 33), 'egg_info')
# Obtaining the member 'egg_info' of a type (line 55)
egg_info_28957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 33), egg_info_28956, 'egg_info')
# Getting the type of 'numpy_cmdclass' (line 55)
numpy_cmdclass_28958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'numpy_cmdclass')
str_28959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 19), 'str', 'egg_info')
# Storing an element on a container (line 55)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 4), numpy_cmdclass_28958, (str_28959, egg_info_28957))
# SSA join for if statement (line 48)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def _dict_append(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_dict_append'
    module_type_store = module_type_store.open_function_context('_dict_append', 57, 0, False)
    
    # Passed parameters checking function
    _dict_append.stypy_localization = localization
    _dict_append.stypy_type_of_self = None
    _dict_append.stypy_type_store = module_type_store
    _dict_append.stypy_function_name = '_dict_append'
    _dict_append.stypy_param_names_list = ['d']
    _dict_append.stypy_varargs_param_name = None
    _dict_append.stypy_kwargs_param_name = 'kws'
    _dict_append.stypy_call_defaults = defaults
    _dict_append.stypy_call_varargs = varargs
    _dict_append.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_dict_append', ['d'], None, 'kws', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_dict_append', localization, ['d'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_dict_append(...)' code ##################

    
    
    # Call to items(...): (line 58)
    # Processing the call keyword arguments (line 58)
    kwargs_28962 = {}
    # Getting the type of 'kws' (line 58)
    kws_28960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'kws', False)
    # Obtaining the member 'items' of a type (line 58)
    items_28961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), kws_28960, 'items')
    # Calling items(args, kwargs) (line 58)
    items_call_result_28963 = invoke(stypy.reporting.localization.Localization(__file__, 58, 16), items_28961, *[], **kwargs_28962)
    
    # Testing the type of a for loop iterable (line 58)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 58, 4), items_call_result_28963)
    # Getting the type of the for loop variable (line 58)
    for_loop_var_28964 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 58, 4), items_call_result_28963)
    # Assigning a type to the variable 'k' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 4), for_loop_var_28964))
    # Assigning a type to the variable 'v' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 4), for_loop_var_28964))
    # SSA begins for a for statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'k' (line 59)
    k_28965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'k')
    # Getting the type of 'd' (line 59)
    d_28966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'd')
    # Applying the binary operator 'notin' (line 59)
    result_contains_28967 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 11), 'notin', k_28965, d_28966)
    
    # Testing the type of an if condition (line 59)
    if_condition_28968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), result_contains_28967)
    # Assigning a type to the variable 'if_condition_28968' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'if_condition_28968', if_condition_28968)
    # SSA begins for if statement (line 59)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 60):
    
    # Assigning a Name to a Subscript (line 60):
    # Getting the type of 'v' (line 60)
    v_28969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), 'v')
    # Getting the type of 'd' (line 60)
    d_28970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'd')
    # Getting the type of 'k' (line 60)
    k_28971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 14), 'k')
    # Storing an element on a container (line 60)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 12), d_28970, (k_28971, v_28969))
    # SSA join for if statement (line 59)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 62):
    
    # Assigning a Subscript to a Name (line 62):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 62)
    k_28972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'k')
    # Getting the type of 'd' (line 62)
    d_28973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 13), 'd')
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___28974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 13), d_28973, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_28975 = invoke(stypy.reporting.localization.Localization(__file__, 62, 13), getitem___28974, k_28972)
    
    # Assigning a type to the variable 'dv' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'dv', subscript_call_result_28975)
    
    # Type idiom detected: calculating its left and rigth part (line 63)
    # Getting the type of 'tuple' (line 63)
    tuple_28976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 26), 'tuple')
    # Getting the type of 'dv' (line 63)
    dv_28977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'dv')
    
    (may_be_28978, more_types_in_union_28979) = may_be_subtype(tuple_28976, dv_28977)

    if may_be_28978:

        if more_types_in_union_28979:
            # Runtime conditional SSA (line 63)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'dv' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'dv', remove_not_subtype_from_union(dv_28977, tuple))
        
        # Assigning a BinOp to a Subscript (line 64):
        
        # Assigning a BinOp to a Subscript (line 64):
        # Getting the type of 'dv' (line 64)
        dv_28980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 19), 'dv')
        
        # Call to tuple(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'v' (line 64)
        v_28982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 30), 'v', False)
        # Processing the call keyword arguments (line 64)
        kwargs_28983 = {}
        # Getting the type of 'tuple' (line 64)
        tuple_28981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'tuple', False)
        # Calling tuple(args, kwargs) (line 64)
        tuple_call_result_28984 = invoke(stypy.reporting.localization.Localization(__file__, 64, 24), tuple_28981, *[v_28982], **kwargs_28983)
        
        # Applying the binary operator '+' (line 64)
        result_add_28985 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 19), '+', dv_28980, tuple_call_result_28984)
        
        # Getting the type of 'd' (line 64)
        d_28986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'd')
        # Getting the type of 'k' (line 64)
        k_28987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 14), 'k')
        # Storing an element on a container (line 64)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 12), d_28986, (k_28987, result_add_28985))

        if more_types_in_union_28979:
            # Runtime conditional SSA for else branch (line 63)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_28978) or more_types_in_union_28979):
        # Assigning a type to the variable 'dv' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'dv', remove_subtype_from_union(dv_28977, tuple))
        
        # Type idiom detected: calculating its left and rigth part (line 65)
        # Getting the type of 'list' (line 65)
        list_28988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'list')
        # Getting the type of 'dv' (line 65)
        dv_28989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 24), 'dv')
        
        (may_be_28990, more_types_in_union_28991) = may_be_subtype(list_28988, dv_28989)

        if may_be_28990:

            if more_types_in_union_28991:
                # Runtime conditional SSA (line 65)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'dv' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 13), 'dv', remove_not_subtype_from_union(dv_28989, list))
            
            # Assigning a BinOp to a Subscript (line 66):
            
            # Assigning a BinOp to a Subscript (line 66):
            # Getting the type of 'dv' (line 66)
            dv_28992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 'dv')
            
            # Call to list(...): (line 66)
            # Processing the call arguments (line 66)
            # Getting the type of 'v' (line 66)
            v_28994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 29), 'v', False)
            # Processing the call keyword arguments (line 66)
            kwargs_28995 = {}
            # Getting the type of 'list' (line 66)
            list_28993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'list', False)
            # Calling list(args, kwargs) (line 66)
            list_call_result_28996 = invoke(stypy.reporting.localization.Localization(__file__, 66, 24), list_28993, *[v_28994], **kwargs_28995)
            
            # Applying the binary operator '+' (line 66)
            result_add_28997 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 19), '+', dv_28992, list_call_result_28996)
            
            # Getting the type of 'd' (line 66)
            d_28998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'd')
            # Getting the type of 'k' (line 66)
            k_28999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 14), 'k')
            # Storing an element on a container (line 66)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 12), d_28998, (k_28999, result_add_28997))

            if more_types_in_union_28991:
                # Runtime conditional SSA for else branch (line 65)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_28990) or more_types_in_union_28991):
            # Assigning a type to the variable 'dv' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 13), 'dv', remove_subtype_from_union(dv_28989, list))
            
            # Type idiom detected: calculating its left and rigth part (line 67)
            # Getting the type of 'dict' (line 67)
            dict_29000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 28), 'dict')
            # Getting the type of 'dv' (line 67)
            dv_29001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'dv')
            
            (may_be_29002, more_types_in_union_29003) = may_be_subtype(dict_29000, dv_29001)

            if may_be_29002:

                if more_types_in_union_29003:
                    # Runtime conditional SSA (line 67)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'dv' (line 67)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 13), 'dv', remove_not_subtype_from_union(dv_29001, dict))
                
                # Call to _dict_append(...): (line 68)
                # Processing the call arguments (line 68)
                # Getting the type of 'dv' (line 68)
                dv_29005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 'dv', False)
                # Processing the call keyword arguments (line 68)
                # Getting the type of 'v' (line 68)
                v_29006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 31), 'v', False)
                kwargs_29007 = {'v_29006': v_29006}
                # Getting the type of '_dict_append' (line 68)
                _dict_append_29004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), '_dict_append', False)
                # Calling _dict_append(args, kwargs) (line 68)
                _dict_append_call_result_29008 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), _dict_append_29004, *[dv_29005], **kwargs_29007)
                

                if more_types_in_union_29003:
                    # Runtime conditional SSA for else branch (line 67)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_29002) or more_types_in_union_29003):
                # Assigning a type to the variable 'dv' (line 67)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 13), 'dv', remove_subtype_from_union(dv_29001, dict))
                
                
                # Call to is_string(...): (line 69)
                # Processing the call arguments (line 69)
                # Getting the type of 'dv' (line 69)
                dv_29010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), 'dv', False)
                # Processing the call keyword arguments (line 69)
                kwargs_29011 = {}
                # Getting the type of 'is_string' (line 69)
                is_string_29009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'is_string', False)
                # Calling is_string(args, kwargs) (line 69)
                is_string_call_result_29012 = invoke(stypy.reporting.localization.Localization(__file__, 69, 13), is_string_29009, *[dv_29010], **kwargs_29011)
                
                # Testing the type of an if condition (line 69)
                if_condition_29013 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 13), is_string_call_result_29012)
                # Assigning a type to the variable 'if_condition_29013' (line 69)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'if_condition_29013', if_condition_29013)
                # SSA begins for if statement (line 69)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Subscript (line 70):
                
                # Assigning a BinOp to a Subscript (line 70):
                # Getting the type of 'dv' (line 70)
                dv_29014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'dv')
                # Getting the type of 'v' (line 70)
                v_29015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 24), 'v')
                # Applying the binary operator '+' (line 70)
                result_add_29016 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 19), '+', dv_29014, v_29015)
                
                # Getting the type of 'd' (line 70)
                d_29017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'd')
                # Getting the type of 'k' (line 70)
                k_29018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 14), 'k')
                # Storing an element on a container (line 70)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 12), d_29017, (k_29018, result_add_29016))
                # SSA branch for the else part of an if statement (line 69)
                module_type_store.open_ssa_branch('else')
                
                # Call to TypeError(...): (line 72)
                # Processing the call arguments (line 72)
                
                # Call to repr(...): (line 72)
                # Processing the call arguments (line 72)
                
                # Call to type(...): (line 72)
                # Processing the call arguments (line 72)
                # Getting the type of 'dv' (line 72)
                dv_29022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 38), 'dv', False)
                # Processing the call keyword arguments (line 72)
                kwargs_29023 = {}
                # Getting the type of 'type' (line 72)
                type_29021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 33), 'type', False)
                # Calling type(args, kwargs) (line 72)
                type_call_result_29024 = invoke(stypy.reporting.localization.Localization(__file__, 72, 33), type_29021, *[dv_29022], **kwargs_29023)
                
                # Processing the call keyword arguments (line 72)
                kwargs_29025 = {}
                # Getting the type of 'repr' (line 72)
                repr_29020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'repr', False)
                # Calling repr(args, kwargs) (line 72)
                repr_call_result_29026 = invoke(stypy.reporting.localization.Localization(__file__, 72, 28), repr_29020, *[type_call_result_29024], **kwargs_29025)
                
                # Processing the call keyword arguments (line 72)
                kwargs_29027 = {}
                # Getting the type of 'TypeError' (line 72)
                TypeError_29019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 18), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 72)
                TypeError_call_result_29028 = invoke(stypy.reporting.localization.Localization(__file__, 72, 18), TypeError_29019, *[repr_call_result_29026], **kwargs_29027)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 72, 12), TypeError_call_result_29028, 'raise parameter', BaseException)
                # SSA join for if statement (line 69)
                module_type_store = module_type_store.join_ssa_context()
                

                if (may_be_29002 and more_types_in_union_29003):
                    # SSA join for if statement (line 67)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_28990 and more_types_in_union_28991):
                # SSA join for if statement (line 65)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_28978 and more_types_in_union_28979):
            # SSA join for if statement (line 63)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_dict_append(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_dict_append' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_29029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29029)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_dict_append'
    return stypy_return_type_29029

# Assigning a type to the variable '_dict_append' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), '_dict_append', _dict_append)

@norecursion
def _command_line_ok(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'list' (line 74)
    list_29030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 74)
    
    defaults = [list_29030]
    # Create a new context for function '_command_line_ok'
    module_type_store = module_type_store.open_function_context('_command_line_ok', 74, 0, False)
    
    # Passed parameters checking function
    _command_line_ok.stypy_localization = localization
    _command_line_ok.stypy_type_of_self = None
    _command_line_ok.stypy_type_store = module_type_store
    _command_line_ok.stypy_function_name = '_command_line_ok'
    _command_line_ok.stypy_param_names_list = ['_cache']
    _command_line_ok.stypy_varargs_param_name = None
    _command_line_ok.stypy_kwargs_param_name = None
    _command_line_ok.stypy_call_defaults = defaults
    _command_line_ok.stypy_call_varargs = varargs
    _command_line_ok.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_command_line_ok', ['_cache'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_command_line_ok', localization, ['_cache'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_command_line_ok(...)' code ##################

    str_29031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, (-1)), 'str', ' Return True if command line does not contain any\n    help or display requests.\n    ')
    
    # Getting the type of '_cache' (line 78)
    _cache_29032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 7), '_cache')
    # Testing the type of an if condition (line 78)
    if_condition_29033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 4), _cache_29032)
    # Assigning a type to the variable 'if_condition_29033' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'if_condition_29033', if_condition_29033)
    # SSA begins for if statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_29034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 22), 'int')
    # Getting the type of '_cache' (line 79)
    _cache_29035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), '_cache')
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___29036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 15), _cache_29035, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_29037 = invoke(stypy.reporting.localization.Localization(__file__, 79, 15), getitem___29036, int_29034)
    
    # Assigning a type to the variable 'stypy_return_type' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'stypy_return_type', subscript_call_result_29037)
    # SSA join for if statement (line 78)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 80):
    
    # Assigning a Name to a Name (line 80):
    # Getting the type of 'True' (line 80)
    True_29038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 9), 'True')
    # Assigning a type to the variable 'ok' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'ok', True_29038)
    
    # Assigning a ListComp to a Name (line 81):
    
    # Assigning a ListComp to a Name (line 81):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'Distribution' (line 81)
    Distribution_29042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 36), 'Distribution')
    # Obtaining the member 'display_option_names' of a type (line 81)
    display_option_names_29043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 36), Distribution_29042, 'display_option_names')
    comprehension_29044 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 20), display_option_names_29043)
    # Assigning a type to the variable 'n' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'n', comprehension_29044)
    str_29039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'str', '--')
    # Getting the type of 'n' (line 81)
    n_29040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 25), 'n')
    # Applying the binary operator '+' (line 81)
    result_add_29041 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 20), '+', str_29039, n_29040)
    
    list_29045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 20), list_29045, result_add_29041)
    # Assigning a type to the variable 'display_opts' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'display_opts', list_29045)
    
    # Getting the type of 'Distribution' (line 82)
    Distribution_29046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'Distribution')
    # Obtaining the member 'display_options' of a type (line 82)
    display_options_29047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 13), Distribution_29046, 'display_options')
    # Testing the type of a for loop iterable (line 82)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 82, 4), display_options_29047)
    # Getting the type of the for loop variable (line 82)
    for_loop_var_29048 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 82, 4), display_options_29047)
    # Assigning a type to the variable 'o' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'o', for_loop_var_29048)
    # SSA begins for a for statement (line 82)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Obtaining the type of the subscript
    int_29049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 13), 'int')
    # Getting the type of 'o' (line 83)
    o_29050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'o')
    # Obtaining the member '__getitem__' of a type (line 83)
    getitem___29051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 11), o_29050, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 83)
    subscript_call_result_29052 = invoke(stypy.reporting.localization.Localization(__file__, 83, 11), getitem___29051, int_29049)
    
    # Testing the type of an if condition (line 83)
    if_condition_29053 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 8), subscript_call_result_29052)
    # Assigning a type to the variable 'if_condition_29053' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'if_condition_29053', if_condition_29053)
    # SSA begins for if statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 84)
    # Processing the call arguments (line 84)
    str_29056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 32), 'str', '-')
    
    # Obtaining the type of the subscript
    int_29057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 38), 'int')
    # Getting the type of 'o' (line 84)
    o_29058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 36), 'o', False)
    # Obtaining the member '__getitem__' of a type (line 84)
    getitem___29059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 36), o_29058, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 84)
    subscript_call_result_29060 = invoke(stypy.reporting.localization.Localization(__file__, 84, 36), getitem___29059, int_29057)
    
    # Applying the binary operator '+' (line 84)
    result_add_29061 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 32), '+', str_29056, subscript_call_result_29060)
    
    # Processing the call keyword arguments (line 84)
    kwargs_29062 = {}
    # Getting the type of 'display_opts' (line 84)
    display_opts_29054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'display_opts', False)
    # Obtaining the member 'append' of a type (line 84)
    append_29055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), display_opts_29054, 'append')
    # Calling append(args, kwargs) (line 84)
    append_call_result_29063 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), append_29055, *[result_add_29061], **kwargs_29062)
    
    # SSA join for if statement (line 83)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'sys' (line 85)
    sys_29064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'sys')
    # Obtaining the member 'argv' of a type (line 85)
    argv_29065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 15), sys_29064, 'argv')
    # Testing the type of a for loop iterable (line 85)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 85, 4), argv_29065)
    # Getting the type of the for loop variable (line 85)
    for_loop_var_29066 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 85, 4), argv_29065)
    # Assigning a type to the variable 'arg' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'arg', for_loop_var_29066)
    # SSA begins for a for statement (line 85)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Call to startswith(...): (line 86)
    # Processing the call arguments (line 86)
    str_29069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 26), 'str', '--help')
    # Processing the call keyword arguments (line 86)
    kwargs_29070 = {}
    # Getting the type of 'arg' (line 86)
    arg_29067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'arg', False)
    # Obtaining the member 'startswith' of a type (line 86)
    startswith_29068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 11), arg_29067, 'startswith')
    # Calling startswith(args, kwargs) (line 86)
    startswith_call_result_29071 = invoke(stypy.reporting.localization.Localization(__file__, 86, 11), startswith_29068, *[str_29069], **kwargs_29070)
    
    
    # Getting the type of 'arg' (line 86)
    arg_29072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 39), 'arg')
    str_29073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 44), 'str', '-h')
    # Applying the binary operator '==' (line 86)
    result_eq_29074 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 39), '==', arg_29072, str_29073)
    
    # Applying the binary operator 'or' (line 86)
    result_or_keyword_29075 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 11), 'or', startswith_call_result_29071, result_eq_29074)
    
    # Getting the type of 'arg' (line 86)
    arg_29076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 52), 'arg')
    # Getting the type of 'display_opts' (line 86)
    display_opts_29077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 59), 'display_opts')
    # Applying the binary operator 'in' (line 86)
    result_contains_29078 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 52), 'in', arg_29076, display_opts_29077)
    
    # Applying the binary operator 'or' (line 86)
    result_or_keyword_29079 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 11), 'or', result_or_keyword_29075, result_contains_29078)
    
    # Testing the type of an if condition (line 86)
    if_condition_29080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 8), result_or_keyword_29079)
    # Assigning a type to the variable 'if_condition_29080' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'if_condition_29080', if_condition_29080)
    # SSA begins for if statement (line 86)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 87):
    
    # Assigning a Name to a Name (line 87):
    # Getting the type of 'False' (line 87)
    False_29081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 17), 'False')
    # Assigning a type to the variable 'ok' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'ok', False_29081)
    # SSA join for if statement (line 86)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'ok' (line 89)
    ok_29084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'ok', False)
    # Processing the call keyword arguments (line 89)
    kwargs_29085 = {}
    # Getting the type of '_cache' (line 89)
    _cache_29082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), '_cache', False)
    # Obtaining the member 'append' of a type (line 89)
    append_29083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 4), _cache_29082, 'append')
    # Calling append(args, kwargs) (line 89)
    append_call_result_29086 = invoke(stypy.reporting.localization.Localization(__file__, 89, 4), append_29083, *[ok_29084], **kwargs_29085)
    
    # Getting the type of 'ok' (line 90)
    ok_29087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 11), 'ok')
    # Assigning a type to the variable 'stypy_return_type' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type', ok_29087)
    
    # ################# End of '_command_line_ok(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_command_line_ok' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_29088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29088)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_command_line_ok'
    return stypy_return_type_29088

# Assigning a type to the variable '_command_line_ok' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), '_command_line_ok', _command_line_ok)

@norecursion
def get_distribution(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 92)
    False_29089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'False')
    defaults = [False_29089]
    # Create a new context for function 'get_distribution'
    module_type_store = module_type_store.open_function_context('get_distribution', 92, 0, False)
    
    # Passed parameters checking function
    get_distribution.stypy_localization = localization
    get_distribution.stypy_type_of_self = None
    get_distribution.stypy_type_store = module_type_store
    get_distribution.stypy_function_name = 'get_distribution'
    get_distribution.stypy_param_names_list = ['always']
    get_distribution.stypy_varargs_param_name = None
    get_distribution.stypy_kwargs_param_name = None
    get_distribution.stypy_call_defaults = defaults
    get_distribution.stypy_call_varargs = varargs
    get_distribution.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_distribution', ['always'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_distribution', localization, ['always'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_distribution(...)' code ##################

    
    # Assigning a Attribute to a Name (line 93):
    
    # Assigning a Attribute to a Name (line 93):
    # Getting the type of 'distutils' (line 93)
    distutils_29090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'distutils')
    # Obtaining the member 'core' of a type (line 93)
    core_29091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 11), distutils_29090, 'core')
    # Obtaining the member '_setup_distribution' of a type (line 93)
    _setup_distribution_29092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 11), core_29091, '_setup_distribution')
    # Assigning a type to the variable 'dist' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'dist', _setup_distribution_29092)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dist' (line 100)
    dist_29093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 7), 'dist')
    # Getting the type of 'None' (line 100)
    None_29094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 19), 'None')
    # Applying the binary operator 'isnot' (line 100)
    result_is_not_29095 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 7), 'isnot', dist_29093, None_29094)
    
    
    str_29096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 12), 'str', 'DistributionWithoutHelpCommands')
    
    # Call to repr(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'dist' (line 101)
    dist_29098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 54), 'dist', False)
    # Processing the call keyword arguments (line 101)
    kwargs_29099 = {}
    # Getting the type of 'repr' (line 101)
    repr_29097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 49), 'repr', False)
    # Calling repr(args, kwargs) (line 101)
    repr_call_result_29100 = invoke(stypy.reporting.localization.Localization(__file__, 101, 49), repr_29097, *[dist_29098], **kwargs_29099)
    
    # Applying the binary operator 'in' (line 101)
    result_contains_29101 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 12), 'in', str_29096, repr_call_result_29100)
    
    # Applying the binary operator 'and' (line 100)
    result_and_keyword_29102 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 7), 'and', result_is_not_29095, result_contains_29101)
    
    # Testing the type of an if condition (line 100)
    if_condition_29103 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 4), result_and_keyword_29102)
    # Assigning a type to the variable 'if_condition_29103' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'if_condition_29103', if_condition_29103)
    # SSA begins for if statement (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 102):
    
    # Assigning a Name to a Name (line 102):
    # Getting the type of 'None' (line 102)
    None_29104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'None')
    # Assigning a type to the variable 'dist' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'dist', None_29104)
    # SSA join for if statement (line 100)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'always' (line 103)
    always_29105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 7), 'always')
    
    # Getting the type of 'dist' (line 103)
    dist_29106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 18), 'dist')
    # Getting the type of 'None' (line 103)
    None_29107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'None')
    # Applying the binary operator 'is' (line 103)
    result_is__29108 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 18), 'is', dist_29106, None_29107)
    
    # Applying the binary operator 'and' (line 103)
    result_and_keyword_29109 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 7), 'and', always_29105, result_is__29108)
    
    # Testing the type of an if condition (line 103)
    if_condition_29110 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 4), result_and_keyword_29109)
    # Assigning a type to the variable 'if_condition_29110' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'if_condition_29110', if_condition_29110)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 104):
    
    # Assigning a Call to a Name (line 104):
    
    # Call to NumpyDistribution(...): (line 104)
    # Processing the call keyword arguments (line 104)
    kwargs_29112 = {}
    # Getting the type of 'NumpyDistribution' (line 104)
    NumpyDistribution_29111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'NumpyDistribution', False)
    # Calling NumpyDistribution(args, kwargs) (line 104)
    NumpyDistribution_call_result_29113 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), NumpyDistribution_29111, *[], **kwargs_29112)
    
    # Assigning a type to the variable 'dist' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'dist', NumpyDistribution_call_result_29113)
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'dist' (line 105)
    dist_29114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'dist')
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', dist_29114)
    
    # ################# End of 'get_distribution(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_distribution' in the type store
    # Getting the type of 'stypy_return_type' (line 92)
    stypy_return_type_29115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29115)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_distribution'
    return stypy_return_type_29115

# Assigning a type to the variable 'get_distribution' (line 92)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 0), 'get_distribution', get_distribution)

@norecursion
def setup(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'setup'
    module_type_store = module_type_store.open_function_context('setup', 107, 0, False)
    
    # Passed parameters checking function
    setup.stypy_localization = localization
    setup.stypy_type_of_self = None
    setup.stypy_type_store = module_type_store
    setup.stypy_function_name = 'setup'
    setup.stypy_param_names_list = []
    setup.stypy_varargs_param_name = None
    setup.stypy_kwargs_param_name = 'attr'
    setup.stypy_call_defaults = defaults
    setup.stypy_call_varargs = varargs
    setup.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'setup', [], None, 'attr', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'setup', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'setup(...)' code ##################

    
    # Assigning a Call to a Name (line 109):
    
    # Assigning a Call to a Name (line 109):
    
    # Call to copy(...): (line 109)
    # Processing the call keyword arguments (line 109)
    kwargs_29118 = {}
    # Getting the type of 'numpy_cmdclass' (line 109)
    numpy_cmdclass_29116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'numpy_cmdclass', False)
    # Obtaining the member 'copy' of a type (line 109)
    copy_29117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 15), numpy_cmdclass_29116, 'copy')
    # Calling copy(args, kwargs) (line 109)
    copy_call_result_29119 = invoke(stypy.reporting.localization.Localization(__file__, 109, 15), copy_29117, *[], **kwargs_29118)
    
    # Assigning a type to the variable 'cmdclass' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'cmdclass', copy_call_result_29119)
    
    # Assigning a Call to a Name (line 111):
    
    # Assigning a Call to a Name (line 111):
    
    # Call to copy(...): (line 111)
    # Processing the call keyword arguments (line 111)
    kwargs_29122 = {}
    # Getting the type of 'attr' (line 111)
    attr_29120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'attr', False)
    # Obtaining the member 'copy' of a type (line 111)
    copy_29121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), attr_29120, 'copy')
    # Calling copy(args, kwargs) (line 111)
    copy_call_result_29123 = invoke(stypy.reporting.localization.Localization(__file__, 111, 15), copy_29121, *[], **kwargs_29122)
    
    # Assigning a type to the variable 'new_attr' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'new_attr', copy_call_result_29123)
    
    
    str_29124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 7), 'str', 'cmdclass')
    # Getting the type of 'new_attr' (line 112)
    new_attr_29125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 21), 'new_attr')
    # Applying the binary operator 'in' (line 112)
    result_contains_29126 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 7), 'in', str_29124, new_attr_29125)
    
    # Testing the type of an if condition (line 112)
    if_condition_29127 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 4), result_contains_29126)
    # Assigning a type to the variable 'if_condition_29127' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'if_condition_29127', if_condition_29127)
    # SSA begins for if statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to update(...): (line 113)
    # Processing the call arguments (line 113)
    
    # Obtaining the type of the subscript
    str_29130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 33), 'str', 'cmdclass')
    # Getting the type of 'new_attr' (line 113)
    new_attr_29131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'new_attr', False)
    # Obtaining the member '__getitem__' of a type (line 113)
    getitem___29132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 24), new_attr_29131, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 113)
    subscript_call_result_29133 = invoke(stypy.reporting.localization.Localization(__file__, 113, 24), getitem___29132, str_29130)
    
    # Processing the call keyword arguments (line 113)
    kwargs_29134 = {}
    # Getting the type of 'cmdclass' (line 113)
    cmdclass_29128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'cmdclass', False)
    # Obtaining the member 'update' of a type (line 113)
    update_29129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), cmdclass_29128, 'update')
    # Calling update(args, kwargs) (line 113)
    update_call_result_29135 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), update_29129, *[subscript_call_result_29133], **kwargs_29134)
    
    # SSA join for if statement (line 112)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 114):
    
    # Assigning a Name to a Subscript (line 114):
    # Getting the type of 'cmdclass' (line 114)
    cmdclass_29136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 27), 'cmdclass')
    # Getting the type of 'new_attr' (line 114)
    new_attr_29137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'new_attr')
    str_29138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 13), 'str', 'cmdclass')
    # Storing an element on a container (line 114)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 4), new_attr_29137, (str_29138, cmdclass_29136))
    
    
    str_29139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 7), 'str', 'configuration')
    # Getting the type of 'new_attr' (line 116)
    new_attr_29140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 'new_attr')
    # Applying the binary operator 'in' (line 116)
    result_contains_29141 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 7), 'in', str_29139, new_attr_29140)
    
    # Testing the type of an if condition (line 116)
    if_condition_29142 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 4), result_contains_29141)
    # Assigning a type to the variable 'if_condition_29142' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'if_condition_29142', if_condition_29142)
    # SSA begins for if statement (line 116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 119):
    
    # Assigning a Call to a Name (line 119):
    
    # Call to pop(...): (line 119)
    # Processing the call arguments (line 119)
    str_29145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 37), 'str', 'configuration')
    # Processing the call keyword arguments (line 119)
    kwargs_29146 = {}
    # Getting the type of 'new_attr' (line 119)
    new_attr_29143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), 'new_attr', False)
    # Obtaining the member 'pop' of a type (line 119)
    pop_29144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 24), new_attr_29143, 'pop')
    # Calling pop(args, kwargs) (line 119)
    pop_call_result_29147 = invoke(stypy.reporting.localization.Localization(__file__, 119, 24), pop_29144, *[str_29145], **kwargs_29146)
    
    # Assigning a type to the variable 'configuration' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'configuration', pop_call_result_29147)
    
    # Assigning a Attribute to a Name (line 121):
    
    # Assigning a Attribute to a Name (line 121):
    # Getting the type of 'distutils' (line 121)
    distutils_29148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'distutils')
    # Obtaining the member 'core' of a type (line 121)
    core_29149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 19), distutils_29148, 'core')
    # Obtaining the member '_setup_distribution' of a type (line 121)
    _setup_distribution_29150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 19), core_29149, '_setup_distribution')
    # Assigning a type to the variable 'old_dist' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'old_dist', _setup_distribution_29150)
    
    # Assigning a Attribute to a Name (line 122):
    
    # Assigning a Attribute to a Name (line 122):
    # Getting the type of 'distutils' (line 122)
    distutils_29151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'distutils')
    # Obtaining the member 'core' of a type (line 122)
    core_29152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 19), distutils_29151, 'core')
    # Obtaining the member '_setup_stop_after' of a type (line 122)
    _setup_stop_after_29153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 19), core_29152, '_setup_stop_after')
    # Assigning a type to the variable 'old_stop' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'old_stop', _setup_stop_after_29153)
    
    # Assigning a Name to a Attribute (line 123):
    
    # Assigning a Name to a Attribute (line 123):
    # Getting the type of 'None' (line 123)
    None_29154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 45), 'None')
    # Getting the type of 'distutils' (line 123)
    distutils_29155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'distutils')
    # Obtaining the member 'core' of a type (line 123)
    core_29156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), distutils_29155, 'core')
    # Setting the type of the member '_setup_distribution' of a type (line 123)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), core_29156, '_setup_distribution', None_29154)
    
    # Assigning a Str to a Attribute (line 124):
    
    # Assigning a Str to a Attribute (line 124):
    str_29157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 43), 'str', 'commandline')
    # Getting the type of 'distutils' (line 124)
    distutils_29158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'distutils')
    # Obtaining the member 'core' of a type (line 124)
    core_29159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), distutils_29158, 'core')
    # Setting the type of the member '_setup_stop_after' of a type (line 124)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), core_29159, '_setup_stop_after', str_29157)
    
    # Try-finally block (line 125)
    
    # Assigning a Call to a Name (line 126):
    
    # Assigning a Call to a Name (line 126):
    
    # Call to setup(...): (line 126)
    # Processing the call keyword arguments (line 126)
    # Getting the type of 'new_attr' (line 126)
    new_attr_29161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 27), 'new_attr', False)
    kwargs_29162 = {'new_attr_29161': new_attr_29161}
    # Getting the type of 'setup' (line 126)
    setup_29160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 19), 'setup', False)
    # Calling setup(args, kwargs) (line 126)
    setup_call_result_29163 = invoke(stypy.reporting.localization.Localization(__file__, 126, 19), setup_29160, *[], **kwargs_29162)
    
    # Assigning a type to the variable 'dist' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'dist', setup_call_result_29163)
    
    # finally branch of the try-finally block (line 125)
    
    # Assigning a Name to a Attribute (line 128):
    
    # Assigning a Name to a Attribute (line 128):
    # Getting the type of 'old_dist' (line 128)
    old_dist_29164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 49), 'old_dist')
    # Getting the type of 'distutils' (line 128)
    distutils_29165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'distutils')
    # Obtaining the member 'core' of a type (line 128)
    core_29166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), distutils_29165, 'core')
    # Setting the type of the member '_setup_distribution' of a type (line 128)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), core_29166, '_setup_distribution', old_dist_29164)
    
    # Assigning a Name to a Attribute (line 129):
    
    # Assigning a Name to a Attribute (line 129):
    # Getting the type of 'old_stop' (line 129)
    old_stop_29167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 47), 'old_stop')
    # Getting the type of 'distutils' (line 129)
    distutils_29168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'distutils')
    # Obtaining the member 'core' of a type (line 129)
    core_29169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), distutils_29168, 'core')
    # Setting the type of the member '_setup_stop_after' of a type (line 129)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), core_29169, '_setup_stop_after', old_stop_29167)
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'dist' (line 130)
    dist_29170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), 'dist')
    # Obtaining the member 'help' of a type (line 130)
    help_29171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 11), dist_29170, 'help')
    
    
    # Call to _command_line_ok(...): (line 130)
    # Processing the call keyword arguments (line 130)
    kwargs_29173 = {}
    # Getting the type of '_command_line_ok' (line 130)
    _command_line_ok_29172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 28), '_command_line_ok', False)
    # Calling _command_line_ok(args, kwargs) (line 130)
    _command_line_ok_call_result_29174 = invoke(stypy.reporting.localization.Localization(__file__, 130, 28), _command_line_ok_29172, *[], **kwargs_29173)
    
    # Applying the 'not' unary operator (line 130)
    result_not__29175 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 24), 'not', _command_line_ok_call_result_29174)
    
    # Applying the binary operator 'or' (line 130)
    result_or_keyword_29176 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 11), 'or', help_29171, result_not__29175)
    
    # Testing the type of an if condition (line 130)
    if_condition_29177 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), result_or_keyword_29176)
    # Assigning a type to the variable 'if_condition_29177' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_29177', if_condition_29177)
    # SSA begins for if statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'dist' (line 132)
    dist_29178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 19), 'dist')
    # Assigning a type to the variable 'stypy_return_type' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'stypy_return_type', dist_29178)
    # SSA join for if statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 135):
    
    # Assigning a Call to a Name (line 135):
    
    # Call to configuration(...): (line 135)
    # Processing the call keyword arguments (line 135)
    kwargs_29180 = {}
    # Getting the type of 'configuration' (line 135)
    configuration_29179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'configuration', False)
    # Calling configuration(args, kwargs) (line 135)
    configuration_call_result_29181 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), configuration_29179, *[], **kwargs_29180)
    
    # Assigning a type to the variable 'config' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'config', configuration_call_result_29181)
    
    # Type idiom detected: calculating its left and rigth part (line 136)
    str_29182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 27), 'str', 'todict')
    # Getting the type of 'config' (line 136)
    config_29183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'config')
    
    (may_be_29184, more_types_in_union_29185) = may_provide_member(str_29182, config_29183)

    if may_be_29184:

        if more_types_in_union_29185:
            # Runtime conditional SSA (line 136)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'config' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'config', remove_not_member_provider_from_union(config_29183, 'todict'))
        
        # Assigning a Call to a Name (line 137):
        
        # Assigning a Call to a Name (line 137):
        
        # Call to todict(...): (line 137)
        # Processing the call keyword arguments (line 137)
        kwargs_29188 = {}
        # Getting the type of 'config' (line 137)
        config_29186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 'config', False)
        # Obtaining the member 'todict' of a type (line 137)
        todict_29187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 21), config_29186, 'todict')
        # Calling todict(args, kwargs) (line 137)
        todict_call_result_29189 = invoke(stypy.reporting.localization.Localization(__file__, 137, 21), todict_29187, *[], **kwargs_29188)
        
        # Assigning a type to the variable 'config' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'config', todict_call_result_29189)

        if more_types_in_union_29185:
            # SSA join for if statement (line 136)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to _dict_append(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'new_attr' (line 138)
    new_attr_29191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 21), 'new_attr', False)
    # Processing the call keyword arguments (line 138)
    # Getting the type of 'config' (line 138)
    config_29192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 33), 'config', False)
    kwargs_29193 = {'config_29192': config_29192}
    # Getting the type of '_dict_append' (line 138)
    _dict_append_29190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), '_dict_append', False)
    # Calling _dict_append(args, kwargs) (line 138)
    _dict_append_call_result_29194 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), _dict_append_29190, *[new_attr_29191], **kwargs_29193)
    
    # SSA join for if statement (line 116)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 141):
    
    # Assigning a List to a Name (line 141):
    
    # Obtaining an instance of the builtin type 'list' (line 141)
    list_29195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 141)
    
    # Assigning a type to the variable 'libraries' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'libraries', list_29195)
    
    
    # Call to get(...): (line 142)
    # Processing the call arguments (line 142)
    str_29198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 28), 'str', 'ext_modules')
    
    # Obtaining an instance of the builtin type 'list' (line 142)
    list_29199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 142)
    
    # Processing the call keyword arguments (line 142)
    kwargs_29200 = {}
    # Getting the type of 'new_attr' (line 142)
    new_attr_29196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'new_attr', False)
    # Obtaining the member 'get' of a type (line 142)
    get_29197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 15), new_attr_29196, 'get')
    # Calling get(args, kwargs) (line 142)
    get_call_result_29201 = invoke(stypy.reporting.localization.Localization(__file__, 142, 15), get_29197, *[str_29198, list_29199], **kwargs_29200)
    
    # Testing the type of a for loop iterable (line 142)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 142, 4), get_call_result_29201)
    # Getting the type of the for loop variable (line 142)
    for_loop_var_29202 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 142, 4), get_call_result_29201)
    # Assigning a type to the variable 'ext' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'ext', for_loop_var_29202)
    # SSA begins for a for statement (line 142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a List to a Name (line 143):
    
    # Assigning a List to a Name (line 143):
    
    # Obtaining an instance of the builtin type 'list' (line 143)
    list_29203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 143)
    
    # Assigning a type to the variable 'new_libraries' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'new_libraries', list_29203)
    
    # Getting the type of 'ext' (line 144)
    ext_29204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'ext')
    # Obtaining the member 'libraries' of a type (line 144)
    libraries_29205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 20), ext_29204, 'libraries')
    # Testing the type of a for loop iterable (line 144)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 144, 8), libraries_29205)
    # Getting the type of the for loop variable (line 144)
    for_loop_var_29206 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 144, 8), libraries_29205)
    # Assigning a type to the variable 'item' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'item', for_loop_var_29206)
    # SSA begins for a for statement (line 144)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to is_sequence(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'item' (line 145)
    item_29208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 27), 'item', False)
    # Processing the call keyword arguments (line 145)
    kwargs_29209 = {}
    # Getting the type of 'is_sequence' (line 145)
    is_sequence_29207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 15), 'is_sequence', False)
    # Calling is_sequence(args, kwargs) (line 145)
    is_sequence_call_result_29210 = invoke(stypy.reporting.localization.Localization(__file__, 145, 15), is_sequence_29207, *[item_29208], **kwargs_29209)
    
    # Testing the type of an if condition (line 145)
    if_condition_29211 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 12), is_sequence_call_result_29210)
    # Assigning a type to the variable 'if_condition_29211' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'if_condition_29211', if_condition_29211)
    # SSA begins for if statement (line 145)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Tuple (line 146):
    
    # Assigning a Subscript to a Name (line 146):
    
    # Obtaining the type of the subscript
    int_29212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 16), 'int')
    # Getting the type of 'item' (line 146)
    item_29213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 39), 'item')
    # Obtaining the member '__getitem__' of a type (line 146)
    getitem___29214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), item_29213, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 146)
    subscript_call_result_29215 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), getitem___29214, int_29212)
    
    # Assigning a type to the variable 'tuple_var_assignment_28864' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'tuple_var_assignment_28864', subscript_call_result_29215)
    
    # Assigning a Subscript to a Name (line 146):
    
    # Obtaining the type of the subscript
    int_29216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 16), 'int')
    # Getting the type of 'item' (line 146)
    item_29217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 39), 'item')
    # Obtaining the member '__getitem__' of a type (line 146)
    getitem___29218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), item_29217, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 146)
    subscript_call_result_29219 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), getitem___29218, int_29216)
    
    # Assigning a type to the variable 'tuple_var_assignment_28865' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'tuple_var_assignment_28865', subscript_call_result_29219)
    
    # Assigning a Name to a Name (line 146):
    # Getting the type of 'tuple_var_assignment_28864' (line 146)
    tuple_var_assignment_28864_29220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'tuple_var_assignment_28864')
    # Assigning a type to the variable 'lib_name' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'lib_name', tuple_var_assignment_28864_29220)
    
    # Assigning a Name to a Name (line 146):
    # Getting the type of 'tuple_var_assignment_28865' (line 146)
    tuple_var_assignment_28865_29221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'tuple_var_assignment_28865')
    # Assigning a type to the variable 'build_info' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'build_info', tuple_var_assignment_28865_29221)
    
    # Call to _check_append_ext_library(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'libraries' (line 147)
    libraries_29223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 42), 'libraries', False)
    # Getting the type of 'lib_name' (line 147)
    lib_name_29224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 53), 'lib_name', False)
    # Getting the type of 'build_info' (line 147)
    build_info_29225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 63), 'build_info', False)
    # Processing the call keyword arguments (line 147)
    kwargs_29226 = {}
    # Getting the type of '_check_append_ext_library' (line 147)
    _check_append_ext_library_29222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 16), '_check_append_ext_library', False)
    # Calling _check_append_ext_library(args, kwargs) (line 147)
    _check_append_ext_library_call_result_29227 = invoke(stypy.reporting.localization.Localization(__file__, 147, 16), _check_append_ext_library_29222, *[libraries_29223, lib_name_29224, build_info_29225], **kwargs_29226)
    
    
    # Call to append(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'lib_name' (line 148)
    lib_name_29230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 37), 'lib_name', False)
    # Processing the call keyword arguments (line 148)
    kwargs_29231 = {}
    # Getting the type of 'new_libraries' (line 148)
    new_libraries_29228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'new_libraries', False)
    # Obtaining the member 'append' of a type (line 148)
    append_29229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 16), new_libraries_29228, 'append')
    # Calling append(args, kwargs) (line 148)
    append_call_result_29232 = invoke(stypy.reporting.localization.Localization(__file__, 148, 16), append_29229, *[lib_name_29230], **kwargs_29231)
    
    # SSA branch for the else part of an if statement (line 145)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to is_string(...): (line 149)
    # Processing the call arguments (line 149)
    # Getting the type of 'item' (line 149)
    item_29234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 27), 'item', False)
    # Processing the call keyword arguments (line 149)
    kwargs_29235 = {}
    # Getting the type of 'is_string' (line 149)
    is_string_29233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), 'is_string', False)
    # Calling is_string(args, kwargs) (line 149)
    is_string_call_result_29236 = invoke(stypy.reporting.localization.Localization(__file__, 149, 17), is_string_29233, *[item_29234], **kwargs_29235)
    
    # Testing the type of an if condition (line 149)
    if_condition_29237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 17), is_string_call_result_29236)
    # Assigning a type to the variable 'if_condition_29237' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), 'if_condition_29237', if_condition_29237)
    # SSA begins for if statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'item' (line 150)
    item_29240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 37), 'item', False)
    # Processing the call keyword arguments (line 150)
    kwargs_29241 = {}
    # Getting the type of 'new_libraries' (line 150)
    new_libraries_29238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'new_libraries', False)
    # Obtaining the member 'append' of a type (line 150)
    append_29239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), new_libraries_29238, 'append')
    # Calling append(args, kwargs) (line 150)
    append_call_result_29242 = invoke(stypy.reporting.localization.Localization(__file__, 150, 16), append_29239, *[item_29240], **kwargs_29241)
    
    # SSA branch for the else part of an if statement (line 149)
    module_type_store.open_ssa_branch('else')
    
    # Call to TypeError(...): (line 152)
    # Processing the call arguments (line 152)
    str_29244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 32), 'str', 'invalid description of extension module library %r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 153)
    tuple_29245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 48), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 153)
    # Adding element type (line 153)
    # Getting the type of 'item' (line 153)
    item_29246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 48), 'item', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 48), tuple_29245, item_29246)
    
    # Applying the binary operator '%' (line 152)
    result_mod_29247 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 32), '%', str_29244, tuple_29245)
    
    # Processing the call keyword arguments (line 152)
    kwargs_29248 = {}
    # Getting the type of 'TypeError' (line 152)
    TypeError_29243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 22), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 152)
    TypeError_call_result_29249 = invoke(stypy.reporting.localization.Localization(__file__, 152, 22), TypeError_29243, *[result_mod_29247], **kwargs_29248)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 152, 16), TypeError_call_result_29249, 'raise parameter', BaseException)
    # SSA join for if statement (line 149)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 145)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Attribute (line 154):
    
    # Assigning a Name to a Attribute (line 154):
    # Getting the type of 'new_libraries' (line 154)
    new_libraries_29250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 24), 'new_libraries')
    # Getting the type of 'ext' (line 154)
    ext_29251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'ext')
    # Setting the type of the member 'libraries' of a type (line 154)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), ext_29251, 'libraries', new_libraries_29250)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'libraries' (line 155)
    libraries_29252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 7), 'libraries')
    # Testing the type of an if condition (line 155)
    if_condition_29253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 4), libraries_29252)
    # Assigning a type to the variable 'if_condition_29253' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'if_condition_29253', if_condition_29253)
    # SSA begins for if statement (line 155)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    str_29254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 11), 'str', 'libraries')
    # Getting the type of 'new_attr' (line 156)
    new_attr_29255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 30), 'new_attr')
    # Applying the binary operator 'notin' (line 156)
    result_contains_29256 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 11), 'notin', str_29254, new_attr_29255)
    
    # Testing the type of an if condition (line 156)
    if_condition_29257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 8), result_contains_29256)
    # Assigning a type to the variable 'if_condition_29257' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'if_condition_29257', if_condition_29257)
    # SSA begins for if statement (line 156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Subscript (line 157):
    
    # Assigning a List to a Subscript (line 157):
    
    # Obtaining an instance of the builtin type 'list' (line 157)
    list_29258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 157)
    
    # Getting the type of 'new_attr' (line 157)
    new_attr_29259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'new_attr')
    str_29260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 21), 'str', 'libraries')
    # Storing an element on a container (line 157)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 12), new_attr_29259, (str_29260, list_29258))
    # SSA join for if statement (line 156)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'libraries' (line 158)
    libraries_29261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 20), 'libraries')
    # Testing the type of a for loop iterable (line 158)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 158, 8), libraries_29261)
    # Getting the type of the for loop variable (line 158)
    for_loop_var_29262 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 158, 8), libraries_29261)
    # Assigning a type to the variable 'item' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'item', for_loop_var_29262)
    # SSA begins for a for statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to _check_append_library(...): (line 159)
    # Processing the call arguments (line 159)
    
    # Obtaining the type of the subscript
    str_29264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 43), 'str', 'libraries')
    # Getting the type of 'new_attr' (line 159)
    new_attr_29265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 34), 'new_attr', False)
    # Obtaining the member '__getitem__' of a type (line 159)
    getitem___29266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 34), new_attr_29265, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 159)
    subscript_call_result_29267 = invoke(stypy.reporting.localization.Localization(__file__, 159, 34), getitem___29266, str_29264)
    
    # Getting the type of 'item' (line 159)
    item_29268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 57), 'item', False)
    # Processing the call keyword arguments (line 159)
    kwargs_29269 = {}
    # Getting the type of '_check_append_library' (line 159)
    _check_append_library_29263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), '_check_append_library', False)
    # Calling _check_append_library(args, kwargs) (line 159)
    _check_append_library_call_result_29270 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), _check_append_library_29263, *[subscript_call_result_29267, item_29268], **kwargs_29269)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 155)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    str_29271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 8), 'str', 'ext_modules')
    # Getting the type of 'new_attr' (line 162)
    new_attr_29272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), 'new_attr')
    # Applying the binary operator 'in' (line 162)
    result_contains_29273 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 8), 'in', str_29271, new_attr_29272)
    
    
    str_29274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 37), 'str', 'libraries')
    # Getting the type of 'new_attr' (line 162)
    new_attr_29275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 52), 'new_attr')
    # Applying the binary operator 'in' (line 162)
    result_contains_29276 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 37), 'in', str_29274, new_attr_29275)
    
    # Applying the binary operator 'or' (line 162)
    result_or_keyword_29277 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 8), 'or', result_contains_29273, result_contains_29276)
    
    
    str_29278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 11), 'str', 'headers')
    # Getting the type of 'new_attr' (line 163)
    new_attr_29279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 28), 'new_attr')
    # Applying the binary operator 'notin' (line 163)
    result_contains_29280 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 11), 'notin', str_29278, new_attr_29279)
    
    # Applying the binary operator 'and' (line 162)
    result_and_keyword_29281 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 7), 'and', result_or_keyword_29277, result_contains_29280)
    
    # Testing the type of an if condition (line 162)
    if_condition_29282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 4), result_and_keyword_29281)
    # Assigning a type to the variable 'if_condition_29282' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'if_condition_29282', if_condition_29282)
    # SSA begins for if statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Subscript (line 164):
    
    # Assigning a List to a Subscript (line 164):
    
    # Obtaining an instance of the builtin type 'list' (line 164)
    list_29283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 164)
    
    # Getting the type of 'new_attr' (line 164)
    new_attr_29284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'new_attr')
    str_29285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 17), 'str', 'headers')
    # Storing an element on a container (line 164)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 8), new_attr_29284, (str_29285, list_29283))
    # SSA join for if statement (line 162)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 167):
    
    # Assigning a Name to a Subscript (line 167):
    # Getting the type of 'NumpyDistribution' (line 167)
    NumpyDistribution_29286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'NumpyDistribution')
    # Getting the type of 'new_attr' (line 167)
    new_attr_29287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'new_attr')
    str_29288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 13), 'str', 'distclass')
    # Storing an element on a container (line 167)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 4), new_attr_29287, (str_29288, NumpyDistribution_29286))
    
    # Call to old_setup(...): (line 169)
    # Processing the call keyword arguments (line 169)
    # Getting the type of 'new_attr' (line 169)
    new_attr_29290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 'new_attr', False)
    kwargs_29291 = {'new_attr_29290': new_attr_29290}
    # Getting the type of 'old_setup' (line 169)
    old_setup_29289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 11), 'old_setup', False)
    # Calling old_setup(args, kwargs) (line 169)
    old_setup_call_result_29292 = invoke(stypy.reporting.localization.Localization(__file__, 169, 11), old_setup_29289, *[], **kwargs_29291)
    
    # Assigning a type to the variable 'stypy_return_type' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'stypy_return_type', old_setup_call_result_29292)
    
    # ################# End of 'setup(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'setup' in the type store
    # Getting the type of 'stypy_return_type' (line 107)
    stypy_return_type_29293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29293)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'setup'
    return stypy_return_type_29293

# Assigning a type to the variable 'setup' (line 107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'setup', setup)

@norecursion
def _check_append_library(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_check_append_library'
    module_type_store = module_type_store.open_function_context('_check_append_library', 171, 0, False)
    
    # Passed parameters checking function
    _check_append_library.stypy_localization = localization
    _check_append_library.stypy_type_of_self = None
    _check_append_library.stypy_type_store = module_type_store
    _check_append_library.stypy_function_name = '_check_append_library'
    _check_append_library.stypy_param_names_list = ['libraries', 'item']
    _check_append_library.stypy_varargs_param_name = None
    _check_append_library.stypy_kwargs_param_name = None
    _check_append_library.stypy_call_defaults = defaults
    _check_append_library.stypy_call_varargs = varargs
    _check_append_library.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_append_library', ['libraries', 'item'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_append_library', localization, ['libraries', 'item'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_append_library(...)' code ##################

    
    # Getting the type of 'libraries' (line 172)
    libraries_29294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 19), 'libraries')
    # Testing the type of a for loop iterable (line 172)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 172, 4), libraries_29294)
    # Getting the type of the for loop variable (line 172)
    for_loop_var_29295 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 172, 4), libraries_29294)
    # Assigning a type to the variable 'libitem' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'libitem', for_loop_var_29295)
    # SSA begins for a for statement (line 172)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to is_sequence(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'libitem' (line 173)
    libitem_29297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 23), 'libitem', False)
    # Processing the call keyword arguments (line 173)
    kwargs_29298 = {}
    # Getting the type of 'is_sequence' (line 173)
    is_sequence_29296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'is_sequence', False)
    # Calling is_sequence(args, kwargs) (line 173)
    is_sequence_call_result_29299 = invoke(stypy.reporting.localization.Localization(__file__, 173, 11), is_sequence_29296, *[libitem_29297], **kwargs_29298)
    
    # Testing the type of an if condition (line 173)
    if_condition_29300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 8), is_sequence_call_result_29299)
    # Assigning a type to the variable 'if_condition_29300' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'if_condition_29300', if_condition_29300)
    # SSA begins for if statement (line 173)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to is_sequence(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'item' (line 174)
    item_29302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'item', False)
    # Processing the call keyword arguments (line 174)
    kwargs_29303 = {}
    # Getting the type of 'is_sequence' (line 174)
    is_sequence_29301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), 'is_sequence', False)
    # Calling is_sequence(args, kwargs) (line 174)
    is_sequence_call_result_29304 = invoke(stypy.reporting.localization.Localization(__file__, 174, 15), is_sequence_29301, *[item_29302], **kwargs_29303)
    
    # Testing the type of an if condition (line 174)
    if_condition_29305 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 12), is_sequence_call_result_29304)
    # Assigning a type to the variable 'if_condition_29305' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'if_condition_29305', if_condition_29305)
    # SSA begins for if statement (line 174)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_29306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 24), 'int')
    # Getting the type of 'item' (line 175)
    item_29307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 19), 'item')
    # Obtaining the member '__getitem__' of a type (line 175)
    getitem___29308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 19), item_29307, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 175)
    subscript_call_result_29309 = invoke(stypy.reporting.localization.Localization(__file__, 175, 19), getitem___29308, int_29306)
    
    
    # Obtaining the type of the subscript
    int_29310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 36), 'int')
    # Getting the type of 'libitem' (line 175)
    libitem_29311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 28), 'libitem')
    # Obtaining the member '__getitem__' of a type (line 175)
    getitem___29312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 28), libitem_29311, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 175)
    subscript_call_result_29313 = invoke(stypy.reporting.localization.Localization(__file__, 175, 28), getitem___29312, int_29310)
    
    # Applying the binary operator '==' (line 175)
    result_eq_29314 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 19), '==', subscript_call_result_29309, subscript_call_result_29313)
    
    # Testing the type of an if condition (line 175)
    if_condition_29315 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 16), result_eq_29314)
    # Assigning a type to the variable 'if_condition_29315' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 16), 'if_condition_29315', if_condition_29315)
    # SSA begins for if statement (line 175)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_29316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'int')
    # Getting the type of 'item' (line 176)
    item_29317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 23), 'item')
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___29318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 23), item_29317, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_29319 = invoke(stypy.reporting.localization.Localization(__file__, 176, 23), getitem___29318, int_29316)
    
    
    # Obtaining the type of the subscript
    int_29320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 42), 'int')
    # Getting the type of 'libitem' (line 176)
    libitem_29321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 34), 'libitem')
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___29322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 34), libitem_29321, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_29323 = invoke(stypy.reporting.localization.Localization(__file__, 176, 34), getitem___29322, int_29320)
    
    # Applying the binary operator 'is' (line 176)
    result_is__29324 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 23), 'is', subscript_call_result_29319, subscript_call_result_29323)
    
    # Testing the type of an if condition (line 176)
    if_condition_29325 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 20), result_is__29324)
    # Assigning a type to the variable 'if_condition_29325' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 20), 'if_condition_29325', if_condition_29325)
    # SSA begins for if statement (line 176)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 176)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to warn(...): (line 178)
    # Processing the call arguments (line 178)
    str_29328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 34), 'str', '[0] libraries list contains %r with different build_info')
    
    # Obtaining an instance of the builtin type 'tuple' (line 179)
    tuple_29329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 61), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 179)
    # Adding element type (line 179)
    
    # Obtaining the type of the subscript
    int_29330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 66), 'int')
    # Getting the type of 'item' (line 179)
    item_29331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 61), 'item', False)
    # Obtaining the member '__getitem__' of a type (line 179)
    getitem___29332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 61), item_29331, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 179)
    subscript_call_result_29333 = invoke(stypy.reporting.localization.Localization(__file__, 179, 61), getitem___29332, int_29330)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 61), tuple_29329, subscript_call_result_29333)
    
    # Applying the binary operator '%' (line 178)
    result_mod_29334 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 34), '%', str_29328, tuple_29329)
    
    # Processing the call keyword arguments (line 178)
    kwargs_29335 = {}
    # Getting the type of 'warnings' (line 178)
    warnings_29326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 20), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 178)
    warn_29327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 20), warnings_29326, 'warn')
    # Calling warn(args, kwargs) (line 178)
    warn_call_result_29336 = invoke(stypy.reporting.localization.Localization(__file__, 178, 20), warn_29327, *[result_mod_29334], **kwargs_29335)
    
    # SSA join for if statement (line 175)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 174)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'item' (line 182)
    item_29337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 19), 'item')
    
    # Obtaining the type of the subscript
    int_29338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 33), 'int')
    # Getting the type of 'libitem' (line 182)
    libitem_29339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 25), 'libitem')
    # Obtaining the member '__getitem__' of a type (line 182)
    getitem___29340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 25), libitem_29339, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 182)
    subscript_call_result_29341 = invoke(stypy.reporting.localization.Localization(__file__, 182, 25), getitem___29340, int_29338)
    
    # Applying the binary operator '==' (line 182)
    result_eq_29342 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 19), '==', item_29337, subscript_call_result_29341)
    
    # Testing the type of an if condition (line 182)
    if_condition_29343 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 16), result_eq_29342)
    # Assigning a type to the variable 'if_condition_29343' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'if_condition_29343', if_condition_29343)
    # SSA begins for if statement (line 182)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 183)
    # Processing the call arguments (line 183)
    str_29346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 34), 'str', '[1] libraries list contains %r with no build_info')
    
    # Obtaining an instance of the builtin type 'tuple' (line 184)
    tuple_29347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 184)
    # Adding element type (line 184)
    
    # Obtaining the type of the subscript
    int_29348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 59), 'int')
    # Getting the type of 'item' (line 184)
    item_29349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 54), 'item', False)
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___29350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 54), item_29349, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_29351 = invoke(stypy.reporting.localization.Localization(__file__, 184, 54), getitem___29350, int_29348)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 54), tuple_29347, subscript_call_result_29351)
    
    # Applying the binary operator '%' (line 183)
    result_mod_29352 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 34), '%', str_29346, tuple_29347)
    
    # Processing the call keyword arguments (line 183)
    kwargs_29353 = {}
    # Getting the type of 'warnings' (line 183)
    warnings_29344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 183)
    warn_29345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 20), warnings_29344, 'warn')
    # Calling warn(args, kwargs) (line 183)
    warn_call_result_29354 = invoke(stypy.reporting.localization.Localization(__file__, 183, 20), warn_29345, *[result_mod_29352], **kwargs_29353)
    
    # SSA join for if statement (line 182)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 174)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 173)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to is_sequence(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'item' (line 187)
    item_29356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 27), 'item', False)
    # Processing the call keyword arguments (line 187)
    kwargs_29357 = {}
    # Getting the type of 'is_sequence' (line 187)
    is_sequence_29355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'is_sequence', False)
    # Calling is_sequence(args, kwargs) (line 187)
    is_sequence_call_result_29358 = invoke(stypy.reporting.localization.Localization(__file__, 187, 15), is_sequence_29355, *[item_29356], **kwargs_29357)
    
    # Testing the type of an if condition (line 187)
    if_condition_29359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 12), is_sequence_call_result_29358)
    # Assigning a type to the variable 'if_condition_29359' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'if_condition_29359', if_condition_29359)
    # SSA begins for if statement (line 187)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_29360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 24), 'int')
    # Getting the type of 'item' (line 188)
    item_29361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 19), 'item')
    # Obtaining the member '__getitem__' of a type (line 188)
    getitem___29362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 19), item_29361, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 188)
    subscript_call_result_29363 = invoke(stypy.reporting.localization.Localization(__file__, 188, 19), getitem___29362, int_29360)
    
    # Getting the type of 'libitem' (line 188)
    libitem_29364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 28), 'libitem')
    # Applying the binary operator '==' (line 188)
    result_eq_29365 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 19), '==', subscript_call_result_29363, libitem_29364)
    
    # Testing the type of an if condition (line 188)
    if_condition_29366 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 16), result_eq_29365)
    # Assigning a type to the variable 'if_condition_29366' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'if_condition_29366', if_condition_29366)
    # SSA begins for if statement (line 188)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 189)
    # Processing the call arguments (line 189)
    str_29369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 34), 'str', '[2] libraries list contains %r with no build_info')
    
    # Obtaining an instance of the builtin type 'tuple' (line 190)
    tuple_29370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 190)
    # Adding element type (line 190)
    
    # Obtaining the type of the subscript
    int_29371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 59), 'int')
    # Getting the type of 'item' (line 190)
    item_29372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 54), 'item', False)
    # Obtaining the member '__getitem__' of a type (line 190)
    getitem___29373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 54), item_29372, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 190)
    subscript_call_result_29374 = invoke(stypy.reporting.localization.Localization(__file__, 190, 54), getitem___29373, int_29371)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 54), tuple_29370, subscript_call_result_29374)
    
    # Applying the binary operator '%' (line 189)
    result_mod_29375 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 34), '%', str_29369, tuple_29370)
    
    # Processing the call keyword arguments (line 189)
    kwargs_29376 = {}
    # Getting the type of 'warnings' (line 189)
    warnings_29367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 189)
    warn_29368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 20), warnings_29367, 'warn')
    # Calling warn(args, kwargs) (line 189)
    warn_call_result_29377 = invoke(stypy.reporting.localization.Localization(__file__, 189, 20), warn_29368, *[result_mod_29375], **kwargs_29376)
    
    # SSA join for if statement (line 188)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 187)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'item' (line 193)
    item_29378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 19), 'item')
    # Getting the type of 'libitem' (line 193)
    libitem_29379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 25), 'libitem')
    # Applying the binary operator '==' (line 193)
    result_eq_29380 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 19), '==', item_29378, libitem_29379)
    
    # Testing the type of an if condition (line 193)
    if_condition_29381 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 16), result_eq_29380)
    # Assigning a type to the variable 'if_condition_29381' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'if_condition_29381', if_condition_29381)
    # SSA begins for if statement (line 193)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 20), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 193)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 187)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 173)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'item' (line 195)
    item_29384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 21), 'item', False)
    # Processing the call keyword arguments (line 195)
    kwargs_29385 = {}
    # Getting the type of 'libraries' (line 195)
    libraries_29382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'libraries', False)
    # Obtaining the member 'append' of a type (line 195)
    append_29383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 4), libraries_29382, 'append')
    # Calling append(args, kwargs) (line 195)
    append_call_result_29386 = invoke(stypy.reporting.localization.Localization(__file__, 195, 4), append_29383, *[item_29384], **kwargs_29385)
    
    
    # ################# End of '_check_append_library(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_append_library' in the type store
    # Getting the type of 'stypy_return_type' (line 171)
    stypy_return_type_29387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29387)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_append_library'
    return stypy_return_type_29387

# Assigning a type to the variable '_check_append_library' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), '_check_append_library', _check_append_library)

@norecursion
def _check_append_ext_library(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_check_append_ext_library'
    module_type_store = module_type_store.open_function_context('_check_append_ext_library', 197, 0, False)
    
    # Passed parameters checking function
    _check_append_ext_library.stypy_localization = localization
    _check_append_ext_library.stypy_type_of_self = None
    _check_append_ext_library.stypy_type_store = module_type_store
    _check_append_ext_library.stypy_function_name = '_check_append_ext_library'
    _check_append_ext_library.stypy_param_names_list = ['libraries', 'lib_name', 'build_info']
    _check_append_ext_library.stypy_varargs_param_name = None
    _check_append_ext_library.stypy_kwargs_param_name = None
    _check_append_ext_library.stypy_call_defaults = defaults
    _check_append_ext_library.stypy_call_varargs = varargs
    _check_append_ext_library.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_append_ext_library', ['libraries', 'lib_name', 'build_info'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_append_ext_library', localization, ['libraries', 'lib_name', 'build_info'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_append_ext_library(...)' code ##################

    
    # Getting the type of 'libraries' (line 198)
    libraries_29388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'libraries')
    # Testing the type of a for loop iterable (line 198)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 198, 4), libraries_29388)
    # Getting the type of the for loop variable (line 198)
    for_loop_var_29389 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 198, 4), libraries_29388)
    # Assigning a type to the variable 'item' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'item', for_loop_var_29389)
    # SSA begins for a for statement (line 198)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to is_sequence(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'item' (line 199)
    item_29391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 23), 'item', False)
    # Processing the call keyword arguments (line 199)
    kwargs_29392 = {}
    # Getting the type of 'is_sequence' (line 199)
    is_sequence_29390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'is_sequence', False)
    # Calling is_sequence(args, kwargs) (line 199)
    is_sequence_call_result_29393 = invoke(stypy.reporting.localization.Localization(__file__, 199, 11), is_sequence_29390, *[item_29391], **kwargs_29392)
    
    # Testing the type of an if condition (line 199)
    if_condition_29394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 8), is_sequence_call_result_29393)
    # Assigning a type to the variable 'if_condition_29394' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'if_condition_29394', if_condition_29394)
    # SSA begins for if statement (line 199)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_29395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 20), 'int')
    # Getting the type of 'item' (line 200)
    item_29396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'item')
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___29397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 15), item_29396, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_29398 = invoke(stypy.reporting.localization.Localization(__file__, 200, 15), getitem___29397, int_29395)
    
    # Getting the type of 'lib_name' (line 200)
    lib_name_29399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 24), 'lib_name')
    # Applying the binary operator '==' (line 200)
    result_eq_29400 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 15), '==', subscript_call_result_29398, lib_name_29399)
    
    # Testing the type of an if condition (line 200)
    if_condition_29401 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 12), result_eq_29400)
    # Assigning a type to the variable 'if_condition_29401' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'if_condition_29401', if_condition_29401)
    # SSA begins for if statement (line 200)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_29402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 24), 'int')
    # Getting the type of 'item' (line 201)
    item_29403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 'item')
    # Obtaining the member '__getitem__' of a type (line 201)
    getitem___29404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 19), item_29403, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 201)
    subscript_call_result_29405 = invoke(stypy.reporting.localization.Localization(__file__, 201, 19), getitem___29404, int_29402)
    
    # Getting the type of 'build_info' (line 201)
    build_info_29406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 30), 'build_info')
    # Applying the binary operator 'is' (line 201)
    result_is__29407 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 19), 'is', subscript_call_result_29405, build_info_29406)
    
    # Testing the type of an if condition (line 201)
    if_condition_29408 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 16), result_is__29407)
    # Assigning a type to the variable 'if_condition_29408' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'if_condition_29408', if_condition_29408)
    # SSA begins for if statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 201)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to warn(...): (line 203)
    # Processing the call arguments (line 203)
    str_29411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 30), 'str', '[3] libraries list contains %r with different build_info')
    
    # Obtaining an instance of the builtin type 'tuple' (line 204)
    tuple_29412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 204)
    # Adding element type (line 204)
    # Getting the type of 'lib_name' (line 204)
    lib_name_29413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 57), 'lib_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 57), tuple_29412, lib_name_29413)
    
    # Applying the binary operator '%' (line 203)
    result_mod_29414 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 30), '%', str_29411, tuple_29412)
    
    # Processing the call keyword arguments (line 203)
    kwargs_29415 = {}
    # Getting the type of 'warnings' (line 203)
    warnings_29409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 203)
    warn_29410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 16), warnings_29409, 'warn')
    # Calling warn(args, kwargs) (line 203)
    warn_call_result_29416 = invoke(stypy.reporting.localization.Localization(__file__, 203, 16), warn_29410, *[result_mod_29414], **kwargs_29415)
    
    # SSA join for if statement (line 200)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 199)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'item' (line 206)
    item_29417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 13), 'item')
    # Getting the type of 'lib_name' (line 206)
    lib_name_29418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 19), 'lib_name')
    # Applying the binary operator '==' (line 206)
    result_eq_29419 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 13), '==', item_29417, lib_name_29418)
    
    # Testing the type of an if condition (line 206)
    if_condition_29420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 13), result_eq_29419)
    # Assigning a type to the variable 'if_condition_29420' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 13), 'if_condition_29420', if_condition_29420)
    # SSA begins for if statement (line 206)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 207)
    # Processing the call arguments (line 207)
    str_29423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 26), 'str', '[4] libraries list contains %r with no build_info')
    
    # Obtaining an instance of the builtin type 'tuple' (line 208)
    tuple_29424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 46), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 208)
    # Adding element type (line 208)
    # Getting the type of 'lib_name' (line 208)
    lib_name_29425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 46), 'lib_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 208, 46), tuple_29424, lib_name_29425)
    
    # Applying the binary operator '%' (line 207)
    result_mod_29426 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 26), '%', str_29423, tuple_29424)
    
    # Processing the call keyword arguments (line 207)
    kwargs_29427 = {}
    # Getting the type of 'warnings' (line 207)
    warnings_29421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 207)
    warn_29422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), warnings_29421, 'warn')
    # Calling warn(args, kwargs) (line 207)
    warn_call_result_29428 = invoke(stypy.reporting.localization.Localization(__file__, 207, 12), warn_29422, *[result_mod_29426], **kwargs_29427)
    
    # SSA join for if statement (line 206)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 199)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 210)
    # Processing the call arguments (line 210)
    
    # Obtaining an instance of the builtin type 'tuple' (line 210)
    tuple_29431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 210)
    # Adding element type (line 210)
    # Getting the type of 'lib_name' (line 210)
    lib_name_29432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 22), 'lib_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 22), tuple_29431, lib_name_29432)
    # Adding element type (line 210)
    # Getting the type of 'build_info' (line 210)
    build_info_29433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 32), 'build_info', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 22), tuple_29431, build_info_29433)
    
    # Processing the call keyword arguments (line 210)
    kwargs_29434 = {}
    # Getting the type of 'libraries' (line 210)
    libraries_29429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'libraries', False)
    # Obtaining the member 'append' of a type (line 210)
    append_29430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 4), libraries_29429, 'append')
    # Calling append(args, kwargs) (line 210)
    append_call_result_29435 = invoke(stypy.reporting.localization.Localization(__file__, 210, 4), append_29430, *[tuple_29431], **kwargs_29434)
    
    
    # ################# End of '_check_append_ext_library(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_append_ext_library' in the type store
    # Getting the type of 'stypy_return_type' (line 197)
    stypy_return_type_29436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_29436)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_append_ext_library'
    return stypy_return_type_29436

# Assigning a type to the variable '_check_append_ext_library' (line 197)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 0), '_check_append_ext_library', _check_append_ext_library)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
