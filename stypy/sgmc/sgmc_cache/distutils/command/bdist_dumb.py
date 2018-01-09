
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.bdist_dumb
2: 
3: Implements the Distutils 'bdist_dumb' command (create a "dumb" built
4: distribution -- i.e., just an archive to be unpacked under $prefix or
5: $exec_prefix).'''
6: 
7: __revision__ = "$Id$"
8: 
9: import os
10: 
11: from sysconfig import get_python_version
12: 
13: from distutils.util import get_platform
14: from distutils.core import Command
15: from distutils.dir_util import remove_tree, ensure_relative
16: from distutils.errors import DistutilsPlatformError
17: from distutils import log
18: 
19: class bdist_dumb (Command):
20: 
21:     description = 'create a "dumb" built distribution'
22: 
23:     user_options = [('bdist-dir=', 'd',
24:                      "temporary directory for creating the distribution"),
25:                     ('plat-name=', 'p',
26:                      "platform name to embed in generated filenames "
27:                      "(default: %s)" % get_platform()),
28:                     ('format=', 'f',
29:                      "archive format to create (tar, ztar, gztar, zip)"),
30:                     ('keep-temp', 'k',
31:                      "keep the pseudo-installation tree around after " +
32:                      "creating the distribution archive"),
33:                     ('dist-dir=', 'd',
34:                      "directory to put final built distributions in"),
35:                     ('skip-build', None,
36:                      "skip rebuilding everything (for testing/debugging)"),
37:                     ('relative', None,
38:                      "build the archive using relative paths"
39:                      "(default: false)"),
40:                     ('owner=', 'u',
41:                      "Owner name used when creating a tar file"
42:                      " [default: current user]"),
43:                     ('group=', 'g',
44:                      "Group name used when creating a tar file"
45:                      " [default: current group]"),
46:                    ]
47: 
48:     boolean_options = ['keep-temp', 'skip-build', 'relative']
49: 
50:     default_format = { 'posix': 'gztar',
51:                        'nt': 'zip',
52:                        'os2': 'zip' }
53: 
54: 
55:     def initialize_options (self):
56:         self.bdist_dir = None
57:         self.plat_name = None
58:         self.format = None
59:         self.keep_temp = 0
60:         self.dist_dir = None
61:         self.skip_build = None
62:         self.relative = 0
63:         self.owner = None
64:         self.group = None
65: 
66:     def finalize_options(self):
67:         if self.bdist_dir is None:
68:             bdist_base = self.get_finalized_command('bdist').bdist_base
69:             self.bdist_dir = os.path.join(bdist_base, 'dumb')
70: 
71:         if self.format is None:
72:             try:
73:                 self.format = self.default_format[os.name]
74:             except KeyError:
75:                 raise DistutilsPlatformError, \
76:                       ("don't know how to create dumb built distributions " +
77:                        "on platform %s") % os.name
78: 
79:         self.set_undefined_options('bdist',
80:                                    ('dist_dir', 'dist_dir'),
81:                                    ('plat_name', 'plat_name'),
82:                                    ('skip_build', 'skip_build'))
83: 
84:     def run(self):
85:         if not self.skip_build:
86:             self.run_command('build')
87: 
88:         install = self.reinitialize_command('install', reinit_subcommands=1)
89:         install.root = self.bdist_dir
90:         install.skip_build = self.skip_build
91:         install.warn_dir = 0
92: 
93:         log.info("installing to %s" % self.bdist_dir)
94:         self.run_command('install')
95: 
96:         # And make an archive relative to the root of the
97:         # pseudo-installation tree.
98:         archive_basename = "%s.%s" % (self.distribution.get_fullname(),
99:                                       self.plat_name)
100: 
101:         # OS/2 objects to any ":" characters in a filename (such as when
102:         # a timestamp is used in a version) so change them to hyphens.
103:         if os.name == "os2":
104:             archive_basename = archive_basename.replace(":", "-")
105: 
106:         pseudoinstall_root = os.path.join(self.dist_dir, archive_basename)
107:         if not self.relative:
108:             archive_root = self.bdist_dir
109:         else:
110:             if (self.distribution.has_ext_modules() and
111:                 (install.install_base != install.install_platbase)):
112:                 raise DistutilsPlatformError, \
113:                       ("can't make a dumb built distribution where "
114:                        "base and platbase are different (%s, %s)"
115:                        % (repr(install.install_base),
116:                           repr(install.install_platbase)))
117:             else:
118:                 archive_root = os.path.join(self.bdist_dir,
119:                                    ensure_relative(install.install_base))
120: 
121:         # Make the archive
122:         filename = self.make_archive(pseudoinstall_root,
123:                                      self.format, root_dir=archive_root,
124:                                      owner=self.owner, group=self.group)
125:         if self.distribution.has_ext_modules():
126:             pyversion = get_python_version()
127:         else:
128:             pyversion = 'any'
129:         self.distribution.dist_files.append(('bdist_dumb', pyversion,
130:                                              filename))
131: 
132:         if not self.keep_temp:
133:             remove_tree(self.bdist_dir, dry_run=self.dry_run)
134: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_11978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', 'distutils.command.bdist_dumb\n\nImplements the Distutils \'bdist_dumb\' command (create a "dumb" built\ndistribution -- i.e., just an archive to be unpacked under $prefix or\n$exec_prefix).')

# Assigning a Str to a Name (line 7):
str_11979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__revision__', str_11979)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import os' statement (line 9)
import os

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from sysconfig import get_python_version' statement (line 11)
try:
    from sysconfig import get_python_version

except:
    get_python_version = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'sysconfig', None, module_type_store, ['get_python_version'], [get_python_version])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.util import get_platform' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_11980 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.util')

if (type(import_11980) is not StypyTypeError):

    if (import_11980 != 'pyd_module'):
        __import__(import_11980)
        sys_modules_11981 = sys.modules[import_11980]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.util', sys_modules_11981.module_type_store, module_type_store, ['get_platform'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_11981, sys_modules_11981.module_type_store, module_type_store)
    else:
        from distutils.util import get_platform

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.util', None, module_type_store, ['get_platform'], [get_platform])

else:
    # Assigning a type to the variable 'distutils.util' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.util', import_11980)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.core import Command' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_11982 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.core')

if (type(import_11982) is not StypyTypeError):

    if (import_11982 != 'pyd_module'):
        __import__(import_11982)
        sys_modules_11983 = sys.modules[import_11982]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.core', sys_modules_11983.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_11983, sys_modules_11983.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.core', import_11982)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.dir_util import remove_tree, ensure_relative' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_11984 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.dir_util')

if (type(import_11984) is not StypyTypeError):

    if (import_11984 != 'pyd_module'):
        __import__(import_11984)
        sys_modules_11985 = sys.modules[import_11984]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.dir_util', sys_modules_11985.module_type_store, module_type_store, ['remove_tree', 'ensure_relative'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_11985, sys_modules_11985.module_type_store, module_type_store)
    else:
        from distutils.dir_util import remove_tree, ensure_relative

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.dir_util', None, module_type_store, ['remove_tree', 'ensure_relative'], [remove_tree, ensure_relative])

else:
    # Assigning a type to the variable 'distutils.dir_util' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.dir_util', import_11984)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils.errors import DistutilsPlatformError' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_11986 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors')

if (type(import_11986) is not StypyTypeError):

    if (import_11986 != 'pyd_module'):
        __import__(import_11986)
        sys_modules_11987 = sys.modules[import_11986]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', sys_modules_11987.module_type_store, module_type_store, ['DistutilsPlatformError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_11987, sys_modules_11987.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsPlatformError

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', None, module_type_store, ['DistutilsPlatformError'], [DistutilsPlatformError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils.errors', import_11986)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 0))

# 'from distutils import log' statement (line 17)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 17, 0), 'distutils', None, module_type_store, ['log'], [log])

# Declaration of the 'bdist_dumb' class
# Getting the type of 'Command' (line 19)
Command_11988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 18), 'Command')

class bdist_dumb(Command_11988, ):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_dumb.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        bdist_dumb.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_dumb.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_dumb.initialize_options.__dict__.__setitem__('stypy_function_name', 'bdist_dumb.initialize_options')
        bdist_dumb.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_dumb.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_dumb.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_dumb.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_dumb.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_dumb.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_dumb.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_dumb.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 56):
        # Getting the type of 'None' (line 56)
        None_11989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'None')
        # Getting the type of 'self' (line 56)
        self_11990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member 'bdist_dir' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_11990, 'bdist_dir', None_11989)
        
        # Assigning a Name to a Attribute (line 57):
        # Getting the type of 'None' (line 57)
        None_11991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'None')
        # Getting the type of 'self' (line 57)
        self_11992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member 'plat_name' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_11992, 'plat_name', None_11991)
        
        # Assigning a Name to a Attribute (line 58):
        # Getting the type of 'None' (line 58)
        None_11993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 22), 'None')
        # Getting the type of 'self' (line 58)
        self_11994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self')
        # Setting the type of the member 'format' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_11994, 'format', None_11993)
        
        # Assigning a Num to a Attribute (line 59):
        int_11995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 25), 'int')
        # Getting the type of 'self' (line 59)
        self_11996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member 'keep_temp' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_11996, 'keep_temp', int_11995)
        
        # Assigning a Name to a Attribute (line 60):
        # Getting the type of 'None' (line 60)
        None_11997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 24), 'None')
        # Getting the type of 'self' (line 60)
        self_11998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self')
        # Setting the type of the member 'dist_dir' of a type (line 60)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_11998, 'dist_dir', None_11997)
        
        # Assigning a Name to a Attribute (line 61):
        # Getting the type of 'None' (line 61)
        None_11999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'None')
        # Getting the type of 'self' (line 61)
        self_12000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
        # Setting the type of the member 'skip_build' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_12000, 'skip_build', None_11999)
        
        # Assigning a Num to a Attribute (line 62):
        int_12001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 24), 'int')
        # Getting the type of 'self' (line 62)
        self_12002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member 'relative' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_12002, 'relative', int_12001)
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'None' (line 63)
        None_12003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 21), 'None')
        # Getting the type of 'self' (line 63)
        self_12004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member 'owner' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_12004, 'owner', None_12003)
        
        # Assigning a Name to a Attribute (line 64):
        # Getting the type of 'None' (line 64)
        None_12005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'None')
        # Getting the type of 'self' (line 64)
        self_12006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Setting the type of the member 'group' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_12006, 'group', None_12005)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_12007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12007)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_12007


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_dumb.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        bdist_dumb.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_dumb.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_dumb.finalize_options.__dict__.__setitem__('stypy_function_name', 'bdist_dumb.finalize_options')
        bdist_dumb.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_dumb.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_dumb.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_dumb.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_dumb.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_dumb.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_dumb.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_dumb.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 67)
        # Getting the type of 'self' (line 67)
        self_12008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'self')
        # Obtaining the member 'bdist_dir' of a type (line 67)
        bdist_dir_12009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 11), self_12008, 'bdist_dir')
        # Getting the type of 'None' (line 67)
        None_12010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 29), 'None')
        
        (may_be_12011, more_types_in_union_12012) = may_be_none(bdist_dir_12009, None_12010)

        if may_be_12011:

            if more_types_in_union_12012:
                # Runtime conditional SSA (line 67)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 68):
            
            # Call to get_finalized_command(...): (line 68)
            # Processing the call arguments (line 68)
            str_12015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 52), 'str', 'bdist')
            # Processing the call keyword arguments (line 68)
            kwargs_12016 = {}
            # Getting the type of 'self' (line 68)
            self_12013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 'self', False)
            # Obtaining the member 'get_finalized_command' of a type (line 68)
            get_finalized_command_12014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 25), self_12013, 'get_finalized_command')
            # Calling get_finalized_command(args, kwargs) (line 68)
            get_finalized_command_call_result_12017 = invoke(stypy.reporting.localization.Localization(__file__, 68, 25), get_finalized_command_12014, *[str_12015], **kwargs_12016)
            
            # Obtaining the member 'bdist_base' of a type (line 68)
            bdist_base_12018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 25), get_finalized_command_call_result_12017, 'bdist_base')
            # Assigning a type to the variable 'bdist_base' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'bdist_base', bdist_base_12018)
            
            # Assigning a Call to a Attribute (line 69):
            
            # Call to join(...): (line 69)
            # Processing the call arguments (line 69)
            # Getting the type of 'bdist_base' (line 69)
            bdist_base_12022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 42), 'bdist_base', False)
            str_12023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 54), 'str', 'dumb')
            # Processing the call keyword arguments (line 69)
            kwargs_12024 = {}
            # Getting the type of 'os' (line 69)
            os_12019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 29), 'os', False)
            # Obtaining the member 'path' of a type (line 69)
            path_12020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 29), os_12019, 'path')
            # Obtaining the member 'join' of a type (line 69)
            join_12021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 29), path_12020, 'join')
            # Calling join(args, kwargs) (line 69)
            join_call_result_12025 = invoke(stypy.reporting.localization.Localization(__file__, 69, 29), join_12021, *[bdist_base_12022, str_12023], **kwargs_12024)
            
            # Getting the type of 'self' (line 69)
            self_12026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'self')
            # Setting the type of the member 'bdist_dir' of a type (line 69)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), self_12026, 'bdist_dir', join_call_result_12025)

            if more_types_in_union_12012:
                # SSA join for if statement (line 67)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 71)
        # Getting the type of 'self' (line 71)
        self_12027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'self')
        # Obtaining the member 'format' of a type (line 71)
        format_12028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 11), self_12027, 'format')
        # Getting the type of 'None' (line 71)
        None_12029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 26), 'None')
        
        (may_be_12030, more_types_in_union_12031) = may_be_none(format_12028, None_12029)

        if may_be_12030:

            if more_types_in_union_12031:
                # Runtime conditional SSA (line 71)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # SSA begins for try-except statement (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Subscript to a Attribute (line 73):
            
            # Obtaining the type of the subscript
            # Getting the type of 'os' (line 73)
            os_12032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 50), 'os')
            # Obtaining the member 'name' of a type (line 73)
            name_12033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 50), os_12032, 'name')
            # Getting the type of 'self' (line 73)
            self_12034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'self')
            # Obtaining the member 'default_format' of a type (line 73)
            default_format_12035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 30), self_12034, 'default_format')
            # Obtaining the member '__getitem__' of a type (line 73)
            getitem___12036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 30), default_format_12035, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 73)
            subscript_call_result_12037 = invoke(stypy.reporting.localization.Localization(__file__, 73, 30), getitem___12036, name_12033)
            
            # Getting the type of 'self' (line 73)
            self_12038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'self')
            # Setting the type of the member 'format' of a type (line 73)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), self_12038, 'format', subscript_call_result_12037)
            # SSA branch for the except part of a try statement (line 72)
            # SSA branch for the except 'KeyError' branch of a try statement (line 72)
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'DistutilsPlatformError' (line 75)
            DistutilsPlatformError_12039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'DistutilsPlatformError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 75, 16), DistutilsPlatformError_12039, 'raise parameter', BaseException)
            # SSA join for try-except statement (line 72)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_12031:
                # SSA join for if statement (line 71)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to set_undefined_options(...): (line 79)
        # Processing the call arguments (line 79)
        str_12042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 35), 'str', 'bdist')
        
        # Obtaining an instance of the builtin type 'tuple' (line 80)
        tuple_12043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 80)
        # Adding element type (line 80)
        str_12044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 36), 'str', 'dist_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 36), tuple_12043, str_12044)
        # Adding element type (line 80)
        str_12045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 48), 'str', 'dist_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 36), tuple_12043, str_12045)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 81)
        tuple_12046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 81)
        # Adding element type (line 81)
        str_12047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 36), 'str', 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 36), tuple_12046, str_12047)
        # Adding element type (line 81)
        str_12048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 49), 'str', 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 36), tuple_12046, str_12048)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 82)
        tuple_12049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 82)
        # Adding element type (line 82)
        str_12050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 36), 'str', 'skip_build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 36), tuple_12049, str_12050)
        # Adding element type (line 82)
        str_12051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 50), 'str', 'skip_build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 36), tuple_12049, str_12051)
        
        # Processing the call keyword arguments (line 79)
        kwargs_12052 = {}
        # Getting the type of 'self' (line 79)
        self_12040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 79)
        set_undefined_options_12041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_12040, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 79)
        set_undefined_options_call_result_12053 = invoke(stypy.reporting.localization.Localization(__file__, 79, 8), set_undefined_options_12041, *[str_12042, tuple_12043, tuple_12046, tuple_12049], **kwargs_12052)
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_12054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12054)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_12054


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist_dumb.run.__dict__.__setitem__('stypy_localization', localization)
        bdist_dumb.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist_dumb.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist_dumb.run.__dict__.__setitem__('stypy_function_name', 'bdist_dumb.run')
        bdist_dumb.run.__dict__.__setitem__('stypy_param_names_list', [])
        bdist_dumb.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist_dumb.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist_dumb.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist_dumb.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist_dumb.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist_dumb.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_dumb.run', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 85)
        self_12055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'self')
        # Obtaining the member 'skip_build' of a type (line 85)
        skip_build_12056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 15), self_12055, 'skip_build')
        # Applying the 'not' unary operator (line 85)
        result_not__12057 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 11), 'not', skip_build_12056)
        
        # Testing the type of an if condition (line 85)
        if_condition_12058 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 8), result_not__12057)
        # Assigning a type to the variable 'if_condition_12058' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'if_condition_12058', if_condition_12058)
        # SSA begins for if statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to run_command(...): (line 86)
        # Processing the call arguments (line 86)
        str_12061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 29), 'str', 'build')
        # Processing the call keyword arguments (line 86)
        kwargs_12062 = {}
        # Getting the type of 'self' (line 86)
        self_12059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'self', False)
        # Obtaining the member 'run_command' of a type (line 86)
        run_command_12060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), self_12059, 'run_command')
        # Calling run_command(args, kwargs) (line 86)
        run_command_call_result_12063 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), run_command_12060, *[str_12061], **kwargs_12062)
        
        # SSA join for if statement (line 85)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 88):
        
        # Call to reinitialize_command(...): (line 88)
        # Processing the call arguments (line 88)
        str_12066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 44), 'str', 'install')
        # Processing the call keyword arguments (line 88)
        int_12067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 74), 'int')
        keyword_12068 = int_12067
        kwargs_12069 = {'reinit_subcommands': keyword_12068}
        # Getting the type of 'self' (line 88)
        self_12064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'self', False)
        # Obtaining the member 'reinitialize_command' of a type (line 88)
        reinitialize_command_12065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 18), self_12064, 'reinitialize_command')
        # Calling reinitialize_command(args, kwargs) (line 88)
        reinitialize_command_call_result_12070 = invoke(stypy.reporting.localization.Localization(__file__, 88, 18), reinitialize_command_12065, *[str_12066], **kwargs_12069)
        
        # Assigning a type to the variable 'install' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'install', reinitialize_command_call_result_12070)
        
        # Assigning a Attribute to a Attribute (line 89):
        # Getting the type of 'self' (line 89)
        self_12071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 23), 'self')
        # Obtaining the member 'bdist_dir' of a type (line 89)
        bdist_dir_12072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 23), self_12071, 'bdist_dir')
        # Getting the type of 'install' (line 89)
        install_12073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'install')
        # Setting the type of the member 'root' of a type (line 89)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), install_12073, 'root', bdist_dir_12072)
        
        # Assigning a Attribute to a Attribute (line 90):
        # Getting the type of 'self' (line 90)
        self_12074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 29), 'self')
        # Obtaining the member 'skip_build' of a type (line 90)
        skip_build_12075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 29), self_12074, 'skip_build')
        # Getting the type of 'install' (line 90)
        install_12076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'install')
        # Setting the type of the member 'skip_build' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), install_12076, 'skip_build', skip_build_12075)
        
        # Assigning a Num to a Attribute (line 91):
        int_12077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 27), 'int')
        # Getting the type of 'install' (line 91)
        install_12078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'install')
        # Setting the type of the member 'warn_dir' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), install_12078, 'warn_dir', int_12077)
        
        # Call to info(...): (line 93)
        # Processing the call arguments (line 93)
        str_12081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 17), 'str', 'installing to %s')
        # Getting the type of 'self' (line 93)
        self_12082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 38), 'self', False)
        # Obtaining the member 'bdist_dir' of a type (line 93)
        bdist_dir_12083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 38), self_12082, 'bdist_dir')
        # Applying the binary operator '%' (line 93)
        result_mod_12084 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 17), '%', str_12081, bdist_dir_12083)
        
        # Processing the call keyword arguments (line 93)
        kwargs_12085 = {}
        # Getting the type of 'log' (line 93)
        log_12079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 93)
        info_12080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), log_12079, 'info')
        # Calling info(args, kwargs) (line 93)
        info_call_result_12086 = invoke(stypy.reporting.localization.Localization(__file__, 93, 8), info_12080, *[result_mod_12084], **kwargs_12085)
        
        
        # Call to run_command(...): (line 94)
        # Processing the call arguments (line 94)
        str_12089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 25), 'str', 'install')
        # Processing the call keyword arguments (line 94)
        kwargs_12090 = {}
        # Getting the type of 'self' (line 94)
        self_12087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self', False)
        # Obtaining the member 'run_command' of a type (line 94)
        run_command_12088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_12087, 'run_command')
        # Calling run_command(args, kwargs) (line 94)
        run_command_call_result_12091 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), run_command_12088, *[str_12089], **kwargs_12090)
        
        
        # Assigning a BinOp to a Name (line 98):
        str_12092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 27), 'str', '%s.%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 98)
        tuple_12093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 38), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 98)
        # Adding element type (line 98)
        
        # Call to get_fullname(...): (line 98)
        # Processing the call keyword arguments (line 98)
        kwargs_12097 = {}
        # Getting the type of 'self' (line 98)
        self_12094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 38), 'self', False)
        # Obtaining the member 'distribution' of a type (line 98)
        distribution_12095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 38), self_12094, 'distribution')
        # Obtaining the member 'get_fullname' of a type (line 98)
        get_fullname_12096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 38), distribution_12095, 'get_fullname')
        # Calling get_fullname(args, kwargs) (line 98)
        get_fullname_call_result_12098 = invoke(stypy.reporting.localization.Localization(__file__, 98, 38), get_fullname_12096, *[], **kwargs_12097)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 38), tuple_12093, get_fullname_call_result_12098)
        # Adding element type (line 98)
        # Getting the type of 'self' (line 99)
        self_12099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 38), 'self')
        # Obtaining the member 'plat_name' of a type (line 99)
        plat_name_12100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 38), self_12099, 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 38), tuple_12093, plat_name_12100)
        
        # Applying the binary operator '%' (line 98)
        result_mod_12101 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 27), '%', str_12092, tuple_12093)
        
        # Assigning a type to the variable 'archive_basename' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'archive_basename', result_mod_12101)
        
        
        # Getting the type of 'os' (line 103)
        os_12102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'os')
        # Obtaining the member 'name' of a type (line 103)
        name_12103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 11), os_12102, 'name')
        str_12104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 22), 'str', 'os2')
        # Applying the binary operator '==' (line 103)
        result_eq_12105 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 11), '==', name_12103, str_12104)
        
        # Testing the type of an if condition (line 103)
        if_condition_12106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 8), result_eq_12105)
        # Assigning a type to the variable 'if_condition_12106' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'if_condition_12106', if_condition_12106)
        # SSA begins for if statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 104):
        
        # Call to replace(...): (line 104)
        # Processing the call arguments (line 104)
        str_12109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 56), 'str', ':')
        str_12110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 61), 'str', '-')
        # Processing the call keyword arguments (line 104)
        kwargs_12111 = {}
        # Getting the type of 'archive_basename' (line 104)
        archive_basename_12107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 31), 'archive_basename', False)
        # Obtaining the member 'replace' of a type (line 104)
        replace_12108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 31), archive_basename_12107, 'replace')
        # Calling replace(args, kwargs) (line 104)
        replace_call_result_12112 = invoke(stypy.reporting.localization.Localization(__file__, 104, 31), replace_12108, *[str_12109, str_12110], **kwargs_12111)
        
        # Assigning a type to the variable 'archive_basename' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'archive_basename', replace_call_result_12112)
        # SSA join for if statement (line 103)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 106):
        
        # Call to join(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'self' (line 106)
        self_12116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 42), 'self', False)
        # Obtaining the member 'dist_dir' of a type (line 106)
        dist_dir_12117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 42), self_12116, 'dist_dir')
        # Getting the type of 'archive_basename' (line 106)
        archive_basename_12118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 57), 'archive_basename', False)
        # Processing the call keyword arguments (line 106)
        kwargs_12119 = {}
        # Getting the type of 'os' (line 106)
        os_12113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 'os', False)
        # Obtaining the member 'path' of a type (line 106)
        path_12114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 29), os_12113, 'path')
        # Obtaining the member 'join' of a type (line 106)
        join_12115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 29), path_12114, 'join')
        # Calling join(args, kwargs) (line 106)
        join_call_result_12120 = invoke(stypy.reporting.localization.Localization(__file__, 106, 29), join_12115, *[dist_dir_12117, archive_basename_12118], **kwargs_12119)
        
        # Assigning a type to the variable 'pseudoinstall_root' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'pseudoinstall_root', join_call_result_12120)
        
        
        # Getting the type of 'self' (line 107)
        self_12121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'self')
        # Obtaining the member 'relative' of a type (line 107)
        relative_12122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 15), self_12121, 'relative')
        # Applying the 'not' unary operator (line 107)
        result_not__12123 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 11), 'not', relative_12122)
        
        # Testing the type of an if condition (line 107)
        if_condition_12124 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 8), result_not__12123)
        # Assigning a type to the variable 'if_condition_12124' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'if_condition_12124', if_condition_12124)
        # SSA begins for if statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 108):
        # Getting the type of 'self' (line 108)
        self_12125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'self')
        # Obtaining the member 'bdist_dir' of a type (line 108)
        bdist_dir_12126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 27), self_12125, 'bdist_dir')
        # Assigning a type to the variable 'archive_root' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'archive_root', bdist_dir_12126)
        # SSA branch for the else part of an if statement (line 107)
        module_type_store.open_ssa_branch('else')
        
        
        # Evaluating a boolean operation
        
        # Call to has_ext_modules(...): (line 110)
        # Processing the call keyword arguments (line 110)
        kwargs_12130 = {}
        # Getting the type of 'self' (line 110)
        self_12127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'self', False)
        # Obtaining the member 'distribution' of a type (line 110)
        distribution_12128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), self_12127, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 110)
        has_ext_modules_12129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), distribution_12128, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 110)
        has_ext_modules_call_result_12131 = invoke(stypy.reporting.localization.Localization(__file__, 110, 16), has_ext_modules_12129, *[], **kwargs_12130)
        
        
        # Getting the type of 'install' (line 111)
        install_12132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 17), 'install')
        # Obtaining the member 'install_base' of a type (line 111)
        install_base_12133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 17), install_12132, 'install_base')
        # Getting the type of 'install' (line 111)
        install_12134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 41), 'install')
        # Obtaining the member 'install_platbase' of a type (line 111)
        install_platbase_12135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 41), install_12134, 'install_platbase')
        # Applying the binary operator '!=' (line 111)
        result_ne_12136 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 17), '!=', install_base_12133, install_platbase_12135)
        
        # Applying the binary operator 'and' (line 110)
        result_and_keyword_12137 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 16), 'and', has_ext_modules_call_result_12131, result_ne_12136)
        
        # Testing the type of an if condition (line 110)
        if_condition_12138 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 12), result_and_keyword_12137)
        # Assigning a type to the variable 'if_condition_12138' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'if_condition_12138', if_condition_12138)
        # SSA begins for if statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsPlatformError' (line 112)
        DistutilsPlatformError_12139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'DistutilsPlatformError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 112, 16), DistutilsPlatformError_12139, 'raise parameter', BaseException)
        # SSA branch for the else part of an if statement (line 110)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 118):
        
        # Call to join(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'self' (line 118)
        self_12143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 44), 'self', False)
        # Obtaining the member 'bdist_dir' of a type (line 118)
        bdist_dir_12144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 44), self_12143, 'bdist_dir')
        
        # Call to ensure_relative(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'install' (line 119)
        install_12146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 51), 'install', False)
        # Obtaining the member 'install_base' of a type (line 119)
        install_base_12147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 51), install_12146, 'install_base')
        # Processing the call keyword arguments (line 119)
        kwargs_12148 = {}
        # Getting the type of 'ensure_relative' (line 119)
        ensure_relative_12145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 35), 'ensure_relative', False)
        # Calling ensure_relative(args, kwargs) (line 119)
        ensure_relative_call_result_12149 = invoke(stypy.reporting.localization.Localization(__file__, 119, 35), ensure_relative_12145, *[install_base_12147], **kwargs_12148)
        
        # Processing the call keyword arguments (line 118)
        kwargs_12150 = {}
        # Getting the type of 'os' (line 118)
        os_12140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 118)
        path_12141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 31), os_12140, 'path')
        # Obtaining the member 'join' of a type (line 118)
        join_12142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 31), path_12141, 'join')
        # Calling join(args, kwargs) (line 118)
        join_call_result_12151 = invoke(stypy.reporting.localization.Localization(__file__, 118, 31), join_12142, *[bdist_dir_12144, ensure_relative_call_result_12149], **kwargs_12150)
        
        # Assigning a type to the variable 'archive_root' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'archive_root', join_call_result_12151)
        # SSA join for if statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 107)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 122):
        
        # Call to make_archive(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'pseudoinstall_root' (line 122)
        pseudoinstall_root_12154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 37), 'pseudoinstall_root', False)
        # Getting the type of 'self' (line 123)
        self_12155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 37), 'self', False)
        # Obtaining the member 'format' of a type (line 123)
        format_12156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 37), self_12155, 'format')
        # Processing the call keyword arguments (line 122)
        # Getting the type of 'archive_root' (line 123)
        archive_root_12157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 59), 'archive_root', False)
        keyword_12158 = archive_root_12157
        # Getting the type of 'self' (line 124)
        self_12159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 43), 'self', False)
        # Obtaining the member 'owner' of a type (line 124)
        owner_12160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 43), self_12159, 'owner')
        keyword_12161 = owner_12160
        # Getting the type of 'self' (line 124)
        self_12162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 61), 'self', False)
        # Obtaining the member 'group' of a type (line 124)
        group_12163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 61), self_12162, 'group')
        keyword_12164 = group_12163
        kwargs_12165 = {'owner': keyword_12161, 'group': keyword_12164, 'root_dir': keyword_12158}
        # Getting the type of 'self' (line 122)
        self_12152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'self', False)
        # Obtaining the member 'make_archive' of a type (line 122)
        make_archive_12153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 19), self_12152, 'make_archive')
        # Calling make_archive(args, kwargs) (line 122)
        make_archive_call_result_12166 = invoke(stypy.reporting.localization.Localization(__file__, 122, 19), make_archive_12153, *[pseudoinstall_root_12154, format_12156], **kwargs_12165)
        
        # Assigning a type to the variable 'filename' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'filename', make_archive_call_result_12166)
        
        
        # Call to has_ext_modules(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_12170 = {}
        # Getting the type of 'self' (line 125)
        self_12167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'self', False)
        # Obtaining the member 'distribution' of a type (line 125)
        distribution_12168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 11), self_12167, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 125)
        has_ext_modules_12169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 11), distribution_12168, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 125)
        has_ext_modules_call_result_12171 = invoke(stypy.reporting.localization.Localization(__file__, 125, 11), has_ext_modules_12169, *[], **kwargs_12170)
        
        # Testing the type of an if condition (line 125)
        if_condition_12172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), has_ext_modules_call_result_12171)
        # Assigning a type to the variable 'if_condition_12172' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'if_condition_12172', if_condition_12172)
        # SSA begins for if statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 126):
        
        # Call to get_python_version(...): (line 126)
        # Processing the call keyword arguments (line 126)
        kwargs_12174 = {}
        # Getting the type of 'get_python_version' (line 126)
        get_python_version_12173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'get_python_version', False)
        # Calling get_python_version(args, kwargs) (line 126)
        get_python_version_call_result_12175 = invoke(stypy.reporting.localization.Localization(__file__, 126, 24), get_python_version_12173, *[], **kwargs_12174)
        
        # Assigning a type to the variable 'pyversion' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'pyversion', get_python_version_call_result_12175)
        # SSA branch for the else part of an if statement (line 125)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 128):
        str_12176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 24), 'str', 'any')
        # Assigning a type to the variable 'pyversion' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'pyversion', str_12176)
        # SSA join for if statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Obtaining an instance of the builtin type 'tuple' (line 129)
        tuple_12181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 129)
        # Adding element type (line 129)
        str_12182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 45), 'str', 'bdist_dumb')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 45), tuple_12181, str_12182)
        # Adding element type (line 129)
        # Getting the type of 'pyversion' (line 129)
        pyversion_12183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 59), 'pyversion', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 45), tuple_12181, pyversion_12183)
        # Adding element type (line 129)
        # Getting the type of 'filename' (line 130)
        filename_12184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 45), 'filename', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 45), tuple_12181, filename_12184)
        
        # Processing the call keyword arguments (line 129)
        kwargs_12185 = {}
        # Getting the type of 'self' (line 129)
        self_12177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self', False)
        # Obtaining the member 'distribution' of a type (line 129)
        distribution_12178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_12177, 'distribution')
        # Obtaining the member 'dist_files' of a type (line 129)
        dist_files_12179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), distribution_12178, 'dist_files')
        # Obtaining the member 'append' of a type (line 129)
        append_12180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), dist_files_12179, 'append')
        # Calling append(args, kwargs) (line 129)
        append_call_result_12186 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), append_12180, *[tuple_12181], **kwargs_12185)
        
        
        
        # Getting the type of 'self' (line 132)
        self_12187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'self')
        # Obtaining the member 'keep_temp' of a type (line 132)
        keep_temp_12188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 15), self_12187, 'keep_temp')
        # Applying the 'not' unary operator (line 132)
        result_not__12189 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 11), 'not', keep_temp_12188)
        
        # Testing the type of an if condition (line 132)
        if_condition_12190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 8), result_not__12189)
        # Assigning a type to the variable 'if_condition_12190' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'if_condition_12190', if_condition_12190)
        # SSA begins for if statement (line 132)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove_tree(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'self' (line 133)
        self_12192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'self', False)
        # Obtaining the member 'bdist_dir' of a type (line 133)
        bdist_dir_12193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 24), self_12192, 'bdist_dir')
        # Processing the call keyword arguments (line 133)
        # Getting the type of 'self' (line 133)
        self_12194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 48), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 133)
        dry_run_12195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 48), self_12194, 'dry_run')
        keyword_12196 = dry_run_12195
        kwargs_12197 = {'dry_run': keyword_12196}
        # Getting the type of 'remove_tree' (line 133)
        remove_tree_12191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'remove_tree', False)
        # Calling remove_tree(args, kwargs) (line 133)
        remove_tree_call_result_12198 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), remove_tree_12191, *[bdist_dir_12193], **kwargs_12197)
        
        # SSA join for if statement (line 132)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_12199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12199)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_12199


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 0, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist_dumb.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'bdist_dumb' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'bdist_dumb', bdist_dumb)

# Assigning a Str to a Name (line 21):
str_12200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 18), 'str', 'create a "dumb" built distribution')
# Getting the type of 'bdist_dumb'
bdist_dumb_12201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist_dumb')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_dumb_12201, 'description', str_12200)

# Assigning a List to a Name (line 23):

# Obtaining an instance of the builtin type 'list' (line 23)
list_12202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)

# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_12203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
str_12204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'str', 'bdist-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 21), tuple_12203, str_12204)
# Adding element type (line 23)
str_12205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 35), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 21), tuple_12203, str_12205)
# Adding element type (line 23)
str_12206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 21), 'str', 'temporary directory for creating the distribution')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 21), tuple_12203, str_12206)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_12202, tuple_12203)
# Adding element type (line 23)

# Obtaining an instance of the builtin type 'tuple' (line 25)
tuple_12207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 25)
# Adding element type (line 25)
str_12208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'str', 'plat-name=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 21), tuple_12207, str_12208)
# Adding element type (line 25)
str_12209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 35), 'str', 'p')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 21), tuple_12207, str_12209)
# Adding element type (line 25)
str_12210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'str', 'platform name to embed in generated filenames (default: %s)')

# Call to get_platform(...): (line 27)
# Processing the call keyword arguments (line 27)
kwargs_12212 = {}
# Getting the type of 'get_platform' (line 27)
get_platform_12211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 39), 'get_platform', False)
# Calling get_platform(args, kwargs) (line 27)
get_platform_call_result_12213 = invoke(stypy.reporting.localization.Localization(__file__, 27, 39), get_platform_12211, *[], **kwargs_12212)

# Applying the binary operator '%' (line 26)
result_mod_12214 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 21), '%', str_12210, get_platform_call_result_12213)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 21), tuple_12207, result_mod_12214)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_12202, tuple_12207)
# Adding element type (line 23)

# Obtaining an instance of the builtin type 'tuple' (line 28)
tuple_12215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 28)
# Adding element type (line 28)
str_12216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 21), 'str', 'format=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 21), tuple_12215, str_12216)
# Adding element type (line 28)
str_12217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 32), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 21), tuple_12215, str_12217)
# Adding element type (line 28)
str_12218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 21), 'str', 'archive format to create (tar, ztar, gztar, zip)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 21), tuple_12215, str_12218)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_12202, tuple_12215)
# Adding element type (line 23)

# Obtaining an instance of the builtin type 'tuple' (line 30)
tuple_12219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 30)
# Adding element type (line 30)
str_12220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 21), 'str', 'keep-temp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 21), tuple_12219, str_12220)
# Adding element type (line 30)
str_12221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 34), 'str', 'k')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 21), tuple_12219, str_12221)
# Adding element type (line 30)
str_12222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 21), 'str', 'keep the pseudo-installation tree around after ')
str_12223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'str', 'creating the distribution archive')
# Applying the binary operator '+' (line 31)
result_add_12224 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 21), '+', str_12222, str_12223)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 21), tuple_12219, result_add_12224)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_12202, tuple_12219)
# Adding element type (line 23)

# Obtaining an instance of the builtin type 'tuple' (line 33)
tuple_12225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 33)
# Adding element type (line 33)
str_12226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'str', 'dist-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), tuple_12225, str_12226)
# Adding element type (line 33)
str_12227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 34), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), tuple_12225, str_12227)
# Adding element type (line 33)
str_12228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 21), 'str', 'directory to put final built distributions in')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), tuple_12225, str_12228)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_12202, tuple_12225)
# Adding element type (line 23)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_12229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
str_12230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 21), 'str', 'skip-build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 21), tuple_12229, str_12230)
# Adding element type (line 35)
# Getting the type of 'None' (line 35)
None_12231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 35), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 21), tuple_12229, None_12231)
# Adding element type (line 35)
str_12232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 21), 'str', 'skip rebuilding everything (for testing/debugging)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 21), tuple_12229, str_12232)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_12202, tuple_12229)
# Adding element type (line 23)

# Obtaining an instance of the builtin type 'tuple' (line 37)
tuple_12233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 37)
# Adding element type (line 37)
str_12234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'str', 'relative')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 21), tuple_12233, str_12234)
# Adding element type (line 37)
# Getting the type of 'None' (line 37)
None_12235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 33), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 21), tuple_12233, None_12235)
# Adding element type (line 37)
str_12236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'str', 'build the archive using relative paths(default: false)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 21), tuple_12233, str_12236)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_12202, tuple_12233)
# Adding element type (line 23)

# Obtaining an instance of the builtin type 'tuple' (line 40)
tuple_12237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 40)
# Adding element type (line 40)
str_12238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 21), 'str', 'owner=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 21), tuple_12237, str_12238)
# Adding element type (line 40)
str_12239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 31), 'str', 'u')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 21), tuple_12237, str_12239)
# Adding element type (line 40)
str_12240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 21), 'str', 'Owner name used when creating a tar file [default: current user]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 21), tuple_12237, str_12240)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_12202, tuple_12237)
# Adding element type (line 23)

# Obtaining an instance of the builtin type 'tuple' (line 43)
tuple_12241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 43)
# Adding element type (line 43)
str_12242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'str', 'group=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), tuple_12241, str_12242)
# Adding element type (line 43)
str_12243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 31), 'str', 'g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), tuple_12241, str_12243)
# Adding element type (line 43)
str_12244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 21), 'str', 'Group name used when creating a tar file [default: current group]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), tuple_12241, str_12244)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_12202, tuple_12241)

# Getting the type of 'bdist_dumb'
bdist_dumb_12245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist_dumb')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_dumb_12245, 'user_options', list_12202)

# Assigning a List to a Name (line 48):

# Obtaining an instance of the builtin type 'list' (line 48)
list_12246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 48)
# Adding element type (line 48)
str_12247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 23), 'str', 'keep-temp')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 22), list_12246, str_12247)
# Adding element type (line 48)
str_12248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 36), 'str', 'skip-build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 22), list_12246, str_12248)
# Adding element type (line 48)
str_12249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 50), 'str', 'relative')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 22), list_12246, str_12249)

# Getting the type of 'bdist_dumb'
bdist_dumb_12250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist_dumb')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_dumb_12250, 'boolean_options', list_12246)

# Assigning a Dict to a Name (line 50):

# Obtaining an instance of the builtin type 'dict' (line 50)
dict_12251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 21), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 50)
# Adding element type (key, value) (line 50)
str_12252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 23), 'str', 'posix')
str_12253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 32), 'str', 'gztar')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 21), dict_12251, (str_12252, str_12253))
# Adding element type (key, value) (line 50)
str_12254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 23), 'str', 'nt')
str_12255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 29), 'str', 'zip')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 21), dict_12251, (str_12254, str_12255))
# Adding element type (key, value) (line 50)
str_12256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 23), 'str', 'os2')
str_12257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 30), 'str', 'zip')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 21), dict_12251, (str_12256, str_12257))

# Getting the type of 'bdist_dumb'
bdist_dumb_12258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist_dumb')
# Setting the type of the member 'default_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_dumb_12258, 'default_format', dict_12251)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
