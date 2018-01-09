
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.build
2: 
3: Implements the Distutils 'build' command.'''
4: 
5: __revision__ = "$Id$"
6: 
7: import sys, os
8: 
9: from distutils.util import get_platform
10: from distutils.core import Command
11: from distutils.errors import DistutilsOptionError
12: 
13: def show_compilers():
14:     from distutils.ccompiler import show_compilers
15:     show_compilers()
16: 
17: class build(Command):
18: 
19:     description = "build everything needed to install"
20: 
21:     user_options = [
22:         ('build-base=', 'b',
23:          "base directory for build library"),
24:         ('build-purelib=', None,
25:          "build directory for platform-neutral distributions"),
26:         ('build-platlib=', None,
27:          "build directory for platform-specific distributions"),
28:         ('build-lib=', None,
29:          "build directory for all distribution (defaults to either " +
30:          "build-purelib or build-platlib"),
31:         ('build-scripts=', None,
32:          "build directory for scripts"),
33:         ('build-temp=', 't',
34:          "temporary build directory"),
35:         ('plat-name=', 'p',
36:          "platform name to build for, if supported "
37:          "(default: %s)" % get_platform()),
38:         ('compiler=', 'c',
39:          "specify the compiler type"),
40:         ('debug', 'g',
41:          "compile extensions and libraries with debugging information"),
42:         ('force', 'f',
43:          "forcibly build everything (ignore file timestamps)"),
44:         ('executable=', 'e',
45:          "specify final destination interpreter path (build.py)"),
46:         ]
47: 
48:     boolean_options = ['debug', 'force']
49: 
50:     help_options = [
51:         ('help-compiler', None,
52:          "list available compilers", show_compilers),
53:         ]
54: 
55:     def initialize_options(self):
56:         self.build_base = 'build'
57:         # these are decided only after 'build_base' has its final value
58:         # (unless overridden by the user or client)
59:         self.build_purelib = None
60:         self.build_platlib = None
61:         self.build_lib = None
62:         self.build_temp = None
63:         self.build_scripts = None
64:         self.compiler = None
65:         self.plat_name = None
66:         self.debug = None
67:         self.force = 0
68:         self.executable = None
69: 
70:     def finalize_options(self):
71:         if self.plat_name is None:
72:             self.plat_name = get_platform()
73:         else:
74:             # plat-name only supported for windows (other platforms are
75:             # supported via ./configure flags, if at all).  Avoid misleading
76:             # other platforms.
77:             if os.name != 'nt':
78:                 raise DistutilsOptionError(
79:                             "--plat-name only supported on Windows (try "
80:                             "using './configure --help' on your platform)")
81: 
82:         plat_specifier = ".%s-%s" % (self.plat_name, sys.version[0:3])
83: 
84:         # Make it so Python 2.x and Python 2.x with --with-pydebug don't
85:         # share the same build directories. Doing so confuses the build
86:         # process for C modules
87:         if hasattr(sys, 'gettotalrefcount'):
88:             plat_specifier += '-pydebug'
89: 
90:         # 'build_purelib' and 'build_platlib' just default to 'lib' and
91:         # 'lib.<plat>' under the base build directory.  We only use one of
92:         # them for a given distribution, though --
93:         if self.build_purelib is None:
94:             self.build_purelib = os.path.join(self.build_base, 'lib')
95:         if self.build_platlib is None:
96:             self.build_platlib = os.path.join(self.build_base,
97:                                               'lib' + plat_specifier)
98: 
99:         # 'build_lib' is the actual directory that we will use for this
100:         # particular module distribution -- if user didn't supply it, pick
101:         # one of 'build_purelib' or 'build_platlib'.
102:         if self.build_lib is None:
103:             if self.distribution.ext_modules:
104:                 self.build_lib = self.build_platlib
105:             else:
106:                 self.build_lib = self.build_purelib
107: 
108:         # 'build_temp' -- temporary directory for compiler turds,
109:         # "build/temp.<plat>"
110:         if self.build_temp is None:
111:             self.build_temp = os.path.join(self.build_base,
112:                                            'temp' + plat_specifier)
113:         if self.build_scripts is None:
114:             self.build_scripts = os.path.join(self.build_base,
115:                                               'scripts-' + sys.version[0:3])
116: 
117:         if self.executable is None:
118:             self.executable = os.path.normpath(sys.executable)
119: 
120:     def run(self):
121:         # Run all relevant sub-commands.  This will be some subset of:
122:         #  - build_py      - pure Python modules
123:         #  - build_clib    - standalone C libraries
124:         #  - build_ext     - Python extensions
125:         #  - build_scripts - (Python) scripts
126:         for cmd_name in self.get_sub_commands():
127:             self.run_command(cmd_name)
128: 
129:     # -- Predicates for the sub-command list ---------------------------
130: 
131:     def has_pure_modules (self):
132:         return self.distribution.has_pure_modules()
133: 
134:     def has_c_libraries (self):
135:         return self.distribution.has_c_libraries()
136: 
137:     def has_ext_modules (self):
138:         return self.distribution.has_ext_modules()
139: 
140:     def has_scripts (self):
141:         return self.distribution.has_scripts()
142: 
143:     sub_commands = [('build_py',      has_pure_modules),
144:                     ('build_clib',    has_c_libraries),
145:                     ('build_ext',     has_ext_modules),
146:                     ('build_scripts', has_scripts),
147:                    ]
148: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_17183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', "distutils.command.build\n\nImplements the Distutils 'build' command.")

# Assigning a Str to a Name (line 5):
str_17184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__revision__', str_17184)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# Multiple import statement. import sys (1/2) (line 7)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'sys', sys, module_type_store)
# Multiple import statement. import os (2/2) (line 7)
import os

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.util import get_platform' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_17185 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.util')

if (type(import_17185) is not StypyTypeError):

    if (import_17185 != 'pyd_module'):
        __import__(import_17185)
        sys_modules_17186 = sys.modules[import_17185]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.util', sys_modules_17186.module_type_store, module_type_store, ['get_platform'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_17186, sys_modules_17186.module_type_store, module_type_store)
    else:
        from distutils.util import get_platform

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.util', None, module_type_store, ['get_platform'], [get_platform])

else:
    # Assigning a type to the variable 'distutils.util' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.util', import_17185)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.core import Command' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_17187 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.core')

if (type(import_17187) is not StypyTypeError):

    if (import_17187 != 'pyd_module'):
        __import__(import_17187)
        sys_modules_17188 = sys.modules[import_17187]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.core', sys_modules_17188.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_17188, sys_modules_17188.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.core', import_17187)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.errors import DistutilsOptionError' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_17189 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.errors')

if (type(import_17189) is not StypyTypeError):

    if (import_17189 != 'pyd_module'):
        __import__(import_17189)
        sys_modules_17190 = sys.modules[import_17189]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.errors', sys_modules_17190.module_type_store, module_type_store, ['DistutilsOptionError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_17190, sys_modules_17190.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsOptionError

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.errors', None, module_type_store, ['DistutilsOptionError'], [DistutilsOptionError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.errors', import_17189)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')


@norecursion
def show_compilers(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'show_compilers'
    module_type_store = module_type_store.open_function_context('show_compilers', 13, 0, False)
    
    # Passed parameters checking function
    show_compilers.stypy_localization = localization
    show_compilers.stypy_type_of_self = None
    show_compilers.stypy_type_store = module_type_store
    show_compilers.stypy_function_name = 'show_compilers'
    show_compilers.stypy_param_names_list = []
    show_compilers.stypy_varargs_param_name = None
    show_compilers.stypy_kwargs_param_name = None
    show_compilers.stypy_call_defaults = defaults
    show_compilers.stypy_call_varargs = varargs
    show_compilers.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'show_compilers', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'show_compilers', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'show_compilers(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 4))
    
    # 'from distutils.ccompiler import show_compilers' statement (line 14)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
    import_17191 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 4), 'distutils.ccompiler')

    if (type(import_17191) is not StypyTypeError):

        if (import_17191 != 'pyd_module'):
            __import__(import_17191)
            sys_modules_17192 = sys.modules[import_17191]
            import_from_module(stypy.reporting.localization.Localization(__file__, 14, 4), 'distutils.ccompiler', sys_modules_17192.module_type_store, module_type_store, ['show_compilers'])
            nest_module(stypy.reporting.localization.Localization(__file__, 14, 4), __file__, sys_modules_17192, sys_modules_17192.module_type_store, module_type_store)
        else:
            from distutils.ccompiler import show_compilers

            import_from_module(stypy.reporting.localization.Localization(__file__, 14, 4), 'distutils.ccompiler', None, module_type_store, ['show_compilers'], [show_compilers])

    else:
        # Assigning a type to the variable 'distutils.ccompiler' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'distutils.ccompiler', import_17191)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
    
    
    # Call to show_compilers(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_17194 = {}
    # Getting the type of 'show_compilers' (line 15)
    show_compilers_17193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'show_compilers', False)
    # Calling show_compilers(args, kwargs) (line 15)
    show_compilers_call_result_17195 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), show_compilers_17193, *[], **kwargs_17194)
    
    
    # ################# End of 'show_compilers(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'show_compilers' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_17196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17196)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'show_compilers'
    return stypy_return_type_17196

# Assigning a type to the variable 'show_compilers' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'show_compilers', show_compilers)
# Declaration of the 'build' class
# Getting the type of 'Command' (line 17)
Command_17197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'Command')

class build(Command_17197, ):

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
        build.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        build.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build.initialize_options.__dict__.__setitem__('stypy_function_name', 'build.initialize_options')
        build.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Attribute (line 56):
        str_17198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 26), 'str', 'build')
        # Getting the type of 'self' (line 56)
        self_17199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member 'build_base' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_17199, 'build_base', str_17198)
        
        # Assigning a Name to a Attribute (line 59):
        # Getting the type of 'None' (line 59)
        None_17200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'None')
        # Getting the type of 'self' (line 59)
        self_17201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member 'build_purelib' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_17201, 'build_purelib', None_17200)
        
        # Assigning a Name to a Attribute (line 60):
        # Getting the type of 'None' (line 60)
        None_17202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'None')
        # Getting the type of 'self' (line 60)
        self_17203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self')
        # Setting the type of the member 'build_platlib' of a type (line 60)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_17203, 'build_platlib', None_17202)
        
        # Assigning a Name to a Attribute (line 61):
        # Getting the type of 'None' (line 61)
        None_17204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'None')
        # Getting the type of 'self' (line 61)
        self_17205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
        # Setting the type of the member 'build_lib' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_17205, 'build_lib', None_17204)
        
        # Assigning a Name to a Attribute (line 62):
        # Getting the type of 'None' (line 62)
        None_17206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 26), 'None')
        # Getting the type of 'self' (line 62)
        self_17207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member 'build_temp' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_17207, 'build_temp', None_17206)
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'None' (line 63)
        None_17208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 29), 'None')
        # Getting the type of 'self' (line 63)
        self_17209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member 'build_scripts' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_17209, 'build_scripts', None_17208)
        
        # Assigning a Name to a Attribute (line 64):
        # Getting the type of 'None' (line 64)
        None_17210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'None')
        # Getting the type of 'self' (line 64)
        self_17211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Setting the type of the member 'compiler' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_17211, 'compiler', None_17210)
        
        # Assigning a Name to a Attribute (line 65):
        # Getting the type of 'None' (line 65)
        None_17212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'None')
        # Getting the type of 'self' (line 65)
        self_17213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self')
        # Setting the type of the member 'plat_name' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_17213, 'plat_name', None_17212)
        
        # Assigning a Name to a Attribute (line 66):
        # Getting the type of 'None' (line 66)
        None_17214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'None')
        # Getting the type of 'self' (line 66)
        self_17215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self')
        # Setting the type of the member 'debug' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_17215, 'debug', None_17214)
        
        # Assigning a Num to a Attribute (line 67):
        int_17216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 21), 'int')
        # Getting the type of 'self' (line 67)
        self_17217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self')
        # Setting the type of the member 'force' of a type (line 67)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_17217, 'force', int_17216)
        
        # Assigning a Name to a Attribute (line 68):
        # Getting the type of 'None' (line 68)
        None_17218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 26), 'None')
        # Getting the type of 'self' (line 68)
        self_17219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self')
        # Setting the type of the member 'executable' of a type (line 68)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_17219, 'executable', None_17218)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_17220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17220)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_17220


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 70, 4, False)
        # Assigning a type to the variable 'self' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        build.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build.finalize_options.__dict__.__setitem__('stypy_function_name', 'build.finalize_options')
        build.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 71)
        # Getting the type of 'self' (line 71)
        self_17221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'self')
        # Obtaining the member 'plat_name' of a type (line 71)
        plat_name_17222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 11), self_17221, 'plat_name')
        # Getting the type of 'None' (line 71)
        None_17223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 29), 'None')
        
        (may_be_17224, more_types_in_union_17225) = may_be_none(plat_name_17222, None_17223)

        if may_be_17224:

            if more_types_in_union_17225:
                # Runtime conditional SSA (line 71)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 72):
            
            # Call to get_platform(...): (line 72)
            # Processing the call keyword arguments (line 72)
            kwargs_17227 = {}
            # Getting the type of 'get_platform' (line 72)
            get_platform_17226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 29), 'get_platform', False)
            # Calling get_platform(args, kwargs) (line 72)
            get_platform_call_result_17228 = invoke(stypy.reporting.localization.Localization(__file__, 72, 29), get_platform_17226, *[], **kwargs_17227)
            
            # Getting the type of 'self' (line 72)
            self_17229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'self')
            # Setting the type of the member 'plat_name' of a type (line 72)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), self_17229, 'plat_name', get_platform_call_result_17228)

            if more_types_in_union_17225:
                # Runtime conditional SSA for else branch (line 71)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_17224) or more_types_in_union_17225):
            
            
            # Getting the type of 'os' (line 77)
            os_17230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 15), 'os')
            # Obtaining the member 'name' of a type (line 77)
            name_17231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 15), os_17230, 'name')
            str_17232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 26), 'str', 'nt')
            # Applying the binary operator '!=' (line 77)
            result_ne_17233 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 15), '!=', name_17231, str_17232)
            
            # Testing the type of an if condition (line 77)
            if_condition_17234 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 12), result_ne_17233)
            # Assigning a type to the variable 'if_condition_17234' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'if_condition_17234', if_condition_17234)
            # SSA begins for if statement (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to DistutilsOptionError(...): (line 78)
            # Processing the call arguments (line 78)
            str_17236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 28), 'str', "--plat-name only supported on Windows (try using './configure --help' on your platform)")
            # Processing the call keyword arguments (line 78)
            kwargs_17237 = {}
            # Getting the type of 'DistutilsOptionError' (line 78)
            DistutilsOptionError_17235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 22), 'DistutilsOptionError', False)
            # Calling DistutilsOptionError(args, kwargs) (line 78)
            DistutilsOptionError_call_result_17238 = invoke(stypy.reporting.localization.Localization(__file__, 78, 22), DistutilsOptionError_17235, *[str_17236], **kwargs_17237)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 78, 16), DistutilsOptionError_call_result_17238, 'raise parameter', BaseException)
            # SSA join for if statement (line 77)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_17224 and more_types_in_union_17225):
                # SSA join for if statement (line 71)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 82):
        str_17239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 25), 'str', '.%s-%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 82)
        tuple_17240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 82)
        # Adding element type (line 82)
        # Getting the type of 'self' (line 82)
        self_17241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 37), 'self')
        # Obtaining the member 'plat_name' of a type (line 82)
        plat_name_17242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 37), self_17241, 'plat_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 37), tuple_17240, plat_name_17242)
        # Adding element type (line 82)
        
        # Obtaining the type of the subscript
        int_17243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 65), 'int')
        int_17244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 67), 'int')
        slice_17245 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 82, 53), int_17243, int_17244, None)
        # Getting the type of 'sys' (line 82)
        sys_17246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 53), 'sys')
        # Obtaining the member 'version' of a type (line 82)
        version_17247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 53), sys_17246, 'version')
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___17248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 53), version_17247, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_17249 = invoke(stypy.reporting.localization.Localization(__file__, 82, 53), getitem___17248, slice_17245)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 37), tuple_17240, subscript_call_result_17249)
        
        # Applying the binary operator '%' (line 82)
        result_mod_17250 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 25), '%', str_17239, tuple_17240)
        
        # Assigning a type to the variable 'plat_specifier' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'plat_specifier', result_mod_17250)
        
        # Type idiom detected: calculating its left and rigth part (line 87)
        str_17251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 24), 'str', 'gettotalrefcount')
        # Getting the type of 'sys' (line 87)
        sys_17252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'sys')
        
        (may_be_17253, more_types_in_union_17254) = may_provide_member(str_17251, sys_17252)

        if may_be_17253:

            if more_types_in_union_17254:
                # Runtime conditional SSA (line 87)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'sys' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'sys', remove_not_member_provider_from_union(sys_17252, 'gettotalrefcount'))
            
            # Getting the type of 'plat_specifier' (line 88)
            plat_specifier_17255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'plat_specifier')
            str_17256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 30), 'str', '-pydebug')
            # Applying the binary operator '+=' (line 88)
            result_iadd_17257 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 12), '+=', plat_specifier_17255, str_17256)
            # Assigning a type to the variable 'plat_specifier' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'plat_specifier', result_iadd_17257)
            

            if more_types_in_union_17254:
                # SSA join for if statement (line 87)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 93)
        # Getting the type of 'self' (line 93)
        self_17258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'self')
        # Obtaining the member 'build_purelib' of a type (line 93)
        build_purelib_17259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 11), self_17258, 'build_purelib')
        # Getting the type of 'None' (line 93)
        None_17260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 33), 'None')
        
        (may_be_17261, more_types_in_union_17262) = may_be_none(build_purelib_17259, None_17260)

        if may_be_17261:

            if more_types_in_union_17262:
                # Runtime conditional SSA (line 93)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 94):
            
            # Call to join(...): (line 94)
            # Processing the call arguments (line 94)
            # Getting the type of 'self' (line 94)
            self_17266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 46), 'self', False)
            # Obtaining the member 'build_base' of a type (line 94)
            build_base_17267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 46), self_17266, 'build_base')
            str_17268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 63), 'str', 'lib')
            # Processing the call keyword arguments (line 94)
            kwargs_17269 = {}
            # Getting the type of 'os' (line 94)
            os_17263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 33), 'os', False)
            # Obtaining the member 'path' of a type (line 94)
            path_17264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 33), os_17263, 'path')
            # Obtaining the member 'join' of a type (line 94)
            join_17265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 33), path_17264, 'join')
            # Calling join(args, kwargs) (line 94)
            join_call_result_17270 = invoke(stypy.reporting.localization.Localization(__file__, 94, 33), join_17265, *[build_base_17267, str_17268], **kwargs_17269)
            
            # Getting the type of 'self' (line 94)
            self_17271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'self')
            # Setting the type of the member 'build_purelib' of a type (line 94)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), self_17271, 'build_purelib', join_call_result_17270)

            if more_types_in_union_17262:
                # SSA join for if statement (line 93)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 95)
        # Getting the type of 'self' (line 95)
        self_17272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'self')
        # Obtaining the member 'build_platlib' of a type (line 95)
        build_platlib_17273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 11), self_17272, 'build_platlib')
        # Getting the type of 'None' (line 95)
        None_17274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 33), 'None')
        
        (may_be_17275, more_types_in_union_17276) = may_be_none(build_platlib_17273, None_17274)

        if may_be_17275:

            if more_types_in_union_17276:
                # Runtime conditional SSA (line 95)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 96):
            
            # Call to join(...): (line 96)
            # Processing the call arguments (line 96)
            # Getting the type of 'self' (line 96)
            self_17280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 46), 'self', False)
            # Obtaining the member 'build_base' of a type (line 96)
            build_base_17281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 46), self_17280, 'build_base')
            str_17282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 46), 'str', 'lib')
            # Getting the type of 'plat_specifier' (line 97)
            plat_specifier_17283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 54), 'plat_specifier', False)
            # Applying the binary operator '+' (line 97)
            result_add_17284 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 46), '+', str_17282, plat_specifier_17283)
            
            # Processing the call keyword arguments (line 96)
            kwargs_17285 = {}
            # Getting the type of 'os' (line 96)
            os_17277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'os', False)
            # Obtaining the member 'path' of a type (line 96)
            path_17278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 33), os_17277, 'path')
            # Obtaining the member 'join' of a type (line 96)
            join_17279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 33), path_17278, 'join')
            # Calling join(args, kwargs) (line 96)
            join_call_result_17286 = invoke(stypy.reporting.localization.Localization(__file__, 96, 33), join_17279, *[build_base_17281, result_add_17284], **kwargs_17285)
            
            # Getting the type of 'self' (line 96)
            self_17287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'self')
            # Setting the type of the member 'build_platlib' of a type (line 96)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), self_17287, 'build_platlib', join_call_result_17286)

            if more_types_in_union_17276:
                # SSA join for if statement (line 95)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 102)
        # Getting the type of 'self' (line 102)
        self_17288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'self')
        # Obtaining the member 'build_lib' of a type (line 102)
        build_lib_17289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 11), self_17288, 'build_lib')
        # Getting the type of 'None' (line 102)
        None_17290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 29), 'None')
        
        (may_be_17291, more_types_in_union_17292) = may_be_none(build_lib_17289, None_17290)

        if may_be_17291:

            if more_types_in_union_17292:
                # Runtime conditional SSA (line 102)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'self' (line 103)
            self_17293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'self')
            # Obtaining the member 'distribution' of a type (line 103)
            distribution_17294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), self_17293, 'distribution')
            # Obtaining the member 'ext_modules' of a type (line 103)
            ext_modules_17295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), distribution_17294, 'ext_modules')
            # Testing the type of an if condition (line 103)
            if_condition_17296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 12), ext_modules_17295)
            # Assigning a type to the variable 'if_condition_17296' (line 103)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'if_condition_17296', if_condition_17296)
            # SSA begins for if statement (line 103)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Attribute (line 104):
            # Getting the type of 'self' (line 104)
            self_17297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 33), 'self')
            # Obtaining the member 'build_platlib' of a type (line 104)
            build_platlib_17298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 33), self_17297, 'build_platlib')
            # Getting the type of 'self' (line 104)
            self_17299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'self')
            # Setting the type of the member 'build_lib' of a type (line 104)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 16), self_17299, 'build_lib', build_platlib_17298)
            # SSA branch for the else part of an if statement (line 103)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Attribute (line 106):
            # Getting the type of 'self' (line 106)
            self_17300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'self')
            # Obtaining the member 'build_purelib' of a type (line 106)
            build_purelib_17301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 33), self_17300, 'build_purelib')
            # Getting the type of 'self' (line 106)
            self_17302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'self')
            # Setting the type of the member 'build_lib' of a type (line 106)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), self_17302, 'build_lib', build_purelib_17301)
            # SSA join for if statement (line 103)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_17292:
                # SSA join for if statement (line 102)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 110)
        # Getting the type of 'self' (line 110)
        self_17303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'self')
        # Obtaining the member 'build_temp' of a type (line 110)
        build_temp_17304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 11), self_17303, 'build_temp')
        # Getting the type of 'None' (line 110)
        None_17305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 30), 'None')
        
        (may_be_17306, more_types_in_union_17307) = may_be_none(build_temp_17304, None_17305)

        if may_be_17306:

            if more_types_in_union_17307:
                # Runtime conditional SSA (line 110)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 111):
            
            # Call to join(...): (line 111)
            # Processing the call arguments (line 111)
            # Getting the type of 'self' (line 111)
            self_17311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 43), 'self', False)
            # Obtaining the member 'build_base' of a type (line 111)
            build_base_17312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 43), self_17311, 'build_base')
            str_17313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 43), 'str', 'temp')
            # Getting the type of 'plat_specifier' (line 112)
            plat_specifier_17314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 52), 'plat_specifier', False)
            # Applying the binary operator '+' (line 112)
            result_add_17315 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 43), '+', str_17313, plat_specifier_17314)
            
            # Processing the call keyword arguments (line 111)
            kwargs_17316 = {}
            # Getting the type of 'os' (line 111)
            os_17308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'os', False)
            # Obtaining the member 'path' of a type (line 111)
            path_17309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 30), os_17308, 'path')
            # Obtaining the member 'join' of a type (line 111)
            join_17310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 30), path_17309, 'join')
            # Calling join(args, kwargs) (line 111)
            join_call_result_17317 = invoke(stypy.reporting.localization.Localization(__file__, 111, 30), join_17310, *[build_base_17312, result_add_17315], **kwargs_17316)
            
            # Getting the type of 'self' (line 111)
            self_17318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'self')
            # Setting the type of the member 'build_temp' of a type (line 111)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), self_17318, 'build_temp', join_call_result_17317)

            if more_types_in_union_17307:
                # SSA join for if statement (line 110)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 113)
        # Getting the type of 'self' (line 113)
        self_17319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 11), 'self')
        # Obtaining the member 'build_scripts' of a type (line 113)
        build_scripts_17320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 11), self_17319, 'build_scripts')
        # Getting the type of 'None' (line 113)
        None_17321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 33), 'None')
        
        (may_be_17322, more_types_in_union_17323) = may_be_none(build_scripts_17320, None_17321)

        if may_be_17322:

            if more_types_in_union_17323:
                # Runtime conditional SSA (line 113)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 114):
            
            # Call to join(...): (line 114)
            # Processing the call arguments (line 114)
            # Getting the type of 'self' (line 114)
            self_17327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 46), 'self', False)
            # Obtaining the member 'build_base' of a type (line 114)
            build_base_17328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 46), self_17327, 'build_base')
            str_17329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 46), 'str', 'scripts-')
            
            # Obtaining the type of the subscript
            int_17330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 71), 'int')
            int_17331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 73), 'int')
            slice_17332 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 115, 59), int_17330, int_17331, None)
            # Getting the type of 'sys' (line 115)
            sys_17333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 59), 'sys', False)
            # Obtaining the member 'version' of a type (line 115)
            version_17334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 59), sys_17333, 'version')
            # Obtaining the member '__getitem__' of a type (line 115)
            getitem___17335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 59), version_17334, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 115)
            subscript_call_result_17336 = invoke(stypy.reporting.localization.Localization(__file__, 115, 59), getitem___17335, slice_17332)
            
            # Applying the binary operator '+' (line 115)
            result_add_17337 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 46), '+', str_17329, subscript_call_result_17336)
            
            # Processing the call keyword arguments (line 114)
            kwargs_17338 = {}
            # Getting the type of 'os' (line 114)
            os_17324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 33), 'os', False)
            # Obtaining the member 'path' of a type (line 114)
            path_17325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 33), os_17324, 'path')
            # Obtaining the member 'join' of a type (line 114)
            join_17326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 33), path_17325, 'join')
            # Calling join(args, kwargs) (line 114)
            join_call_result_17339 = invoke(stypy.reporting.localization.Localization(__file__, 114, 33), join_17326, *[build_base_17328, result_add_17337], **kwargs_17338)
            
            # Getting the type of 'self' (line 114)
            self_17340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'self')
            # Setting the type of the member 'build_scripts' of a type (line 114)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), self_17340, 'build_scripts', join_call_result_17339)

            if more_types_in_union_17323:
                # SSA join for if statement (line 113)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 117)
        # Getting the type of 'self' (line 117)
        self_17341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 11), 'self')
        # Obtaining the member 'executable' of a type (line 117)
        executable_17342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 11), self_17341, 'executable')
        # Getting the type of 'None' (line 117)
        None_17343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 30), 'None')
        
        (may_be_17344, more_types_in_union_17345) = may_be_none(executable_17342, None_17343)

        if may_be_17344:

            if more_types_in_union_17345:
                # Runtime conditional SSA (line 117)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 118):
            
            # Call to normpath(...): (line 118)
            # Processing the call arguments (line 118)
            # Getting the type of 'sys' (line 118)
            sys_17349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 47), 'sys', False)
            # Obtaining the member 'executable' of a type (line 118)
            executable_17350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 47), sys_17349, 'executable')
            # Processing the call keyword arguments (line 118)
            kwargs_17351 = {}
            # Getting the type of 'os' (line 118)
            os_17346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 30), 'os', False)
            # Obtaining the member 'path' of a type (line 118)
            path_17347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 30), os_17346, 'path')
            # Obtaining the member 'normpath' of a type (line 118)
            normpath_17348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 30), path_17347, 'normpath')
            # Calling normpath(args, kwargs) (line 118)
            normpath_call_result_17352 = invoke(stypy.reporting.localization.Localization(__file__, 118, 30), normpath_17348, *[executable_17350], **kwargs_17351)
            
            # Getting the type of 'self' (line 118)
            self_17353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'self')
            # Setting the type of the member 'executable' of a type (line 118)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 12), self_17353, 'executable', normpath_call_result_17352)

            if more_types_in_union_17345:
                # SSA join for if statement (line 117)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_17354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17354)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_17354


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build.run.__dict__.__setitem__('stypy_localization', localization)
        build.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        build.run.__dict__.__setitem__('stypy_function_name', 'build.run')
        build.run.__dict__.__setitem__('stypy_param_names_list', [])
        build.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        build.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        build.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        build.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build.run', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to get_sub_commands(...): (line 126)
        # Processing the call keyword arguments (line 126)
        kwargs_17357 = {}
        # Getting the type of 'self' (line 126)
        self_17355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'self', False)
        # Obtaining the member 'get_sub_commands' of a type (line 126)
        get_sub_commands_17356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 24), self_17355, 'get_sub_commands')
        # Calling get_sub_commands(args, kwargs) (line 126)
        get_sub_commands_call_result_17358 = invoke(stypy.reporting.localization.Localization(__file__, 126, 24), get_sub_commands_17356, *[], **kwargs_17357)
        
        # Testing the type of a for loop iterable (line 126)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 126, 8), get_sub_commands_call_result_17358)
        # Getting the type of the for loop variable (line 126)
        for_loop_var_17359 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 126, 8), get_sub_commands_call_result_17358)
        # Assigning a type to the variable 'cmd_name' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'cmd_name', for_loop_var_17359)
        # SSA begins for a for statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to run_command(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'cmd_name' (line 127)
        cmd_name_17362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 29), 'cmd_name', False)
        # Processing the call keyword arguments (line 127)
        kwargs_17363 = {}
        # Getting the type of 'self' (line 127)
        self_17360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'self', False)
        # Obtaining the member 'run_command' of a type (line 127)
        run_command_17361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), self_17360, 'run_command')
        # Calling run_command(args, kwargs) (line 127)
        run_command_call_result_17364 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), run_command_17361, *[cmd_name_17362], **kwargs_17363)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_17365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17365)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_17365


    @norecursion
    def has_pure_modules(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_pure_modules'
        module_type_store = module_type_store.open_function_context('has_pure_modules', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build.has_pure_modules.__dict__.__setitem__('stypy_localization', localization)
        build.has_pure_modules.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build.has_pure_modules.__dict__.__setitem__('stypy_type_store', module_type_store)
        build.has_pure_modules.__dict__.__setitem__('stypy_function_name', 'build.has_pure_modules')
        build.has_pure_modules.__dict__.__setitem__('stypy_param_names_list', [])
        build.has_pure_modules.__dict__.__setitem__('stypy_varargs_param_name', None)
        build.has_pure_modules.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build.has_pure_modules.__dict__.__setitem__('stypy_call_defaults', defaults)
        build.has_pure_modules.__dict__.__setitem__('stypy_call_varargs', varargs)
        build.has_pure_modules.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build.has_pure_modules.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build.has_pure_modules', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_pure_modules', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_pure_modules(...)' code ##################

        
        # Call to has_pure_modules(...): (line 132)
        # Processing the call keyword arguments (line 132)
        kwargs_17369 = {}
        # Getting the type of 'self' (line 132)
        self_17366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'self', False)
        # Obtaining the member 'distribution' of a type (line 132)
        distribution_17367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 15), self_17366, 'distribution')
        # Obtaining the member 'has_pure_modules' of a type (line 132)
        has_pure_modules_17368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 15), distribution_17367, 'has_pure_modules')
        # Calling has_pure_modules(args, kwargs) (line 132)
        has_pure_modules_call_result_17370 = invoke(stypy.reporting.localization.Localization(__file__, 132, 15), has_pure_modules_17368, *[], **kwargs_17369)
        
        # Assigning a type to the variable 'stypy_return_type' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'stypy_return_type', has_pure_modules_call_result_17370)
        
        # ################# End of 'has_pure_modules(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_pure_modules' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_17371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17371)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_pure_modules'
        return stypy_return_type_17371


    @norecursion
    def has_c_libraries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_c_libraries'
        module_type_store = module_type_store.open_function_context('has_c_libraries', 134, 4, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build.has_c_libraries.__dict__.__setitem__('stypy_localization', localization)
        build.has_c_libraries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build.has_c_libraries.__dict__.__setitem__('stypy_type_store', module_type_store)
        build.has_c_libraries.__dict__.__setitem__('stypy_function_name', 'build.has_c_libraries')
        build.has_c_libraries.__dict__.__setitem__('stypy_param_names_list', [])
        build.has_c_libraries.__dict__.__setitem__('stypy_varargs_param_name', None)
        build.has_c_libraries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build.has_c_libraries.__dict__.__setitem__('stypy_call_defaults', defaults)
        build.has_c_libraries.__dict__.__setitem__('stypy_call_varargs', varargs)
        build.has_c_libraries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build.has_c_libraries.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build.has_c_libraries', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_c_libraries', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_c_libraries(...)' code ##################

        
        # Call to has_c_libraries(...): (line 135)
        # Processing the call keyword arguments (line 135)
        kwargs_17375 = {}
        # Getting the type of 'self' (line 135)
        self_17372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'self', False)
        # Obtaining the member 'distribution' of a type (line 135)
        distribution_17373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 15), self_17372, 'distribution')
        # Obtaining the member 'has_c_libraries' of a type (line 135)
        has_c_libraries_17374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 15), distribution_17373, 'has_c_libraries')
        # Calling has_c_libraries(args, kwargs) (line 135)
        has_c_libraries_call_result_17376 = invoke(stypy.reporting.localization.Localization(__file__, 135, 15), has_c_libraries_17374, *[], **kwargs_17375)
        
        # Assigning a type to the variable 'stypy_return_type' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'stypy_return_type', has_c_libraries_call_result_17376)
        
        # ################# End of 'has_c_libraries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_c_libraries' in the type store
        # Getting the type of 'stypy_return_type' (line 134)
        stypy_return_type_17377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17377)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_c_libraries'
        return stypy_return_type_17377


    @norecursion
    def has_ext_modules(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_ext_modules'
        module_type_store = module_type_store.open_function_context('has_ext_modules', 137, 4, False)
        # Assigning a type to the variable 'self' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build.has_ext_modules.__dict__.__setitem__('stypy_localization', localization)
        build.has_ext_modules.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build.has_ext_modules.__dict__.__setitem__('stypy_type_store', module_type_store)
        build.has_ext_modules.__dict__.__setitem__('stypy_function_name', 'build.has_ext_modules')
        build.has_ext_modules.__dict__.__setitem__('stypy_param_names_list', [])
        build.has_ext_modules.__dict__.__setitem__('stypy_varargs_param_name', None)
        build.has_ext_modules.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build.has_ext_modules.__dict__.__setitem__('stypy_call_defaults', defaults)
        build.has_ext_modules.__dict__.__setitem__('stypy_call_varargs', varargs)
        build.has_ext_modules.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build.has_ext_modules.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build.has_ext_modules', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_ext_modules', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_ext_modules(...)' code ##################

        
        # Call to has_ext_modules(...): (line 138)
        # Processing the call keyword arguments (line 138)
        kwargs_17381 = {}
        # Getting the type of 'self' (line 138)
        self_17378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'self', False)
        # Obtaining the member 'distribution' of a type (line 138)
        distribution_17379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), self_17378, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 138)
        has_ext_modules_17380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), distribution_17379, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 138)
        has_ext_modules_call_result_17382 = invoke(stypy.reporting.localization.Localization(__file__, 138, 15), has_ext_modules_17380, *[], **kwargs_17381)
        
        # Assigning a type to the variable 'stypy_return_type' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'stypy_return_type', has_ext_modules_call_result_17382)
        
        # ################# End of 'has_ext_modules(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_ext_modules' in the type store
        # Getting the type of 'stypy_return_type' (line 137)
        stypy_return_type_17383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17383)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_ext_modules'
        return stypy_return_type_17383


    @norecursion
    def has_scripts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_scripts'
        module_type_store = module_type_store.open_function_context('has_scripts', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build.has_scripts.__dict__.__setitem__('stypy_localization', localization)
        build.has_scripts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build.has_scripts.__dict__.__setitem__('stypy_type_store', module_type_store)
        build.has_scripts.__dict__.__setitem__('stypy_function_name', 'build.has_scripts')
        build.has_scripts.__dict__.__setitem__('stypy_param_names_list', [])
        build.has_scripts.__dict__.__setitem__('stypy_varargs_param_name', None)
        build.has_scripts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build.has_scripts.__dict__.__setitem__('stypy_call_defaults', defaults)
        build.has_scripts.__dict__.__setitem__('stypy_call_varargs', varargs)
        build.has_scripts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build.has_scripts.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build.has_scripts', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_scripts', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_scripts(...)' code ##################

        
        # Call to has_scripts(...): (line 141)
        # Processing the call keyword arguments (line 141)
        kwargs_17387 = {}
        # Getting the type of 'self' (line 141)
        self_17384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'self', False)
        # Obtaining the member 'distribution' of a type (line 141)
        distribution_17385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 15), self_17384, 'distribution')
        # Obtaining the member 'has_scripts' of a type (line 141)
        has_scripts_17386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 15), distribution_17385, 'has_scripts')
        # Calling has_scripts(args, kwargs) (line 141)
        has_scripts_call_result_17388 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), has_scripts_17386, *[], **kwargs_17387)
        
        # Assigning a type to the variable 'stypy_return_type' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'stypy_return_type', has_scripts_call_result_17388)
        
        # ################# End of 'has_scripts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_scripts' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_17389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17389)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_scripts'
        return stypy_return_type_17389


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 17, 0, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'build' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'build', build)

# Assigning a Str to a Name (line 19):
str_17390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 18), 'str', 'build everything needed to install')
# Getting the type of 'build'
build_17391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_17391, 'description', str_17390)

# Assigning a List to a Name (line 21):

# Obtaining an instance of the builtin type 'list' (line 21)
list_17392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 22)
tuple_17393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 22)
# Adding element type (line 22)
str_17394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'str', 'build-base=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_17393, str_17394)
# Adding element type (line 22)
str_17395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 24), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_17393, str_17395)
# Adding element type (line 22)
str_17396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'str', 'base directory for build library')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_17393, str_17396)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_17392, tuple_17393)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_17397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
str_17398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'str', 'build-purelib=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_17397, str_17398)
# Adding element type (line 24)
# Getting the type of 'None' (line 24)
None_17399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 27), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_17397, None_17399)
# Adding element type (line 24)
str_17400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'str', 'build directory for platform-neutral distributions')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_17397, str_17400)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_17392, tuple_17397)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 26)
tuple_17401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 26)
# Adding element type (line 26)
str_17402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 9), 'str', 'build-platlib=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 9), tuple_17401, str_17402)
# Adding element type (line 26)
# Getting the type of 'None' (line 26)
None_17403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 27), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 9), tuple_17401, None_17403)
# Adding element type (line 26)
str_17404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 9), 'str', 'build directory for platform-specific distributions')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 9), tuple_17401, str_17404)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_17392, tuple_17401)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 28)
tuple_17405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 28)
# Adding element type (line 28)
str_17406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 9), 'str', 'build-lib=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), tuple_17405, str_17406)
# Adding element type (line 28)
# Getting the type of 'None' (line 28)
None_17407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), tuple_17405, None_17407)
# Adding element type (line 28)
str_17408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 9), 'str', 'build directory for all distribution (defaults to either ')
str_17409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'str', 'build-purelib or build-platlib')
# Applying the binary operator '+' (line 29)
result_add_17410 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 9), '+', str_17408, str_17409)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), tuple_17405, result_add_17410)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_17392, tuple_17405)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 31)
tuple_17411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 31)
# Adding element type (line 31)
str_17412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 9), 'str', 'build-scripts=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), tuple_17411, str_17412)
# Adding element type (line 31)
# Getting the type of 'None' (line 31)
None_17413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 27), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), tuple_17411, None_17413)
# Adding element type (line 31)
str_17414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 9), 'str', 'build directory for scripts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), tuple_17411, str_17414)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_17392, tuple_17411)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 33)
tuple_17415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 33)
# Adding element type (line 33)
str_17416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 9), 'str', 'build-temp=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_17415, str_17416)
# Adding element type (line 33)
str_17417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 24), 'str', 't')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_17415, str_17417)
# Adding element type (line 33)
str_17418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 9), 'str', 'temporary build directory')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_17415, str_17418)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_17392, tuple_17415)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_17419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
str_17420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 9), 'str', 'plat-name=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_17419, str_17420)
# Adding element type (line 35)
str_17421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'str', 'p')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_17419, str_17421)
# Adding element type (line 35)
str_17422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 9), 'str', 'platform name to build for, if supported (default: %s)')

# Call to get_platform(...): (line 37)
# Processing the call keyword arguments (line 37)
kwargs_17424 = {}
# Getting the type of 'get_platform' (line 37)
get_platform_17423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 27), 'get_platform', False)
# Calling get_platform(args, kwargs) (line 37)
get_platform_call_result_17425 = invoke(stypy.reporting.localization.Localization(__file__, 37, 27), get_platform_17423, *[], **kwargs_17424)

# Applying the binary operator '%' (line 36)
result_mod_17426 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 9), '%', str_17422, get_platform_call_result_17425)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_17419, result_mod_17426)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_17392, tuple_17419)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 38)
tuple_17427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 38)
# Adding element type (line 38)
str_17428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 9), 'str', 'compiler=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 9), tuple_17427, str_17428)
# Adding element type (line 38)
str_17429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 22), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 9), tuple_17427, str_17429)
# Adding element type (line 38)
str_17430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 9), 'str', 'specify the compiler type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 9), tuple_17427, str_17430)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_17392, tuple_17427)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 40)
tuple_17431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 40)
# Adding element type (line 40)
str_17432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 9), 'str', 'debug')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 9), tuple_17431, str_17432)
# Adding element type (line 40)
str_17433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 18), 'str', 'g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 9), tuple_17431, str_17433)
# Adding element type (line 40)
str_17434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 9), 'str', 'compile extensions and libraries with debugging information')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 9), tuple_17431, str_17434)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_17392, tuple_17431)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 42)
tuple_17435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 42)
# Adding element type (line 42)
str_17436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 9), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), tuple_17435, str_17436)
# Adding element type (line 42)
str_17437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 18), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), tuple_17435, str_17437)
# Adding element type (line 42)
str_17438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 9), 'str', 'forcibly build everything (ignore file timestamps)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), tuple_17435, str_17438)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_17392, tuple_17435)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 44)
tuple_17439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 44)
# Adding element type (line 44)
str_17440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 9), 'str', 'executable=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 9), tuple_17439, str_17440)
# Adding element type (line 44)
str_17441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 24), 'str', 'e')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 9), tuple_17439, str_17441)
# Adding element type (line 44)
str_17442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 9), 'str', 'specify final destination interpreter path (build.py)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 9), tuple_17439, str_17442)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_17392, tuple_17439)

# Getting the type of 'build'
build_17443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_17443, 'user_options', list_17392)

# Assigning a List to a Name (line 48):

# Obtaining an instance of the builtin type 'list' (line 48)
list_17444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 48)
# Adding element type (line 48)
str_17445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 23), 'str', 'debug')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 22), list_17444, str_17445)
# Adding element type (line 48)
str_17446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 32), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 22), list_17444, str_17446)

# Getting the type of 'build'
build_17447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_17447, 'boolean_options', list_17444)

# Assigning a List to a Name (line 50):

# Obtaining an instance of the builtin type 'list' (line 50)
list_17448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 50)
# Adding element type (line 50)

# Obtaining an instance of the builtin type 'tuple' (line 51)
tuple_17449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 51)
# Adding element type (line 51)
str_17450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 9), 'str', 'help-compiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 9), tuple_17449, str_17450)
# Adding element type (line 51)
# Getting the type of 'None' (line 51)
None_17451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 9), tuple_17449, None_17451)
# Adding element type (line 51)
str_17452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 9), 'str', 'list available compilers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 9), tuple_17449, str_17452)
# Adding element type (line 51)
# Getting the type of 'show_compilers' (line 52)
show_compilers_17453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 37), 'show_compilers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 9), tuple_17449, show_compilers_17453)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 19), list_17448, tuple_17449)

# Getting the type of 'build'
build_17454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build')
# Setting the type of the member 'help_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_17454, 'help_options', list_17448)

# Assigning a List to a Name (line 143):

# Obtaining an instance of the builtin type 'list' (line 143)
list_17455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 143)
# Adding element type (line 143)

# Obtaining an instance of the builtin type 'tuple' (line 143)
tuple_17456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 143)
# Adding element type (line 143)
str_17457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 21), 'str', 'build_py')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 21), tuple_17456, str_17457)
# Adding element type (line 143)
# Getting the type of 'build'
build_17458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build')
# Obtaining the member 'has_pure_modules' of a type
has_pure_modules_17459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_17458, 'has_pure_modules')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 21), tuple_17456, has_pure_modules_17459)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 19), list_17455, tuple_17456)
# Adding element type (line 143)

# Obtaining an instance of the builtin type 'tuple' (line 144)
tuple_17460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 144)
# Adding element type (line 144)
str_17461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 21), 'str', 'build_clib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 21), tuple_17460, str_17461)
# Adding element type (line 144)
# Getting the type of 'build'
build_17462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build')
# Obtaining the member 'has_c_libraries' of a type
has_c_libraries_17463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_17462, 'has_c_libraries')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 21), tuple_17460, has_c_libraries_17463)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 19), list_17455, tuple_17460)
# Adding element type (line 143)

# Obtaining an instance of the builtin type 'tuple' (line 145)
tuple_17464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 145)
# Adding element type (line 145)
str_17465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 21), 'str', 'build_ext')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 21), tuple_17464, str_17465)
# Adding element type (line 145)
# Getting the type of 'build'
build_17466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build')
# Obtaining the member 'has_ext_modules' of a type
has_ext_modules_17467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_17466, 'has_ext_modules')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 21), tuple_17464, has_ext_modules_17467)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 19), list_17455, tuple_17464)
# Adding element type (line 143)

# Obtaining an instance of the builtin type 'tuple' (line 146)
tuple_17468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 146)
# Adding element type (line 146)
str_17469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 21), 'str', 'build_scripts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 21), tuple_17468, str_17469)
# Adding element type (line 146)
# Getting the type of 'build'
build_17470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build')
# Obtaining the member 'has_scripts' of a type
has_scripts_17471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_17470, 'has_scripts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 21), tuple_17468, has_scripts_17471)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 19), list_17455, tuple_17468)

# Getting the type of 'build'
build_17472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build')
# Setting the type of the member 'sub_commands' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_17472, 'sub_commands', list_17455)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
