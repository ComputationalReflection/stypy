
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.build_scripts
2: 
3: Implements the Distutils 'build_scripts' command.'''
4: 
5: __revision__ = "$Id$"
6: 
7: import os, re
8: from stat import ST_MODE
9: from distutils.core import Command
10: from distutils.dep_util import newer
11: from distutils.util import convert_path
12: from distutils import log
13: 
14: # check if Python is called on the first line with this expression
15: first_line_re = re.compile('^#!.*python[0-9.]*([ \t].*)?$')
16: 
17: class build_scripts (Command):
18: 
19:     description = "\"build\" scripts (copy and fixup #! line)"
20: 
21:     user_options = [
22:         ('build-dir=', 'd', "directory to \"build\" (copy) to"),
23:         ('force', 'f', "forcibly build everything (ignore file timestamps"),
24:         ('executable=', 'e', "specify final destination interpreter path"),
25:         ]
26: 
27:     boolean_options = ['force']
28: 
29: 
30:     def initialize_options (self):
31:         self.build_dir = None
32:         self.scripts = None
33:         self.force = None
34:         self.executable = None
35:         self.outfiles = None
36: 
37:     def finalize_options (self):
38:         self.set_undefined_options('build',
39:                                    ('build_scripts', 'build_dir'),
40:                                    ('force', 'force'),
41:                                    ('executable', 'executable'))
42:         self.scripts = self.distribution.scripts
43: 
44:     def get_source_files(self):
45:         return self.scripts
46: 
47:     def run (self):
48:         if not self.scripts:
49:             return
50:         self.copy_scripts()
51: 
52: 
53:     def copy_scripts (self):
54:         '''Copy each script listed in 'self.scripts'; if it's marked as a
55:         Python script in the Unix way (first line matches 'first_line_re',
56:         ie. starts with "\#!" and contains "python"), then adjust the first
57:         line to refer to the current Python interpreter as we copy.
58:         '''
59:         _sysconfig = __import__('sysconfig')
60:         self.mkpath(self.build_dir)
61:         outfiles = []
62:         for script in self.scripts:
63:             adjust = 0
64:             script = convert_path(script)
65:             outfile = os.path.join(self.build_dir, os.path.basename(script))
66:             outfiles.append(outfile)
67: 
68:             if not self.force and not newer(script, outfile):
69:                 log.debug("not copying %s (up-to-date)", script)
70:                 continue
71: 
72:             # Always open the file, but ignore failures in dry-run mode --
73:             # that way, we'll get accurate feedback if we can read the
74:             # script.
75:             try:
76:                 f = open(script, "r")
77:             except IOError:
78:                 if not self.dry_run:
79:                     raise
80:                 f = None
81:             else:
82:                 first_line = f.readline()
83:                 if not first_line:
84:                     self.warn("%s is an empty file (skipping)" % script)
85:                     continue
86: 
87:                 match = first_line_re.match(first_line)
88:                 if match:
89:                     adjust = 1
90:                     post_interp = match.group(1) or ''
91: 
92:             if adjust:
93:                 log.info("copying and adjusting %s -> %s", script,
94:                          self.build_dir)
95:                 if not self.dry_run:
96:                     outf = open(outfile, "w")
97:                     if not _sysconfig.is_python_build():
98:                         outf.write("#!%s%s\n" %
99:                                    (self.executable,
100:                                     post_interp))
101:                     else:
102:                         outf.write("#!%s%s\n" %
103:                                    (os.path.join(
104:                             _sysconfig.get_config_var("BINDIR"),
105:                            "python%s%s" % (_sysconfig.get_config_var("VERSION"),
106:                                            _sysconfig.get_config_var("EXE"))),
107:                                     post_interp))
108:                     outf.writelines(f.readlines())
109:                     outf.close()
110:                 if f:
111:                     f.close()
112:             else:
113:                 if f:
114:                     f.close()
115:                 self.copy_file(script, outfile)
116: 
117:         if os.name == 'posix':
118:             for file in outfiles:
119:                 if self.dry_run:
120:                     log.info("changing mode of %s", file)
121:                 else:
122:                     oldmode = os.stat(file)[ST_MODE] & 07777
123:                     newmode = (oldmode | 0555) & 07777
124:                     if newmode != oldmode:
125:                         log.info("changing mode of %s from %o to %o",
126:                                  file, oldmode, newmode)
127:                         os.chmod(file, newmode)
128: 
129:     # copy_scripts ()
130: 
131: # class build_scripts
132: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_20625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', "distutils.command.build_scripts\n\nImplements the Distutils 'build_scripts' command.")

# Assigning a Str to a Name (line 5):
str_20626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__revision__', str_20626)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# Multiple import statement. import os (1/2) (line 7)
import os

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'os', os, module_type_store)
# Multiple import statement. import re (2/2) (line 7)
import re

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from stat import ST_MODE' statement (line 8)
try:
    from stat import ST_MODE

except:
    ST_MODE = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stat', None, module_type_store, ['ST_MODE'], [ST_MODE])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.core import Command' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_20627 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.core')

if (type(import_20627) is not StypyTypeError):

    if (import_20627 != 'pyd_module'):
        __import__(import_20627)
        sys_modules_20628 = sys.modules[import_20627]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.core', sys_modules_20628.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_20628, sys_modules_20628.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.core', import_20627)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.dep_util import newer' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_20629 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.dep_util')

if (type(import_20629) is not StypyTypeError):

    if (import_20629 != 'pyd_module'):
        __import__(import_20629)
        sys_modules_20630 = sys.modules[import_20629]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.dep_util', sys_modules_20630.module_type_store, module_type_store, ['newer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_20630, sys_modules_20630.module_type_store, module_type_store)
    else:
        from distutils.dep_util import newer

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.dep_util', None, module_type_store, ['newer'], [newer])

else:
    # Assigning a type to the variable 'distutils.dep_util' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.dep_util', import_20629)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.util import convert_path' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_20631 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.util')

if (type(import_20631) is not StypyTypeError):

    if (import_20631 != 'pyd_module'):
        __import__(import_20631)
        sys_modules_20632 = sys.modules[import_20631]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.util', sys_modules_20632.module_type_store, module_type_store, ['convert_path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_20632, sys_modules_20632.module_type_store, module_type_store)
    else:
        from distutils.util import convert_path

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.util', None, module_type_store, ['convert_path'], [convert_path])

else:
    # Assigning a type to the variable 'distutils.util' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.util', import_20631)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils import log' statement (line 12)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils', None, module_type_store, ['log'], [log])


# Assigning a Call to a Name (line 15):

# Call to compile(...): (line 15)
# Processing the call arguments (line 15)
str_20635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 27), 'str', '^#!.*python[0-9.]*([ \t].*)?$')
# Processing the call keyword arguments (line 15)
kwargs_20636 = {}
# Getting the type of 're' (line 15)
re_20633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 16), 're', False)
# Obtaining the member 'compile' of a type (line 15)
compile_20634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 16), re_20633, 'compile')
# Calling compile(args, kwargs) (line 15)
compile_call_result_20637 = invoke(stypy.reporting.localization.Localization(__file__, 15, 16), compile_20634, *[str_20635], **kwargs_20636)

# Assigning a type to the variable 'first_line_re' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'first_line_re', compile_call_result_20637)
# Declaration of the 'build_scripts' class
# Getting the type of 'Command' (line 17)
Command_20638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 21), 'Command')

class build_scripts(Command_20638, ):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_scripts.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        build_scripts.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_scripts.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_scripts.initialize_options.__dict__.__setitem__('stypy_function_name', 'build_scripts.initialize_options')
        build_scripts.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build_scripts.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_scripts.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_scripts.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_scripts.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_scripts.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_scripts.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_scripts.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 31):
        # Getting the type of 'None' (line 31)
        None_20639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'None')
        # Getting the type of 'self' (line 31)
        self_20640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'build_dir' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_20640, 'build_dir', None_20639)
        
        # Assigning a Name to a Attribute (line 32):
        # Getting the type of 'None' (line 32)
        None_20641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'None')
        # Getting the type of 'self' (line 32)
        self_20642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member 'scripts' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_20642, 'scripts', None_20641)
        
        # Assigning a Name to a Attribute (line 33):
        # Getting the type of 'None' (line 33)
        None_20643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 21), 'None')
        # Getting the type of 'self' (line 33)
        self_20644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'force' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_20644, 'force', None_20643)
        
        # Assigning a Name to a Attribute (line 34):
        # Getting the type of 'None' (line 34)
        None_20645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 26), 'None')
        # Getting the type of 'self' (line 34)
        self_20646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'executable' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_20646, 'executable', None_20645)
        
        # Assigning a Name to a Attribute (line 35):
        # Getting the type of 'None' (line 35)
        None_20647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 24), 'None')
        # Getting the type of 'self' (line 35)
        self_20648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self')
        # Setting the type of the member 'outfiles' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_20648, 'outfiles', None_20647)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_20649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20649)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_20649


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_scripts.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        build_scripts.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_scripts.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_scripts.finalize_options.__dict__.__setitem__('stypy_function_name', 'build_scripts.finalize_options')
        build_scripts.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build_scripts.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_scripts.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_scripts.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_scripts.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_scripts.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_scripts.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_scripts.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_undefined_options(...): (line 38)
        # Processing the call arguments (line 38)
        str_20652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 35), 'str', 'build')
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_20653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        str_20654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'str', 'build_scripts')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 36), tuple_20653, str_20654)
        # Adding element type (line 39)
        str_20655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 53), 'str', 'build_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 36), tuple_20653, str_20655)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 40)
        tuple_20656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 40)
        # Adding element type (line 40)
        str_20657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 36), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 36), tuple_20656, str_20657)
        # Adding element type (line 40)
        str_20658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 45), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 36), tuple_20656, str_20658)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_20659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        str_20660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 36), 'str', 'executable')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 36), tuple_20659, str_20660)
        # Adding element type (line 41)
        str_20661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 50), 'str', 'executable')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 36), tuple_20659, str_20661)
        
        # Processing the call keyword arguments (line 38)
        kwargs_20662 = {}
        # Getting the type of 'self' (line 38)
        self_20650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 38)
        set_undefined_options_20651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_20650, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 38)
        set_undefined_options_call_result_20663 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), set_undefined_options_20651, *[str_20652, tuple_20653, tuple_20656, tuple_20659], **kwargs_20662)
        
        
        # Assigning a Attribute to a Attribute (line 42):
        # Getting the type of 'self' (line 42)
        self_20664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'self')
        # Obtaining the member 'distribution' of a type (line 42)
        distribution_20665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 23), self_20664, 'distribution')
        # Obtaining the member 'scripts' of a type (line 42)
        scripts_20666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 23), distribution_20665, 'scripts')
        # Getting the type of 'self' (line 42)
        self_20667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self')
        # Setting the type of the member 'scripts' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_20667, 'scripts', scripts_20666)
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_20668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20668)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_20668


    @norecursion
    def get_source_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_source_files'
        module_type_store = module_type_store.open_function_context('get_source_files', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_scripts.get_source_files.__dict__.__setitem__('stypy_localization', localization)
        build_scripts.get_source_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_scripts.get_source_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_scripts.get_source_files.__dict__.__setitem__('stypy_function_name', 'build_scripts.get_source_files')
        build_scripts.get_source_files.__dict__.__setitem__('stypy_param_names_list', [])
        build_scripts.get_source_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_scripts.get_source_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_scripts.get_source_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_scripts.get_source_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_scripts.get_source_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_scripts.get_source_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_scripts.get_source_files', [], None, None, defaults, varargs, kwargs)

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

        # Getting the type of 'self' (line 45)
        self_20669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'self')
        # Obtaining the member 'scripts' of a type (line 45)
        scripts_20670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 15), self_20669, 'scripts')
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', scripts_20670)
        
        # ################# End of 'get_source_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_source_files' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_20671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20671)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_source_files'
        return stypy_return_type_20671


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_scripts.run.__dict__.__setitem__('stypy_localization', localization)
        build_scripts.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_scripts.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_scripts.run.__dict__.__setitem__('stypy_function_name', 'build_scripts.run')
        build_scripts.run.__dict__.__setitem__('stypy_param_names_list', [])
        build_scripts.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_scripts.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_scripts.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_scripts.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_scripts.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_scripts.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_scripts.run', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 48)
        self_20672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'self')
        # Obtaining the member 'scripts' of a type (line 48)
        scripts_20673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 15), self_20672, 'scripts')
        # Applying the 'not' unary operator (line 48)
        result_not__20674 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 11), 'not', scripts_20673)
        
        # Testing the type of an if condition (line 48)
        if_condition_20675 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 8), result_not__20674)
        # Assigning a type to the variable 'if_condition_20675' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'if_condition_20675', if_condition_20675)
        # SSA begins for if statement (line 48)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 48)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to copy_scripts(...): (line 50)
        # Processing the call keyword arguments (line 50)
        kwargs_20678 = {}
        # Getting the type of 'self' (line 50)
        self_20676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'self', False)
        # Obtaining the member 'copy_scripts' of a type (line 50)
        copy_scripts_20677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), self_20676, 'copy_scripts')
        # Calling copy_scripts(args, kwargs) (line 50)
        copy_scripts_call_result_20679 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), copy_scripts_20677, *[], **kwargs_20678)
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_20680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20680)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_20680


    @norecursion
    def copy_scripts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy_scripts'
        module_type_store = module_type_store.open_function_context('copy_scripts', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_scripts.copy_scripts.__dict__.__setitem__('stypy_localization', localization)
        build_scripts.copy_scripts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_scripts.copy_scripts.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_scripts.copy_scripts.__dict__.__setitem__('stypy_function_name', 'build_scripts.copy_scripts')
        build_scripts.copy_scripts.__dict__.__setitem__('stypy_param_names_list', [])
        build_scripts.copy_scripts.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_scripts.copy_scripts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_scripts.copy_scripts.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_scripts.copy_scripts.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_scripts.copy_scripts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_scripts.copy_scripts.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_scripts.copy_scripts', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy_scripts', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy_scripts(...)' code ##################

        str_20681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, (-1)), 'str', 'Copy each script listed in \'self.scripts\'; if it\'s marked as a\n        Python script in the Unix way (first line matches \'first_line_re\',\n        ie. starts with "\\#!" and contains "python"), then adjust the first\n        line to refer to the current Python interpreter as we copy.\n        ')
        
        # Assigning a Call to a Name (line 59):
        
        # Call to __import__(...): (line 59)
        # Processing the call arguments (line 59)
        str_20683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 32), 'str', 'sysconfig')
        # Processing the call keyword arguments (line 59)
        kwargs_20684 = {}
        # Getting the type of '__import__' (line 59)
        import___20682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), '__import__', False)
        # Calling __import__(args, kwargs) (line 59)
        import___call_result_20685 = invoke(stypy.reporting.localization.Localization(__file__, 59, 21), import___20682, *[str_20683], **kwargs_20684)
        
        # Assigning a type to the variable '_sysconfig' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), '_sysconfig', import___call_result_20685)
        
        # Call to mkpath(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'self' (line 60)
        self_20688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'self', False)
        # Obtaining the member 'build_dir' of a type (line 60)
        build_dir_20689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 20), self_20688, 'build_dir')
        # Processing the call keyword arguments (line 60)
        kwargs_20690 = {}
        # Getting the type of 'self' (line 60)
        self_20686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 60)
        mkpath_20687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_20686, 'mkpath')
        # Calling mkpath(args, kwargs) (line 60)
        mkpath_call_result_20691 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), mkpath_20687, *[build_dir_20689], **kwargs_20690)
        
        
        # Assigning a List to a Name (line 61):
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_20692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        
        # Assigning a type to the variable 'outfiles' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'outfiles', list_20692)
        
        # Getting the type of 'self' (line 62)
        self_20693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 22), 'self')
        # Obtaining the member 'scripts' of a type (line 62)
        scripts_20694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 22), self_20693, 'scripts')
        # Testing the type of a for loop iterable (line 62)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 62, 8), scripts_20694)
        # Getting the type of the for loop variable (line 62)
        for_loop_var_20695 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 62, 8), scripts_20694)
        # Assigning a type to the variable 'script' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'script', for_loop_var_20695)
        # SSA begins for a for statement (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Name (line 63):
        int_20696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 21), 'int')
        # Assigning a type to the variable 'adjust' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'adjust', int_20696)
        
        # Assigning a Call to a Name (line 64):
        
        # Call to convert_path(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'script' (line 64)
        script_20698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'script', False)
        # Processing the call keyword arguments (line 64)
        kwargs_20699 = {}
        # Getting the type of 'convert_path' (line 64)
        convert_path_20697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'convert_path', False)
        # Calling convert_path(args, kwargs) (line 64)
        convert_path_call_result_20700 = invoke(stypy.reporting.localization.Localization(__file__, 64, 21), convert_path_20697, *[script_20698], **kwargs_20699)
        
        # Assigning a type to the variable 'script' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'script', convert_path_call_result_20700)
        
        # Assigning a Call to a Name (line 65):
        
        # Call to join(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'self' (line 65)
        self_20704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 35), 'self', False)
        # Obtaining the member 'build_dir' of a type (line 65)
        build_dir_20705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 35), self_20704, 'build_dir')
        
        # Call to basename(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'script' (line 65)
        script_20709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 68), 'script', False)
        # Processing the call keyword arguments (line 65)
        kwargs_20710 = {}
        # Getting the type of 'os' (line 65)
        os_20706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 51), 'os', False)
        # Obtaining the member 'path' of a type (line 65)
        path_20707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 51), os_20706, 'path')
        # Obtaining the member 'basename' of a type (line 65)
        basename_20708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 51), path_20707, 'basename')
        # Calling basename(args, kwargs) (line 65)
        basename_call_result_20711 = invoke(stypy.reporting.localization.Localization(__file__, 65, 51), basename_20708, *[script_20709], **kwargs_20710)
        
        # Processing the call keyword arguments (line 65)
        kwargs_20712 = {}
        # Getting the type of 'os' (line 65)
        os_20701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 65)
        path_20702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 22), os_20701, 'path')
        # Obtaining the member 'join' of a type (line 65)
        join_20703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 22), path_20702, 'join')
        # Calling join(args, kwargs) (line 65)
        join_call_result_20713 = invoke(stypy.reporting.localization.Localization(__file__, 65, 22), join_20703, *[build_dir_20705, basename_call_result_20711], **kwargs_20712)
        
        # Assigning a type to the variable 'outfile' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'outfile', join_call_result_20713)
        
        # Call to append(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'outfile' (line 66)
        outfile_20716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 28), 'outfile', False)
        # Processing the call keyword arguments (line 66)
        kwargs_20717 = {}
        # Getting the type of 'outfiles' (line 66)
        outfiles_20714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'outfiles', False)
        # Obtaining the member 'append' of a type (line 66)
        append_20715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), outfiles_20714, 'append')
        # Calling append(args, kwargs) (line 66)
        append_call_result_20718 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), append_20715, *[outfile_20716], **kwargs_20717)
        
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 68)
        self_20719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'self')
        # Obtaining the member 'force' of a type (line 68)
        force_20720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), self_20719, 'force')
        # Applying the 'not' unary operator (line 68)
        result_not__20721 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 15), 'not', force_20720)
        
        
        
        # Call to newer(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'script' (line 68)
        script_20723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 44), 'script', False)
        # Getting the type of 'outfile' (line 68)
        outfile_20724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 52), 'outfile', False)
        # Processing the call keyword arguments (line 68)
        kwargs_20725 = {}
        # Getting the type of 'newer' (line 68)
        newer_20722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 38), 'newer', False)
        # Calling newer(args, kwargs) (line 68)
        newer_call_result_20726 = invoke(stypy.reporting.localization.Localization(__file__, 68, 38), newer_20722, *[script_20723, outfile_20724], **kwargs_20725)
        
        # Applying the 'not' unary operator (line 68)
        result_not__20727 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 34), 'not', newer_call_result_20726)
        
        # Applying the binary operator 'and' (line 68)
        result_and_keyword_20728 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 15), 'and', result_not__20721, result_not__20727)
        
        # Testing the type of an if condition (line 68)
        if_condition_20729 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 12), result_and_keyword_20728)
        # Assigning a type to the variable 'if_condition_20729' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'if_condition_20729', if_condition_20729)
        # SSA begins for if statement (line 68)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to debug(...): (line 69)
        # Processing the call arguments (line 69)
        str_20732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 26), 'str', 'not copying %s (up-to-date)')
        # Getting the type of 'script' (line 69)
        script_20733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 57), 'script', False)
        # Processing the call keyword arguments (line 69)
        kwargs_20734 = {}
        # Getting the type of 'log' (line 69)
        log_20730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'log', False)
        # Obtaining the member 'debug' of a type (line 69)
        debug_20731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 16), log_20730, 'debug')
        # Calling debug(args, kwargs) (line 69)
        debug_call_result_20735 = invoke(stypy.reporting.localization.Localization(__file__, 69, 16), debug_20731, *[str_20732, script_20733], **kwargs_20734)
        
        # SSA join for if statement (line 68)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 76):
        
        # Call to open(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'script' (line 76)
        script_20737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'script', False)
        str_20738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 33), 'str', 'r')
        # Processing the call keyword arguments (line 76)
        kwargs_20739 = {}
        # Getting the type of 'open' (line 76)
        open_20736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'open', False)
        # Calling open(args, kwargs) (line 76)
        open_call_result_20740 = invoke(stypy.reporting.localization.Localization(__file__, 76, 20), open_20736, *[script_20737, str_20738], **kwargs_20739)
        
        # Assigning a type to the variable 'f' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'f', open_call_result_20740)
        # SSA branch for the except part of a try statement (line 75)
        # SSA branch for the except 'IOError' branch of a try statement (line 75)
        module_type_store.open_ssa_branch('except')
        
        
        # Getting the type of 'self' (line 78)
        self_20741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'self')
        # Obtaining the member 'dry_run' of a type (line 78)
        dry_run_20742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 23), self_20741, 'dry_run')
        # Applying the 'not' unary operator (line 78)
        result_not__20743 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 19), 'not', dry_run_20742)
        
        # Testing the type of an if condition (line 78)
        if_condition_20744 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 16), result_not__20743)
        # Assigning a type to the variable 'if_condition_20744' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'if_condition_20744', if_condition_20744)
        # SSA begins for if statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 78)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 80):
        # Getting the type of 'None' (line 80)
        None_20745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'None')
        # Assigning a type to the variable 'f' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'f', None_20745)
        # SSA branch for the else branch of a try statement (line 75)
        module_type_store.open_ssa_branch('except else')
        
        # Assigning a Call to a Name (line 82):
        
        # Call to readline(...): (line 82)
        # Processing the call keyword arguments (line 82)
        kwargs_20748 = {}
        # Getting the type of 'f' (line 82)
        f_20746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'f', False)
        # Obtaining the member 'readline' of a type (line 82)
        readline_20747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 29), f_20746, 'readline')
        # Calling readline(args, kwargs) (line 82)
        readline_call_result_20749 = invoke(stypy.reporting.localization.Localization(__file__, 82, 29), readline_20747, *[], **kwargs_20748)
        
        # Assigning a type to the variable 'first_line' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'first_line', readline_call_result_20749)
        
        
        # Getting the type of 'first_line' (line 83)
        first_line_20750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 23), 'first_line')
        # Applying the 'not' unary operator (line 83)
        result_not__20751 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 19), 'not', first_line_20750)
        
        # Testing the type of an if condition (line 83)
        if_condition_20752 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 16), result_not__20751)
        # Assigning a type to the variable 'if_condition_20752' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'if_condition_20752', if_condition_20752)
        # SSA begins for if statement (line 83)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 84)
        # Processing the call arguments (line 84)
        str_20755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 30), 'str', '%s is an empty file (skipping)')
        # Getting the type of 'script' (line 84)
        script_20756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 65), 'script', False)
        # Applying the binary operator '%' (line 84)
        result_mod_20757 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 30), '%', str_20755, script_20756)
        
        # Processing the call keyword arguments (line 84)
        kwargs_20758 = {}
        # Getting the type of 'self' (line 84)
        self_20753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'self', False)
        # Obtaining the member 'warn' of a type (line 84)
        warn_20754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 20), self_20753, 'warn')
        # Calling warn(args, kwargs) (line 84)
        warn_call_result_20759 = invoke(stypy.reporting.localization.Localization(__file__, 84, 20), warn_20754, *[result_mod_20757], **kwargs_20758)
        
        # SSA join for if statement (line 83)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 87):
        
        # Call to match(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'first_line' (line 87)
        first_line_20762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 44), 'first_line', False)
        # Processing the call keyword arguments (line 87)
        kwargs_20763 = {}
        # Getting the type of 'first_line_re' (line 87)
        first_line_re_20760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'first_line_re', False)
        # Obtaining the member 'match' of a type (line 87)
        match_20761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 24), first_line_re_20760, 'match')
        # Calling match(args, kwargs) (line 87)
        match_call_result_20764 = invoke(stypy.reporting.localization.Localization(__file__, 87, 24), match_20761, *[first_line_20762], **kwargs_20763)
        
        # Assigning a type to the variable 'match' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'match', match_call_result_20764)
        
        # Getting the type of 'match' (line 88)
        match_20765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'match')
        # Testing the type of an if condition (line 88)
        if_condition_20766 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 16), match_20765)
        # Assigning a type to the variable 'if_condition_20766' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'if_condition_20766', if_condition_20766)
        # SSA begins for if statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 89):
        int_20767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 29), 'int')
        # Assigning a type to the variable 'adjust' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'adjust', int_20767)
        
        # Assigning a BoolOp to a Name (line 90):
        
        # Evaluating a boolean operation
        
        # Call to group(...): (line 90)
        # Processing the call arguments (line 90)
        int_20770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 46), 'int')
        # Processing the call keyword arguments (line 90)
        kwargs_20771 = {}
        # Getting the type of 'match' (line 90)
        match_20768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 34), 'match', False)
        # Obtaining the member 'group' of a type (line 90)
        group_20769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 34), match_20768, 'group')
        # Calling group(args, kwargs) (line 90)
        group_call_result_20772 = invoke(stypy.reporting.localization.Localization(__file__, 90, 34), group_20769, *[int_20770], **kwargs_20771)
        
        str_20773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 52), 'str', '')
        # Applying the binary operator 'or' (line 90)
        result_or_keyword_20774 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 34), 'or', group_call_result_20772, str_20773)
        
        # Assigning a type to the variable 'post_interp' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'post_interp', result_or_keyword_20774)
        # SSA join for if statement (line 88)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for try-except statement (line 75)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'adjust' (line 92)
        adjust_20775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'adjust')
        # Testing the type of an if condition (line 92)
        if_condition_20776 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 12), adjust_20775)
        # Assigning a type to the variable 'if_condition_20776' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'if_condition_20776', if_condition_20776)
        # SSA begins for if statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 93)
        # Processing the call arguments (line 93)
        str_20779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 25), 'str', 'copying and adjusting %s -> %s')
        # Getting the type of 'script' (line 93)
        script_20780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 59), 'script', False)
        # Getting the type of 'self' (line 94)
        self_20781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'self', False)
        # Obtaining the member 'build_dir' of a type (line 94)
        build_dir_20782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 25), self_20781, 'build_dir')
        # Processing the call keyword arguments (line 93)
        kwargs_20783 = {}
        # Getting the type of 'log' (line 93)
        log_20777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'log', False)
        # Obtaining the member 'info' of a type (line 93)
        info_20778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 16), log_20777, 'info')
        # Calling info(args, kwargs) (line 93)
        info_call_result_20784 = invoke(stypy.reporting.localization.Localization(__file__, 93, 16), info_20778, *[str_20779, script_20780, build_dir_20782], **kwargs_20783)
        
        
        
        # Getting the type of 'self' (line 95)
        self_20785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'self')
        # Obtaining the member 'dry_run' of a type (line 95)
        dry_run_20786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 23), self_20785, 'dry_run')
        # Applying the 'not' unary operator (line 95)
        result_not__20787 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 19), 'not', dry_run_20786)
        
        # Testing the type of an if condition (line 95)
        if_condition_20788 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 16), result_not__20787)
        # Assigning a type to the variable 'if_condition_20788' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'if_condition_20788', if_condition_20788)
        # SSA begins for if statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 96):
        
        # Call to open(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'outfile' (line 96)
        outfile_20790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 32), 'outfile', False)
        str_20791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 41), 'str', 'w')
        # Processing the call keyword arguments (line 96)
        kwargs_20792 = {}
        # Getting the type of 'open' (line 96)
        open_20789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 27), 'open', False)
        # Calling open(args, kwargs) (line 96)
        open_call_result_20793 = invoke(stypy.reporting.localization.Localization(__file__, 96, 27), open_20789, *[outfile_20790, str_20791], **kwargs_20792)
        
        # Assigning a type to the variable 'outf' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 20), 'outf', open_call_result_20793)
        
        
        
        # Call to is_python_build(...): (line 97)
        # Processing the call keyword arguments (line 97)
        kwargs_20796 = {}
        # Getting the type of '_sysconfig' (line 97)
        _sysconfig_20794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 27), '_sysconfig', False)
        # Obtaining the member 'is_python_build' of a type (line 97)
        is_python_build_20795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 27), _sysconfig_20794, 'is_python_build')
        # Calling is_python_build(args, kwargs) (line 97)
        is_python_build_call_result_20797 = invoke(stypy.reporting.localization.Localization(__file__, 97, 27), is_python_build_20795, *[], **kwargs_20796)
        
        # Applying the 'not' unary operator (line 97)
        result_not__20798 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 23), 'not', is_python_build_call_result_20797)
        
        # Testing the type of an if condition (line 97)
        if_condition_20799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 20), result_not__20798)
        # Assigning a type to the variable 'if_condition_20799' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'if_condition_20799', if_condition_20799)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 98)
        # Processing the call arguments (line 98)
        str_20802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 35), 'str', '#!%s%s\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 99)
        tuple_20803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 99)
        # Adding element type (line 99)
        # Getting the type of 'self' (line 99)
        self_20804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 36), 'self', False)
        # Obtaining the member 'executable' of a type (line 99)
        executable_20805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 36), self_20804, 'executable')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 36), tuple_20803, executable_20805)
        # Adding element type (line 99)
        # Getting the type of 'post_interp' (line 100)
        post_interp_20806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 36), 'post_interp', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 36), tuple_20803, post_interp_20806)
        
        # Applying the binary operator '%' (line 98)
        result_mod_20807 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 35), '%', str_20802, tuple_20803)
        
        # Processing the call keyword arguments (line 98)
        kwargs_20808 = {}
        # Getting the type of 'outf' (line 98)
        outf_20800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'outf', False)
        # Obtaining the member 'write' of a type (line 98)
        write_20801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 24), outf_20800, 'write')
        # Calling write(args, kwargs) (line 98)
        write_call_result_20809 = invoke(stypy.reporting.localization.Localization(__file__, 98, 24), write_20801, *[result_mod_20807], **kwargs_20808)
        
        # SSA branch for the else part of an if statement (line 97)
        module_type_store.open_ssa_branch('else')
        
        # Call to write(...): (line 102)
        # Processing the call arguments (line 102)
        str_20812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 35), 'str', '#!%s%s\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 103)
        tuple_20813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 103)
        # Adding element type (line 103)
        
        # Call to join(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Call to get_config_var(...): (line 104)
        # Processing the call arguments (line 104)
        str_20819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 54), 'str', 'BINDIR')
        # Processing the call keyword arguments (line 104)
        kwargs_20820 = {}
        # Getting the type of '_sysconfig' (line 104)
        _sysconfig_20817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 28), '_sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 104)
        get_config_var_20818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 28), _sysconfig_20817, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 104)
        get_config_var_call_result_20821 = invoke(stypy.reporting.localization.Localization(__file__, 104, 28), get_config_var_20818, *[str_20819], **kwargs_20820)
        
        str_20822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 27), 'str', 'python%s%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 105)
        tuple_20823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 43), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 105)
        # Adding element type (line 105)
        
        # Call to get_config_var(...): (line 105)
        # Processing the call arguments (line 105)
        str_20826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 69), 'str', 'VERSION')
        # Processing the call keyword arguments (line 105)
        kwargs_20827 = {}
        # Getting the type of '_sysconfig' (line 105)
        _sysconfig_20824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 43), '_sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 105)
        get_config_var_20825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 43), _sysconfig_20824, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 105)
        get_config_var_call_result_20828 = invoke(stypy.reporting.localization.Localization(__file__, 105, 43), get_config_var_20825, *[str_20826], **kwargs_20827)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 43), tuple_20823, get_config_var_call_result_20828)
        # Adding element type (line 105)
        
        # Call to get_config_var(...): (line 106)
        # Processing the call arguments (line 106)
        str_20831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 69), 'str', 'EXE')
        # Processing the call keyword arguments (line 106)
        kwargs_20832 = {}
        # Getting the type of '_sysconfig' (line 106)
        _sysconfig_20829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 43), '_sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 106)
        get_config_var_20830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 43), _sysconfig_20829, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 106)
        get_config_var_call_result_20833 = invoke(stypy.reporting.localization.Localization(__file__, 106, 43), get_config_var_20830, *[str_20831], **kwargs_20832)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 43), tuple_20823, get_config_var_call_result_20833)
        
        # Applying the binary operator '%' (line 105)
        result_mod_20834 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 27), '%', str_20822, tuple_20823)
        
        # Processing the call keyword arguments (line 103)
        kwargs_20835 = {}
        # Getting the type of 'os' (line 103)
        os_20814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 36), 'os', False)
        # Obtaining the member 'path' of a type (line 103)
        path_20815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 36), os_20814, 'path')
        # Obtaining the member 'join' of a type (line 103)
        join_20816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 36), path_20815, 'join')
        # Calling join(args, kwargs) (line 103)
        join_call_result_20836 = invoke(stypy.reporting.localization.Localization(__file__, 103, 36), join_20816, *[get_config_var_call_result_20821, result_mod_20834], **kwargs_20835)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 36), tuple_20813, join_call_result_20836)
        # Adding element type (line 103)
        # Getting the type of 'post_interp' (line 107)
        post_interp_20837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 36), 'post_interp', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 36), tuple_20813, post_interp_20837)
        
        # Applying the binary operator '%' (line 102)
        result_mod_20838 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 35), '%', str_20812, tuple_20813)
        
        # Processing the call keyword arguments (line 102)
        kwargs_20839 = {}
        # Getting the type of 'outf' (line 102)
        outf_20810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'outf', False)
        # Obtaining the member 'write' of a type (line 102)
        write_20811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 24), outf_20810, 'write')
        # Calling write(args, kwargs) (line 102)
        write_call_result_20840 = invoke(stypy.reporting.localization.Localization(__file__, 102, 24), write_20811, *[result_mod_20838], **kwargs_20839)
        
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to writelines(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Call to readlines(...): (line 108)
        # Processing the call keyword arguments (line 108)
        kwargs_20845 = {}
        # Getting the type of 'f' (line 108)
        f_20843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 36), 'f', False)
        # Obtaining the member 'readlines' of a type (line 108)
        readlines_20844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 36), f_20843, 'readlines')
        # Calling readlines(args, kwargs) (line 108)
        readlines_call_result_20846 = invoke(stypy.reporting.localization.Localization(__file__, 108, 36), readlines_20844, *[], **kwargs_20845)
        
        # Processing the call keyword arguments (line 108)
        kwargs_20847 = {}
        # Getting the type of 'outf' (line 108)
        outf_20841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'outf', False)
        # Obtaining the member 'writelines' of a type (line 108)
        writelines_20842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 20), outf_20841, 'writelines')
        # Calling writelines(args, kwargs) (line 108)
        writelines_call_result_20848 = invoke(stypy.reporting.localization.Localization(__file__, 108, 20), writelines_20842, *[readlines_call_result_20846], **kwargs_20847)
        
        
        # Call to close(...): (line 109)
        # Processing the call keyword arguments (line 109)
        kwargs_20851 = {}
        # Getting the type of 'outf' (line 109)
        outf_20849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 20), 'outf', False)
        # Obtaining the member 'close' of a type (line 109)
        close_20850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 20), outf_20849, 'close')
        # Calling close(args, kwargs) (line 109)
        close_call_result_20852 = invoke(stypy.reporting.localization.Localization(__file__, 109, 20), close_20850, *[], **kwargs_20851)
        
        # SSA join for if statement (line 95)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'f' (line 110)
        f_20853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'f')
        # Testing the type of an if condition (line 110)
        if_condition_20854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 16), f_20853)
        # Assigning a type to the variable 'if_condition_20854' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'if_condition_20854', if_condition_20854)
        # SSA begins for if statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to close(...): (line 111)
        # Processing the call keyword arguments (line 111)
        kwargs_20857 = {}
        # Getting the type of 'f' (line 111)
        f_20855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'f', False)
        # Obtaining the member 'close' of a type (line 111)
        close_20856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 20), f_20855, 'close')
        # Calling close(args, kwargs) (line 111)
        close_call_result_20858 = invoke(stypy.reporting.localization.Localization(__file__, 111, 20), close_20856, *[], **kwargs_20857)
        
        # SSA join for if statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 92)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'f' (line 113)
        f_20859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'f')
        # Testing the type of an if condition (line 113)
        if_condition_20860 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 16), f_20859)
        # Assigning a type to the variable 'if_condition_20860' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'if_condition_20860', if_condition_20860)
        # SSA begins for if statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to close(...): (line 114)
        # Processing the call keyword arguments (line 114)
        kwargs_20863 = {}
        # Getting the type of 'f' (line 114)
        f_20861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 20), 'f', False)
        # Obtaining the member 'close' of a type (line 114)
        close_20862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 20), f_20861, 'close')
        # Calling close(args, kwargs) (line 114)
        close_call_result_20864 = invoke(stypy.reporting.localization.Localization(__file__, 114, 20), close_20862, *[], **kwargs_20863)
        
        # SSA join for if statement (line 113)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to copy_file(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'script' (line 115)
        script_20867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 31), 'script', False)
        # Getting the type of 'outfile' (line 115)
        outfile_20868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 39), 'outfile', False)
        # Processing the call keyword arguments (line 115)
        kwargs_20869 = {}
        # Getting the type of 'self' (line 115)
        self_20865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'self', False)
        # Obtaining the member 'copy_file' of a type (line 115)
        copy_file_20866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 16), self_20865, 'copy_file')
        # Calling copy_file(args, kwargs) (line 115)
        copy_file_call_result_20870 = invoke(stypy.reporting.localization.Localization(__file__, 115, 16), copy_file_20866, *[script_20867, outfile_20868], **kwargs_20869)
        
        # SSA join for if statement (line 92)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'os' (line 117)
        os_20871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 11), 'os')
        # Obtaining the member 'name' of a type (line 117)
        name_20872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 11), os_20871, 'name')
        str_20873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 22), 'str', 'posix')
        # Applying the binary operator '==' (line 117)
        result_eq_20874 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 11), '==', name_20872, str_20873)
        
        # Testing the type of an if condition (line 117)
        if_condition_20875 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 8), result_eq_20874)
        # Assigning a type to the variable 'if_condition_20875' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'if_condition_20875', if_condition_20875)
        # SSA begins for if statement (line 117)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'outfiles' (line 118)
        outfiles_20876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), 'outfiles')
        # Testing the type of a for loop iterable (line 118)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 12), outfiles_20876)
        # Getting the type of the for loop variable (line 118)
        for_loop_var_20877 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 12), outfiles_20876)
        # Assigning a type to the variable 'file' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'file', for_loop_var_20877)
        # SSA begins for a for statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'self' (line 119)
        self_20878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 19), 'self')
        # Obtaining the member 'dry_run' of a type (line 119)
        dry_run_20879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 19), self_20878, 'dry_run')
        # Testing the type of an if condition (line 119)
        if_condition_20880 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 16), dry_run_20879)
        # Assigning a type to the variable 'if_condition_20880' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'if_condition_20880', if_condition_20880)
        # SSA begins for if statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 120)
        # Processing the call arguments (line 120)
        str_20883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 29), 'str', 'changing mode of %s')
        # Getting the type of 'file' (line 120)
        file_20884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 52), 'file', False)
        # Processing the call keyword arguments (line 120)
        kwargs_20885 = {}
        # Getting the type of 'log' (line 120)
        log_20881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'log', False)
        # Obtaining the member 'info' of a type (line 120)
        info_20882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 20), log_20881, 'info')
        # Calling info(args, kwargs) (line 120)
        info_call_result_20886 = invoke(stypy.reporting.localization.Localization(__file__, 120, 20), info_20882, *[str_20883, file_20884], **kwargs_20885)
        
        # SSA branch for the else part of an if statement (line 119)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 122):
        
        # Obtaining the type of the subscript
        # Getting the type of 'ST_MODE' (line 122)
        ST_MODE_20887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 44), 'ST_MODE')
        
        # Call to stat(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'file' (line 122)
        file_20890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 38), 'file', False)
        # Processing the call keyword arguments (line 122)
        kwargs_20891 = {}
        # Getting the type of 'os' (line 122)
        os_20888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 30), 'os', False)
        # Obtaining the member 'stat' of a type (line 122)
        stat_20889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 30), os_20888, 'stat')
        # Calling stat(args, kwargs) (line 122)
        stat_call_result_20892 = invoke(stypy.reporting.localization.Localization(__file__, 122, 30), stat_20889, *[file_20890], **kwargs_20891)
        
        # Obtaining the member '__getitem__' of a type (line 122)
        getitem___20893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 30), stat_call_result_20892, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 122)
        subscript_call_result_20894 = invoke(stypy.reporting.localization.Localization(__file__, 122, 30), getitem___20893, ST_MODE_20887)
        
        int_20895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 55), 'int')
        # Applying the binary operator '&' (line 122)
        result_and__20896 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 30), '&', subscript_call_result_20894, int_20895)
        
        # Assigning a type to the variable 'oldmode' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 20), 'oldmode', result_and__20896)
        
        # Assigning a BinOp to a Name (line 123):
        # Getting the type of 'oldmode' (line 123)
        oldmode_20897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 31), 'oldmode')
        int_20898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 41), 'int')
        # Applying the binary operator '|' (line 123)
        result_or__20899 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 31), '|', oldmode_20897, int_20898)
        
        int_20900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 49), 'int')
        # Applying the binary operator '&' (line 123)
        result_and__20901 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 30), '&', result_or__20899, int_20900)
        
        # Assigning a type to the variable 'newmode' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 20), 'newmode', result_and__20901)
        
        
        # Getting the type of 'newmode' (line 124)
        newmode_20902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 23), 'newmode')
        # Getting the type of 'oldmode' (line 124)
        oldmode_20903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 34), 'oldmode')
        # Applying the binary operator '!=' (line 124)
        result_ne_20904 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 23), '!=', newmode_20902, oldmode_20903)
        
        # Testing the type of an if condition (line 124)
        if_condition_20905 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 20), result_ne_20904)
        # Assigning a type to the variable 'if_condition_20905' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'if_condition_20905', if_condition_20905)
        # SSA begins for if statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 125)
        # Processing the call arguments (line 125)
        str_20908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 33), 'str', 'changing mode of %s from %o to %o')
        # Getting the type of 'file' (line 126)
        file_20909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 33), 'file', False)
        # Getting the type of 'oldmode' (line 126)
        oldmode_20910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 39), 'oldmode', False)
        # Getting the type of 'newmode' (line 126)
        newmode_20911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 48), 'newmode', False)
        # Processing the call keyword arguments (line 125)
        kwargs_20912 = {}
        # Getting the type of 'log' (line 125)
        log_20906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 24), 'log', False)
        # Obtaining the member 'info' of a type (line 125)
        info_20907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 24), log_20906, 'info')
        # Calling info(args, kwargs) (line 125)
        info_call_result_20913 = invoke(stypy.reporting.localization.Localization(__file__, 125, 24), info_20907, *[str_20908, file_20909, oldmode_20910, newmode_20911], **kwargs_20912)
        
        
        # Call to chmod(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'file' (line 127)
        file_20916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 33), 'file', False)
        # Getting the type of 'newmode' (line 127)
        newmode_20917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 39), 'newmode', False)
        # Processing the call keyword arguments (line 127)
        kwargs_20918 = {}
        # Getting the type of 'os' (line 127)
        os_20914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'os', False)
        # Obtaining the member 'chmod' of a type (line 127)
        chmod_20915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 24), os_20914, 'chmod')
        # Calling chmod(args, kwargs) (line 127)
        chmod_call_result_20919 = invoke(stypy.reporting.localization.Localization(__file__, 127, 24), chmod_20915, *[file_20916, newmode_20917], **kwargs_20918)
        
        # SSA join for if statement (line 124)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 119)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 117)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'copy_scripts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy_scripts' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_20920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20920)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy_scripts'
        return stypy_return_type_20920


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_scripts.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'build_scripts' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'build_scripts', build_scripts)

# Assigning a Str to a Name (line 19):
str_20921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 18), 'str', '"build" scripts (copy and fixup #! line)')
# Getting the type of 'build_scripts'
build_scripts_20922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_scripts')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_scripts_20922, 'description', str_20921)

# Assigning a List to a Name (line 21):

# Obtaining an instance of the builtin type 'list' (line 21)
list_20923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 22)
tuple_20924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 22)
# Adding element type (line 22)
str_20925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'str', 'build-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_20924, str_20925)
# Adding element type (line 22)
str_20926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_20924, str_20926)
# Adding element type (line 22)
str_20927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 28), 'str', 'directory to "build" (copy) to')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_20924, str_20927)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_20923, tuple_20924)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_20928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
str_20929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_20928, str_20929)
# Adding element type (line 23)
str_20930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 18), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_20928, str_20930)
# Adding element type (line 23)
str_20931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'str', 'forcibly build everything (ignore file timestamps')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_20928, str_20931)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_20923, tuple_20928)
# Adding element type (line 21)

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_20932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
str_20933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'str', 'executable=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_20932, str_20933)
# Adding element type (line 24)
str_20934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 24), 'str', 'e')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_20932, str_20934)
# Adding element type (line 24)
str_20935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 29), 'str', 'specify final destination interpreter path')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_20932, str_20935)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_20923, tuple_20932)

# Getting the type of 'build_scripts'
build_scripts_20936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_scripts')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_scripts_20936, 'user_options', list_20923)

# Assigning a List to a Name (line 27):

# Obtaining an instance of the builtin type 'list' (line 27)
list_20937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)
str_20938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), list_20937, str_20938)

# Getting the type of 'build_scripts'
build_scripts_20939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_scripts')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_scripts_20939, 'boolean_options', list_20937)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
