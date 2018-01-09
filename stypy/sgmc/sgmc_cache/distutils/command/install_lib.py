
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.install_lib
2: 
3: Implements the Distutils 'install_lib' command
4: (install all Python modules).'''
5: 
6: __revision__ = "$Id$"
7: 
8: import os
9: import sys
10: 
11: from distutils.core import Command
12: from distutils.errors import DistutilsOptionError
13: 
14: 
15: # Extension for Python source files.
16: if hasattr(os, 'extsep'):
17:     PYTHON_SOURCE_EXTENSION = os.extsep + "py"
18: else:
19:     PYTHON_SOURCE_EXTENSION = ".py"
20: 
21: class install_lib(Command):
22: 
23:     description = "install all Python modules (extensions and pure Python)"
24: 
25:     # The byte-compilation options are a tad confusing.  Here are the
26:     # possible scenarios:
27:     #   1) no compilation at all (--no-compile --no-optimize)
28:     #   2) compile .pyc only (--compile --no-optimize; default)
29:     #   3) compile .pyc and "level 1" .pyo (--compile --optimize)
30:     #   4) compile "level 1" .pyo only (--no-compile --optimize)
31:     #   5) compile .pyc and "level 2" .pyo (--compile --optimize-more)
32:     #   6) compile "level 2" .pyo only (--no-compile --optimize-more)
33:     #
34:     # The UI for this is two option, 'compile' and 'optimize'.
35:     # 'compile' is strictly boolean, and only decides whether to
36:     # generate .pyc files.  'optimize' is three-way (0, 1, or 2), and
37:     # decides both whether to generate .pyo files and what level of
38:     # optimization to use.
39: 
40:     user_options = [
41:         ('install-dir=', 'd', "directory to install to"),
42:         ('build-dir=','b', "build directory (where to install from)"),
43:         ('force', 'f', "force installation (overwrite existing files)"),
44:         ('compile', 'c', "compile .py to .pyc [default]"),
45:         ('no-compile', None, "don't compile .py files"),
46:         ('optimize=', 'O',
47:          "also compile with optimization: -O1 for \"python -O\", "
48:          "-O2 for \"python -OO\", and -O0 to disable [default: -O0]"),
49:         ('skip-build', None, "skip the build steps"),
50:         ]
51: 
52:     boolean_options = ['force', 'compile', 'skip-build']
53:     negative_opt = {'no-compile' : 'compile'}
54: 
55:     def initialize_options(self):
56:         # let the 'install' command dictate our installation directory
57:         self.install_dir = None
58:         self.build_dir = None
59:         self.force = 0
60:         self.compile = None
61:         self.optimize = None
62:         self.skip_build = None
63: 
64:     def finalize_options(self):
65:         # Get all the information we need to install pure Python modules
66:         # from the umbrella 'install' command -- build (source) directory,
67:         # install (target) directory, and whether to compile .py files.
68:         self.set_undefined_options('install',
69:                                    ('build_lib', 'build_dir'),
70:                                    ('install_lib', 'install_dir'),
71:                                    ('force', 'force'),
72:                                    ('compile', 'compile'),
73:                                    ('optimize', 'optimize'),
74:                                    ('skip_build', 'skip_build'),
75:                                   )
76: 
77:         if self.compile is None:
78:             self.compile = 1
79:         if self.optimize is None:
80:             self.optimize = 0
81: 
82:         if not isinstance(self.optimize, int):
83:             try:
84:                 self.optimize = int(self.optimize)
85:                 if self.optimize not in (0, 1, 2):
86:                     raise AssertionError
87:             except (ValueError, AssertionError):
88:                 raise DistutilsOptionError, "optimize must be 0, 1, or 2"
89: 
90:     def run(self):
91:         # Make sure we have built everything we need first
92:         self.build()
93: 
94:         # Install everything: simply dump the entire contents of the build
95:         # directory to the installation directory (that's the beauty of
96:         # having a build directory!)
97:         outfiles = self.install()
98: 
99:         # (Optionally) compile .py to .pyc
100:         if outfiles is not None and self.distribution.has_pure_modules():
101:             self.byte_compile(outfiles)
102: 
103:     # -- Top-level worker functions ------------------------------------
104:     # (called from 'run()')
105: 
106:     def build(self):
107:         if not self.skip_build:
108:             if self.distribution.has_pure_modules():
109:                 self.run_command('build_py')
110:             if self.distribution.has_ext_modules():
111:                 self.run_command('build_ext')
112: 
113:     def install(self):
114:         if os.path.isdir(self.build_dir):
115:             outfiles = self.copy_tree(self.build_dir, self.install_dir)
116:         else:
117:             self.warn("'%s' does not exist -- no Python modules to install" %
118:                       self.build_dir)
119:             return
120:         return outfiles
121: 
122:     def byte_compile(self, files):
123:         if sys.dont_write_bytecode:
124:             self.warn('byte-compiling is disabled, skipping.')
125:             return
126: 
127:         from distutils.util import byte_compile
128: 
129:         # Get the "--root" directory supplied to the "install" command,
130:         # and use it as a prefix to strip off the purported filename
131:         # encoded in bytecode files.  This is far from complete, but it
132:         # should at least generate usable bytecode in RPM distributions.
133:         install_root = self.get_finalized_command('install').root
134: 
135:         if self.compile:
136:             byte_compile(files, optimize=0,
137:                          force=self.force, prefix=install_root,
138:                          dry_run=self.dry_run)
139:         if self.optimize > 0:
140:             byte_compile(files, optimize=self.optimize,
141:                          force=self.force, prefix=install_root,
142:                          verbose=self.verbose, dry_run=self.dry_run)
143: 
144: 
145:     # -- Utility methods -----------------------------------------------
146: 
147:     def _mutate_outputs(self, has_any, build_cmd, cmd_option, output_dir):
148:         if not has_any:
149:             return []
150: 
151:         build_cmd = self.get_finalized_command(build_cmd)
152:         build_files = build_cmd.get_outputs()
153:         build_dir = getattr(build_cmd, cmd_option)
154: 
155:         prefix_len = len(build_dir) + len(os.sep)
156:         outputs = []
157:         for file in build_files:
158:             outputs.append(os.path.join(output_dir, file[prefix_len:]))
159: 
160:         return outputs
161: 
162:     def _bytecode_filenames(self, py_filenames):
163:         bytecode_files = []
164:         for py_file in py_filenames:
165:             # Since build_py handles package data installation, the
166:             # list of outputs can contain more than just .py files.
167:             # Make sure we only report bytecode for the .py files.
168:             ext = os.path.splitext(os.path.normcase(py_file))[1]
169:             if ext != PYTHON_SOURCE_EXTENSION:
170:                 continue
171:             if self.compile:
172:                 bytecode_files.append(py_file + "c")
173:             if self.optimize > 0:
174:                 bytecode_files.append(py_file + "o")
175: 
176:         return bytecode_files
177: 
178: 
179:     # -- External interface --------------------------------------------
180:     # (called by outsiders)
181: 
182:     def get_outputs(self):
183:         '''Return the list of files that would be installed if this command
184:         were actually run.  Not affected by the "dry-run" flag or whether
185:         modules have actually been built yet.
186:         '''
187:         pure_outputs = \
188:             self._mutate_outputs(self.distribution.has_pure_modules(),
189:                                  'build_py', 'build_lib',
190:                                  self.install_dir)
191:         if self.compile:
192:             bytecode_outputs = self._bytecode_filenames(pure_outputs)
193:         else:
194:             bytecode_outputs = []
195: 
196:         ext_outputs = \
197:             self._mutate_outputs(self.distribution.has_ext_modules(),
198:                                  'build_ext', 'build_lib',
199:                                  self.install_dir)
200: 
201:         return pure_outputs + bytecode_outputs + ext_outputs
202: 
203:     def get_inputs(self):
204:         '''Get the list of files that are input to this command, ie. the
205:         files that get installed as they are named in the build tree.
206:         The files in this list correspond one-to-one to the output
207:         filenames returned by 'get_outputs()'.
208:         '''
209:         inputs = []
210: 
211:         if self.distribution.has_pure_modules():
212:             build_py = self.get_finalized_command('build_py')
213:             inputs.extend(build_py.get_outputs())
214: 
215:         if self.distribution.has_ext_modules():
216:             build_ext = self.get_finalized_command('build_ext')
217:             inputs.extend(build_ext.get_outputs())
218: 
219:         return inputs
220: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_24144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', "distutils.command.install_lib\n\nImplements the Distutils 'install_lib' command\n(install all Python modules).")

# Assigning a Str to a Name (line 6):
str_24145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__revision__', str_24145)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import os' statement (line 8)
import os

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import sys' statement (line 9)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.core import Command' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_24146 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core')

if (type(import_24146) is not StypyTypeError):

    if (import_24146 != 'pyd_module'):
        __import__(import_24146)
        sys_modules_24147 = sys.modules[import_24146]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core', sys_modules_24147.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_24147, sys_modules_24147.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core', import_24146)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.errors import DistutilsOptionError' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_24148 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors')

if (type(import_24148) is not StypyTypeError):

    if (import_24148 != 'pyd_module'):
        __import__(import_24148)
        sys_modules_24149 = sys.modules[import_24148]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', sys_modules_24149.module_type_store, module_type_store, ['DistutilsOptionError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_24149, sys_modules_24149.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsOptionError

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', None, module_type_store, ['DistutilsOptionError'], [DistutilsOptionError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', import_24148)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')


# Type idiom detected: calculating its left and rigth part (line 16)
str_24150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 15), 'str', 'extsep')
# Getting the type of 'os' (line 16)
os_24151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'os')

(may_be_24152, more_types_in_union_24153) = may_provide_member(str_24150, os_24151)

if may_be_24152:

    if more_types_in_union_24153:
        # Runtime conditional SSA (line 16)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    # Assigning a type to the variable 'os' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'os', remove_not_member_provider_from_union(os_24151, 'extsep'))
    
    # Assigning a BinOp to a Name (line 17):
    # Getting the type of 'os' (line 17)
    os_24154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 30), 'os')
    # Obtaining the member 'extsep' of a type (line 17)
    extsep_24155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 30), os_24154, 'extsep')
    str_24156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 42), 'str', 'py')
    # Applying the binary operator '+' (line 17)
    result_add_24157 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 30), '+', extsep_24155, str_24156)
    
    # Assigning a type to the variable 'PYTHON_SOURCE_EXTENSION' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'PYTHON_SOURCE_EXTENSION', result_add_24157)

    if more_types_in_union_24153:
        # Runtime conditional SSA for else branch (line 16)
        module_type_store.open_ssa_branch('idiom else')



if ((not may_be_24152) or more_types_in_union_24153):
    # Assigning a type to the variable 'os' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'os', remove_member_provider_from_union(os_24151, 'extsep'))
    
    # Assigning a Str to a Name (line 19):
    str_24158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 30), 'str', '.py')
    # Assigning a type to the variable 'PYTHON_SOURCE_EXTENSION' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'PYTHON_SOURCE_EXTENSION', str_24158)

    if (may_be_24152 and more_types_in_union_24153):
        # SSA join for if statement (line 16)
        module_type_store = module_type_store.join_ssa_context()



# Declaration of the 'install_lib' class
# Getting the type of 'Command' (line 21)
Command_24159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), 'Command')

class install_lib(Command_24159, ):

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
        install_lib.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        install_lib.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_lib.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_lib.initialize_options.__dict__.__setitem__('stypy_function_name', 'install_lib.initialize_options')
        install_lib.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install_lib.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_lib.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_lib.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_lib.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_lib.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_lib.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_lib.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 57):
        # Getting the type of 'None' (line 57)
        None_24160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 27), 'None')
        # Getting the type of 'self' (line 57)
        self_24161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member 'install_dir' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_24161, 'install_dir', None_24160)
        
        # Assigning a Name to a Attribute (line 58):
        # Getting the type of 'None' (line 58)
        None_24162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 25), 'None')
        # Getting the type of 'self' (line 58)
        self_24163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self')
        # Setting the type of the member 'build_dir' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_24163, 'build_dir', None_24162)
        
        # Assigning a Num to a Attribute (line 59):
        int_24164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 21), 'int')
        # Getting the type of 'self' (line 59)
        self_24165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member 'force' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_24165, 'force', int_24164)
        
        # Assigning a Name to a Attribute (line 60):
        # Getting the type of 'None' (line 60)
        None_24166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'None')
        # Getting the type of 'self' (line 60)
        self_24167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self')
        # Setting the type of the member 'compile' of a type (line 60)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_24167, 'compile', None_24166)
        
        # Assigning a Name to a Attribute (line 61):
        # Getting the type of 'None' (line 61)
        None_24168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'None')
        # Getting the type of 'self' (line 61)
        self_24169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
        # Setting the type of the member 'optimize' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_24169, 'optimize', None_24168)
        
        # Assigning a Name to a Attribute (line 62):
        # Getting the type of 'None' (line 62)
        None_24170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 26), 'None')
        # Getting the type of 'self' (line 62)
        self_24171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member 'skip_build' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_24171, 'skip_build', None_24170)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_24172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24172)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_24172


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_lib.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        install_lib.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_lib.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_lib.finalize_options.__dict__.__setitem__('stypy_function_name', 'install_lib.finalize_options')
        install_lib.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install_lib.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_lib.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_lib.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_lib.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_lib.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_lib.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_lib.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_undefined_options(...): (line 68)
        # Processing the call arguments (line 68)
        str_24175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 35), 'str', 'install')
        
        # Obtaining an instance of the builtin type 'tuple' (line 69)
        tuple_24176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 69)
        # Adding element type (line 69)
        str_24177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 36), 'str', 'build_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 36), tuple_24176, str_24177)
        # Adding element type (line 69)
        str_24178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 49), 'str', 'build_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 36), tuple_24176, str_24178)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 70)
        tuple_24179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 70)
        # Adding element type (line 70)
        str_24180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 36), 'str', 'install_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 36), tuple_24179, str_24180)
        # Adding element type (line 70)
        str_24181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 51), 'str', 'install_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 36), tuple_24179, str_24181)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 71)
        tuple_24182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 71)
        # Adding element type (line 71)
        str_24183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 36), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 36), tuple_24182, str_24183)
        # Adding element type (line 71)
        str_24184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 45), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 36), tuple_24182, str_24184)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 72)
        tuple_24185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 72)
        # Adding element type (line 72)
        str_24186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 36), 'str', 'compile')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 36), tuple_24185, str_24186)
        # Adding element type (line 72)
        str_24187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 47), 'str', 'compile')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 36), tuple_24185, str_24187)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 73)
        tuple_24188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 73)
        # Adding element type (line 73)
        str_24189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 36), 'str', 'optimize')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 36), tuple_24188, str_24189)
        # Adding element type (line 73)
        str_24190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 48), 'str', 'optimize')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 36), tuple_24188, str_24190)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 74)
        tuple_24191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 74)
        # Adding element type (line 74)
        str_24192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 36), 'str', 'skip_build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 36), tuple_24191, str_24192)
        # Adding element type (line 74)
        str_24193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 50), 'str', 'skip_build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 36), tuple_24191, str_24193)
        
        # Processing the call keyword arguments (line 68)
        kwargs_24194 = {}
        # Getting the type of 'self' (line 68)
        self_24173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 68)
        set_undefined_options_24174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_24173, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 68)
        set_undefined_options_call_result_24195 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), set_undefined_options_24174, *[str_24175, tuple_24176, tuple_24179, tuple_24182, tuple_24185, tuple_24188, tuple_24191], **kwargs_24194)
        
        
        # Type idiom detected: calculating its left and rigth part (line 77)
        # Getting the type of 'self' (line 77)
        self_24196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'self')
        # Obtaining the member 'compile' of a type (line 77)
        compile_24197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 11), self_24196, 'compile')
        # Getting the type of 'None' (line 77)
        None_24198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 27), 'None')
        
        (may_be_24199, more_types_in_union_24200) = may_be_none(compile_24197, None_24198)

        if may_be_24199:

            if more_types_in_union_24200:
                # Runtime conditional SSA (line 77)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Attribute (line 78):
            int_24201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 27), 'int')
            # Getting the type of 'self' (line 78)
            self_24202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'self')
            # Setting the type of the member 'compile' of a type (line 78)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), self_24202, 'compile', int_24201)

            if more_types_in_union_24200:
                # SSA join for if statement (line 77)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 79)
        # Getting the type of 'self' (line 79)
        self_24203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'self')
        # Obtaining the member 'optimize' of a type (line 79)
        optimize_24204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 11), self_24203, 'optimize')
        # Getting the type of 'None' (line 79)
        None_24205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 28), 'None')
        
        (may_be_24206, more_types_in_union_24207) = may_be_none(optimize_24204, None_24205)

        if may_be_24206:

            if more_types_in_union_24207:
                # Runtime conditional SSA (line 79)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Attribute (line 80):
            int_24208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 28), 'int')
            # Getting the type of 'self' (line 80)
            self_24209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'self')
            # Setting the type of the member 'optimize' of a type (line 80)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), self_24209, 'optimize', int_24208)

            if more_types_in_union_24207:
                # SSA join for if statement (line 79)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 82)
        # Getting the type of 'int' (line 82)
        int_24210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 41), 'int')
        # Getting the type of 'self' (line 82)
        self_24211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'self')
        # Obtaining the member 'optimize' of a type (line 82)
        optimize_24212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 26), self_24211, 'optimize')
        
        (may_be_24213, more_types_in_union_24214) = may_not_be_subtype(int_24210, optimize_24212)

        if may_be_24213:

            if more_types_in_union_24214:
                # Runtime conditional SSA (line 82)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 82)
            self_24215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self')
            # Obtaining the member 'optimize' of a type (line 82)
            optimize_24216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_24215, 'optimize')
            # Setting the type of the member 'optimize' of a type (line 82)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_24215, 'optimize', remove_subtype_from_union(optimize_24212, int))
            
            
            # SSA begins for try-except statement (line 83)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Attribute (line 84):
            
            # Call to int(...): (line 84)
            # Processing the call arguments (line 84)
            # Getting the type of 'self' (line 84)
            self_24218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 36), 'self', False)
            # Obtaining the member 'optimize' of a type (line 84)
            optimize_24219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 36), self_24218, 'optimize')
            # Processing the call keyword arguments (line 84)
            kwargs_24220 = {}
            # Getting the type of 'int' (line 84)
            int_24217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'int', False)
            # Calling int(args, kwargs) (line 84)
            int_call_result_24221 = invoke(stypy.reporting.localization.Localization(__file__, 84, 32), int_24217, *[optimize_24219], **kwargs_24220)
            
            # Getting the type of 'self' (line 84)
            self_24222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'self')
            # Setting the type of the member 'optimize' of a type (line 84)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 16), self_24222, 'optimize', int_call_result_24221)
            
            
            # Getting the type of 'self' (line 85)
            self_24223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 19), 'self')
            # Obtaining the member 'optimize' of a type (line 85)
            optimize_24224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 19), self_24223, 'optimize')
            
            # Obtaining an instance of the builtin type 'tuple' (line 85)
            tuple_24225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 41), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 85)
            # Adding element type (line 85)
            int_24226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 41), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 41), tuple_24225, int_24226)
            # Adding element type (line 85)
            int_24227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 44), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 41), tuple_24225, int_24227)
            # Adding element type (line 85)
            int_24228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 47), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 41), tuple_24225, int_24228)
            
            # Applying the binary operator 'notin' (line 85)
            result_contains_24229 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 19), 'notin', optimize_24224, tuple_24225)
            
            # Testing the type of an if condition (line 85)
            if_condition_24230 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 16), result_contains_24229)
            # Assigning a type to the variable 'if_condition_24230' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'if_condition_24230', if_condition_24230)
            # SSA begins for if statement (line 85)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'AssertionError' (line 86)
            AssertionError_24231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 26), 'AssertionError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 86, 20), AssertionError_24231, 'raise parameter', BaseException)
            # SSA join for if statement (line 85)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the except part of a try statement (line 83)
            # SSA branch for the except 'Tuple' branch of a try statement (line 83)
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'DistutilsOptionError' (line 88)
            DistutilsOptionError_24232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 22), 'DistutilsOptionError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 88, 16), DistutilsOptionError_24232, 'raise parameter', BaseException)
            # SSA join for try-except statement (line 83)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_24214:
                # SSA join for if statement (line 82)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_24233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24233)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_24233


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 90, 4, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_lib.run.__dict__.__setitem__('stypy_localization', localization)
        install_lib.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_lib.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_lib.run.__dict__.__setitem__('stypy_function_name', 'install_lib.run')
        install_lib.run.__dict__.__setitem__('stypy_param_names_list', [])
        install_lib.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_lib.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_lib.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_lib.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_lib.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_lib.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_lib.run', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to build(...): (line 92)
        # Processing the call keyword arguments (line 92)
        kwargs_24236 = {}
        # Getting the type of 'self' (line 92)
        self_24234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'self', False)
        # Obtaining the member 'build' of a type (line 92)
        build_24235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 8), self_24234, 'build')
        # Calling build(args, kwargs) (line 92)
        build_call_result_24237 = invoke(stypy.reporting.localization.Localization(__file__, 92, 8), build_24235, *[], **kwargs_24236)
        
        
        # Assigning a Call to a Name (line 97):
        
        # Call to install(...): (line 97)
        # Processing the call keyword arguments (line 97)
        kwargs_24240 = {}
        # Getting the type of 'self' (line 97)
        self_24238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 19), 'self', False)
        # Obtaining the member 'install' of a type (line 97)
        install_24239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 19), self_24238, 'install')
        # Calling install(args, kwargs) (line 97)
        install_call_result_24241 = invoke(stypy.reporting.localization.Localization(__file__, 97, 19), install_24239, *[], **kwargs_24240)
        
        # Assigning a type to the variable 'outfiles' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'outfiles', install_call_result_24241)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'outfiles' (line 100)
        outfiles_24242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 11), 'outfiles')
        # Getting the type of 'None' (line 100)
        None_24243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'None')
        # Applying the binary operator 'isnot' (line 100)
        result_is_not_24244 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 11), 'isnot', outfiles_24242, None_24243)
        
        
        # Call to has_pure_modules(...): (line 100)
        # Processing the call keyword arguments (line 100)
        kwargs_24248 = {}
        # Getting the type of 'self' (line 100)
        self_24245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 36), 'self', False)
        # Obtaining the member 'distribution' of a type (line 100)
        distribution_24246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 36), self_24245, 'distribution')
        # Obtaining the member 'has_pure_modules' of a type (line 100)
        has_pure_modules_24247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 36), distribution_24246, 'has_pure_modules')
        # Calling has_pure_modules(args, kwargs) (line 100)
        has_pure_modules_call_result_24249 = invoke(stypy.reporting.localization.Localization(__file__, 100, 36), has_pure_modules_24247, *[], **kwargs_24248)
        
        # Applying the binary operator 'and' (line 100)
        result_and_keyword_24250 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 11), 'and', result_is_not_24244, has_pure_modules_call_result_24249)
        
        # Testing the type of an if condition (line 100)
        if_condition_24251 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 8), result_and_keyword_24250)
        # Assigning a type to the variable 'if_condition_24251' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'if_condition_24251', if_condition_24251)
        # SSA begins for if statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to byte_compile(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'outfiles' (line 101)
        outfiles_24254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'outfiles', False)
        # Processing the call keyword arguments (line 101)
        kwargs_24255 = {}
        # Getting the type of 'self' (line 101)
        self_24252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'self', False)
        # Obtaining the member 'byte_compile' of a type (line 101)
        byte_compile_24253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), self_24252, 'byte_compile')
        # Calling byte_compile(args, kwargs) (line 101)
        byte_compile_call_result_24256 = invoke(stypy.reporting.localization.Localization(__file__, 101, 12), byte_compile_24253, *[outfiles_24254], **kwargs_24255)
        
        # SSA join for if statement (line 100)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 90)
        stypy_return_type_24257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24257)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_24257


    @norecursion
    def build(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build'
        module_type_store = module_type_store.open_function_context('build', 106, 4, False)
        # Assigning a type to the variable 'self' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_lib.build.__dict__.__setitem__('stypy_localization', localization)
        install_lib.build.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_lib.build.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_lib.build.__dict__.__setitem__('stypy_function_name', 'install_lib.build')
        install_lib.build.__dict__.__setitem__('stypy_param_names_list', [])
        install_lib.build.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_lib.build.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_lib.build.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_lib.build.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_lib.build.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_lib.build.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_lib.build', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build(...)' code ##################

        
        
        # Getting the type of 'self' (line 107)
        self_24258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'self')
        # Obtaining the member 'skip_build' of a type (line 107)
        skip_build_24259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 15), self_24258, 'skip_build')
        # Applying the 'not' unary operator (line 107)
        result_not__24260 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 11), 'not', skip_build_24259)
        
        # Testing the type of an if condition (line 107)
        if_condition_24261 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 8), result_not__24260)
        # Assigning a type to the variable 'if_condition_24261' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'if_condition_24261', if_condition_24261)
        # SSA begins for if statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to has_pure_modules(...): (line 108)
        # Processing the call keyword arguments (line 108)
        kwargs_24265 = {}
        # Getting the type of 'self' (line 108)
        self_24262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'self', False)
        # Obtaining the member 'distribution' of a type (line 108)
        distribution_24263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 15), self_24262, 'distribution')
        # Obtaining the member 'has_pure_modules' of a type (line 108)
        has_pure_modules_24264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 15), distribution_24263, 'has_pure_modules')
        # Calling has_pure_modules(args, kwargs) (line 108)
        has_pure_modules_call_result_24266 = invoke(stypy.reporting.localization.Localization(__file__, 108, 15), has_pure_modules_24264, *[], **kwargs_24265)
        
        # Testing the type of an if condition (line 108)
        if_condition_24267 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 12), has_pure_modules_call_result_24266)
        # Assigning a type to the variable 'if_condition_24267' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'if_condition_24267', if_condition_24267)
        # SSA begins for if statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to run_command(...): (line 109)
        # Processing the call arguments (line 109)
        str_24270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 33), 'str', 'build_py')
        # Processing the call keyword arguments (line 109)
        kwargs_24271 = {}
        # Getting the type of 'self' (line 109)
        self_24268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'self', False)
        # Obtaining the member 'run_command' of a type (line 109)
        run_command_24269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 16), self_24268, 'run_command')
        # Calling run_command(args, kwargs) (line 109)
        run_command_call_result_24272 = invoke(stypy.reporting.localization.Localization(__file__, 109, 16), run_command_24269, *[str_24270], **kwargs_24271)
        
        # SSA join for if statement (line 108)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to has_ext_modules(...): (line 110)
        # Processing the call keyword arguments (line 110)
        kwargs_24276 = {}
        # Getting the type of 'self' (line 110)
        self_24273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'self', False)
        # Obtaining the member 'distribution' of a type (line 110)
        distribution_24274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 15), self_24273, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 110)
        has_ext_modules_24275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 15), distribution_24274, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 110)
        has_ext_modules_call_result_24277 = invoke(stypy.reporting.localization.Localization(__file__, 110, 15), has_ext_modules_24275, *[], **kwargs_24276)
        
        # Testing the type of an if condition (line 110)
        if_condition_24278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 12), has_ext_modules_call_result_24277)
        # Assigning a type to the variable 'if_condition_24278' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'if_condition_24278', if_condition_24278)
        # SSA begins for if statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to run_command(...): (line 111)
        # Processing the call arguments (line 111)
        str_24281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 33), 'str', 'build_ext')
        # Processing the call keyword arguments (line 111)
        kwargs_24282 = {}
        # Getting the type of 'self' (line 111)
        self_24279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'self', False)
        # Obtaining the member 'run_command' of a type (line 111)
        run_command_24280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), self_24279, 'run_command')
        # Calling run_command(args, kwargs) (line 111)
        run_command_call_result_24283 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), run_command_24280, *[str_24281], **kwargs_24282)
        
        # SSA join for if statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 107)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'build(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build' in the type store
        # Getting the type of 'stypy_return_type' (line 106)
        stypy_return_type_24284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24284)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build'
        return stypy_return_type_24284


    @norecursion
    def install(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'install'
        module_type_store = module_type_store.open_function_context('install', 113, 4, False)
        # Assigning a type to the variable 'self' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_lib.install.__dict__.__setitem__('stypy_localization', localization)
        install_lib.install.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_lib.install.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_lib.install.__dict__.__setitem__('stypy_function_name', 'install_lib.install')
        install_lib.install.__dict__.__setitem__('stypy_param_names_list', [])
        install_lib.install.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_lib.install.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_lib.install.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_lib.install.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_lib.install.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_lib.install.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_lib.install', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'install', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'install(...)' code ##################

        
        
        # Call to isdir(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'self' (line 114)
        self_24288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 25), 'self', False)
        # Obtaining the member 'build_dir' of a type (line 114)
        build_dir_24289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 25), self_24288, 'build_dir')
        # Processing the call keyword arguments (line 114)
        kwargs_24290 = {}
        # Getting the type of 'os' (line 114)
        os_24285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 114)
        path_24286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 11), os_24285, 'path')
        # Obtaining the member 'isdir' of a type (line 114)
        isdir_24287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 11), path_24286, 'isdir')
        # Calling isdir(args, kwargs) (line 114)
        isdir_call_result_24291 = invoke(stypy.reporting.localization.Localization(__file__, 114, 11), isdir_24287, *[build_dir_24289], **kwargs_24290)
        
        # Testing the type of an if condition (line 114)
        if_condition_24292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 8), isdir_call_result_24291)
        # Assigning a type to the variable 'if_condition_24292' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'if_condition_24292', if_condition_24292)
        # SSA begins for if statement (line 114)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 115):
        
        # Call to copy_tree(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'self' (line 115)
        self_24295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 38), 'self', False)
        # Obtaining the member 'build_dir' of a type (line 115)
        build_dir_24296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 38), self_24295, 'build_dir')
        # Getting the type of 'self' (line 115)
        self_24297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 54), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 115)
        install_dir_24298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 54), self_24297, 'install_dir')
        # Processing the call keyword arguments (line 115)
        kwargs_24299 = {}
        # Getting the type of 'self' (line 115)
        self_24293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 23), 'self', False)
        # Obtaining the member 'copy_tree' of a type (line 115)
        copy_tree_24294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 23), self_24293, 'copy_tree')
        # Calling copy_tree(args, kwargs) (line 115)
        copy_tree_call_result_24300 = invoke(stypy.reporting.localization.Localization(__file__, 115, 23), copy_tree_24294, *[build_dir_24296, install_dir_24298], **kwargs_24299)
        
        # Assigning a type to the variable 'outfiles' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'outfiles', copy_tree_call_result_24300)
        # SSA branch for the else part of an if statement (line 114)
        module_type_store.open_ssa_branch('else')
        
        # Call to warn(...): (line 117)
        # Processing the call arguments (line 117)
        str_24303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 22), 'str', "'%s' does not exist -- no Python modules to install")
        # Getting the type of 'self' (line 118)
        self_24304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 22), 'self', False)
        # Obtaining the member 'build_dir' of a type (line 118)
        build_dir_24305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 22), self_24304, 'build_dir')
        # Applying the binary operator '%' (line 117)
        result_mod_24306 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 22), '%', str_24303, build_dir_24305)
        
        # Processing the call keyword arguments (line 117)
        kwargs_24307 = {}
        # Getting the type of 'self' (line 117)
        self_24301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'self', False)
        # Obtaining the member 'warn' of a type (line 117)
        warn_24302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 12), self_24301, 'warn')
        # Calling warn(args, kwargs) (line 117)
        warn_call_result_24308 = invoke(stypy.reporting.localization.Localization(__file__, 117, 12), warn_24302, *[result_mod_24306], **kwargs_24307)
        
        # Assigning a type to the variable 'stypy_return_type' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 114)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'outfiles' (line 120)
        outfiles_24309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'outfiles')
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', outfiles_24309)
        
        # ################# End of 'install(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'install' in the type store
        # Getting the type of 'stypy_return_type' (line 113)
        stypy_return_type_24310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24310)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'install'
        return stypy_return_type_24310


    @norecursion
    def byte_compile(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'byte_compile'
        module_type_store = module_type_store.open_function_context('byte_compile', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_lib.byte_compile.__dict__.__setitem__('stypy_localization', localization)
        install_lib.byte_compile.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_lib.byte_compile.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_lib.byte_compile.__dict__.__setitem__('stypy_function_name', 'install_lib.byte_compile')
        install_lib.byte_compile.__dict__.__setitem__('stypy_param_names_list', ['files'])
        install_lib.byte_compile.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_lib.byte_compile.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_lib.byte_compile.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_lib.byte_compile.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_lib.byte_compile.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_lib.byte_compile.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_lib.byte_compile', ['files'], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'sys' (line 123)
        sys_24311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'sys')
        # Obtaining the member 'dont_write_bytecode' of a type (line 123)
        dont_write_bytecode_24312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 11), sys_24311, 'dont_write_bytecode')
        # Testing the type of an if condition (line 123)
        if_condition_24313 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 8), dont_write_bytecode_24312)
        # Assigning a type to the variable 'if_condition_24313' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'if_condition_24313', if_condition_24313)
        # SSA begins for if statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 124)
        # Processing the call arguments (line 124)
        str_24316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 22), 'str', 'byte-compiling is disabled, skipping.')
        # Processing the call keyword arguments (line 124)
        kwargs_24317 = {}
        # Getting the type of 'self' (line 124)
        self_24314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'self', False)
        # Obtaining the member 'warn' of a type (line 124)
        warn_24315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), self_24314, 'warn')
        # Calling warn(args, kwargs) (line 124)
        warn_call_result_24318 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), warn_24315, *[str_24316], **kwargs_24317)
        
        # Assigning a type to the variable 'stypy_return_type' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 127, 8))
        
        # 'from distutils.util import byte_compile' statement (line 127)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
        import_24319 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 127, 8), 'distutils.util')

        if (type(import_24319) is not StypyTypeError):

            if (import_24319 != 'pyd_module'):
                __import__(import_24319)
                sys_modules_24320 = sys.modules[import_24319]
                import_from_module(stypy.reporting.localization.Localization(__file__, 127, 8), 'distutils.util', sys_modules_24320.module_type_store, module_type_store, ['byte_compile'])
                nest_module(stypy.reporting.localization.Localization(__file__, 127, 8), __file__, sys_modules_24320, sys_modules_24320.module_type_store, module_type_store)
            else:
                from distutils.util import byte_compile

                import_from_module(stypy.reporting.localization.Localization(__file__, 127, 8), 'distutils.util', None, module_type_store, ['byte_compile'], [byte_compile])

        else:
            # Assigning a type to the variable 'distutils.util' (line 127)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'distutils.util', import_24319)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
        
        
        # Assigning a Attribute to a Name (line 133):
        
        # Call to get_finalized_command(...): (line 133)
        # Processing the call arguments (line 133)
        str_24323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 50), 'str', 'install')
        # Processing the call keyword arguments (line 133)
        kwargs_24324 = {}
        # Getting the type of 'self' (line 133)
        self_24321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 133)
        get_finalized_command_24322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), self_24321, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 133)
        get_finalized_command_call_result_24325 = invoke(stypy.reporting.localization.Localization(__file__, 133, 23), get_finalized_command_24322, *[str_24323], **kwargs_24324)
        
        # Obtaining the member 'root' of a type (line 133)
        root_24326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 23), get_finalized_command_call_result_24325, 'root')
        # Assigning a type to the variable 'install_root' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'install_root', root_24326)
        
        # Getting the type of 'self' (line 135)
        self_24327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 11), 'self')
        # Obtaining the member 'compile' of a type (line 135)
        compile_24328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 11), self_24327, 'compile')
        # Testing the type of an if condition (line 135)
        if_condition_24329 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 8), compile_24328)
        # Assigning a type to the variable 'if_condition_24329' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'if_condition_24329', if_condition_24329)
        # SSA begins for if statement (line 135)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to byte_compile(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'files' (line 136)
        files_24331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 25), 'files', False)
        # Processing the call keyword arguments (line 136)
        int_24332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 41), 'int')
        keyword_24333 = int_24332
        # Getting the type of 'self' (line 137)
        self_24334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 31), 'self', False)
        # Obtaining the member 'force' of a type (line 137)
        force_24335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 31), self_24334, 'force')
        keyword_24336 = force_24335
        # Getting the type of 'install_root' (line 137)
        install_root_24337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 50), 'install_root', False)
        keyword_24338 = install_root_24337
        # Getting the type of 'self' (line 138)
        self_24339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 33), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 138)
        dry_run_24340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 33), self_24339, 'dry_run')
        keyword_24341 = dry_run_24340
        kwargs_24342 = {'prefix': keyword_24338, 'force': keyword_24336, 'optimize': keyword_24333, 'dry_run': keyword_24341}
        # Getting the type of 'byte_compile' (line 136)
        byte_compile_24330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'byte_compile', False)
        # Calling byte_compile(args, kwargs) (line 136)
        byte_compile_call_result_24343 = invoke(stypy.reporting.localization.Localization(__file__, 136, 12), byte_compile_24330, *[files_24331], **kwargs_24342)
        
        # SSA join for if statement (line 135)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 139)
        self_24344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 'self')
        # Obtaining the member 'optimize' of a type (line 139)
        optimize_24345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 11), self_24344, 'optimize')
        int_24346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 27), 'int')
        # Applying the binary operator '>' (line 139)
        result_gt_24347 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 11), '>', optimize_24345, int_24346)
        
        # Testing the type of an if condition (line 139)
        if_condition_24348 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 8), result_gt_24347)
        # Assigning a type to the variable 'if_condition_24348' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'if_condition_24348', if_condition_24348)
        # SSA begins for if statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to byte_compile(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'files' (line 140)
        files_24350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 25), 'files', False)
        # Processing the call keyword arguments (line 140)
        # Getting the type of 'self' (line 140)
        self_24351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 41), 'self', False)
        # Obtaining the member 'optimize' of a type (line 140)
        optimize_24352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 41), self_24351, 'optimize')
        keyword_24353 = optimize_24352
        # Getting the type of 'self' (line 141)
        self_24354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 31), 'self', False)
        # Obtaining the member 'force' of a type (line 141)
        force_24355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 31), self_24354, 'force')
        keyword_24356 = force_24355
        # Getting the type of 'install_root' (line 141)
        install_root_24357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 50), 'install_root', False)
        keyword_24358 = install_root_24357
        # Getting the type of 'self' (line 142)
        self_24359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 33), 'self', False)
        # Obtaining the member 'verbose' of a type (line 142)
        verbose_24360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 33), self_24359, 'verbose')
        keyword_24361 = verbose_24360
        # Getting the type of 'self' (line 142)
        self_24362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 55), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 142)
        dry_run_24363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 55), self_24362, 'dry_run')
        keyword_24364 = dry_run_24363
        kwargs_24365 = {'prefix': keyword_24358, 'force': keyword_24356, 'optimize': keyword_24353, 'dry_run': keyword_24364, 'verbose': keyword_24361}
        # Getting the type of 'byte_compile' (line 140)
        byte_compile_24349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'byte_compile', False)
        # Calling byte_compile(args, kwargs) (line 140)
        byte_compile_call_result_24366 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), byte_compile_24349, *[files_24350], **kwargs_24365)
        
        # SSA join for if statement (line 139)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'byte_compile(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'byte_compile' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_24367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24367)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'byte_compile'
        return stypy_return_type_24367


    @norecursion
    def _mutate_outputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_mutate_outputs'
        module_type_store = module_type_store.open_function_context('_mutate_outputs', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_lib._mutate_outputs.__dict__.__setitem__('stypy_localization', localization)
        install_lib._mutate_outputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_lib._mutate_outputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_lib._mutate_outputs.__dict__.__setitem__('stypy_function_name', 'install_lib._mutate_outputs')
        install_lib._mutate_outputs.__dict__.__setitem__('stypy_param_names_list', ['has_any', 'build_cmd', 'cmd_option', 'output_dir'])
        install_lib._mutate_outputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_lib._mutate_outputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_lib._mutate_outputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_lib._mutate_outputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_lib._mutate_outputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_lib._mutate_outputs.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_lib._mutate_outputs', ['has_any', 'build_cmd', 'cmd_option', 'output_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_mutate_outputs', localization, ['has_any', 'build_cmd', 'cmd_option', 'output_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_mutate_outputs(...)' code ##################

        
        
        # Getting the type of 'has_any' (line 148)
        has_any_24368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'has_any')
        # Applying the 'not' unary operator (line 148)
        result_not__24369 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 11), 'not', has_any_24368)
        
        # Testing the type of an if condition (line 148)
        if_condition_24370 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 8), result_not__24369)
        # Assigning a type to the variable 'if_condition_24370' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'if_condition_24370', if_condition_24370)
        # SSA begins for if statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 149)
        list_24371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 149)
        
        # Assigning a type to the variable 'stypy_return_type' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'stypy_return_type', list_24371)
        # SSA join for if statement (line 148)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 151):
        
        # Call to get_finalized_command(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'build_cmd' (line 151)
        build_cmd_24374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 47), 'build_cmd', False)
        # Processing the call keyword arguments (line 151)
        kwargs_24375 = {}
        # Getting the type of 'self' (line 151)
        self_24372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 151)
        get_finalized_command_24373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 20), self_24372, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 151)
        get_finalized_command_call_result_24376 = invoke(stypy.reporting.localization.Localization(__file__, 151, 20), get_finalized_command_24373, *[build_cmd_24374], **kwargs_24375)
        
        # Assigning a type to the variable 'build_cmd' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'build_cmd', get_finalized_command_call_result_24376)
        
        # Assigning a Call to a Name (line 152):
        
        # Call to get_outputs(...): (line 152)
        # Processing the call keyword arguments (line 152)
        kwargs_24379 = {}
        # Getting the type of 'build_cmd' (line 152)
        build_cmd_24377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 22), 'build_cmd', False)
        # Obtaining the member 'get_outputs' of a type (line 152)
        get_outputs_24378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 22), build_cmd_24377, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 152)
        get_outputs_call_result_24380 = invoke(stypy.reporting.localization.Localization(__file__, 152, 22), get_outputs_24378, *[], **kwargs_24379)
        
        # Assigning a type to the variable 'build_files' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'build_files', get_outputs_call_result_24380)
        
        # Assigning a Call to a Name (line 153):
        
        # Call to getattr(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'build_cmd' (line 153)
        build_cmd_24382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 28), 'build_cmd', False)
        # Getting the type of 'cmd_option' (line 153)
        cmd_option_24383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 39), 'cmd_option', False)
        # Processing the call keyword arguments (line 153)
        kwargs_24384 = {}
        # Getting the type of 'getattr' (line 153)
        getattr_24381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 153)
        getattr_call_result_24385 = invoke(stypy.reporting.localization.Localization(__file__, 153, 20), getattr_24381, *[build_cmd_24382, cmd_option_24383], **kwargs_24384)
        
        # Assigning a type to the variable 'build_dir' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'build_dir', getattr_call_result_24385)
        
        # Assigning a BinOp to a Name (line 155):
        
        # Call to len(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'build_dir' (line 155)
        build_dir_24387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 25), 'build_dir', False)
        # Processing the call keyword arguments (line 155)
        kwargs_24388 = {}
        # Getting the type of 'len' (line 155)
        len_24386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 21), 'len', False)
        # Calling len(args, kwargs) (line 155)
        len_call_result_24389 = invoke(stypy.reporting.localization.Localization(__file__, 155, 21), len_24386, *[build_dir_24387], **kwargs_24388)
        
        
        # Call to len(...): (line 155)
        # Processing the call arguments (line 155)
        # Getting the type of 'os' (line 155)
        os_24391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 42), 'os', False)
        # Obtaining the member 'sep' of a type (line 155)
        sep_24392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 42), os_24391, 'sep')
        # Processing the call keyword arguments (line 155)
        kwargs_24393 = {}
        # Getting the type of 'len' (line 155)
        len_24390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 38), 'len', False)
        # Calling len(args, kwargs) (line 155)
        len_call_result_24394 = invoke(stypy.reporting.localization.Localization(__file__, 155, 38), len_24390, *[sep_24392], **kwargs_24393)
        
        # Applying the binary operator '+' (line 155)
        result_add_24395 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 21), '+', len_call_result_24389, len_call_result_24394)
        
        # Assigning a type to the variable 'prefix_len' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'prefix_len', result_add_24395)
        
        # Assigning a List to a Name (line 156):
        
        # Obtaining an instance of the builtin type 'list' (line 156)
        list_24396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 156)
        
        # Assigning a type to the variable 'outputs' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'outputs', list_24396)
        
        # Getting the type of 'build_files' (line 157)
        build_files_24397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'build_files')
        # Testing the type of a for loop iterable (line 157)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 157, 8), build_files_24397)
        # Getting the type of the for loop variable (line 157)
        for_loop_var_24398 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 157, 8), build_files_24397)
        # Assigning a type to the variable 'file' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'file', for_loop_var_24398)
        # SSA begins for a for statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 158)
        # Processing the call arguments (line 158)
        
        # Call to join(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'output_dir' (line 158)
        output_dir_24404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 40), 'output_dir', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'prefix_len' (line 158)
        prefix_len_24405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 57), 'prefix_len', False)
        slice_24406 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 158, 52), prefix_len_24405, None, None)
        # Getting the type of 'file' (line 158)
        file_24407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 52), 'file', False)
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___24408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 52), file_24407, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_24409 = invoke(stypy.reporting.localization.Localization(__file__, 158, 52), getitem___24408, slice_24406)
        
        # Processing the call keyword arguments (line 158)
        kwargs_24410 = {}
        # Getting the type of 'os' (line 158)
        os_24401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 158)
        path_24402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 27), os_24401, 'path')
        # Obtaining the member 'join' of a type (line 158)
        join_24403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 27), path_24402, 'join')
        # Calling join(args, kwargs) (line 158)
        join_call_result_24411 = invoke(stypy.reporting.localization.Localization(__file__, 158, 27), join_24403, *[output_dir_24404, subscript_call_result_24409], **kwargs_24410)
        
        # Processing the call keyword arguments (line 158)
        kwargs_24412 = {}
        # Getting the type of 'outputs' (line 158)
        outputs_24399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'outputs', False)
        # Obtaining the member 'append' of a type (line 158)
        append_24400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 12), outputs_24399, 'append')
        # Calling append(args, kwargs) (line 158)
        append_call_result_24413 = invoke(stypy.reporting.localization.Localization(__file__, 158, 12), append_24400, *[join_call_result_24411], **kwargs_24412)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'outputs' (line 160)
        outputs_24414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'outputs')
        # Assigning a type to the variable 'stypy_return_type' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'stypy_return_type', outputs_24414)
        
        # ################# End of '_mutate_outputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_mutate_outputs' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_24415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24415)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_mutate_outputs'
        return stypy_return_type_24415


    @norecursion
    def _bytecode_filenames(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_bytecode_filenames'
        module_type_store = module_type_store.open_function_context('_bytecode_filenames', 162, 4, False)
        # Assigning a type to the variable 'self' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_lib._bytecode_filenames.__dict__.__setitem__('stypy_localization', localization)
        install_lib._bytecode_filenames.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_lib._bytecode_filenames.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_lib._bytecode_filenames.__dict__.__setitem__('stypy_function_name', 'install_lib._bytecode_filenames')
        install_lib._bytecode_filenames.__dict__.__setitem__('stypy_param_names_list', ['py_filenames'])
        install_lib._bytecode_filenames.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_lib._bytecode_filenames.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_lib._bytecode_filenames.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_lib._bytecode_filenames.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_lib._bytecode_filenames.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_lib._bytecode_filenames.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_lib._bytecode_filenames', ['py_filenames'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_bytecode_filenames', localization, ['py_filenames'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_bytecode_filenames(...)' code ##################

        
        # Assigning a List to a Name (line 163):
        
        # Obtaining an instance of the builtin type 'list' (line 163)
        list_24416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 163)
        
        # Assigning a type to the variable 'bytecode_files' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'bytecode_files', list_24416)
        
        # Getting the type of 'py_filenames' (line 164)
        py_filenames_24417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 23), 'py_filenames')
        # Testing the type of a for loop iterable (line 164)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 164, 8), py_filenames_24417)
        # Getting the type of the for loop variable (line 164)
        for_loop_var_24418 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 164, 8), py_filenames_24417)
        # Assigning a type to the variable 'py_file' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'py_file', for_loop_var_24418)
        # SSA begins for a for statement (line 164)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 168):
        
        # Obtaining the type of the subscript
        int_24419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 62), 'int')
        
        # Call to splitext(...): (line 168)
        # Processing the call arguments (line 168)
        
        # Call to normcase(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'py_file' (line 168)
        py_file_24426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 52), 'py_file', False)
        # Processing the call keyword arguments (line 168)
        kwargs_24427 = {}
        # Getting the type of 'os' (line 168)
        os_24423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 35), 'os', False)
        # Obtaining the member 'path' of a type (line 168)
        path_24424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 35), os_24423, 'path')
        # Obtaining the member 'normcase' of a type (line 168)
        normcase_24425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 35), path_24424, 'normcase')
        # Calling normcase(args, kwargs) (line 168)
        normcase_call_result_24428 = invoke(stypy.reporting.localization.Localization(__file__, 168, 35), normcase_24425, *[py_file_24426], **kwargs_24427)
        
        # Processing the call keyword arguments (line 168)
        kwargs_24429 = {}
        # Getting the type of 'os' (line 168)
        os_24420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 168)
        path_24421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 18), os_24420, 'path')
        # Obtaining the member 'splitext' of a type (line 168)
        splitext_24422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 18), path_24421, 'splitext')
        # Calling splitext(args, kwargs) (line 168)
        splitext_call_result_24430 = invoke(stypy.reporting.localization.Localization(__file__, 168, 18), splitext_24422, *[normcase_call_result_24428], **kwargs_24429)
        
        # Obtaining the member '__getitem__' of a type (line 168)
        getitem___24431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 18), splitext_call_result_24430, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 168)
        subscript_call_result_24432 = invoke(stypy.reporting.localization.Localization(__file__, 168, 18), getitem___24431, int_24419)
        
        # Assigning a type to the variable 'ext' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'ext', subscript_call_result_24432)
        
        
        # Getting the type of 'ext' (line 169)
        ext_24433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'ext')
        # Getting the type of 'PYTHON_SOURCE_EXTENSION' (line 169)
        PYTHON_SOURCE_EXTENSION_24434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 22), 'PYTHON_SOURCE_EXTENSION')
        # Applying the binary operator '!=' (line 169)
        result_ne_24435 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 15), '!=', ext_24433, PYTHON_SOURCE_EXTENSION_24434)
        
        # Testing the type of an if condition (line 169)
        if_condition_24436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 12), result_ne_24435)
        # Assigning a type to the variable 'if_condition_24436' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'if_condition_24436', if_condition_24436)
        # SSA begins for if statement (line 169)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 169)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 171)
        self_24437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'self')
        # Obtaining the member 'compile' of a type (line 171)
        compile_24438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 15), self_24437, 'compile')
        # Testing the type of an if condition (line 171)
        if_condition_24439 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 12), compile_24438)
        # Assigning a type to the variable 'if_condition_24439' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'if_condition_24439', if_condition_24439)
        # SSA begins for if statement (line 171)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'py_file' (line 172)
        py_file_24442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 38), 'py_file', False)
        str_24443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 48), 'str', 'c')
        # Applying the binary operator '+' (line 172)
        result_add_24444 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 38), '+', py_file_24442, str_24443)
        
        # Processing the call keyword arguments (line 172)
        kwargs_24445 = {}
        # Getting the type of 'bytecode_files' (line 172)
        bytecode_files_24440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'bytecode_files', False)
        # Obtaining the member 'append' of a type (line 172)
        append_24441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 16), bytecode_files_24440, 'append')
        # Calling append(args, kwargs) (line 172)
        append_call_result_24446 = invoke(stypy.reporting.localization.Localization(__file__, 172, 16), append_24441, *[result_add_24444], **kwargs_24445)
        
        # SSA join for if statement (line 171)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 173)
        self_24447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'self')
        # Obtaining the member 'optimize' of a type (line 173)
        optimize_24448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 15), self_24447, 'optimize')
        int_24449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 31), 'int')
        # Applying the binary operator '>' (line 173)
        result_gt_24450 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 15), '>', optimize_24448, int_24449)
        
        # Testing the type of an if condition (line 173)
        if_condition_24451 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 12), result_gt_24450)
        # Assigning a type to the variable 'if_condition_24451' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'if_condition_24451', if_condition_24451)
        # SSA begins for if statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'py_file' (line 174)
        py_file_24454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 38), 'py_file', False)
        str_24455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 48), 'str', 'o')
        # Applying the binary operator '+' (line 174)
        result_add_24456 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 38), '+', py_file_24454, str_24455)
        
        # Processing the call keyword arguments (line 174)
        kwargs_24457 = {}
        # Getting the type of 'bytecode_files' (line 174)
        bytecode_files_24452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'bytecode_files', False)
        # Obtaining the member 'append' of a type (line 174)
        append_24453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), bytecode_files_24452, 'append')
        # Calling append(args, kwargs) (line 174)
        append_call_result_24458 = invoke(stypy.reporting.localization.Localization(__file__, 174, 16), append_24453, *[result_add_24456], **kwargs_24457)
        
        # SSA join for if statement (line 173)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'bytecode_files' (line 176)
        bytecode_files_24459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'bytecode_files')
        # Assigning a type to the variable 'stypy_return_type' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'stypy_return_type', bytecode_files_24459)
        
        # ################# End of '_bytecode_filenames(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_bytecode_filenames' in the type store
        # Getting the type of 'stypy_return_type' (line 162)
        stypy_return_type_24460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24460)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_bytecode_filenames'
        return stypy_return_type_24460


    @norecursion
    def get_outputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_outputs'
        module_type_store = module_type_store.open_function_context('get_outputs', 182, 4, False)
        # Assigning a type to the variable 'self' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_lib.get_outputs.__dict__.__setitem__('stypy_localization', localization)
        install_lib.get_outputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_lib.get_outputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_lib.get_outputs.__dict__.__setitem__('stypy_function_name', 'install_lib.get_outputs')
        install_lib.get_outputs.__dict__.__setitem__('stypy_param_names_list', [])
        install_lib.get_outputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_lib.get_outputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_lib.get_outputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_lib.get_outputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_lib.get_outputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_lib.get_outputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_lib.get_outputs', [], None, None, defaults, varargs, kwargs)

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

        str_24461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, (-1)), 'str', 'Return the list of files that would be installed if this command\n        were actually run.  Not affected by the "dry-run" flag or whether\n        modules have actually been built yet.\n        ')
        
        # Assigning a Call to a Name (line 187):
        
        # Call to _mutate_outputs(...): (line 188)
        # Processing the call arguments (line 188)
        
        # Call to has_pure_modules(...): (line 188)
        # Processing the call keyword arguments (line 188)
        kwargs_24467 = {}
        # Getting the type of 'self' (line 188)
        self_24464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 33), 'self', False)
        # Obtaining the member 'distribution' of a type (line 188)
        distribution_24465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 33), self_24464, 'distribution')
        # Obtaining the member 'has_pure_modules' of a type (line 188)
        has_pure_modules_24466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 33), distribution_24465, 'has_pure_modules')
        # Calling has_pure_modules(args, kwargs) (line 188)
        has_pure_modules_call_result_24468 = invoke(stypy.reporting.localization.Localization(__file__, 188, 33), has_pure_modules_24466, *[], **kwargs_24467)
        
        str_24469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 33), 'str', 'build_py')
        str_24470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 45), 'str', 'build_lib')
        # Getting the type of 'self' (line 190)
        self_24471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 33), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 190)
        install_dir_24472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 33), self_24471, 'install_dir')
        # Processing the call keyword arguments (line 188)
        kwargs_24473 = {}
        # Getting the type of 'self' (line 188)
        self_24462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'self', False)
        # Obtaining the member '_mutate_outputs' of a type (line 188)
        _mutate_outputs_24463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), self_24462, '_mutate_outputs')
        # Calling _mutate_outputs(args, kwargs) (line 188)
        _mutate_outputs_call_result_24474 = invoke(stypy.reporting.localization.Localization(__file__, 188, 12), _mutate_outputs_24463, *[has_pure_modules_call_result_24468, str_24469, str_24470, install_dir_24472], **kwargs_24473)
        
        # Assigning a type to the variable 'pure_outputs' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'pure_outputs', _mutate_outputs_call_result_24474)
        
        # Getting the type of 'self' (line 191)
        self_24475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'self')
        # Obtaining the member 'compile' of a type (line 191)
        compile_24476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 11), self_24475, 'compile')
        # Testing the type of an if condition (line 191)
        if_condition_24477 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 8), compile_24476)
        # Assigning a type to the variable 'if_condition_24477' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'if_condition_24477', if_condition_24477)
        # SSA begins for if statement (line 191)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 192):
        
        # Call to _bytecode_filenames(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'pure_outputs' (line 192)
        pure_outputs_24480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 56), 'pure_outputs', False)
        # Processing the call keyword arguments (line 192)
        kwargs_24481 = {}
        # Getting the type of 'self' (line 192)
        self_24478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 31), 'self', False)
        # Obtaining the member '_bytecode_filenames' of a type (line 192)
        _bytecode_filenames_24479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 31), self_24478, '_bytecode_filenames')
        # Calling _bytecode_filenames(args, kwargs) (line 192)
        _bytecode_filenames_call_result_24482 = invoke(stypy.reporting.localization.Localization(__file__, 192, 31), _bytecode_filenames_24479, *[pure_outputs_24480], **kwargs_24481)
        
        # Assigning a type to the variable 'bytecode_outputs' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'bytecode_outputs', _bytecode_filenames_call_result_24482)
        # SSA branch for the else part of an if statement (line 191)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 194):
        
        # Obtaining an instance of the builtin type 'list' (line 194)
        list_24483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 194)
        
        # Assigning a type to the variable 'bytecode_outputs' (line 194)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'bytecode_outputs', list_24483)
        # SSA join for if statement (line 191)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 196):
        
        # Call to _mutate_outputs(...): (line 197)
        # Processing the call arguments (line 197)
        
        # Call to has_ext_modules(...): (line 197)
        # Processing the call keyword arguments (line 197)
        kwargs_24489 = {}
        # Getting the type of 'self' (line 197)
        self_24486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 33), 'self', False)
        # Obtaining the member 'distribution' of a type (line 197)
        distribution_24487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 33), self_24486, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 197)
        has_ext_modules_24488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 33), distribution_24487, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 197)
        has_ext_modules_call_result_24490 = invoke(stypy.reporting.localization.Localization(__file__, 197, 33), has_ext_modules_24488, *[], **kwargs_24489)
        
        str_24491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 33), 'str', 'build_ext')
        str_24492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 46), 'str', 'build_lib')
        # Getting the type of 'self' (line 199)
        self_24493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 33), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 199)
        install_dir_24494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 33), self_24493, 'install_dir')
        # Processing the call keyword arguments (line 197)
        kwargs_24495 = {}
        # Getting the type of 'self' (line 197)
        self_24484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'self', False)
        # Obtaining the member '_mutate_outputs' of a type (line 197)
        _mutate_outputs_24485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 12), self_24484, '_mutate_outputs')
        # Calling _mutate_outputs(args, kwargs) (line 197)
        _mutate_outputs_call_result_24496 = invoke(stypy.reporting.localization.Localization(__file__, 197, 12), _mutate_outputs_24485, *[has_ext_modules_call_result_24490, str_24491, str_24492, install_dir_24494], **kwargs_24495)
        
        # Assigning a type to the variable 'ext_outputs' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'ext_outputs', _mutate_outputs_call_result_24496)
        # Getting the type of 'pure_outputs' (line 201)
        pure_outputs_24497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'pure_outputs')
        # Getting the type of 'bytecode_outputs' (line 201)
        bytecode_outputs_24498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 30), 'bytecode_outputs')
        # Applying the binary operator '+' (line 201)
        result_add_24499 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 15), '+', pure_outputs_24497, bytecode_outputs_24498)
        
        # Getting the type of 'ext_outputs' (line 201)
        ext_outputs_24500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 49), 'ext_outputs')
        # Applying the binary operator '+' (line 201)
        result_add_24501 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 47), '+', result_add_24499, ext_outputs_24500)
        
        # Assigning a type to the variable 'stypy_return_type' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'stypy_return_type', result_add_24501)
        
        # ################# End of 'get_outputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_outputs' in the type store
        # Getting the type of 'stypy_return_type' (line 182)
        stypy_return_type_24502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24502)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_outputs'
        return stypy_return_type_24502


    @norecursion
    def get_inputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_inputs'
        module_type_store = module_type_store.open_function_context('get_inputs', 203, 4, False)
        # Assigning a type to the variable 'self' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_lib.get_inputs.__dict__.__setitem__('stypy_localization', localization)
        install_lib.get_inputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_lib.get_inputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_lib.get_inputs.__dict__.__setitem__('stypy_function_name', 'install_lib.get_inputs')
        install_lib.get_inputs.__dict__.__setitem__('stypy_param_names_list', [])
        install_lib.get_inputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_lib.get_inputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_lib.get_inputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_lib.get_inputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_lib.get_inputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_lib.get_inputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_lib.get_inputs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_inputs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_inputs(...)' code ##################

        str_24503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, (-1)), 'str', "Get the list of files that are input to this command, ie. the\n        files that get installed as they are named in the build tree.\n        The files in this list correspond one-to-one to the output\n        filenames returned by 'get_outputs()'.\n        ")
        
        # Assigning a List to a Name (line 209):
        
        # Obtaining an instance of the builtin type 'list' (line 209)
        list_24504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 209)
        
        # Assigning a type to the variable 'inputs' (line 209)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'inputs', list_24504)
        
        
        # Call to has_pure_modules(...): (line 211)
        # Processing the call keyword arguments (line 211)
        kwargs_24508 = {}
        # Getting the type of 'self' (line 211)
        self_24505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'self', False)
        # Obtaining the member 'distribution' of a type (line 211)
        distribution_24506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), self_24505, 'distribution')
        # Obtaining the member 'has_pure_modules' of a type (line 211)
        has_pure_modules_24507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), distribution_24506, 'has_pure_modules')
        # Calling has_pure_modules(args, kwargs) (line 211)
        has_pure_modules_call_result_24509 = invoke(stypy.reporting.localization.Localization(__file__, 211, 11), has_pure_modules_24507, *[], **kwargs_24508)
        
        # Testing the type of an if condition (line 211)
        if_condition_24510 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 8), has_pure_modules_call_result_24509)
        # Assigning a type to the variable 'if_condition_24510' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'if_condition_24510', if_condition_24510)
        # SSA begins for if statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 212):
        
        # Call to get_finalized_command(...): (line 212)
        # Processing the call arguments (line 212)
        str_24513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 50), 'str', 'build_py')
        # Processing the call keyword arguments (line 212)
        kwargs_24514 = {}
        # Getting the type of 'self' (line 212)
        self_24511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 23), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 212)
        get_finalized_command_24512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 23), self_24511, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 212)
        get_finalized_command_call_result_24515 = invoke(stypy.reporting.localization.Localization(__file__, 212, 23), get_finalized_command_24512, *[str_24513], **kwargs_24514)
        
        # Assigning a type to the variable 'build_py' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'build_py', get_finalized_command_call_result_24515)
        
        # Call to extend(...): (line 213)
        # Processing the call arguments (line 213)
        
        # Call to get_outputs(...): (line 213)
        # Processing the call keyword arguments (line 213)
        kwargs_24520 = {}
        # Getting the type of 'build_py' (line 213)
        build_py_24518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 26), 'build_py', False)
        # Obtaining the member 'get_outputs' of a type (line 213)
        get_outputs_24519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 26), build_py_24518, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 213)
        get_outputs_call_result_24521 = invoke(stypy.reporting.localization.Localization(__file__, 213, 26), get_outputs_24519, *[], **kwargs_24520)
        
        # Processing the call keyword arguments (line 213)
        kwargs_24522 = {}
        # Getting the type of 'inputs' (line 213)
        inputs_24516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'inputs', False)
        # Obtaining the member 'extend' of a type (line 213)
        extend_24517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 12), inputs_24516, 'extend')
        # Calling extend(args, kwargs) (line 213)
        extend_call_result_24523 = invoke(stypy.reporting.localization.Localization(__file__, 213, 12), extend_24517, *[get_outputs_call_result_24521], **kwargs_24522)
        
        # SSA join for if statement (line 211)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to has_ext_modules(...): (line 215)
        # Processing the call keyword arguments (line 215)
        kwargs_24527 = {}
        # Getting the type of 'self' (line 215)
        self_24524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'self', False)
        # Obtaining the member 'distribution' of a type (line 215)
        distribution_24525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 11), self_24524, 'distribution')
        # Obtaining the member 'has_ext_modules' of a type (line 215)
        has_ext_modules_24526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 11), distribution_24525, 'has_ext_modules')
        # Calling has_ext_modules(args, kwargs) (line 215)
        has_ext_modules_call_result_24528 = invoke(stypy.reporting.localization.Localization(__file__, 215, 11), has_ext_modules_24526, *[], **kwargs_24527)
        
        # Testing the type of an if condition (line 215)
        if_condition_24529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 8), has_ext_modules_call_result_24528)
        # Assigning a type to the variable 'if_condition_24529' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'if_condition_24529', if_condition_24529)
        # SSA begins for if statement (line 215)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 216):
        
        # Call to get_finalized_command(...): (line 216)
        # Processing the call arguments (line 216)
        str_24532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 51), 'str', 'build_ext')
        # Processing the call keyword arguments (line 216)
        kwargs_24533 = {}
        # Getting the type of 'self' (line 216)
        self_24530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 24), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 216)
        get_finalized_command_24531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 24), self_24530, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 216)
        get_finalized_command_call_result_24534 = invoke(stypy.reporting.localization.Localization(__file__, 216, 24), get_finalized_command_24531, *[str_24532], **kwargs_24533)
        
        # Assigning a type to the variable 'build_ext' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'build_ext', get_finalized_command_call_result_24534)
        
        # Call to extend(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Call to get_outputs(...): (line 217)
        # Processing the call keyword arguments (line 217)
        kwargs_24539 = {}
        # Getting the type of 'build_ext' (line 217)
        build_ext_24537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 26), 'build_ext', False)
        # Obtaining the member 'get_outputs' of a type (line 217)
        get_outputs_24538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 26), build_ext_24537, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 217)
        get_outputs_call_result_24540 = invoke(stypy.reporting.localization.Localization(__file__, 217, 26), get_outputs_24538, *[], **kwargs_24539)
        
        # Processing the call keyword arguments (line 217)
        kwargs_24541 = {}
        # Getting the type of 'inputs' (line 217)
        inputs_24535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'inputs', False)
        # Obtaining the member 'extend' of a type (line 217)
        extend_24536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 12), inputs_24535, 'extend')
        # Calling extend(args, kwargs) (line 217)
        extend_call_result_24542 = invoke(stypy.reporting.localization.Localization(__file__, 217, 12), extend_24536, *[get_outputs_call_result_24540], **kwargs_24541)
        
        # SSA join for if statement (line 215)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'inputs' (line 219)
        inputs_24543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 15), 'inputs')
        # Assigning a type to the variable 'stypy_return_type' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'stypy_return_type', inputs_24543)
        
        # ################# End of 'get_inputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_inputs' in the type store
        # Getting the type of 'stypy_return_type' (line 203)
        stypy_return_type_24544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24544)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_inputs'
        return stypy_return_type_24544


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 21, 0, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_lib.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'install_lib' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'install_lib', install_lib)

# Assigning a Str to a Name (line 23):
str_24545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 18), 'str', 'install all Python modules (extensions and pure Python)')
# Getting the type of 'install_lib'
install_lib_24546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_lib')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_lib_24546, 'description', str_24545)

# Assigning a List to a Name (line 40):

# Obtaining an instance of the builtin type 'list' (line 40)
list_24547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 40)
# Adding element type (line 40)

# Obtaining an instance of the builtin type 'tuple' (line 41)
tuple_24548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 41)
# Adding element type (line 41)
str_24549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 9), 'str', 'install-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 9), tuple_24548, str_24549)
# Adding element type (line 41)
str_24550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 9), tuple_24548, str_24550)
# Adding element type (line 41)
str_24551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'str', 'directory to install to')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 9), tuple_24548, str_24551)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), list_24547, tuple_24548)
# Adding element type (line 40)

# Obtaining an instance of the builtin type 'tuple' (line 42)
tuple_24552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 42)
# Adding element type (line 42)
str_24553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 9), 'str', 'build-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), tuple_24552, str_24553)
# Adding element type (line 42)
str_24554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 22), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), tuple_24552, str_24554)
# Adding element type (line 42)
str_24555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 27), 'str', 'build directory (where to install from)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 9), tuple_24552, str_24555)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), list_24547, tuple_24552)
# Adding element type (line 40)

# Obtaining an instance of the builtin type 'tuple' (line 43)
tuple_24556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 43)
# Adding element type (line 43)
str_24557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 9), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 9), tuple_24556, str_24557)
# Adding element type (line 43)
str_24558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 18), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 9), tuple_24556, str_24558)
# Adding element type (line 43)
str_24559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 23), 'str', 'force installation (overwrite existing files)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 9), tuple_24556, str_24559)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), list_24547, tuple_24556)
# Adding element type (line 40)

# Obtaining an instance of the builtin type 'tuple' (line 44)
tuple_24560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 44)
# Adding element type (line 44)
str_24561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 9), 'str', 'compile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 9), tuple_24560, str_24561)
# Adding element type (line 44)
str_24562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 20), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 9), tuple_24560, str_24562)
# Adding element type (line 44)
str_24563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 25), 'str', 'compile .py to .pyc [default]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 9), tuple_24560, str_24563)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), list_24547, tuple_24560)
# Adding element type (line 40)

# Obtaining an instance of the builtin type 'tuple' (line 45)
tuple_24564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 45)
# Adding element type (line 45)
str_24565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 9), 'str', 'no-compile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 9), tuple_24564, str_24565)
# Adding element type (line 45)
# Getting the type of 'None' (line 45)
None_24566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 9), tuple_24564, None_24566)
# Adding element type (line 45)
str_24567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 29), 'str', "don't compile .py files")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 9), tuple_24564, str_24567)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), list_24547, tuple_24564)
# Adding element type (line 40)

# Obtaining an instance of the builtin type 'tuple' (line 46)
tuple_24568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 46)
# Adding element type (line 46)
str_24569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 9), 'str', 'optimize=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 9), tuple_24568, str_24569)
# Adding element type (line 46)
str_24570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'str', 'O')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 9), tuple_24568, str_24570)
# Adding element type (line 46)
str_24571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 9), 'str', 'also compile with optimization: -O1 for "python -O", -O2 for "python -OO", and -O0 to disable [default: -O0]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 9), tuple_24568, str_24571)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), list_24547, tuple_24568)
# Adding element type (line 40)

# Obtaining an instance of the builtin type 'tuple' (line 49)
tuple_24572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 49)
# Adding element type (line 49)
str_24573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 9), 'str', 'skip-build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 9), tuple_24572, str_24573)
# Adding element type (line 49)
# Getting the type of 'None' (line 49)
None_24574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 9), tuple_24572, None_24574)
# Adding element type (line 49)
str_24575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 29), 'str', 'skip the build steps')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 9), tuple_24572, str_24575)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), list_24547, tuple_24572)

# Getting the type of 'install_lib'
install_lib_24576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_lib')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_lib_24576, 'user_options', list_24547)

# Assigning a List to a Name (line 52):

# Obtaining an instance of the builtin type 'list' (line 52)
list_24577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 52)
# Adding element type (line 52)
str_24578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 23), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 22), list_24577, str_24578)
# Adding element type (line 52)
str_24579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 32), 'str', 'compile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 22), list_24577, str_24579)
# Adding element type (line 52)
str_24580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 43), 'str', 'skip-build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 22), list_24577, str_24580)

# Getting the type of 'install_lib'
install_lib_24581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_lib')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_lib_24581, 'boolean_options', list_24577)

# Assigning a Dict to a Name (line 53):

# Obtaining an instance of the builtin type 'dict' (line 53)
dict_24582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 53)
# Adding element type (key, value) (line 53)
str_24583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 20), 'str', 'no-compile')
str_24584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 35), 'str', 'compile')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 19), dict_24582, (str_24583, str_24584))

# Getting the type of 'install_lib'
install_lib_24585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_lib')
# Setting the type of the member 'negative_opt' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_lib_24585, 'negative_opt', dict_24582)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
