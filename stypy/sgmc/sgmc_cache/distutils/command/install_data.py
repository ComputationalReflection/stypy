
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.install_data
2: 
3: Implements the Distutils 'install_data' command, for installing
4: platform-independent data files.'''
5: 
6: # contributed by Bastian Kleineidam
7: 
8: __revision__ = "$Id$"
9: 
10: import os
11: from distutils.core import Command
12: from distutils.util import change_root, convert_path
13: 
14: class install_data(Command):
15: 
16:     description = "install data files"
17: 
18:     user_options = [
19:         ('install-dir=', 'd',
20:          "base directory for installing data files "
21:          "(default: installation base dir)"),
22:         ('root=', None,
23:          "install everything relative to this alternate root directory"),
24:         ('force', 'f', "force installation (overwrite existing files)"),
25:         ]
26: 
27:     boolean_options = ['force']
28: 
29:     def initialize_options(self):
30:         self.install_dir = None
31:         self.outfiles = []
32:         self.root = None
33:         self.force = 0
34:         self.data_files = self.distribution.data_files
35:         self.warn_dir = 1
36: 
37:     def finalize_options(self):
38:         self.set_undefined_options('install',
39:                                    ('install_data', 'install_dir'),
40:                                    ('root', 'root'),
41:                                    ('force', 'force'),
42:                                   )
43: 
44:     def run(self):
45:         self.mkpath(self.install_dir)
46:         for f in self.data_files:
47:             if isinstance(f, str):
48:                 # it's a simple file, so copy it
49:                 f = convert_path(f)
50:                 if self.warn_dir:
51:                     self.warn("setup script did not provide a directory for "
52:                               "'%s' -- installing right in '%s'" %
53:                               (f, self.install_dir))
54:                 (out, _) = self.copy_file(f, self.install_dir)
55:                 self.outfiles.append(out)
56:             else:
57:                 # it's a tuple with path to install to and a list of files
58:                 dir = convert_path(f[0])
59:                 if not os.path.isabs(dir):
60:                     dir = os.path.join(self.install_dir, dir)
61:                 elif self.root:
62:                     dir = change_root(self.root, dir)
63:                 self.mkpath(dir)
64: 
65:                 if f[1] == []:
66:                     # If there are no files listed, the user must be
67:                     # trying to create an empty directory, so add the
68:                     # directory to the list of output files.
69:                     self.outfiles.append(dir)
70:                 else:
71:                     # Copy files, adding them to the list of output files.
72:                     for data in f[1]:
73:                         data = convert_path(data)
74:                         (out, _) = self.copy_file(data, dir)
75:                         self.outfiles.append(out)
76: 
77:     def get_inputs(self):
78:         return self.data_files or []
79: 
80:     def get_outputs(self):
81:         return self.outfiles
82: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_23646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', "distutils.command.install_data\n\nImplements the Distutils 'install_data' command, for installing\nplatform-independent data files.")

# Assigning a Str to a Name (line 8):

# Assigning a Str to a Name (line 8):
str_23647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__revision__', str_23647)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import os' statement (line 10)
import os

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.core import Command' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_23648 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core')

if (type(import_23648) is not StypyTypeError):

    if (import_23648 != 'pyd_module'):
        __import__(import_23648)
        sys_modules_23649 = sys.modules[import_23648]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core', sys_modules_23649.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_23649, sys_modules_23649.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core', import_23648)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.util import change_root, convert_path' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_23650 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.util')

if (type(import_23650) is not StypyTypeError):

    if (import_23650 != 'pyd_module'):
        __import__(import_23650)
        sys_modules_23651 = sys.modules[import_23650]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.util', sys_modules_23651.module_type_store, module_type_store, ['change_root', 'convert_path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_23651, sys_modules_23651.module_type_store, module_type_store)
    else:
        from distutils.util import change_root, convert_path

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.util', None, module_type_store, ['change_root', 'convert_path'], [change_root, convert_path])

else:
    # Assigning a type to the variable 'distutils.util' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.util', import_23650)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

# Declaration of the 'install_data' class
# Getting the type of 'Command' (line 14)
Command_23652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'Command')

class install_data(Command_23652, ):
    
    # Assigning a Str to a Name (line 16):
    
    # Assigning a List to a Name (line 18):
    
    # Assigning a List to a Name (line 27):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_data.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        install_data.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_data.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_data.initialize_options.__dict__.__setitem__('stypy_function_name', 'install_data.initialize_options')
        install_data.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install_data.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_data.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_data.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_data.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_data.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_data.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_data.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 30):
        
        # Assigning a Name to a Attribute (line 30):
        # Getting the type of 'None' (line 30)
        None_23653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 27), 'None')
        # Getting the type of 'self' (line 30)
        self_23654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'install_dir' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_23654, 'install_dir', None_23653)
        
        # Assigning a List to a Attribute (line 31):
        
        # Assigning a List to a Attribute (line 31):
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_23655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        
        # Getting the type of 'self' (line 31)
        self_23656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'outfiles' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_23656, 'outfiles', list_23655)
        
        # Assigning a Name to a Attribute (line 32):
        
        # Assigning a Name to a Attribute (line 32):
        # Getting the type of 'None' (line 32)
        None_23657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'None')
        # Getting the type of 'self' (line 32)
        self_23658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member 'root' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_23658, 'root', None_23657)
        
        # Assigning a Num to a Attribute (line 33):
        
        # Assigning a Num to a Attribute (line 33):
        int_23659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'int')
        # Getting the type of 'self' (line 33)
        self_23660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'force' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_23660, 'force', int_23659)
        
        # Assigning a Attribute to a Attribute (line 34):
        
        # Assigning a Attribute to a Attribute (line 34):
        # Getting the type of 'self' (line 34)
        self_23661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 26), 'self')
        # Obtaining the member 'distribution' of a type (line 34)
        distribution_23662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 26), self_23661, 'distribution')
        # Obtaining the member 'data_files' of a type (line 34)
        data_files_23663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 26), distribution_23662, 'data_files')
        # Getting the type of 'self' (line 34)
        self_23664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'data_files' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_23664, 'data_files', data_files_23663)
        
        # Assigning a Num to a Attribute (line 35):
        
        # Assigning a Num to a Attribute (line 35):
        int_23665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 24), 'int')
        # Getting the type of 'self' (line 35)
        self_23666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self')
        # Setting the type of the member 'warn_dir' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_23666, 'warn_dir', int_23665)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_23667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23667)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_23667


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
        install_data.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        install_data.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_data.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_data.finalize_options.__dict__.__setitem__('stypy_function_name', 'install_data.finalize_options')
        install_data.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install_data.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_data.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_data.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_data.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_data.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_data.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_data.finalize_options', [], None, None, defaults, varargs, kwargs)

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
        str_23670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 35), 'str', 'install')
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_23671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        str_23672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'str', 'install_data')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 36), tuple_23671, str_23672)
        # Adding element type (line 39)
        str_23673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 52), 'str', 'install_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 36), tuple_23671, str_23673)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 40)
        tuple_23674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 40)
        # Adding element type (line 40)
        str_23675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 36), 'str', 'root')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 36), tuple_23674, str_23675)
        # Adding element type (line 40)
        str_23676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 44), 'str', 'root')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 36), tuple_23674, str_23676)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_23677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        str_23678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 36), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 36), tuple_23677, str_23678)
        # Adding element type (line 41)
        str_23679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 45), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 36), tuple_23677, str_23679)
        
        # Processing the call keyword arguments (line 38)
        kwargs_23680 = {}
        # Getting the type of 'self' (line 38)
        self_23668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 38)
        set_undefined_options_23669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_23668, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 38)
        set_undefined_options_call_result_23681 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), set_undefined_options_23669, *[str_23670, tuple_23671, tuple_23674, tuple_23677], **kwargs_23680)
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_23682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23682)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_23682


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_data.run.__dict__.__setitem__('stypy_localization', localization)
        install_data.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_data.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_data.run.__dict__.__setitem__('stypy_function_name', 'install_data.run')
        install_data.run.__dict__.__setitem__('stypy_param_names_list', [])
        install_data.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_data.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_data.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_data.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_data.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_data.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_data.run', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to mkpath(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'self' (line 45)
        self_23685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 45)
        install_dir_23686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 20), self_23685, 'install_dir')
        # Processing the call keyword arguments (line 45)
        kwargs_23687 = {}
        # Getting the type of 'self' (line 45)
        self_23683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 45)
        mkpath_23684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_23683, 'mkpath')
        # Calling mkpath(args, kwargs) (line 45)
        mkpath_call_result_23688 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), mkpath_23684, *[install_dir_23686], **kwargs_23687)
        
        
        # Getting the type of 'self' (line 46)
        self_23689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 17), 'self')
        # Obtaining the member 'data_files' of a type (line 46)
        data_files_23690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 17), self_23689, 'data_files')
        # Testing the type of a for loop iterable (line 46)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 46, 8), data_files_23690)
        # Getting the type of the for loop variable (line 46)
        for_loop_var_23691 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 46, 8), data_files_23690)
        # Assigning a type to the variable 'f' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'f', for_loop_var_23691)
        # SSA begins for a for statement (line 46)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 47)
        # Getting the type of 'str' (line 47)
        str_23692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 29), 'str')
        # Getting the type of 'f' (line 47)
        f_23693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'f')
        
        (may_be_23694, more_types_in_union_23695) = may_be_subtype(str_23692, f_23693)

        if may_be_23694:

            if more_types_in_union_23695:
                # Runtime conditional SSA (line 47)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'f' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'f', remove_not_subtype_from_union(f_23693, str))
            
            # Assigning a Call to a Name (line 49):
            
            # Assigning a Call to a Name (line 49):
            
            # Call to convert_path(...): (line 49)
            # Processing the call arguments (line 49)
            # Getting the type of 'f' (line 49)
            f_23697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 33), 'f', False)
            # Processing the call keyword arguments (line 49)
            kwargs_23698 = {}
            # Getting the type of 'convert_path' (line 49)
            convert_path_23696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 20), 'convert_path', False)
            # Calling convert_path(args, kwargs) (line 49)
            convert_path_call_result_23699 = invoke(stypy.reporting.localization.Localization(__file__, 49, 20), convert_path_23696, *[f_23697], **kwargs_23698)
            
            # Assigning a type to the variable 'f' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'f', convert_path_call_result_23699)
            
            # Getting the type of 'self' (line 50)
            self_23700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'self')
            # Obtaining the member 'warn_dir' of a type (line 50)
            warn_dir_23701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 19), self_23700, 'warn_dir')
            # Testing the type of an if condition (line 50)
            if_condition_23702 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 16), warn_dir_23701)
            # Assigning a type to the variable 'if_condition_23702' (line 50)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 16), 'if_condition_23702', if_condition_23702)
            # SSA begins for if statement (line 50)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to warn(...): (line 51)
            # Processing the call arguments (line 51)
            str_23705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 30), 'str', "setup script did not provide a directory for '%s' -- installing right in '%s'")
            
            # Obtaining an instance of the builtin type 'tuple' (line 53)
            tuple_23706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 31), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 53)
            # Adding element type (line 53)
            # Getting the type of 'f' (line 53)
            f_23707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 31), 'f', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 31), tuple_23706, f_23707)
            # Adding element type (line 53)
            # Getting the type of 'self' (line 53)
            self_23708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 34), 'self', False)
            # Obtaining the member 'install_dir' of a type (line 53)
            install_dir_23709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 34), self_23708, 'install_dir')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 31), tuple_23706, install_dir_23709)
            
            # Applying the binary operator '%' (line 51)
            result_mod_23710 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 30), '%', str_23705, tuple_23706)
            
            # Processing the call keyword arguments (line 51)
            kwargs_23711 = {}
            # Getting the type of 'self' (line 51)
            self_23703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'self', False)
            # Obtaining the member 'warn' of a type (line 51)
            warn_23704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 20), self_23703, 'warn')
            # Calling warn(args, kwargs) (line 51)
            warn_call_result_23712 = invoke(stypy.reporting.localization.Localization(__file__, 51, 20), warn_23704, *[result_mod_23710], **kwargs_23711)
            
            # SSA join for if statement (line 50)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Tuple (line 54):
            
            # Assigning a Subscript to a Name (line 54):
            
            # Obtaining the type of the subscript
            int_23713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 16), 'int')
            
            # Call to copy_file(...): (line 54)
            # Processing the call arguments (line 54)
            # Getting the type of 'f' (line 54)
            f_23716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 42), 'f', False)
            # Getting the type of 'self' (line 54)
            self_23717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 45), 'self', False)
            # Obtaining the member 'install_dir' of a type (line 54)
            install_dir_23718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 45), self_23717, 'install_dir')
            # Processing the call keyword arguments (line 54)
            kwargs_23719 = {}
            # Getting the type of 'self' (line 54)
            self_23714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 27), 'self', False)
            # Obtaining the member 'copy_file' of a type (line 54)
            copy_file_23715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 27), self_23714, 'copy_file')
            # Calling copy_file(args, kwargs) (line 54)
            copy_file_call_result_23720 = invoke(stypy.reporting.localization.Localization(__file__, 54, 27), copy_file_23715, *[f_23716, install_dir_23718], **kwargs_23719)
            
            # Obtaining the member '__getitem__' of a type (line 54)
            getitem___23721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), copy_file_call_result_23720, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 54)
            subscript_call_result_23722 = invoke(stypy.reporting.localization.Localization(__file__, 54, 16), getitem___23721, int_23713)
            
            # Assigning a type to the variable 'tuple_var_assignment_23642' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'tuple_var_assignment_23642', subscript_call_result_23722)
            
            # Assigning a Subscript to a Name (line 54):
            
            # Obtaining the type of the subscript
            int_23723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 16), 'int')
            
            # Call to copy_file(...): (line 54)
            # Processing the call arguments (line 54)
            # Getting the type of 'f' (line 54)
            f_23726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 42), 'f', False)
            # Getting the type of 'self' (line 54)
            self_23727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 45), 'self', False)
            # Obtaining the member 'install_dir' of a type (line 54)
            install_dir_23728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 45), self_23727, 'install_dir')
            # Processing the call keyword arguments (line 54)
            kwargs_23729 = {}
            # Getting the type of 'self' (line 54)
            self_23724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 27), 'self', False)
            # Obtaining the member 'copy_file' of a type (line 54)
            copy_file_23725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 27), self_23724, 'copy_file')
            # Calling copy_file(args, kwargs) (line 54)
            copy_file_call_result_23730 = invoke(stypy.reporting.localization.Localization(__file__, 54, 27), copy_file_23725, *[f_23726, install_dir_23728], **kwargs_23729)
            
            # Obtaining the member '__getitem__' of a type (line 54)
            getitem___23731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), copy_file_call_result_23730, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 54)
            subscript_call_result_23732 = invoke(stypy.reporting.localization.Localization(__file__, 54, 16), getitem___23731, int_23723)
            
            # Assigning a type to the variable 'tuple_var_assignment_23643' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'tuple_var_assignment_23643', subscript_call_result_23732)
            
            # Assigning a Name to a Name (line 54):
            # Getting the type of 'tuple_var_assignment_23642' (line 54)
            tuple_var_assignment_23642_23733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'tuple_var_assignment_23642')
            # Assigning a type to the variable 'out' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'out', tuple_var_assignment_23642_23733)
            
            # Assigning a Name to a Name (line 54):
            # Getting the type of 'tuple_var_assignment_23643' (line 54)
            tuple_var_assignment_23643_23734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'tuple_var_assignment_23643')
            # Assigning a type to the variable '_' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 22), '_', tuple_var_assignment_23643_23734)
            
            # Call to append(...): (line 55)
            # Processing the call arguments (line 55)
            # Getting the type of 'out' (line 55)
            out_23738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 37), 'out', False)
            # Processing the call keyword arguments (line 55)
            kwargs_23739 = {}
            # Getting the type of 'self' (line 55)
            self_23735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'self', False)
            # Obtaining the member 'outfiles' of a type (line 55)
            outfiles_23736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), self_23735, 'outfiles')
            # Obtaining the member 'append' of a type (line 55)
            append_23737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), outfiles_23736, 'append')
            # Calling append(args, kwargs) (line 55)
            append_call_result_23740 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), append_23737, *[out_23738], **kwargs_23739)
            

            if more_types_in_union_23695:
                # Runtime conditional SSA for else branch (line 47)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_23694) or more_types_in_union_23695):
            # Assigning a type to the variable 'f' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'f', remove_subtype_from_union(f_23693, str))
            
            # Assigning a Call to a Name (line 58):
            
            # Assigning a Call to a Name (line 58):
            
            # Call to convert_path(...): (line 58)
            # Processing the call arguments (line 58)
            
            # Obtaining the type of the subscript
            int_23742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 37), 'int')
            # Getting the type of 'f' (line 58)
            f_23743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 35), 'f', False)
            # Obtaining the member '__getitem__' of a type (line 58)
            getitem___23744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 35), f_23743, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 58)
            subscript_call_result_23745 = invoke(stypy.reporting.localization.Localization(__file__, 58, 35), getitem___23744, int_23742)
            
            # Processing the call keyword arguments (line 58)
            kwargs_23746 = {}
            # Getting the type of 'convert_path' (line 58)
            convert_path_23741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 22), 'convert_path', False)
            # Calling convert_path(args, kwargs) (line 58)
            convert_path_call_result_23747 = invoke(stypy.reporting.localization.Localization(__file__, 58, 22), convert_path_23741, *[subscript_call_result_23745], **kwargs_23746)
            
            # Assigning a type to the variable 'dir' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'dir', convert_path_call_result_23747)
            
            
            
            # Call to isabs(...): (line 59)
            # Processing the call arguments (line 59)
            # Getting the type of 'dir' (line 59)
            dir_23751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 37), 'dir', False)
            # Processing the call keyword arguments (line 59)
            kwargs_23752 = {}
            # Getting the type of 'os' (line 59)
            os_23748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'os', False)
            # Obtaining the member 'path' of a type (line 59)
            path_23749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 23), os_23748, 'path')
            # Obtaining the member 'isabs' of a type (line 59)
            isabs_23750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 23), path_23749, 'isabs')
            # Calling isabs(args, kwargs) (line 59)
            isabs_call_result_23753 = invoke(stypy.reporting.localization.Localization(__file__, 59, 23), isabs_23750, *[dir_23751], **kwargs_23752)
            
            # Applying the 'not' unary operator (line 59)
            result_not__23754 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 19), 'not', isabs_call_result_23753)
            
            # Testing the type of an if condition (line 59)
            if_condition_23755 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 16), result_not__23754)
            # Assigning a type to the variable 'if_condition_23755' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'if_condition_23755', if_condition_23755)
            # SSA begins for if statement (line 59)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 60):
            
            # Assigning a Call to a Name (line 60):
            
            # Call to join(...): (line 60)
            # Processing the call arguments (line 60)
            # Getting the type of 'self' (line 60)
            self_23759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 39), 'self', False)
            # Obtaining the member 'install_dir' of a type (line 60)
            install_dir_23760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 39), self_23759, 'install_dir')
            # Getting the type of 'dir' (line 60)
            dir_23761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 57), 'dir', False)
            # Processing the call keyword arguments (line 60)
            kwargs_23762 = {}
            # Getting the type of 'os' (line 60)
            os_23756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 26), 'os', False)
            # Obtaining the member 'path' of a type (line 60)
            path_23757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 26), os_23756, 'path')
            # Obtaining the member 'join' of a type (line 60)
            join_23758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 26), path_23757, 'join')
            # Calling join(args, kwargs) (line 60)
            join_call_result_23763 = invoke(stypy.reporting.localization.Localization(__file__, 60, 26), join_23758, *[install_dir_23760, dir_23761], **kwargs_23762)
            
            # Assigning a type to the variable 'dir' (line 60)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'dir', join_call_result_23763)
            # SSA branch for the else part of an if statement (line 59)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'self' (line 61)
            self_23764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 'self')
            # Obtaining the member 'root' of a type (line 61)
            root_23765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 21), self_23764, 'root')
            # Testing the type of an if condition (line 61)
            if_condition_23766 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 21), root_23765)
            # Assigning a type to the variable 'if_condition_23766' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 'if_condition_23766', if_condition_23766)
            # SSA begins for if statement (line 61)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 62):
            
            # Assigning a Call to a Name (line 62):
            
            # Call to change_root(...): (line 62)
            # Processing the call arguments (line 62)
            # Getting the type of 'self' (line 62)
            self_23768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 38), 'self', False)
            # Obtaining the member 'root' of a type (line 62)
            root_23769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 38), self_23768, 'root')
            # Getting the type of 'dir' (line 62)
            dir_23770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 49), 'dir', False)
            # Processing the call keyword arguments (line 62)
            kwargs_23771 = {}
            # Getting the type of 'change_root' (line 62)
            change_root_23767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 26), 'change_root', False)
            # Calling change_root(args, kwargs) (line 62)
            change_root_call_result_23772 = invoke(stypy.reporting.localization.Localization(__file__, 62, 26), change_root_23767, *[root_23769, dir_23770], **kwargs_23771)
            
            # Assigning a type to the variable 'dir' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'dir', change_root_call_result_23772)
            # SSA join for if statement (line 61)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 59)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to mkpath(...): (line 63)
            # Processing the call arguments (line 63)
            # Getting the type of 'dir' (line 63)
            dir_23775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'dir', False)
            # Processing the call keyword arguments (line 63)
            kwargs_23776 = {}
            # Getting the type of 'self' (line 63)
            self_23773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'self', False)
            # Obtaining the member 'mkpath' of a type (line 63)
            mkpath_23774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 16), self_23773, 'mkpath')
            # Calling mkpath(args, kwargs) (line 63)
            mkpath_call_result_23777 = invoke(stypy.reporting.localization.Localization(__file__, 63, 16), mkpath_23774, *[dir_23775], **kwargs_23776)
            
            
            
            
            # Obtaining the type of the subscript
            int_23778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 21), 'int')
            # Getting the type of 'f' (line 65)
            f_23779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'f')
            # Obtaining the member '__getitem__' of a type (line 65)
            getitem___23780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), f_23779, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 65)
            subscript_call_result_23781 = invoke(stypy.reporting.localization.Localization(__file__, 65, 19), getitem___23780, int_23778)
            
            
            # Obtaining an instance of the builtin type 'list' (line 65)
            list_23782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 27), 'list')
            # Adding type elements to the builtin type 'list' instance (line 65)
            
            # Applying the binary operator '==' (line 65)
            result_eq_23783 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 19), '==', subscript_call_result_23781, list_23782)
            
            # Testing the type of an if condition (line 65)
            if_condition_23784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 16), result_eq_23783)
            # Assigning a type to the variable 'if_condition_23784' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'if_condition_23784', if_condition_23784)
            # SSA begins for if statement (line 65)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 69)
            # Processing the call arguments (line 69)
            # Getting the type of 'dir' (line 69)
            dir_23788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 41), 'dir', False)
            # Processing the call keyword arguments (line 69)
            kwargs_23789 = {}
            # Getting the type of 'self' (line 69)
            self_23785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'self', False)
            # Obtaining the member 'outfiles' of a type (line 69)
            outfiles_23786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 20), self_23785, 'outfiles')
            # Obtaining the member 'append' of a type (line 69)
            append_23787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 20), outfiles_23786, 'append')
            # Calling append(args, kwargs) (line 69)
            append_call_result_23790 = invoke(stypy.reporting.localization.Localization(__file__, 69, 20), append_23787, *[dir_23788], **kwargs_23789)
            
            # SSA branch for the else part of an if statement (line 65)
            module_type_store.open_ssa_branch('else')
            
            
            # Obtaining the type of the subscript
            int_23791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 34), 'int')
            # Getting the type of 'f' (line 72)
            f_23792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'f')
            # Obtaining the member '__getitem__' of a type (line 72)
            getitem___23793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 32), f_23792, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 72)
            subscript_call_result_23794 = invoke(stypy.reporting.localization.Localization(__file__, 72, 32), getitem___23793, int_23791)
            
            # Testing the type of a for loop iterable (line 72)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 72, 20), subscript_call_result_23794)
            # Getting the type of the for loop variable (line 72)
            for_loop_var_23795 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 72, 20), subscript_call_result_23794)
            # Assigning a type to the variable 'data' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'data', for_loop_var_23795)
            # SSA begins for a for statement (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 73):
            
            # Assigning a Call to a Name (line 73):
            
            # Call to convert_path(...): (line 73)
            # Processing the call arguments (line 73)
            # Getting the type of 'data' (line 73)
            data_23797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'data', False)
            # Processing the call keyword arguments (line 73)
            kwargs_23798 = {}
            # Getting the type of 'convert_path' (line 73)
            convert_path_23796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 31), 'convert_path', False)
            # Calling convert_path(args, kwargs) (line 73)
            convert_path_call_result_23799 = invoke(stypy.reporting.localization.Localization(__file__, 73, 31), convert_path_23796, *[data_23797], **kwargs_23798)
            
            # Assigning a type to the variable 'data' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 24), 'data', convert_path_call_result_23799)
            
            # Assigning a Call to a Tuple (line 74):
            
            # Assigning a Subscript to a Name (line 74):
            
            # Obtaining the type of the subscript
            int_23800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 24), 'int')
            
            # Call to copy_file(...): (line 74)
            # Processing the call arguments (line 74)
            # Getting the type of 'data' (line 74)
            data_23803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 50), 'data', False)
            # Getting the type of 'dir' (line 74)
            dir_23804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 56), 'dir', False)
            # Processing the call keyword arguments (line 74)
            kwargs_23805 = {}
            # Getting the type of 'self' (line 74)
            self_23801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 35), 'self', False)
            # Obtaining the member 'copy_file' of a type (line 74)
            copy_file_23802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 35), self_23801, 'copy_file')
            # Calling copy_file(args, kwargs) (line 74)
            copy_file_call_result_23806 = invoke(stypy.reporting.localization.Localization(__file__, 74, 35), copy_file_23802, *[data_23803, dir_23804], **kwargs_23805)
            
            # Obtaining the member '__getitem__' of a type (line 74)
            getitem___23807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 24), copy_file_call_result_23806, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 74)
            subscript_call_result_23808 = invoke(stypy.reporting.localization.Localization(__file__, 74, 24), getitem___23807, int_23800)
            
            # Assigning a type to the variable 'tuple_var_assignment_23644' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 24), 'tuple_var_assignment_23644', subscript_call_result_23808)
            
            # Assigning a Subscript to a Name (line 74):
            
            # Obtaining the type of the subscript
            int_23809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 24), 'int')
            
            # Call to copy_file(...): (line 74)
            # Processing the call arguments (line 74)
            # Getting the type of 'data' (line 74)
            data_23812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 50), 'data', False)
            # Getting the type of 'dir' (line 74)
            dir_23813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 56), 'dir', False)
            # Processing the call keyword arguments (line 74)
            kwargs_23814 = {}
            # Getting the type of 'self' (line 74)
            self_23810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 35), 'self', False)
            # Obtaining the member 'copy_file' of a type (line 74)
            copy_file_23811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 35), self_23810, 'copy_file')
            # Calling copy_file(args, kwargs) (line 74)
            copy_file_call_result_23815 = invoke(stypy.reporting.localization.Localization(__file__, 74, 35), copy_file_23811, *[data_23812, dir_23813], **kwargs_23814)
            
            # Obtaining the member '__getitem__' of a type (line 74)
            getitem___23816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 24), copy_file_call_result_23815, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 74)
            subscript_call_result_23817 = invoke(stypy.reporting.localization.Localization(__file__, 74, 24), getitem___23816, int_23809)
            
            # Assigning a type to the variable 'tuple_var_assignment_23645' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 24), 'tuple_var_assignment_23645', subscript_call_result_23817)
            
            # Assigning a Name to a Name (line 74):
            # Getting the type of 'tuple_var_assignment_23644' (line 74)
            tuple_var_assignment_23644_23818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 24), 'tuple_var_assignment_23644')
            # Assigning a type to the variable 'out' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'out', tuple_var_assignment_23644_23818)
            
            # Assigning a Name to a Name (line 74):
            # Getting the type of 'tuple_var_assignment_23645' (line 74)
            tuple_var_assignment_23645_23819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 24), 'tuple_var_assignment_23645')
            # Assigning a type to the variable '_' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 30), '_', tuple_var_assignment_23645_23819)
            
            # Call to append(...): (line 75)
            # Processing the call arguments (line 75)
            # Getting the type of 'out' (line 75)
            out_23823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 45), 'out', False)
            # Processing the call keyword arguments (line 75)
            kwargs_23824 = {}
            # Getting the type of 'self' (line 75)
            self_23820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 24), 'self', False)
            # Obtaining the member 'outfiles' of a type (line 75)
            outfiles_23821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 24), self_23820, 'outfiles')
            # Obtaining the member 'append' of a type (line 75)
            append_23822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 24), outfiles_23821, 'append')
            # Calling append(args, kwargs) (line 75)
            append_call_result_23825 = invoke(stypy.reporting.localization.Localization(__file__, 75, 24), append_23822, *[out_23823], **kwargs_23824)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 65)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_23694 and more_types_in_union_23695):
                # SSA join for if statement (line 47)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_23826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23826)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_23826


    @norecursion
    def get_inputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_inputs'
        module_type_store = module_type_store.open_function_context('get_inputs', 77, 4, False)
        # Assigning a type to the variable 'self' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_data.get_inputs.__dict__.__setitem__('stypy_localization', localization)
        install_data.get_inputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_data.get_inputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_data.get_inputs.__dict__.__setitem__('stypy_function_name', 'install_data.get_inputs')
        install_data.get_inputs.__dict__.__setitem__('stypy_param_names_list', [])
        install_data.get_inputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_data.get_inputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_data.get_inputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_data.get_inputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_data.get_inputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_data.get_inputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_data.get_inputs', [], None, None, defaults, varargs, kwargs)

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

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 78)
        self_23827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'self')
        # Obtaining the member 'data_files' of a type (line 78)
        data_files_23828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 15), self_23827, 'data_files')
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_23829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        
        # Applying the binary operator 'or' (line 78)
        result_or_keyword_23830 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 15), 'or', data_files_23828, list_23829)
        
        # Assigning a type to the variable 'stypy_return_type' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'stypy_return_type', result_or_keyword_23830)
        
        # ################# End of 'get_inputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_inputs' in the type store
        # Getting the type of 'stypy_return_type' (line 77)
        stypy_return_type_23831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23831)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_inputs'
        return stypy_return_type_23831


    @norecursion
    def get_outputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_outputs'
        module_type_store = module_type_store.open_function_context('get_outputs', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_data.get_outputs.__dict__.__setitem__('stypy_localization', localization)
        install_data.get_outputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_data.get_outputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_data.get_outputs.__dict__.__setitem__('stypy_function_name', 'install_data.get_outputs')
        install_data.get_outputs.__dict__.__setitem__('stypy_param_names_list', [])
        install_data.get_outputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_data.get_outputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_data.get_outputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_data.get_outputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_data.get_outputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_data.get_outputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_data.get_outputs', [], None, None, defaults, varargs, kwargs)

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

        # Getting the type of 'self' (line 81)
        self_23832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'self')
        # Obtaining the member 'outfiles' of a type (line 81)
        outfiles_23833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 15), self_23832, 'outfiles')
        # Assigning a type to the variable 'stypy_return_type' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'stypy_return_type', outfiles_23833)
        
        # ################# End of 'get_outputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_outputs' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_23834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23834)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_outputs'
        return stypy_return_type_23834


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 14, 0, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_data.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'install_data' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'install_data', install_data)

# Assigning a Str to a Name (line 16):
str_23835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 18), 'str', 'install data files')
# Getting the type of 'install_data'
install_data_23836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_data')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_data_23836, 'description', str_23835)

# Assigning a List to a Name (line 18):

# Obtaining an instance of the builtin type 'list' (line 18)
list_23837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)

# Obtaining an instance of the builtin type 'tuple' (line 19)
tuple_23838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 19)
# Adding element type (line 19)
str_23839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 9), 'str', 'install-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 9), tuple_23838, str_23839)
# Adding element type (line 19)
str_23840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 9), tuple_23838, str_23840)
# Adding element type (line 19)
str_23841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 9), 'str', 'base directory for installing data files (default: installation base dir)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 9), tuple_23838, str_23841)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 19), list_23837, tuple_23838)
# Adding element type (line 18)

# Obtaining an instance of the builtin type 'tuple' (line 22)
tuple_23842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 22)
# Adding element type (line 22)
str_23843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'str', 'root=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_23842, str_23843)
# Adding element type (line 22)
# Getting the type of 'None' (line 22)
None_23844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 18), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_23842, None_23844)
# Adding element type (line 22)
str_23845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'str', 'install everything relative to this alternate root directory')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_23842, str_23845)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 19), list_23837, tuple_23842)
# Adding element type (line 18)

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_23846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
str_23847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_23846, str_23847)
# Adding element type (line 24)
str_23848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 18), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_23846, str_23848)
# Adding element type (line 24)
str_23849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'str', 'force installation (overwrite existing files)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_23846, str_23849)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 19), list_23837, tuple_23846)

# Getting the type of 'install_data'
install_data_23850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_data')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_data_23850, 'user_options', list_23837)

# Assigning a List to a Name (line 27):

# Obtaining an instance of the builtin type 'list' (line 27)
list_23851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)
str_23852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 22), list_23851, str_23852)

# Getting the type of 'install_data'
install_data_23853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_data')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_data_23853, 'boolean_options', list_23851)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
