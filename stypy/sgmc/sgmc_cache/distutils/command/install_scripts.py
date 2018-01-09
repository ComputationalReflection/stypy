
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.install_scripts
2: 
3: Implements the Distutils 'install_scripts' command, for installing
4: Python scripts.'''
5: 
6: # contributed by Bastian Kleineidam
7: 
8: __revision__ = "$Id$"
9: 
10: import os
11: from distutils.core import Command
12: from distutils import log
13: from stat import ST_MODE
14: 
15: class install_scripts (Command):
16: 
17:     description = "install scripts (Python or otherwise)"
18: 
19:     user_options = [
20:         ('install-dir=', 'd', "directory to install scripts to"),
21:         ('build-dir=','b', "build directory (where to install from)"),
22:         ('force', 'f', "force installation (overwrite existing files)"),
23:         ('skip-build', None, "skip the build steps"),
24:     ]
25: 
26:     boolean_options = ['force', 'skip-build']
27: 
28: 
29:     def initialize_options (self):
30:         self.install_dir = None
31:         self.force = 0
32:         self.build_dir = None
33:         self.skip_build = None
34: 
35:     def finalize_options (self):
36:         self.set_undefined_options('build', ('build_scripts', 'build_dir'))
37:         self.set_undefined_options('install',
38:                                    ('install_scripts', 'install_dir'),
39:                                    ('force', 'force'),
40:                                    ('skip_build', 'skip_build'),
41:                                   )
42: 
43:     def run (self):
44:         if not self.skip_build:
45:             self.run_command('build_scripts')
46:         self.outfiles = self.copy_tree(self.build_dir, self.install_dir)
47:         if os.name == 'posix':
48:             # Set the executable bits (owner, group, and world) on
49:             # all the scripts we just installed.
50:             for file in self.get_outputs():
51:                 if self.dry_run:
52:                     log.info("changing mode of %s", file)
53:                 else:
54:                     mode = ((os.stat(file)[ST_MODE]) | 0555) & 07777
55:                     log.info("changing mode of %s to %o", file, mode)
56:                     os.chmod(file, mode)
57: 
58:     def get_inputs (self):
59:         return self.distribution.scripts or []
60: 
61:     def get_outputs(self):
62:         return self.outfiles or []
63: 
64: # class install_scripts
65: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_24586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', "distutils.command.install_scripts\n\nImplements the Distutils 'install_scripts' command, for installing\nPython scripts.")

# Assigning a Str to a Name (line 8):
str_24587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__revision__', str_24587)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import os' statement (line 10)
import os

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.core import Command' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_24588 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core')

if (type(import_24588) is not StypyTypeError):

    if (import_24588 != 'pyd_module'):
        __import__(import_24588)
        sys_modules_24589 = sys.modules[import_24588]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core', sys_modules_24589.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_24589, sys_modules_24589.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core', import_24588)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils import log' statement (line 12)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils', None, module_type_store, ['log'], [log])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from stat import ST_MODE' statement (line 13)
try:
    from stat import ST_MODE

except:
    ST_MODE = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'stat', None, module_type_store, ['ST_MODE'], [ST_MODE])

# Declaration of the 'install_scripts' class
# Getting the type of 'Command' (line 15)
Command_24590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 23), 'Command')

class install_scripts(Command_24590, ):

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
        install_scripts.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        install_scripts.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_scripts.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_scripts.initialize_options.__dict__.__setitem__('stypy_function_name', 'install_scripts.initialize_options')
        install_scripts.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install_scripts.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_scripts.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_scripts.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_scripts.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_scripts.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_scripts.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_scripts.initialize_options', [], None, None, defaults, varargs, kwargs)

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
        # Getting the type of 'None' (line 30)
        None_24591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 27), 'None')
        # Getting the type of 'self' (line 30)
        self_24592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'install_dir' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_24592, 'install_dir', None_24591)
        
        # Assigning a Num to a Attribute (line 31):
        int_24593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 21), 'int')
        # Getting the type of 'self' (line 31)
        self_24594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'force' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_24594, 'force', int_24593)
        
        # Assigning a Name to a Attribute (line 32):
        # Getting the type of 'None' (line 32)
        None_24595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'None')
        # Getting the type of 'self' (line 32)
        self_24596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member 'build_dir' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_24596, 'build_dir', None_24595)
        
        # Assigning a Name to a Attribute (line 33):
        # Getting the type of 'None' (line 33)
        None_24597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 26), 'None')
        # Getting the type of 'self' (line 33)
        self_24598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'skip_build' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_24598, 'skip_build', None_24597)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_24599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24599)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_24599


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_scripts.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        install_scripts.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_scripts.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_scripts.finalize_options.__dict__.__setitem__('stypy_function_name', 'install_scripts.finalize_options')
        install_scripts.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install_scripts.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_scripts.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_scripts.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_scripts.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_scripts.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_scripts.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_scripts.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_undefined_options(...): (line 36)
        # Processing the call arguments (line 36)
        str_24602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 35), 'str', 'build')
        
        # Obtaining an instance of the builtin type 'tuple' (line 36)
        tuple_24603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 45), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 36)
        # Adding element type (line 36)
        str_24604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 45), 'str', 'build_scripts')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 45), tuple_24603, str_24604)
        # Adding element type (line 36)
        str_24605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 62), 'str', 'build_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 45), tuple_24603, str_24605)
        
        # Processing the call keyword arguments (line 36)
        kwargs_24606 = {}
        # Getting the type of 'self' (line 36)
        self_24600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 36)
        set_undefined_options_24601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_24600, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 36)
        set_undefined_options_call_result_24607 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), set_undefined_options_24601, *[str_24602, tuple_24603], **kwargs_24606)
        
        
        # Call to set_undefined_options(...): (line 37)
        # Processing the call arguments (line 37)
        str_24610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 35), 'str', 'install')
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_24611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        str_24612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 36), 'str', 'install_scripts')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 36), tuple_24611, str_24612)
        # Adding element type (line 38)
        str_24613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 55), 'str', 'install_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 36), tuple_24611, str_24613)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_24614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        str_24615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 36), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 36), tuple_24614, str_24615)
        # Adding element type (line 39)
        str_24616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 45), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 36), tuple_24614, str_24616)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 40)
        tuple_24617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 40)
        # Adding element type (line 40)
        str_24618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 36), 'str', 'skip_build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 36), tuple_24617, str_24618)
        # Adding element type (line 40)
        str_24619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 50), 'str', 'skip_build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 36), tuple_24617, str_24619)
        
        # Processing the call keyword arguments (line 37)
        kwargs_24620 = {}
        # Getting the type of 'self' (line 37)
        self_24608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 37)
        set_undefined_options_24609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_24608, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 37)
        set_undefined_options_call_result_24621 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), set_undefined_options_24609, *[str_24610, tuple_24611, tuple_24614, tuple_24617], **kwargs_24620)
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_24622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24622)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_24622


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_scripts.run.__dict__.__setitem__('stypy_localization', localization)
        install_scripts.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_scripts.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_scripts.run.__dict__.__setitem__('stypy_function_name', 'install_scripts.run')
        install_scripts.run.__dict__.__setitem__('stypy_param_names_list', [])
        install_scripts.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_scripts.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_scripts.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_scripts.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_scripts.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_scripts.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_scripts.run', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 44)
        self_24623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'self')
        # Obtaining the member 'skip_build' of a type (line 44)
        skip_build_24624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 15), self_24623, 'skip_build')
        # Applying the 'not' unary operator (line 44)
        result_not__24625 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), 'not', skip_build_24624)
        
        # Testing the type of an if condition (line 44)
        if_condition_24626 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 8), result_not__24625)
        # Assigning a type to the variable 'if_condition_24626' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'if_condition_24626', if_condition_24626)
        # SSA begins for if statement (line 44)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to run_command(...): (line 45)
        # Processing the call arguments (line 45)
        str_24629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 29), 'str', 'build_scripts')
        # Processing the call keyword arguments (line 45)
        kwargs_24630 = {}
        # Getting the type of 'self' (line 45)
        self_24627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'self', False)
        # Obtaining the member 'run_command' of a type (line 45)
        run_command_24628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), self_24627, 'run_command')
        # Calling run_command(args, kwargs) (line 45)
        run_command_call_result_24631 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), run_command_24628, *[str_24629], **kwargs_24630)
        
        # SSA join for if statement (line 44)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 46):
        
        # Call to copy_tree(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'self' (line 46)
        self_24634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 39), 'self', False)
        # Obtaining the member 'build_dir' of a type (line 46)
        build_dir_24635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 39), self_24634, 'build_dir')
        # Getting the type of 'self' (line 46)
        self_24636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 55), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 46)
        install_dir_24637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 55), self_24636, 'install_dir')
        # Processing the call keyword arguments (line 46)
        kwargs_24638 = {}
        # Getting the type of 'self' (line 46)
        self_24632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 24), 'self', False)
        # Obtaining the member 'copy_tree' of a type (line 46)
        copy_tree_24633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 24), self_24632, 'copy_tree')
        # Calling copy_tree(args, kwargs) (line 46)
        copy_tree_call_result_24639 = invoke(stypy.reporting.localization.Localization(__file__, 46, 24), copy_tree_24633, *[build_dir_24635, install_dir_24637], **kwargs_24638)
        
        # Getting the type of 'self' (line 46)
        self_24640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self')
        # Setting the type of the member 'outfiles' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_24640, 'outfiles', copy_tree_call_result_24639)
        
        
        # Getting the type of 'os' (line 47)
        os_24641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'os')
        # Obtaining the member 'name' of a type (line 47)
        name_24642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), os_24641, 'name')
        str_24643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 22), 'str', 'posix')
        # Applying the binary operator '==' (line 47)
        result_eq_24644 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), '==', name_24642, str_24643)
        
        # Testing the type of an if condition (line 47)
        if_condition_24645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 8), result_eq_24644)
        # Assigning a type to the variable 'if_condition_24645' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'if_condition_24645', if_condition_24645)
        # SSA begins for if statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to get_outputs(...): (line 50)
        # Processing the call keyword arguments (line 50)
        kwargs_24648 = {}
        # Getting the type of 'self' (line 50)
        self_24646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'self', False)
        # Obtaining the member 'get_outputs' of a type (line 50)
        get_outputs_24647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 24), self_24646, 'get_outputs')
        # Calling get_outputs(args, kwargs) (line 50)
        get_outputs_call_result_24649 = invoke(stypy.reporting.localization.Localization(__file__, 50, 24), get_outputs_24647, *[], **kwargs_24648)
        
        # Testing the type of a for loop iterable (line 50)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 12), get_outputs_call_result_24649)
        # Getting the type of the for loop variable (line 50)
        for_loop_var_24650 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 12), get_outputs_call_result_24649)
        # Assigning a type to the variable 'file' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'file', for_loop_var_24650)
        # SSA begins for a for statement (line 50)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'self' (line 51)
        self_24651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'self')
        # Obtaining the member 'dry_run' of a type (line 51)
        dry_run_24652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 19), self_24651, 'dry_run')
        # Testing the type of an if condition (line 51)
        if_condition_24653 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 16), dry_run_24652)
        # Assigning a type to the variable 'if_condition_24653' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'if_condition_24653', if_condition_24653)
        # SSA begins for if statement (line 51)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 52)
        # Processing the call arguments (line 52)
        str_24656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 29), 'str', 'changing mode of %s')
        # Getting the type of 'file' (line 52)
        file_24657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 52), 'file', False)
        # Processing the call keyword arguments (line 52)
        kwargs_24658 = {}
        # Getting the type of 'log' (line 52)
        log_24654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'log', False)
        # Obtaining the member 'info' of a type (line 52)
        info_24655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 20), log_24654, 'info')
        # Calling info(args, kwargs) (line 52)
        info_call_result_24659 = invoke(stypy.reporting.localization.Localization(__file__, 52, 20), info_24655, *[str_24656, file_24657], **kwargs_24658)
        
        # SSA branch for the else part of an if statement (line 51)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Name (line 54):
        
        # Obtaining the type of the subscript
        # Getting the type of 'ST_MODE' (line 54)
        ST_MODE_24660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 43), 'ST_MODE')
        
        # Call to stat(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'file' (line 54)
        file_24663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 37), 'file', False)
        # Processing the call keyword arguments (line 54)
        kwargs_24664 = {}
        # Getting the type of 'os' (line 54)
        os_24661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 29), 'os', False)
        # Obtaining the member 'stat' of a type (line 54)
        stat_24662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 29), os_24661, 'stat')
        # Calling stat(args, kwargs) (line 54)
        stat_call_result_24665 = invoke(stypy.reporting.localization.Localization(__file__, 54, 29), stat_24662, *[file_24663], **kwargs_24664)
        
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___24666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 29), stat_call_result_24665, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_24667 = invoke(stypy.reporting.localization.Localization(__file__, 54, 29), getitem___24666, ST_MODE_24660)
        
        int_24668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 55), 'int')
        # Applying the binary operator '|' (line 54)
        result_or__24669 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 28), '|', subscript_call_result_24667, int_24668)
        
        int_24670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 63), 'int')
        # Applying the binary operator '&' (line 54)
        result_and__24671 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 27), '&', result_or__24669, int_24670)
        
        # Assigning a type to the variable 'mode' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'mode', result_and__24671)
        
        # Call to info(...): (line 55)
        # Processing the call arguments (line 55)
        str_24674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 29), 'str', 'changing mode of %s to %o')
        # Getting the type of 'file' (line 55)
        file_24675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 58), 'file', False)
        # Getting the type of 'mode' (line 55)
        mode_24676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 64), 'mode', False)
        # Processing the call keyword arguments (line 55)
        kwargs_24677 = {}
        # Getting the type of 'log' (line 55)
        log_24672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 20), 'log', False)
        # Obtaining the member 'info' of a type (line 55)
        info_24673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 20), log_24672, 'info')
        # Calling info(args, kwargs) (line 55)
        info_call_result_24678 = invoke(stypy.reporting.localization.Localization(__file__, 55, 20), info_24673, *[str_24674, file_24675, mode_24676], **kwargs_24677)
        
        
        # Call to chmod(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'file' (line 56)
        file_24681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 29), 'file', False)
        # Getting the type of 'mode' (line 56)
        mode_24682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 35), 'mode', False)
        # Processing the call keyword arguments (line 56)
        kwargs_24683 = {}
        # Getting the type of 'os' (line 56)
        os_24679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'os', False)
        # Obtaining the member 'chmod' of a type (line 56)
        chmod_24680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 20), os_24679, 'chmod')
        # Calling chmod(args, kwargs) (line 56)
        chmod_call_result_24684 = invoke(stypy.reporting.localization.Localization(__file__, 56, 20), chmod_24680, *[file_24681, mode_24682], **kwargs_24683)
        
        # SSA join for if statement (line 51)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 47)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_24685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24685)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_24685


    @norecursion
    def get_inputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_inputs'
        module_type_store = module_type_store.open_function_context('get_inputs', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_scripts.get_inputs.__dict__.__setitem__('stypy_localization', localization)
        install_scripts.get_inputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_scripts.get_inputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_scripts.get_inputs.__dict__.__setitem__('stypy_function_name', 'install_scripts.get_inputs')
        install_scripts.get_inputs.__dict__.__setitem__('stypy_param_names_list', [])
        install_scripts.get_inputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_scripts.get_inputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_scripts.get_inputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_scripts.get_inputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_scripts.get_inputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_scripts.get_inputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_scripts.get_inputs', [], None, None, defaults, varargs, kwargs)

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
        # Getting the type of 'self' (line 59)
        self_24686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'self')
        # Obtaining the member 'distribution' of a type (line 59)
        distribution_24687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 15), self_24686, 'distribution')
        # Obtaining the member 'scripts' of a type (line 59)
        scripts_24688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 15), distribution_24687, 'scripts')
        
        # Obtaining an instance of the builtin type 'list' (line 59)
        list_24689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 59)
        
        # Applying the binary operator 'or' (line 59)
        result_or_keyword_24690 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 15), 'or', scripts_24688, list_24689)
        
        # Assigning a type to the variable 'stypy_return_type' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type', result_or_keyword_24690)
        
        # ################# End of 'get_inputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_inputs' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_24691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24691)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_inputs'
        return stypy_return_type_24691


    @norecursion
    def get_outputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_outputs'
        module_type_store = module_type_store.open_function_context('get_outputs', 61, 4, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_scripts.get_outputs.__dict__.__setitem__('stypy_localization', localization)
        install_scripts.get_outputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_scripts.get_outputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_scripts.get_outputs.__dict__.__setitem__('stypy_function_name', 'install_scripts.get_outputs')
        install_scripts.get_outputs.__dict__.__setitem__('stypy_param_names_list', [])
        install_scripts.get_outputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_scripts.get_outputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_scripts.get_outputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_scripts.get_outputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_scripts.get_outputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_scripts.get_outputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_scripts.get_outputs', [], None, None, defaults, varargs, kwargs)

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

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 62)
        self_24692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'self')
        # Obtaining the member 'outfiles' of a type (line 62)
        outfiles_24693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 15), self_24692, 'outfiles')
        
        # Obtaining an instance of the builtin type 'list' (line 62)
        list_24694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 62)
        
        # Applying the binary operator 'or' (line 62)
        result_or_keyword_24695 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 15), 'or', outfiles_24693, list_24694)
        
        # Assigning a type to the variable 'stypy_return_type' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'stypy_return_type', result_or_keyword_24695)
        
        # ################# End of 'get_outputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_outputs' in the type store
        # Getting the type of 'stypy_return_type' (line 61)
        stypy_return_type_24696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24696)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_outputs'
        return stypy_return_type_24696


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 15, 0, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_scripts.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'install_scripts' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'install_scripts', install_scripts)

# Assigning a Str to a Name (line 17):
str_24697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'str', 'install scripts (Python or otherwise)')
# Getting the type of 'install_scripts'
install_scripts_24698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_scripts')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_scripts_24698, 'description', str_24697)

# Assigning a List to a Name (line 19):

# Obtaining an instance of the builtin type 'list' (line 19)
list_24699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)

# Obtaining an instance of the builtin type 'tuple' (line 20)
tuple_24700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 20)
# Adding element type (line 20)
str_24701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 9), 'str', 'install-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 9), tuple_24700, str_24701)
# Adding element type (line 20)
str_24702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 9), tuple_24700, str_24702)
# Adding element type (line 20)
str_24703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 30), 'str', 'directory to install scripts to')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 9), tuple_24700, str_24703)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 19), list_24699, tuple_24700)
# Adding element type (line 19)

# Obtaining an instance of the builtin type 'tuple' (line 21)
tuple_24704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 21)
# Adding element type (line 21)
str_24705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 9), 'str', 'build-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 9), tuple_24704, str_24705)
# Adding element type (line 21)
str_24706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 22), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 9), tuple_24704, str_24706)
# Adding element type (line 21)
str_24707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 27), 'str', 'build directory (where to install from)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 9), tuple_24704, str_24707)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 19), list_24699, tuple_24704)
# Adding element type (line 19)

# Obtaining an instance of the builtin type 'tuple' (line 22)
tuple_24708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 22)
# Adding element type (line 22)
str_24709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_24708, str_24709)
# Adding element type (line 22)
str_24710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_24708, str_24710)
# Adding element type (line 22)
str_24711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'str', 'force installation (overwrite existing files)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_24708, str_24711)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 19), list_24699, tuple_24708)
# Adding element type (line 19)

# Obtaining an instance of the builtin type 'tuple' (line 23)
tuple_24712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 23)
# Adding element type (line 23)
str_24713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'str', 'skip-build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_24712, str_24713)
# Adding element type (line 23)
# Getting the type of 'None' (line 23)
None_24714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_24712, None_24714)
# Adding element type (line 23)
str_24715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 29), 'str', 'skip the build steps')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 9), tuple_24712, str_24715)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 19), list_24699, tuple_24712)

# Getting the type of 'install_scripts'
install_scripts_24716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_scripts')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_scripts_24716, 'user_options', list_24699)

# Assigning a List to a Name (line 26):

# Obtaining an instance of the builtin type 'list' (line 26)
list_24717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
str_24718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 23), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 22), list_24717, str_24718)
# Adding element type (line 26)
str_24719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 32), 'str', 'skip-build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 22), list_24717, str_24719)

# Getting the type of 'install_scripts'
install_scripts_24720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_scripts')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_scripts_24720, 'boolean_options', list_24717)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
