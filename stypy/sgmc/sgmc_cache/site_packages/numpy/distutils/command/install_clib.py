
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import os
4: from distutils.core import Command
5: from distutils.ccompiler import new_compiler
6: from numpy.distutils.misc_util import get_cmd
7: 
8: class install_clib(Command):
9:     description = "Command to install installable C libraries"
10: 
11:     user_options = []
12: 
13:     def initialize_options(self):
14:         self.install_dir = None
15:         self.outfiles = []
16: 
17:     def finalize_options(self):
18:         self.set_undefined_options('install', ('install_lib', 'install_dir'))
19: 
20:     def run (self):
21:         build_clib_cmd = get_cmd("build_clib")
22:         build_dir = build_clib_cmd.build_clib
23: 
24:         # We need the compiler to get the library name -> filename association
25:         if not build_clib_cmd.compiler:
26:             compiler = new_compiler(compiler=None)
27:             compiler.customize(self.distribution)
28:         else:
29:             compiler = build_clib_cmd.compiler
30: 
31:         for l in self.distribution.installed_libraries:
32:             target_dir = os.path.join(self.install_dir, l.target_dir)
33:             name = compiler.library_filename(l.name)
34:             source = os.path.join(build_dir, name)
35:             self.mkpath(target_dir)
36:             self.outfiles.append(self.copy_file(source, target_dir)[0])
37: 
38:     def get_outputs(self):
39:         return self.outfiles
40: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from distutils.core import Command' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_59485 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.core')

if (type(import_59485) is not StypyTypeError):

    if (import_59485 != 'pyd_module'):
        __import__(import_59485)
        sys_modules_59486 = sys.modules[import_59485]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.core', sys_modules_59486.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_59486, sys_modules_59486.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.core', import_59485)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from distutils.ccompiler import new_compiler' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_59487 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.ccompiler')

if (type(import_59487) is not StypyTypeError):

    if (import_59487 != 'pyd_module'):
        __import__(import_59487)
        sys_modules_59488 = sys.modules[import_59487]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.ccompiler', sys_modules_59488.module_type_store, module_type_store, ['new_compiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_59488, sys_modules_59488.module_type_store, module_type_store)
    else:
        from distutils.ccompiler import new_compiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.ccompiler', None, module_type_store, ['new_compiler'], [new_compiler])

else:
    # Assigning a type to the variable 'distutils.ccompiler' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.ccompiler', import_59487)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.distutils.misc_util import get_cmd' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_59489 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.distutils.misc_util')

if (type(import_59489) is not StypyTypeError):

    if (import_59489 != 'pyd_module'):
        __import__(import_59489)
        sys_modules_59490 = sys.modules[import_59489]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.distutils.misc_util', sys_modules_59490.module_type_store, module_type_store, ['get_cmd'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_59490, sys_modules_59490.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import get_cmd

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.distutils.misc_util', None, module_type_store, ['get_cmd'], [get_cmd])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.distutils.misc_util', import_59489)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

# Declaration of the 'install_clib' class
# Getting the type of 'Command' (line 8)
Command_59491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 19), 'Command')

class install_clib(Command_59491, ):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_clib.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        install_clib.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_clib.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_clib.initialize_options.__dict__.__setitem__('stypy_function_name', 'install_clib.initialize_options')
        install_clib.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install_clib.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_clib.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_clib.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_clib.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_clib.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_clib.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_clib.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 14):
        # Getting the type of 'None' (line 14)
        None_59492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 27), 'None')
        # Getting the type of 'self' (line 14)
        self_59493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self')
        # Setting the type of the member 'install_dir' of a type (line 14)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), self_59493, 'install_dir', None_59492)
        
        # Assigning a List to a Attribute (line 15):
        
        # Obtaining an instance of the builtin type 'list' (line 15)
        list_59494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 15)
        
        # Getting the type of 'self' (line 15)
        self_59495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self')
        # Setting the type of the member 'outfiles' of a type (line 15)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_59495, 'outfiles', list_59494)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_59496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59496)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_59496


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_clib.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        install_clib.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_clib.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_clib.finalize_options.__dict__.__setitem__('stypy_function_name', 'install_clib.finalize_options')
        install_clib.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install_clib.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_clib.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_clib.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_clib.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_clib.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_clib.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_clib.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_undefined_options(...): (line 18)
        # Processing the call arguments (line 18)
        str_59499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'str', 'install')
        
        # Obtaining an instance of the builtin type 'tuple' (line 18)
        tuple_59500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 47), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 18)
        # Adding element type (line 18)
        str_59501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 47), 'str', 'install_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 47), tuple_59500, str_59501)
        # Adding element type (line 18)
        str_59502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 62), 'str', 'install_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 47), tuple_59500, str_59502)
        
        # Processing the call keyword arguments (line 18)
        kwargs_59503 = {}
        # Getting the type of 'self' (line 18)
        self_59497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 18)
        set_undefined_options_59498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), self_59497, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 18)
        set_undefined_options_call_result_59504 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), set_undefined_options_59498, *[str_59499, tuple_59500], **kwargs_59503)
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_59505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59505)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_59505


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_clib.run.__dict__.__setitem__('stypy_localization', localization)
        install_clib.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_clib.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_clib.run.__dict__.__setitem__('stypy_function_name', 'install_clib.run')
        install_clib.run.__dict__.__setitem__('stypy_param_names_list', [])
        install_clib.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_clib.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_clib.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_clib.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_clib.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_clib.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_clib.run', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 21):
        
        # Call to get_cmd(...): (line 21)
        # Processing the call arguments (line 21)
        str_59507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 33), 'str', 'build_clib')
        # Processing the call keyword arguments (line 21)
        kwargs_59508 = {}
        # Getting the type of 'get_cmd' (line 21)
        get_cmd_59506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 25), 'get_cmd', False)
        # Calling get_cmd(args, kwargs) (line 21)
        get_cmd_call_result_59509 = invoke(stypy.reporting.localization.Localization(__file__, 21, 25), get_cmd_59506, *[str_59507], **kwargs_59508)
        
        # Assigning a type to the variable 'build_clib_cmd' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'build_clib_cmd', get_cmd_call_result_59509)
        
        # Assigning a Attribute to a Name (line 22):
        # Getting the type of 'build_clib_cmd' (line 22)
        build_clib_cmd_59510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'build_clib_cmd')
        # Obtaining the member 'build_clib' of a type (line 22)
        build_clib_59511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 20), build_clib_cmd_59510, 'build_clib')
        # Assigning a type to the variable 'build_dir' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'build_dir', build_clib_59511)
        
        
        # Getting the type of 'build_clib_cmd' (line 25)
        build_clib_cmd_59512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'build_clib_cmd')
        # Obtaining the member 'compiler' of a type (line 25)
        compiler_59513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 15), build_clib_cmd_59512, 'compiler')
        # Applying the 'not' unary operator (line 25)
        result_not__59514 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 11), 'not', compiler_59513)
        
        # Testing the type of an if condition (line 25)
        if_condition_59515 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 8), result_not__59514)
        # Assigning a type to the variable 'if_condition_59515' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'if_condition_59515', if_condition_59515)
        # SSA begins for if statement (line 25)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 26):
        
        # Call to new_compiler(...): (line 26)
        # Processing the call keyword arguments (line 26)
        # Getting the type of 'None' (line 26)
        None_59517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 45), 'None', False)
        keyword_59518 = None_59517
        kwargs_59519 = {'compiler': keyword_59518}
        # Getting the type of 'new_compiler' (line 26)
        new_compiler_59516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'new_compiler', False)
        # Calling new_compiler(args, kwargs) (line 26)
        new_compiler_call_result_59520 = invoke(stypy.reporting.localization.Localization(__file__, 26, 23), new_compiler_59516, *[], **kwargs_59519)
        
        # Assigning a type to the variable 'compiler' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'compiler', new_compiler_call_result_59520)
        
        # Call to customize(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'self' (line 27)
        self_59523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 31), 'self', False)
        # Obtaining the member 'distribution' of a type (line 27)
        distribution_59524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 31), self_59523, 'distribution')
        # Processing the call keyword arguments (line 27)
        kwargs_59525 = {}
        # Getting the type of 'compiler' (line 27)
        compiler_59521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'compiler', False)
        # Obtaining the member 'customize' of a type (line 27)
        customize_59522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), compiler_59521, 'customize')
        # Calling customize(args, kwargs) (line 27)
        customize_call_result_59526 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), customize_59522, *[distribution_59524], **kwargs_59525)
        
        # SSA branch for the else part of an if statement (line 25)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 29):
        # Getting the type of 'build_clib_cmd' (line 29)
        build_clib_cmd_59527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'build_clib_cmd')
        # Obtaining the member 'compiler' of a type (line 29)
        compiler_59528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 23), build_clib_cmd_59527, 'compiler')
        # Assigning a type to the variable 'compiler' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'compiler', compiler_59528)
        # SSA join for if statement (line 25)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 31)
        self_59529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'self')
        # Obtaining the member 'distribution' of a type (line 31)
        distribution_59530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 17), self_59529, 'distribution')
        # Obtaining the member 'installed_libraries' of a type (line 31)
        installed_libraries_59531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 17), distribution_59530, 'installed_libraries')
        # Testing the type of a for loop iterable (line 31)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 31, 8), installed_libraries_59531)
        # Getting the type of the for loop variable (line 31)
        for_loop_var_59532 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 31, 8), installed_libraries_59531)
        # Assigning a type to the variable 'l' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'l', for_loop_var_59532)
        # SSA begins for a for statement (line 31)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 32):
        
        # Call to join(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'self' (line 32)
        self_59536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 38), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 32)
        install_dir_59537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 38), self_59536, 'install_dir')
        # Getting the type of 'l' (line 32)
        l_59538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 56), 'l', False)
        # Obtaining the member 'target_dir' of a type (line 32)
        target_dir_59539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 56), l_59538, 'target_dir')
        # Processing the call keyword arguments (line 32)
        kwargs_59540 = {}
        # Getting the type of 'os' (line 32)
        os_59533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 32)
        path_59534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 25), os_59533, 'path')
        # Obtaining the member 'join' of a type (line 32)
        join_59535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 25), path_59534, 'join')
        # Calling join(args, kwargs) (line 32)
        join_call_result_59541 = invoke(stypy.reporting.localization.Localization(__file__, 32, 25), join_59535, *[install_dir_59537, target_dir_59539], **kwargs_59540)
        
        # Assigning a type to the variable 'target_dir' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'target_dir', join_call_result_59541)
        
        # Assigning a Call to a Name (line 33):
        
        # Call to library_filename(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'l' (line 33)
        l_59544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 45), 'l', False)
        # Obtaining the member 'name' of a type (line 33)
        name_59545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 45), l_59544, 'name')
        # Processing the call keyword arguments (line 33)
        kwargs_59546 = {}
        # Getting the type of 'compiler' (line 33)
        compiler_59542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'compiler', False)
        # Obtaining the member 'library_filename' of a type (line 33)
        library_filename_59543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 19), compiler_59542, 'library_filename')
        # Calling library_filename(args, kwargs) (line 33)
        library_filename_call_result_59547 = invoke(stypy.reporting.localization.Localization(__file__, 33, 19), library_filename_59543, *[name_59545], **kwargs_59546)
        
        # Assigning a type to the variable 'name' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'name', library_filename_call_result_59547)
        
        # Assigning a Call to a Name (line 34):
        
        # Call to join(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'build_dir' (line 34)
        build_dir_59551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 34), 'build_dir', False)
        # Getting the type of 'name' (line 34)
        name_59552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 45), 'name', False)
        # Processing the call keyword arguments (line 34)
        kwargs_59553 = {}
        # Getting the type of 'os' (line 34)
        os_59548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'os', False)
        # Obtaining the member 'path' of a type (line 34)
        path_59549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 21), os_59548, 'path')
        # Obtaining the member 'join' of a type (line 34)
        join_59550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 21), path_59549, 'join')
        # Calling join(args, kwargs) (line 34)
        join_call_result_59554 = invoke(stypy.reporting.localization.Localization(__file__, 34, 21), join_59550, *[build_dir_59551, name_59552], **kwargs_59553)
        
        # Assigning a type to the variable 'source' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'source', join_call_result_59554)
        
        # Call to mkpath(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'target_dir' (line 35)
        target_dir_59557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 24), 'target_dir', False)
        # Processing the call keyword arguments (line 35)
        kwargs_59558 = {}
        # Getting the type of 'self' (line 35)
        self_59555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 35)
        mkpath_59556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), self_59555, 'mkpath')
        # Calling mkpath(args, kwargs) (line 35)
        mkpath_call_result_59559 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), mkpath_59556, *[target_dir_59557], **kwargs_59558)
        
        
        # Call to append(...): (line 36)
        # Processing the call arguments (line 36)
        
        # Obtaining the type of the subscript
        int_59563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 68), 'int')
        
        # Call to copy_file(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'source' (line 36)
        source_59566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 48), 'source', False)
        # Getting the type of 'target_dir' (line 36)
        target_dir_59567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 56), 'target_dir', False)
        # Processing the call keyword arguments (line 36)
        kwargs_59568 = {}
        # Getting the type of 'self' (line 36)
        self_59564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 33), 'self', False)
        # Obtaining the member 'copy_file' of a type (line 36)
        copy_file_59565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 33), self_59564, 'copy_file')
        # Calling copy_file(args, kwargs) (line 36)
        copy_file_call_result_59569 = invoke(stypy.reporting.localization.Localization(__file__, 36, 33), copy_file_59565, *[source_59566, target_dir_59567], **kwargs_59568)
        
        # Obtaining the member '__getitem__' of a type (line 36)
        getitem___59570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 33), copy_file_call_result_59569, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 36)
        subscript_call_result_59571 = invoke(stypy.reporting.localization.Localization(__file__, 36, 33), getitem___59570, int_59563)
        
        # Processing the call keyword arguments (line 36)
        kwargs_59572 = {}
        # Getting the type of 'self' (line 36)
        self_59560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'self', False)
        # Obtaining the member 'outfiles' of a type (line 36)
        outfiles_59561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), self_59560, 'outfiles')
        # Obtaining the member 'append' of a type (line 36)
        append_59562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), outfiles_59561, 'append')
        # Calling append(args, kwargs) (line 36)
        append_call_result_59573 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), append_59562, *[subscript_call_result_59571], **kwargs_59572)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_59574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59574)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_59574


    @norecursion
    def get_outputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_outputs'
        module_type_store = module_type_store.open_function_context('get_outputs', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_clib.get_outputs.__dict__.__setitem__('stypy_localization', localization)
        install_clib.get_outputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_clib.get_outputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_clib.get_outputs.__dict__.__setitem__('stypy_function_name', 'install_clib.get_outputs')
        install_clib.get_outputs.__dict__.__setitem__('stypy_param_names_list', [])
        install_clib.get_outputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_clib.get_outputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_clib.get_outputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_clib.get_outputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_clib.get_outputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_clib.get_outputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_clib.get_outputs', [], None, None, defaults, varargs, kwargs)

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

        # Getting the type of 'self' (line 39)
        self_59575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'self')
        # Obtaining the member 'outfiles' of a type (line 39)
        outfiles_59576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 15), self_59575, 'outfiles')
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', outfiles_59576)
        
        # ################# End of 'get_outputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_outputs' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_59577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59577)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_outputs'
        return stypy_return_type_59577


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 8, 0, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_clib.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'install_clib' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'install_clib', install_clib)

# Assigning a Str to a Name (line 9):
str_59578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 18), 'str', 'Command to install installable C libraries')
# Getting the type of 'install_clib'
install_clib_59579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_clib')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_clib_59579, 'description', str_59578)

# Assigning a List to a Name (line 11):

# Obtaining an instance of the builtin type 'list' (line 11)
list_59580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)

# Getting the type of 'install_clib'
install_clib_59581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_clib')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_clib_59581, 'user_options', list_59580)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
