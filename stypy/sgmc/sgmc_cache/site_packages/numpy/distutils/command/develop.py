
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Override the develop command from setuptools so we can ensure that our
2: generated files (from build_src or build_scripts) are properly converted to real
3: files with filenames.
4: 
5: '''
6: from __future__ import division, absolute_import, print_function
7: 
8: from setuptools.command.develop import develop as old_develop
9: 
10: class develop(old_develop):
11:     __doc__ = old_develop.__doc__
12:     def install_for_development(self):
13:         # Build sources in-place, too.
14:         self.reinitialize_command('build_src', inplace=1)
15:         # Make sure scripts are built.
16:         self.run_command('build_scripts')
17:         old_develop.install_for_development(self)
18: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_59290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', ' Override the develop command from setuptools so we can ensure that our\ngenerated files (from build_src or build_scripts) are properly converted to real\nfiles with filenames.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from setuptools.command.develop import old_develop' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_59291 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'setuptools.command.develop')

if (type(import_59291) is not StypyTypeError):

    if (import_59291 != 'pyd_module'):
        __import__(import_59291)
        sys_modules_59292 = sys.modules[import_59291]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'setuptools.command.develop', sys_modules_59292.module_type_store, module_type_store, ['develop'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_59292, sys_modules_59292.module_type_store, module_type_store)
    else:
        from setuptools.command.develop import develop as old_develop

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'setuptools.command.develop', None, module_type_store, ['develop'], [old_develop])

else:
    # Assigning a type to the variable 'setuptools.command.develop' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'setuptools.command.develop', import_59291)

# Adding an alias
module_type_store.add_alias('old_develop', 'develop')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

# Declaration of the 'develop' class
# Getting the type of 'old_develop' (line 10)
old_develop_59293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 14), 'old_develop')

class develop(old_develop_59293, ):

    @norecursion
    def install_for_development(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'install_for_development'
        module_type_store = module_type_store.open_function_context('install_for_development', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        develop.install_for_development.__dict__.__setitem__('stypy_localization', localization)
        develop.install_for_development.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        develop.install_for_development.__dict__.__setitem__('stypy_type_store', module_type_store)
        develop.install_for_development.__dict__.__setitem__('stypy_function_name', 'develop.install_for_development')
        develop.install_for_development.__dict__.__setitem__('stypy_param_names_list', [])
        develop.install_for_development.__dict__.__setitem__('stypy_varargs_param_name', None)
        develop.install_for_development.__dict__.__setitem__('stypy_kwargs_param_name', None)
        develop.install_for_development.__dict__.__setitem__('stypy_call_defaults', defaults)
        develop.install_for_development.__dict__.__setitem__('stypy_call_varargs', varargs)
        develop.install_for_development.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        develop.install_for_development.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'develop.install_for_development', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'install_for_development', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'install_for_development(...)' code ##################

        
        # Call to reinitialize_command(...): (line 14)
        # Processing the call arguments (line 14)
        str_59296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 34), 'str', 'build_src')
        # Processing the call keyword arguments (line 14)
        int_59297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 55), 'int')
        keyword_59298 = int_59297
        kwargs_59299 = {'inplace': keyword_59298}
        # Getting the type of 'self' (line 14)
        self_59294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self', False)
        # Obtaining the member 'reinitialize_command' of a type (line 14)
        reinitialize_command_59295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), self_59294, 'reinitialize_command')
        # Calling reinitialize_command(args, kwargs) (line 14)
        reinitialize_command_call_result_59300 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), reinitialize_command_59295, *[str_59296], **kwargs_59299)
        
        
        # Call to run_command(...): (line 16)
        # Processing the call arguments (line 16)
        str_59303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'str', 'build_scripts')
        # Processing the call keyword arguments (line 16)
        kwargs_59304 = {}
        # Getting the type of 'self' (line 16)
        self_59301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self', False)
        # Obtaining the member 'run_command' of a type (line 16)
        run_command_59302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), self_59301, 'run_command')
        # Calling run_command(args, kwargs) (line 16)
        run_command_call_result_59305 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), run_command_59302, *[str_59303], **kwargs_59304)
        
        
        # Call to install_for_development(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'self' (line 17)
        self_59308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 44), 'self', False)
        # Processing the call keyword arguments (line 17)
        kwargs_59309 = {}
        # Getting the type of 'old_develop' (line 17)
        old_develop_59306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'old_develop', False)
        # Obtaining the member 'install_for_development' of a type (line 17)
        install_for_development_59307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), old_develop_59306, 'install_for_development')
        # Calling install_for_development(args, kwargs) (line 17)
        install_for_development_call_result_59310 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), install_for_development_59307, *[self_59308], **kwargs_59309)
        
        
        # ################# End of 'install_for_development(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'install_for_development' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_59311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59311)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'install_for_development'
        return stypy_return_type_59311


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 10, 0, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'develop.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'develop' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'develop', develop)

# Assigning a Attribute to a Name (line 11):
# Getting the type of 'old_develop' (line 11)
old_develop_59312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'old_develop')
# Obtaining the member '__doc__' of a type (line 11)
doc___59313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 14), old_develop_59312, '__doc__')
# Getting the type of 'develop'
develop_59314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'develop')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), develop_59314, '__doc__', doc___59313)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
