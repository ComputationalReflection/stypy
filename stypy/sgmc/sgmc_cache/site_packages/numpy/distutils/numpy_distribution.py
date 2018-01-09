
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # XXX: Handle setuptools ?
2: from __future__ import division, absolute_import, print_function
3: 
4: from distutils.core import Distribution
5: 
6: # This class is used because we add new files (sconscripts, and so on) with the
7: # scons command
8: class NumpyDistribution(Distribution):
9:     def __init__(self, attrs = None):
10:         # A list of (sconscripts, pre_hook, post_hook, src, parent_names)
11:         self.scons_data = []
12:         # A list of installable libraries
13:         self.installed_libraries = []
14:         # A dict of pkg_config files to generate/install
15:         self.installed_pkg_config = {}
16:         Distribution.__init__(self, attrs)
17: 
18:     def has_scons_scripts(self):
19:         return bool(self.scons_data)
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from distutils.core import Distribution' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_45223 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.core')

if (type(import_45223) is not StypyTypeError):

    if (import_45223 != 'pyd_module'):
        __import__(import_45223)
        sys_modules_45224 = sys.modules[import_45223]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.core', sys_modules_45224.module_type_store, module_type_store, ['Distribution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_45224, sys_modules_45224.module_type_store, module_type_store)
    else:
        from distutils.core import Distribution

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.core', None, module_type_store, ['Distribution'], [Distribution])

else:
    # Assigning a type to the variable 'distutils.core' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.core', import_45223)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

# Declaration of the 'NumpyDistribution' class
# Getting the type of 'Distribution' (line 8)
Distribution_45225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 24), 'Distribution')

class NumpyDistribution(Distribution_45225, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 9)
        None_45226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 31), 'None')
        defaults = [None_45226]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 9, 4, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyDistribution.__init__', ['attrs'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['attrs'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a List to a Attribute (line 11):
        
        # Obtaining an instance of the builtin type 'list' (line 11)
        list_45227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 11)
        
        # Getting the type of 'self' (line 11)
        self_45228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'self')
        # Setting the type of the member 'scons_data' of a type (line 11)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), self_45228, 'scons_data', list_45227)
        
        # Assigning a List to a Attribute (line 13):
        
        # Obtaining an instance of the builtin type 'list' (line 13)
        list_45229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 13)
        
        # Getting the type of 'self' (line 13)
        self_45230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'self')
        # Setting the type of the member 'installed_libraries' of a type (line 13)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), self_45230, 'installed_libraries', list_45229)
        
        # Assigning a Dict to a Attribute (line 15):
        
        # Obtaining an instance of the builtin type 'dict' (line 15)
        dict_45231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 36), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 15)
        
        # Getting the type of 'self' (line 15)
        self_45232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self')
        # Setting the type of the member 'installed_pkg_config' of a type (line 15)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_45232, 'installed_pkg_config', dict_45231)
        
        # Call to __init__(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'self' (line 16)
        self_45235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 30), 'self', False)
        # Getting the type of 'attrs' (line 16)
        attrs_45236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 36), 'attrs', False)
        # Processing the call keyword arguments (line 16)
        kwargs_45237 = {}
        # Getting the type of 'Distribution' (line 16)
        Distribution_45233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'Distribution', False)
        # Obtaining the member '__init__' of a type (line 16)
        init___45234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), Distribution_45233, '__init__')
        # Calling __init__(args, kwargs) (line 16)
        init___call_result_45238 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), init___45234, *[self_45235, attrs_45236], **kwargs_45237)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def has_scons_scripts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_scons_scripts'
        module_type_store = module_type_store.open_function_context('has_scons_scripts', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyDistribution.has_scons_scripts.__dict__.__setitem__('stypy_localization', localization)
        NumpyDistribution.has_scons_scripts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyDistribution.has_scons_scripts.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyDistribution.has_scons_scripts.__dict__.__setitem__('stypy_function_name', 'NumpyDistribution.has_scons_scripts')
        NumpyDistribution.has_scons_scripts.__dict__.__setitem__('stypy_param_names_list', [])
        NumpyDistribution.has_scons_scripts.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyDistribution.has_scons_scripts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyDistribution.has_scons_scripts.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyDistribution.has_scons_scripts.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyDistribution.has_scons_scripts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyDistribution.has_scons_scripts.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyDistribution.has_scons_scripts', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_scons_scripts', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_scons_scripts(...)' code ##################

        
        # Call to bool(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'self' (line 19)
        self_45240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'self', False)
        # Obtaining the member 'scons_data' of a type (line 19)
        scons_data_45241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 20), self_45240, 'scons_data')
        # Processing the call keyword arguments (line 19)
        kwargs_45242 = {}
        # Getting the type of 'bool' (line 19)
        bool_45239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'bool', False)
        # Calling bool(args, kwargs) (line 19)
        bool_call_result_45243 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), bool_45239, *[scons_data_45241], **kwargs_45242)
        
        # Assigning a type to the variable 'stypy_return_type' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'stypy_return_type', bool_call_result_45243)
        
        # ################# End of 'has_scons_scripts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_scons_scripts' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_45244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45244)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_scons_scripts'
        return stypy_return_type_45244


# Assigning a type to the variable 'NumpyDistribution' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'NumpyDistribution', NumpyDistribution)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
