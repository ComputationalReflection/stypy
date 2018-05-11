
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from ...python_types_copy.non_python_type_copy import NonPythonType
2: 
3: 
4: class UndefinedType(NonPythonType):
5:     '''
6:     The type of an undefined variable
7:     '''
8: 
9:     def __str__(self):
10:         return 'Undefined'
11: 
12:     def __repr__(self):
13:         return self.__str__()
14: 
15:     def __eq__(self, other):
16:         return isinstance(other, UndefinedType)
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy import NonPythonType' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_12570 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy')

if (type(import_12570) is not StypyTypeError):

    if (import_12570 != 'pyd_module'):
        __import__(import_12570)
        sys_modules_12571 = sys.modules[import_12570]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy', sys_modules_12571.module_type_store, module_type_store, ['NonPythonType'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_12571, sys_modules_12571.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy import NonPythonType

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy', None, module_type_store, ['NonPythonType'], [NonPythonType])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy', import_12570)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

# Declaration of the 'UndefinedType' class
# Getting the type of 'NonPythonType' (line 4)
NonPythonType_12572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 20), 'NonPythonType')

class UndefinedType(NonPythonType_12572, ):
    str_12573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\n    The type of an undefined variable\n    ')

    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 9, 4, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_function_name', 'UndefinedType.stypy__str__')
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UndefinedType.stypy__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        str_12574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 15), 'str', 'Undefined')
        # Assigning a type to the variable 'stypy_return_type' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'stypy_return_type', str_12574)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 9)
        stypy_return_type_12575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12575)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_12575


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'UndefinedType.stypy__repr__')
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UndefinedType.stypy__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        
        # Call to __str__(...): (line 13)
        # Processing the call keyword arguments (line 13)
        kwargs_12578 = {}
        # Getting the type of 'self' (line 13)
        self_12576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 15), 'self', False)
        # Obtaining the member '__str__' of a type (line 13)
        str___12577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 15), self_12576, '__str__')
        # Calling __str__(args, kwargs) (line 13)
        str___call_result_12579 = invoke(stypy.reporting.localization.Localization(__file__, 13, 15), str___12577, *[], **kwargs_12578)
        
        # Assigning a type to the variable 'stypy_return_type' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', str___call_result_12579)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_12580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12580)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_12580


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'UndefinedType.stypy__eq__')
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UndefinedType.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Call to isinstance(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'other' (line 16)
        other_12582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 26), 'other', False)
        # Getting the type of 'UndefinedType' (line 16)
        UndefinedType_12583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 33), 'UndefinedType', False)
        # Processing the call keyword arguments (line 16)
        kwargs_12584 = {}
        # Getting the type of 'isinstance' (line 16)
        isinstance_12581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 16)
        isinstance_call_result_12585 = invoke(stypy.reporting.localization.Localization(__file__, 16, 15), isinstance_12581, *[other_12582, UndefinedType_12583], **kwargs_12584)
        
        # Assigning a type to the variable 'stypy_return_type' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'stypy_return_type', isinstance_call_result_12585)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_12586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12586)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_12586


# Assigning a type to the variable 'UndefinedType' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'UndefinedType', UndefinedType)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
