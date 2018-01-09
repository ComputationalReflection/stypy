
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: class WrongFoo:
2:     def __abs__(self):
3:         return "4"
4: 
5: 
6: x = (abs(WrongFoo()) + 3)  # Not reported, runtime crash
7: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'WrongFoo' class

class WrongFoo:

    @norecursion
    def __abs__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__abs__'
        module_type_store = module_type_store.open_function_context('__abs__', 2, 4, False)
        # Assigning a type to the variable 'self' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        WrongFoo.__abs__.__dict__.__setitem__('stypy_localization', localization)
        WrongFoo.__abs__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        WrongFoo.__abs__.__dict__.__setitem__('stypy_type_store', module_type_store)
        WrongFoo.__abs__.__dict__.__setitem__('stypy_function_name', 'WrongFoo.__abs__')
        WrongFoo.__abs__.__dict__.__setitem__('stypy_param_names_list', [])
        WrongFoo.__abs__.__dict__.__setitem__('stypy_varargs_param_name', None)
        WrongFoo.__abs__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        WrongFoo.__abs__.__dict__.__setitem__('stypy_call_defaults', defaults)
        WrongFoo.__abs__.__dict__.__setitem__('stypy_call_varargs', varargs)
        WrongFoo.__abs__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        WrongFoo.__abs__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'WrongFoo.__abs__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__abs__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__abs__(...)' code ##################

        str_7763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 15), 'str', '4')
        # Assigning a type to the variable 'stypy_return_type' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'stypy_return_type', str_7763)
        
        # ################# End of '__abs__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__abs__' in the type store
        # Getting the type of 'stypy_return_type' (line 2)
        stypy_return_type_7764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7764)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__abs__'
        return stypy_return_type_7764


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1, 0, False)
        # Assigning a type to the variable 'self' (line 2)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'WrongFoo.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'WrongFoo' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'WrongFoo', WrongFoo)

# Assigning a BinOp to a Name (line 6):

# Call to abs(...): (line 6)
# Processing the call arguments (line 6)

# Call to WrongFoo(...): (line 6)
# Processing the call keyword arguments (line 6)
kwargs_7767 = {}
# Getting the type of 'WrongFoo' (line 6)
WrongFoo_7766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 9), 'WrongFoo', False)
# Calling WrongFoo(args, kwargs) (line 6)
WrongFoo_call_result_7768 = invoke(stypy.reporting.localization.Localization(__file__, 6, 9), WrongFoo_7766, *[], **kwargs_7767)

# Processing the call keyword arguments (line 6)
kwargs_7769 = {}
# Getting the type of 'abs' (line 6)
abs_7765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'abs', False)
# Calling abs(args, kwargs) (line 6)
abs_call_result_7770 = invoke(stypy.reporting.localization.Localization(__file__, 6, 5), abs_7765, *[WrongFoo_call_result_7768], **kwargs_7769)

int_7771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 23), 'int')
# Applying the binary operator '+' (line 6)
result_add_7772 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 5), '+', abs_call_result_7770, int_7771)

# Assigning a type to the variable 'x' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'x', result_add_7772)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
