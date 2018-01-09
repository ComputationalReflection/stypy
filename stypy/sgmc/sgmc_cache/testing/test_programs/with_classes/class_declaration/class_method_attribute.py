
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: class MaskedArray:
2:     def var(self):
3:         pass
4: 
5:     var.__doc__ = "hi"
6: 
7: 
8: m = MaskedArray()
9: y = m.var.__doc__
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'MaskedArray' class

class MaskedArray:

    @norecursion
    def var(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'var'
        module_type_store = module_type_store.open_function_context('var', 2, 4, False)
        # Assigning a type to the variable 'self' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MaskedArray.var.__dict__.__setitem__('stypy_localization', localization)
        MaskedArray.var.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MaskedArray.var.__dict__.__setitem__('stypy_type_store', module_type_store)
        MaskedArray.var.__dict__.__setitem__('stypy_function_name', 'MaskedArray.var')
        MaskedArray.var.__dict__.__setitem__('stypy_param_names_list', [])
        MaskedArray.var.__dict__.__setitem__('stypy_varargs_param_name', None)
        MaskedArray.var.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MaskedArray.var.__dict__.__setitem__('stypy_call_defaults', defaults)
        MaskedArray.var.__dict__.__setitem__('stypy_call_varargs', varargs)
        MaskedArray.var.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MaskedArray.var.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedArray.var', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'var', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'var(...)' code ##################

        pass
        
        # ################# End of 'var(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'var' in the type store
        # Getting the type of 'stypy_return_type' (line 2)
        stypy_return_type_2219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2219)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'var'
        return stypy_return_type_2219


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MaskedArray.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MaskedArray' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'MaskedArray', MaskedArray)

# Assigning a Str to a Attribute (line 5):
str_2220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 18), 'str', 'hi')
# Getting the type of 'MaskedArray'
MaskedArray_2221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MaskedArray')
# Obtaining the member 'var' of a type
var_2222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MaskedArray_2221, 'var')
# Setting the type of the member '__doc__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), var_2222, '__doc__', str_2220)

# Assigning a Call to a Name (line 8):

# Call to MaskedArray(...): (line 8)
# Processing the call keyword arguments (line 8)
kwargs_2224 = {}
# Getting the type of 'MaskedArray' (line 8)
MaskedArray_2223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'MaskedArray', False)
# Calling MaskedArray(args, kwargs) (line 8)
MaskedArray_call_result_2225 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), MaskedArray_2223, *[], **kwargs_2224)

# Assigning a type to the variable 'm' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'm', MaskedArray_call_result_2225)

# Assigning a Attribute to a Name (line 9):
# Getting the type of 'm' (line 9)
m_2226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'm')
# Obtaining the member 'var' of a type (line 9)
var_2227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), m_2226, 'var')
# Obtaining the member '__doc__' of a type (line 9)
doc___2228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), var_2227, '__doc__')
# Assigning a type to the variable 'y' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'y', doc___2228)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
