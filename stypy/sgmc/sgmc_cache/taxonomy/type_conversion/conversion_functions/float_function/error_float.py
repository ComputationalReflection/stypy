
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "One single parameter of any type not convertible to float"
3: 
4: if __name__ == '__main__':
5:     class Sample:
6:         pass
7: 
8: 
9:     # Type error #
10:     print float(Sample()) + 3
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'One single parameter of any type not convertible to float')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 5, 4, False)
            # Assigning a type to the variable 'self' (line 6)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sample.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Sample' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'Sample', Sample)
    
    # Call to float(...): (line 10)
    # Processing the call arguments (line 10)
    
    # Call to Sample(...): (line 10)
    # Processing the call keyword arguments (line 10)
    kwargs_4 = {}
    # Getting the type of 'Sample' (line 10)
    Sample_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 16), 'Sample', False)
    # Calling Sample(args, kwargs) (line 10)
    Sample_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 10, 16), Sample_3, *[], **kwargs_4)
    
    # Processing the call keyword arguments (line 10)
    kwargs_6 = {}
    # Getting the type of 'float' (line 10)
    float_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 10), 'float', False)
    # Calling float(args, kwargs) (line 10)
    float_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 10, 10), float_2, *[Sample_call_result_5], **kwargs_6)
    
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 28), 'int')
    # Applying the binary operator '+' (line 10)
    result_add_9 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 10), '+', float_call_result_7, int_8)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
