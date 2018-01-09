
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "delattr builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (AnyType, Str) -> types.NoneType
7: 
8:     class Sample:
9:         def __init__(self):
10:             self.att_to_delete = 0
11: 
12: 
13:     # Call the builtin
14:     # No error
15:     ret = delattr(Sample(), 'att_to_delete')
16: 
17:     # Type error
18:     ret.unexisting_method()
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'delattr builtin is invoked and its return type is used to call an non existing method')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Sample' class

    class Sample:

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 9, 8, False)
            # Assigning a type to the variable 'self' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'self', type_of_self)
            
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

            
            # Assigning a Num to a Attribute (line 10):
            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 33), 'int')
            # Getting the type of 'self' (line 10)
            self_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'self')
            # Setting the type of the member 'att_to_delete' of a type (line 10)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 12), self_3, 'att_to_delete', int_2)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'Sample' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'Sample', Sample)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to delattr(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to Sample(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_6 = {}
    # Getting the type of 'Sample' (line 15)
    Sample_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 18), 'Sample', False)
    # Calling Sample(args, kwargs) (line 15)
    Sample_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 15, 18), Sample_5, *[], **kwargs_6)
    
    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 28), 'str', 'att_to_delete')
    # Processing the call keyword arguments (line 15)
    kwargs_9 = {}
    # Getting the type of 'delattr' (line 15)
    delattr_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'delattr', False)
    # Calling delattr(args, kwargs) (line 15)
    delattr_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), delattr_4, *[Sample_call_result_7, str_8], **kwargs_9)
    
    # Assigning a type to the variable 'ret' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ret', delattr_call_result_10)
    
    # Call to unexisting_method(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_13 = {}
    # Getting the type of 'ret' (line 18)
    ret_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', False)
    # Obtaining the member 'unexisting_method' of a type (line 18)
    unexisting_method_12 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), ret_11, 'unexisting_method')
    # Calling unexisting_method(args, kwargs) (line 18)
    unexisting_method_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), unexisting_method_12, *[], **kwargs_13)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
