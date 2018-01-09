
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "getattr builtin is invoked and its return type is used to call an non existing method"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (AnyType, Str) -> DynamicType
7:     # (AnyType, Str, AnyType) -> DynamicType
8: 
9: 
10:     class Sample:
11:         def __init__(self):
12:             self.att = 0
13: 
14: 
15:     # Call the builtin with incorrect types of parameters
16:     # No error
17:     ret = getattr(Sample(), 'att', 3)
18: 
19:     # Type error
20:     ret.unexisting_method()
21: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'getattr builtin is invoked and its return type is used to call an non existing method')
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
            module_type_store = module_type_store.open_function_context('__init__', 11, 8, False)
            # Assigning a type to the variable 'self' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self', type_of_self)
            
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

            
            # Assigning a Num to a Attribute (line 12):
            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 23), 'int')
            # Getting the type of 'self' (line 12)
            self_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'self')
            # Setting the type of the member 'att' of a type (line 12)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 12), self_3, 'att', int_2)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'Sample' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'Sample', Sample)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to getattr(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Call to Sample(...): (line 17)
    # Processing the call keyword arguments (line 17)
    kwargs_6 = {}
    # Getting the type of 'Sample' (line 17)
    Sample_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 18), 'Sample', False)
    # Calling Sample(args, kwargs) (line 17)
    Sample_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 17, 18), Sample_5, *[], **kwargs_6)
    
    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 28), 'str', 'att')
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 35), 'int')
    # Processing the call keyword arguments (line 17)
    kwargs_10 = {}
    # Getting the type of 'getattr' (line 17)
    getattr_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'getattr', False)
    # Calling getattr(args, kwargs) (line 17)
    getattr_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), getattr_4, *[Sample_call_result_7, str_8, int_9], **kwargs_10)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', getattr_call_result_11)
    
    # Call to unexisting_method(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_14 = {}
    # Getting the type of 'ret' (line 20)
    ret_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'ret', False)
    # Obtaining the member 'unexisting_method' of a type (line 20)
    unexisting_method_13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), ret_12, 'unexisting_method')
    # Calling unexisting_method(args, kwargs) (line 20)
    unexisting_method_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), unexisting_method_13, *[], **kwargs_14)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
