
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "getattr method is present, but is invoked with a wrong number of parameters"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (AnyType, Str) -> DynamicType
7:     # (AnyType, Str, AnyType) -> DynamicType
8: 
9: 
10:     # Call the builtin with incorrect number of parameters
11:     class Sample:
12:         def __init__(self):
13:             self.att = 0
14: 
15: 
16:     # Call the builtin with incorrect types of parameters
17:     # Type error
18:     ret = getattr(Sample(), 'att', 3, 5)
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'getattr method is present, but is invoked with a wrong number of parameters')
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
            module_type_store = module_type_store.open_function_context('__init__', 12, 8, False)
            # Assigning a type to the variable 'self' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'self', type_of_self)
            
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

            
            # Assigning a Num to a Attribute (line 13):
            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 23), 'int')
            # Getting the type of 'self' (line 13)
            self_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'self')
            # Setting the type of the member 'att' of a type (line 13)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 12), self_3, 'att', int_2)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'Sample' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'Sample', Sample)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to getattr(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Call to Sample(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_6 = {}
    # Getting the type of 'Sample' (line 18)
    Sample_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 18), 'Sample', False)
    # Calling Sample(args, kwargs) (line 18)
    Sample_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 18, 18), Sample_5, *[], **kwargs_6)
    
    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 28), 'str', 'att')
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'int')
    int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 38), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_11 = {}
    # Getting the type of 'getattr' (line 18)
    getattr_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'getattr', False)
    # Calling getattr(args, kwargs) (line 18)
    getattr_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), getattr_4, *[Sample_call_result_7, str_8, int_9, int_10], **kwargs_11)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', getattr_call_result_12)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
