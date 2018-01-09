
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "getattr builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (AnyType, Str) -> DynamicType
7:     # (AnyType, Str, AnyType) -> DynamicType
8:     class Sample:
9:         def __init__(self):
10:             self.att = 0
11: 
12: 
13:     # Call the builtin with correct parameters
14:     # No error
15:     ret = getattr(Sample(), 'att')
16:     # No error
17:     ret = getattr(Sample(), 'att', 5)
18: 
19:     # Call the builtin with incorrect types of parameters
20:     # Type error
21:     ret = getattr(Sample(), 'not_exist')
22:     # Type error
23:     ret = getattr(Sample, 'att')
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'getattr builtin is invoked, but incorrect parameter types are passed')
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
            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 23), 'int')
            # Getting the type of 'self' (line 10)
            self_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'self')
            # Setting the type of the member 'att' of a type (line 10)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 12), self_3, 'att', int_2)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'Sample' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'Sample', Sample)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to getattr(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to Sample(...): (line 15)
    # Processing the call keyword arguments (line 15)
    kwargs_6 = {}
    # Getting the type of 'Sample' (line 15)
    Sample_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 18), 'Sample', False)
    # Calling Sample(args, kwargs) (line 15)
    Sample_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 15, 18), Sample_5, *[], **kwargs_6)
    
    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 28), 'str', 'att')
    # Processing the call keyword arguments (line 15)
    kwargs_9 = {}
    # Getting the type of 'getattr' (line 15)
    getattr_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 10), 'getattr', False)
    # Calling getattr(args, kwargs) (line 15)
    getattr_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 15, 10), getattr_4, *[Sample_call_result_7, str_8], **kwargs_9)
    
    # Assigning a type to the variable 'ret' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ret', getattr_call_result_10)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to getattr(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Call to Sample(...): (line 17)
    # Processing the call keyword arguments (line 17)
    kwargs_13 = {}
    # Getting the type of 'Sample' (line 17)
    Sample_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 18), 'Sample', False)
    # Calling Sample(args, kwargs) (line 17)
    Sample_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 17, 18), Sample_12, *[], **kwargs_13)
    
    str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 28), 'str', 'att')
    int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 35), 'int')
    # Processing the call keyword arguments (line 17)
    kwargs_17 = {}
    # Getting the type of 'getattr' (line 17)
    getattr_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 10), 'getattr', False)
    # Calling getattr(args, kwargs) (line 17)
    getattr_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 17, 10), getattr_11, *[Sample_call_result_14, str_15, int_16], **kwargs_17)
    
    # Assigning a type to the variable 'ret' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'ret', getattr_call_result_18)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to getattr(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Call to Sample(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_21 = {}
    # Getting the type of 'Sample' (line 21)
    Sample_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 18), 'Sample', False)
    # Calling Sample(args, kwargs) (line 21)
    Sample_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 21, 18), Sample_20, *[], **kwargs_21)
    
    str_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'str', 'not_exist')
    # Processing the call keyword arguments (line 21)
    kwargs_24 = {}
    # Getting the type of 'getattr' (line 21)
    getattr_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'getattr', False)
    # Calling getattr(args, kwargs) (line 21)
    getattr_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), getattr_19, *[Sample_call_result_22, str_23], **kwargs_24)
    
    # Assigning a type to the variable 'ret' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ret', getattr_call_result_25)
    
    # Assigning a Call to a Name (line 23):
    
    # Call to getattr(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'Sample' (line 23)
    Sample_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 18), 'Sample', False)
    str_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 26), 'str', 'att')
    # Processing the call keyword arguments (line 23)
    kwargs_29 = {}
    # Getting the type of 'getattr' (line 23)
    getattr_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'getattr', False)
    # Calling getattr(args, kwargs) (line 23)
    getattr_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), getattr_26, *[Sample_27, str_28], **kwargs_29)
    
    # Assigning a type to the variable 'ret' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'ret', getattr_call_result_30)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
