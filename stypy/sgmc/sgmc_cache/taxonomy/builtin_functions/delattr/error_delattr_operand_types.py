
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "delattr builtin is invoked, but incorrect parameter types are passed"
3: 
4: if __name__ == '__main__':
5:     # Call options
6:     # (AnyType, Str) -> types.NoneType
7:     class Sample:
8:         def __init__(self):
9:             self.att_to_delete = 0
10: 
11: 
12:     # Call the builtin with correct parameters
13:     # No error
14:     ret = delattr(Sample(), 'att_to_delete')
15: 
16:     # Call the builtin with incorrect types of parameters
17:     # Type error
18:     ret = delattr(Sample(), 'not_exist')
19:     # Type error
20:     ret = delattr(Sample, 'att_to_delete')
21:     # Type error
22:     ret = delattr(int, '__doc__')
23: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'delattr builtin is invoked, but incorrect parameter types are passed')
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
            module_type_store = module_type_store.open_function_context('__init__', 8, 8, False)
            # Assigning a type to the variable 'self' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'self', type_of_self)
            
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

            
            # Assigning a Num to a Attribute (line 9):
            int_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 33), 'int')
            # Getting the type of 'self' (line 9)
            self_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'self')
            # Setting the type of the member 'att_to_delete' of a type (line 9)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 12), self_3, 'att_to_delete', int_2)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'Sample' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'Sample', Sample)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to delattr(...): (line 14)
    # Processing the call arguments (line 14)
    
    # Call to Sample(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_6 = {}
    # Getting the type of 'Sample' (line 14)
    Sample_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 18), 'Sample', False)
    # Calling Sample(args, kwargs) (line 14)
    Sample_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 14, 18), Sample_5, *[], **kwargs_6)
    
    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 28), 'str', 'att_to_delete')
    # Processing the call keyword arguments (line 14)
    kwargs_9 = {}
    # Getting the type of 'delattr' (line 14)
    delattr_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 10), 'delattr', False)
    # Calling delattr(args, kwargs) (line 14)
    delattr_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 14, 10), delattr_4, *[Sample_call_result_7, str_8], **kwargs_9)
    
    # Assigning a type to the variable 'ret' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'ret', delattr_call_result_10)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to delattr(...): (line 18)
    # Processing the call arguments (line 18)
    
    # Call to Sample(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_13 = {}
    # Getting the type of 'Sample' (line 18)
    Sample_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 18), 'Sample', False)
    # Calling Sample(args, kwargs) (line 18)
    Sample_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 18, 18), Sample_12, *[], **kwargs_13)
    
    str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 28), 'str', 'not_exist')
    # Processing the call keyword arguments (line 18)
    kwargs_16 = {}
    # Getting the type of 'delattr' (line 18)
    delattr_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 10), 'delattr', False)
    # Calling delattr(args, kwargs) (line 18)
    delattr_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 18, 10), delattr_11, *[Sample_call_result_14, str_15], **kwargs_16)
    
    # Assigning a type to the variable 'ret' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'ret', delattr_call_result_17)
    
    # Assigning a Call to a Name (line 20):
    
    # Call to delattr(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'Sample' (line 20)
    Sample_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 18), 'Sample', False)
    str_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'str', 'att_to_delete')
    # Processing the call keyword arguments (line 20)
    kwargs_21 = {}
    # Getting the type of 'delattr' (line 20)
    delattr_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 10), 'delattr', False)
    # Calling delattr(args, kwargs) (line 20)
    delattr_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 20, 10), delattr_18, *[Sample_19, str_20], **kwargs_21)
    
    # Assigning a type to the variable 'ret' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'ret', delattr_call_result_22)
    
    # Assigning a Call to a Name (line 22):
    
    # Call to delattr(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'int' (line 22)
    int_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 18), 'int', False)
    str_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'str', '__doc__')
    # Processing the call keyword arguments (line 22)
    kwargs_26 = {}
    # Getting the type of 'delattr' (line 22)
    delattr_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), 'delattr', False)
    # Calling delattr(args, kwargs) (line 22)
    delattr_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 22, 10), delattr_23, *[int_24, str_25], **kwargs_26)
    
    # Assigning a type to the variable 'ret' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'ret', delattr_call_result_27)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
