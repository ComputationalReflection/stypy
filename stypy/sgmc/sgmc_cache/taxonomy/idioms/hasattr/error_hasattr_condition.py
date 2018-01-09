
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "If hasattr condition with dynamic type inspection"
4: 
5: if __name__ == '__main__':
6: 
7:     class Test:
8:         attribute = None
9: 
10: 
11:     test = Test()
12: 
13:     if True:
14:         test.attribute = 3
15:     else:
16:         test.attribute = "3"
17: 
18:     b = None
19:     if hasattr(test, 'attribute'):
20:         # Type warning
21:         r = test.attribute / 3
22:         # Type warning
23:         r2 = test.attribute[0]
24:         b = 3
25: 
26:     # Type warning
27:     r3 = test.attribute / 3
28: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'If hasattr condition with dynamic type inspection')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Test' class

    class Test:
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 7, 4, False)
            # Assigning a type to the variable 'self' (line 8)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Test' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'Test', Test)
    
    # Assigning a Name to a Name (line 8):
    # Getting the type of 'None' (line 8)
    None_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 20), 'None')
    # Getting the type of 'Test'
    Test_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Test')
    # Setting the type of the member 'attribute' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Test_3, 'attribute', None_2)
    
    # Assigning a Call to a Name (line 11):
    
    # Call to Test(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_5 = {}
    # Getting the type of 'Test' (line 11)
    Test_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'Test', False)
    # Calling Test(args, kwargs) (line 11)
    Test_call_result_6 = invoke(stypy.reporting.localization.Localization(__file__, 11, 11), Test_4, *[], **kwargs_5)
    
    # Assigning a type to the variable 'test' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'test', Test_call_result_6)
    
    # Getting the type of 'True' (line 13)
    True_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 7), 'True')
    # Testing the type of an if condition (line 13)
    if_condition_8 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 13, 4), True_7)
    # Assigning a type to the variable 'if_condition_8' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'if_condition_8', if_condition_8)
    # SSA begins for if statement (line 13)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Attribute (line 14):
    int_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'int')
    # Getting the type of 'test' (line 14)
    test_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'test')
    # Setting the type of the member 'attribute' of a type (line 14)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), test_10, 'attribute', int_9)
    # SSA branch for the else part of an if statement (line 13)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Attribute (line 16):
    str_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'str', '3')
    # Getting the type of 'test' (line 16)
    test_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'test')
    # Setting the type of the member 'attribute' of a type (line 16)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), test_12, 'attribute', str_11)
    # SSA join for if statement (line 13)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 18):
    # Getting the type of 'None' (line 18)
    None_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'None')
    # Assigning a type to the variable 'b' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'b', None_13)
    
    # Type idiom detected: calculating its left and rigth part (line 19)
    str_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 21), 'str', 'attribute')
    # Getting the type of 'test' (line 19)
    test_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'test')
    
    (may_be_16, more_types_in_union_17) = may_provide_member(str_14, test_15)

    if may_be_16:

        if more_types_in_union_17:
            # Runtime conditional SSA (line 19)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'test' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'test', remove_not_member_provider_from_union(test_15, 'attribute'))
        
        # Assigning a BinOp to a Name (line 21):
        # Getting the type of 'test' (line 21)
        test_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'test')
        # Obtaining the member 'attribute' of a type (line 21)
        attribute_19 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 12), test_18, 'attribute')
        int_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 29), 'int')
        # Applying the binary operator 'div' (line 21)
        result_div_21 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 12), 'div', attribute_19, int_20)
        
        # Assigning a type to the variable 'r' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'r', result_div_21)
        
        # Assigning a Subscript to a Name (line 23):
        
        # Obtaining the type of the subscript
        int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 28), 'int')
        # Getting the type of 'test' (line 23)
        test_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'test')
        # Obtaining the member 'attribute' of a type (line 23)
        attribute_24 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 13), test_23, 'attribute')
        # Obtaining the member '__getitem__' of a type (line 23)
        getitem___25 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 13), attribute_24, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 23)
        subscript_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 23, 13), getitem___25, int_22)
        
        # Assigning a type to the variable 'r2' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'r2', subscript_call_result_26)
        
        # Assigning a Num to a Name (line 24):
        int_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 12), 'int')
        # Assigning a type to the variable 'b' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'b', int_27)

        if more_types_in_union_17:
            # SSA join for if statement (line 19)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 27):
    # Getting the type of 'test' (line 27)
    test_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 9), 'test')
    # Obtaining the member 'attribute' of a type (line 27)
    attribute_29 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 9), test_28, 'attribute')
    int_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 26), 'int')
    # Applying the binary operator 'div' (line 27)
    result_div_31 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 9), 'div', attribute_29, int_30)
    
    # Assigning a type to the variable 'r3' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'r3', result_div_31)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
