
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "If / else hasattr condition with dynamic type inspection"
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
18:     if hasattr(test, 'attribute'):
19:         # Type warning
20:         r = test.attribute / 3
21:         # Type warning
22:         r2 = test.attribute[0]
23:     else:
24:         # Never executed
25:         r3 = test.attribute[0]
26:         r4 = test.attribute / 3
27: 
28:     # Type warning
29:     r5 = test.attribute / 3
30: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'If / else hasattr condition with dynamic type inspection')
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
    
    
    # Type idiom detected: calculating its left and rigth part (line 18)
    str_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 21), 'str', 'attribute')
    # Getting the type of 'test' (line 18)
    test_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'test')
    
    (may_be_15, more_types_in_union_16) = may_provide_member(str_13, test_14)

    if may_be_15:

        if more_types_in_union_16:
            # Runtime conditional SSA (line 18)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'test' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'test', remove_not_member_provider_from_union(test_14, 'attribute'))
        
        # Assigning a BinOp to a Name (line 20):
        # Getting the type of 'test' (line 20)
        test_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'test')
        # Obtaining the member 'attribute' of a type (line 20)
        attribute_18 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), test_17, 'attribute')
        int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 29), 'int')
        # Applying the binary operator 'div' (line 20)
        result_div_20 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 12), 'div', attribute_18, int_19)
        
        # Assigning a type to the variable 'r' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'r', result_div_20)
        
        # Assigning a Subscript to a Name (line 22):
        
        # Obtaining the type of the subscript
        int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 28), 'int')
        # Getting the type of 'test' (line 22)
        test_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 13), 'test')
        # Obtaining the member 'attribute' of a type (line 22)
        attribute_23 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 13), test_22, 'attribute')
        # Obtaining the member '__getitem__' of a type (line 22)
        getitem___24 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 13), attribute_23, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 22)
        subscript_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 22, 13), getitem___24, int_21)
        
        # Assigning a type to the variable 'r2' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'r2', subscript_call_result_25)

        if more_types_in_union_16:
            # Runtime conditional SSA for else branch (line 18)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_15) or more_types_in_union_16):
        # Assigning a type to the variable 'test' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'test', remove_member_provider_from_union(test_14, 'attribute'))
        
        # Assigning a Subscript to a Name (line 25):
        
        # Obtaining the type of the subscript
        int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 28), 'int')
        # Getting the type of 'test' (line 25)
        test_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 13), 'test')
        # Obtaining the member 'attribute' of a type (line 25)
        attribute_28 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 13), test_27, 'attribute')
        # Obtaining the member '__getitem__' of a type (line 25)
        getitem___29 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 13), attribute_28, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 25)
        subscript_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 25, 13), getitem___29, int_26)
        
        # Assigning a type to the variable 'r3' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'r3', subscript_call_result_30)
        
        # Assigning a BinOp to a Name (line 26):
        # Getting the type of 'test' (line 26)
        test_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 13), 'test')
        # Obtaining the member 'attribute' of a type (line 26)
        attribute_32 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 13), test_31, 'attribute')
        int_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 30), 'int')
        # Applying the binary operator 'div' (line 26)
        result_div_34 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 13), 'div', attribute_32, int_33)
        
        # Assigning a type to the variable 'r4' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'r4', result_div_34)

        if (may_be_15 and more_types_in_union_16):
            # SSA join for if statement (line 18)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 29):
    # Getting the type of 'test' (line 29)
    test_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 9), 'test')
    # Obtaining the member 'attribute' of a type (line 29)
    attribute_36 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 9), test_35, 'attribute')
    int_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 26), 'int')
    # Applying the binary operator 'div' (line 29)
    result_div_38 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 9), 'div', attribute_36, int_37)
    
    # Assigning a type to the variable 'r5' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'r5', result_div_38)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
