
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import ast
2: 
3: import data_structures_copy
4: 
5: '''
6: Helper functions to create conditional statements
7: '''
8: 
9: 
10: # ############################################### IF STATEMENTS ######################################################
11: 
12: 
13: def create_if(test, body, orelse=list()):
14:     '''
15:     Creates an If AST Node, with its body and else statements
16:     :param test: Test of the if statement
17:     :param body: Statements of the body part
18:     :param orelse: Statements of the else part (optional
19:     :return: AST If Node
20:     '''
21:     if_ = ast.If()
22: 
23:     if_.test = test
24:     if data_structures_copy.is_iterable(body):
25:         if_.body = body
26:     else:
27:         if_.body = [body]
28: 
29:     if data_structures_copy.is_iterable(orelse):
30:         if_.orelse = orelse
31:     else:
32:         if_.orelse = [orelse]
33: 
34:     return if_
35: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import ast' statement (line 1)
import ast

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'ast', ast, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import data_structures_copy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_15277 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'data_structures_copy')

if (type(import_15277) is not StypyTypeError):

    if (import_15277 != 'pyd_module'):
        __import__(import_15277)
        sys_modules_15278 = sys.modules[import_15277]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'data_structures_copy', sys_modules_15278.module_type_store, module_type_store)
    else:
        import data_structures_copy

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'data_structures_copy', data_structures_copy, module_type_store)

else:
    # Assigning a type to the variable 'data_structures_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'data_structures_copy', import_15277)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

str_15279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\nHelper functions to create conditional statements\n')

@norecursion
def create_if(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Call to list(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_15281 = {}
    # Getting the type of 'list' (line 13)
    list_15280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 33), 'list', False)
    # Calling list(args, kwargs) (line 13)
    list_call_result_15282 = invoke(stypy.reporting.localization.Localization(__file__, 13, 33), list_15280, *[], **kwargs_15281)
    
    defaults = [list_call_result_15282]
    # Create a new context for function 'create_if'
    module_type_store = module_type_store.open_function_context('create_if', 13, 0, False)
    
    # Passed parameters checking function
    create_if.stypy_localization = localization
    create_if.stypy_type_of_self = None
    create_if.stypy_type_store = module_type_store
    create_if.stypy_function_name = 'create_if'
    create_if.stypy_param_names_list = ['test', 'body', 'orelse']
    create_if.stypy_varargs_param_name = None
    create_if.stypy_kwargs_param_name = None
    create_if.stypy_call_defaults = defaults
    create_if.stypy_call_varargs = varargs
    create_if.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_if', ['test', 'body', 'orelse'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_if', localization, ['test', 'body', 'orelse'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_if(...)' code ##################

    str_15283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, (-1)), 'str', '\n    Creates an If AST Node, with its body and else statements\n    :param test: Test of the if statement\n    :param body: Statements of the body part\n    :param orelse: Statements of the else part (optional\n    :return: AST If Node\n    ')
    
    # Assigning a Call to a Name (line 21):
    
    # Call to If(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_15286 = {}
    # Getting the type of 'ast' (line 21)
    ast_15284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'ast', False)
    # Obtaining the member 'If' of a type (line 21)
    If_15285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 10), ast_15284, 'If')
    # Calling If(args, kwargs) (line 21)
    If_call_result_15287 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), If_15285, *[], **kwargs_15286)
    
    # Assigning a type to the variable 'if_' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'if_', If_call_result_15287)
    
    # Assigning a Name to a Attribute (line 23):
    # Getting the type of 'test' (line 23)
    test_15288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'test')
    # Getting the type of 'if_' (line 23)
    if__15289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'if_')
    # Setting the type of the member 'test' of a type (line 23)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 4), if__15289, 'test', test_15288)
    
    # Call to is_iterable(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'body' (line 24)
    body_15292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 40), 'body', False)
    # Processing the call keyword arguments (line 24)
    kwargs_15293 = {}
    # Getting the type of 'data_structures_copy' (line 24)
    data_structures_copy_15290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 7), 'data_structures_copy', False)
    # Obtaining the member 'is_iterable' of a type (line 24)
    is_iterable_15291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 7), data_structures_copy_15290, 'is_iterable')
    # Calling is_iterable(args, kwargs) (line 24)
    is_iterable_call_result_15294 = invoke(stypy.reporting.localization.Localization(__file__, 24, 7), is_iterable_15291, *[body_15292], **kwargs_15293)
    
    # Testing if the type of an if condition is none (line 24)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 24, 4), is_iterable_call_result_15294):
        
        # Assigning a List to a Attribute (line 27):
        
        # Obtaining an instance of the builtin type 'list' (line 27)
        list_15298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 27)
        # Adding element type (line 27)
        # Getting the type of 'body' (line 27)
        body_15299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), 'body')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 19), list_15298, body_15299)
        
        # Getting the type of 'if_' (line 27)
        if__15300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'if_')
        # Setting the type of the member 'body' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), if__15300, 'body', list_15298)
    else:
        
        # Testing the type of an if condition (line 24)
        if_condition_15295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 4), is_iterable_call_result_15294)
        # Assigning a type to the variable 'if_condition_15295' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'if_condition_15295', if_condition_15295)
        # SSA begins for if statement (line 24)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 25):
        # Getting the type of 'body' (line 25)
        body_15296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'body')
        # Getting the type of 'if_' (line 25)
        if__15297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'if_')
        # Setting the type of the member 'body' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), if__15297, 'body', body_15296)
        # SSA branch for the else part of an if statement (line 24)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Attribute (line 27):
        
        # Obtaining an instance of the builtin type 'list' (line 27)
        list_15298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 27)
        # Adding element type (line 27)
        # Getting the type of 'body' (line 27)
        body_15299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), 'body')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 19), list_15298, body_15299)
        
        # Getting the type of 'if_' (line 27)
        if__15300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'if_')
        # Setting the type of the member 'body' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), if__15300, 'body', list_15298)
        # SSA join for if statement (line 24)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to is_iterable(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'orelse' (line 29)
    orelse_15303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 40), 'orelse', False)
    # Processing the call keyword arguments (line 29)
    kwargs_15304 = {}
    # Getting the type of 'data_structures_copy' (line 29)
    data_structures_copy_15301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 7), 'data_structures_copy', False)
    # Obtaining the member 'is_iterable' of a type (line 29)
    is_iterable_15302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 7), data_structures_copy_15301, 'is_iterable')
    # Calling is_iterable(args, kwargs) (line 29)
    is_iterable_call_result_15305 = invoke(stypy.reporting.localization.Localization(__file__, 29, 7), is_iterable_15302, *[orelse_15303], **kwargs_15304)
    
    # Testing if the type of an if condition is none (line 29)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 29, 4), is_iterable_call_result_15305):
        
        # Assigning a List to a Attribute (line 32):
        
        # Obtaining an instance of the builtin type 'list' (line 32)
        list_15309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 32)
        # Adding element type (line 32)
        # Getting the type of 'orelse' (line 32)
        orelse_15310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), 'orelse')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 21), list_15309, orelse_15310)
        
        # Getting the type of 'if_' (line 32)
        if__15311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'if_')
        # Setting the type of the member 'orelse' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), if__15311, 'orelse', list_15309)
    else:
        
        # Testing the type of an if condition (line 29)
        if_condition_15306 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 4), is_iterable_call_result_15305)
        # Assigning a type to the variable 'if_condition_15306' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'if_condition_15306', if_condition_15306)
        # SSA begins for if statement (line 29)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 30):
        # Getting the type of 'orelse' (line 30)
        orelse_15307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 21), 'orelse')
        # Getting the type of 'if_' (line 30)
        if__15308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'if_')
        # Setting the type of the member 'orelse' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), if__15308, 'orelse', orelse_15307)
        # SSA branch for the else part of an if statement (line 29)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Attribute (line 32):
        
        # Obtaining an instance of the builtin type 'list' (line 32)
        list_15309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 32)
        # Adding element type (line 32)
        # Getting the type of 'orelse' (line 32)
        orelse_15310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), 'orelse')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 21), list_15309, orelse_15310)
        
        # Getting the type of 'if_' (line 32)
        if__15311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'if_')
        # Setting the type of the member 'orelse' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), if__15311, 'orelse', list_15309)
        # SSA join for if statement (line 29)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'if_' (line 34)
    if__15312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'if_')
    # Assigning a type to the variable 'stypy_return_type' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type', if__15312)
    
    # ################# End of 'create_if(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_if' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_15313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15313)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_if'
    return stypy_return_type_15313

# Assigning a type to the variable 'create_if' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'create_if', create_if)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
