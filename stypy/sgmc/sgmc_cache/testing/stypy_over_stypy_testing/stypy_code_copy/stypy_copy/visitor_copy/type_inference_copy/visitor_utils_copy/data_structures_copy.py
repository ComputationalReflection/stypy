
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from collections import Iterable
2: import ast
3: 
4: import core_language_copy
5: 
6: '''
7: Helper functions to create data structures related nodes in the type inference program AST tree
8: '''
9: 
10: 
11: # ######################################## COLLECTIONS HANDLING FUNCTIONS #############################################
12: 
13: 
14: def is_iterable(obj):
15:     '''
16:     Determines if the parameter is iterable
17:     :param obj: Any instance
18:     :return: Boolean value
19:     '''
20:     return isinstance(obj, Iterable)
21: 
22: 
23: def create_list(contents):
24:     list_node = ast.List(ctx=ast.Load())
25:     list_node.elts = contents
26: 
27:     return list_node
28: 
29: 
30: def create_keyword_dict(keywords):
31:     dict_node = ast.Dict(ctx=ast.Load(), keys=[], values=[])
32: 
33:     if keywords is not None:
34:         for elem in keywords:
35:             dict_node.keys.append(core_language.create_str(elem))
36:             dict_node.values.append(keywords[elem])
37: 
38:     return dict_node
39: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from collections import Iterable' statement (line 1)
try:
    from collections import Iterable

except:
    Iterable = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'collections', None, module_type_store, ['Iterable'], [Iterable])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import ast' statement (line 2)
import ast

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'ast', ast, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import core_language_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_15501 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'core_language_copy')

if (type(import_15501) is not StypyTypeError):

    if (import_15501 != 'pyd_module'):
        __import__(import_15501)
        sys_modules_15502 = sys.modules[import_15501]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'core_language_copy', sys_modules_15502.module_type_store, module_type_store)
    else:
        import core_language_copy

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'core_language_copy', core_language_copy, module_type_store)

else:
    # Assigning a type to the variable 'core_language_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'core_language_copy', import_15501)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

str_15503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', '\nHelper functions to create data structures related nodes in the type inference program AST tree\n')

@norecursion
def is_iterable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_iterable'
    module_type_store = module_type_store.open_function_context('is_iterable', 14, 0, False)
    
    # Passed parameters checking function
    is_iterable.stypy_localization = localization
    is_iterable.stypy_type_of_self = None
    is_iterable.stypy_type_store = module_type_store
    is_iterable.stypy_function_name = 'is_iterable'
    is_iterable.stypy_param_names_list = ['obj']
    is_iterable.stypy_varargs_param_name = None
    is_iterable.stypy_kwargs_param_name = None
    is_iterable.stypy_call_defaults = defaults
    is_iterable.stypy_call_varargs = varargs
    is_iterable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_iterable', ['obj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_iterable', localization, ['obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_iterable(...)' code ##################

    str_15504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', '\n    Determines if the parameter is iterable\n    :param obj: Any instance\n    :return: Boolean value\n    ')
    
    # Call to isinstance(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'obj' (line 20)
    obj_15506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'obj', False)
    # Getting the type of 'Iterable' (line 20)
    Iterable_15507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 27), 'Iterable', False)
    # Processing the call keyword arguments (line 20)
    kwargs_15508 = {}
    # Getting the type of 'isinstance' (line 20)
    isinstance_15505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 20)
    isinstance_call_result_15509 = invoke(stypy.reporting.localization.Localization(__file__, 20, 11), isinstance_15505, *[obj_15506, Iterable_15507], **kwargs_15508)
    
    # Assigning a type to the variable 'stypy_return_type' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type', isinstance_call_result_15509)
    
    # ################# End of 'is_iterable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_iterable' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_15510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15510)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_iterable'
    return stypy_return_type_15510

# Assigning a type to the variable 'is_iterable' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'is_iterable', is_iterable)

@norecursion
def create_list(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_list'
    module_type_store = module_type_store.open_function_context('create_list', 23, 0, False)
    
    # Passed parameters checking function
    create_list.stypy_localization = localization
    create_list.stypy_type_of_self = None
    create_list.stypy_type_store = module_type_store
    create_list.stypy_function_name = 'create_list'
    create_list.stypy_param_names_list = ['contents']
    create_list.stypy_varargs_param_name = None
    create_list.stypy_kwargs_param_name = None
    create_list.stypy_call_defaults = defaults
    create_list.stypy_call_varargs = varargs
    create_list.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_list', ['contents'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_list', localization, ['contents'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_list(...)' code ##################

    
    # Assigning a Call to a Name (line 24):
    
    # Call to List(...): (line 24)
    # Processing the call keyword arguments (line 24)
    
    # Call to Load(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_15515 = {}
    # Getting the type of 'ast' (line 24)
    ast_15513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'ast', False)
    # Obtaining the member 'Load' of a type (line 24)
    Load_15514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 29), ast_15513, 'Load')
    # Calling Load(args, kwargs) (line 24)
    Load_call_result_15516 = invoke(stypy.reporting.localization.Localization(__file__, 24, 29), Load_15514, *[], **kwargs_15515)
    
    keyword_15517 = Load_call_result_15516
    kwargs_15518 = {'ctx': keyword_15517}
    # Getting the type of 'ast' (line 24)
    ast_15511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'ast', False)
    # Obtaining the member 'List' of a type (line 24)
    List_15512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 16), ast_15511, 'List')
    # Calling List(args, kwargs) (line 24)
    List_call_result_15519 = invoke(stypy.reporting.localization.Localization(__file__, 24, 16), List_15512, *[], **kwargs_15518)
    
    # Assigning a type to the variable 'list_node' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'list_node', List_call_result_15519)
    
    # Assigning a Name to a Attribute (line 25):
    # Getting the type of 'contents' (line 25)
    contents_15520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 21), 'contents')
    # Getting the type of 'list_node' (line 25)
    list_node_15521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'list_node')
    # Setting the type of the member 'elts' of a type (line 25)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), list_node_15521, 'elts', contents_15520)
    # Getting the type of 'list_node' (line 27)
    list_node_15522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'list_node')
    # Assigning a type to the variable 'stypy_return_type' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type', list_node_15522)
    
    # ################# End of 'create_list(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_list' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_15523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15523)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_list'
    return stypy_return_type_15523

# Assigning a type to the variable 'create_list' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'create_list', create_list)

@norecursion
def create_keyword_dict(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_keyword_dict'
    module_type_store = module_type_store.open_function_context('create_keyword_dict', 30, 0, False)
    
    # Passed parameters checking function
    create_keyword_dict.stypy_localization = localization
    create_keyword_dict.stypy_type_of_self = None
    create_keyword_dict.stypy_type_store = module_type_store
    create_keyword_dict.stypy_function_name = 'create_keyword_dict'
    create_keyword_dict.stypy_param_names_list = ['keywords']
    create_keyword_dict.stypy_varargs_param_name = None
    create_keyword_dict.stypy_kwargs_param_name = None
    create_keyword_dict.stypy_call_defaults = defaults
    create_keyword_dict.stypy_call_varargs = varargs
    create_keyword_dict.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_keyword_dict', ['keywords'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_keyword_dict', localization, ['keywords'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_keyword_dict(...)' code ##################

    
    # Assigning a Call to a Name (line 31):
    
    # Call to Dict(...): (line 31)
    # Processing the call keyword arguments (line 31)
    
    # Call to Load(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_15528 = {}
    # Getting the type of 'ast' (line 31)
    ast_15526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 29), 'ast', False)
    # Obtaining the member 'Load' of a type (line 31)
    Load_15527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 29), ast_15526, 'Load')
    # Calling Load(args, kwargs) (line 31)
    Load_call_result_15529 = invoke(stypy.reporting.localization.Localization(__file__, 31, 29), Load_15527, *[], **kwargs_15528)
    
    keyword_15530 = Load_call_result_15529
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_15531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    
    keyword_15532 = list_15531
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_15533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 57), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    
    keyword_15534 = list_15533
    kwargs_15535 = {'keys': keyword_15532, 'values': keyword_15534, 'ctx': keyword_15530}
    # Getting the type of 'ast' (line 31)
    ast_15524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'ast', False)
    # Obtaining the member 'Dict' of a type (line 31)
    Dict_15525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 16), ast_15524, 'Dict')
    # Calling Dict(args, kwargs) (line 31)
    Dict_call_result_15536 = invoke(stypy.reporting.localization.Localization(__file__, 31, 16), Dict_15525, *[], **kwargs_15535)
    
    # Assigning a type to the variable 'dict_node' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'dict_node', Dict_call_result_15536)
    
    # Type idiom detected: calculating its left and rigth part (line 33)
    # Getting the type of 'keywords' (line 33)
    keywords_15537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'keywords')
    # Getting the type of 'None' (line 33)
    None_15538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 23), 'None')
    
    (may_be_15539, more_types_in_union_15540) = may_not_be_none(keywords_15537, None_15538)

    if may_be_15539:

        if more_types_in_union_15540:
            # Runtime conditional SSA (line 33)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'keywords' (line 34)
        keywords_15541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'keywords')
        # Assigning a type to the variable 'keywords_15541' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'keywords_15541', keywords_15541)
        # Testing if the for loop is going to be iterated (line 34)
        # Testing the type of a for loop iterable (line 34)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 8), keywords_15541)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 34, 8), keywords_15541):
            # Getting the type of the for loop variable (line 34)
            for_loop_var_15542 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 8), keywords_15541)
            # Assigning a type to the variable 'elem' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'elem', for_loop_var_15542)
            # SSA begins for a for statement (line 34)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 35)
            # Processing the call arguments (line 35)
            
            # Call to create_str(...): (line 35)
            # Processing the call arguments (line 35)
            # Getting the type of 'elem' (line 35)
            elem_15548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 59), 'elem', False)
            # Processing the call keyword arguments (line 35)
            kwargs_15549 = {}
            # Getting the type of 'core_language' (line 35)
            core_language_15546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'core_language', False)
            # Obtaining the member 'create_str' of a type (line 35)
            create_str_15547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 34), core_language_15546, 'create_str')
            # Calling create_str(args, kwargs) (line 35)
            create_str_call_result_15550 = invoke(stypy.reporting.localization.Localization(__file__, 35, 34), create_str_15547, *[elem_15548], **kwargs_15549)
            
            # Processing the call keyword arguments (line 35)
            kwargs_15551 = {}
            # Getting the type of 'dict_node' (line 35)
            dict_node_15543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'dict_node', False)
            # Obtaining the member 'keys' of a type (line 35)
            keys_15544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), dict_node_15543, 'keys')
            # Obtaining the member 'append' of a type (line 35)
            append_15545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), keys_15544, 'append')
            # Calling append(args, kwargs) (line 35)
            append_call_result_15552 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), append_15545, *[create_str_call_result_15550], **kwargs_15551)
            
            
            # Call to append(...): (line 36)
            # Processing the call arguments (line 36)
            
            # Obtaining the type of the subscript
            # Getting the type of 'elem' (line 36)
            elem_15556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 45), 'elem', False)
            # Getting the type of 'keywords' (line 36)
            keywords_15557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 36), 'keywords', False)
            # Obtaining the member '__getitem__' of a type (line 36)
            getitem___15558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 36), keywords_15557, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 36)
            subscript_call_result_15559 = invoke(stypy.reporting.localization.Localization(__file__, 36, 36), getitem___15558, elem_15556)
            
            # Processing the call keyword arguments (line 36)
            kwargs_15560 = {}
            # Getting the type of 'dict_node' (line 36)
            dict_node_15553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'dict_node', False)
            # Obtaining the member 'values' of a type (line 36)
            values_15554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), dict_node_15553, 'values')
            # Obtaining the member 'append' of a type (line 36)
            append_15555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), values_15554, 'append')
            # Calling append(args, kwargs) (line 36)
            append_call_result_15561 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), append_15555, *[subscript_call_result_15559], **kwargs_15560)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        

        if more_types_in_union_15540:
            # SSA join for if statement (line 33)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'dict_node' (line 38)
    dict_node_15562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'dict_node')
    # Assigning a type to the variable 'stypy_return_type' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type', dict_node_15562)
    
    # ################# End of 'create_keyword_dict(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_keyword_dict' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_15563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15563)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_keyword_dict'
    return stypy_return_type_15563

# Assigning a type to the variable 'create_keyword_dict' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'create_keyword_dict', create_keyword_dict)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
