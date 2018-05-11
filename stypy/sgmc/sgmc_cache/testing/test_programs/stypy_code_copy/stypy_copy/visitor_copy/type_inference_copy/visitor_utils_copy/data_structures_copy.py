
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
35:             dict_node.keys.append(core_language_copy.create_str(elem))
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
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_31066 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'core_language_copy')

if (type(import_31066) is not StypyTypeError):

    if (import_31066 != 'pyd_module'):
        __import__(import_31066)
        sys_modules_31067 = sys.modules[import_31066]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'core_language_copy', sys_modules_31067.module_type_store, module_type_store)
    else:
        import core_language_copy

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'core_language_copy', core_language_copy, module_type_store)

else:
    # Assigning a type to the variable 'core_language_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'core_language_copy', import_31066)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

str_31068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', '\nHelper functions to create data structures related nodes in the type inference program AST tree\n')

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

    str_31069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', '\n    Determines if the parameter is iterable\n    :param obj: Any instance\n    :return: Boolean value\n    ')
    
    # Call to isinstance(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'obj' (line 20)
    obj_31071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'obj', False)
    # Getting the type of 'Iterable' (line 20)
    Iterable_31072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 27), 'Iterable', False)
    # Processing the call keyword arguments (line 20)
    kwargs_31073 = {}
    # Getting the type of 'isinstance' (line 20)
    isinstance_31070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 20)
    isinstance_call_result_31074 = invoke(stypy.reporting.localization.Localization(__file__, 20, 11), isinstance_31070, *[obj_31071, Iterable_31072], **kwargs_31073)
    
    # Assigning a type to the variable 'stypy_return_type' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type', isinstance_call_result_31074)
    
    # ################# End of 'is_iterable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_iterable' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_31075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31075)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_iterable'
    return stypy_return_type_31075

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
    kwargs_31080 = {}
    # Getting the type of 'ast' (line 24)
    ast_31078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'ast', False)
    # Obtaining the member 'Load' of a type (line 24)
    Load_31079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 29), ast_31078, 'Load')
    # Calling Load(args, kwargs) (line 24)
    Load_call_result_31081 = invoke(stypy.reporting.localization.Localization(__file__, 24, 29), Load_31079, *[], **kwargs_31080)
    
    keyword_31082 = Load_call_result_31081
    kwargs_31083 = {'ctx': keyword_31082}
    # Getting the type of 'ast' (line 24)
    ast_31076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'ast', False)
    # Obtaining the member 'List' of a type (line 24)
    List_31077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 16), ast_31076, 'List')
    # Calling List(args, kwargs) (line 24)
    List_call_result_31084 = invoke(stypy.reporting.localization.Localization(__file__, 24, 16), List_31077, *[], **kwargs_31083)
    
    # Assigning a type to the variable 'list_node' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'list_node', List_call_result_31084)
    
    # Assigning a Name to a Attribute (line 25):
    # Getting the type of 'contents' (line 25)
    contents_31085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 21), 'contents')
    # Getting the type of 'list_node' (line 25)
    list_node_31086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'list_node')
    # Setting the type of the member 'elts' of a type (line 25)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), list_node_31086, 'elts', contents_31085)
    # Getting the type of 'list_node' (line 27)
    list_node_31087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'list_node')
    # Assigning a type to the variable 'stypy_return_type' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type', list_node_31087)
    
    # ################# End of 'create_list(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_list' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_31088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31088)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_list'
    return stypy_return_type_31088

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
    kwargs_31093 = {}
    # Getting the type of 'ast' (line 31)
    ast_31091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 29), 'ast', False)
    # Obtaining the member 'Load' of a type (line 31)
    Load_31092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 29), ast_31091, 'Load')
    # Calling Load(args, kwargs) (line 31)
    Load_call_result_31094 = invoke(stypy.reporting.localization.Localization(__file__, 31, 29), Load_31092, *[], **kwargs_31093)
    
    keyword_31095 = Load_call_result_31094
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_31096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    
    keyword_31097 = list_31096
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_31098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 57), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    
    keyword_31099 = list_31098
    kwargs_31100 = {'keys': keyword_31097, 'values': keyword_31099, 'ctx': keyword_31095}
    # Getting the type of 'ast' (line 31)
    ast_31089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'ast', False)
    # Obtaining the member 'Dict' of a type (line 31)
    Dict_31090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 16), ast_31089, 'Dict')
    # Calling Dict(args, kwargs) (line 31)
    Dict_call_result_31101 = invoke(stypy.reporting.localization.Localization(__file__, 31, 16), Dict_31090, *[], **kwargs_31100)
    
    # Assigning a type to the variable 'dict_node' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'dict_node', Dict_call_result_31101)
    
    # Type idiom detected: calculating its left and rigth part (line 33)
    # Getting the type of 'keywords' (line 33)
    keywords_31102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'keywords')
    # Getting the type of 'None' (line 33)
    None_31103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 23), 'None')
    
    (may_be_31104, more_types_in_union_31105) = may_not_be_none(keywords_31102, None_31103)

    if may_be_31104:

        if more_types_in_union_31105:
            # Runtime conditional SSA (line 33)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'keywords' (line 34)
        keywords_31106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'keywords')
        # Assigning a type to the variable 'keywords_31106' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'keywords_31106', keywords_31106)
        # Testing if the for loop is going to be iterated (line 34)
        # Testing the type of a for loop iterable (line 34)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 8), keywords_31106)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 34, 8), keywords_31106):
            # Getting the type of the for loop variable (line 34)
            for_loop_var_31107 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 8), keywords_31106)
            # Assigning a type to the variable 'elem' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'elem', for_loop_var_31107)
            # SSA begins for a for statement (line 34)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 35)
            # Processing the call arguments (line 35)
            
            # Call to create_str(...): (line 35)
            # Processing the call arguments (line 35)
            # Getting the type of 'elem' (line 35)
            elem_31113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 64), 'elem', False)
            # Processing the call keyword arguments (line 35)
            kwargs_31114 = {}
            # Getting the type of 'core_language_copy' (line 35)
            core_language_copy_31111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'core_language_copy', False)
            # Obtaining the member 'create_str' of a type (line 35)
            create_str_31112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 34), core_language_copy_31111, 'create_str')
            # Calling create_str(args, kwargs) (line 35)
            create_str_call_result_31115 = invoke(stypy.reporting.localization.Localization(__file__, 35, 34), create_str_31112, *[elem_31113], **kwargs_31114)
            
            # Processing the call keyword arguments (line 35)
            kwargs_31116 = {}
            # Getting the type of 'dict_node' (line 35)
            dict_node_31108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'dict_node', False)
            # Obtaining the member 'keys' of a type (line 35)
            keys_31109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), dict_node_31108, 'keys')
            # Obtaining the member 'append' of a type (line 35)
            append_31110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), keys_31109, 'append')
            # Calling append(args, kwargs) (line 35)
            append_call_result_31117 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), append_31110, *[create_str_call_result_31115], **kwargs_31116)
            
            
            # Call to append(...): (line 36)
            # Processing the call arguments (line 36)
            
            # Obtaining the type of the subscript
            # Getting the type of 'elem' (line 36)
            elem_31121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 45), 'elem', False)
            # Getting the type of 'keywords' (line 36)
            keywords_31122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 36), 'keywords', False)
            # Obtaining the member '__getitem__' of a type (line 36)
            getitem___31123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 36), keywords_31122, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 36)
            subscript_call_result_31124 = invoke(stypy.reporting.localization.Localization(__file__, 36, 36), getitem___31123, elem_31121)
            
            # Processing the call keyword arguments (line 36)
            kwargs_31125 = {}
            # Getting the type of 'dict_node' (line 36)
            dict_node_31118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'dict_node', False)
            # Obtaining the member 'values' of a type (line 36)
            values_31119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), dict_node_31118, 'values')
            # Obtaining the member 'append' of a type (line 36)
            append_31120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), values_31119, 'append')
            # Calling append(args, kwargs) (line 36)
            append_call_result_31126 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), append_31120, *[subscript_call_result_31124], **kwargs_31125)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        

        if more_types_in_union_31105:
            # SSA join for if statement (line 33)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'dict_node' (line 38)
    dict_node_31127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'dict_node')
    # Assigning a type to the variable 'stypy_return_type' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type', dict_node_31127)
    
    # ################# End of 'create_keyword_dict(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_keyword_dict' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_31128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31128)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_keyword_dict'
    return stypy_return_type_31128

# Assigning a type to the variable 'create_keyword_dict' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'create_keyword_dict', create_keyword_dict)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
