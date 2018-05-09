
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import *
2: from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy
3: import ast
4: 
5: class ClassInitVisitor(ast.NodeTransformer):
6:     '''
7:     This transformer ensures that every declared class has an __init__ method. If the original class do not declare
8:     one, an empty one is added. This is needed to avoid code generation errors
9:     '''
10: 
11:     def __search_init_method(self, node):
12:         if hasattr(node, 'name'):
13:             return node.name == '__init__'
14:         else:
15:             return False
16: 
17:     def visit_ClassDef(self, node):
18:         # Test if we have an __init__ method
19:         init_method = filter(lambda n: self.__search_init_method(n), node.body)
20:         if len(init_method) > 0:
21:             return node
22: 
23:         # If no __init__ method is declared, declare an empty one
24:         function_def_arguments = ast.arguments()
25: 
26:         function_def_arguments.args = [core_language_copy.create_Name('self')]
27: 
28:         function_def = ast.FunctionDef()
29:         function_def.lineno = node.lineno
30:         function_def.col_offset = node.col_offset
31:         function_def.name = '__init__'
32: 
33:         function_def.args = function_def_arguments
34:         function_def_arguments.kwarg = None
35:         function_def_arguments.vararg = None
36:         function_def_arguments.defaults = []
37:         function_def.decorator_list = []
38: 
39:         function_def.body = []
40: 
41:         function_def.body.append(ast.Pass())
42: 
43:         node.body.append(function_def)
44: 
45:         return node
46: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import ' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')
import_13683 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy')

if (type(import_13683) is not StypyTypeError):

    if (import_13683 != 'pyd_module'):
        __import__(import_13683)
        sys_modules_13684 = sys.modules[import_13683]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', sys_modules_13684.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_13684, sys_modules_13684.module_type_store, module_type_store)
    else:
        from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', import_13683)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')
import_13685 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy')

if (type(import_13685) is not StypyTypeError):

    if (import_13685 != 'pyd_module'):
        __import__(import_13685)
        sys_modules_13686 = sys.modules[import_13685]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', sys_modules_13686.module_type_store, module_type_store, ['core_language_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_13686, sys_modules_13686.module_type_store, module_type_store)
    else:
        from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', None, module_type_store, ['core_language_copy'], [core_language_copy])

else:
    # Assigning a type to the variable 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', import_13685)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import ast' statement (line 3)
import ast

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'ast', ast, module_type_store)

# Declaration of the 'ClassInitVisitor' class
# Getting the type of 'ast' (line 5)
ast_13687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 23), 'ast')
# Obtaining the member 'NodeTransformer' of a type (line 5)
NodeTransformer_13688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 23), ast_13687, 'NodeTransformer')

class ClassInitVisitor(NodeTransformer_13688, ):
    str_13689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, (-1)), 'str', '\n    This transformer ensures that every declared class has an __init__ method. If the original class do not declare\n    one, an empty one is added. This is needed to avoid code generation errors\n    ')

    @norecursion
    def __search_init_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__search_init_method'
        module_type_store = module_type_store.open_function_context('__search_init_method', 11, 4, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ClassInitVisitor.__search_init_method.__dict__.__setitem__('stypy_localization', localization)
        ClassInitVisitor.__search_init_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ClassInitVisitor.__search_init_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        ClassInitVisitor.__search_init_method.__dict__.__setitem__('stypy_function_name', 'ClassInitVisitor.__search_init_method')
        ClassInitVisitor.__search_init_method.__dict__.__setitem__('stypy_param_names_list', ['node'])
        ClassInitVisitor.__search_init_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        ClassInitVisitor.__search_init_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ClassInitVisitor.__search_init_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        ClassInitVisitor.__search_init_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        ClassInitVisitor.__search_init_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ClassInitVisitor.__search_init_method.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ClassInitVisitor.__search_init_method', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__search_init_method', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__search_init_method(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 12)
        str_13690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 25), 'str', 'name')
        # Getting the type of 'node' (line 12)
        node_13691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 19), 'node')
        
        (may_be_13692, more_types_in_union_13693) = may_provide_member(str_13690, node_13691)

        if may_be_13692:

            if more_types_in_union_13693:
                # Runtime conditional SSA (line 12)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'node' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'node', remove_not_member_provider_from_union(node_13691, 'name'))
            
            # Getting the type of 'node' (line 13)
            node_13694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 19), 'node')
            # Obtaining the member 'name' of a type (line 13)
            name_13695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 19), node_13694, 'name')
            str_13696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 32), 'str', '__init__')
            # Applying the binary operator '==' (line 13)
            result_eq_13697 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 19), '==', name_13695, str_13696)
            
            # Assigning a type to the variable 'stypy_return_type' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'stypy_return_type', result_eq_13697)

            if more_types_in_union_13693:
                # Runtime conditional SSA for else branch (line 12)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_13692) or more_types_in_union_13693):
            # Assigning a type to the variable 'node' (line 12)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'node', remove_member_provider_from_union(node_13691, 'name'))
            # Getting the type of 'False' (line 15)
            False_13698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'stypy_return_type', False_13698)

            if (may_be_13692 and more_types_in_union_13693):
                # SSA join for if statement (line 12)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '__search_init_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__search_init_method' in the type store
        # Getting the type of 'stypy_return_type' (line 11)
        stypy_return_type_13699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13699)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__search_init_method'
        return stypy_return_type_13699


    @norecursion
    def visit_ClassDef(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_ClassDef'
        module_type_store = module_type_store.open_function_context('visit_ClassDef', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ClassInitVisitor.visit_ClassDef.__dict__.__setitem__('stypy_localization', localization)
        ClassInitVisitor.visit_ClassDef.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ClassInitVisitor.visit_ClassDef.__dict__.__setitem__('stypy_type_store', module_type_store)
        ClassInitVisitor.visit_ClassDef.__dict__.__setitem__('stypy_function_name', 'ClassInitVisitor.visit_ClassDef')
        ClassInitVisitor.visit_ClassDef.__dict__.__setitem__('stypy_param_names_list', ['node'])
        ClassInitVisitor.visit_ClassDef.__dict__.__setitem__('stypy_varargs_param_name', None)
        ClassInitVisitor.visit_ClassDef.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ClassInitVisitor.visit_ClassDef.__dict__.__setitem__('stypy_call_defaults', defaults)
        ClassInitVisitor.visit_ClassDef.__dict__.__setitem__('stypy_call_varargs', varargs)
        ClassInitVisitor.visit_ClassDef.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ClassInitVisitor.visit_ClassDef.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ClassInitVisitor.visit_ClassDef', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_ClassDef', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_ClassDef(...)' code ##################

        
        # Assigning a Call to a Name (line 19):
        
        # Call to filter(...): (line 19)
        # Processing the call arguments (line 19)

        @norecursion
        def _stypy_temp_lambda_19(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_19'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_19', 19, 29, True)
            # Passed parameters checking function
            _stypy_temp_lambda_19.stypy_localization = localization
            _stypy_temp_lambda_19.stypy_type_of_self = None
            _stypy_temp_lambda_19.stypy_type_store = module_type_store
            _stypy_temp_lambda_19.stypy_function_name = '_stypy_temp_lambda_19'
            _stypy_temp_lambda_19.stypy_param_names_list = ['n']
            _stypy_temp_lambda_19.stypy_varargs_param_name = None
            _stypy_temp_lambda_19.stypy_kwargs_param_name = None
            _stypy_temp_lambda_19.stypy_call_defaults = defaults
            _stypy_temp_lambda_19.stypy_call_varargs = varargs
            _stypy_temp_lambda_19.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_19', ['n'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_19', ['n'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to __search_init_method(...): (line 19)
            # Processing the call arguments (line 19)
            # Getting the type of 'n' (line 19)
            n_13703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 65), 'n', False)
            # Processing the call keyword arguments (line 19)
            kwargs_13704 = {}
            # Getting the type of 'self' (line 19)
            self_13701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 39), 'self', False)
            # Obtaining the member '__search_init_method' of a type (line 19)
            search_init_method_13702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 39), self_13701, '__search_init_method')
            # Calling __search_init_method(args, kwargs) (line 19)
            search_init_method_call_result_13705 = invoke(stypy.reporting.localization.Localization(__file__, 19, 39), search_init_method_13702, *[n_13703], **kwargs_13704)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 29), 'stypy_return_type', search_init_method_call_result_13705)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_19' in the type store
            # Getting the type of 'stypy_return_type' (line 19)
            stypy_return_type_13706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 29), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_13706)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_19'
            return stypy_return_type_13706

        # Assigning a type to the variable '_stypy_temp_lambda_19' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 29), '_stypy_temp_lambda_19', _stypy_temp_lambda_19)
        # Getting the type of '_stypy_temp_lambda_19' (line 19)
        _stypy_temp_lambda_19_13707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 29), '_stypy_temp_lambda_19')
        # Getting the type of 'node' (line 19)
        node_13708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 69), 'node', False)
        # Obtaining the member 'body' of a type (line 19)
        body_13709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 69), node_13708, 'body')
        # Processing the call keyword arguments (line 19)
        kwargs_13710 = {}
        # Getting the type of 'filter' (line 19)
        filter_13700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'filter', False)
        # Calling filter(args, kwargs) (line 19)
        filter_call_result_13711 = invoke(stypy.reporting.localization.Localization(__file__, 19, 22), filter_13700, *[_stypy_temp_lambda_19_13707, body_13709], **kwargs_13710)
        
        # Assigning a type to the variable 'init_method' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'init_method', filter_call_result_13711)
        
        
        # Call to len(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'init_method' (line 20)
        init_method_13713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'init_method', False)
        # Processing the call keyword arguments (line 20)
        kwargs_13714 = {}
        # Getting the type of 'len' (line 20)
        len_13712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'len', False)
        # Calling len(args, kwargs) (line 20)
        len_call_result_13715 = invoke(stypy.reporting.localization.Localization(__file__, 20, 11), len_13712, *[init_method_13713], **kwargs_13714)
        
        int_13716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 30), 'int')
        # Applying the binary operator '>' (line 20)
        result_gt_13717 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 11), '>', len_call_result_13715, int_13716)
        
        # Testing if the type of an if condition is none (line 20)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 20, 8), result_gt_13717):
            pass
        else:
            
            # Testing the type of an if condition (line 20)
            if_condition_13718 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 8), result_gt_13717)
            # Assigning a type to the variable 'if_condition_13718' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'if_condition_13718', if_condition_13718)
            # SSA begins for if statement (line 20)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'node' (line 21)
            node_13719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'node')
            # Assigning a type to the variable 'stypy_return_type' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'stypy_return_type', node_13719)
            # SSA join for if statement (line 20)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 24):
        
        # Call to arguments(...): (line 24)
        # Processing the call keyword arguments (line 24)
        kwargs_13722 = {}
        # Getting the type of 'ast' (line 24)
        ast_13720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 33), 'ast', False)
        # Obtaining the member 'arguments' of a type (line 24)
        arguments_13721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 33), ast_13720, 'arguments')
        # Calling arguments(args, kwargs) (line 24)
        arguments_call_result_13723 = invoke(stypy.reporting.localization.Localization(__file__, 24, 33), arguments_13721, *[], **kwargs_13722)
        
        # Assigning a type to the variable 'function_def_arguments' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'function_def_arguments', arguments_call_result_13723)
        
        # Assigning a List to a Attribute (line 26):
        
        # Obtaining an instance of the builtin type 'list' (line 26)
        list_13724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 26)
        # Adding element type (line 26)
        
        # Call to create_Name(...): (line 26)
        # Processing the call arguments (line 26)
        str_13727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 70), 'str', 'self')
        # Processing the call keyword arguments (line 26)
        kwargs_13728 = {}
        # Getting the type of 'core_language_copy' (line 26)
        core_language_copy_13725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 39), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 26)
        create_Name_13726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 39), core_language_copy_13725, 'create_Name')
        # Calling create_Name(args, kwargs) (line 26)
        create_Name_call_result_13729 = invoke(stypy.reporting.localization.Localization(__file__, 26, 39), create_Name_13726, *[str_13727], **kwargs_13728)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 38), list_13724, create_Name_call_result_13729)
        
        # Getting the type of 'function_def_arguments' (line 26)
        function_def_arguments_13730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'function_def_arguments')
        # Setting the type of the member 'args' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), function_def_arguments_13730, 'args', list_13724)
        
        # Assigning a Call to a Name (line 28):
        
        # Call to FunctionDef(...): (line 28)
        # Processing the call keyword arguments (line 28)
        kwargs_13733 = {}
        # Getting the type of 'ast' (line 28)
        ast_13731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'ast', False)
        # Obtaining the member 'FunctionDef' of a type (line 28)
        FunctionDef_13732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 23), ast_13731, 'FunctionDef')
        # Calling FunctionDef(args, kwargs) (line 28)
        FunctionDef_call_result_13734 = invoke(stypy.reporting.localization.Localization(__file__, 28, 23), FunctionDef_13732, *[], **kwargs_13733)
        
        # Assigning a type to the variable 'function_def' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'function_def', FunctionDef_call_result_13734)
        
        # Assigning a Attribute to a Attribute (line 29):
        # Getting the type of 'node' (line 29)
        node_13735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 30), 'node')
        # Obtaining the member 'lineno' of a type (line 29)
        lineno_13736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 30), node_13735, 'lineno')
        # Getting the type of 'function_def' (line 29)
        function_def_13737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'function_def')
        # Setting the type of the member 'lineno' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), function_def_13737, 'lineno', lineno_13736)
        
        # Assigning a Attribute to a Attribute (line 30):
        # Getting the type of 'node' (line 30)
        node_13738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 34), 'node')
        # Obtaining the member 'col_offset' of a type (line 30)
        col_offset_13739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 34), node_13738, 'col_offset')
        # Getting the type of 'function_def' (line 30)
        function_def_13740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'function_def')
        # Setting the type of the member 'col_offset' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), function_def_13740, 'col_offset', col_offset_13739)
        
        # Assigning a Str to a Attribute (line 31):
        str_13741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 28), 'str', '__init__')
        # Getting the type of 'function_def' (line 31)
        function_def_13742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'function_def')
        # Setting the type of the member 'name' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), function_def_13742, 'name', str_13741)
        
        # Assigning a Name to a Attribute (line 33):
        # Getting the type of 'function_def_arguments' (line 33)
        function_def_arguments_13743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 28), 'function_def_arguments')
        # Getting the type of 'function_def' (line 33)
        function_def_13744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'function_def')
        # Setting the type of the member 'args' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), function_def_13744, 'args', function_def_arguments_13743)
        
        # Assigning a Name to a Attribute (line 34):
        # Getting the type of 'None' (line 34)
        None_13745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 39), 'None')
        # Getting the type of 'function_def_arguments' (line 34)
        function_def_arguments_13746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'function_def_arguments')
        # Setting the type of the member 'kwarg' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), function_def_arguments_13746, 'kwarg', None_13745)
        
        # Assigning a Name to a Attribute (line 35):
        # Getting the type of 'None' (line 35)
        None_13747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 40), 'None')
        # Getting the type of 'function_def_arguments' (line 35)
        function_def_arguments_13748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'function_def_arguments')
        # Setting the type of the member 'vararg' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), function_def_arguments_13748, 'vararg', None_13747)
        
        # Assigning a List to a Attribute (line 36):
        
        # Obtaining an instance of the builtin type 'list' (line 36)
        list_13749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 36)
        
        # Getting the type of 'function_def_arguments' (line 36)
        function_def_arguments_13750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'function_def_arguments')
        # Setting the type of the member 'defaults' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), function_def_arguments_13750, 'defaults', list_13749)
        
        # Assigning a List to a Attribute (line 37):
        
        # Obtaining an instance of the builtin type 'list' (line 37)
        list_13751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 37)
        
        # Getting the type of 'function_def' (line 37)
        function_def_13752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'function_def')
        # Setting the type of the member 'decorator_list' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), function_def_13752, 'decorator_list', list_13751)
        
        # Assigning a List to a Attribute (line 39):
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_13753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        
        # Getting the type of 'function_def' (line 39)
        function_def_13754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'function_def')
        # Setting the type of the member 'body' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), function_def_13754, 'body', list_13753)
        
        # Call to append(...): (line 41)
        # Processing the call arguments (line 41)
        
        # Call to Pass(...): (line 41)
        # Processing the call keyword arguments (line 41)
        kwargs_13760 = {}
        # Getting the type of 'ast' (line 41)
        ast_13758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 33), 'ast', False)
        # Obtaining the member 'Pass' of a type (line 41)
        Pass_13759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 33), ast_13758, 'Pass')
        # Calling Pass(args, kwargs) (line 41)
        Pass_call_result_13761 = invoke(stypy.reporting.localization.Localization(__file__, 41, 33), Pass_13759, *[], **kwargs_13760)
        
        # Processing the call keyword arguments (line 41)
        kwargs_13762 = {}
        # Getting the type of 'function_def' (line 41)
        function_def_13755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'function_def', False)
        # Obtaining the member 'body' of a type (line 41)
        body_13756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), function_def_13755, 'body')
        # Obtaining the member 'append' of a type (line 41)
        append_13757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), body_13756, 'append')
        # Calling append(args, kwargs) (line 41)
        append_call_result_13763 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), append_13757, *[Pass_call_result_13761], **kwargs_13762)
        
        
        # Call to append(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'function_def' (line 43)
        function_def_13767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 25), 'function_def', False)
        # Processing the call keyword arguments (line 43)
        kwargs_13768 = {}
        # Getting the type of 'node' (line 43)
        node_13764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'node', False)
        # Obtaining the member 'body' of a type (line 43)
        body_13765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), node_13764, 'body')
        # Obtaining the member 'append' of a type (line 43)
        append_13766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), body_13765, 'append')
        # Calling append(args, kwargs) (line 43)
        append_call_result_13769 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), append_13766, *[function_def_13767], **kwargs_13768)
        
        # Getting the type of 'node' (line 45)
        node_13770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', node_13770)
        
        # ################# End of 'visit_ClassDef(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ClassDef' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_13771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13771)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ClassDef'
        return stypy_return_type_13771


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 5, 0, False)
        # Assigning a type to the variable 'self' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ClassInitVisitor.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ClassInitVisitor' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'ClassInitVisitor', ClassInitVisitor)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
