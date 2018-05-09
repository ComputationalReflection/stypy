
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import *
2: from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, stypy_functions_copy
3: import ast
4: 
5: class ClassAttributesVisitor(ast.NodeTransformer):
6:     '''
7:     This desugaring visitor converts class-bound attributes such as:
8: 
9:     class Example:
10:         att = "hello"
11:         <rest of the members that are not attributes>
12: 
13:     into this equivalent form:
14: 
15:     class Example:
16:         <rest of the members that are not attributes>
17: 
18:     Example.att = "hello"
19: 
20:     The first form cannot be properly processed by stypy due to limitations in the way they are transformed into AST
21:     nodes. The second form is completely processable using the same assignment processing code we already have.
22: 
23:     '''
24: 
25:     @staticmethod
26:     def __extract_attribute_attached_comments(attr, node):
27:         attr_index = node.body.index(attr)
28:         separator_comment = None
29:         comment = None
30: 
31:         for i in range(attr_index):
32:             if stypy_functions_copy.is_blank_line(node.body[i]):
33:                 separator_comment = node.body[i]
34:             if stypy_functions_copy.is_src_comment(node.body[i]):
35:                 comment = node.body[i]
36:                 comment.value.id = comment.value.id.replace("# Assignment", "# Class-bound assignment")
37: 
38:         return separator_comment, comment
39: 
40:     def visit_ClassDef(self, node):
41:         class_attributes = filter(lambda element: isinstance(element, ast.Assign), node.body)
42: 
43:         attr_stmts = []
44:         for attr in class_attributes:
45:             separator_comment, comment = self.__extract_attribute_attached_comments(attr, node)
46:             if separator_comment is not None:
47:                 node.body.remove(separator_comment)
48:                 attr_stmts.append(separator_comment)
49: 
50:             if separator_comment is not None:
51:                 node.body.remove(comment)
52:                 attr_stmts.append(comment)
53: 
54:             node.body.remove(attr)
55: 
56:             temp_class_attr = core_language_copy.create_attribute(node.name, attr.targets[0].id)
57:             if len(filter(lambda class_attr: class_attr.targets[0] == attr.value, class_attributes)) == 0:
58:                 attr_stmts.append(core_language_copy.create_Assign(temp_class_attr, attr.value))
59:             else:
60:                 temp_class_value = core_language_copy.create_attribute(node.name, attr.value.id)
61:                 attr_stmts.append(core_language_copy.create_Assign(temp_class_attr, temp_class_value))
62: 
63:         # Extracting all attributes from a class may leave the program in an incorrect state if all the members in the
64:         # class are attributes. An empty class body is an error, we add a pass node in that special case
65:         if len(node.body) == 0:
66:             node.body.append(stypy_functions_copy.create_pass_node())
67: 
68:         return stypy_functions_copy.flatten_lists(node, attr_stmts)
69: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import ' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')
import_13469 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy')

if (type(import_13469) is not StypyTypeError):

    if (import_13469 != 'pyd_module'):
        __import__(import_13469)
        sys_modules_13470 = sys.modules[import_13469]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', sys_modules_13470.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_13470, sys_modules_13470.module_type_store, module_type_store)
    else:
        from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', import_13469)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, stypy_functions_copy' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')
import_13471 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy')

if (type(import_13471) is not StypyTypeError):

    if (import_13471 != 'pyd_module'):
        __import__(import_13471)
        sys_modules_13472 = sys.modules[import_13471]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', sys_modules_13472.module_type_store, module_type_store, ['core_language_copy', 'stypy_functions_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_13472, sys_modules_13472.module_type_store, module_type_store)
    else:
        from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, stypy_functions_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', None, module_type_store, ['core_language_copy', 'stypy_functions_copy'], [core_language_copy, stypy_functions_copy])

else:
    # Assigning a type to the variable 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', import_13471)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import ast' statement (line 3)
import ast

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'ast', ast, module_type_store)

# Declaration of the 'ClassAttributesVisitor' class
# Getting the type of 'ast' (line 5)
ast_13473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 29), 'ast')
# Obtaining the member 'NodeTransformer' of a type (line 5)
NodeTransformer_13474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 29), ast_13473, 'NodeTransformer')

class ClassAttributesVisitor(NodeTransformer_13474, ):
    str_13475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, (-1)), 'str', '\n    This desugaring visitor converts class-bound attributes such as:\n\n    class Example:\n        att = "hello"\n        <rest of the members that are not attributes>\n\n    into this equivalent form:\n\n    class Example:\n        <rest of the members that are not attributes>\n\n    Example.att = "hello"\n\n    The first form cannot be properly processed by stypy due to limitations in the way they are transformed into AST\n    nodes. The second form is completely processable using the same assignment processing code we already have.\n\n    ')

    @staticmethod
    @norecursion
    def __extract_attribute_attached_comments(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__extract_attribute_attached_comments'
        module_type_store = module_type_store.open_function_context('__extract_attribute_attached_comments', 25, 4, False)
        
        # Passed parameters checking function
        ClassAttributesVisitor.__extract_attribute_attached_comments.__dict__.__setitem__('stypy_localization', localization)
        ClassAttributesVisitor.__extract_attribute_attached_comments.__dict__.__setitem__('stypy_type_of_self', None)
        ClassAttributesVisitor.__extract_attribute_attached_comments.__dict__.__setitem__('stypy_type_store', module_type_store)
        ClassAttributesVisitor.__extract_attribute_attached_comments.__dict__.__setitem__('stypy_function_name', '__extract_attribute_attached_comments')
        ClassAttributesVisitor.__extract_attribute_attached_comments.__dict__.__setitem__('stypy_param_names_list', ['attr', 'node'])
        ClassAttributesVisitor.__extract_attribute_attached_comments.__dict__.__setitem__('stypy_varargs_param_name', None)
        ClassAttributesVisitor.__extract_attribute_attached_comments.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ClassAttributesVisitor.__extract_attribute_attached_comments.__dict__.__setitem__('stypy_call_defaults', defaults)
        ClassAttributesVisitor.__extract_attribute_attached_comments.__dict__.__setitem__('stypy_call_varargs', varargs)
        ClassAttributesVisitor.__extract_attribute_attached_comments.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ClassAttributesVisitor.__extract_attribute_attached_comments.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, '__extract_attribute_attached_comments', ['attr', 'node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__extract_attribute_attached_comments', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__extract_attribute_attached_comments(...)' code ##################

        
        # Assigning a Call to a Name (line 27):
        
        # Assigning a Call to a Name (line 27):
        
        # Call to index(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'attr' (line 27)
        attr_13479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 37), 'attr', False)
        # Processing the call keyword arguments (line 27)
        kwargs_13480 = {}
        # Getting the type of 'node' (line 27)
        node_13476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'node', False)
        # Obtaining the member 'body' of a type (line 27)
        body_13477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 21), node_13476, 'body')
        # Obtaining the member 'index' of a type (line 27)
        index_13478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 21), body_13477, 'index')
        # Calling index(args, kwargs) (line 27)
        index_call_result_13481 = invoke(stypy.reporting.localization.Localization(__file__, 27, 21), index_13478, *[attr_13479], **kwargs_13480)
        
        # Assigning a type to the variable 'attr_index' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'attr_index', index_call_result_13481)
        
        # Assigning a Name to a Name (line 28):
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'None' (line 28)
        None_13482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 28), 'None')
        # Assigning a type to the variable 'separator_comment' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'separator_comment', None_13482)
        
        # Assigning a Name to a Name (line 29):
        
        # Assigning a Name to a Name (line 29):
        # Getting the type of 'None' (line 29)
        None_13483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'None')
        # Assigning a type to the variable 'comment' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'comment', None_13483)
        
        
        # Call to range(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'attr_index' (line 31)
        attr_index_13485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'attr_index', False)
        # Processing the call keyword arguments (line 31)
        kwargs_13486 = {}
        # Getting the type of 'range' (line 31)
        range_13484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'range', False)
        # Calling range(args, kwargs) (line 31)
        range_call_result_13487 = invoke(stypy.reporting.localization.Localization(__file__, 31, 17), range_13484, *[attr_index_13485], **kwargs_13486)
        
        # Assigning a type to the variable 'range_call_result_13487' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'range_call_result_13487', range_call_result_13487)
        # Testing if the for loop is going to be iterated (line 31)
        # Testing the type of a for loop iterable (line 31)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 31, 8), range_call_result_13487)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 31, 8), range_call_result_13487):
            # Getting the type of the for loop variable (line 31)
            for_loop_var_13488 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 31, 8), range_call_result_13487)
            # Assigning a type to the variable 'i' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'i', for_loop_var_13488)
            # SSA begins for a for statement (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to is_blank_line(...): (line 32)
            # Processing the call arguments (line 32)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 32)
            i_13491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 60), 'i', False)
            # Getting the type of 'node' (line 32)
            node_13492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 50), 'node', False)
            # Obtaining the member 'body' of a type (line 32)
            body_13493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 50), node_13492, 'body')
            # Obtaining the member '__getitem__' of a type (line 32)
            getitem___13494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 50), body_13493, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 32)
            subscript_call_result_13495 = invoke(stypy.reporting.localization.Localization(__file__, 32, 50), getitem___13494, i_13491)
            
            # Processing the call keyword arguments (line 32)
            kwargs_13496 = {}
            # Getting the type of 'stypy_functions_copy' (line 32)
            stypy_functions_copy_13489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'stypy_functions_copy', False)
            # Obtaining the member 'is_blank_line' of a type (line 32)
            is_blank_line_13490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 15), stypy_functions_copy_13489, 'is_blank_line')
            # Calling is_blank_line(args, kwargs) (line 32)
            is_blank_line_call_result_13497 = invoke(stypy.reporting.localization.Localization(__file__, 32, 15), is_blank_line_13490, *[subscript_call_result_13495], **kwargs_13496)
            
            # Testing if the type of an if condition is none (line 32)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 32, 12), is_blank_line_call_result_13497):
                pass
            else:
                
                # Testing the type of an if condition (line 32)
                if_condition_13498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 12), is_blank_line_call_result_13497)
                # Assigning a type to the variable 'if_condition_13498' (line 32)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'if_condition_13498', if_condition_13498)
                # SSA begins for if statement (line 32)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Name (line 33):
                
                # Assigning a Subscript to a Name (line 33):
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 33)
                i_13499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 46), 'i')
                # Getting the type of 'node' (line 33)
                node_13500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 36), 'node')
                # Obtaining the member 'body' of a type (line 33)
                body_13501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 36), node_13500, 'body')
                # Obtaining the member '__getitem__' of a type (line 33)
                getitem___13502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 36), body_13501, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 33)
                subscript_call_result_13503 = invoke(stypy.reporting.localization.Localization(__file__, 33, 36), getitem___13502, i_13499)
                
                # Assigning a type to the variable 'separator_comment' (line 33)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'separator_comment', subscript_call_result_13503)
                # SSA join for if statement (line 32)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to is_src_comment(...): (line 34)
            # Processing the call arguments (line 34)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 34)
            i_13506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 61), 'i', False)
            # Getting the type of 'node' (line 34)
            node_13507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 51), 'node', False)
            # Obtaining the member 'body' of a type (line 34)
            body_13508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 51), node_13507, 'body')
            # Obtaining the member '__getitem__' of a type (line 34)
            getitem___13509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 51), body_13508, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 34)
            subscript_call_result_13510 = invoke(stypy.reporting.localization.Localization(__file__, 34, 51), getitem___13509, i_13506)
            
            # Processing the call keyword arguments (line 34)
            kwargs_13511 = {}
            # Getting the type of 'stypy_functions_copy' (line 34)
            stypy_functions_copy_13504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'stypy_functions_copy', False)
            # Obtaining the member 'is_src_comment' of a type (line 34)
            is_src_comment_13505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 15), stypy_functions_copy_13504, 'is_src_comment')
            # Calling is_src_comment(args, kwargs) (line 34)
            is_src_comment_call_result_13512 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), is_src_comment_13505, *[subscript_call_result_13510], **kwargs_13511)
            
            # Testing if the type of an if condition is none (line 34)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 34, 12), is_src_comment_call_result_13512):
                pass
            else:
                
                # Testing the type of an if condition (line 34)
                if_condition_13513 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 12), is_src_comment_call_result_13512)
                # Assigning a type to the variable 'if_condition_13513' (line 34)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'if_condition_13513', if_condition_13513)
                # SSA begins for if statement (line 34)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Name (line 35):
                
                # Assigning a Subscript to a Name (line 35):
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 35)
                i_13514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 36), 'i')
                # Getting the type of 'node' (line 35)
                node_13515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 26), 'node')
                # Obtaining the member 'body' of a type (line 35)
                body_13516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 26), node_13515, 'body')
                # Obtaining the member '__getitem__' of a type (line 35)
                getitem___13517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 26), body_13516, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 35)
                subscript_call_result_13518 = invoke(stypy.reporting.localization.Localization(__file__, 35, 26), getitem___13517, i_13514)
                
                # Assigning a type to the variable 'comment' (line 35)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'comment', subscript_call_result_13518)
                
                # Assigning a Call to a Attribute (line 36):
                
                # Assigning a Call to a Attribute (line 36):
                
                # Call to replace(...): (line 36)
                # Processing the call arguments (line 36)
                str_13523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 60), 'str', '# Assignment')
                str_13524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 76), 'str', '# Class-bound assignment')
                # Processing the call keyword arguments (line 36)
                kwargs_13525 = {}
                # Getting the type of 'comment' (line 36)
                comment_13519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 35), 'comment', False)
                # Obtaining the member 'value' of a type (line 36)
                value_13520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 35), comment_13519, 'value')
                # Obtaining the member 'id' of a type (line 36)
                id_13521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 35), value_13520, 'id')
                # Obtaining the member 'replace' of a type (line 36)
                replace_13522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 35), id_13521, 'replace')
                # Calling replace(args, kwargs) (line 36)
                replace_call_result_13526 = invoke(stypy.reporting.localization.Localization(__file__, 36, 35), replace_13522, *[str_13523, str_13524], **kwargs_13525)
                
                # Getting the type of 'comment' (line 36)
                comment_13527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'comment')
                # Obtaining the member 'value' of a type (line 36)
                value_13528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), comment_13527, 'value')
                # Setting the type of the member 'id' of a type (line 36)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), value_13528, 'id', replace_call_result_13526)
                # SSA join for if statement (line 34)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_13529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        # Getting the type of 'separator_comment' (line 38)
        separator_comment_13530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'separator_comment')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 15), tuple_13529, separator_comment_13530)
        # Adding element type (line 38)
        # Getting the type of 'comment' (line 38)
        comment_13531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 34), 'comment')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 15), tuple_13529, comment_13531)
        
        # Assigning a type to the variable 'stypy_return_type' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', tuple_13529)
        
        # ################# End of '__extract_attribute_attached_comments(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__extract_attribute_attached_comments' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_13532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13532)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__extract_attribute_attached_comments'
        return stypy_return_type_13532


    @norecursion
    def visit_ClassDef(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_ClassDef'
        module_type_store = module_type_store.open_function_context('visit_ClassDef', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ClassAttributesVisitor.visit_ClassDef.__dict__.__setitem__('stypy_localization', localization)
        ClassAttributesVisitor.visit_ClassDef.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ClassAttributesVisitor.visit_ClassDef.__dict__.__setitem__('stypy_type_store', module_type_store)
        ClassAttributesVisitor.visit_ClassDef.__dict__.__setitem__('stypy_function_name', 'ClassAttributesVisitor.visit_ClassDef')
        ClassAttributesVisitor.visit_ClassDef.__dict__.__setitem__('stypy_param_names_list', ['node'])
        ClassAttributesVisitor.visit_ClassDef.__dict__.__setitem__('stypy_varargs_param_name', None)
        ClassAttributesVisitor.visit_ClassDef.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ClassAttributesVisitor.visit_ClassDef.__dict__.__setitem__('stypy_call_defaults', defaults)
        ClassAttributesVisitor.visit_ClassDef.__dict__.__setitem__('stypy_call_varargs', varargs)
        ClassAttributesVisitor.visit_ClassDef.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ClassAttributesVisitor.visit_ClassDef.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ClassAttributesVisitor.visit_ClassDef', ['node'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 41):
        
        # Assigning a Call to a Name (line 41):
        
        # Call to filter(...): (line 41)
        # Processing the call arguments (line 41)

        @norecursion
        def _stypy_temp_lambda_17(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_17'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_17', 41, 34, True)
            # Passed parameters checking function
            _stypy_temp_lambda_17.stypy_localization = localization
            _stypy_temp_lambda_17.stypy_type_of_self = None
            _stypy_temp_lambda_17.stypy_type_store = module_type_store
            _stypy_temp_lambda_17.stypy_function_name = '_stypy_temp_lambda_17'
            _stypy_temp_lambda_17.stypy_param_names_list = ['element']
            _stypy_temp_lambda_17.stypy_varargs_param_name = None
            _stypy_temp_lambda_17.stypy_kwargs_param_name = None
            _stypy_temp_lambda_17.stypy_call_defaults = defaults
            _stypy_temp_lambda_17.stypy_call_varargs = varargs
            _stypy_temp_lambda_17.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_17', ['element'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_17', ['element'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to isinstance(...): (line 41)
            # Processing the call arguments (line 41)
            # Getting the type of 'element' (line 41)
            element_13535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 61), 'element', False)
            # Getting the type of 'ast' (line 41)
            ast_13536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 70), 'ast', False)
            # Obtaining the member 'Assign' of a type (line 41)
            Assign_13537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 70), ast_13536, 'Assign')
            # Processing the call keyword arguments (line 41)
            kwargs_13538 = {}
            # Getting the type of 'isinstance' (line 41)
            isinstance_13534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 50), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 41)
            isinstance_call_result_13539 = invoke(stypy.reporting.localization.Localization(__file__, 41, 50), isinstance_13534, *[element_13535, Assign_13537], **kwargs_13538)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), 'stypy_return_type', isinstance_call_result_13539)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_17' in the type store
            # Getting the type of 'stypy_return_type' (line 41)
            stypy_return_type_13540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_13540)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_17'
            return stypy_return_type_13540

        # Assigning a type to the variable '_stypy_temp_lambda_17' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), '_stypy_temp_lambda_17', _stypy_temp_lambda_17)
        # Getting the type of '_stypy_temp_lambda_17' (line 41)
        _stypy_temp_lambda_17_13541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), '_stypy_temp_lambda_17')
        # Getting the type of 'node' (line 41)
        node_13542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 83), 'node', False)
        # Obtaining the member 'body' of a type (line 41)
        body_13543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 83), node_13542, 'body')
        # Processing the call keyword arguments (line 41)
        kwargs_13544 = {}
        # Getting the type of 'filter' (line 41)
        filter_13533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 27), 'filter', False)
        # Calling filter(args, kwargs) (line 41)
        filter_call_result_13545 = invoke(stypy.reporting.localization.Localization(__file__, 41, 27), filter_13533, *[_stypy_temp_lambda_17_13541, body_13543], **kwargs_13544)
        
        # Assigning a type to the variable 'class_attributes' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'class_attributes', filter_call_result_13545)
        
        # Assigning a List to a Name (line 43):
        
        # Assigning a List to a Name (line 43):
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_13546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        
        # Assigning a type to the variable 'attr_stmts' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'attr_stmts', list_13546)
        
        # Getting the type of 'class_attributes' (line 44)
        class_attributes_13547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'class_attributes')
        # Assigning a type to the variable 'class_attributes_13547' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'class_attributes_13547', class_attributes_13547)
        # Testing if the for loop is going to be iterated (line 44)
        # Testing the type of a for loop iterable (line 44)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 44, 8), class_attributes_13547)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 44, 8), class_attributes_13547):
            # Getting the type of the for loop variable (line 44)
            for_loop_var_13548 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 44, 8), class_attributes_13547)
            # Assigning a type to the variable 'attr' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'attr', for_loop_var_13548)
            # SSA begins for a for statement (line 44)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Tuple (line 45):
            
            # Assigning a Call to a Name:
            
            # Call to __extract_attribute_attached_comments(...): (line 45)
            # Processing the call arguments (line 45)
            # Getting the type of 'attr' (line 45)
            attr_13551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 84), 'attr', False)
            # Getting the type of 'node' (line 45)
            node_13552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 90), 'node', False)
            # Processing the call keyword arguments (line 45)
            kwargs_13553 = {}
            # Getting the type of 'self' (line 45)
            self_13549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 41), 'self', False)
            # Obtaining the member '__extract_attribute_attached_comments' of a type (line 45)
            extract_attribute_attached_comments_13550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 41), self_13549, '__extract_attribute_attached_comments')
            # Calling __extract_attribute_attached_comments(args, kwargs) (line 45)
            extract_attribute_attached_comments_call_result_13554 = invoke(stypy.reporting.localization.Localization(__file__, 45, 41), extract_attribute_attached_comments_13550, *[attr_13551, node_13552], **kwargs_13553)
            
            # Assigning a type to the variable 'call_assignment_13466' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'call_assignment_13466', extract_attribute_attached_comments_call_result_13554)
            
            # Assigning a Call to a Name (line 45):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_13466' (line 45)
            call_assignment_13466_13555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'call_assignment_13466', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_13556 = stypy_get_value_from_tuple(call_assignment_13466_13555, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_13467' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'call_assignment_13467', stypy_get_value_from_tuple_call_result_13556)
            
            # Assigning a Name to a Name (line 45):
            # Getting the type of 'call_assignment_13467' (line 45)
            call_assignment_13467_13557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'call_assignment_13467')
            # Assigning a type to the variable 'separator_comment' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'separator_comment', call_assignment_13467_13557)
            
            # Assigning a Call to a Name (line 45):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_13466' (line 45)
            call_assignment_13466_13558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'call_assignment_13466', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_13559 = stypy_get_value_from_tuple(call_assignment_13466_13558, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_13468' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'call_assignment_13468', stypy_get_value_from_tuple_call_result_13559)
            
            # Assigning a Name to a Name (line 45):
            # Getting the type of 'call_assignment_13468' (line 45)
            call_assignment_13468_13560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'call_assignment_13468')
            # Assigning a type to the variable 'comment' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'comment', call_assignment_13468_13560)
            
            # Type idiom detected: calculating its left and rigth part (line 46)
            # Getting the type of 'separator_comment' (line 46)
            separator_comment_13561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'separator_comment')
            # Getting the type of 'None' (line 46)
            None_13562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 40), 'None')
            
            (may_be_13563, more_types_in_union_13564) = may_not_be_none(separator_comment_13561, None_13562)

            if may_be_13563:

                if more_types_in_union_13564:
                    # Runtime conditional SSA (line 46)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to remove(...): (line 47)
                # Processing the call arguments (line 47)
                # Getting the type of 'separator_comment' (line 47)
                separator_comment_13568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'separator_comment', False)
                # Processing the call keyword arguments (line 47)
                kwargs_13569 = {}
                # Getting the type of 'node' (line 47)
                node_13565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'node', False)
                # Obtaining the member 'body' of a type (line 47)
                body_13566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), node_13565, 'body')
                # Obtaining the member 'remove' of a type (line 47)
                remove_13567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), body_13566, 'remove')
                # Calling remove(args, kwargs) (line 47)
                remove_call_result_13570 = invoke(stypy.reporting.localization.Localization(__file__, 47, 16), remove_13567, *[separator_comment_13568], **kwargs_13569)
                
                
                # Call to append(...): (line 48)
                # Processing the call arguments (line 48)
                # Getting the type of 'separator_comment' (line 48)
                separator_comment_13573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 34), 'separator_comment', False)
                # Processing the call keyword arguments (line 48)
                kwargs_13574 = {}
                # Getting the type of 'attr_stmts' (line 48)
                attr_stmts_13571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'attr_stmts', False)
                # Obtaining the member 'append' of a type (line 48)
                append_13572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 16), attr_stmts_13571, 'append')
                # Calling append(args, kwargs) (line 48)
                append_call_result_13575 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), append_13572, *[separator_comment_13573], **kwargs_13574)
                

                if more_types_in_union_13564:
                    # SSA join for if statement (line 46)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Type idiom detected: calculating its left and rigth part (line 50)
            # Getting the type of 'separator_comment' (line 50)
            separator_comment_13576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'separator_comment')
            # Getting the type of 'None' (line 50)
            None_13577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 40), 'None')
            
            (may_be_13578, more_types_in_union_13579) = may_not_be_none(separator_comment_13576, None_13577)

            if may_be_13578:

                if more_types_in_union_13579:
                    # Runtime conditional SSA (line 50)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to remove(...): (line 51)
                # Processing the call arguments (line 51)
                # Getting the type of 'comment' (line 51)
                comment_13583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 33), 'comment', False)
                # Processing the call keyword arguments (line 51)
                kwargs_13584 = {}
                # Getting the type of 'node' (line 51)
                node_13580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'node', False)
                # Obtaining the member 'body' of a type (line 51)
                body_13581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), node_13580, 'body')
                # Obtaining the member 'remove' of a type (line 51)
                remove_13582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), body_13581, 'remove')
                # Calling remove(args, kwargs) (line 51)
                remove_call_result_13585 = invoke(stypy.reporting.localization.Localization(__file__, 51, 16), remove_13582, *[comment_13583], **kwargs_13584)
                
                
                # Call to append(...): (line 52)
                # Processing the call arguments (line 52)
                # Getting the type of 'comment' (line 52)
                comment_13588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 34), 'comment', False)
                # Processing the call keyword arguments (line 52)
                kwargs_13589 = {}
                # Getting the type of 'attr_stmts' (line 52)
                attr_stmts_13586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'attr_stmts', False)
                # Obtaining the member 'append' of a type (line 52)
                append_13587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 16), attr_stmts_13586, 'append')
                # Calling append(args, kwargs) (line 52)
                append_call_result_13590 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), append_13587, *[comment_13588], **kwargs_13589)
                

                if more_types_in_union_13579:
                    # SSA join for if statement (line 50)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Call to remove(...): (line 54)
            # Processing the call arguments (line 54)
            # Getting the type of 'attr' (line 54)
            attr_13594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 29), 'attr', False)
            # Processing the call keyword arguments (line 54)
            kwargs_13595 = {}
            # Getting the type of 'node' (line 54)
            node_13591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'node', False)
            # Obtaining the member 'body' of a type (line 54)
            body_13592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), node_13591, 'body')
            # Obtaining the member 'remove' of a type (line 54)
            remove_13593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), body_13592, 'remove')
            # Calling remove(args, kwargs) (line 54)
            remove_call_result_13596 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), remove_13593, *[attr_13594], **kwargs_13595)
            
            
            # Assigning a Call to a Name (line 56):
            
            # Assigning a Call to a Name (line 56):
            
            # Call to create_attribute(...): (line 56)
            # Processing the call arguments (line 56)
            # Getting the type of 'node' (line 56)
            node_13599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 66), 'node', False)
            # Obtaining the member 'name' of a type (line 56)
            name_13600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 66), node_13599, 'name')
            
            # Obtaining the type of the subscript
            int_13601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 90), 'int')
            # Getting the type of 'attr' (line 56)
            attr_13602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 77), 'attr', False)
            # Obtaining the member 'targets' of a type (line 56)
            targets_13603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 77), attr_13602, 'targets')
            # Obtaining the member '__getitem__' of a type (line 56)
            getitem___13604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 77), targets_13603, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 56)
            subscript_call_result_13605 = invoke(stypy.reporting.localization.Localization(__file__, 56, 77), getitem___13604, int_13601)
            
            # Obtaining the member 'id' of a type (line 56)
            id_13606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 77), subscript_call_result_13605, 'id')
            # Processing the call keyword arguments (line 56)
            kwargs_13607 = {}
            # Getting the type of 'core_language_copy' (line 56)
            core_language_copy_13597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'core_language_copy', False)
            # Obtaining the member 'create_attribute' of a type (line 56)
            create_attribute_13598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 30), core_language_copy_13597, 'create_attribute')
            # Calling create_attribute(args, kwargs) (line 56)
            create_attribute_call_result_13608 = invoke(stypy.reporting.localization.Localization(__file__, 56, 30), create_attribute_13598, *[name_13600, id_13606], **kwargs_13607)
            
            # Assigning a type to the variable 'temp_class_attr' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'temp_class_attr', create_attribute_call_result_13608)
            
            
            # Call to len(...): (line 57)
            # Processing the call arguments (line 57)
            
            # Call to filter(...): (line 57)
            # Processing the call arguments (line 57)

            @norecursion
            def _stypy_temp_lambda_18(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_18'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_18', 57, 26, True)
                # Passed parameters checking function
                _stypy_temp_lambda_18.stypy_localization = localization
                _stypy_temp_lambda_18.stypy_type_of_self = None
                _stypy_temp_lambda_18.stypy_type_store = module_type_store
                _stypy_temp_lambda_18.stypy_function_name = '_stypy_temp_lambda_18'
                _stypy_temp_lambda_18.stypy_param_names_list = ['class_attr']
                _stypy_temp_lambda_18.stypy_varargs_param_name = None
                _stypy_temp_lambda_18.stypy_kwargs_param_name = None
                _stypy_temp_lambda_18.stypy_call_defaults = defaults
                _stypy_temp_lambda_18.stypy_call_varargs = varargs
                _stypy_temp_lambda_18.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_18', ['class_attr'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_18', ['class_attr'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                
                # Obtaining the type of the subscript
                int_13611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 64), 'int')
                # Getting the type of 'class_attr' (line 57)
                class_attr_13612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 45), 'class_attr', False)
                # Obtaining the member 'targets' of a type (line 57)
                targets_13613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 45), class_attr_13612, 'targets')
                # Obtaining the member '__getitem__' of a type (line 57)
                getitem___13614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 45), targets_13613, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 57)
                subscript_call_result_13615 = invoke(stypy.reporting.localization.Localization(__file__, 57, 45), getitem___13614, int_13611)
                
                # Getting the type of 'attr' (line 57)
                attr_13616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 70), 'attr', False)
                # Obtaining the member 'value' of a type (line 57)
                value_13617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 70), attr_13616, 'value')
                # Applying the binary operator '==' (line 57)
                result_eq_13618 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 45), '==', subscript_call_result_13615, value_13617)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 57)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), 'stypy_return_type', result_eq_13618)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_18' in the type store
                # Getting the type of 'stypy_return_type' (line 57)
                stypy_return_type_13619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_13619)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_18'
                return stypy_return_type_13619

            # Assigning a type to the variable '_stypy_temp_lambda_18' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), '_stypy_temp_lambda_18', _stypy_temp_lambda_18)
            # Getting the type of '_stypy_temp_lambda_18' (line 57)
            _stypy_temp_lambda_18_13620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), '_stypy_temp_lambda_18')
            # Getting the type of 'class_attributes' (line 57)
            class_attributes_13621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 82), 'class_attributes', False)
            # Processing the call keyword arguments (line 57)
            kwargs_13622 = {}
            # Getting the type of 'filter' (line 57)
            filter_13610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'filter', False)
            # Calling filter(args, kwargs) (line 57)
            filter_call_result_13623 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), filter_13610, *[_stypy_temp_lambda_18_13620, class_attributes_13621], **kwargs_13622)
            
            # Processing the call keyword arguments (line 57)
            kwargs_13624 = {}
            # Getting the type of 'len' (line 57)
            len_13609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'len', False)
            # Calling len(args, kwargs) (line 57)
            len_call_result_13625 = invoke(stypy.reporting.localization.Localization(__file__, 57, 15), len_13609, *[filter_call_result_13623], **kwargs_13624)
            
            int_13626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 104), 'int')
            # Applying the binary operator '==' (line 57)
            result_eq_13627 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 15), '==', len_call_result_13625, int_13626)
            
            # Testing if the type of an if condition is none (line 57)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 57, 12), result_eq_13627):
                
                # Assigning a Call to a Name (line 60):
                
                # Assigning a Call to a Name (line 60):
                
                # Call to create_attribute(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'node' (line 60)
                node_13642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 71), 'node', False)
                # Obtaining the member 'name' of a type (line 60)
                name_13643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 71), node_13642, 'name')
                # Getting the type of 'attr' (line 60)
                attr_13644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 82), 'attr', False)
                # Obtaining the member 'value' of a type (line 60)
                value_13645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 82), attr_13644, 'value')
                # Obtaining the member 'id' of a type (line 60)
                id_13646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 82), value_13645, 'id')
                # Processing the call keyword arguments (line 60)
                kwargs_13647 = {}
                # Getting the type of 'core_language_copy' (line 60)
                core_language_copy_13640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 35), 'core_language_copy', False)
                # Obtaining the member 'create_attribute' of a type (line 60)
                create_attribute_13641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 35), core_language_copy_13640, 'create_attribute')
                # Calling create_attribute(args, kwargs) (line 60)
                create_attribute_call_result_13648 = invoke(stypy.reporting.localization.Localization(__file__, 60, 35), create_attribute_13641, *[name_13643, id_13646], **kwargs_13647)
                
                # Assigning a type to the variable 'temp_class_value' (line 60)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'temp_class_value', create_attribute_call_result_13648)
                
                # Call to append(...): (line 61)
                # Processing the call arguments (line 61)
                
                # Call to create_Assign(...): (line 61)
                # Processing the call arguments (line 61)
                # Getting the type of 'temp_class_attr' (line 61)
                temp_class_attr_13653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 67), 'temp_class_attr', False)
                # Getting the type of 'temp_class_value' (line 61)
                temp_class_value_13654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 84), 'temp_class_value', False)
                # Processing the call keyword arguments (line 61)
                kwargs_13655 = {}
                # Getting the type of 'core_language_copy' (line 61)
                core_language_copy_13651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'core_language_copy', False)
                # Obtaining the member 'create_Assign' of a type (line 61)
                create_Assign_13652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 34), core_language_copy_13651, 'create_Assign')
                # Calling create_Assign(args, kwargs) (line 61)
                create_Assign_call_result_13656 = invoke(stypy.reporting.localization.Localization(__file__, 61, 34), create_Assign_13652, *[temp_class_attr_13653, temp_class_value_13654], **kwargs_13655)
                
                # Processing the call keyword arguments (line 61)
                kwargs_13657 = {}
                # Getting the type of 'attr_stmts' (line 61)
                attr_stmts_13649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'attr_stmts', False)
                # Obtaining the member 'append' of a type (line 61)
                append_13650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 16), attr_stmts_13649, 'append')
                # Calling append(args, kwargs) (line 61)
                append_call_result_13658 = invoke(stypy.reporting.localization.Localization(__file__, 61, 16), append_13650, *[create_Assign_call_result_13656], **kwargs_13657)
                
            else:
                
                # Testing the type of an if condition (line 57)
                if_condition_13628 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 12), result_eq_13627)
                # Assigning a type to the variable 'if_condition_13628' (line 57)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'if_condition_13628', if_condition_13628)
                # SSA begins for if statement (line 57)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 58)
                # Processing the call arguments (line 58)
                
                # Call to create_Assign(...): (line 58)
                # Processing the call arguments (line 58)
                # Getting the type of 'temp_class_attr' (line 58)
                temp_class_attr_13633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 67), 'temp_class_attr', False)
                # Getting the type of 'attr' (line 58)
                attr_13634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 84), 'attr', False)
                # Obtaining the member 'value' of a type (line 58)
                value_13635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 84), attr_13634, 'value')
                # Processing the call keyword arguments (line 58)
                kwargs_13636 = {}
                # Getting the type of 'core_language_copy' (line 58)
                core_language_copy_13631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 34), 'core_language_copy', False)
                # Obtaining the member 'create_Assign' of a type (line 58)
                create_Assign_13632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 34), core_language_copy_13631, 'create_Assign')
                # Calling create_Assign(args, kwargs) (line 58)
                create_Assign_call_result_13637 = invoke(stypy.reporting.localization.Localization(__file__, 58, 34), create_Assign_13632, *[temp_class_attr_13633, value_13635], **kwargs_13636)
                
                # Processing the call keyword arguments (line 58)
                kwargs_13638 = {}
                # Getting the type of 'attr_stmts' (line 58)
                attr_stmts_13629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'attr_stmts', False)
                # Obtaining the member 'append' of a type (line 58)
                append_13630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), attr_stmts_13629, 'append')
                # Calling append(args, kwargs) (line 58)
                append_call_result_13639 = invoke(stypy.reporting.localization.Localization(__file__, 58, 16), append_13630, *[create_Assign_call_result_13637], **kwargs_13638)
                
                # SSA branch for the else part of an if statement (line 57)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 60):
                
                # Assigning a Call to a Name (line 60):
                
                # Call to create_attribute(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'node' (line 60)
                node_13642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 71), 'node', False)
                # Obtaining the member 'name' of a type (line 60)
                name_13643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 71), node_13642, 'name')
                # Getting the type of 'attr' (line 60)
                attr_13644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 82), 'attr', False)
                # Obtaining the member 'value' of a type (line 60)
                value_13645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 82), attr_13644, 'value')
                # Obtaining the member 'id' of a type (line 60)
                id_13646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 82), value_13645, 'id')
                # Processing the call keyword arguments (line 60)
                kwargs_13647 = {}
                # Getting the type of 'core_language_copy' (line 60)
                core_language_copy_13640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 35), 'core_language_copy', False)
                # Obtaining the member 'create_attribute' of a type (line 60)
                create_attribute_13641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 35), core_language_copy_13640, 'create_attribute')
                # Calling create_attribute(args, kwargs) (line 60)
                create_attribute_call_result_13648 = invoke(stypy.reporting.localization.Localization(__file__, 60, 35), create_attribute_13641, *[name_13643, id_13646], **kwargs_13647)
                
                # Assigning a type to the variable 'temp_class_value' (line 60)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'temp_class_value', create_attribute_call_result_13648)
                
                # Call to append(...): (line 61)
                # Processing the call arguments (line 61)
                
                # Call to create_Assign(...): (line 61)
                # Processing the call arguments (line 61)
                # Getting the type of 'temp_class_attr' (line 61)
                temp_class_attr_13653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 67), 'temp_class_attr', False)
                # Getting the type of 'temp_class_value' (line 61)
                temp_class_value_13654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 84), 'temp_class_value', False)
                # Processing the call keyword arguments (line 61)
                kwargs_13655 = {}
                # Getting the type of 'core_language_copy' (line 61)
                core_language_copy_13651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'core_language_copy', False)
                # Obtaining the member 'create_Assign' of a type (line 61)
                create_Assign_13652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 34), core_language_copy_13651, 'create_Assign')
                # Calling create_Assign(args, kwargs) (line 61)
                create_Assign_call_result_13656 = invoke(stypy.reporting.localization.Localization(__file__, 61, 34), create_Assign_13652, *[temp_class_attr_13653, temp_class_value_13654], **kwargs_13655)
                
                # Processing the call keyword arguments (line 61)
                kwargs_13657 = {}
                # Getting the type of 'attr_stmts' (line 61)
                attr_stmts_13649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'attr_stmts', False)
                # Obtaining the member 'append' of a type (line 61)
                append_13650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 16), attr_stmts_13649, 'append')
                # Calling append(args, kwargs) (line 61)
                append_call_result_13658 = invoke(stypy.reporting.localization.Localization(__file__, 61, 16), append_13650, *[create_Assign_call_result_13656], **kwargs_13657)
                
                # SSA join for if statement (line 57)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'node' (line 65)
        node_13660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'node', False)
        # Obtaining the member 'body' of a type (line 65)
        body_13661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 15), node_13660, 'body')
        # Processing the call keyword arguments (line 65)
        kwargs_13662 = {}
        # Getting the type of 'len' (line 65)
        len_13659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'len', False)
        # Calling len(args, kwargs) (line 65)
        len_call_result_13663 = invoke(stypy.reporting.localization.Localization(__file__, 65, 11), len_13659, *[body_13661], **kwargs_13662)
        
        int_13664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 29), 'int')
        # Applying the binary operator '==' (line 65)
        result_eq_13665 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 11), '==', len_call_result_13663, int_13664)
        
        # Testing if the type of an if condition is none (line 65)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 65, 8), result_eq_13665):
            pass
        else:
            
            # Testing the type of an if condition (line 65)
            if_condition_13666 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 8), result_eq_13665)
            # Assigning a type to the variable 'if_condition_13666' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'if_condition_13666', if_condition_13666)
            # SSA begins for if statement (line 65)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 66)
            # Processing the call arguments (line 66)
            
            # Call to create_pass_node(...): (line 66)
            # Processing the call keyword arguments (line 66)
            kwargs_13672 = {}
            # Getting the type of 'stypy_functions_copy' (line 66)
            stypy_functions_copy_13670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 29), 'stypy_functions_copy', False)
            # Obtaining the member 'create_pass_node' of a type (line 66)
            create_pass_node_13671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 29), stypy_functions_copy_13670, 'create_pass_node')
            # Calling create_pass_node(args, kwargs) (line 66)
            create_pass_node_call_result_13673 = invoke(stypy.reporting.localization.Localization(__file__, 66, 29), create_pass_node_13671, *[], **kwargs_13672)
            
            # Processing the call keyword arguments (line 66)
            kwargs_13674 = {}
            # Getting the type of 'node' (line 66)
            node_13667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'node', False)
            # Obtaining the member 'body' of a type (line 66)
            body_13668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), node_13667, 'body')
            # Obtaining the member 'append' of a type (line 66)
            append_13669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), body_13668, 'append')
            # Calling append(args, kwargs) (line 66)
            append_call_result_13675 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), append_13669, *[create_pass_node_call_result_13673], **kwargs_13674)
            
            # SSA join for if statement (line 65)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to flatten_lists(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'node' (line 68)
        node_13678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 50), 'node', False)
        # Getting the type of 'attr_stmts' (line 68)
        attr_stmts_13679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 56), 'attr_stmts', False)
        # Processing the call keyword arguments (line 68)
        kwargs_13680 = {}
        # Getting the type of 'stypy_functions_copy' (line 68)
        stypy_functions_copy_13676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'stypy_functions_copy', False)
        # Obtaining the member 'flatten_lists' of a type (line 68)
        flatten_lists_13677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 15), stypy_functions_copy_13676, 'flatten_lists')
        # Calling flatten_lists(args, kwargs) (line 68)
        flatten_lists_call_result_13681 = invoke(stypy.reporting.localization.Localization(__file__, 68, 15), flatten_lists_13677, *[node_13678, attr_stmts_13679], **kwargs_13680)
        
        # Assigning a type to the variable 'stypy_return_type' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'stypy_return_type', flatten_lists_call_result_13681)
        
        # ################# End of 'visit_ClassDef(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ClassDef' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_13682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13682)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ClassDef'
        return stypy_return_type_13682


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ClassAttributesVisitor.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ClassAttributesVisitor' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'ClassAttributesVisitor', ClassAttributesVisitor)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
