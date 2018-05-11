
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from ....code_generation_copy.type_inference_programs_copy.aux_functions_copy import *
2: from ....visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, stypy_functions_copy
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

# 'from testing.test_programs.stypy_code_copy.stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import ' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')
import_30050 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy')

if (type(import_30050) is not StypyTypeError):

    if (import_30050 != 'pyd_module'):
        __import__(import_30050)
        sys_modules_30051 = sys.modules[import_30050]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', sys_modules_30051.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_30051, sys_modules_30051.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', import_30050)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, stypy_functions_copy' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')
import_30052 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy')

if (type(import_30052) is not StypyTypeError):

    if (import_30052 != 'pyd_module'):
        __import__(import_30052)
        sys_modules_30053 = sys.modules[import_30052]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', sys_modules_30053.module_type_store, module_type_store, ['core_language_copy', 'stypy_functions_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_30053, sys_modules_30053.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, stypy_functions_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', None, module_type_store, ['core_language_copy', 'stypy_functions_copy'], [core_language_copy, stypy_functions_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', import_30052)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import ast' statement (line 3)
import ast

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'ast', ast, module_type_store)

# Declaration of the 'ClassAttributesVisitor' class
# Getting the type of 'ast' (line 5)
ast_30054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 29), 'ast')
# Obtaining the member 'NodeTransformer' of a type (line 5)
NodeTransformer_30055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 29), ast_30054, 'NodeTransformer')

class ClassAttributesVisitor(NodeTransformer_30055, ):
    str_30056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, (-1)), 'str', '\n    This desugaring visitor converts class-bound attributes such as:\n\n    class Example:\n        att = "hello"\n        <rest of the members that are not attributes>\n\n    into this equivalent form:\n\n    class Example:\n        <rest of the members that are not attributes>\n\n    Example.att = "hello"\n\n    The first form cannot be properly processed by stypy due to limitations in the way they are transformed into AST\n    nodes. The second form is completely processable using the same assignment processing code we already have.\n\n    ')

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
        attr_30060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 37), 'attr', False)
        # Processing the call keyword arguments (line 27)
        kwargs_30061 = {}
        # Getting the type of 'node' (line 27)
        node_30057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'node', False)
        # Obtaining the member 'body' of a type (line 27)
        body_30058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 21), node_30057, 'body')
        # Obtaining the member 'index' of a type (line 27)
        index_30059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 21), body_30058, 'index')
        # Calling index(args, kwargs) (line 27)
        index_call_result_30062 = invoke(stypy.reporting.localization.Localization(__file__, 27, 21), index_30059, *[attr_30060], **kwargs_30061)
        
        # Assigning a type to the variable 'attr_index' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'attr_index', index_call_result_30062)
        
        # Assigning a Name to a Name (line 28):
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'None' (line 28)
        None_30063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 28), 'None')
        # Assigning a type to the variable 'separator_comment' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'separator_comment', None_30063)
        
        # Assigning a Name to a Name (line 29):
        
        # Assigning a Name to a Name (line 29):
        # Getting the type of 'None' (line 29)
        None_30064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'None')
        # Assigning a type to the variable 'comment' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'comment', None_30064)
        
        
        # Call to range(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'attr_index' (line 31)
        attr_index_30066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'attr_index', False)
        # Processing the call keyword arguments (line 31)
        kwargs_30067 = {}
        # Getting the type of 'range' (line 31)
        range_30065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'range', False)
        # Calling range(args, kwargs) (line 31)
        range_call_result_30068 = invoke(stypy.reporting.localization.Localization(__file__, 31, 17), range_30065, *[attr_index_30066], **kwargs_30067)
        
        # Assigning a type to the variable 'range_call_result_30068' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'range_call_result_30068', range_call_result_30068)
        # Testing if the for loop is going to be iterated (line 31)
        # Testing the type of a for loop iterable (line 31)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 31, 8), range_call_result_30068)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 31, 8), range_call_result_30068):
            # Getting the type of the for loop variable (line 31)
            for_loop_var_30069 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 31, 8), range_call_result_30068)
            # Assigning a type to the variable 'i' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'i', for_loop_var_30069)
            # SSA begins for a for statement (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to is_blank_line(...): (line 32)
            # Processing the call arguments (line 32)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 32)
            i_30072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 60), 'i', False)
            # Getting the type of 'node' (line 32)
            node_30073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 50), 'node', False)
            # Obtaining the member 'body' of a type (line 32)
            body_30074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 50), node_30073, 'body')
            # Obtaining the member '__getitem__' of a type (line 32)
            getitem___30075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 50), body_30074, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 32)
            subscript_call_result_30076 = invoke(stypy.reporting.localization.Localization(__file__, 32, 50), getitem___30075, i_30072)
            
            # Processing the call keyword arguments (line 32)
            kwargs_30077 = {}
            # Getting the type of 'stypy_functions_copy' (line 32)
            stypy_functions_copy_30070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'stypy_functions_copy', False)
            # Obtaining the member 'is_blank_line' of a type (line 32)
            is_blank_line_30071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 15), stypy_functions_copy_30070, 'is_blank_line')
            # Calling is_blank_line(args, kwargs) (line 32)
            is_blank_line_call_result_30078 = invoke(stypy.reporting.localization.Localization(__file__, 32, 15), is_blank_line_30071, *[subscript_call_result_30076], **kwargs_30077)
            
            # Testing if the type of an if condition is none (line 32)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 32, 12), is_blank_line_call_result_30078):
                pass
            else:
                
                # Testing the type of an if condition (line 32)
                if_condition_30079 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 12), is_blank_line_call_result_30078)
                # Assigning a type to the variable 'if_condition_30079' (line 32)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'if_condition_30079', if_condition_30079)
                # SSA begins for if statement (line 32)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Name (line 33):
                
                # Assigning a Subscript to a Name (line 33):
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 33)
                i_30080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 46), 'i')
                # Getting the type of 'node' (line 33)
                node_30081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 36), 'node')
                # Obtaining the member 'body' of a type (line 33)
                body_30082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 36), node_30081, 'body')
                # Obtaining the member '__getitem__' of a type (line 33)
                getitem___30083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 36), body_30082, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 33)
                subscript_call_result_30084 = invoke(stypy.reporting.localization.Localization(__file__, 33, 36), getitem___30083, i_30080)
                
                # Assigning a type to the variable 'separator_comment' (line 33)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'separator_comment', subscript_call_result_30084)
                # SSA join for if statement (line 32)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to is_src_comment(...): (line 34)
            # Processing the call arguments (line 34)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 34)
            i_30087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 61), 'i', False)
            # Getting the type of 'node' (line 34)
            node_30088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 51), 'node', False)
            # Obtaining the member 'body' of a type (line 34)
            body_30089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 51), node_30088, 'body')
            # Obtaining the member '__getitem__' of a type (line 34)
            getitem___30090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 51), body_30089, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 34)
            subscript_call_result_30091 = invoke(stypy.reporting.localization.Localization(__file__, 34, 51), getitem___30090, i_30087)
            
            # Processing the call keyword arguments (line 34)
            kwargs_30092 = {}
            # Getting the type of 'stypy_functions_copy' (line 34)
            stypy_functions_copy_30085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'stypy_functions_copy', False)
            # Obtaining the member 'is_src_comment' of a type (line 34)
            is_src_comment_30086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 15), stypy_functions_copy_30085, 'is_src_comment')
            # Calling is_src_comment(args, kwargs) (line 34)
            is_src_comment_call_result_30093 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), is_src_comment_30086, *[subscript_call_result_30091], **kwargs_30092)
            
            # Testing if the type of an if condition is none (line 34)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 34, 12), is_src_comment_call_result_30093):
                pass
            else:
                
                # Testing the type of an if condition (line 34)
                if_condition_30094 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 12), is_src_comment_call_result_30093)
                # Assigning a type to the variable 'if_condition_30094' (line 34)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'if_condition_30094', if_condition_30094)
                # SSA begins for if statement (line 34)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Name (line 35):
                
                # Assigning a Subscript to a Name (line 35):
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 35)
                i_30095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 36), 'i')
                # Getting the type of 'node' (line 35)
                node_30096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 26), 'node')
                # Obtaining the member 'body' of a type (line 35)
                body_30097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 26), node_30096, 'body')
                # Obtaining the member '__getitem__' of a type (line 35)
                getitem___30098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 26), body_30097, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 35)
                subscript_call_result_30099 = invoke(stypy.reporting.localization.Localization(__file__, 35, 26), getitem___30098, i_30095)
                
                # Assigning a type to the variable 'comment' (line 35)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'comment', subscript_call_result_30099)
                
                # Assigning a Call to a Attribute (line 36):
                
                # Assigning a Call to a Attribute (line 36):
                
                # Call to replace(...): (line 36)
                # Processing the call arguments (line 36)
                str_30104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 60), 'str', '# Assignment')
                str_30105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 76), 'str', '# Class-bound assignment')
                # Processing the call keyword arguments (line 36)
                kwargs_30106 = {}
                # Getting the type of 'comment' (line 36)
                comment_30100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 35), 'comment', False)
                # Obtaining the member 'value' of a type (line 36)
                value_30101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 35), comment_30100, 'value')
                # Obtaining the member 'id' of a type (line 36)
                id_30102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 35), value_30101, 'id')
                # Obtaining the member 'replace' of a type (line 36)
                replace_30103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 35), id_30102, 'replace')
                # Calling replace(args, kwargs) (line 36)
                replace_call_result_30107 = invoke(stypy.reporting.localization.Localization(__file__, 36, 35), replace_30103, *[str_30104, str_30105], **kwargs_30106)
                
                # Getting the type of 'comment' (line 36)
                comment_30108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'comment')
                # Obtaining the member 'value' of a type (line 36)
                value_30109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), comment_30108, 'value')
                # Setting the type of the member 'id' of a type (line 36)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), value_30109, 'id', replace_call_result_30107)
                # SSA join for if statement (line 34)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 38)
        tuple_30110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 38)
        # Adding element type (line 38)
        # Getting the type of 'separator_comment' (line 38)
        separator_comment_30111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'separator_comment')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 15), tuple_30110, separator_comment_30111)
        # Adding element type (line 38)
        # Getting the type of 'comment' (line 38)
        comment_30112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 34), 'comment')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 15), tuple_30110, comment_30112)
        
        # Assigning a type to the variable 'stypy_return_type' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', tuple_30110)
        
        # ################# End of '__extract_attribute_attached_comments(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__extract_attribute_attached_comments' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_30113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30113)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__extract_attribute_attached_comments'
        return stypy_return_type_30113


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
        def _stypy_temp_lambda_41(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_41'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_41', 41, 34, True)
            # Passed parameters checking function
            _stypy_temp_lambda_41.stypy_localization = localization
            _stypy_temp_lambda_41.stypy_type_of_self = None
            _stypy_temp_lambda_41.stypy_type_store = module_type_store
            _stypy_temp_lambda_41.stypy_function_name = '_stypy_temp_lambda_41'
            _stypy_temp_lambda_41.stypy_param_names_list = ['element']
            _stypy_temp_lambda_41.stypy_varargs_param_name = None
            _stypy_temp_lambda_41.stypy_kwargs_param_name = None
            _stypy_temp_lambda_41.stypy_call_defaults = defaults
            _stypy_temp_lambda_41.stypy_call_varargs = varargs
            _stypy_temp_lambda_41.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_41', ['element'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_41', ['element'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to isinstance(...): (line 41)
            # Processing the call arguments (line 41)
            # Getting the type of 'element' (line 41)
            element_30116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 61), 'element', False)
            # Getting the type of 'ast' (line 41)
            ast_30117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 70), 'ast', False)
            # Obtaining the member 'Assign' of a type (line 41)
            Assign_30118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 70), ast_30117, 'Assign')
            # Processing the call keyword arguments (line 41)
            kwargs_30119 = {}
            # Getting the type of 'isinstance' (line 41)
            isinstance_30115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 50), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 41)
            isinstance_call_result_30120 = invoke(stypy.reporting.localization.Localization(__file__, 41, 50), isinstance_30115, *[element_30116, Assign_30118], **kwargs_30119)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), 'stypy_return_type', isinstance_call_result_30120)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_41' in the type store
            # Getting the type of 'stypy_return_type' (line 41)
            stypy_return_type_30121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_30121)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_41'
            return stypy_return_type_30121

        # Assigning a type to the variable '_stypy_temp_lambda_41' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), '_stypy_temp_lambda_41', _stypy_temp_lambda_41)
        # Getting the type of '_stypy_temp_lambda_41' (line 41)
        _stypy_temp_lambda_41_30122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), '_stypy_temp_lambda_41')
        # Getting the type of 'node' (line 41)
        node_30123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 83), 'node', False)
        # Obtaining the member 'body' of a type (line 41)
        body_30124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 83), node_30123, 'body')
        # Processing the call keyword arguments (line 41)
        kwargs_30125 = {}
        # Getting the type of 'filter' (line 41)
        filter_30114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 27), 'filter', False)
        # Calling filter(args, kwargs) (line 41)
        filter_call_result_30126 = invoke(stypy.reporting.localization.Localization(__file__, 41, 27), filter_30114, *[_stypy_temp_lambda_41_30122, body_30124], **kwargs_30125)
        
        # Assigning a type to the variable 'class_attributes' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'class_attributes', filter_call_result_30126)
        
        # Assigning a List to a Name (line 43):
        
        # Assigning a List to a Name (line 43):
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_30127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        
        # Assigning a type to the variable 'attr_stmts' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'attr_stmts', list_30127)
        
        # Getting the type of 'class_attributes' (line 44)
        class_attributes_30128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'class_attributes')
        # Assigning a type to the variable 'class_attributes_30128' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'class_attributes_30128', class_attributes_30128)
        # Testing if the for loop is going to be iterated (line 44)
        # Testing the type of a for loop iterable (line 44)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 44, 8), class_attributes_30128)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 44, 8), class_attributes_30128):
            # Getting the type of the for loop variable (line 44)
            for_loop_var_30129 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 44, 8), class_attributes_30128)
            # Assigning a type to the variable 'attr' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'attr', for_loop_var_30129)
            # SSA begins for a for statement (line 44)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Tuple (line 45):
            
            # Assigning a Call to a Name:
            
            # Call to __extract_attribute_attached_comments(...): (line 45)
            # Processing the call arguments (line 45)
            # Getting the type of 'attr' (line 45)
            attr_30132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 84), 'attr', False)
            # Getting the type of 'node' (line 45)
            node_30133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 90), 'node', False)
            # Processing the call keyword arguments (line 45)
            kwargs_30134 = {}
            # Getting the type of 'self' (line 45)
            self_30130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 41), 'self', False)
            # Obtaining the member '__extract_attribute_attached_comments' of a type (line 45)
            extract_attribute_attached_comments_30131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 41), self_30130, '__extract_attribute_attached_comments')
            # Calling __extract_attribute_attached_comments(args, kwargs) (line 45)
            extract_attribute_attached_comments_call_result_30135 = invoke(stypy.reporting.localization.Localization(__file__, 45, 41), extract_attribute_attached_comments_30131, *[attr_30132, node_30133], **kwargs_30134)
            
            # Assigning a type to the variable 'call_assignment_30047' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'call_assignment_30047', extract_attribute_attached_comments_call_result_30135)
            
            # Assigning a Call to a Name (line 45):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_30047' (line 45)
            call_assignment_30047_30136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'call_assignment_30047', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_30137 = stypy_get_value_from_tuple(call_assignment_30047_30136, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_30048' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'call_assignment_30048', stypy_get_value_from_tuple_call_result_30137)
            
            # Assigning a Name to a Name (line 45):
            # Getting the type of 'call_assignment_30048' (line 45)
            call_assignment_30048_30138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'call_assignment_30048')
            # Assigning a type to the variable 'separator_comment' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'separator_comment', call_assignment_30048_30138)
            
            # Assigning a Call to a Name (line 45):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_30047' (line 45)
            call_assignment_30047_30139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'call_assignment_30047', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_30140 = stypy_get_value_from_tuple(call_assignment_30047_30139, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_30049' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'call_assignment_30049', stypy_get_value_from_tuple_call_result_30140)
            
            # Assigning a Name to a Name (line 45):
            # Getting the type of 'call_assignment_30049' (line 45)
            call_assignment_30049_30141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'call_assignment_30049')
            # Assigning a type to the variable 'comment' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'comment', call_assignment_30049_30141)
            
            # Type idiom detected: calculating its left and rigth part (line 46)
            # Getting the type of 'separator_comment' (line 46)
            separator_comment_30142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'separator_comment')
            # Getting the type of 'None' (line 46)
            None_30143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 40), 'None')
            
            (may_be_30144, more_types_in_union_30145) = may_not_be_none(separator_comment_30142, None_30143)

            if may_be_30144:

                if more_types_in_union_30145:
                    # Runtime conditional SSA (line 46)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to remove(...): (line 47)
                # Processing the call arguments (line 47)
                # Getting the type of 'separator_comment' (line 47)
                separator_comment_30149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'separator_comment', False)
                # Processing the call keyword arguments (line 47)
                kwargs_30150 = {}
                # Getting the type of 'node' (line 47)
                node_30146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'node', False)
                # Obtaining the member 'body' of a type (line 47)
                body_30147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), node_30146, 'body')
                # Obtaining the member 'remove' of a type (line 47)
                remove_30148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 16), body_30147, 'remove')
                # Calling remove(args, kwargs) (line 47)
                remove_call_result_30151 = invoke(stypy.reporting.localization.Localization(__file__, 47, 16), remove_30148, *[separator_comment_30149], **kwargs_30150)
                
                
                # Call to append(...): (line 48)
                # Processing the call arguments (line 48)
                # Getting the type of 'separator_comment' (line 48)
                separator_comment_30154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 34), 'separator_comment', False)
                # Processing the call keyword arguments (line 48)
                kwargs_30155 = {}
                # Getting the type of 'attr_stmts' (line 48)
                attr_stmts_30152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'attr_stmts', False)
                # Obtaining the member 'append' of a type (line 48)
                append_30153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 16), attr_stmts_30152, 'append')
                # Calling append(args, kwargs) (line 48)
                append_call_result_30156 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), append_30153, *[separator_comment_30154], **kwargs_30155)
                

                if more_types_in_union_30145:
                    # SSA join for if statement (line 46)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Type idiom detected: calculating its left and rigth part (line 50)
            # Getting the type of 'separator_comment' (line 50)
            separator_comment_30157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'separator_comment')
            # Getting the type of 'None' (line 50)
            None_30158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 40), 'None')
            
            (may_be_30159, more_types_in_union_30160) = may_not_be_none(separator_comment_30157, None_30158)

            if may_be_30159:

                if more_types_in_union_30160:
                    # Runtime conditional SSA (line 50)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to remove(...): (line 51)
                # Processing the call arguments (line 51)
                # Getting the type of 'comment' (line 51)
                comment_30164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 33), 'comment', False)
                # Processing the call keyword arguments (line 51)
                kwargs_30165 = {}
                # Getting the type of 'node' (line 51)
                node_30161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'node', False)
                # Obtaining the member 'body' of a type (line 51)
                body_30162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), node_30161, 'body')
                # Obtaining the member 'remove' of a type (line 51)
                remove_30163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 16), body_30162, 'remove')
                # Calling remove(args, kwargs) (line 51)
                remove_call_result_30166 = invoke(stypy.reporting.localization.Localization(__file__, 51, 16), remove_30163, *[comment_30164], **kwargs_30165)
                
                
                # Call to append(...): (line 52)
                # Processing the call arguments (line 52)
                # Getting the type of 'comment' (line 52)
                comment_30169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 34), 'comment', False)
                # Processing the call keyword arguments (line 52)
                kwargs_30170 = {}
                # Getting the type of 'attr_stmts' (line 52)
                attr_stmts_30167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'attr_stmts', False)
                # Obtaining the member 'append' of a type (line 52)
                append_30168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 16), attr_stmts_30167, 'append')
                # Calling append(args, kwargs) (line 52)
                append_call_result_30171 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), append_30168, *[comment_30169], **kwargs_30170)
                

                if more_types_in_union_30160:
                    # SSA join for if statement (line 50)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Call to remove(...): (line 54)
            # Processing the call arguments (line 54)
            # Getting the type of 'attr' (line 54)
            attr_30175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 29), 'attr', False)
            # Processing the call keyword arguments (line 54)
            kwargs_30176 = {}
            # Getting the type of 'node' (line 54)
            node_30172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'node', False)
            # Obtaining the member 'body' of a type (line 54)
            body_30173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), node_30172, 'body')
            # Obtaining the member 'remove' of a type (line 54)
            remove_30174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), body_30173, 'remove')
            # Calling remove(args, kwargs) (line 54)
            remove_call_result_30177 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), remove_30174, *[attr_30175], **kwargs_30176)
            
            
            # Assigning a Call to a Name (line 56):
            
            # Assigning a Call to a Name (line 56):
            
            # Call to create_attribute(...): (line 56)
            # Processing the call arguments (line 56)
            # Getting the type of 'node' (line 56)
            node_30180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 66), 'node', False)
            # Obtaining the member 'name' of a type (line 56)
            name_30181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 66), node_30180, 'name')
            
            # Obtaining the type of the subscript
            int_30182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 90), 'int')
            # Getting the type of 'attr' (line 56)
            attr_30183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 77), 'attr', False)
            # Obtaining the member 'targets' of a type (line 56)
            targets_30184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 77), attr_30183, 'targets')
            # Obtaining the member '__getitem__' of a type (line 56)
            getitem___30185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 77), targets_30184, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 56)
            subscript_call_result_30186 = invoke(stypy.reporting.localization.Localization(__file__, 56, 77), getitem___30185, int_30182)
            
            # Obtaining the member 'id' of a type (line 56)
            id_30187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 77), subscript_call_result_30186, 'id')
            # Processing the call keyword arguments (line 56)
            kwargs_30188 = {}
            # Getting the type of 'core_language_copy' (line 56)
            core_language_copy_30178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'core_language_copy', False)
            # Obtaining the member 'create_attribute' of a type (line 56)
            create_attribute_30179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 30), core_language_copy_30178, 'create_attribute')
            # Calling create_attribute(args, kwargs) (line 56)
            create_attribute_call_result_30189 = invoke(stypy.reporting.localization.Localization(__file__, 56, 30), create_attribute_30179, *[name_30181, id_30187], **kwargs_30188)
            
            # Assigning a type to the variable 'temp_class_attr' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'temp_class_attr', create_attribute_call_result_30189)
            
            
            # Call to len(...): (line 57)
            # Processing the call arguments (line 57)
            
            # Call to filter(...): (line 57)
            # Processing the call arguments (line 57)

            @norecursion
            def _stypy_temp_lambda_42(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_42'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_42', 57, 26, True)
                # Passed parameters checking function
                _stypy_temp_lambda_42.stypy_localization = localization
                _stypy_temp_lambda_42.stypy_type_of_self = None
                _stypy_temp_lambda_42.stypy_type_store = module_type_store
                _stypy_temp_lambda_42.stypy_function_name = '_stypy_temp_lambda_42'
                _stypy_temp_lambda_42.stypy_param_names_list = ['class_attr']
                _stypy_temp_lambda_42.stypy_varargs_param_name = None
                _stypy_temp_lambda_42.stypy_kwargs_param_name = None
                _stypy_temp_lambda_42.stypy_call_defaults = defaults
                _stypy_temp_lambda_42.stypy_call_varargs = varargs
                _stypy_temp_lambda_42.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_42', ['class_attr'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_42', ['class_attr'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                
                # Obtaining the type of the subscript
                int_30192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 64), 'int')
                # Getting the type of 'class_attr' (line 57)
                class_attr_30193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 45), 'class_attr', False)
                # Obtaining the member 'targets' of a type (line 57)
                targets_30194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 45), class_attr_30193, 'targets')
                # Obtaining the member '__getitem__' of a type (line 57)
                getitem___30195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 45), targets_30194, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 57)
                subscript_call_result_30196 = invoke(stypy.reporting.localization.Localization(__file__, 57, 45), getitem___30195, int_30192)
                
                # Getting the type of 'attr' (line 57)
                attr_30197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 70), 'attr', False)
                # Obtaining the member 'value' of a type (line 57)
                value_30198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 70), attr_30197, 'value')
                # Applying the binary operator '==' (line 57)
                result_eq_30199 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 45), '==', subscript_call_result_30196, value_30198)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 57)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), 'stypy_return_type', result_eq_30199)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_42' in the type store
                # Getting the type of 'stypy_return_type' (line 57)
                stypy_return_type_30200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_30200)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_42'
                return stypy_return_type_30200

            # Assigning a type to the variable '_stypy_temp_lambda_42' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), '_stypy_temp_lambda_42', _stypy_temp_lambda_42)
            # Getting the type of '_stypy_temp_lambda_42' (line 57)
            _stypy_temp_lambda_42_30201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 26), '_stypy_temp_lambda_42')
            # Getting the type of 'class_attributes' (line 57)
            class_attributes_30202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 82), 'class_attributes', False)
            # Processing the call keyword arguments (line 57)
            kwargs_30203 = {}
            # Getting the type of 'filter' (line 57)
            filter_30191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'filter', False)
            # Calling filter(args, kwargs) (line 57)
            filter_call_result_30204 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), filter_30191, *[_stypy_temp_lambda_42_30201, class_attributes_30202], **kwargs_30203)
            
            # Processing the call keyword arguments (line 57)
            kwargs_30205 = {}
            # Getting the type of 'len' (line 57)
            len_30190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'len', False)
            # Calling len(args, kwargs) (line 57)
            len_call_result_30206 = invoke(stypy.reporting.localization.Localization(__file__, 57, 15), len_30190, *[filter_call_result_30204], **kwargs_30205)
            
            int_30207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 104), 'int')
            # Applying the binary operator '==' (line 57)
            result_eq_30208 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 15), '==', len_call_result_30206, int_30207)
            
            # Testing if the type of an if condition is none (line 57)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 57, 12), result_eq_30208):
                
                # Assigning a Call to a Name (line 60):
                
                # Assigning a Call to a Name (line 60):
                
                # Call to create_attribute(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'node' (line 60)
                node_30223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 71), 'node', False)
                # Obtaining the member 'name' of a type (line 60)
                name_30224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 71), node_30223, 'name')
                # Getting the type of 'attr' (line 60)
                attr_30225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 82), 'attr', False)
                # Obtaining the member 'value' of a type (line 60)
                value_30226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 82), attr_30225, 'value')
                # Obtaining the member 'id' of a type (line 60)
                id_30227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 82), value_30226, 'id')
                # Processing the call keyword arguments (line 60)
                kwargs_30228 = {}
                # Getting the type of 'core_language_copy' (line 60)
                core_language_copy_30221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 35), 'core_language_copy', False)
                # Obtaining the member 'create_attribute' of a type (line 60)
                create_attribute_30222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 35), core_language_copy_30221, 'create_attribute')
                # Calling create_attribute(args, kwargs) (line 60)
                create_attribute_call_result_30229 = invoke(stypy.reporting.localization.Localization(__file__, 60, 35), create_attribute_30222, *[name_30224, id_30227], **kwargs_30228)
                
                # Assigning a type to the variable 'temp_class_value' (line 60)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'temp_class_value', create_attribute_call_result_30229)
                
                # Call to append(...): (line 61)
                # Processing the call arguments (line 61)
                
                # Call to create_Assign(...): (line 61)
                # Processing the call arguments (line 61)
                # Getting the type of 'temp_class_attr' (line 61)
                temp_class_attr_30234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 67), 'temp_class_attr', False)
                # Getting the type of 'temp_class_value' (line 61)
                temp_class_value_30235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 84), 'temp_class_value', False)
                # Processing the call keyword arguments (line 61)
                kwargs_30236 = {}
                # Getting the type of 'core_language_copy' (line 61)
                core_language_copy_30232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'core_language_copy', False)
                # Obtaining the member 'create_Assign' of a type (line 61)
                create_Assign_30233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 34), core_language_copy_30232, 'create_Assign')
                # Calling create_Assign(args, kwargs) (line 61)
                create_Assign_call_result_30237 = invoke(stypy.reporting.localization.Localization(__file__, 61, 34), create_Assign_30233, *[temp_class_attr_30234, temp_class_value_30235], **kwargs_30236)
                
                # Processing the call keyword arguments (line 61)
                kwargs_30238 = {}
                # Getting the type of 'attr_stmts' (line 61)
                attr_stmts_30230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'attr_stmts', False)
                # Obtaining the member 'append' of a type (line 61)
                append_30231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 16), attr_stmts_30230, 'append')
                # Calling append(args, kwargs) (line 61)
                append_call_result_30239 = invoke(stypy.reporting.localization.Localization(__file__, 61, 16), append_30231, *[create_Assign_call_result_30237], **kwargs_30238)
                
            else:
                
                # Testing the type of an if condition (line 57)
                if_condition_30209 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 12), result_eq_30208)
                # Assigning a type to the variable 'if_condition_30209' (line 57)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'if_condition_30209', if_condition_30209)
                # SSA begins for if statement (line 57)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 58)
                # Processing the call arguments (line 58)
                
                # Call to create_Assign(...): (line 58)
                # Processing the call arguments (line 58)
                # Getting the type of 'temp_class_attr' (line 58)
                temp_class_attr_30214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 67), 'temp_class_attr', False)
                # Getting the type of 'attr' (line 58)
                attr_30215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 84), 'attr', False)
                # Obtaining the member 'value' of a type (line 58)
                value_30216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 84), attr_30215, 'value')
                # Processing the call keyword arguments (line 58)
                kwargs_30217 = {}
                # Getting the type of 'core_language_copy' (line 58)
                core_language_copy_30212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 34), 'core_language_copy', False)
                # Obtaining the member 'create_Assign' of a type (line 58)
                create_Assign_30213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 34), core_language_copy_30212, 'create_Assign')
                # Calling create_Assign(args, kwargs) (line 58)
                create_Assign_call_result_30218 = invoke(stypy.reporting.localization.Localization(__file__, 58, 34), create_Assign_30213, *[temp_class_attr_30214, value_30216], **kwargs_30217)
                
                # Processing the call keyword arguments (line 58)
                kwargs_30219 = {}
                # Getting the type of 'attr_stmts' (line 58)
                attr_stmts_30210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'attr_stmts', False)
                # Obtaining the member 'append' of a type (line 58)
                append_30211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), attr_stmts_30210, 'append')
                # Calling append(args, kwargs) (line 58)
                append_call_result_30220 = invoke(stypy.reporting.localization.Localization(__file__, 58, 16), append_30211, *[create_Assign_call_result_30218], **kwargs_30219)
                
                # SSA branch for the else part of an if statement (line 57)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 60):
                
                # Assigning a Call to a Name (line 60):
                
                # Call to create_attribute(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'node' (line 60)
                node_30223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 71), 'node', False)
                # Obtaining the member 'name' of a type (line 60)
                name_30224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 71), node_30223, 'name')
                # Getting the type of 'attr' (line 60)
                attr_30225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 82), 'attr', False)
                # Obtaining the member 'value' of a type (line 60)
                value_30226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 82), attr_30225, 'value')
                # Obtaining the member 'id' of a type (line 60)
                id_30227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 82), value_30226, 'id')
                # Processing the call keyword arguments (line 60)
                kwargs_30228 = {}
                # Getting the type of 'core_language_copy' (line 60)
                core_language_copy_30221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 35), 'core_language_copy', False)
                # Obtaining the member 'create_attribute' of a type (line 60)
                create_attribute_30222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 35), core_language_copy_30221, 'create_attribute')
                # Calling create_attribute(args, kwargs) (line 60)
                create_attribute_call_result_30229 = invoke(stypy.reporting.localization.Localization(__file__, 60, 35), create_attribute_30222, *[name_30224, id_30227], **kwargs_30228)
                
                # Assigning a type to the variable 'temp_class_value' (line 60)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'temp_class_value', create_attribute_call_result_30229)
                
                # Call to append(...): (line 61)
                # Processing the call arguments (line 61)
                
                # Call to create_Assign(...): (line 61)
                # Processing the call arguments (line 61)
                # Getting the type of 'temp_class_attr' (line 61)
                temp_class_attr_30234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 67), 'temp_class_attr', False)
                # Getting the type of 'temp_class_value' (line 61)
                temp_class_value_30235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 84), 'temp_class_value', False)
                # Processing the call keyword arguments (line 61)
                kwargs_30236 = {}
                # Getting the type of 'core_language_copy' (line 61)
                core_language_copy_30232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'core_language_copy', False)
                # Obtaining the member 'create_Assign' of a type (line 61)
                create_Assign_30233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 34), core_language_copy_30232, 'create_Assign')
                # Calling create_Assign(args, kwargs) (line 61)
                create_Assign_call_result_30237 = invoke(stypy.reporting.localization.Localization(__file__, 61, 34), create_Assign_30233, *[temp_class_attr_30234, temp_class_value_30235], **kwargs_30236)
                
                # Processing the call keyword arguments (line 61)
                kwargs_30238 = {}
                # Getting the type of 'attr_stmts' (line 61)
                attr_stmts_30230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'attr_stmts', False)
                # Obtaining the member 'append' of a type (line 61)
                append_30231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 16), attr_stmts_30230, 'append')
                # Calling append(args, kwargs) (line 61)
                append_call_result_30239 = invoke(stypy.reporting.localization.Localization(__file__, 61, 16), append_30231, *[create_Assign_call_result_30237], **kwargs_30238)
                
                # SSA join for if statement (line 57)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'node' (line 65)
        node_30241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'node', False)
        # Obtaining the member 'body' of a type (line 65)
        body_30242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 15), node_30241, 'body')
        # Processing the call keyword arguments (line 65)
        kwargs_30243 = {}
        # Getting the type of 'len' (line 65)
        len_30240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'len', False)
        # Calling len(args, kwargs) (line 65)
        len_call_result_30244 = invoke(stypy.reporting.localization.Localization(__file__, 65, 11), len_30240, *[body_30242], **kwargs_30243)
        
        int_30245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 29), 'int')
        # Applying the binary operator '==' (line 65)
        result_eq_30246 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 11), '==', len_call_result_30244, int_30245)
        
        # Testing if the type of an if condition is none (line 65)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 65, 8), result_eq_30246):
            pass
        else:
            
            # Testing the type of an if condition (line 65)
            if_condition_30247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 8), result_eq_30246)
            # Assigning a type to the variable 'if_condition_30247' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'if_condition_30247', if_condition_30247)
            # SSA begins for if statement (line 65)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 66)
            # Processing the call arguments (line 66)
            
            # Call to create_pass_node(...): (line 66)
            # Processing the call keyword arguments (line 66)
            kwargs_30253 = {}
            # Getting the type of 'stypy_functions_copy' (line 66)
            stypy_functions_copy_30251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 29), 'stypy_functions_copy', False)
            # Obtaining the member 'create_pass_node' of a type (line 66)
            create_pass_node_30252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 29), stypy_functions_copy_30251, 'create_pass_node')
            # Calling create_pass_node(args, kwargs) (line 66)
            create_pass_node_call_result_30254 = invoke(stypy.reporting.localization.Localization(__file__, 66, 29), create_pass_node_30252, *[], **kwargs_30253)
            
            # Processing the call keyword arguments (line 66)
            kwargs_30255 = {}
            # Getting the type of 'node' (line 66)
            node_30248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'node', False)
            # Obtaining the member 'body' of a type (line 66)
            body_30249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), node_30248, 'body')
            # Obtaining the member 'append' of a type (line 66)
            append_30250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), body_30249, 'append')
            # Calling append(args, kwargs) (line 66)
            append_call_result_30256 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), append_30250, *[create_pass_node_call_result_30254], **kwargs_30255)
            
            # SSA join for if statement (line 65)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to flatten_lists(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'node' (line 68)
        node_30259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 50), 'node', False)
        # Getting the type of 'attr_stmts' (line 68)
        attr_stmts_30260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 56), 'attr_stmts', False)
        # Processing the call keyword arguments (line 68)
        kwargs_30261 = {}
        # Getting the type of 'stypy_functions_copy' (line 68)
        stypy_functions_copy_30257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'stypy_functions_copy', False)
        # Obtaining the member 'flatten_lists' of a type (line 68)
        flatten_lists_30258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 15), stypy_functions_copy_30257, 'flatten_lists')
        # Calling flatten_lists(args, kwargs) (line 68)
        flatten_lists_call_result_30262 = invoke(stypy.reporting.localization.Localization(__file__, 68, 15), flatten_lists_30258, *[node_30259, attr_stmts_30260], **kwargs_30261)
        
        # Assigning a type to the variable 'stypy_return_type' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'stypy_return_type', flatten_lists_call_result_30262)
        
        # ################# End of 'visit_ClassDef(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ClassDef' in the type store
        # Getting the type of 'stypy_return_type' (line 40)
        stypy_return_type_30263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30263)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ClassDef'
        return stypy_return_type_30263


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
