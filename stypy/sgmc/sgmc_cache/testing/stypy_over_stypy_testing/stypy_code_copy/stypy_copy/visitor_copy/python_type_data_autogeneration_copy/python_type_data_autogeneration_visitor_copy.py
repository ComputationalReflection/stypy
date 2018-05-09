
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import ast
2: 
3: type_test_name = "type_test"
4: 
5: 
6: class TypeDataAutoGeneratorVisitor(ast.NodeVisitor):
7:     '''
8:     This visitor is used to generate a version of the original source code that dynamically captures the types of
9:     variables in functions, methods and the global scope. This program can be executed and code is inserted in key
10:     places to capture the value of the variables at the end of the execution of any of the previously mentioned
11:     elements. This has several limitations:
12:     - If a variable changes types during the execution of a function, only the last type is captured.
13:     - If the program has errors at runtime, nothing is captured
14:     - The technique may fail with certain Python constructs.
15: 
16:     In general, this visitor is only used as a helper for creating Python data files that can be used to unit test
17:     the type inference code generation modules and classes, not being a part of the end-user functionality of stypy.
18:      As a tool to facilitate the development of stypy, the code is not polished at the same level as the rest of the
19:      code, as its only function is to generate an approximation of the types of the variables that a correct execution
20:      of the tested programs should return. Normally the generated table has to be modified by hand, as this technique
21:      is not 100% accurate.
22:     '''
23: 
24:     def visit_Module(self, node):
25:         alias = ast.alias(name="TypeDataFileWriter", asname=None)
26:         import_ = ast.ImportFrom(level=0,
27:                                  module="stypy.code_generation.type_inference_programs.checking.type_data_file_writer",
28:                                  names=[alias])
29: 
30:         name = ast.Name(id="__file__", ctx=ast.Load())
31:         # attribute_module = ast.Attribute(attr="type_data_file_writer", ctx=ast.Load(),
32:         #                                   value=ast.Name(id="", ctx=ast.Load()))
33:         attribute = ast.Name(id="TypeDataFileWriter", ctx=ast.Load())  # , value=attribute_module)
34:         call = ast.Call(args=[name], func=attribute, keywords=[], kwargs=None, starargs=None)
35:         assign = ast.Assign(targets=[ast.Name(id=type_test_name, ctx=ast.Store())],
36:                             value=call)
37: 
38:         node.body.insert(0, assign)
39:         node.body.insert(0, import_)
40: 
41:         for stmt in node.body:
42:             self.visit(stmt)
43: 
44:         locals_call = ast.Call(args=[], func=ast.Name(id="globals", ctx=ast.Load()), keywords=[], kwargs=None,
45:                                starargs=None)
46:         attribute = ast.Attribute(attr="add_type_dict_for_main_context", ctx=ast.Load(),
47:                                   value=ast.Name(id=type_test_name, ctx=ast.Load()))
48: 
49:         call = ast.Call(args=[locals_call], func=attribute, keywords=[], kwargs=None, starargs=None)
50:         expr = ast.Expr(value=call)
51: 
52:         attribute_generate = ast.Attribute(attr="generate_type_data_file", ctx=ast.Load(),
53:                                            value=ast.Name(id=type_test_name, ctx=ast.Load()))
54:         call_generate = ast.Call(args=[], func=attribute_generate, keywords=[], kwargs=None, starargs=None)
55:         expr_final = ast.Expr(value=call_generate, ctx=ast.Load())
56:         node.body.append(expr)
57:         node.body.append(expr_final)
58:         return node
59: 
60:     def visit_FunctionDef(self, node):
61:         for stmt in node.body:
62:             self.visit(stmt)
63: 
64:         locals_call = ast.Call(args=[], func=ast.Name(id="locals", ctx=ast.Load()), keywords=[], kwargs=None,
65:                                starargs=None)
66:         attribute = ast.Attribute(attr="add_type_dict_for_context", ctx=ast.Load(),
67:                                   value=ast.Name(id=type_test_name, ctx=ast.Load()))
68:         call = ast.Call(args=[locals_call], func=attribute, keywords=[], kwargs=None, starargs=None)
69:         expr = ast.Expr(value=call)
70:         node.body.append(expr)
71: 
72:         return node
73: 
74:     def visit_Return(self, node):
75:         self.visit(node.value)
76: 
77:         index = ast.Index(value=ast.Num(n=0))
78:         locals_call = ast.Call(args=[], func=ast.Name(id="locals", ctx=ast.Load()), keywords=[], kwargs=None,
79:                                starargs=None)
80:         attribute = ast.Attribute(attr="add_type_dict_for_context", ctx=ast.Load(),
81:                                   value=ast.Name(id=type_test_name, ctx=ast.Load()))
82:         call = ast.Call(args=[locals_call], func=attribute, keywords=[], kwargs=None, starargs=None)
83:         tuple_ = ast.Tuple(ctx=ast.Load(), elts=[node.value, call])
84:         subscript = ast.Subscript(ctx=ast.Load(), slice=index, value=tuple_)
85:         node.value = subscript
86: 
87:         return node
88: 

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


# Assigning a Str to a Name (line 3):
str_4779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 17), 'str', 'type_test')
# Assigning a type to the variable 'type_test_name' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'type_test_name', str_4779)
# Declaration of the 'TypeDataAutoGeneratorVisitor' class
# Getting the type of 'ast' (line 6)
ast_4780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 35), 'ast')
# Obtaining the member 'NodeVisitor' of a type (line 6)
NodeVisitor_4781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 35), ast_4780, 'NodeVisitor')

class TypeDataAutoGeneratorVisitor(NodeVisitor_4781, ):
    str_4782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', '\n    This visitor is used to generate a version of the original source code that dynamically captures the types of\n    variables in functions, methods and the global scope. This program can be executed and code is inserted in key\n    places to capture the value of the variables at the end of the execution of any of the previously mentioned\n    elements. This has several limitations:\n    - If a variable changes types during the execution of a function, only the last type is captured.\n    - If the program has errors at runtime, nothing is captured\n    - The technique may fail with certain Python constructs.\n\n    In general, this visitor is only used as a helper for creating Python data files that can be used to unit test\n    the type inference code generation modules and classes, not being a part of the end-user functionality of stypy.\n     As a tool to facilitate the development of stypy, the code is not polished at the same level as the rest of the\n     code, as its only function is to generate an approximation of the types of the variables that a correct execution\n     of the tested programs should return. Normally the generated table has to be modified by hand, as this technique\n     is not 100% accurate.\n    ')

    @norecursion
    def visit_Module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Module'
        module_type_store = module_type_store.open_function_context('visit_Module', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeDataAutoGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_localization', localization)
        TypeDataAutoGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeDataAutoGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeDataAutoGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_function_name', 'TypeDataAutoGeneratorVisitor.visit_Module')
        TypeDataAutoGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_param_names_list', ['node'])
        TypeDataAutoGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeDataAutoGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeDataAutoGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeDataAutoGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeDataAutoGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeDataAutoGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeDataAutoGeneratorVisitor.visit_Module', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Module', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Module(...)' code ##################

        
        # Assigning a Call to a Name (line 25):
        
        # Call to alias(...): (line 25)
        # Processing the call keyword arguments (line 25)
        str_4785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 31), 'str', 'TypeDataFileWriter')
        keyword_4786 = str_4785
        # Getting the type of 'None' (line 25)
        None_4787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 60), 'None', False)
        keyword_4788 = None_4787
        kwargs_4789 = {'name': keyword_4786, 'asname': keyword_4788}
        # Getting the type of 'ast' (line 25)
        ast_4783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'ast', False)
        # Obtaining the member 'alias' of a type (line 25)
        alias_4784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 16), ast_4783, 'alias')
        # Calling alias(args, kwargs) (line 25)
        alias_call_result_4790 = invoke(stypy.reporting.localization.Localization(__file__, 25, 16), alias_4784, *[], **kwargs_4789)
        
        # Assigning a type to the variable 'alias' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'alias', alias_call_result_4790)
        
        # Assigning a Call to a Name (line 26):
        
        # Call to ImportFrom(...): (line 26)
        # Processing the call keyword arguments (line 26)
        int_4793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 39), 'int')
        keyword_4794 = int_4793
        str_4795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 40), 'str', 'stypy.code_generation.type_inference_programs.checking.type_data_file_writer')
        keyword_4796 = str_4795
        
        # Obtaining an instance of the builtin type 'list' (line 28)
        list_4797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 28)
        # Adding element type (line 28)
        # Getting the type of 'alias' (line 28)
        alias_4798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 40), 'alias', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 39), list_4797, alias_4798)
        
        keyword_4799 = list_4797
        kwargs_4800 = {'names': keyword_4799, 'module': keyword_4796, 'level': keyword_4794}
        # Getting the type of 'ast' (line 26)
        ast_4791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'ast', False)
        # Obtaining the member 'ImportFrom' of a type (line 26)
        ImportFrom_4792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 18), ast_4791, 'ImportFrom')
        # Calling ImportFrom(args, kwargs) (line 26)
        ImportFrom_call_result_4801 = invoke(stypy.reporting.localization.Localization(__file__, 26, 18), ImportFrom_4792, *[], **kwargs_4800)
        
        # Assigning a type to the variable 'import_' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'import_', ImportFrom_call_result_4801)
        
        # Assigning a Call to a Name (line 30):
        
        # Call to Name(...): (line 30)
        # Processing the call keyword arguments (line 30)
        str_4804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 27), 'str', '__file__')
        keyword_4805 = str_4804
        
        # Call to Load(...): (line 30)
        # Processing the call keyword arguments (line 30)
        kwargs_4808 = {}
        # Getting the type of 'ast' (line 30)
        ast_4806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 43), 'ast', False)
        # Obtaining the member 'Load' of a type (line 30)
        Load_4807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 43), ast_4806, 'Load')
        # Calling Load(args, kwargs) (line 30)
        Load_call_result_4809 = invoke(stypy.reporting.localization.Localization(__file__, 30, 43), Load_4807, *[], **kwargs_4808)
        
        keyword_4810 = Load_call_result_4809
        kwargs_4811 = {'ctx': keyword_4810, 'id': keyword_4805}
        # Getting the type of 'ast' (line 30)
        ast_4802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'ast', False)
        # Obtaining the member 'Name' of a type (line 30)
        Name_4803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 15), ast_4802, 'Name')
        # Calling Name(args, kwargs) (line 30)
        Name_call_result_4812 = invoke(stypy.reporting.localization.Localization(__file__, 30, 15), Name_4803, *[], **kwargs_4811)
        
        # Assigning a type to the variable 'name' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'name', Name_call_result_4812)
        
        # Assigning a Call to a Name (line 33):
        
        # Call to Name(...): (line 33)
        # Processing the call keyword arguments (line 33)
        str_4815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 32), 'str', 'TypeDataFileWriter')
        keyword_4816 = str_4815
        
        # Call to Load(...): (line 33)
        # Processing the call keyword arguments (line 33)
        kwargs_4819 = {}
        # Getting the type of 'ast' (line 33)
        ast_4817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 58), 'ast', False)
        # Obtaining the member 'Load' of a type (line 33)
        Load_4818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 58), ast_4817, 'Load')
        # Calling Load(args, kwargs) (line 33)
        Load_call_result_4820 = invoke(stypy.reporting.localization.Localization(__file__, 33, 58), Load_4818, *[], **kwargs_4819)
        
        keyword_4821 = Load_call_result_4820
        kwargs_4822 = {'ctx': keyword_4821, 'id': keyword_4816}
        # Getting the type of 'ast' (line 33)
        ast_4813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'ast', False)
        # Obtaining the member 'Name' of a type (line 33)
        Name_4814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 20), ast_4813, 'Name')
        # Calling Name(args, kwargs) (line 33)
        Name_call_result_4823 = invoke(stypy.reporting.localization.Localization(__file__, 33, 20), Name_4814, *[], **kwargs_4822)
        
        # Assigning a type to the variable 'attribute' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'attribute', Name_call_result_4823)
        
        # Assigning a Call to a Name (line 34):
        
        # Call to Call(...): (line 34)
        # Processing the call keyword arguments (line 34)
        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_4826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        # Adding element type (line 34)
        # Getting the type of 'name' (line 34)
        name_4827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 29), list_4826, name_4827)
        
        keyword_4828 = list_4826
        # Getting the type of 'attribute' (line 34)
        attribute_4829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 42), 'attribute', False)
        keyword_4830 = attribute_4829
        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_4831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 62), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        
        keyword_4832 = list_4831
        # Getting the type of 'None' (line 34)
        None_4833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 73), 'None', False)
        keyword_4834 = None_4833
        # Getting the type of 'None' (line 34)
        None_4835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 88), 'None', False)
        keyword_4836 = None_4835
        kwargs_4837 = {'keywords': keyword_4832, 'starargs': keyword_4836, 'args': keyword_4828, 'func': keyword_4830, 'kwargs': keyword_4834}
        # Getting the type of 'ast' (line 34)
        ast_4824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'ast', False)
        # Obtaining the member 'Call' of a type (line 34)
        Call_4825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 15), ast_4824, 'Call')
        # Calling Call(args, kwargs) (line 34)
        Call_call_result_4838 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), Call_4825, *[], **kwargs_4837)
        
        # Assigning a type to the variable 'call' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'call', Call_call_result_4838)
        
        # Assigning a Call to a Name (line 35):
        
        # Call to Assign(...): (line 35)
        # Processing the call keyword arguments (line 35)
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_4841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        # Adding element type (line 35)
        
        # Call to Name(...): (line 35)
        # Processing the call keyword arguments (line 35)
        # Getting the type of 'type_test_name' (line 35)
        type_test_name_4844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 49), 'type_test_name', False)
        keyword_4845 = type_test_name_4844
        
        # Call to Store(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_4848 = {}
        # Getting the type of 'ast' (line 35)
        ast_4846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 69), 'ast', False)
        # Obtaining the member 'Store' of a type (line 35)
        Store_4847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 69), ast_4846, 'Store')
        # Calling Store(args, kwargs) (line 35)
        Store_call_result_4849 = invoke(stypy.reporting.localization.Localization(__file__, 35, 69), Store_4847, *[], **kwargs_4848)
        
        keyword_4850 = Store_call_result_4849
        kwargs_4851 = {'ctx': keyword_4850, 'id': keyword_4845}
        # Getting the type of 'ast' (line 35)
        ast_4842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 37), 'ast', False)
        # Obtaining the member 'Name' of a type (line 35)
        Name_4843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 37), ast_4842, 'Name')
        # Calling Name(args, kwargs) (line 35)
        Name_call_result_4852 = invoke(stypy.reporting.localization.Localization(__file__, 35, 37), Name_4843, *[], **kwargs_4851)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 36), list_4841, Name_call_result_4852)
        
        keyword_4853 = list_4841
        # Getting the type of 'call' (line 36)
        call_4854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 34), 'call', False)
        keyword_4855 = call_4854
        kwargs_4856 = {'targets': keyword_4853, 'value': keyword_4855}
        # Getting the type of 'ast' (line 35)
        ast_4839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'ast', False)
        # Obtaining the member 'Assign' of a type (line 35)
        Assign_4840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 17), ast_4839, 'Assign')
        # Calling Assign(args, kwargs) (line 35)
        Assign_call_result_4857 = invoke(stypy.reporting.localization.Localization(__file__, 35, 17), Assign_4840, *[], **kwargs_4856)
        
        # Assigning a type to the variable 'assign' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'assign', Assign_call_result_4857)
        
        # Call to insert(...): (line 38)
        # Processing the call arguments (line 38)
        int_4861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 25), 'int')
        # Getting the type of 'assign' (line 38)
        assign_4862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 28), 'assign', False)
        # Processing the call keyword arguments (line 38)
        kwargs_4863 = {}
        # Getting the type of 'node' (line 38)
        node_4858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'node', False)
        # Obtaining the member 'body' of a type (line 38)
        body_4859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), node_4858, 'body')
        # Obtaining the member 'insert' of a type (line 38)
        insert_4860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), body_4859, 'insert')
        # Calling insert(args, kwargs) (line 38)
        insert_call_result_4864 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), insert_4860, *[int_4861, assign_4862], **kwargs_4863)
        
        
        # Call to insert(...): (line 39)
        # Processing the call arguments (line 39)
        int_4868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'int')
        # Getting the type of 'import_' (line 39)
        import__4869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 28), 'import_', False)
        # Processing the call keyword arguments (line 39)
        kwargs_4870 = {}
        # Getting the type of 'node' (line 39)
        node_4865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'node', False)
        # Obtaining the member 'body' of a type (line 39)
        body_4866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), node_4865, 'body')
        # Obtaining the member 'insert' of a type (line 39)
        insert_4867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), body_4866, 'insert')
        # Calling insert(args, kwargs) (line 39)
        insert_call_result_4871 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), insert_4867, *[int_4868, import__4869], **kwargs_4870)
        
        
        # Getting the type of 'node' (line 41)
        node_4872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'node')
        # Obtaining the member 'body' of a type (line 41)
        body_4873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 20), node_4872, 'body')
        # Assigning a type to the variable 'body_4873' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'body_4873', body_4873)
        # Testing if the for loop is going to be iterated (line 41)
        # Testing the type of a for loop iterable (line 41)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 41, 8), body_4873)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 41, 8), body_4873):
            # Getting the type of the for loop variable (line 41)
            for_loop_var_4874 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 41, 8), body_4873)
            # Assigning a type to the variable 'stmt' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stmt', for_loop_var_4874)
            # SSA begins for a for statement (line 41)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 42)
            # Processing the call arguments (line 42)
            # Getting the type of 'stmt' (line 42)
            stmt_4877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'stmt', False)
            # Processing the call keyword arguments (line 42)
            kwargs_4878 = {}
            # Getting the type of 'self' (line 42)
            self_4875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 42)
            visit_4876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), self_4875, 'visit')
            # Calling visit(args, kwargs) (line 42)
            visit_call_result_4879 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), visit_4876, *[stmt_4877], **kwargs_4878)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 44):
        
        # Call to Call(...): (line 44)
        # Processing the call keyword arguments (line 44)
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_4882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        
        keyword_4883 = list_4882
        
        # Call to Name(...): (line 44)
        # Processing the call keyword arguments (line 44)
        str_4886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 57), 'str', 'globals')
        keyword_4887 = str_4886
        
        # Call to Load(...): (line 44)
        # Processing the call keyword arguments (line 44)
        kwargs_4890 = {}
        # Getting the type of 'ast' (line 44)
        ast_4888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 72), 'ast', False)
        # Obtaining the member 'Load' of a type (line 44)
        Load_4889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 72), ast_4888, 'Load')
        # Calling Load(args, kwargs) (line 44)
        Load_call_result_4891 = invoke(stypy.reporting.localization.Localization(__file__, 44, 72), Load_4889, *[], **kwargs_4890)
        
        keyword_4892 = Load_call_result_4891
        kwargs_4893 = {'ctx': keyword_4892, 'id': keyword_4887}
        # Getting the type of 'ast' (line 44)
        ast_4884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 45), 'ast', False)
        # Obtaining the member 'Name' of a type (line 44)
        Name_4885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 45), ast_4884, 'Name')
        # Calling Name(args, kwargs) (line 44)
        Name_call_result_4894 = invoke(stypy.reporting.localization.Localization(__file__, 44, 45), Name_4885, *[], **kwargs_4893)
        
        keyword_4895 = Name_call_result_4894
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_4896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 94), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        
        keyword_4897 = list_4896
        # Getting the type of 'None' (line 44)
        None_4898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 105), 'None', False)
        keyword_4899 = None_4898
        # Getting the type of 'None' (line 45)
        None_4900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 40), 'None', False)
        keyword_4901 = None_4900
        kwargs_4902 = {'keywords': keyword_4897, 'starargs': keyword_4901, 'args': keyword_4883, 'func': keyword_4895, 'kwargs': keyword_4899}
        # Getting the type of 'ast' (line 44)
        ast_4880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'ast', False)
        # Obtaining the member 'Call' of a type (line 44)
        Call_4881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 22), ast_4880, 'Call')
        # Calling Call(args, kwargs) (line 44)
        Call_call_result_4903 = invoke(stypy.reporting.localization.Localization(__file__, 44, 22), Call_4881, *[], **kwargs_4902)
        
        # Assigning a type to the variable 'locals_call' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'locals_call', Call_call_result_4903)
        
        # Assigning a Call to a Name (line 46):
        
        # Call to Attribute(...): (line 46)
        # Processing the call keyword arguments (line 46)
        str_4906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 39), 'str', 'add_type_dict_for_main_context')
        keyword_4907 = str_4906
        
        # Call to Load(...): (line 46)
        # Processing the call keyword arguments (line 46)
        kwargs_4910 = {}
        # Getting the type of 'ast' (line 46)
        ast_4908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 77), 'ast', False)
        # Obtaining the member 'Load' of a type (line 46)
        Load_4909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 77), ast_4908, 'Load')
        # Calling Load(args, kwargs) (line 46)
        Load_call_result_4911 = invoke(stypy.reporting.localization.Localization(__file__, 46, 77), Load_4909, *[], **kwargs_4910)
        
        keyword_4912 = Load_call_result_4911
        
        # Call to Name(...): (line 47)
        # Processing the call keyword arguments (line 47)
        # Getting the type of 'type_test_name' (line 47)
        type_test_name_4915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 52), 'type_test_name', False)
        keyword_4916 = type_test_name_4915
        
        # Call to Load(...): (line 47)
        # Processing the call keyword arguments (line 47)
        kwargs_4919 = {}
        # Getting the type of 'ast' (line 47)
        ast_4917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 72), 'ast', False)
        # Obtaining the member 'Load' of a type (line 47)
        Load_4918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 72), ast_4917, 'Load')
        # Calling Load(args, kwargs) (line 47)
        Load_call_result_4920 = invoke(stypy.reporting.localization.Localization(__file__, 47, 72), Load_4918, *[], **kwargs_4919)
        
        keyword_4921 = Load_call_result_4920
        kwargs_4922 = {'ctx': keyword_4921, 'id': keyword_4916}
        # Getting the type of 'ast' (line 47)
        ast_4913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 40), 'ast', False)
        # Obtaining the member 'Name' of a type (line 47)
        Name_4914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 40), ast_4913, 'Name')
        # Calling Name(args, kwargs) (line 47)
        Name_call_result_4923 = invoke(stypy.reporting.localization.Localization(__file__, 47, 40), Name_4914, *[], **kwargs_4922)
        
        keyword_4924 = Name_call_result_4923
        kwargs_4925 = {'ctx': keyword_4912, 'attr': keyword_4907, 'value': keyword_4924}
        # Getting the type of 'ast' (line 46)
        ast_4904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'ast', False)
        # Obtaining the member 'Attribute' of a type (line 46)
        Attribute_4905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 20), ast_4904, 'Attribute')
        # Calling Attribute(args, kwargs) (line 46)
        Attribute_call_result_4926 = invoke(stypy.reporting.localization.Localization(__file__, 46, 20), Attribute_4905, *[], **kwargs_4925)
        
        # Assigning a type to the variable 'attribute' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'attribute', Attribute_call_result_4926)
        
        # Assigning a Call to a Name (line 49):
        
        # Call to Call(...): (line 49)
        # Processing the call keyword arguments (line 49)
        
        # Obtaining an instance of the builtin type 'list' (line 49)
        list_4929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 49)
        # Adding element type (line 49)
        # Getting the type of 'locals_call' (line 49)
        locals_call_4930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'locals_call', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 29), list_4929, locals_call_4930)
        
        keyword_4931 = list_4929
        # Getting the type of 'attribute' (line 49)
        attribute_4932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 49), 'attribute', False)
        keyword_4933 = attribute_4932
        
        # Obtaining an instance of the builtin type 'list' (line 49)
        list_4934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 69), 'list')
        # Adding type elements to the builtin type 'list' instance (line 49)
        
        keyword_4935 = list_4934
        # Getting the type of 'None' (line 49)
        None_4936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 80), 'None', False)
        keyword_4937 = None_4936
        # Getting the type of 'None' (line 49)
        None_4938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 95), 'None', False)
        keyword_4939 = None_4938
        kwargs_4940 = {'keywords': keyword_4935, 'starargs': keyword_4939, 'args': keyword_4931, 'func': keyword_4933, 'kwargs': keyword_4937}
        # Getting the type of 'ast' (line 49)
        ast_4927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'ast', False)
        # Obtaining the member 'Call' of a type (line 49)
        Call_4928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 15), ast_4927, 'Call')
        # Calling Call(args, kwargs) (line 49)
        Call_call_result_4941 = invoke(stypy.reporting.localization.Localization(__file__, 49, 15), Call_4928, *[], **kwargs_4940)
        
        # Assigning a type to the variable 'call' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'call', Call_call_result_4941)
        
        # Assigning a Call to a Name (line 50):
        
        # Call to Expr(...): (line 50)
        # Processing the call keyword arguments (line 50)
        # Getting the type of 'call' (line 50)
        call_4944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 30), 'call', False)
        keyword_4945 = call_4944
        kwargs_4946 = {'value': keyword_4945}
        # Getting the type of 'ast' (line 50)
        ast_4942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'ast', False)
        # Obtaining the member 'Expr' of a type (line 50)
        Expr_4943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 15), ast_4942, 'Expr')
        # Calling Expr(args, kwargs) (line 50)
        Expr_call_result_4947 = invoke(stypy.reporting.localization.Localization(__file__, 50, 15), Expr_4943, *[], **kwargs_4946)
        
        # Assigning a type to the variable 'expr' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'expr', Expr_call_result_4947)
        
        # Assigning a Call to a Name (line 52):
        
        # Call to Attribute(...): (line 52)
        # Processing the call keyword arguments (line 52)
        str_4950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 48), 'str', 'generate_type_data_file')
        keyword_4951 = str_4950
        
        # Call to Load(...): (line 52)
        # Processing the call keyword arguments (line 52)
        kwargs_4954 = {}
        # Getting the type of 'ast' (line 52)
        ast_4952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 79), 'ast', False)
        # Obtaining the member 'Load' of a type (line 52)
        Load_4953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 79), ast_4952, 'Load')
        # Calling Load(args, kwargs) (line 52)
        Load_call_result_4955 = invoke(stypy.reporting.localization.Localization(__file__, 52, 79), Load_4953, *[], **kwargs_4954)
        
        keyword_4956 = Load_call_result_4955
        
        # Call to Name(...): (line 53)
        # Processing the call keyword arguments (line 53)
        # Getting the type of 'type_test_name' (line 53)
        type_test_name_4959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 61), 'type_test_name', False)
        keyword_4960 = type_test_name_4959
        
        # Call to Load(...): (line 53)
        # Processing the call keyword arguments (line 53)
        kwargs_4963 = {}
        # Getting the type of 'ast' (line 53)
        ast_4961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 81), 'ast', False)
        # Obtaining the member 'Load' of a type (line 53)
        Load_4962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 81), ast_4961, 'Load')
        # Calling Load(args, kwargs) (line 53)
        Load_call_result_4964 = invoke(stypy.reporting.localization.Localization(__file__, 53, 81), Load_4962, *[], **kwargs_4963)
        
        keyword_4965 = Load_call_result_4964
        kwargs_4966 = {'ctx': keyword_4965, 'id': keyword_4960}
        # Getting the type of 'ast' (line 53)
        ast_4957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 49), 'ast', False)
        # Obtaining the member 'Name' of a type (line 53)
        Name_4958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 49), ast_4957, 'Name')
        # Calling Name(args, kwargs) (line 53)
        Name_call_result_4967 = invoke(stypy.reporting.localization.Localization(__file__, 53, 49), Name_4958, *[], **kwargs_4966)
        
        keyword_4968 = Name_call_result_4967
        kwargs_4969 = {'ctx': keyword_4956, 'attr': keyword_4951, 'value': keyword_4968}
        # Getting the type of 'ast' (line 52)
        ast_4948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 29), 'ast', False)
        # Obtaining the member 'Attribute' of a type (line 52)
        Attribute_4949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 29), ast_4948, 'Attribute')
        # Calling Attribute(args, kwargs) (line 52)
        Attribute_call_result_4970 = invoke(stypy.reporting.localization.Localization(__file__, 52, 29), Attribute_4949, *[], **kwargs_4969)
        
        # Assigning a type to the variable 'attribute_generate' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'attribute_generate', Attribute_call_result_4970)
        
        # Assigning a Call to a Name (line 54):
        
        # Call to Call(...): (line 54)
        # Processing the call keyword arguments (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_4973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        
        keyword_4974 = list_4973
        # Getting the type of 'attribute_generate' (line 54)
        attribute_generate_4975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 47), 'attribute_generate', False)
        keyword_4976 = attribute_generate_4975
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_4977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 76), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        
        keyword_4978 = list_4977
        # Getting the type of 'None' (line 54)
        None_4979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 87), 'None', False)
        keyword_4980 = None_4979
        # Getting the type of 'None' (line 54)
        None_4981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 102), 'None', False)
        keyword_4982 = None_4981
        kwargs_4983 = {'keywords': keyword_4978, 'starargs': keyword_4982, 'args': keyword_4974, 'func': keyword_4976, 'kwargs': keyword_4980}
        # Getting the type of 'ast' (line 54)
        ast_4971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'ast', False)
        # Obtaining the member 'Call' of a type (line 54)
        Call_4972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 24), ast_4971, 'Call')
        # Calling Call(args, kwargs) (line 54)
        Call_call_result_4984 = invoke(stypy.reporting.localization.Localization(__file__, 54, 24), Call_4972, *[], **kwargs_4983)
        
        # Assigning a type to the variable 'call_generate' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'call_generate', Call_call_result_4984)
        
        # Assigning a Call to a Name (line 55):
        
        # Call to Expr(...): (line 55)
        # Processing the call keyword arguments (line 55)
        # Getting the type of 'call_generate' (line 55)
        call_generate_4987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 36), 'call_generate', False)
        keyword_4988 = call_generate_4987
        
        # Call to Load(...): (line 55)
        # Processing the call keyword arguments (line 55)
        kwargs_4991 = {}
        # Getting the type of 'ast' (line 55)
        ast_4989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 55), 'ast', False)
        # Obtaining the member 'Load' of a type (line 55)
        Load_4990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 55), ast_4989, 'Load')
        # Calling Load(args, kwargs) (line 55)
        Load_call_result_4992 = invoke(stypy.reporting.localization.Localization(__file__, 55, 55), Load_4990, *[], **kwargs_4991)
        
        keyword_4993 = Load_call_result_4992
        kwargs_4994 = {'ctx': keyword_4993, 'value': keyword_4988}
        # Getting the type of 'ast' (line 55)
        ast_4985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'ast', False)
        # Obtaining the member 'Expr' of a type (line 55)
        Expr_4986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 21), ast_4985, 'Expr')
        # Calling Expr(args, kwargs) (line 55)
        Expr_call_result_4995 = invoke(stypy.reporting.localization.Localization(__file__, 55, 21), Expr_4986, *[], **kwargs_4994)
        
        # Assigning a type to the variable 'expr_final' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'expr_final', Expr_call_result_4995)
        
        # Call to append(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'expr' (line 56)
        expr_4999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'expr', False)
        # Processing the call keyword arguments (line 56)
        kwargs_5000 = {}
        # Getting the type of 'node' (line 56)
        node_4996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'node', False)
        # Obtaining the member 'body' of a type (line 56)
        body_4997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), node_4996, 'body')
        # Obtaining the member 'append' of a type (line 56)
        append_4998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), body_4997, 'append')
        # Calling append(args, kwargs) (line 56)
        append_call_result_5001 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), append_4998, *[expr_4999], **kwargs_5000)
        
        
        # Call to append(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'expr_final' (line 57)
        expr_final_5005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'expr_final', False)
        # Processing the call keyword arguments (line 57)
        kwargs_5006 = {}
        # Getting the type of 'node' (line 57)
        node_5002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'node', False)
        # Obtaining the member 'body' of a type (line 57)
        body_5003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), node_5002, 'body')
        # Obtaining the member 'append' of a type (line 57)
        append_5004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), body_5003, 'append')
        # Calling append(args, kwargs) (line 57)
        append_call_result_5007 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), append_5004, *[expr_final_5005], **kwargs_5006)
        
        # Getting the type of 'node' (line 58)
        node_5008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'stypy_return_type', node_5008)
        
        # ################# End of 'visit_Module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Module' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_5009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5009)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Module'
        return stypy_return_type_5009


    @norecursion
    def visit_FunctionDef(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_FunctionDef'
        module_type_store = module_type_store.open_function_context('visit_FunctionDef', 60, 4, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeDataAutoGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_localization', localization)
        TypeDataAutoGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeDataAutoGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeDataAutoGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_function_name', 'TypeDataAutoGeneratorVisitor.visit_FunctionDef')
        TypeDataAutoGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_param_names_list', ['node'])
        TypeDataAutoGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeDataAutoGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeDataAutoGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeDataAutoGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeDataAutoGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeDataAutoGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeDataAutoGeneratorVisitor.visit_FunctionDef', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_FunctionDef', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_FunctionDef(...)' code ##################

        
        # Getting the type of 'node' (line 61)
        node_5010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'node')
        # Obtaining the member 'body' of a type (line 61)
        body_5011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 20), node_5010, 'body')
        # Assigning a type to the variable 'body_5011' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'body_5011', body_5011)
        # Testing if the for loop is going to be iterated (line 61)
        # Testing the type of a for loop iterable (line 61)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 61, 8), body_5011)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 61, 8), body_5011):
            # Getting the type of the for loop variable (line 61)
            for_loop_var_5012 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 61, 8), body_5011)
            # Assigning a type to the variable 'stmt' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'stmt', for_loop_var_5012)
            # SSA begins for a for statement (line 61)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 62)
            # Processing the call arguments (line 62)
            # Getting the type of 'stmt' (line 62)
            stmt_5015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'stmt', False)
            # Processing the call keyword arguments (line 62)
            kwargs_5016 = {}
            # Getting the type of 'self' (line 62)
            self_5013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 62)
            visit_5014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), self_5013, 'visit')
            # Calling visit(args, kwargs) (line 62)
            visit_call_result_5017 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), visit_5014, *[stmt_5015], **kwargs_5016)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 64):
        
        # Call to Call(...): (line 64)
        # Processing the call keyword arguments (line 64)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_5020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        
        keyword_5021 = list_5020
        
        # Call to Name(...): (line 64)
        # Processing the call keyword arguments (line 64)
        str_5024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 57), 'str', 'locals')
        keyword_5025 = str_5024
        
        # Call to Load(...): (line 64)
        # Processing the call keyword arguments (line 64)
        kwargs_5028 = {}
        # Getting the type of 'ast' (line 64)
        ast_5026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 71), 'ast', False)
        # Obtaining the member 'Load' of a type (line 64)
        Load_5027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 71), ast_5026, 'Load')
        # Calling Load(args, kwargs) (line 64)
        Load_call_result_5029 = invoke(stypy.reporting.localization.Localization(__file__, 64, 71), Load_5027, *[], **kwargs_5028)
        
        keyword_5030 = Load_call_result_5029
        kwargs_5031 = {'ctx': keyword_5030, 'id': keyword_5025}
        # Getting the type of 'ast' (line 64)
        ast_5022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 45), 'ast', False)
        # Obtaining the member 'Name' of a type (line 64)
        Name_5023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 45), ast_5022, 'Name')
        # Calling Name(args, kwargs) (line 64)
        Name_call_result_5032 = invoke(stypy.reporting.localization.Localization(__file__, 64, 45), Name_5023, *[], **kwargs_5031)
        
        keyword_5033 = Name_call_result_5032
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_5034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 93), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        
        keyword_5035 = list_5034
        # Getting the type of 'None' (line 64)
        None_5036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 104), 'None', False)
        keyword_5037 = None_5036
        # Getting the type of 'None' (line 65)
        None_5038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 40), 'None', False)
        keyword_5039 = None_5038
        kwargs_5040 = {'keywords': keyword_5035, 'starargs': keyword_5039, 'args': keyword_5021, 'func': keyword_5033, 'kwargs': keyword_5037}
        # Getting the type of 'ast' (line 64)
        ast_5018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 22), 'ast', False)
        # Obtaining the member 'Call' of a type (line 64)
        Call_5019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 22), ast_5018, 'Call')
        # Calling Call(args, kwargs) (line 64)
        Call_call_result_5041 = invoke(stypy.reporting.localization.Localization(__file__, 64, 22), Call_5019, *[], **kwargs_5040)
        
        # Assigning a type to the variable 'locals_call' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'locals_call', Call_call_result_5041)
        
        # Assigning a Call to a Name (line 66):
        
        # Call to Attribute(...): (line 66)
        # Processing the call keyword arguments (line 66)
        str_5044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 39), 'str', 'add_type_dict_for_context')
        keyword_5045 = str_5044
        
        # Call to Load(...): (line 66)
        # Processing the call keyword arguments (line 66)
        kwargs_5048 = {}
        # Getting the type of 'ast' (line 66)
        ast_5046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 72), 'ast', False)
        # Obtaining the member 'Load' of a type (line 66)
        Load_5047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 72), ast_5046, 'Load')
        # Calling Load(args, kwargs) (line 66)
        Load_call_result_5049 = invoke(stypy.reporting.localization.Localization(__file__, 66, 72), Load_5047, *[], **kwargs_5048)
        
        keyword_5050 = Load_call_result_5049
        
        # Call to Name(...): (line 67)
        # Processing the call keyword arguments (line 67)
        # Getting the type of 'type_test_name' (line 67)
        type_test_name_5053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 52), 'type_test_name', False)
        keyword_5054 = type_test_name_5053
        
        # Call to Load(...): (line 67)
        # Processing the call keyword arguments (line 67)
        kwargs_5057 = {}
        # Getting the type of 'ast' (line 67)
        ast_5055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 72), 'ast', False)
        # Obtaining the member 'Load' of a type (line 67)
        Load_5056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 72), ast_5055, 'Load')
        # Calling Load(args, kwargs) (line 67)
        Load_call_result_5058 = invoke(stypy.reporting.localization.Localization(__file__, 67, 72), Load_5056, *[], **kwargs_5057)
        
        keyword_5059 = Load_call_result_5058
        kwargs_5060 = {'ctx': keyword_5059, 'id': keyword_5054}
        # Getting the type of 'ast' (line 67)
        ast_5051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 40), 'ast', False)
        # Obtaining the member 'Name' of a type (line 67)
        Name_5052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 40), ast_5051, 'Name')
        # Calling Name(args, kwargs) (line 67)
        Name_call_result_5061 = invoke(stypy.reporting.localization.Localization(__file__, 67, 40), Name_5052, *[], **kwargs_5060)
        
        keyword_5062 = Name_call_result_5061
        kwargs_5063 = {'ctx': keyword_5050, 'attr': keyword_5045, 'value': keyword_5062}
        # Getting the type of 'ast' (line 66)
        ast_5042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'ast', False)
        # Obtaining the member 'Attribute' of a type (line 66)
        Attribute_5043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 20), ast_5042, 'Attribute')
        # Calling Attribute(args, kwargs) (line 66)
        Attribute_call_result_5064 = invoke(stypy.reporting.localization.Localization(__file__, 66, 20), Attribute_5043, *[], **kwargs_5063)
        
        # Assigning a type to the variable 'attribute' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'attribute', Attribute_call_result_5064)
        
        # Assigning a Call to a Name (line 68):
        
        # Call to Call(...): (line 68)
        # Processing the call keyword arguments (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_5067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        # Getting the type of 'locals_call' (line 68)
        locals_call_5068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'locals_call', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 29), list_5067, locals_call_5068)
        
        keyword_5069 = list_5067
        # Getting the type of 'attribute' (line 68)
        attribute_5070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 49), 'attribute', False)
        keyword_5071 = attribute_5070
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_5072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 69), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        
        keyword_5073 = list_5072
        # Getting the type of 'None' (line 68)
        None_5074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 80), 'None', False)
        keyword_5075 = None_5074
        # Getting the type of 'None' (line 68)
        None_5076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 95), 'None', False)
        keyword_5077 = None_5076
        kwargs_5078 = {'keywords': keyword_5073, 'starargs': keyword_5077, 'args': keyword_5069, 'func': keyword_5071, 'kwargs': keyword_5075}
        # Getting the type of 'ast' (line 68)
        ast_5065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'ast', False)
        # Obtaining the member 'Call' of a type (line 68)
        Call_5066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 15), ast_5065, 'Call')
        # Calling Call(args, kwargs) (line 68)
        Call_call_result_5079 = invoke(stypy.reporting.localization.Localization(__file__, 68, 15), Call_5066, *[], **kwargs_5078)
        
        # Assigning a type to the variable 'call' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'call', Call_call_result_5079)
        
        # Assigning a Call to a Name (line 69):
        
        # Call to Expr(...): (line 69)
        # Processing the call keyword arguments (line 69)
        # Getting the type of 'call' (line 69)
        call_5082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 30), 'call', False)
        keyword_5083 = call_5082
        kwargs_5084 = {'value': keyword_5083}
        # Getting the type of 'ast' (line 69)
        ast_5080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'ast', False)
        # Obtaining the member 'Expr' of a type (line 69)
        Expr_5081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 15), ast_5080, 'Expr')
        # Calling Expr(args, kwargs) (line 69)
        Expr_call_result_5085 = invoke(stypy.reporting.localization.Localization(__file__, 69, 15), Expr_5081, *[], **kwargs_5084)
        
        # Assigning a type to the variable 'expr' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'expr', Expr_call_result_5085)
        
        # Call to append(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'expr' (line 70)
        expr_5089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'expr', False)
        # Processing the call keyword arguments (line 70)
        kwargs_5090 = {}
        # Getting the type of 'node' (line 70)
        node_5086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'node', False)
        # Obtaining the member 'body' of a type (line 70)
        body_5087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), node_5086, 'body')
        # Obtaining the member 'append' of a type (line 70)
        append_5088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), body_5087, 'append')
        # Calling append(args, kwargs) (line 70)
        append_call_result_5091 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), append_5088, *[expr_5089], **kwargs_5090)
        
        # Getting the type of 'node' (line 72)
        node_5092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'stypy_return_type', node_5092)
        
        # ################# End of 'visit_FunctionDef(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_FunctionDef' in the type store
        # Getting the type of 'stypy_return_type' (line 60)
        stypy_return_type_5093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5093)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_FunctionDef'
        return stypy_return_type_5093


    @norecursion
    def visit_Return(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Return'
        module_type_store = module_type_store.open_function_context('visit_Return', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeDataAutoGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_localization', localization)
        TypeDataAutoGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeDataAutoGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeDataAutoGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_function_name', 'TypeDataAutoGeneratorVisitor.visit_Return')
        TypeDataAutoGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_param_names_list', ['node'])
        TypeDataAutoGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeDataAutoGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeDataAutoGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeDataAutoGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeDataAutoGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeDataAutoGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeDataAutoGeneratorVisitor.visit_Return', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Return', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Return(...)' code ##################

        
        # Call to visit(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'node' (line 75)
        node_5096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 19), 'node', False)
        # Obtaining the member 'value' of a type (line 75)
        value_5097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 19), node_5096, 'value')
        # Processing the call keyword arguments (line 75)
        kwargs_5098 = {}
        # Getting the type of 'self' (line 75)
        self_5094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 75)
        visit_5095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_5094, 'visit')
        # Calling visit(args, kwargs) (line 75)
        visit_call_result_5099 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), visit_5095, *[value_5097], **kwargs_5098)
        
        
        # Assigning a Call to a Name (line 77):
        
        # Call to Index(...): (line 77)
        # Processing the call keyword arguments (line 77)
        
        # Call to Num(...): (line 77)
        # Processing the call keyword arguments (line 77)
        int_5104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 42), 'int')
        keyword_5105 = int_5104
        kwargs_5106 = {'n': keyword_5105}
        # Getting the type of 'ast' (line 77)
        ast_5102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 32), 'ast', False)
        # Obtaining the member 'Num' of a type (line 77)
        Num_5103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 32), ast_5102, 'Num')
        # Calling Num(args, kwargs) (line 77)
        Num_call_result_5107 = invoke(stypy.reporting.localization.Localization(__file__, 77, 32), Num_5103, *[], **kwargs_5106)
        
        keyword_5108 = Num_call_result_5107
        kwargs_5109 = {'value': keyword_5108}
        # Getting the type of 'ast' (line 77)
        ast_5100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'ast', False)
        # Obtaining the member 'Index' of a type (line 77)
        Index_5101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 16), ast_5100, 'Index')
        # Calling Index(args, kwargs) (line 77)
        Index_call_result_5110 = invoke(stypy.reporting.localization.Localization(__file__, 77, 16), Index_5101, *[], **kwargs_5109)
        
        # Assigning a type to the variable 'index' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'index', Index_call_result_5110)
        
        # Assigning a Call to a Name (line 78):
        
        # Call to Call(...): (line 78)
        # Processing the call keyword arguments (line 78)
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_5113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        
        keyword_5114 = list_5113
        
        # Call to Name(...): (line 78)
        # Processing the call keyword arguments (line 78)
        str_5117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 57), 'str', 'locals')
        keyword_5118 = str_5117
        
        # Call to Load(...): (line 78)
        # Processing the call keyword arguments (line 78)
        kwargs_5121 = {}
        # Getting the type of 'ast' (line 78)
        ast_5119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 71), 'ast', False)
        # Obtaining the member 'Load' of a type (line 78)
        Load_5120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 71), ast_5119, 'Load')
        # Calling Load(args, kwargs) (line 78)
        Load_call_result_5122 = invoke(stypy.reporting.localization.Localization(__file__, 78, 71), Load_5120, *[], **kwargs_5121)
        
        keyword_5123 = Load_call_result_5122
        kwargs_5124 = {'ctx': keyword_5123, 'id': keyword_5118}
        # Getting the type of 'ast' (line 78)
        ast_5115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 45), 'ast', False)
        # Obtaining the member 'Name' of a type (line 78)
        Name_5116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 45), ast_5115, 'Name')
        # Calling Name(args, kwargs) (line 78)
        Name_call_result_5125 = invoke(stypy.reporting.localization.Localization(__file__, 78, 45), Name_5116, *[], **kwargs_5124)
        
        keyword_5126 = Name_call_result_5125
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_5127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 93), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        
        keyword_5128 = list_5127
        # Getting the type of 'None' (line 78)
        None_5129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 104), 'None', False)
        keyword_5130 = None_5129
        # Getting the type of 'None' (line 79)
        None_5131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 40), 'None', False)
        keyword_5132 = None_5131
        kwargs_5133 = {'keywords': keyword_5128, 'starargs': keyword_5132, 'args': keyword_5114, 'func': keyword_5126, 'kwargs': keyword_5130}
        # Getting the type of 'ast' (line 78)
        ast_5111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 22), 'ast', False)
        # Obtaining the member 'Call' of a type (line 78)
        Call_5112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 22), ast_5111, 'Call')
        # Calling Call(args, kwargs) (line 78)
        Call_call_result_5134 = invoke(stypy.reporting.localization.Localization(__file__, 78, 22), Call_5112, *[], **kwargs_5133)
        
        # Assigning a type to the variable 'locals_call' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'locals_call', Call_call_result_5134)
        
        # Assigning a Call to a Name (line 80):
        
        # Call to Attribute(...): (line 80)
        # Processing the call keyword arguments (line 80)
        str_5137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 39), 'str', 'add_type_dict_for_context')
        keyword_5138 = str_5137
        
        # Call to Load(...): (line 80)
        # Processing the call keyword arguments (line 80)
        kwargs_5141 = {}
        # Getting the type of 'ast' (line 80)
        ast_5139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 72), 'ast', False)
        # Obtaining the member 'Load' of a type (line 80)
        Load_5140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 72), ast_5139, 'Load')
        # Calling Load(args, kwargs) (line 80)
        Load_call_result_5142 = invoke(stypy.reporting.localization.Localization(__file__, 80, 72), Load_5140, *[], **kwargs_5141)
        
        keyword_5143 = Load_call_result_5142
        
        # Call to Name(...): (line 81)
        # Processing the call keyword arguments (line 81)
        # Getting the type of 'type_test_name' (line 81)
        type_test_name_5146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 52), 'type_test_name', False)
        keyword_5147 = type_test_name_5146
        
        # Call to Load(...): (line 81)
        # Processing the call keyword arguments (line 81)
        kwargs_5150 = {}
        # Getting the type of 'ast' (line 81)
        ast_5148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 72), 'ast', False)
        # Obtaining the member 'Load' of a type (line 81)
        Load_5149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 72), ast_5148, 'Load')
        # Calling Load(args, kwargs) (line 81)
        Load_call_result_5151 = invoke(stypy.reporting.localization.Localization(__file__, 81, 72), Load_5149, *[], **kwargs_5150)
        
        keyword_5152 = Load_call_result_5151
        kwargs_5153 = {'ctx': keyword_5152, 'id': keyword_5147}
        # Getting the type of 'ast' (line 81)
        ast_5144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 40), 'ast', False)
        # Obtaining the member 'Name' of a type (line 81)
        Name_5145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 40), ast_5144, 'Name')
        # Calling Name(args, kwargs) (line 81)
        Name_call_result_5154 = invoke(stypy.reporting.localization.Localization(__file__, 81, 40), Name_5145, *[], **kwargs_5153)
        
        keyword_5155 = Name_call_result_5154
        kwargs_5156 = {'ctx': keyword_5143, 'attr': keyword_5138, 'value': keyword_5155}
        # Getting the type of 'ast' (line 80)
        ast_5135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'ast', False)
        # Obtaining the member 'Attribute' of a type (line 80)
        Attribute_5136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 20), ast_5135, 'Attribute')
        # Calling Attribute(args, kwargs) (line 80)
        Attribute_call_result_5157 = invoke(stypy.reporting.localization.Localization(__file__, 80, 20), Attribute_5136, *[], **kwargs_5156)
        
        # Assigning a type to the variable 'attribute' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'attribute', Attribute_call_result_5157)
        
        # Assigning a Call to a Name (line 82):
        
        # Call to Call(...): (line 82)
        # Processing the call keyword arguments (line 82)
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_5160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        # Adding element type (line 82)
        # Getting the type of 'locals_call' (line 82)
        locals_call_5161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 30), 'locals_call', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 29), list_5160, locals_call_5161)
        
        keyword_5162 = list_5160
        # Getting the type of 'attribute' (line 82)
        attribute_5163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 49), 'attribute', False)
        keyword_5164 = attribute_5163
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_5165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 69), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        
        keyword_5166 = list_5165
        # Getting the type of 'None' (line 82)
        None_5167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 80), 'None', False)
        keyword_5168 = None_5167
        # Getting the type of 'None' (line 82)
        None_5169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 95), 'None', False)
        keyword_5170 = None_5169
        kwargs_5171 = {'keywords': keyword_5166, 'starargs': keyword_5170, 'args': keyword_5162, 'func': keyword_5164, 'kwargs': keyword_5168}
        # Getting the type of 'ast' (line 82)
        ast_5158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'ast', False)
        # Obtaining the member 'Call' of a type (line 82)
        Call_5159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), ast_5158, 'Call')
        # Calling Call(args, kwargs) (line 82)
        Call_call_result_5172 = invoke(stypy.reporting.localization.Localization(__file__, 82, 15), Call_5159, *[], **kwargs_5171)
        
        # Assigning a type to the variable 'call' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'call', Call_call_result_5172)
        
        # Assigning a Call to a Name (line 83):
        
        # Call to Tuple(...): (line 83)
        # Processing the call keyword arguments (line 83)
        
        # Call to Load(...): (line 83)
        # Processing the call keyword arguments (line 83)
        kwargs_5177 = {}
        # Getting the type of 'ast' (line 83)
        ast_5175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 31), 'ast', False)
        # Obtaining the member 'Load' of a type (line 83)
        Load_5176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 31), ast_5175, 'Load')
        # Calling Load(args, kwargs) (line 83)
        Load_call_result_5178 = invoke(stypy.reporting.localization.Localization(__file__, 83, 31), Load_5176, *[], **kwargs_5177)
        
        keyword_5179 = Load_call_result_5178
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_5180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        # Adding element type (line 83)
        # Getting the type of 'node' (line 83)
        node_5181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 49), 'node', False)
        # Obtaining the member 'value' of a type (line 83)
        value_5182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 49), node_5181, 'value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 48), list_5180, value_5182)
        # Adding element type (line 83)
        # Getting the type of 'call' (line 83)
        call_5183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 61), 'call', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 48), list_5180, call_5183)
        
        keyword_5184 = list_5180
        kwargs_5185 = {'elts': keyword_5184, 'ctx': keyword_5179}
        # Getting the type of 'ast' (line 83)
        ast_5173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 17), 'ast', False)
        # Obtaining the member 'Tuple' of a type (line 83)
        Tuple_5174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 17), ast_5173, 'Tuple')
        # Calling Tuple(args, kwargs) (line 83)
        Tuple_call_result_5186 = invoke(stypy.reporting.localization.Localization(__file__, 83, 17), Tuple_5174, *[], **kwargs_5185)
        
        # Assigning a type to the variable 'tuple_' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'tuple_', Tuple_call_result_5186)
        
        # Assigning a Call to a Name (line 84):
        
        # Call to Subscript(...): (line 84)
        # Processing the call keyword arguments (line 84)
        
        # Call to Load(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_5191 = {}
        # Getting the type of 'ast' (line 84)
        ast_5189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 38), 'ast', False)
        # Obtaining the member 'Load' of a type (line 84)
        Load_5190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 38), ast_5189, 'Load')
        # Calling Load(args, kwargs) (line 84)
        Load_call_result_5192 = invoke(stypy.reporting.localization.Localization(__file__, 84, 38), Load_5190, *[], **kwargs_5191)
        
        keyword_5193 = Load_call_result_5192
        # Getting the type of 'index' (line 84)
        index_5194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 56), 'index', False)
        keyword_5195 = index_5194
        # Getting the type of 'tuple_' (line 84)
        tuple__5196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 69), 'tuple_', False)
        keyword_5197 = tuple__5196
        kwargs_5198 = {'slice': keyword_5195, 'ctx': keyword_5193, 'value': keyword_5197}
        # Getting the type of 'ast' (line 84)
        ast_5187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'ast', False)
        # Obtaining the member 'Subscript' of a type (line 84)
        Subscript_5188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 20), ast_5187, 'Subscript')
        # Calling Subscript(args, kwargs) (line 84)
        Subscript_call_result_5199 = invoke(stypy.reporting.localization.Localization(__file__, 84, 20), Subscript_5188, *[], **kwargs_5198)
        
        # Assigning a type to the variable 'subscript' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'subscript', Subscript_call_result_5199)
        
        # Assigning a Name to a Attribute (line 85):
        # Getting the type of 'subscript' (line 85)
        subscript_5200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'subscript')
        # Getting the type of 'node' (line 85)
        node_5201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'node')
        # Setting the type of the member 'value' of a type (line 85)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), node_5201, 'value', subscript_5200)
        # Getting the type of 'node' (line 87)
        node_5202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type', node_5202)
        
        # ################# End of 'visit_Return(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Return' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_5203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5203)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Return'
        return stypy_return_type_5203


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 6, 0, False)
        # Assigning a type to the variable 'self' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeDataAutoGeneratorVisitor.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TypeDataAutoGeneratorVisitor' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'TypeDataAutoGeneratorVisitor', TypeDataAutoGeneratorVisitor)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
