
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
str_21360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 17), 'str', 'type_test')
# Assigning a type to the variable 'type_test_name' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'type_test_name', str_21360)
# Declaration of the 'TypeDataAutoGeneratorVisitor' class
# Getting the type of 'ast' (line 6)
ast_21361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 35), 'ast')
# Obtaining the member 'NodeVisitor' of a type (line 6)
NodeVisitor_21362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 35), ast_21361, 'NodeVisitor')

class TypeDataAutoGeneratorVisitor(NodeVisitor_21362, ):
    str_21363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', '\n    This visitor is used to generate a version of the original source code that dynamically captures the types of\n    variables in functions, methods and the global scope. This program can be executed and code is inserted in key\n    places to capture the value of the variables at the end of the execution of any of the previously mentioned\n    elements. This has several limitations:\n    - If a variable changes types during the execution of a function, only the last type is captured.\n    - If the program has errors at runtime, nothing is captured\n    - The technique may fail with certain Python constructs.\n\n    In general, this visitor is only used as a helper for creating Python data files that can be used to unit test\n    the type inference code generation modules and classes, not being a part of the end-user functionality of stypy.\n     As a tool to facilitate the development of stypy, the code is not polished at the same level as the rest of the\n     code, as its only function is to generate an approximation of the types of the variables that a correct execution\n     of the tested programs should return. Normally the generated table has to be modified by hand, as this technique\n     is not 100% accurate.\n    ')

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
        str_21366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 31), 'str', 'TypeDataFileWriter')
        keyword_21367 = str_21366
        # Getting the type of 'None' (line 25)
        None_21368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 60), 'None', False)
        keyword_21369 = None_21368
        kwargs_21370 = {'name': keyword_21367, 'asname': keyword_21369}
        # Getting the type of 'ast' (line 25)
        ast_21364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'ast', False)
        # Obtaining the member 'alias' of a type (line 25)
        alias_21365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 16), ast_21364, 'alias')
        # Calling alias(args, kwargs) (line 25)
        alias_call_result_21371 = invoke(stypy.reporting.localization.Localization(__file__, 25, 16), alias_21365, *[], **kwargs_21370)
        
        # Assigning a type to the variable 'alias' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'alias', alias_call_result_21371)
        
        # Assigning a Call to a Name (line 26):
        
        # Call to ImportFrom(...): (line 26)
        # Processing the call keyword arguments (line 26)
        int_21374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 39), 'int')
        keyword_21375 = int_21374
        str_21376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 40), 'str', 'stypy.code_generation.type_inference_programs.checking.type_data_file_writer')
        keyword_21377 = str_21376
        
        # Obtaining an instance of the builtin type 'list' (line 28)
        list_21378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 39), 'list')
        # Adding type elements to the builtin type 'list' instance (line 28)
        # Adding element type (line 28)
        # Getting the type of 'alias' (line 28)
        alias_21379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 40), 'alias', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 39), list_21378, alias_21379)
        
        keyword_21380 = list_21378
        kwargs_21381 = {'names': keyword_21380, 'module': keyword_21377, 'level': keyword_21375}
        # Getting the type of 'ast' (line 26)
        ast_21372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'ast', False)
        # Obtaining the member 'ImportFrom' of a type (line 26)
        ImportFrom_21373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 18), ast_21372, 'ImportFrom')
        # Calling ImportFrom(args, kwargs) (line 26)
        ImportFrom_call_result_21382 = invoke(stypy.reporting.localization.Localization(__file__, 26, 18), ImportFrom_21373, *[], **kwargs_21381)
        
        # Assigning a type to the variable 'import_' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'import_', ImportFrom_call_result_21382)
        
        # Assigning a Call to a Name (line 30):
        
        # Call to Name(...): (line 30)
        # Processing the call keyword arguments (line 30)
        str_21385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 27), 'str', '__file__')
        keyword_21386 = str_21385
        
        # Call to Load(...): (line 30)
        # Processing the call keyword arguments (line 30)
        kwargs_21389 = {}
        # Getting the type of 'ast' (line 30)
        ast_21387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 43), 'ast', False)
        # Obtaining the member 'Load' of a type (line 30)
        Load_21388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 43), ast_21387, 'Load')
        # Calling Load(args, kwargs) (line 30)
        Load_call_result_21390 = invoke(stypy.reporting.localization.Localization(__file__, 30, 43), Load_21388, *[], **kwargs_21389)
        
        keyword_21391 = Load_call_result_21390
        kwargs_21392 = {'ctx': keyword_21391, 'id': keyword_21386}
        # Getting the type of 'ast' (line 30)
        ast_21383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'ast', False)
        # Obtaining the member 'Name' of a type (line 30)
        Name_21384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 15), ast_21383, 'Name')
        # Calling Name(args, kwargs) (line 30)
        Name_call_result_21393 = invoke(stypy.reporting.localization.Localization(__file__, 30, 15), Name_21384, *[], **kwargs_21392)
        
        # Assigning a type to the variable 'name' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'name', Name_call_result_21393)
        
        # Assigning a Call to a Name (line 33):
        
        # Call to Name(...): (line 33)
        # Processing the call keyword arguments (line 33)
        str_21396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 32), 'str', 'TypeDataFileWriter')
        keyword_21397 = str_21396
        
        # Call to Load(...): (line 33)
        # Processing the call keyword arguments (line 33)
        kwargs_21400 = {}
        # Getting the type of 'ast' (line 33)
        ast_21398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 58), 'ast', False)
        # Obtaining the member 'Load' of a type (line 33)
        Load_21399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 58), ast_21398, 'Load')
        # Calling Load(args, kwargs) (line 33)
        Load_call_result_21401 = invoke(stypy.reporting.localization.Localization(__file__, 33, 58), Load_21399, *[], **kwargs_21400)
        
        keyword_21402 = Load_call_result_21401
        kwargs_21403 = {'ctx': keyword_21402, 'id': keyword_21397}
        # Getting the type of 'ast' (line 33)
        ast_21394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'ast', False)
        # Obtaining the member 'Name' of a type (line 33)
        Name_21395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 20), ast_21394, 'Name')
        # Calling Name(args, kwargs) (line 33)
        Name_call_result_21404 = invoke(stypy.reporting.localization.Localization(__file__, 33, 20), Name_21395, *[], **kwargs_21403)
        
        # Assigning a type to the variable 'attribute' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'attribute', Name_call_result_21404)
        
        # Assigning a Call to a Name (line 34):
        
        # Call to Call(...): (line 34)
        # Processing the call keyword arguments (line 34)
        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_21407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        # Adding element type (line 34)
        # Getting the type of 'name' (line 34)
        name_21408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 29), list_21407, name_21408)
        
        keyword_21409 = list_21407
        # Getting the type of 'attribute' (line 34)
        attribute_21410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 42), 'attribute', False)
        keyword_21411 = attribute_21410
        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_21412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 62), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        
        keyword_21413 = list_21412
        # Getting the type of 'None' (line 34)
        None_21414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 73), 'None', False)
        keyword_21415 = None_21414
        # Getting the type of 'None' (line 34)
        None_21416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 88), 'None', False)
        keyword_21417 = None_21416
        kwargs_21418 = {'keywords': keyword_21413, 'starargs': keyword_21417, 'args': keyword_21409, 'func': keyword_21411, 'kwargs': keyword_21415}
        # Getting the type of 'ast' (line 34)
        ast_21405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'ast', False)
        # Obtaining the member 'Call' of a type (line 34)
        Call_21406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 15), ast_21405, 'Call')
        # Calling Call(args, kwargs) (line 34)
        Call_call_result_21419 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), Call_21406, *[], **kwargs_21418)
        
        # Assigning a type to the variable 'call' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'call', Call_call_result_21419)
        
        # Assigning a Call to a Name (line 35):
        
        # Call to Assign(...): (line 35)
        # Processing the call keyword arguments (line 35)
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_21422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        # Adding element type (line 35)
        
        # Call to Name(...): (line 35)
        # Processing the call keyword arguments (line 35)
        # Getting the type of 'type_test_name' (line 35)
        type_test_name_21425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 49), 'type_test_name', False)
        keyword_21426 = type_test_name_21425
        
        # Call to Store(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_21429 = {}
        # Getting the type of 'ast' (line 35)
        ast_21427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 69), 'ast', False)
        # Obtaining the member 'Store' of a type (line 35)
        Store_21428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 69), ast_21427, 'Store')
        # Calling Store(args, kwargs) (line 35)
        Store_call_result_21430 = invoke(stypy.reporting.localization.Localization(__file__, 35, 69), Store_21428, *[], **kwargs_21429)
        
        keyword_21431 = Store_call_result_21430
        kwargs_21432 = {'ctx': keyword_21431, 'id': keyword_21426}
        # Getting the type of 'ast' (line 35)
        ast_21423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 37), 'ast', False)
        # Obtaining the member 'Name' of a type (line 35)
        Name_21424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 37), ast_21423, 'Name')
        # Calling Name(args, kwargs) (line 35)
        Name_call_result_21433 = invoke(stypy.reporting.localization.Localization(__file__, 35, 37), Name_21424, *[], **kwargs_21432)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 36), list_21422, Name_call_result_21433)
        
        keyword_21434 = list_21422
        # Getting the type of 'call' (line 36)
        call_21435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 34), 'call', False)
        keyword_21436 = call_21435
        kwargs_21437 = {'targets': keyword_21434, 'value': keyword_21436}
        # Getting the type of 'ast' (line 35)
        ast_21420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'ast', False)
        # Obtaining the member 'Assign' of a type (line 35)
        Assign_21421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 17), ast_21420, 'Assign')
        # Calling Assign(args, kwargs) (line 35)
        Assign_call_result_21438 = invoke(stypy.reporting.localization.Localization(__file__, 35, 17), Assign_21421, *[], **kwargs_21437)
        
        # Assigning a type to the variable 'assign' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'assign', Assign_call_result_21438)
        
        # Call to insert(...): (line 38)
        # Processing the call arguments (line 38)
        int_21442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 25), 'int')
        # Getting the type of 'assign' (line 38)
        assign_21443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 28), 'assign', False)
        # Processing the call keyword arguments (line 38)
        kwargs_21444 = {}
        # Getting the type of 'node' (line 38)
        node_21439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'node', False)
        # Obtaining the member 'body' of a type (line 38)
        body_21440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), node_21439, 'body')
        # Obtaining the member 'insert' of a type (line 38)
        insert_21441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), body_21440, 'insert')
        # Calling insert(args, kwargs) (line 38)
        insert_call_result_21445 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), insert_21441, *[int_21442, assign_21443], **kwargs_21444)
        
        
        # Call to insert(...): (line 39)
        # Processing the call arguments (line 39)
        int_21449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'int')
        # Getting the type of 'import_' (line 39)
        import__21450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 28), 'import_', False)
        # Processing the call keyword arguments (line 39)
        kwargs_21451 = {}
        # Getting the type of 'node' (line 39)
        node_21446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'node', False)
        # Obtaining the member 'body' of a type (line 39)
        body_21447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), node_21446, 'body')
        # Obtaining the member 'insert' of a type (line 39)
        insert_21448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), body_21447, 'insert')
        # Calling insert(args, kwargs) (line 39)
        insert_call_result_21452 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), insert_21448, *[int_21449, import__21450], **kwargs_21451)
        
        
        # Getting the type of 'node' (line 41)
        node_21453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'node')
        # Obtaining the member 'body' of a type (line 41)
        body_21454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 20), node_21453, 'body')
        # Assigning a type to the variable 'body_21454' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'body_21454', body_21454)
        # Testing if the for loop is going to be iterated (line 41)
        # Testing the type of a for loop iterable (line 41)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 41, 8), body_21454)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 41, 8), body_21454):
            # Getting the type of the for loop variable (line 41)
            for_loop_var_21455 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 41, 8), body_21454)
            # Assigning a type to the variable 'stmt' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stmt', for_loop_var_21455)
            # SSA begins for a for statement (line 41)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 42)
            # Processing the call arguments (line 42)
            # Getting the type of 'stmt' (line 42)
            stmt_21458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'stmt', False)
            # Processing the call keyword arguments (line 42)
            kwargs_21459 = {}
            # Getting the type of 'self' (line 42)
            self_21456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 42)
            visit_21457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), self_21456, 'visit')
            # Calling visit(args, kwargs) (line 42)
            visit_call_result_21460 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), visit_21457, *[stmt_21458], **kwargs_21459)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 44):
        
        # Call to Call(...): (line 44)
        # Processing the call keyword arguments (line 44)
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_21463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        
        keyword_21464 = list_21463
        
        # Call to Name(...): (line 44)
        # Processing the call keyword arguments (line 44)
        str_21467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 57), 'str', 'globals')
        keyword_21468 = str_21467
        
        # Call to Load(...): (line 44)
        # Processing the call keyword arguments (line 44)
        kwargs_21471 = {}
        # Getting the type of 'ast' (line 44)
        ast_21469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 72), 'ast', False)
        # Obtaining the member 'Load' of a type (line 44)
        Load_21470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 72), ast_21469, 'Load')
        # Calling Load(args, kwargs) (line 44)
        Load_call_result_21472 = invoke(stypy.reporting.localization.Localization(__file__, 44, 72), Load_21470, *[], **kwargs_21471)
        
        keyword_21473 = Load_call_result_21472
        kwargs_21474 = {'ctx': keyword_21473, 'id': keyword_21468}
        # Getting the type of 'ast' (line 44)
        ast_21465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 45), 'ast', False)
        # Obtaining the member 'Name' of a type (line 44)
        Name_21466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 45), ast_21465, 'Name')
        # Calling Name(args, kwargs) (line 44)
        Name_call_result_21475 = invoke(stypy.reporting.localization.Localization(__file__, 44, 45), Name_21466, *[], **kwargs_21474)
        
        keyword_21476 = Name_call_result_21475
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_21477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 94), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        
        keyword_21478 = list_21477
        # Getting the type of 'None' (line 44)
        None_21479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 105), 'None', False)
        keyword_21480 = None_21479
        # Getting the type of 'None' (line 45)
        None_21481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 40), 'None', False)
        keyword_21482 = None_21481
        kwargs_21483 = {'keywords': keyword_21478, 'starargs': keyword_21482, 'args': keyword_21464, 'func': keyword_21476, 'kwargs': keyword_21480}
        # Getting the type of 'ast' (line 44)
        ast_21461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'ast', False)
        # Obtaining the member 'Call' of a type (line 44)
        Call_21462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 22), ast_21461, 'Call')
        # Calling Call(args, kwargs) (line 44)
        Call_call_result_21484 = invoke(stypy.reporting.localization.Localization(__file__, 44, 22), Call_21462, *[], **kwargs_21483)
        
        # Assigning a type to the variable 'locals_call' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'locals_call', Call_call_result_21484)
        
        # Assigning a Call to a Name (line 46):
        
        # Call to Attribute(...): (line 46)
        # Processing the call keyword arguments (line 46)
        str_21487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 39), 'str', 'add_type_dict_for_main_context')
        keyword_21488 = str_21487
        
        # Call to Load(...): (line 46)
        # Processing the call keyword arguments (line 46)
        kwargs_21491 = {}
        # Getting the type of 'ast' (line 46)
        ast_21489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 77), 'ast', False)
        # Obtaining the member 'Load' of a type (line 46)
        Load_21490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 77), ast_21489, 'Load')
        # Calling Load(args, kwargs) (line 46)
        Load_call_result_21492 = invoke(stypy.reporting.localization.Localization(__file__, 46, 77), Load_21490, *[], **kwargs_21491)
        
        keyword_21493 = Load_call_result_21492
        
        # Call to Name(...): (line 47)
        # Processing the call keyword arguments (line 47)
        # Getting the type of 'type_test_name' (line 47)
        type_test_name_21496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 52), 'type_test_name', False)
        keyword_21497 = type_test_name_21496
        
        # Call to Load(...): (line 47)
        # Processing the call keyword arguments (line 47)
        kwargs_21500 = {}
        # Getting the type of 'ast' (line 47)
        ast_21498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 72), 'ast', False)
        # Obtaining the member 'Load' of a type (line 47)
        Load_21499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 72), ast_21498, 'Load')
        # Calling Load(args, kwargs) (line 47)
        Load_call_result_21501 = invoke(stypy.reporting.localization.Localization(__file__, 47, 72), Load_21499, *[], **kwargs_21500)
        
        keyword_21502 = Load_call_result_21501
        kwargs_21503 = {'ctx': keyword_21502, 'id': keyword_21497}
        # Getting the type of 'ast' (line 47)
        ast_21494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 40), 'ast', False)
        # Obtaining the member 'Name' of a type (line 47)
        Name_21495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 40), ast_21494, 'Name')
        # Calling Name(args, kwargs) (line 47)
        Name_call_result_21504 = invoke(stypy.reporting.localization.Localization(__file__, 47, 40), Name_21495, *[], **kwargs_21503)
        
        keyword_21505 = Name_call_result_21504
        kwargs_21506 = {'ctx': keyword_21493, 'attr': keyword_21488, 'value': keyword_21505}
        # Getting the type of 'ast' (line 46)
        ast_21485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'ast', False)
        # Obtaining the member 'Attribute' of a type (line 46)
        Attribute_21486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 20), ast_21485, 'Attribute')
        # Calling Attribute(args, kwargs) (line 46)
        Attribute_call_result_21507 = invoke(stypy.reporting.localization.Localization(__file__, 46, 20), Attribute_21486, *[], **kwargs_21506)
        
        # Assigning a type to the variable 'attribute' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'attribute', Attribute_call_result_21507)
        
        # Assigning a Call to a Name (line 49):
        
        # Call to Call(...): (line 49)
        # Processing the call keyword arguments (line 49)
        
        # Obtaining an instance of the builtin type 'list' (line 49)
        list_21510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 49)
        # Adding element type (line 49)
        # Getting the type of 'locals_call' (line 49)
        locals_call_21511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'locals_call', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 29), list_21510, locals_call_21511)
        
        keyword_21512 = list_21510
        # Getting the type of 'attribute' (line 49)
        attribute_21513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 49), 'attribute', False)
        keyword_21514 = attribute_21513
        
        # Obtaining an instance of the builtin type 'list' (line 49)
        list_21515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 69), 'list')
        # Adding type elements to the builtin type 'list' instance (line 49)
        
        keyword_21516 = list_21515
        # Getting the type of 'None' (line 49)
        None_21517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 80), 'None', False)
        keyword_21518 = None_21517
        # Getting the type of 'None' (line 49)
        None_21519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 95), 'None', False)
        keyword_21520 = None_21519
        kwargs_21521 = {'keywords': keyword_21516, 'starargs': keyword_21520, 'args': keyword_21512, 'func': keyword_21514, 'kwargs': keyword_21518}
        # Getting the type of 'ast' (line 49)
        ast_21508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'ast', False)
        # Obtaining the member 'Call' of a type (line 49)
        Call_21509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 15), ast_21508, 'Call')
        # Calling Call(args, kwargs) (line 49)
        Call_call_result_21522 = invoke(stypy.reporting.localization.Localization(__file__, 49, 15), Call_21509, *[], **kwargs_21521)
        
        # Assigning a type to the variable 'call' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'call', Call_call_result_21522)
        
        # Assigning a Call to a Name (line 50):
        
        # Call to Expr(...): (line 50)
        # Processing the call keyword arguments (line 50)
        # Getting the type of 'call' (line 50)
        call_21525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 30), 'call', False)
        keyword_21526 = call_21525
        kwargs_21527 = {'value': keyword_21526}
        # Getting the type of 'ast' (line 50)
        ast_21523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'ast', False)
        # Obtaining the member 'Expr' of a type (line 50)
        Expr_21524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 15), ast_21523, 'Expr')
        # Calling Expr(args, kwargs) (line 50)
        Expr_call_result_21528 = invoke(stypy.reporting.localization.Localization(__file__, 50, 15), Expr_21524, *[], **kwargs_21527)
        
        # Assigning a type to the variable 'expr' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'expr', Expr_call_result_21528)
        
        # Assigning a Call to a Name (line 52):
        
        # Call to Attribute(...): (line 52)
        # Processing the call keyword arguments (line 52)
        str_21531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 48), 'str', 'generate_type_data_file')
        keyword_21532 = str_21531
        
        # Call to Load(...): (line 52)
        # Processing the call keyword arguments (line 52)
        kwargs_21535 = {}
        # Getting the type of 'ast' (line 52)
        ast_21533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 79), 'ast', False)
        # Obtaining the member 'Load' of a type (line 52)
        Load_21534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 79), ast_21533, 'Load')
        # Calling Load(args, kwargs) (line 52)
        Load_call_result_21536 = invoke(stypy.reporting.localization.Localization(__file__, 52, 79), Load_21534, *[], **kwargs_21535)
        
        keyword_21537 = Load_call_result_21536
        
        # Call to Name(...): (line 53)
        # Processing the call keyword arguments (line 53)
        # Getting the type of 'type_test_name' (line 53)
        type_test_name_21540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 61), 'type_test_name', False)
        keyword_21541 = type_test_name_21540
        
        # Call to Load(...): (line 53)
        # Processing the call keyword arguments (line 53)
        kwargs_21544 = {}
        # Getting the type of 'ast' (line 53)
        ast_21542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 81), 'ast', False)
        # Obtaining the member 'Load' of a type (line 53)
        Load_21543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 81), ast_21542, 'Load')
        # Calling Load(args, kwargs) (line 53)
        Load_call_result_21545 = invoke(stypy.reporting.localization.Localization(__file__, 53, 81), Load_21543, *[], **kwargs_21544)
        
        keyword_21546 = Load_call_result_21545
        kwargs_21547 = {'ctx': keyword_21546, 'id': keyword_21541}
        # Getting the type of 'ast' (line 53)
        ast_21538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 49), 'ast', False)
        # Obtaining the member 'Name' of a type (line 53)
        Name_21539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 49), ast_21538, 'Name')
        # Calling Name(args, kwargs) (line 53)
        Name_call_result_21548 = invoke(stypy.reporting.localization.Localization(__file__, 53, 49), Name_21539, *[], **kwargs_21547)
        
        keyword_21549 = Name_call_result_21548
        kwargs_21550 = {'ctx': keyword_21537, 'attr': keyword_21532, 'value': keyword_21549}
        # Getting the type of 'ast' (line 52)
        ast_21529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 29), 'ast', False)
        # Obtaining the member 'Attribute' of a type (line 52)
        Attribute_21530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 29), ast_21529, 'Attribute')
        # Calling Attribute(args, kwargs) (line 52)
        Attribute_call_result_21551 = invoke(stypy.reporting.localization.Localization(__file__, 52, 29), Attribute_21530, *[], **kwargs_21550)
        
        # Assigning a type to the variable 'attribute_generate' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'attribute_generate', Attribute_call_result_21551)
        
        # Assigning a Call to a Name (line 54):
        
        # Call to Call(...): (line 54)
        # Processing the call keyword arguments (line 54)
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_21554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        
        keyword_21555 = list_21554
        # Getting the type of 'attribute_generate' (line 54)
        attribute_generate_21556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 47), 'attribute_generate', False)
        keyword_21557 = attribute_generate_21556
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_21558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 76), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        
        keyword_21559 = list_21558
        # Getting the type of 'None' (line 54)
        None_21560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 87), 'None', False)
        keyword_21561 = None_21560
        # Getting the type of 'None' (line 54)
        None_21562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 102), 'None', False)
        keyword_21563 = None_21562
        kwargs_21564 = {'keywords': keyword_21559, 'starargs': keyword_21563, 'args': keyword_21555, 'func': keyword_21557, 'kwargs': keyword_21561}
        # Getting the type of 'ast' (line 54)
        ast_21552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'ast', False)
        # Obtaining the member 'Call' of a type (line 54)
        Call_21553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 24), ast_21552, 'Call')
        # Calling Call(args, kwargs) (line 54)
        Call_call_result_21565 = invoke(stypy.reporting.localization.Localization(__file__, 54, 24), Call_21553, *[], **kwargs_21564)
        
        # Assigning a type to the variable 'call_generate' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'call_generate', Call_call_result_21565)
        
        # Assigning a Call to a Name (line 55):
        
        # Call to Expr(...): (line 55)
        # Processing the call keyword arguments (line 55)
        # Getting the type of 'call_generate' (line 55)
        call_generate_21568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 36), 'call_generate', False)
        keyword_21569 = call_generate_21568
        
        # Call to Load(...): (line 55)
        # Processing the call keyword arguments (line 55)
        kwargs_21572 = {}
        # Getting the type of 'ast' (line 55)
        ast_21570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 55), 'ast', False)
        # Obtaining the member 'Load' of a type (line 55)
        Load_21571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 55), ast_21570, 'Load')
        # Calling Load(args, kwargs) (line 55)
        Load_call_result_21573 = invoke(stypy.reporting.localization.Localization(__file__, 55, 55), Load_21571, *[], **kwargs_21572)
        
        keyword_21574 = Load_call_result_21573
        kwargs_21575 = {'ctx': keyword_21574, 'value': keyword_21569}
        # Getting the type of 'ast' (line 55)
        ast_21566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'ast', False)
        # Obtaining the member 'Expr' of a type (line 55)
        Expr_21567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 21), ast_21566, 'Expr')
        # Calling Expr(args, kwargs) (line 55)
        Expr_call_result_21576 = invoke(stypy.reporting.localization.Localization(__file__, 55, 21), Expr_21567, *[], **kwargs_21575)
        
        # Assigning a type to the variable 'expr_final' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'expr_final', Expr_call_result_21576)
        
        # Call to append(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'expr' (line 56)
        expr_21580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 25), 'expr', False)
        # Processing the call keyword arguments (line 56)
        kwargs_21581 = {}
        # Getting the type of 'node' (line 56)
        node_21577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'node', False)
        # Obtaining the member 'body' of a type (line 56)
        body_21578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), node_21577, 'body')
        # Obtaining the member 'append' of a type (line 56)
        append_21579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), body_21578, 'append')
        # Calling append(args, kwargs) (line 56)
        append_call_result_21582 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), append_21579, *[expr_21580], **kwargs_21581)
        
        
        # Call to append(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'expr_final' (line 57)
        expr_final_21586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'expr_final', False)
        # Processing the call keyword arguments (line 57)
        kwargs_21587 = {}
        # Getting the type of 'node' (line 57)
        node_21583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'node', False)
        # Obtaining the member 'body' of a type (line 57)
        body_21584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), node_21583, 'body')
        # Obtaining the member 'append' of a type (line 57)
        append_21585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), body_21584, 'append')
        # Calling append(args, kwargs) (line 57)
        append_call_result_21588 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), append_21585, *[expr_final_21586], **kwargs_21587)
        
        # Getting the type of 'node' (line 58)
        node_21589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'stypy_return_type', node_21589)
        
        # ################# End of 'visit_Module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Module' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_21590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21590)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Module'
        return stypy_return_type_21590


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
        node_21591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'node')
        # Obtaining the member 'body' of a type (line 61)
        body_21592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 20), node_21591, 'body')
        # Assigning a type to the variable 'body_21592' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'body_21592', body_21592)
        # Testing if the for loop is going to be iterated (line 61)
        # Testing the type of a for loop iterable (line 61)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 61, 8), body_21592)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 61, 8), body_21592):
            # Getting the type of the for loop variable (line 61)
            for_loop_var_21593 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 61, 8), body_21592)
            # Assigning a type to the variable 'stmt' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'stmt', for_loop_var_21593)
            # SSA begins for a for statement (line 61)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 62)
            # Processing the call arguments (line 62)
            # Getting the type of 'stmt' (line 62)
            stmt_21596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'stmt', False)
            # Processing the call keyword arguments (line 62)
            kwargs_21597 = {}
            # Getting the type of 'self' (line 62)
            self_21594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 62)
            visit_21595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), self_21594, 'visit')
            # Calling visit(args, kwargs) (line 62)
            visit_call_result_21598 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), visit_21595, *[stmt_21596], **kwargs_21597)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 64):
        
        # Call to Call(...): (line 64)
        # Processing the call keyword arguments (line 64)
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_21601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        
        keyword_21602 = list_21601
        
        # Call to Name(...): (line 64)
        # Processing the call keyword arguments (line 64)
        str_21605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 57), 'str', 'locals')
        keyword_21606 = str_21605
        
        # Call to Load(...): (line 64)
        # Processing the call keyword arguments (line 64)
        kwargs_21609 = {}
        # Getting the type of 'ast' (line 64)
        ast_21607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 71), 'ast', False)
        # Obtaining the member 'Load' of a type (line 64)
        Load_21608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 71), ast_21607, 'Load')
        # Calling Load(args, kwargs) (line 64)
        Load_call_result_21610 = invoke(stypy.reporting.localization.Localization(__file__, 64, 71), Load_21608, *[], **kwargs_21609)
        
        keyword_21611 = Load_call_result_21610
        kwargs_21612 = {'ctx': keyword_21611, 'id': keyword_21606}
        # Getting the type of 'ast' (line 64)
        ast_21603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 45), 'ast', False)
        # Obtaining the member 'Name' of a type (line 64)
        Name_21604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 45), ast_21603, 'Name')
        # Calling Name(args, kwargs) (line 64)
        Name_call_result_21613 = invoke(stypy.reporting.localization.Localization(__file__, 64, 45), Name_21604, *[], **kwargs_21612)
        
        keyword_21614 = Name_call_result_21613
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_21615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 93), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        
        keyword_21616 = list_21615
        # Getting the type of 'None' (line 64)
        None_21617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 104), 'None', False)
        keyword_21618 = None_21617
        # Getting the type of 'None' (line 65)
        None_21619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 40), 'None', False)
        keyword_21620 = None_21619
        kwargs_21621 = {'keywords': keyword_21616, 'starargs': keyword_21620, 'args': keyword_21602, 'func': keyword_21614, 'kwargs': keyword_21618}
        # Getting the type of 'ast' (line 64)
        ast_21599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 22), 'ast', False)
        # Obtaining the member 'Call' of a type (line 64)
        Call_21600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 22), ast_21599, 'Call')
        # Calling Call(args, kwargs) (line 64)
        Call_call_result_21622 = invoke(stypy.reporting.localization.Localization(__file__, 64, 22), Call_21600, *[], **kwargs_21621)
        
        # Assigning a type to the variable 'locals_call' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'locals_call', Call_call_result_21622)
        
        # Assigning a Call to a Name (line 66):
        
        # Call to Attribute(...): (line 66)
        # Processing the call keyword arguments (line 66)
        str_21625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 39), 'str', 'add_type_dict_for_context')
        keyword_21626 = str_21625
        
        # Call to Load(...): (line 66)
        # Processing the call keyword arguments (line 66)
        kwargs_21629 = {}
        # Getting the type of 'ast' (line 66)
        ast_21627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 72), 'ast', False)
        # Obtaining the member 'Load' of a type (line 66)
        Load_21628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 72), ast_21627, 'Load')
        # Calling Load(args, kwargs) (line 66)
        Load_call_result_21630 = invoke(stypy.reporting.localization.Localization(__file__, 66, 72), Load_21628, *[], **kwargs_21629)
        
        keyword_21631 = Load_call_result_21630
        
        # Call to Name(...): (line 67)
        # Processing the call keyword arguments (line 67)
        # Getting the type of 'type_test_name' (line 67)
        type_test_name_21634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 52), 'type_test_name', False)
        keyword_21635 = type_test_name_21634
        
        # Call to Load(...): (line 67)
        # Processing the call keyword arguments (line 67)
        kwargs_21638 = {}
        # Getting the type of 'ast' (line 67)
        ast_21636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 72), 'ast', False)
        # Obtaining the member 'Load' of a type (line 67)
        Load_21637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 72), ast_21636, 'Load')
        # Calling Load(args, kwargs) (line 67)
        Load_call_result_21639 = invoke(stypy.reporting.localization.Localization(__file__, 67, 72), Load_21637, *[], **kwargs_21638)
        
        keyword_21640 = Load_call_result_21639
        kwargs_21641 = {'ctx': keyword_21640, 'id': keyword_21635}
        # Getting the type of 'ast' (line 67)
        ast_21632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 40), 'ast', False)
        # Obtaining the member 'Name' of a type (line 67)
        Name_21633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 40), ast_21632, 'Name')
        # Calling Name(args, kwargs) (line 67)
        Name_call_result_21642 = invoke(stypy.reporting.localization.Localization(__file__, 67, 40), Name_21633, *[], **kwargs_21641)
        
        keyword_21643 = Name_call_result_21642
        kwargs_21644 = {'ctx': keyword_21631, 'attr': keyword_21626, 'value': keyword_21643}
        # Getting the type of 'ast' (line 66)
        ast_21623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'ast', False)
        # Obtaining the member 'Attribute' of a type (line 66)
        Attribute_21624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 20), ast_21623, 'Attribute')
        # Calling Attribute(args, kwargs) (line 66)
        Attribute_call_result_21645 = invoke(stypy.reporting.localization.Localization(__file__, 66, 20), Attribute_21624, *[], **kwargs_21644)
        
        # Assigning a type to the variable 'attribute' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'attribute', Attribute_call_result_21645)
        
        # Assigning a Call to a Name (line 68):
        
        # Call to Call(...): (line 68)
        # Processing the call keyword arguments (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_21648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        # Getting the type of 'locals_call' (line 68)
        locals_call_21649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 30), 'locals_call', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 29), list_21648, locals_call_21649)
        
        keyword_21650 = list_21648
        # Getting the type of 'attribute' (line 68)
        attribute_21651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 49), 'attribute', False)
        keyword_21652 = attribute_21651
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_21653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 69), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        
        keyword_21654 = list_21653
        # Getting the type of 'None' (line 68)
        None_21655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 80), 'None', False)
        keyword_21656 = None_21655
        # Getting the type of 'None' (line 68)
        None_21657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 95), 'None', False)
        keyword_21658 = None_21657
        kwargs_21659 = {'keywords': keyword_21654, 'starargs': keyword_21658, 'args': keyword_21650, 'func': keyword_21652, 'kwargs': keyword_21656}
        # Getting the type of 'ast' (line 68)
        ast_21646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'ast', False)
        # Obtaining the member 'Call' of a type (line 68)
        Call_21647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 15), ast_21646, 'Call')
        # Calling Call(args, kwargs) (line 68)
        Call_call_result_21660 = invoke(stypy.reporting.localization.Localization(__file__, 68, 15), Call_21647, *[], **kwargs_21659)
        
        # Assigning a type to the variable 'call' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'call', Call_call_result_21660)
        
        # Assigning a Call to a Name (line 69):
        
        # Call to Expr(...): (line 69)
        # Processing the call keyword arguments (line 69)
        # Getting the type of 'call' (line 69)
        call_21663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 30), 'call', False)
        keyword_21664 = call_21663
        kwargs_21665 = {'value': keyword_21664}
        # Getting the type of 'ast' (line 69)
        ast_21661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'ast', False)
        # Obtaining the member 'Expr' of a type (line 69)
        Expr_21662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 15), ast_21661, 'Expr')
        # Calling Expr(args, kwargs) (line 69)
        Expr_call_result_21666 = invoke(stypy.reporting.localization.Localization(__file__, 69, 15), Expr_21662, *[], **kwargs_21665)
        
        # Assigning a type to the variable 'expr' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'expr', Expr_call_result_21666)
        
        # Call to append(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'expr' (line 70)
        expr_21670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 25), 'expr', False)
        # Processing the call keyword arguments (line 70)
        kwargs_21671 = {}
        # Getting the type of 'node' (line 70)
        node_21667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'node', False)
        # Obtaining the member 'body' of a type (line 70)
        body_21668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), node_21667, 'body')
        # Obtaining the member 'append' of a type (line 70)
        append_21669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), body_21668, 'append')
        # Calling append(args, kwargs) (line 70)
        append_call_result_21672 = invoke(stypy.reporting.localization.Localization(__file__, 70, 8), append_21669, *[expr_21670], **kwargs_21671)
        
        # Getting the type of 'node' (line 72)
        node_21673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'stypy_return_type', node_21673)
        
        # ################# End of 'visit_FunctionDef(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_FunctionDef' in the type store
        # Getting the type of 'stypy_return_type' (line 60)
        stypy_return_type_21674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21674)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_FunctionDef'
        return stypy_return_type_21674


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
        node_21677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 19), 'node', False)
        # Obtaining the member 'value' of a type (line 75)
        value_21678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 19), node_21677, 'value')
        # Processing the call keyword arguments (line 75)
        kwargs_21679 = {}
        # Getting the type of 'self' (line 75)
        self_21675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 75)
        visit_21676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_21675, 'visit')
        # Calling visit(args, kwargs) (line 75)
        visit_call_result_21680 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), visit_21676, *[value_21678], **kwargs_21679)
        
        
        # Assigning a Call to a Name (line 77):
        
        # Call to Index(...): (line 77)
        # Processing the call keyword arguments (line 77)
        
        # Call to Num(...): (line 77)
        # Processing the call keyword arguments (line 77)
        int_21685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 42), 'int')
        keyword_21686 = int_21685
        kwargs_21687 = {'n': keyword_21686}
        # Getting the type of 'ast' (line 77)
        ast_21683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 32), 'ast', False)
        # Obtaining the member 'Num' of a type (line 77)
        Num_21684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 32), ast_21683, 'Num')
        # Calling Num(args, kwargs) (line 77)
        Num_call_result_21688 = invoke(stypy.reporting.localization.Localization(__file__, 77, 32), Num_21684, *[], **kwargs_21687)
        
        keyword_21689 = Num_call_result_21688
        kwargs_21690 = {'value': keyword_21689}
        # Getting the type of 'ast' (line 77)
        ast_21681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'ast', False)
        # Obtaining the member 'Index' of a type (line 77)
        Index_21682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 16), ast_21681, 'Index')
        # Calling Index(args, kwargs) (line 77)
        Index_call_result_21691 = invoke(stypy.reporting.localization.Localization(__file__, 77, 16), Index_21682, *[], **kwargs_21690)
        
        # Assigning a type to the variable 'index' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'index', Index_call_result_21691)
        
        # Assigning a Call to a Name (line 78):
        
        # Call to Call(...): (line 78)
        # Processing the call keyword arguments (line 78)
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_21694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        
        keyword_21695 = list_21694
        
        # Call to Name(...): (line 78)
        # Processing the call keyword arguments (line 78)
        str_21698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 57), 'str', 'locals')
        keyword_21699 = str_21698
        
        # Call to Load(...): (line 78)
        # Processing the call keyword arguments (line 78)
        kwargs_21702 = {}
        # Getting the type of 'ast' (line 78)
        ast_21700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 71), 'ast', False)
        # Obtaining the member 'Load' of a type (line 78)
        Load_21701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 71), ast_21700, 'Load')
        # Calling Load(args, kwargs) (line 78)
        Load_call_result_21703 = invoke(stypy.reporting.localization.Localization(__file__, 78, 71), Load_21701, *[], **kwargs_21702)
        
        keyword_21704 = Load_call_result_21703
        kwargs_21705 = {'ctx': keyword_21704, 'id': keyword_21699}
        # Getting the type of 'ast' (line 78)
        ast_21696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 45), 'ast', False)
        # Obtaining the member 'Name' of a type (line 78)
        Name_21697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 45), ast_21696, 'Name')
        # Calling Name(args, kwargs) (line 78)
        Name_call_result_21706 = invoke(stypy.reporting.localization.Localization(__file__, 78, 45), Name_21697, *[], **kwargs_21705)
        
        keyword_21707 = Name_call_result_21706
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_21708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 93), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        
        keyword_21709 = list_21708
        # Getting the type of 'None' (line 78)
        None_21710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 104), 'None', False)
        keyword_21711 = None_21710
        # Getting the type of 'None' (line 79)
        None_21712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 40), 'None', False)
        keyword_21713 = None_21712
        kwargs_21714 = {'keywords': keyword_21709, 'starargs': keyword_21713, 'args': keyword_21695, 'func': keyword_21707, 'kwargs': keyword_21711}
        # Getting the type of 'ast' (line 78)
        ast_21692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 22), 'ast', False)
        # Obtaining the member 'Call' of a type (line 78)
        Call_21693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 22), ast_21692, 'Call')
        # Calling Call(args, kwargs) (line 78)
        Call_call_result_21715 = invoke(stypy.reporting.localization.Localization(__file__, 78, 22), Call_21693, *[], **kwargs_21714)
        
        # Assigning a type to the variable 'locals_call' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'locals_call', Call_call_result_21715)
        
        # Assigning a Call to a Name (line 80):
        
        # Call to Attribute(...): (line 80)
        # Processing the call keyword arguments (line 80)
        str_21718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 39), 'str', 'add_type_dict_for_context')
        keyword_21719 = str_21718
        
        # Call to Load(...): (line 80)
        # Processing the call keyword arguments (line 80)
        kwargs_21722 = {}
        # Getting the type of 'ast' (line 80)
        ast_21720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 72), 'ast', False)
        # Obtaining the member 'Load' of a type (line 80)
        Load_21721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 72), ast_21720, 'Load')
        # Calling Load(args, kwargs) (line 80)
        Load_call_result_21723 = invoke(stypy.reporting.localization.Localization(__file__, 80, 72), Load_21721, *[], **kwargs_21722)
        
        keyword_21724 = Load_call_result_21723
        
        # Call to Name(...): (line 81)
        # Processing the call keyword arguments (line 81)
        # Getting the type of 'type_test_name' (line 81)
        type_test_name_21727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 52), 'type_test_name', False)
        keyword_21728 = type_test_name_21727
        
        # Call to Load(...): (line 81)
        # Processing the call keyword arguments (line 81)
        kwargs_21731 = {}
        # Getting the type of 'ast' (line 81)
        ast_21729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 72), 'ast', False)
        # Obtaining the member 'Load' of a type (line 81)
        Load_21730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 72), ast_21729, 'Load')
        # Calling Load(args, kwargs) (line 81)
        Load_call_result_21732 = invoke(stypy.reporting.localization.Localization(__file__, 81, 72), Load_21730, *[], **kwargs_21731)
        
        keyword_21733 = Load_call_result_21732
        kwargs_21734 = {'ctx': keyword_21733, 'id': keyword_21728}
        # Getting the type of 'ast' (line 81)
        ast_21725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 40), 'ast', False)
        # Obtaining the member 'Name' of a type (line 81)
        Name_21726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 40), ast_21725, 'Name')
        # Calling Name(args, kwargs) (line 81)
        Name_call_result_21735 = invoke(stypy.reporting.localization.Localization(__file__, 81, 40), Name_21726, *[], **kwargs_21734)
        
        keyword_21736 = Name_call_result_21735
        kwargs_21737 = {'ctx': keyword_21724, 'attr': keyword_21719, 'value': keyword_21736}
        # Getting the type of 'ast' (line 80)
        ast_21716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'ast', False)
        # Obtaining the member 'Attribute' of a type (line 80)
        Attribute_21717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 20), ast_21716, 'Attribute')
        # Calling Attribute(args, kwargs) (line 80)
        Attribute_call_result_21738 = invoke(stypy.reporting.localization.Localization(__file__, 80, 20), Attribute_21717, *[], **kwargs_21737)
        
        # Assigning a type to the variable 'attribute' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'attribute', Attribute_call_result_21738)
        
        # Assigning a Call to a Name (line 82):
        
        # Call to Call(...): (line 82)
        # Processing the call keyword arguments (line 82)
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_21741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        # Adding element type (line 82)
        # Getting the type of 'locals_call' (line 82)
        locals_call_21742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 30), 'locals_call', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 29), list_21741, locals_call_21742)
        
        keyword_21743 = list_21741
        # Getting the type of 'attribute' (line 82)
        attribute_21744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 49), 'attribute', False)
        keyword_21745 = attribute_21744
        
        # Obtaining an instance of the builtin type 'list' (line 82)
        list_21746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 69), 'list')
        # Adding type elements to the builtin type 'list' instance (line 82)
        
        keyword_21747 = list_21746
        # Getting the type of 'None' (line 82)
        None_21748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 80), 'None', False)
        keyword_21749 = None_21748
        # Getting the type of 'None' (line 82)
        None_21750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 95), 'None', False)
        keyword_21751 = None_21750
        kwargs_21752 = {'keywords': keyword_21747, 'starargs': keyword_21751, 'args': keyword_21743, 'func': keyword_21745, 'kwargs': keyword_21749}
        # Getting the type of 'ast' (line 82)
        ast_21739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'ast', False)
        # Obtaining the member 'Call' of a type (line 82)
        Call_21740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), ast_21739, 'Call')
        # Calling Call(args, kwargs) (line 82)
        Call_call_result_21753 = invoke(stypy.reporting.localization.Localization(__file__, 82, 15), Call_21740, *[], **kwargs_21752)
        
        # Assigning a type to the variable 'call' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'call', Call_call_result_21753)
        
        # Assigning a Call to a Name (line 83):
        
        # Call to Tuple(...): (line 83)
        # Processing the call keyword arguments (line 83)
        
        # Call to Load(...): (line 83)
        # Processing the call keyword arguments (line 83)
        kwargs_21758 = {}
        # Getting the type of 'ast' (line 83)
        ast_21756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 31), 'ast', False)
        # Obtaining the member 'Load' of a type (line 83)
        Load_21757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 31), ast_21756, 'Load')
        # Calling Load(args, kwargs) (line 83)
        Load_call_result_21759 = invoke(stypy.reporting.localization.Localization(__file__, 83, 31), Load_21757, *[], **kwargs_21758)
        
        keyword_21760 = Load_call_result_21759
        
        # Obtaining an instance of the builtin type 'list' (line 83)
        list_21761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 83)
        # Adding element type (line 83)
        # Getting the type of 'node' (line 83)
        node_21762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 49), 'node', False)
        # Obtaining the member 'value' of a type (line 83)
        value_21763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 49), node_21762, 'value')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 48), list_21761, value_21763)
        # Adding element type (line 83)
        # Getting the type of 'call' (line 83)
        call_21764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 61), 'call', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 48), list_21761, call_21764)
        
        keyword_21765 = list_21761
        kwargs_21766 = {'elts': keyword_21765, 'ctx': keyword_21760}
        # Getting the type of 'ast' (line 83)
        ast_21754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 17), 'ast', False)
        # Obtaining the member 'Tuple' of a type (line 83)
        Tuple_21755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 17), ast_21754, 'Tuple')
        # Calling Tuple(args, kwargs) (line 83)
        Tuple_call_result_21767 = invoke(stypy.reporting.localization.Localization(__file__, 83, 17), Tuple_21755, *[], **kwargs_21766)
        
        # Assigning a type to the variable 'tuple_' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'tuple_', Tuple_call_result_21767)
        
        # Assigning a Call to a Name (line 84):
        
        # Call to Subscript(...): (line 84)
        # Processing the call keyword arguments (line 84)
        
        # Call to Load(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_21772 = {}
        # Getting the type of 'ast' (line 84)
        ast_21770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 38), 'ast', False)
        # Obtaining the member 'Load' of a type (line 84)
        Load_21771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 38), ast_21770, 'Load')
        # Calling Load(args, kwargs) (line 84)
        Load_call_result_21773 = invoke(stypy.reporting.localization.Localization(__file__, 84, 38), Load_21771, *[], **kwargs_21772)
        
        keyword_21774 = Load_call_result_21773
        # Getting the type of 'index' (line 84)
        index_21775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 56), 'index', False)
        keyword_21776 = index_21775
        # Getting the type of 'tuple_' (line 84)
        tuple__21777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 69), 'tuple_', False)
        keyword_21778 = tuple__21777
        kwargs_21779 = {'slice': keyword_21776, 'ctx': keyword_21774, 'value': keyword_21778}
        # Getting the type of 'ast' (line 84)
        ast_21768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'ast', False)
        # Obtaining the member 'Subscript' of a type (line 84)
        Subscript_21769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 20), ast_21768, 'Subscript')
        # Calling Subscript(args, kwargs) (line 84)
        Subscript_call_result_21780 = invoke(stypy.reporting.localization.Localization(__file__, 84, 20), Subscript_21769, *[], **kwargs_21779)
        
        # Assigning a type to the variable 'subscript' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'subscript', Subscript_call_result_21780)
        
        # Assigning a Name to a Attribute (line 85):
        # Getting the type of 'subscript' (line 85)
        subscript_21781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'subscript')
        # Getting the type of 'node' (line 85)
        node_21782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'node')
        # Setting the type of the member 'value' of a type (line 85)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), node_21782, 'value', subscript_21781)
        # Getting the type of 'node' (line 87)
        node_21783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type', node_21783)
        
        # ################# End of 'visit_Return(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Return' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_21784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21784)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Return'
        return stypy_return_type_21784


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
