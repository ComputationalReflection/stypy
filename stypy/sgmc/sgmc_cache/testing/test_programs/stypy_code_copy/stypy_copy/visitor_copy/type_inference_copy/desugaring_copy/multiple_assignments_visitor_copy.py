
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import copy
2: 
3: from ....code_generation_copy.type_inference_programs_copy.aux_functions_copy import *
4: from ....visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, stypy_functions_copy, functions_copy
5: import ast
6: 
7: '''
8: This visitor decompose several forms of multiple assignments into single assignments, that can be properly processed
9: by stypy. There are various forms of assignments in Python that involve multiple elements, such as:
10: 
11: a, b = c, d
12: a, b = [c, d]
13: a, b = function_that_returns_tuple()
14: a, b = function_that_returns_list()
15: 
16: This visitor reduces the complexity of dealing with assignments when generating type inference code
17: '''
18: 
19: 
20: def multiple_value_call_assignment_handler(target, value, assign_stmts, node, id_str):
21:     '''
22:     Handles code that uses a multiple assignment with a call to a function in the right part of the assignment. The
23:     code var1, var2 = original_call(...) is transformed into:
24: 
25:     temp = original_call(...)
26:     var1 = temp[0]
27:     var2 = temp[1]
28:     ...
29: 
30:     This way we only perform the call once.
31: 
32:     :param target: tuple or list of variables
33:     :param value: call
34:     :param assign_stmts: statements holder list
35:     :param node: current AST node
36:     :param id_str: Type of assignment that we are processing (to create variables)
37:     :return:
38:     '''
39:     target_stmts, value_var = stypy_functions_copy.create_temp_Assign(value, node.lineno, node.col_offset,
40:                                                                  "{0}_assignment".format(id_str))
41:     assign_stmts.append(target_stmts)
42: 
43:     value_var_to_load = copy.deepcopy(value_var)
44:     value_var_to_load.ctx = ast.Load()
45: 
46:     for i in range(len(target.elts)):
47:         # Assign values to each element.
48:         getitem_att = core_language_copy.create_attribute(value_var_to_load, '__getitem__', context=ast.Load(),
49:                                                      line=node.lineno,
50:                                                      column=node.col_offset)
51:         item_call = functions_copy.create_call(getitem_att, [core_language_copy.create_num(i, node.lineno, node.col_offset)])
52:         temp_stmts, temp_value = stypy_functions_copy.create_temp_Assign(item_call, node.lineno, node.col_offset,
53:                                                                     "{0}_assignment".format(id_str))
54:         assign_stmts.append(temp_stmts)
55: 
56:         temp_stmts = core_language_copy.create_Assign(target.elts[i], temp_value)
57:         assign_stmts.append(temp_stmts)
58: 
59: 
60: def multiple_value_assignment_handler(target, value, assign_stmts, node, id_str):
61:     '''
62:     Code to handle assignments like a, b = c, d. This code is converted to:
63:     a = c
64:     b = d
65: 
66:     Length of left and right part is cheched to make sure we are dealing with a valid assignment (an error is produced
67:     otherwise)
68: 
69:     :param target: tuple or list of variables
70:     :param value:  tuple or list of variables
71:     :param assign_stmts: statements holder list
72:     :param node: current AST node
73:     :param id_str: Type of assignment that we are processing (to create variables)
74:     :return:
75:     '''
76:     if len(target.elts) == len(value.elts):
77:         temp_var_names = []
78: 
79:         for i in range(len(value.elts)):
80:             temp_stmts, temp_value = stypy_functions_copy.create_temp_Assign(value.elts[i], node.lineno, node.col_offset,
81:                                                                         "{0}_assignment".format(id_str))
82:             assign_stmts.append(temp_stmts)
83:             temp_var_names.append(temp_value)
84:         for i in range(len(target.elts)):
85:             temp_stmts = core_language_copy.create_Assign(target.elts[i], temp_var_names[i])
86:             assign_stmts.append(temp_stmts)
87:     else:
88:         TypeError(stypy_functions_copy.create_localization(node.lineno, node.col_offset),
89:                   "Multi-value assignments with {0}s must have the same amount of elements on both assignment sides".
90:                   format(id_str))
91: 
92: 
93: def single_assignment_handler(target, value, assign_stmts, node, id_str):
94:     '''
95:     Handles single statements for hte visitor. No change is produced in the code
96:     :param target: Variable
97:     :param value: Value to assign
98:     :param assign_stmts: statements holder list
99:     :param node: current AST node
100:     :param id_str: Type of assignment that we are processing (to create variables)
101:     :return:
102:     '''
103:     temp_stmts = core_language_copy.create_Assign(target, value)
104:     assign_stmts.append(temp_stmts)
105: 
106: 
107: class MultipleAssignmentsDesugaringVisitor(ast.NodeTransformer):
108:     # Table of functions that determines what assignment handler is going to be executed for an assignment. Each
109:     # key is a function that, if evaluated to true, execute the associated value function that adds the necessary
110:     # statements to handle the call
111:     __assignment_handlers = {
112:         (lambda target, value: isinstance(target, ast.Tuple) and (isinstance(value, ast.Tuple) or
113:                                                                   isinstance(value, ast.List))): (
114:             "tuple", multiple_value_assignment_handler),
115: 
116:         (lambda target, value: isinstance(target, ast.List) and (isinstance(value, ast.Tuple) or
117:                                                                  isinstance(value, ast.List))): (
118:             "list", multiple_value_assignment_handler),
119: 
120:         (lambda target, value: (isinstance(target, ast.List) or isinstance(target, ast.Tuple)) and (
121:             isinstance(value, ast.Call))): ("call", multiple_value_call_assignment_handler),
122: 
123:         lambda target, value: isinstance(target, ast.Name):
124:             ("assignment", single_assignment_handler),
125: 
126:         lambda target, value: isinstance(target, ast.Subscript):
127:             ("assignment", single_assignment_handler),
128: 
129:         lambda target, value: isinstance(target, ast.Attribute):
130:             ("assignment", single_assignment_handler),
131:     }
132: 
133:     # ######################################### MAIN MODULE #############################################
134: 
135:     def visit_Assign(self, node):
136:         assign_stmts = []
137:         value = node.value
138:         reversed_targets = node.targets
139:         reversed_targets.reverse()
140:         assign_stmts.append(stypy_functions_copy.create_blank_line())
141:         if len(reversed_targets) > 1:
142:             assign_stmts.append(
143:                 stypy_functions_copy.create_src_comment("Multiple assigment of {0} elements.".format(len(reversed_targets))))
144:         else:
145:             assign_stmts.append(stypy_functions_copy.create_src_comment(
146:                 "Assignment to a {0} from a {1}".format(type(reversed_targets[0]).__name__,
147:                                                         type(value).__name__)))
148: 
149:         for assign_num in range(len(reversed_targets)):
150:             target = reversed_targets[assign_num]
151:             # Function guard is true? execute handler
152:             for handler_func_guard in self.__assignment_handlers:
153:                 if handler_func_guard(target, value):
154:                     id_str, handler_func = self.__assignment_handlers[handler_func_guard]
155:                     handler_func(target, value, assign_stmts, node, id_str)
156:                     assign_stmts = stypy_functions_copy.flatten_lists(assign_stmts)
157:                     value = target
158:                     break
159: 
160:         if len(assign_stmts) > 0:
161:             return assign_stmts
162:         return node
163: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import copy' statement (line 1)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import ' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')
import_30364 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy')

if (type(import_30364) is not StypyTypeError):

    if (import_30364 != 'pyd_module'):
        __import__(import_30364)
        sys_modules_30365 = sys.modules[import_30364]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', sys_modules_30365.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_30365, sys_modules_30365.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', import_30364)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, stypy_functions_copy, functions_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')
import_30366 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy')

if (type(import_30366) is not StypyTypeError):

    if (import_30366 != 'pyd_module'):
        __import__(import_30366)
        sys_modules_30367 = sys.modules[import_30366]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', sys_modules_30367.module_type_store, module_type_store, ['core_language_copy', 'stypy_functions_copy', 'functions_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_30367, sys_modules_30367.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, stypy_functions_copy, functions_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', None, module_type_store, ['core_language_copy', 'stypy_functions_copy', 'functions_copy'], [core_language_copy, stypy_functions_copy, functions_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', import_30366)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import ast' statement (line 5)
import ast

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'ast', ast, module_type_store)

str_30368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\nThis visitor decompose several forms of multiple assignments into single assignments, that can be properly processed\nby stypy. There are various forms of assignments in Python that involve multiple elements, such as:\n\na, b = c, d\na, b = [c, d]\na, b = function_that_returns_tuple()\na, b = function_that_returns_list()\n\nThis visitor reduces the complexity of dealing with assignments when generating type inference code\n')

@norecursion
def multiple_value_call_assignment_handler(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'multiple_value_call_assignment_handler'
    module_type_store = module_type_store.open_function_context('multiple_value_call_assignment_handler', 20, 0, False)
    
    # Passed parameters checking function
    multiple_value_call_assignment_handler.stypy_localization = localization
    multiple_value_call_assignment_handler.stypy_type_of_self = None
    multiple_value_call_assignment_handler.stypy_type_store = module_type_store
    multiple_value_call_assignment_handler.stypy_function_name = 'multiple_value_call_assignment_handler'
    multiple_value_call_assignment_handler.stypy_param_names_list = ['target', 'value', 'assign_stmts', 'node', 'id_str']
    multiple_value_call_assignment_handler.stypy_varargs_param_name = None
    multiple_value_call_assignment_handler.stypy_kwargs_param_name = None
    multiple_value_call_assignment_handler.stypy_call_defaults = defaults
    multiple_value_call_assignment_handler.stypy_call_varargs = varargs
    multiple_value_call_assignment_handler.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'multiple_value_call_assignment_handler', ['target', 'value', 'assign_stmts', 'node', 'id_str'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'multiple_value_call_assignment_handler', localization, ['target', 'value', 'assign_stmts', 'node', 'id_str'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'multiple_value_call_assignment_handler(...)' code ##################

    str_30369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, (-1)), 'str', '\n    Handles code that uses a multiple assignment with a call to a function in the right part of the assignment. The\n    code var1, var2 = original_call(...) is transformed into:\n\n    temp = original_call(...)\n    var1 = temp[0]\n    var2 = temp[1]\n    ...\n\n    This way we only perform the call once.\n\n    :param target: tuple or list of variables\n    :param value: call\n    :param assign_stmts: statements holder list\n    :param node: current AST node\n    :param id_str: Type of assignment that we are processing (to create variables)\n    :return:\n    ')
    
    # Assigning a Call to a Tuple (line 39):
    
    # Assigning a Call to a Name:
    
    # Call to create_temp_Assign(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'value' (line 39)
    value_30372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 70), 'value', False)
    # Getting the type of 'node' (line 39)
    node_30373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 77), 'node', False)
    # Obtaining the member 'lineno' of a type (line 39)
    lineno_30374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 77), node_30373, 'lineno')
    # Getting the type of 'node' (line 39)
    node_30375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 90), 'node', False)
    # Obtaining the member 'col_offset' of a type (line 39)
    col_offset_30376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 90), node_30375, 'col_offset')
    
    # Call to format(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'id_str' (line 40)
    id_str_30379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 89), 'id_str', False)
    # Processing the call keyword arguments (line 40)
    kwargs_30380 = {}
    str_30377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 65), 'str', '{0}_assignment')
    # Obtaining the member 'format' of a type (line 40)
    format_30378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 65), str_30377, 'format')
    # Calling format(args, kwargs) (line 40)
    format_call_result_30381 = invoke(stypy.reporting.localization.Localization(__file__, 40, 65), format_30378, *[id_str_30379], **kwargs_30380)
    
    # Processing the call keyword arguments (line 39)
    kwargs_30382 = {}
    # Getting the type of 'stypy_functions_copy' (line 39)
    stypy_functions_copy_30370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 30), 'stypy_functions_copy', False)
    # Obtaining the member 'create_temp_Assign' of a type (line 39)
    create_temp_Assign_30371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 30), stypy_functions_copy_30370, 'create_temp_Assign')
    # Calling create_temp_Assign(args, kwargs) (line 39)
    create_temp_Assign_call_result_30383 = invoke(stypy.reporting.localization.Localization(__file__, 39, 30), create_temp_Assign_30371, *[value_30372, lineno_30374, col_offset_30376, format_call_result_30381], **kwargs_30382)
    
    # Assigning a type to the variable 'call_assignment_30353' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'call_assignment_30353', create_temp_Assign_call_result_30383)
    
    # Assigning a Call to a Name (line 39):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_30353' (line 39)
    call_assignment_30353_30384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'call_assignment_30353', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_30385 = stypy_get_value_from_tuple(call_assignment_30353_30384, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_30354' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'call_assignment_30354', stypy_get_value_from_tuple_call_result_30385)
    
    # Assigning a Name to a Name (line 39):
    # Getting the type of 'call_assignment_30354' (line 39)
    call_assignment_30354_30386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'call_assignment_30354')
    # Assigning a type to the variable 'target_stmts' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'target_stmts', call_assignment_30354_30386)
    
    # Assigning a Call to a Name (line 39):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_30353' (line 39)
    call_assignment_30353_30387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'call_assignment_30353', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_30388 = stypy_get_value_from_tuple(call_assignment_30353_30387, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_30355' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'call_assignment_30355', stypy_get_value_from_tuple_call_result_30388)
    
    # Assigning a Name to a Name (line 39):
    # Getting the type of 'call_assignment_30355' (line 39)
    call_assignment_30355_30389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'call_assignment_30355')
    # Assigning a type to the variable 'value_var' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 18), 'value_var', call_assignment_30355_30389)
    
    # Call to append(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'target_stmts' (line 41)
    target_stmts_30392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), 'target_stmts', False)
    # Processing the call keyword arguments (line 41)
    kwargs_30393 = {}
    # Getting the type of 'assign_stmts' (line 41)
    assign_stmts_30390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'assign_stmts', False)
    # Obtaining the member 'append' of a type (line 41)
    append_30391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), assign_stmts_30390, 'append')
    # Calling append(args, kwargs) (line 41)
    append_call_result_30394 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), append_30391, *[target_stmts_30392], **kwargs_30393)
    
    
    # Assigning a Call to a Name (line 43):
    
    # Assigning a Call to a Name (line 43):
    
    # Call to deepcopy(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'value_var' (line 43)
    value_var_30397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 38), 'value_var', False)
    # Processing the call keyword arguments (line 43)
    kwargs_30398 = {}
    # Getting the type of 'copy' (line 43)
    copy_30395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'copy', False)
    # Obtaining the member 'deepcopy' of a type (line 43)
    deepcopy_30396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 24), copy_30395, 'deepcopy')
    # Calling deepcopy(args, kwargs) (line 43)
    deepcopy_call_result_30399 = invoke(stypy.reporting.localization.Localization(__file__, 43, 24), deepcopy_30396, *[value_var_30397], **kwargs_30398)
    
    # Assigning a type to the variable 'value_var_to_load' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'value_var_to_load', deepcopy_call_result_30399)
    
    # Assigning a Call to a Attribute (line 44):
    
    # Assigning a Call to a Attribute (line 44):
    
    # Call to Load(...): (line 44)
    # Processing the call keyword arguments (line 44)
    kwargs_30402 = {}
    # Getting the type of 'ast' (line 44)
    ast_30400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 28), 'ast', False)
    # Obtaining the member 'Load' of a type (line 44)
    Load_30401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 28), ast_30400, 'Load')
    # Calling Load(args, kwargs) (line 44)
    Load_call_result_30403 = invoke(stypy.reporting.localization.Localization(__file__, 44, 28), Load_30401, *[], **kwargs_30402)
    
    # Getting the type of 'value_var_to_load' (line 44)
    value_var_to_load_30404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'value_var_to_load')
    # Setting the type of the member 'ctx' of a type (line 44)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 4), value_var_to_load_30404, 'ctx', Load_call_result_30403)
    
    
    # Call to range(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Call to len(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'target' (line 46)
    target_30407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 23), 'target', False)
    # Obtaining the member 'elts' of a type (line 46)
    elts_30408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 23), target_30407, 'elts')
    # Processing the call keyword arguments (line 46)
    kwargs_30409 = {}
    # Getting the type of 'len' (line 46)
    len_30406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'len', False)
    # Calling len(args, kwargs) (line 46)
    len_call_result_30410 = invoke(stypy.reporting.localization.Localization(__file__, 46, 19), len_30406, *[elts_30408], **kwargs_30409)
    
    # Processing the call keyword arguments (line 46)
    kwargs_30411 = {}
    # Getting the type of 'range' (line 46)
    range_30405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'range', False)
    # Calling range(args, kwargs) (line 46)
    range_call_result_30412 = invoke(stypy.reporting.localization.Localization(__file__, 46, 13), range_30405, *[len_call_result_30410], **kwargs_30411)
    
    # Assigning a type to the variable 'range_call_result_30412' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'range_call_result_30412', range_call_result_30412)
    # Testing if the for loop is going to be iterated (line 46)
    # Testing the type of a for loop iterable (line 46)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 46, 4), range_call_result_30412)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 46, 4), range_call_result_30412):
        # Getting the type of the for loop variable (line 46)
        for_loop_var_30413 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 46, 4), range_call_result_30412)
        # Assigning a type to the variable 'i' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'i', for_loop_var_30413)
        # SSA begins for a for statement (line 46)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 48):
        
        # Assigning a Call to a Name (line 48):
        
        # Call to create_attribute(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'value_var_to_load' (line 48)
        value_var_to_load_30416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 58), 'value_var_to_load', False)
        str_30417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 77), 'str', '__getitem__')
        # Processing the call keyword arguments (line 48)
        
        # Call to Load(...): (line 48)
        # Processing the call keyword arguments (line 48)
        kwargs_30420 = {}
        # Getting the type of 'ast' (line 48)
        ast_30418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 100), 'ast', False)
        # Obtaining the member 'Load' of a type (line 48)
        Load_30419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 100), ast_30418, 'Load')
        # Calling Load(args, kwargs) (line 48)
        Load_call_result_30421 = invoke(stypy.reporting.localization.Localization(__file__, 48, 100), Load_30419, *[], **kwargs_30420)
        
        keyword_30422 = Load_call_result_30421
        # Getting the type of 'node' (line 49)
        node_30423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 58), 'node', False)
        # Obtaining the member 'lineno' of a type (line 49)
        lineno_30424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 58), node_30423, 'lineno')
        keyword_30425 = lineno_30424
        # Getting the type of 'node' (line 50)
        node_30426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 60), 'node', False)
        # Obtaining the member 'col_offset' of a type (line 50)
        col_offset_30427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 60), node_30426, 'col_offset')
        keyword_30428 = col_offset_30427
        kwargs_30429 = {'column': keyword_30428, 'line': keyword_30425, 'context': keyword_30422}
        # Getting the type of 'core_language_copy' (line 48)
        core_language_copy_30414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'core_language_copy', False)
        # Obtaining the member 'create_attribute' of a type (line 48)
        create_attribute_30415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 22), core_language_copy_30414, 'create_attribute')
        # Calling create_attribute(args, kwargs) (line 48)
        create_attribute_call_result_30430 = invoke(stypy.reporting.localization.Localization(__file__, 48, 22), create_attribute_30415, *[value_var_to_load_30416, str_30417], **kwargs_30429)
        
        # Assigning a type to the variable 'getitem_att' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'getitem_att', create_attribute_call_result_30430)
        
        # Assigning a Call to a Name (line 51):
        
        # Assigning a Call to a Name (line 51):
        
        # Call to create_call(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'getitem_att' (line 51)
        getitem_att_30433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 47), 'getitem_att', False)
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_30434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        
        # Call to create_num(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'i' (line 51)
        i_30437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 91), 'i', False)
        # Getting the type of 'node' (line 51)
        node_30438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 94), 'node', False)
        # Obtaining the member 'lineno' of a type (line 51)
        lineno_30439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 94), node_30438, 'lineno')
        # Getting the type of 'node' (line 51)
        node_30440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 107), 'node', False)
        # Obtaining the member 'col_offset' of a type (line 51)
        col_offset_30441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 107), node_30440, 'col_offset')
        # Processing the call keyword arguments (line 51)
        kwargs_30442 = {}
        # Getting the type of 'core_language_copy' (line 51)
        core_language_copy_30435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 61), 'core_language_copy', False)
        # Obtaining the member 'create_num' of a type (line 51)
        create_num_30436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 61), core_language_copy_30435, 'create_num')
        # Calling create_num(args, kwargs) (line 51)
        create_num_call_result_30443 = invoke(stypy.reporting.localization.Localization(__file__, 51, 61), create_num_30436, *[i_30437, lineno_30439, col_offset_30441], **kwargs_30442)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 60), list_30434, create_num_call_result_30443)
        
        # Processing the call keyword arguments (line 51)
        kwargs_30444 = {}
        # Getting the type of 'functions_copy' (line 51)
        functions_copy_30431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'functions_copy', False)
        # Obtaining the member 'create_call' of a type (line 51)
        create_call_30432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 20), functions_copy_30431, 'create_call')
        # Calling create_call(args, kwargs) (line 51)
        create_call_call_result_30445 = invoke(stypy.reporting.localization.Localization(__file__, 51, 20), create_call_30432, *[getitem_att_30433, list_30434], **kwargs_30444)
        
        # Assigning a type to the variable 'item_call' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'item_call', create_call_call_result_30445)
        
        # Assigning a Call to a Tuple (line 52):
        
        # Assigning a Call to a Name:
        
        # Call to create_temp_Assign(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'item_call' (line 52)
        item_call_30448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 73), 'item_call', False)
        # Getting the type of 'node' (line 52)
        node_30449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 84), 'node', False)
        # Obtaining the member 'lineno' of a type (line 52)
        lineno_30450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 84), node_30449, 'lineno')
        # Getting the type of 'node' (line 52)
        node_30451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 97), 'node', False)
        # Obtaining the member 'col_offset' of a type (line 52)
        col_offset_30452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 97), node_30451, 'col_offset')
        
        # Call to format(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'id_str' (line 53)
        id_str_30455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 92), 'id_str', False)
        # Processing the call keyword arguments (line 53)
        kwargs_30456 = {}
        str_30453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 68), 'str', '{0}_assignment')
        # Obtaining the member 'format' of a type (line 53)
        format_30454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 68), str_30453, 'format')
        # Calling format(args, kwargs) (line 53)
        format_call_result_30457 = invoke(stypy.reporting.localization.Localization(__file__, 53, 68), format_30454, *[id_str_30455], **kwargs_30456)
        
        # Processing the call keyword arguments (line 52)
        kwargs_30458 = {}
        # Getting the type of 'stypy_functions_copy' (line 52)
        stypy_functions_copy_30446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 33), 'stypy_functions_copy', False)
        # Obtaining the member 'create_temp_Assign' of a type (line 52)
        create_temp_Assign_30447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 33), stypy_functions_copy_30446, 'create_temp_Assign')
        # Calling create_temp_Assign(args, kwargs) (line 52)
        create_temp_Assign_call_result_30459 = invoke(stypy.reporting.localization.Localization(__file__, 52, 33), create_temp_Assign_30447, *[item_call_30448, lineno_30450, col_offset_30452, format_call_result_30457], **kwargs_30458)
        
        # Assigning a type to the variable 'call_assignment_30356' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'call_assignment_30356', create_temp_Assign_call_result_30459)
        
        # Assigning a Call to a Name (line 52):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_30356' (line 52)
        call_assignment_30356_30460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'call_assignment_30356', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_30461 = stypy_get_value_from_tuple(call_assignment_30356_30460, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_30357' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'call_assignment_30357', stypy_get_value_from_tuple_call_result_30461)
        
        # Assigning a Name to a Name (line 52):
        # Getting the type of 'call_assignment_30357' (line 52)
        call_assignment_30357_30462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'call_assignment_30357')
        # Assigning a type to the variable 'temp_stmts' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'temp_stmts', call_assignment_30357_30462)
        
        # Assigning a Call to a Name (line 52):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_30356' (line 52)
        call_assignment_30356_30463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'call_assignment_30356', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_30464 = stypy_get_value_from_tuple(call_assignment_30356_30463, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_30358' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'call_assignment_30358', stypy_get_value_from_tuple_call_result_30464)
        
        # Assigning a Name to a Name (line 52):
        # Getting the type of 'call_assignment_30358' (line 52)
        call_assignment_30358_30465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'call_assignment_30358')
        # Assigning a type to the variable 'temp_value' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'temp_value', call_assignment_30358_30465)
        
        # Call to append(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'temp_stmts' (line 54)
        temp_stmts_30468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'temp_stmts', False)
        # Processing the call keyword arguments (line 54)
        kwargs_30469 = {}
        # Getting the type of 'assign_stmts' (line 54)
        assign_stmts_30466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'assign_stmts', False)
        # Obtaining the member 'append' of a type (line 54)
        append_30467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), assign_stmts_30466, 'append')
        # Calling append(args, kwargs) (line 54)
        append_call_result_30470 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), append_30467, *[temp_stmts_30468], **kwargs_30469)
        
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to create_Assign(...): (line 56)
        # Processing the call arguments (line 56)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 56)
        i_30473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 66), 'i', False)
        # Getting the type of 'target' (line 56)
        target_30474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 54), 'target', False)
        # Obtaining the member 'elts' of a type (line 56)
        elts_30475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 54), target_30474, 'elts')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___30476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 54), elts_30475, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_30477 = invoke(stypy.reporting.localization.Localization(__file__, 56, 54), getitem___30476, i_30473)
        
        # Getting the type of 'temp_value' (line 56)
        temp_value_30478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 70), 'temp_value', False)
        # Processing the call keyword arguments (line 56)
        kwargs_30479 = {}
        # Getting the type of 'core_language_copy' (line 56)
        core_language_copy_30471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'core_language_copy', False)
        # Obtaining the member 'create_Assign' of a type (line 56)
        create_Assign_30472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 21), core_language_copy_30471, 'create_Assign')
        # Calling create_Assign(args, kwargs) (line 56)
        create_Assign_call_result_30480 = invoke(stypy.reporting.localization.Localization(__file__, 56, 21), create_Assign_30472, *[subscript_call_result_30477, temp_value_30478], **kwargs_30479)
        
        # Assigning a type to the variable 'temp_stmts' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'temp_stmts', create_Assign_call_result_30480)
        
        # Call to append(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'temp_stmts' (line 57)
        temp_stmts_30483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'temp_stmts', False)
        # Processing the call keyword arguments (line 57)
        kwargs_30484 = {}
        # Getting the type of 'assign_stmts' (line 57)
        assign_stmts_30481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'assign_stmts', False)
        # Obtaining the member 'append' of a type (line 57)
        append_30482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), assign_stmts_30481, 'append')
        # Calling append(args, kwargs) (line 57)
        append_call_result_30485 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), append_30482, *[temp_stmts_30483], **kwargs_30484)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'multiple_value_call_assignment_handler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'multiple_value_call_assignment_handler' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_30486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30486)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'multiple_value_call_assignment_handler'
    return stypy_return_type_30486

# Assigning a type to the variable 'multiple_value_call_assignment_handler' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'multiple_value_call_assignment_handler', multiple_value_call_assignment_handler)

@norecursion
def multiple_value_assignment_handler(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'multiple_value_assignment_handler'
    module_type_store = module_type_store.open_function_context('multiple_value_assignment_handler', 60, 0, False)
    
    # Passed parameters checking function
    multiple_value_assignment_handler.stypy_localization = localization
    multiple_value_assignment_handler.stypy_type_of_self = None
    multiple_value_assignment_handler.stypy_type_store = module_type_store
    multiple_value_assignment_handler.stypy_function_name = 'multiple_value_assignment_handler'
    multiple_value_assignment_handler.stypy_param_names_list = ['target', 'value', 'assign_stmts', 'node', 'id_str']
    multiple_value_assignment_handler.stypy_varargs_param_name = None
    multiple_value_assignment_handler.stypy_kwargs_param_name = None
    multiple_value_assignment_handler.stypy_call_defaults = defaults
    multiple_value_assignment_handler.stypy_call_varargs = varargs
    multiple_value_assignment_handler.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'multiple_value_assignment_handler', ['target', 'value', 'assign_stmts', 'node', 'id_str'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'multiple_value_assignment_handler', localization, ['target', 'value', 'assign_stmts', 'node', 'id_str'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'multiple_value_assignment_handler(...)' code ##################

    str_30487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, (-1)), 'str', '\n    Code to handle assignments like a, b = c, d. This code is converted to:\n    a = c\n    b = d\n\n    Length of left and right part is cheched to make sure we are dealing with a valid assignment (an error is produced\n    otherwise)\n\n    :param target: tuple or list of variables\n    :param value:  tuple or list of variables\n    :param assign_stmts: statements holder list\n    :param node: current AST node\n    :param id_str: Type of assignment that we are processing (to create variables)\n    :return:\n    ')
    
    
    # Call to len(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'target' (line 76)
    target_30489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'target', False)
    # Obtaining the member 'elts' of a type (line 76)
    elts_30490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 11), target_30489, 'elts')
    # Processing the call keyword arguments (line 76)
    kwargs_30491 = {}
    # Getting the type of 'len' (line 76)
    len_30488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 7), 'len', False)
    # Calling len(args, kwargs) (line 76)
    len_call_result_30492 = invoke(stypy.reporting.localization.Localization(__file__, 76, 7), len_30488, *[elts_30490], **kwargs_30491)
    
    
    # Call to len(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'value' (line 76)
    value_30494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'value', False)
    # Obtaining the member 'elts' of a type (line 76)
    elts_30495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 31), value_30494, 'elts')
    # Processing the call keyword arguments (line 76)
    kwargs_30496 = {}
    # Getting the type of 'len' (line 76)
    len_30493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'len', False)
    # Calling len(args, kwargs) (line 76)
    len_call_result_30497 = invoke(stypy.reporting.localization.Localization(__file__, 76, 27), len_30493, *[elts_30495], **kwargs_30496)
    
    # Applying the binary operator '==' (line 76)
    result_eq_30498 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 7), '==', len_call_result_30492, len_call_result_30497)
    
    # Testing if the type of an if condition is none (line 76)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 4), result_eq_30498):
        
        # Call to TypeError(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Call to create_localization(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'node' (line 88)
        node_30574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 59), 'node', False)
        # Obtaining the member 'lineno' of a type (line 88)
        lineno_30575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 59), node_30574, 'lineno')
        # Getting the type of 'node' (line 88)
        node_30576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 72), 'node', False)
        # Obtaining the member 'col_offset' of a type (line 88)
        col_offset_30577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 72), node_30576, 'col_offset')
        # Processing the call keyword arguments (line 88)
        kwargs_30578 = {}
        # Getting the type of 'stypy_functions_copy' (line 88)
        stypy_functions_copy_30572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'stypy_functions_copy', False)
        # Obtaining the member 'create_localization' of a type (line 88)
        create_localization_30573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 18), stypy_functions_copy_30572, 'create_localization')
        # Calling create_localization(args, kwargs) (line 88)
        create_localization_call_result_30579 = invoke(stypy.reporting.localization.Localization(__file__, 88, 18), create_localization_30573, *[lineno_30575, col_offset_30577], **kwargs_30578)
        
        
        # Call to format(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'id_str' (line 90)
        id_str_30582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'id_str', False)
        # Processing the call keyword arguments (line 89)
        kwargs_30583 = {}
        str_30580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 18), 'str', 'Multi-value assignments with {0}s must have the same amount of elements on both assignment sides')
        # Obtaining the member 'format' of a type (line 89)
        format_30581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 18), str_30580, 'format')
        # Calling format(args, kwargs) (line 89)
        format_call_result_30584 = invoke(stypy.reporting.localization.Localization(__file__, 89, 18), format_30581, *[id_str_30582], **kwargs_30583)
        
        # Processing the call keyword arguments (line 88)
        kwargs_30585 = {}
        # Getting the type of 'TypeError' (line 88)
        TypeError_30571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 88)
        TypeError_call_result_30586 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), TypeError_30571, *[create_localization_call_result_30579, format_call_result_30584], **kwargs_30585)
        
    else:
        
        # Testing the type of an if condition (line 76)
        if_condition_30499 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 4), result_eq_30498)
        # Assigning a type to the variable 'if_condition_30499' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'if_condition_30499', if_condition_30499)
        # SSA begins for if statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 77):
        
        # Assigning a List to a Name (line 77):
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_30500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        
        # Assigning a type to the variable 'temp_var_names' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'temp_var_names', list_30500)
        
        
        # Call to range(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Call to len(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'value' (line 79)
        value_30503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 27), 'value', False)
        # Obtaining the member 'elts' of a type (line 79)
        elts_30504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 27), value_30503, 'elts')
        # Processing the call keyword arguments (line 79)
        kwargs_30505 = {}
        # Getting the type of 'len' (line 79)
        len_30502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'len', False)
        # Calling len(args, kwargs) (line 79)
        len_call_result_30506 = invoke(stypy.reporting.localization.Localization(__file__, 79, 23), len_30502, *[elts_30504], **kwargs_30505)
        
        # Processing the call keyword arguments (line 79)
        kwargs_30507 = {}
        # Getting the type of 'range' (line 79)
        range_30501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'range', False)
        # Calling range(args, kwargs) (line 79)
        range_call_result_30508 = invoke(stypy.reporting.localization.Localization(__file__, 79, 17), range_30501, *[len_call_result_30506], **kwargs_30507)
        
        # Assigning a type to the variable 'range_call_result_30508' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'range_call_result_30508', range_call_result_30508)
        # Testing if the for loop is going to be iterated (line 79)
        # Testing the type of a for loop iterable (line 79)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 79, 8), range_call_result_30508)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 79, 8), range_call_result_30508):
            # Getting the type of the for loop variable (line 79)
            for_loop_var_30509 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 79, 8), range_call_result_30508)
            # Assigning a type to the variable 'i' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'i', for_loop_var_30509)
            # SSA begins for a for statement (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Tuple (line 80):
            
            # Assigning a Call to a Name:
            
            # Call to create_temp_Assign(...): (line 80)
            # Processing the call arguments (line 80)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 80)
            i_30512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 88), 'i', False)
            # Getting the type of 'value' (line 80)
            value_30513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 77), 'value', False)
            # Obtaining the member 'elts' of a type (line 80)
            elts_30514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 77), value_30513, 'elts')
            # Obtaining the member '__getitem__' of a type (line 80)
            getitem___30515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 77), elts_30514, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 80)
            subscript_call_result_30516 = invoke(stypy.reporting.localization.Localization(__file__, 80, 77), getitem___30515, i_30512)
            
            # Getting the type of 'node' (line 80)
            node_30517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 92), 'node', False)
            # Obtaining the member 'lineno' of a type (line 80)
            lineno_30518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 92), node_30517, 'lineno')
            # Getting the type of 'node' (line 80)
            node_30519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 105), 'node', False)
            # Obtaining the member 'col_offset' of a type (line 80)
            col_offset_30520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 105), node_30519, 'col_offset')
            
            # Call to format(...): (line 81)
            # Processing the call arguments (line 81)
            # Getting the type of 'id_str' (line 81)
            id_str_30523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 96), 'id_str', False)
            # Processing the call keyword arguments (line 81)
            kwargs_30524 = {}
            str_30521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 72), 'str', '{0}_assignment')
            # Obtaining the member 'format' of a type (line 81)
            format_30522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 72), str_30521, 'format')
            # Calling format(args, kwargs) (line 81)
            format_call_result_30525 = invoke(stypy.reporting.localization.Localization(__file__, 81, 72), format_30522, *[id_str_30523], **kwargs_30524)
            
            # Processing the call keyword arguments (line 80)
            kwargs_30526 = {}
            # Getting the type of 'stypy_functions_copy' (line 80)
            stypy_functions_copy_30510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 37), 'stypy_functions_copy', False)
            # Obtaining the member 'create_temp_Assign' of a type (line 80)
            create_temp_Assign_30511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 37), stypy_functions_copy_30510, 'create_temp_Assign')
            # Calling create_temp_Assign(args, kwargs) (line 80)
            create_temp_Assign_call_result_30527 = invoke(stypy.reporting.localization.Localization(__file__, 80, 37), create_temp_Assign_30511, *[subscript_call_result_30516, lineno_30518, col_offset_30520, format_call_result_30525], **kwargs_30526)
            
            # Assigning a type to the variable 'call_assignment_30359' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'call_assignment_30359', create_temp_Assign_call_result_30527)
            
            # Assigning a Call to a Name (line 80):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_30359' (line 80)
            call_assignment_30359_30528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'call_assignment_30359', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_30529 = stypy_get_value_from_tuple(call_assignment_30359_30528, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_30360' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'call_assignment_30360', stypy_get_value_from_tuple_call_result_30529)
            
            # Assigning a Name to a Name (line 80):
            # Getting the type of 'call_assignment_30360' (line 80)
            call_assignment_30360_30530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'call_assignment_30360')
            # Assigning a type to the variable 'temp_stmts' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'temp_stmts', call_assignment_30360_30530)
            
            # Assigning a Call to a Name (line 80):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_30359' (line 80)
            call_assignment_30359_30531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'call_assignment_30359', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_30532 = stypy_get_value_from_tuple(call_assignment_30359_30531, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_30361' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'call_assignment_30361', stypy_get_value_from_tuple_call_result_30532)
            
            # Assigning a Name to a Name (line 80):
            # Getting the type of 'call_assignment_30361' (line 80)
            call_assignment_30361_30533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'call_assignment_30361')
            # Assigning a type to the variable 'temp_value' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), 'temp_value', call_assignment_30361_30533)
            
            # Call to append(...): (line 82)
            # Processing the call arguments (line 82)
            # Getting the type of 'temp_stmts' (line 82)
            temp_stmts_30536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 32), 'temp_stmts', False)
            # Processing the call keyword arguments (line 82)
            kwargs_30537 = {}
            # Getting the type of 'assign_stmts' (line 82)
            assign_stmts_30534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'assign_stmts', False)
            # Obtaining the member 'append' of a type (line 82)
            append_30535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), assign_stmts_30534, 'append')
            # Calling append(args, kwargs) (line 82)
            append_call_result_30538 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), append_30535, *[temp_stmts_30536], **kwargs_30537)
            
            
            # Call to append(...): (line 83)
            # Processing the call arguments (line 83)
            # Getting the type of 'temp_value' (line 83)
            temp_value_30541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 34), 'temp_value', False)
            # Processing the call keyword arguments (line 83)
            kwargs_30542 = {}
            # Getting the type of 'temp_var_names' (line 83)
            temp_var_names_30539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'temp_var_names', False)
            # Obtaining the member 'append' of a type (line 83)
            append_30540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), temp_var_names_30539, 'append')
            # Calling append(args, kwargs) (line 83)
            append_call_result_30543 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), append_30540, *[temp_value_30541], **kwargs_30542)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to range(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Call to len(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'target' (line 84)
        target_30546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'target', False)
        # Obtaining the member 'elts' of a type (line 84)
        elts_30547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 27), target_30546, 'elts')
        # Processing the call keyword arguments (line 84)
        kwargs_30548 = {}
        # Getting the type of 'len' (line 84)
        len_30545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'len', False)
        # Calling len(args, kwargs) (line 84)
        len_call_result_30549 = invoke(stypy.reporting.localization.Localization(__file__, 84, 23), len_30545, *[elts_30547], **kwargs_30548)
        
        # Processing the call keyword arguments (line 84)
        kwargs_30550 = {}
        # Getting the type of 'range' (line 84)
        range_30544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), 'range', False)
        # Calling range(args, kwargs) (line 84)
        range_call_result_30551 = invoke(stypy.reporting.localization.Localization(__file__, 84, 17), range_30544, *[len_call_result_30549], **kwargs_30550)
        
        # Assigning a type to the variable 'range_call_result_30551' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'range_call_result_30551', range_call_result_30551)
        # Testing if the for loop is going to be iterated (line 84)
        # Testing the type of a for loop iterable (line 84)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 84, 8), range_call_result_30551)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 84, 8), range_call_result_30551):
            # Getting the type of the for loop variable (line 84)
            for_loop_var_30552 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 84, 8), range_call_result_30551)
            # Assigning a type to the variable 'i' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'i', for_loop_var_30552)
            # SSA begins for a for statement (line 84)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 85):
            
            # Assigning a Call to a Name (line 85):
            
            # Call to create_Assign(...): (line 85)
            # Processing the call arguments (line 85)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 85)
            i_30555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 70), 'i', False)
            # Getting the type of 'target' (line 85)
            target_30556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 58), 'target', False)
            # Obtaining the member 'elts' of a type (line 85)
            elts_30557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 58), target_30556, 'elts')
            # Obtaining the member '__getitem__' of a type (line 85)
            getitem___30558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 58), elts_30557, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 85)
            subscript_call_result_30559 = invoke(stypy.reporting.localization.Localization(__file__, 85, 58), getitem___30558, i_30555)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 85)
            i_30560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 89), 'i', False)
            # Getting the type of 'temp_var_names' (line 85)
            temp_var_names_30561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 74), 'temp_var_names', False)
            # Obtaining the member '__getitem__' of a type (line 85)
            getitem___30562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 74), temp_var_names_30561, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 85)
            subscript_call_result_30563 = invoke(stypy.reporting.localization.Localization(__file__, 85, 74), getitem___30562, i_30560)
            
            # Processing the call keyword arguments (line 85)
            kwargs_30564 = {}
            # Getting the type of 'core_language_copy' (line 85)
            core_language_copy_30553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'core_language_copy', False)
            # Obtaining the member 'create_Assign' of a type (line 85)
            create_Assign_30554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 25), core_language_copy_30553, 'create_Assign')
            # Calling create_Assign(args, kwargs) (line 85)
            create_Assign_call_result_30565 = invoke(stypy.reporting.localization.Localization(__file__, 85, 25), create_Assign_30554, *[subscript_call_result_30559, subscript_call_result_30563], **kwargs_30564)
            
            # Assigning a type to the variable 'temp_stmts' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'temp_stmts', create_Assign_call_result_30565)
            
            # Call to append(...): (line 86)
            # Processing the call arguments (line 86)
            # Getting the type of 'temp_stmts' (line 86)
            temp_stmts_30568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 32), 'temp_stmts', False)
            # Processing the call keyword arguments (line 86)
            kwargs_30569 = {}
            # Getting the type of 'assign_stmts' (line 86)
            assign_stmts_30566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'assign_stmts', False)
            # Obtaining the member 'append' of a type (line 86)
            append_30567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), assign_stmts_30566, 'append')
            # Calling append(args, kwargs) (line 86)
            append_call_result_30570 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), append_30567, *[temp_stmts_30568], **kwargs_30569)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA branch for the else part of an if statement (line 76)
        module_type_store.open_ssa_branch('else')
        
        # Call to TypeError(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Call to create_localization(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'node' (line 88)
        node_30574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 59), 'node', False)
        # Obtaining the member 'lineno' of a type (line 88)
        lineno_30575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 59), node_30574, 'lineno')
        # Getting the type of 'node' (line 88)
        node_30576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 72), 'node', False)
        # Obtaining the member 'col_offset' of a type (line 88)
        col_offset_30577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 72), node_30576, 'col_offset')
        # Processing the call keyword arguments (line 88)
        kwargs_30578 = {}
        # Getting the type of 'stypy_functions_copy' (line 88)
        stypy_functions_copy_30572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'stypy_functions_copy', False)
        # Obtaining the member 'create_localization' of a type (line 88)
        create_localization_30573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 18), stypy_functions_copy_30572, 'create_localization')
        # Calling create_localization(args, kwargs) (line 88)
        create_localization_call_result_30579 = invoke(stypy.reporting.localization.Localization(__file__, 88, 18), create_localization_30573, *[lineno_30575, col_offset_30577], **kwargs_30578)
        
        
        # Call to format(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'id_str' (line 90)
        id_str_30582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'id_str', False)
        # Processing the call keyword arguments (line 89)
        kwargs_30583 = {}
        str_30580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 18), 'str', 'Multi-value assignments with {0}s must have the same amount of elements on both assignment sides')
        # Obtaining the member 'format' of a type (line 89)
        format_30581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 18), str_30580, 'format')
        # Calling format(args, kwargs) (line 89)
        format_call_result_30584 = invoke(stypy.reporting.localization.Localization(__file__, 89, 18), format_30581, *[id_str_30582], **kwargs_30583)
        
        # Processing the call keyword arguments (line 88)
        kwargs_30585 = {}
        # Getting the type of 'TypeError' (line 88)
        TypeError_30571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 88)
        TypeError_call_result_30586 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), TypeError_30571, *[create_localization_call_result_30579, format_call_result_30584], **kwargs_30585)
        
        # SSA join for if statement (line 76)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'multiple_value_assignment_handler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'multiple_value_assignment_handler' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_30587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30587)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'multiple_value_assignment_handler'
    return stypy_return_type_30587

# Assigning a type to the variable 'multiple_value_assignment_handler' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'multiple_value_assignment_handler', multiple_value_assignment_handler)

@norecursion
def single_assignment_handler(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'single_assignment_handler'
    module_type_store = module_type_store.open_function_context('single_assignment_handler', 93, 0, False)
    
    # Passed parameters checking function
    single_assignment_handler.stypy_localization = localization
    single_assignment_handler.stypy_type_of_self = None
    single_assignment_handler.stypy_type_store = module_type_store
    single_assignment_handler.stypy_function_name = 'single_assignment_handler'
    single_assignment_handler.stypy_param_names_list = ['target', 'value', 'assign_stmts', 'node', 'id_str']
    single_assignment_handler.stypy_varargs_param_name = None
    single_assignment_handler.stypy_kwargs_param_name = None
    single_assignment_handler.stypy_call_defaults = defaults
    single_assignment_handler.stypy_call_varargs = varargs
    single_assignment_handler.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'single_assignment_handler', ['target', 'value', 'assign_stmts', 'node', 'id_str'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'single_assignment_handler', localization, ['target', 'value', 'assign_stmts', 'node', 'id_str'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'single_assignment_handler(...)' code ##################

    str_30588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, (-1)), 'str', '\n    Handles single statements for hte visitor. No change is produced in the code\n    :param target: Variable\n    :param value: Value to assign\n    :param assign_stmts: statements holder list\n    :param node: current AST node\n    :param id_str: Type of assignment that we are processing (to create variables)\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 103):
    
    # Assigning a Call to a Name (line 103):
    
    # Call to create_Assign(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'target' (line 103)
    target_30591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 50), 'target', False)
    # Getting the type of 'value' (line 103)
    value_30592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 58), 'value', False)
    # Processing the call keyword arguments (line 103)
    kwargs_30593 = {}
    # Getting the type of 'core_language_copy' (line 103)
    core_language_copy_30589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 17), 'core_language_copy', False)
    # Obtaining the member 'create_Assign' of a type (line 103)
    create_Assign_30590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 17), core_language_copy_30589, 'create_Assign')
    # Calling create_Assign(args, kwargs) (line 103)
    create_Assign_call_result_30594 = invoke(stypy.reporting.localization.Localization(__file__, 103, 17), create_Assign_30590, *[target_30591, value_30592], **kwargs_30593)
    
    # Assigning a type to the variable 'temp_stmts' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'temp_stmts', create_Assign_call_result_30594)
    
    # Call to append(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'temp_stmts' (line 104)
    temp_stmts_30597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'temp_stmts', False)
    # Processing the call keyword arguments (line 104)
    kwargs_30598 = {}
    # Getting the type of 'assign_stmts' (line 104)
    assign_stmts_30595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'assign_stmts', False)
    # Obtaining the member 'append' of a type (line 104)
    append_30596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 4), assign_stmts_30595, 'append')
    # Calling append(args, kwargs) (line 104)
    append_call_result_30599 = invoke(stypy.reporting.localization.Localization(__file__, 104, 4), append_30596, *[temp_stmts_30597], **kwargs_30598)
    
    
    # ################# End of 'single_assignment_handler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'single_assignment_handler' in the type store
    # Getting the type of 'stypy_return_type' (line 93)
    stypy_return_type_30600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30600)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'single_assignment_handler'
    return stypy_return_type_30600

# Assigning a type to the variable 'single_assignment_handler' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'single_assignment_handler', single_assignment_handler)
# Declaration of the 'MultipleAssignmentsDesugaringVisitor' class
# Getting the type of 'ast' (line 107)
ast_30601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 43), 'ast')
# Obtaining the member 'NodeTransformer' of a type (line 107)
NodeTransformer_30602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 43), ast_30601, 'NodeTransformer')

class MultipleAssignmentsDesugaringVisitor(NodeTransformer_30602, ):
    
    # Assigning a Dict to a Name (line 111):

    @norecursion
    def visit_Assign(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Assign'
        module_type_store = module_type_store.open_function_context('visit_Assign', 135, 4, False)
        # Assigning a type to the variable 'self' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MultipleAssignmentsDesugaringVisitor.visit_Assign.__dict__.__setitem__('stypy_localization', localization)
        MultipleAssignmentsDesugaringVisitor.visit_Assign.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MultipleAssignmentsDesugaringVisitor.visit_Assign.__dict__.__setitem__('stypy_type_store', module_type_store)
        MultipleAssignmentsDesugaringVisitor.visit_Assign.__dict__.__setitem__('stypy_function_name', 'MultipleAssignmentsDesugaringVisitor.visit_Assign')
        MultipleAssignmentsDesugaringVisitor.visit_Assign.__dict__.__setitem__('stypy_param_names_list', ['node'])
        MultipleAssignmentsDesugaringVisitor.visit_Assign.__dict__.__setitem__('stypy_varargs_param_name', None)
        MultipleAssignmentsDesugaringVisitor.visit_Assign.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MultipleAssignmentsDesugaringVisitor.visit_Assign.__dict__.__setitem__('stypy_call_defaults', defaults)
        MultipleAssignmentsDesugaringVisitor.visit_Assign.__dict__.__setitem__('stypy_call_varargs', varargs)
        MultipleAssignmentsDesugaringVisitor.visit_Assign.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MultipleAssignmentsDesugaringVisitor.visit_Assign.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MultipleAssignmentsDesugaringVisitor.visit_Assign', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Assign', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Assign(...)' code ##################

        
        # Assigning a List to a Name (line 136):
        
        # Assigning a List to a Name (line 136):
        
        # Obtaining an instance of the builtin type 'list' (line 136)
        list_30603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        
        # Assigning a type to the variable 'assign_stmts' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'assign_stmts', list_30603)
        
        # Assigning a Attribute to a Name (line 137):
        
        # Assigning a Attribute to a Name (line 137):
        # Getting the type of 'node' (line 137)
        node_30604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'node')
        # Obtaining the member 'value' of a type (line 137)
        value_30605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), node_30604, 'value')
        # Assigning a type to the variable 'value' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'value', value_30605)
        
        # Assigning a Attribute to a Name (line 138):
        
        # Assigning a Attribute to a Name (line 138):
        # Getting the type of 'node' (line 138)
        node_30606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 27), 'node')
        # Obtaining the member 'targets' of a type (line 138)
        targets_30607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 27), node_30606, 'targets')
        # Assigning a type to the variable 'reversed_targets' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'reversed_targets', targets_30607)
        
        # Call to reverse(...): (line 139)
        # Processing the call keyword arguments (line 139)
        kwargs_30610 = {}
        # Getting the type of 'reversed_targets' (line 139)
        reversed_targets_30608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'reversed_targets', False)
        # Obtaining the member 'reverse' of a type (line 139)
        reverse_30609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), reversed_targets_30608, 'reverse')
        # Calling reverse(args, kwargs) (line 139)
        reverse_call_result_30611 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), reverse_30609, *[], **kwargs_30610)
        
        
        # Call to append(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Call to create_blank_line(...): (line 140)
        # Processing the call keyword arguments (line 140)
        kwargs_30616 = {}
        # Getting the type of 'stypy_functions_copy' (line 140)
        stypy_functions_copy_30614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 28), 'stypy_functions_copy', False)
        # Obtaining the member 'create_blank_line' of a type (line 140)
        create_blank_line_30615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 28), stypy_functions_copy_30614, 'create_blank_line')
        # Calling create_blank_line(args, kwargs) (line 140)
        create_blank_line_call_result_30617 = invoke(stypy.reporting.localization.Localization(__file__, 140, 28), create_blank_line_30615, *[], **kwargs_30616)
        
        # Processing the call keyword arguments (line 140)
        kwargs_30618 = {}
        # Getting the type of 'assign_stmts' (line 140)
        assign_stmts_30612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'assign_stmts', False)
        # Obtaining the member 'append' of a type (line 140)
        append_30613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), assign_stmts_30612, 'append')
        # Calling append(args, kwargs) (line 140)
        append_call_result_30619 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), append_30613, *[create_blank_line_call_result_30617], **kwargs_30618)
        
        
        
        # Call to len(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'reversed_targets' (line 141)
        reversed_targets_30621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'reversed_targets', False)
        # Processing the call keyword arguments (line 141)
        kwargs_30622 = {}
        # Getting the type of 'len' (line 141)
        len_30620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'len', False)
        # Calling len(args, kwargs) (line 141)
        len_call_result_30623 = invoke(stypy.reporting.localization.Localization(__file__, 141, 11), len_30620, *[reversed_targets_30621], **kwargs_30622)
        
        int_30624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 35), 'int')
        # Applying the binary operator '>' (line 141)
        result_gt_30625 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 11), '>', len_call_result_30623, int_30624)
        
        # Testing if the type of an if condition is none (line 141)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 141, 8), result_gt_30625):
            
            # Call to append(...): (line 145)
            # Processing the call arguments (line 145)
            
            # Call to create_src_comment(...): (line 145)
            # Processing the call arguments (line 145)
            
            # Call to format(...): (line 146)
            # Processing the call arguments (line 146)
            
            # Call to type(...): (line 146)
            # Processing the call arguments (line 146)
            
            # Obtaining the type of the subscript
            int_30650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 78), 'int')
            # Getting the type of 'reversed_targets' (line 146)
            reversed_targets_30651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 61), 'reversed_targets', False)
            # Obtaining the member '__getitem__' of a type (line 146)
            getitem___30652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 61), reversed_targets_30651, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 146)
            subscript_call_result_30653 = invoke(stypy.reporting.localization.Localization(__file__, 146, 61), getitem___30652, int_30650)
            
            # Processing the call keyword arguments (line 146)
            kwargs_30654 = {}
            # Getting the type of 'type' (line 146)
            type_30649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 56), 'type', False)
            # Calling type(args, kwargs) (line 146)
            type_call_result_30655 = invoke(stypy.reporting.localization.Localization(__file__, 146, 56), type_30649, *[subscript_call_result_30653], **kwargs_30654)
            
            # Obtaining the member '__name__' of a type (line 146)
            name___30656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 56), type_call_result_30655, '__name__')
            
            # Call to type(...): (line 147)
            # Processing the call arguments (line 147)
            # Getting the type of 'value' (line 147)
            value_30658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 61), 'value', False)
            # Processing the call keyword arguments (line 147)
            kwargs_30659 = {}
            # Getting the type of 'type' (line 147)
            type_30657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 56), 'type', False)
            # Calling type(args, kwargs) (line 147)
            type_call_result_30660 = invoke(stypy.reporting.localization.Localization(__file__, 147, 56), type_30657, *[value_30658], **kwargs_30659)
            
            # Obtaining the member '__name__' of a type (line 147)
            name___30661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 56), type_call_result_30660, '__name__')
            # Processing the call keyword arguments (line 146)
            kwargs_30662 = {}
            str_30647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 16), 'str', 'Assignment to a {0} from a {1}')
            # Obtaining the member 'format' of a type (line 146)
            format_30648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), str_30647, 'format')
            # Calling format(args, kwargs) (line 146)
            format_call_result_30663 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), format_30648, *[name___30656, name___30661], **kwargs_30662)
            
            # Processing the call keyword arguments (line 145)
            kwargs_30664 = {}
            # Getting the type of 'stypy_functions_copy' (line 145)
            stypy_functions_copy_30645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 32), 'stypy_functions_copy', False)
            # Obtaining the member 'create_src_comment' of a type (line 145)
            create_src_comment_30646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 32), stypy_functions_copy_30645, 'create_src_comment')
            # Calling create_src_comment(args, kwargs) (line 145)
            create_src_comment_call_result_30665 = invoke(stypy.reporting.localization.Localization(__file__, 145, 32), create_src_comment_30646, *[format_call_result_30663], **kwargs_30664)
            
            # Processing the call keyword arguments (line 145)
            kwargs_30666 = {}
            # Getting the type of 'assign_stmts' (line 145)
            assign_stmts_30643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'assign_stmts', False)
            # Obtaining the member 'append' of a type (line 145)
            append_30644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), assign_stmts_30643, 'append')
            # Calling append(args, kwargs) (line 145)
            append_call_result_30667 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), append_30644, *[create_src_comment_call_result_30665], **kwargs_30666)
            
        else:
            
            # Testing the type of an if condition (line 141)
            if_condition_30626 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 8), result_gt_30625)
            # Assigning a type to the variable 'if_condition_30626' (line 141)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'if_condition_30626', if_condition_30626)
            # SSA begins for if statement (line 141)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 142)
            # Processing the call arguments (line 142)
            
            # Call to create_src_comment(...): (line 143)
            # Processing the call arguments (line 143)
            
            # Call to format(...): (line 143)
            # Processing the call arguments (line 143)
            
            # Call to len(...): (line 143)
            # Processing the call arguments (line 143)
            # Getting the type of 'reversed_targets' (line 143)
            reversed_targets_30634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 105), 'reversed_targets', False)
            # Processing the call keyword arguments (line 143)
            kwargs_30635 = {}
            # Getting the type of 'len' (line 143)
            len_30633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 101), 'len', False)
            # Calling len(args, kwargs) (line 143)
            len_call_result_30636 = invoke(stypy.reporting.localization.Localization(__file__, 143, 101), len_30633, *[reversed_targets_30634], **kwargs_30635)
            
            # Processing the call keyword arguments (line 143)
            kwargs_30637 = {}
            str_30631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 56), 'str', 'Multiple assigment of {0} elements.')
            # Obtaining the member 'format' of a type (line 143)
            format_30632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 56), str_30631, 'format')
            # Calling format(args, kwargs) (line 143)
            format_call_result_30638 = invoke(stypy.reporting.localization.Localization(__file__, 143, 56), format_30632, *[len_call_result_30636], **kwargs_30637)
            
            # Processing the call keyword arguments (line 143)
            kwargs_30639 = {}
            # Getting the type of 'stypy_functions_copy' (line 143)
            stypy_functions_copy_30629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'stypy_functions_copy', False)
            # Obtaining the member 'create_src_comment' of a type (line 143)
            create_src_comment_30630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 16), stypy_functions_copy_30629, 'create_src_comment')
            # Calling create_src_comment(args, kwargs) (line 143)
            create_src_comment_call_result_30640 = invoke(stypy.reporting.localization.Localization(__file__, 143, 16), create_src_comment_30630, *[format_call_result_30638], **kwargs_30639)
            
            # Processing the call keyword arguments (line 142)
            kwargs_30641 = {}
            # Getting the type of 'assign_stmts' (line 142)
            assign_stmts_30627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'assign_stmts', False)
            # Obtaining the member 'append' of a type (line 142)
            append_30628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 12), assign_stmts_30627, 'append')
            # Calling append(args, kwargs) (line 142)
            append_call_result_30642 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), append_30628, *[create_src_comment_call_result_30640], **kwargs_30641)
            
            # SSA branch for the else part of an if statement (line 141)
            module_type_store.open_ssa_branch('else')
            
            # Call to append(...): (line 145)
            # Processing the call arguments (line 145)
            
            # Call to create_src_comment(...): (line 145)
            # Processing the call arguments (line 145)
            
            # Call to format(...): (line 146)
            # Processing the call arguments (line 146)
            
            # Call to type(...): (line 146)
            # Processing the call arguments (line 146)
            
            # Obtaining the type of the subscript
            int_30650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 78), 'int')
            # Getting the type of 'reversed_targets' (line 146)
            reversed_targets_30651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 61), 'reversed_targets', False)
            # Obtaining the member '__getitem__' of a type (line 146)
            getitem___30652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 61), reversed_targets_30651, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 146)
            subscript_call_result_30653 = invoke(stypy.reporting.localization.Localization(__file__, 146, 61), getitem___30652, int_30650)
            
            # Processing the call keyword arguments (line 146)
            kwargs_30654 = {}
            # Getting the type of 'type' (line 146)
            type_30649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 56), 'type', False)
            # Calling type(args, kwargs) (line 146)
            type_call_result_30655 = invoke(stypy.reporting.localization.Localization(__file__, 146, 56), type_30649, *[subscript_call_result_30653], **kwargs_30654)
            
            # Obtaining the member '__name__' of a type (line 146)
            name___30656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 56), type_call_result_30655, '__name__')
            
            # Call to type(...): (line 147)
            # Processing the call arguments (line 147)
            # Getting the type of 'value' (line 147)
            value_30658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 61), 'value', False)
            # Processing the call keyword arguments (line 147)
            kwargs_30659 = {}
            # Getting the type of 'type' (line 147)
            type_30657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 56), 'type', False)
            # Calling type(args, kwargs) (line 147)
            type_call_result_30660 = invoke(stypy.reporting.localization.Localization(__file__, 147, 56), type_30657, *[value_30658], **kwargs_30659)
            
            # Obtaining the member '__name__' of a type (line 147)
            name___30661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 56), type_call_result_30660, '__name__')
            # Processing the call keyword arguments (line 146)
            kwargs_30662 = {}
            str_30647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 16), 'str', 'Assignment to a {0} from a {1}')
            # Obtaining the member 'format' of a type (line 146)
            format_30648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), str_30647, 'format')
            # Calling format(args, kwargs) (line 146)
            format_call_result_30663 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), format_30648, *[name___30656, name___30661], **kwargs_30662)
            
            # Processing the call keyword arguments (line 145)
            kwargs_30664 = {}
            # Getting the type of 'stypy_functions_copy' (line 145)
            stypy_functions_copy_30645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 32), 'stypy_functions_copy', False)
            # Obtaining the member 'create_src_comment' of a type (line 145)
            create_src_comment_30646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 32), stypy_functions_copy_30645, 'create_src_comment')
            # Calling create_src_comment(args, kwargs) (line 145)
            create_src_comment_call_result_30665 = invoke(stypy.reporting.localization.Localization(__file__, 145, 32), create_src_comment_30646, *[format_call_result_30663], **kwargs_30664)
            
            # Processing the call keyword arguments (line 145)
            kwargs_30666 = {}
            # Getting the type of 'assign_stmts' (line 145)
            assign_stmts_30643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'assign_stmts', False)
            # Obtaining the member 'append' of a type (line 145)
            append_30644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), assign_stmts_30643, 'append')
            # Calling append(args, kwargs) (line 145)
            append_call_result_30667 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), append_30644, *[create_src_comment_call_result_30665], **kwargs_30666)
            
            # SSA join for if statement (line 141)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to range(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Call to len(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'reversed_targets' (line 149)
        reversed_targets_30670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 36), 'reversed_targets', False)
        # Processing the call keyword arguments (line 149)
        kwargs_30671 = {}
        # Getting the type of 'len' (line 149)
        len_30669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 32), 'len', False)
        # Calling len(args, kwargs) (line 149)
        len_call_result_30672 = invoke(stypy.reporting.localization.Localization(__file__, 149, 32), len_30669, *[reversed_targets_30670], **kwargs_30671)
        
        # Processing the call keyword arguments (line 149)
        kwargs_30673 = {}
        # Getting the type of 'range' (line 149)
        range_30668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 26), 'range', False)
        # Calling range(args, kwargs) (line 149)
        range_call_result_30674 = invoke(stypy.reporting.localization.Localization(__file__, 149, 26), range_30668, *[len_call_result_30672], **kwargs_30673)
        
        # Assigning a type to the variable 'range_call_result_30674' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'range_call_result_30674', range_call_result_30674)
        # Testing if the for loop is going to be iterated (line 149)
        # Testing the type of a for loop iterable (line 149)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 149, 8), range_call_result_30674)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 149, 8), range_call_result_30674):
            # Getting the type of the for loop variable (line 149)
            for_loop_var_30675 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 149, 8), range_call_result_30674)
            # Assigning a type to the variable 'assign_num' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'assign_num', for_loop_var_30675)
            # SSA begins for a for statement (line 149)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 150):
            
            # Assigning a Subscript to a Name (line 150):
            
            # Obtaining the type of the subscript
            # Getting the type of 'assign_num' (line 150)
            assign_num_30676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 38), 'assign_num')
            # Getting the type of 'reversed_targets' (line 150)
            reversed_targets_30677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'reversed_targets')
            # Obtaining the member '__getitem__' of a type (line 150)
            getitem___30678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 21), reversed_targets_30677, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 150)
            subscript_call_result_30679 = invoke(stypy.reporting.localization.Localization(__file__, 150, 21), getitem___30678, assign_num_30676)
            
            # Assigning a type to the variable 'target' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'target', subscript_call_result_30679)
            
            # Getting the type of 'self' (line 152)
            self_30680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 38), 'self')
            # Obtaining the member '__assignment_handlers' of a type (line 152)
            assignment_handlers_30681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 38), self_30680, '__assignment_handlers')
            # Assigning a type to the variable 'assignment_handlers_30681' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'assignment_handlers_30681', assignment_handlers_30681)
            # Testing if the for loop is going to be iterated (line 152)
            # Testing the type of a for loop iterable (line 152)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 152, 12), assignment_handlers_30681)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 152, 12), assignment_handlers_30681):
                # Getting the type of the for loop variable (line 152)
                for_loop_var_30682 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 152, 12), assignment_handlers_30681)
                # Assigning a type to the variable 'handler_func_guard' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'handler_func_guard', for_loop_var_30682)
                # SSA begins for a for statement (line 152)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to handler_func_guard(...): (line 153)
                # Processing the call arguments (line 153)
                # Getting the type of 'target' (line 153)
                target_30684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 38), 'target', False)
                # Getting the type of 'value' (line 153)
                value_30685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 46), 'value', False)
                # Processing the call keyword arguments (line 153)
                kwargs_30686 = {}
                # Getting the type of 'handler_func_guard' (line 153)
                handler_func_guard_30683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 19), 'handler_func_guard', False)
                # Calling handler_func_guard(args, kwargs) (line 153)
                handler_func_guard_call_result_30687 = invoke(stypy.reporting.localization.Localization(__file__, 153, 19), handler_func_guard_30683, *[target_30684, value_30685], **kwargs_30686)
                
                # Testing if the type of an if condition is none (line 153)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 153, 16), handler_func_guard_call_result_30687):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 153)
                    if_condition_30688 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 16), handler_func_guard_call_result_30687)
                    # Assigning a type to the variable 'if_condition_30688' (line 153)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'if_condition_30688', if_condition_30688)
                    # SSA begins for if statement (line 153)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Subscript to a Tuple (line 154):
                    
                    # Assigning a Subscript to a Name (line 154):
                    
                    # Obtaining the type of the subscript
                    int_30689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 20), 'int')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'handler_func_guard' (line 154)
                    handler_func_guard_30690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 70), 'handler_func_guard')
                    # Getting the type of 'self' (line 154)
                    self_30691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 43), 'self')
                    # Obtaining the member '__assignment_handlers' of a type (line 154)
                    assignment_handlers_30692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 43), self_30691, '__assignment_handlers')
                    # Obtaining the member '__getitem__' of a type (line 154)
                    getitem___30693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 43), assignment_handlers_30692, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
                    subscript_call_result_30694 = invoke(stypy.reporting.localization.Localization(__file__, 154, 43), getitem___30693, handler_func_guard_30690)
                    
                    # Obtaining the member '__getitem__' of a type (line 154)
                    getitem___30695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 20), subscript_call_result_30694, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
                    subscript_call_result_30696 = invoke(stypy.reporting.localization.Localization(__file__, 154, 20), getitem___30695, int_30689)
                    
                    # Assigning a type to the variable 'tuple_var_assignment_30362' (line 154)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'tuple_var_assignment_30362', subscript_call_result_30696)
                    
                    # Assigning a Subscript to a Name (line 154):
                    
                    # Obtaining the type of the subscript
                    int_30697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 20), 'int')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'handler_func_guard' (line 154)
                    handler_func_guard_30698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 70), 'handler_func_guard')
                    # Getting the type of 'self' (line 154)
                    self_30699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 43), 'self')
                    # Obtaining the member '__assignment_handlers' of a type (line 154)
                    assignment_handlers_30700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 43), self_30699, '__assignment_handlers')
                    # Obtaining the member '__getitem__' of a type (line 154)
                    getitem___30701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 43), assignment_handlers_30700, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
                    subscript_call_result_30702 = invoke(stypy.reporting.localization.Localization(__file__, 154, 43), getitem___30701, handler_func_guard_30698)
                    
                    # Obtaining the member '__getitem__' of a type (line 154)
                    getitem___30703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 20), subscript_call_result_30702, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
                    subscript_call_result_30704 = invoke(stypy.reporting.localization.Localization(__file__, 154, 20), getitem___30703, int_30697)
                    
                    # Assigning a type to the variable 'tuple_var_assignment_30363' (line 154)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'tuple_var_assignment_30363', subscript_call_result_30704)
                    
                    # Assigning a Name to a Name (line 154):
                    # Getting the type of 'tuple_var_assignment_30362' (line 154)
                    tuple_var_assignment_30362_30705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'tuple_var_assignment_30362')
                    # Assigning a type to the variable 'id_str' (line 154)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'id_str', tuple_var_assignment_30362_30705)
                    
                    # Assigning a Name to a Name (line 154):
                    # Getting the type of 'tuple_var_assignment_30363' (line 154)
                    tuple_var_assignment_30363_30706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'tuple_var_assignment_30363')
                    # Assigning a type to the variable 'handler_func' (line 154)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 28), 'handler_func', tuple_var_assignment_30363_30706)
                    
                    # Call to handler_func(...): (line 155)
                    # Processing the call arguments (line 155)
                    # Getting the type of 'target' (line 155)
                    target_30708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 33), 'target', False)
                    # Getting the type of 'value' (line 155)
                    value_30709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 41), 'value', False)
                    # Getting the type of 'assign_stmts' (line 155)
                    assign_stmts_30710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 48), 'assign_stmts', False)
                    # Getting the type of 'node' (line 155)
                    node_30711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 62), 'node', False)
                    # Getting the type of 'id_str' (line 155)
                    id_str_30712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 68), 'id_str', False)
                    # Processing the call keyword arguments (line 155)
                    kwargs_30713 = {}
                    # Getting the type of 'handler_func' (line 155)
                    handler_func_30707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'handler_func', False)
                    # Calling handler_func(args, kwargs) (line 155)
                    handler_func_call_result_30714 = invoke(stypy.reporting.localization.Localization(__file__, 155, 20), handler_func_30707, *[target_30708, value_30709, assign_stmts_30710, node_30711, id_str_30712], **kwargs_30713)
                    
                    
                    # Assigning a Call to a Name (line 156):
                    
                    # Assigning a Call to a Name (line 156):
                    
                    # Call to flatten_lists(...): (line 156)
                    # Processing the call arguments (line 156)
                    # Getting the type of 'assign_stmts' (line 156)
                    assign_stmts_30717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 70), 'assign_stmts', False)
                    # Processing the call keyword arguments (line 156)
                    kwargs_30718 = {}
                    # Getting the type of 'stypy_functions_copy' (line 156)
                    stypy_functions_copy_30715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 35), 'stypy_functions_copy', False)
                    # Obtaining the member 'flatten_lists' of a type (line 156)
                    flatten_lists_30716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 35), stypy_functions_copy_30715, 'flatten_lists')
                    # Calling flatten_lists(args, kwargs) (line 156)
                    flatten_lists_call_result_30719 = invoke(stypy.reporting.localization.Localization(__file__, 156, 35), flatten_lists_30716, *[assign_stmts_30717], **kwargs_30718)
                    
                    # Assigning a type to the variable 'assign_stmts' (line 156)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 20), 'assign_stmts', flatten_lists_call_result_30719)
                    
                    # Assigning a Name to a Name (line 157):
                    
                    # Assigning a Name to a Name (line 157):
                    # Getting the type of 'target' (line 157)
                    target_30720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 28), 'target')
                    # Assigning a type to the variable 'value' (line 157)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'value', target_30720)
                    # SSA join for if statement (line 153)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'assign_stmts' (line 160)
        assign_stmts_30722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'assign_stmts', False)
        # Processing the call keyword arguments (line 160)
        kwargs_30723 = {}
        # Getting the type of 'len' (line 160)
        len_30721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'len', False)
        # Calling len(args, kwargs) (line 160)
        len_call_result_30724 = invoke(stypy.reporting.localization.Localization(__file__, 160, 11), len_30721, *[assign_stmts_30722], **kwargs_30723)
        
        int_30725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 31), 'int')
        # Applying the binary operator '>' (line 160)
        result_gt_30726 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 11), '>', len_call_result_30724, int_30725)
        
        # Testing if the type of an if condition is none (line 160)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 160, 8), result_gt_30726):
            pass
        else:
            
            # Testing the type of an if condition (line 160)
            if_condition_30727 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 8), result_gt_30726)
            # Assigning a type to the variable 'if_condition_30727' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'if_condition_30727', if_condition_30727)
            # SSA begins for if statement (line 160)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'assign_stmts' (line 161)
            assign_stmts_30728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 19), 'assign_stmts')
            # Assigning a type to the variable 'stypy_return_type' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'stypy_return_type', assign_stmts_30728)
            # SSA join for if statement (line 160)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'node' (line 162)
        node_30729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'stypy_return_type', node_30729)
        
        # ################# End of 'visit_Assign(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Assign' in the type store
        # Getting the type of 'stypy_return_type' (line 135)
        stypy_return_type_30730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_30730)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Assign'
        return stypy_return_type_30730


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 107, 0, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MultipleAssignmentsDesugaringVisitor.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'MultipleAssignmentsDesugaringVisitor' (line 107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'MultipleAssignmentsDesugaringVisitor', MultipleAssignmentsDesugaringVisitor)

# Assigning a Dict to a Name (line 111):

# Obtaining an instance of the builtin type 'dict' (line 111)
dict_30731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 28), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 111)
# Adding element type (key, value) (line 111)

@norecursion
def _stypy_temp_lambda_44(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_44'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_44', 112, 9, True)
    # Passed parameters checking function
    _stypy_temp_lambda_44.stypy_localization = localization
    _stypy_temp_lambda_44.stypy_type_of_self = None
    _stypy_temp_lambda_44.stypy_type_store = module_type_store
    _stypy_temp_lambda_44.stypy_function_name = '_stypy_temp_lambda_44'
    _stypy_temp_lambda_44.stypy_param_names_list = ['target', 'value']
    _stypy_temp_lambda_44.stypy_varargs_param_name = None
    _stypy_temp_lambda_44.stypy_kwargs_param_name = None
    _stypy_temp_lambda_44.stypy_call_defaults = defaults
    _stypy_temp_lambda_44.stypy_call_varargs = varargs
    _stypy_temp_lambda_44.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_44', ['target', 'value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_44', ['target', 'value'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'target' (line 112)
    target_30733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 42), 'target', False)
    # Getting the type of 'ast' (line 112)
    ast_30734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 50), 'ast', False)
    # Obtaining the member 'Tuple' of a type (line 112)
    Tuple_30735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 50), ast_30734, 'Tuple')
    # Processing the call keyword arguments (line 112)
    kwargs_30736 = {}
    # Getting the type of 'isinstance' (line 112)
    isinstance_30732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 112)
    isinstance_call_result_30737 = invoke(stypy.reporting.localization.Localization(__file__, 112, 31), isinstance_30732, *[target_30733, Tuple_30735], **kwargs_30736)
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'value' (line 112)
    value_30739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 77), 'value', False)
    # Getting the type of 'ast' (line 112)
    ast_30740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 84), 'ast', False)
    # Obtaining the member 'Tuple' of a type (line 112)
    Tuple_30741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 84), ast_30740, 'Tuple')
    # Processing the call keyword arguments (line 112)
    kwargs_30742 = {}
    # Getting the type of 'isinstance' (line 112)
    isinstance_30738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 66), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 112)
    isinstance_call_result_30743 = invoke(stypy.reporting.localization.Localization(__file__, 112, 66), isinstance_30738, *[value_30739, Tuple_30741], **kwargs_30742)
    
    
    # Call to isinstance(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'value' (line 113)
    value_30745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 77), 'value', False)
    # Getting the type of 'ast' (line 113)
    ast_30746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 84), 'ast', False)
    # Obtaining the member 'List' of a type (line 113)
    List_30747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 84), ast_30746, 'List')
    # Processing the call keyword arguments (line 113)
    kwargs_30748 = {}
    # Getting the type of 'isinstance' (line 113)
    isinstance_30744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 66), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 113)
    isinstance_call_result_30749 = invoke(stypy.reporting.localization.Localization(__file__, 113, 66), isinstance_30744, *[value_30745, List_30747], **kwargs_30748)
    
    # Applying the binary operator 'or' (line 112)
    result_or_keyword_30750 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 66), 'or', isinstance_call_result_30743, isinstance_call_result_30749)
    
    # Applying the binary operator 'and' (line 112)
    result_and_keyword_30751 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 31), 'and', isinstance_call_result_30737, result_or_keyword_30750)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 9), 'stypy_return_type', result_and_keyword_30751)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_44' in the type store
    # Getting the type of 'stypy_return_type' (line 112)
    stypy_return_type_30752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 9), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30752)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_44'
    return stypy_return_type_30752

# Assigning a type to the variable '_stypy_temp_lambda_44' (line 112)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 9), '_stypy_temp_lambda_44', _stypy_temp_lambda_44)
# Getting the type of '_stypy_temp_lambda_44' (line 112)
_stypy_temp_lambda_44_30753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 9), '_stypy_temp_lambda_44')

# Obtaining an instance of the builtin type 'tuple' (line 114)
tuple_30754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 114)
# Adding element type (line 114)
str_30755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'str', 'tuple')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 12), tuple_30754, str_30755)
# Adding element type (line 114)
# Getting the type of 'multiple_value_assignment_handler' (line 114)
multiple_value_assignment_handler_30756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'multiple_value_assignment_handler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 12), tuple_30754, multiple_value_assignment_handler_30756)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 28), dict_30731, (_stypy_temp_lambda_44_30753, tuple_30754))
# Adding element type (key, value) (line 111)

@norecursion
def _stypy_temp_lambda_45(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_45'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_45', 116, 9, True)
    # Passed parameters checking function
    _stypy_temp_lambda_45.stypy_localization = localization
    _stypy_temp_lambda_45.stypy_type_of_self = None
    _stypy_temp_lambda_45.stypy_type_store = module_type_store
    _stypy_temp_lambda_45.stypy_function_name = '_stypy_temp_lambda_45'
    _stypy_temp_lambda_45.stypy_param_names_list = ['target', 'value']
    _stypy_temp_lambda_45.stypy_varargs_param_name = None
    _stypy_temp_lambda_45.stypy_kwargs_param_name = None
    _stypy_temp_lambda_45.stypy_call_defaults = defaults
    _stypy_temp_lambda_45.stypy_call_varargs = varargs
    _stypy_temp_lambda_45.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_45', ['target', 'value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_45', ['target', 'value'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'target' (line 116)
    target_30758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 42), 'target', False)
    # Getting the type of 'ast' (line 116)
    ast_30759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 50), 'ast', False)
    # Obtaining the member 'List' of a type (line 116)
    List_30760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 50), ast_30759, 'List')
    # Processing the call keyword arguments (line 116)
    kwargs_30761 = {}
    # Getting the type of 'isinstance' (line 116)
    isinstance_30757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 116)
    isinstance_call_result_30762 = invoke(stypy.reporting.localization.Localization(__file__, 116, 31), isinstance_30757, *[target_30758, List_30760], **kwargs_30761)
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'value' (line 116)
    value_30764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 76), 'value', False)
    # Getting the type of 'ast' (line 116)
    ast_30765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 83), 'ast', False)
    # Obtaining the member 'Tuple' of a type (line 116)
    Tuple_30766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 83), ast_30765, 'Tuple')
    # Processing the call keyword arguments (line 116)
    kwargs_30767 = {}
    # Getting the type of 'isinstance' (line 116)
    isinstance_30763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 65), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 116)
    isinstance_call_result_30768 = invoke(stypy.reporting.localization.Localization(__file__, 116, 65), isinstance_30763, *[value_30764, Tuple_30766], **kwargs_30767)
    
    
    # Call to isinstance(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'value' (line 117)
    value_30770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 76), 'value', False)
    # Getting the type of 'ast' (line 117)
    ast_30771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 83), 'ast', False)
    # Obtaining the member 'List' of a type (line 117)
    List_30772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 83), ast_30771, 'List')
    # Processing the call keyword arguments (line 117)
    kwargs_30773 = {}
    # Getting the type of 'isinstance' (line 117)
    isinstance_30769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 65), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 117)
    isinstance_call_result_30774 = invoke(stypy.reporting.localization.Localization(__file__, 117, 65), isinstance_30769, *[value_30770, List_30772], **kwargs_30773)
    
    # Applying the binary operator 'or' (line 116)
    result_or_keyword_30775 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 65), 'or', isinstance_call_result_30768, isinstance_call_result_30774)
    
    # Applying the binary operator 'and' (line 116)
    result_and_keyword_30776 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 31), 'and', isinstance_call_result_30762, result_or_keyword_30775)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), 'stypy_return_type', result_and_keyword_30776)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_45' in the type store
    # Getting the type of 'stypy_return_type' (line 116)
    stypy_return_type_30777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30777)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_45'
    return stypy_return_type_30777

# Assigning a type to the variable '_stypy_temp_lambda_45' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), '_stypy_temp_lambda_45', _stypy_temp_lambda_45)
# Getting the type of '_stypy_temp_lambda_45' (line 116)
_stypy_temp_lambda_45_30778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), '_stypy_temp_lambda_45')

# Obtaining an instance of the builtin type 'tuple' (line 118)
tuple_30779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 118)
# Adding element type (line 118)
str_30780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 12), 'str', 'list')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 12), tuple_30779, str_30780)
# Adding element type (line 118)
# Getting the type of 'multiple_value_assignment_handler' (line 118)
multiple_value_assignment_handler_30781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'multiple_value_assignment_handler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 12), tuple_30779, multiple_value_assignment_handler_30781)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 28), dict_30731, (_stypy_temp_lambda_45_30778, tuple_30779))
# Adding element type (key, value) (line 111)

@norecursion
def _stypy_temp_lambda_46(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_46'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_46', 120, 9, True)
    # Passed parameters checking function
    _stypy_temp_lambda_46.stypy_localization = localization
    _stypy_temp_lambda_46.stypy_type_of_self = None
    _stypy_temp_lambda_46.stypy_type_store = module_type_store
    _stypy_temp_lambda_46.stypy_function_name = '_stypy_temp_lambda_46'
    _stypy_temp_lambda_46.stypy_param_names_list = ['target', 'value']
    _stypy_temp_lambda_46.stypy_varargs_param_name = None
    _stypy_temp_lambda_46.stypy_kwargs_param_name = None
    _stypy_temp_lambda_46.stypy_call_defaults = defaults
    _stypy_temp_lambda_46.stypy_call_varargs = varargs
    _stypy_temp_lambda_46.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_46', ['target', 'value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_46', ['target', 'value'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'target' (line 120)
    target_30783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 43), 'target', False)
    # Getting the type of 'ast' (line 120)
    ast_30784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 51), 'ast', False)
    # Obtaining the member 'List' of a type (line 120)
    List_30785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 51), ast_30784, 'List')
    # Processing the call keyword arguments (line 120)
    kwargs_30786 = {}
    # Getting the type of 'isinstance' (line 120)
    isinstance_30782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 32), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 120)
    isinstance_call_result_30787 = invoke(stypy.reporting.localization.Localization(__file__, 120, 32), isinstance_30782, *[target_30783, List_30785], **kwargs_30786)
    
    
    # Call to isinstance(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'target' (line 120)
    target_30789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 75), 'target', False)
    # Getting the type of 'ast' (line 120)
    ast_30790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 83), 'ast', False)
    # Obtaining the member 'Tuple' of a type (line 120)
    Tuple_30791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 83), ast_30790, 'Tuple')
    # Processing the call keyword arguments (line 120)
    kwargs_30792 = {}
    # Getting the type of 'isinstance' (line 120)
    isinstance_30788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 64), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 120)
    isinstance_call_result_30793 = invoke(stypy.reporting.localization.Localization(__file__, 120, 64), isinstance_30788, *[target_30789, Tuple_30791], **kwargs_30792)
    
    # Applying the binary operator 'or' (line 120)
    result_or_keyword_30794 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 32), 'or', isinstance_call_result_30787, isinstance_call_result_30793)
    
    
    # Call to isinstance(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'value' (line 121)
    value_30796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'value', False)
    # Getting the type of 'ast' (line 121)
    ast_30797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 30), 'ast', False)
    # Obtaining the member 'Call' of a type (line 121)
    Call_30798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 30), ast_30797, 'Call')
    # Processing the call keyword arguments (line 121)
    kwargs_30799 = {}
    # Getting the type of 'isinstance' (line 121)
    isinstance_30795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 121)
    isinstance_call_result_30800 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), isinstance_30795, *[value_30796, Call_30798], **kwargs_30799)
    
    # Applying the binary operator 'and' (line 120)
    result_and_keyword_30801 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 31), 'and', result_or_keyword_30794, isinstance_call_result_30800)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 9), 'stypy_return_type', result_and_keyword_30801)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_46' in the type store
    # Getting the type of 'stypy_return_type' (line 120)
    stypy_return_type_30802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 9), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30802)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_46'
    return stypy_return_type_30802

# Assigning a type to the variable '_stypy_temp_lambda_46' (line 120)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 9), '_stypy_temp_lambda_46', _stypy_temp_lambda_46)
# Getting the type of '_stypy_temp_lambda_46' (line 120)
_stypy_temp_lambda_46_30803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 9), '_stypy_temp_lambda_46')

# Obtaining an instance of the builtin type 'tuple' (line 121)
tuple_30804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 44), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 121)
# Adding element type (line 121)
str_30805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 44), 'str', 'call')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 44), tuple_30804, str_30805)
# Adding element type (line 121)
# Getting the type of 'multiple_value_call_assignment_handler' (line 121)
multiple_value_call_assignment_handler_30806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 52), 'multiple_value_call_assignment_handler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 44), tuple_30804, multiple_value_call_assignment_handler_30806)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 28), dict_30731, (_stypy_temp_lambda_46_30803, tuple_30804))
# Adding element type (key, value) (line 111)

@norecursion
def _stypy_temp_lambda_47(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_47'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_47', 123, 8, True)
    # Passed parameters checking function
    _stypy_temp_lambda_47.stypy_localization = localization
    _stypy_temp_lambda_47.stypy_type_of_self = None
    _stypy_temp_lambda_47.stypy_type_store = module_type_store
    _stypy_temp_lambda_47.stypy_function_name = '_stypy_temp_lambda_47'
    _stypy_temp_lambda_47.stypy_param_names_list = ['target', 'value']
    _stypy_temp_lambda_47.stypy_varargs_param_name = None
    _stypy_temp_lambda_47.stypy_kwargs_param_name = None
    _stypy_temp_lambda_47.stypy_call_defaults = defaults
    _stypy_temp_lambda_47.stypy_call_varargs = varargs
    _stypy_temp_lambda_47.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_47', ['target', 'value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_47', ['target', 'value'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to isinstance(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'target' (line 123)
    target_30808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 41), 'target', False)
    # Getting the type of 'ast' (line 123)
    ast_30809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 49), 'ast', False)
    # Obtaining the member 'Name' of a type (line 123)
    Name_30810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 49), ast_30809, 'Name')
    # Processing the call keyword arguments (line 123)
    kwargs_30811 = {}
    # Getting the type of 'isinstance' (line 123)
    isinstance_30807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 30), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 123)
    isinstance_call_result_30812 = invoke(stypy.reporting.localization.Localization(__file__, 123, 30), isinstance_30807, *[target_30808, Name_30810], **kwargs_30811)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type', isinstance_call_result_30812)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_47' in the type store
    # Getting the type of 'stypy_return_type' (line 123)
    stypy_return_type_30813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30813)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_47'
    return stypy_return_type_30813

# Assigning a type to the variable '_stypy_temp_lambda_47' (line 123)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), '_stypy_temp_lambda_47', _stypy_temp_lambda_47)
# Getting the type of '_stypy_temp_lambda_47' (line 123)
_stypy_temp_lambda_47_30814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), '_stypy_temp_lambda_47')

# Obtaining an instance of the builtin type 'tuple' (line 124)
tuple_30815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 124)
# Adding element type (line 124)
str_30816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 13), 'str', 'assignment')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 13), tuple_30815, str_30816)
# Adding element type (line 124)
# Getting the type of 'single_assignment_handler' (line 124)
single_assignment_handler_30817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'single_assignment_handler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 13), tuple_30815, single_assignment_handler_30817)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 28), dict_30731, (_stypy_temp_lambda_47_30814, tuple_30815))
# Adding element type (key, value) (line 111)

@norecursion
def _stypy_temp_lambda_48(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_48'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_48', 126, 8, True)
    # Passed parameters checking function
    _stypy_temp_lambda_48.stypy_localization = localization
    _stypy_temp_lambda_48.stypy_type_of_self = None
    _stypy_temp_lambda_48.stypy_type_store = module_type_store
    _stypy_temp_lambda_48.stypy_function_name = '_stypy_temp_lambda_48'
    _stypy_temp_lambda_48.stypy_param_names_list = ['target', 'value']
    _stypy_temp_lambda_48.stypy_varargs_param_name = None
    _stypy_temp_lambda_48.stypy_kwargs_param_name = None
    _stypy_temp_lambda_48.stypy_call_defaults = defaults
    _stypy_temp_lambda_48.stypy_call_varargs = varargs
    _stypy_temp_lambda_48.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_48', ['target', 'value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_48', ['target', 'value'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to isinstance(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'target' (line 126)
    target_30819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 41), 'target', False)
    # Getting the type of 'ast' (line 126)
    ast_30820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 49), 'ast', False)
    # Obtaining the member 'Subscript' of a type (line 126)
    Subscript_30821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 49), ast_30820, 'Subscript')
    # Processing the call keyword arguments (line 126)
    kwargs_30822 = {}
    # Getting the type of 'isinstance' (line 126)
    isinstance_30818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 30), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 126)
    isinstance_call_result_30823 = invoke(stypy.reporting.localization.Localization(__file__, 126, 30), isinstance_30818, *[target_30819, Subscript_30821], **kwargs_30822)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type', isinstance_call_result_30823)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_48' in the type store
    # Getting the type of 'stypy_return_type' (line 126)
    stypy_return_type_30824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30824)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_48'
    return stypy_return_type_30824

# Assigning a type to the variable '_stypy_temp_lambda_48' (line 126)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), '_stypy_temp_lambda_48', _stypy_temp_lambda_48)
# Getting the type of '_stypy_temp_lambda_48' (line 126)
_stypy_temp_lambda_48_30825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), '_stypy_temp_lambda_48')

# Obtaining an instance of the builtin type 'tuple' (line 127)
tuple_30826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 127)
# Adding element type (line 127)
str_30827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 13), 'str', 'assignment')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 13), tuple_30826, str_30827)
# Adding element type (line 127)
# Getting the type of 'single_assignment_handler' (line 127)
single_assignment_handler_30828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 27), 'single_assignment_handler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 13), tuple_30826, single_assignment_handler_30828)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 28), dict_30731, (_stypy_temp_lambda_48_30825, tuple_30826))
# Adding element type (key, value) (line 111)

@norecursion
def _stypy_temp_lambda_49(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_49'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_49', 129, 8, True)
    # Passed parameters checking function
    _stypy_temp_lambda_49.stypy_localization = localization
    _stypy_temp_lambda_49.stypy_type_of_self = None
    _stypy_temp_lambda_49.stypy_type_store = module_type_store
    _stypy_temp_lambda_49.stypy_function_name = '_stypy_temp_lambda_49'
    _stypy_temp_lambda_49.stypy_param_names_list = ['target', 'value']
    _stypy_temp_lambda_49.stypy_varargs_param_name = None
    _stypy_temp_lambda_49.stypy_kwargs_param_name = None
    _stypy_temp_lambda_49.stypy_call_defaults = defaults
    _stypy_temp_lambda_49.stypy_call_varargs = varargs
    _stypy_temp_lambda_49.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_49', ['target', 'value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_49', ['target', 'value'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to isinstance(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'target' (line 129)
    target_30830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 41), 'target', False)
    # Getting the type of 'ast' (line 129)
    ast_30831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 49), 'ast', False)
    # Obtaining the member 'Attribute' of a type (line 129)
    Attribute_30832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 49), ast_30831, 'Attribute')
    # Processing the call keyword arguments (line 129)
    kwargs_30833 = {}
    # Getting the type of 'isinstance' (line 129)
    isinstance_30829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 30), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 129)
    isinstance_call_result_30834 = invoke(stypy.reporting.localization.Localization(__file__, 129, 30), isinstance_30829, *[target_30830, Attribute_30832], **kwargs_30833)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type', isinstance_call_result_30834)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_49' in the type store
    # Getting the type of 'stypy_return_type' (line 129)
    stypy_return_type_30835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30835)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_49'
    return stypy_return_type_30835

# Assigning a type to the variable '_stypy_temp_lambda_49' (line 129)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), '_stypy_temp_lambda_49', _stypy_temp_lambda_49)
# Getting the type of '_stypy_temp_lambda_49' (line 129)
_stypy_temp_lambda_49_30836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), '_stypy_temp_lambda_49')

# Obtaining an instance of the builtin type 'tuple' (line 130)
tuple_30837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 130)
# Adding element type (line 130)
str_30838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 13), 'str', 'assignment')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 13), tuple_30837, str_30838)
# Adding element type (line 130)
# Getting the type of 'single_assignment_handler' (line 130)
single_assignment_handler_30839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 27), 'single_assignment_handler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 13), tuple_30837, single_assignment_handler_30839)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 28), dict_30731, (_stypy_temp_lambda_49_30836, tuple_30837))

# Getting the type of 'MultipleAssignmentsDesugaringVisitor'
MultipleAssignmentsDesugaringVisitor_30840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MultipleAssignmentsDesugaringVisitor')
# Setting the type of the member '__assignment_handlers' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MultipleAssignmentsDesugaringVisitor_30840, '__assignment_handlers', dict_30731)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
