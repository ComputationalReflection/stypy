
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import copy
2: 
3: from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import *
4: from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, stypy_functions_copy, functions_copy
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

# 'from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import ' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')
import_13783 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy')

if (type(import_13783) is not StypyTypeError):

    if (import_13783 != 'pyd_module'):
        __import__(import_13783)
        sys_modules_13784 = sys.modules[import_13783]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', sys_modules_13784.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_13784, sys_modules_13784.module_type_store, module_type_store)
    else:
        from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy', import_13783)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, stypy_functions_copy, functions_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')
import_13785 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy')

if (type(import_13785) is not StypyTypeError):

    if (import_13785 != 'pyd_module'):
        __import__(import_13785)
        sys_modules_13786 = sys.modules[import_13785]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', sys_modules_13786.module_type_store, module_type_store, ['core_language_copy', 'stypy_functions_copy', 'functions_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_13786, sys_modules_13786.module_type_store, module_type_store)
    else:
        from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, stypy_functions_copy, functions_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', None, module_type_store, ['core_language_copy', 'stypy_functions_copy', 'functions_copy'], [core_language_copy, stypy_functions_copy, functions_copy])

else:
    # Assigning a type to the variable 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', import_13785)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/desugaring_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import ast' statement (line 5)
import ast

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'ast', ast, module_type_store)

str_13787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\nThis visitor decompose several forms of multiple assignments into single assignments, that can be properly processed\nby stypy. There are various forms of assignments in Python that involve multiple elements, such as:\n\na, b = c, d\na, b = [c, d]\na, b = function_that_returns_tuple()\na, b = function_that_returns_list()\n\nThis visitor reduces the complexity of dealing with assignments when generating type inference code\n')

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

    str_13788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, (-1)), 'str', '\n    Handles code that uses a multiple assignment with a call to a function in the right part of the assignment. The\n    code var1, var2 = original_call(...) is transformed into:\n\n    temp = original_call(...)\n    var1 = temp[0]\n    var2 = temp[1]\n    ...\n\n    This way we only perform the call once.\n\n    :param target: tuple or list of variables\n    :param value: call\n    :param assign_stmts: statements holder list\n    :param node: current AST node\n    :param id_str: Type of assignment that we are processing (to create variables)\n    :return:\n    ')
    
    # Assigning a Call to a Tuple (line 39):
    
    # Assigning a Call to a Name:
    
    # Call to create_temp_Assign(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'value' (line 39)
    value_13791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 70), 'value', False)
    # Getting the type of 'node' (line 39)
    node_13792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 77), 'node', False)
    # Obtaining the member 'lineno' of a type (line 39)
    lineno_13793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 77), node_13792, 'lineno')
    # Getting the type of 'node' (line 39)
    node_13794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 90), 'node', False)
    # Obtaining the member 'col_offset' of a type (line 39)
    col_offset_13795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 90), node_13794, 'col_offset')
    
    # Call to format(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'id_str' (line 40)
    id_str_13798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 89), 'id_str', False)
    # Processing the call keyword arguments (line 40)
    kwargs_13799 = {}
    str_13796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 65), 'str', '{0}_assignment')
    # Obtaining the member 'format' of a type (line 40)
    format_13797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 65), str_13796, 'format')
    # Calling format(args, kwargs) (line 40)
    format_call_result_13800 = invoke(stypy.reporting.localization.Localization(__file__, 40, 65), format_13797, *[id_str_13798], **kwargs_13799)
    
    # Processing the call keyword arguments (line 39)
    kwargs_13801 = {}
    # Getting the type of 'stypy_functions_copy' (line 39)
    stypy_functions_copy_13789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 30), 'stypy_functions_copy', False)
    # Obtaining the member 'create_temp_Assign' of a type (line 39)
    create_temp_Assign_13790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 30), stypy_functions_copy_13789, 'create_temp_Assign')
    # Calling create_temp_Assign(args, kwargs) (line 39)
    create_temp_Assign_call_result_13802 = invoke(stypy.reporting.localization.Localization(__file__, 39, 30), create_temp_Assign_13790, *[value_13791, lineno_13793, col_offset_13795, format_call_result_13800], **kwargs_13801)
    
    # Assigning a type to the variable 'call_assignment_13772' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'call_assignment_13772', create_temp_Assign_call_result_13802)
    
    # Assigning a Call to a Name (line 39):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_13772' (line 39)
    call_assignment_13772_13803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'call_assignment_13772', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_13804 = stypy_get_value_from_tuple(call_assignment_13772_13803, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_13773' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'call_assignment_13773', stypy_get_value_from_tuple_call_result_13804)
    
    # Assigning a Name to a Name (line 39):
    # Getting the type of 'call_assignment_13773' (line 39)
    call_assignment_13773_13805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'call_assignment_13773')
    # Assigning a type to the variable 'target_stmts' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'target_stmts', call_assignment_13773_13805)
    
    # Assigning a Call to a Name (line 39):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_13772' (line 39)
    call_assignment_13772_13806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'call_assignment_13772', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_13807 = stypy_get_value_from_tuple(call_assignment_13772_13806, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_13774' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'call_assignment_13774', stypy_get_value_from_tuple_call_result_13807)
    
    # Assigning a Name to a Name (line 39):
    # Getting the type of 'call_assignment_13774' (line 39)
    call_assignment_13774_13808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'call_assignment_13774')
    # Assigning a type to the variable 'value_var' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 18), 'value_var', call_assignment_13774_13808)
    
    # Call to append(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'target_stmts' (line 41)
    target_stmts_13811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), 'target_stmts', False)
    # Processing the call keyword arguments (line 41)
    kwargs_13812 = {}
    # Getting the type of 'assign_stmts' (line 41)
    assign_stmts_13809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'assign_stmts', False)
    # Obtaining the member 'append' of a type (line 41)
    append_13810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), assign_stmts_13809, 'append')
    # Calling append(args, kwargs) (line 41)
    append_call_result_13813 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), append_13810, *[target_stmts_13811], **kwargs_13812)
    
    
    # Assigning a Call to a Name (line 43):
    
    # Assigning a Call to a Name (line 43):
    
    # Call to deepcopy(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'value_var' (line 43)
    value_var_13816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 38), 'value_var', False)
    # Processing the call keyword arguments (line 43)
    kwargs_13817 = {}
    # Getting the type of 'copy' (line 43)
    copy_13814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 24), 'copy', False)
    # Obtaining the member 'deepcopy' of a type (line 43)
    deepcopy_13815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 24), copy_13814, 'deepcopy')
    # Calling deepcopy(args, kwargs) (line 43)
    deepcopy_call_result_13818 = invoke(stypy.reporting.localization.Localization(__file__, 43, 24), deepcopy_13815, *[value_var_13816], **kwargs_13817)
    
    # Assigning a type to the variable 'value_var_to_load' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'value_var_to_load', deepcopy_call_result_13818)
    
    # Assigning a Call to a Attribute (line 44):
    
    # Assigning a Call to a Attribute (line 44):
    
    # Call to Load(...): (line 44)
    # Processing the call keyword arguments (line 44)
    kwargs_13821 = {}
    # Getting the type of 'ast' (line 44)
    ast_13819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 28), 'ast', False)
    # Obtaining the member 'Load' of a type (line 44)
    Load_13820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 28), ast_13819, 'Load')
    # Calling Load(args, kwargs) (line 44)
    Load_call_result_13822 = invoke(stypy.reporting.localization.Localization(__file__, 44, 28), Load_13820, *[], **kwargs_13821)
    
    # Getting the type of 'value_var_to_load' (line 44)
    value_var_to_load_13823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'value_var_to_load')
    # Setting the type of the member 'ctx' of a type (line 44)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 4), value_var_to_load_13823, 'ctx', Load_call_result_13822)
    
    
    # Call to range(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Call to len(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'target' (line 46)
    target_13826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 23), 'target', False)
    # Obtaining the member 'elts' of a type (line 46)
    elts_13827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 23), target_13826, 'elts')
    # Processing the call keyword arguments (line 46)
    kwargs_13828 = {}
    # Getting the type of 'len' (line 46)
    len_13825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'len', False)
    # Calling len(args, kwargs) (line 46)
    len_call_result_13829 = invoke(stypy.reporting.localization.Localization(__file__, 46, 19), len_13825, *[elts_13827], **kwargs_13828)
    
    # Processing the call keyword arguments (line 46)
    kwargs_13830 = {}
    # Getting the type of 'range' (line 46)
    range_13824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 13), 'range', False)
    # Calling range(args, kwargs) (line 46)
    range_call_result_13831 = invoke(stypy.reporting.localization.Localization(__file__, 46, 13), range_13824, *[len_call_result_13829], **kwargs_13830)
    
    # Assigning a type to the variable 'range_call_result_13831' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'range_call_result_13831', range_call_result_13831)
    # Testing if the for loop is going to be iterated (line 46)
    # Testing the type of a for loop iterable (line 46)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 46, 4), range_call_result_13831)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 46, 4), range_call_result_13831):
        # Getting the type of the for loop variable (line 46)
        for_loop_var_13832 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 46, 4), range_call_result_13831)
        # Assigning a type to the variable 'i' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'i', for_loop_var_13832)
        # SSA begins for a for statement (line 46)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 48):
        
        # Assigning a Call to a Name (line 48):
        
        # Call to create_attribute(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'value_var_to_load' (line 48)
        value_var_to_load_13835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 58), 'value_var_to_load', False)
        str_13836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 77), 'str', '__getitem__')
        # Processing the call keyword arguments (line 48)
        
        # Call to Load(...): (line 48)
        # Processing the call keyword arguments (line 48)
        kwargs_13839 = {}
        # Getting the type of 'ast' (line 48)
        ast_13837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 100), 'ast', False)
        # Obtaining the member 'Load' of a type (line 48)
        Load_13838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 100), ast_13837, 'Load')
        # Calling Load(args, kwargs) (line 48)
        Load_call_result_13840 = invoke(stypy.reporting.localization.Localization(__file__, 48, 100), Load_13838, *[], **kwargs_13839)
        
        keyword_13841 = Load_call_result_13840
        # Getting the type of 'node' (line 49)
        node_13842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 58), 'node', False)
        # Obtaining the member 'lineno' of a type (line 49)
        lineno_13843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 58), node_13842, 'lineno')
        keyword_13844 = lineno_13843
        # Getting the type of 'node' (line 50)
        node_13845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 60), 'node', False)
        # Obtaining the member 'col_offset' of a type (line 50)
        col_offset_13846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 60), node_13845, 'col_offset')
        keyword_13847 = col_offset_13846
        kwargs_13848 = {'column': keyword_13847, 'line': keyword_13844, 'context': keyword_13841}
        # Getting the type of 'core_language_copy' (line 48)
        core_language_copy_13833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'core_language_copy', False)
        # Obtaining the member 'create_attribute' of a type (line 48)
        create_attribute_13834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 22), core_language_copy_13833, 'create_attribute')
        # Calling create_attribute(args, kwargs) (line 48)
        create_attribute_call_result_13849 = invoke(stypy.reporting.localization.Localization(__file__, 48, 22), create_attribute_13834, *[value_var_to_load_13835, str_13836], **kwargs_13848)
        
        # Assigning a type to the variable 'getitem_att' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'getitem_att', create_attribute_call_result_13849)
        
        # Assigning a Call to a Name (line 51):
        
        # Assigning a Call to a Name (line 51):
        
        # Call to create_call(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'getitem_att' (line 51)
        getitem_att_13852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 47), 'getitem_att', False)
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_13853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        
        # Call to create_num(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'i' (line 51)
        i_13856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 91), 'i', False)
        # Getting the type of 'node' (line 51)
        node_13857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 94), 'node', False)
        # Obtaining the member 'lineno' of a type (line 51)
        lineno_13858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 94), node_13857, 'lineno')
        # Getting the type of 'node' (line 51)
        node_13859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 107), 'node', False)
        # Obtaining the member 'col_offset' of a type (line 51)
        col_offset_13860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 107), node_13859, 'col_offset')
        # Processing the call keyword arguments (line 51)
        kwargs_13861 = {}
        # Getting the type of 'core_language_copy' (line 51)
        core_language_copy_13854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 61), 'core_language_copy', False)
        # Obtaining the member 'create_num' of a type (line 51)
        create_num_13855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 61), core_language_copy_13854, 'create_num')
        # Calling create_num(args, kwargs) (line 51)
        create_num_call_result_13862 = invoke(stypy.reporting.localization.Localization(__file__, 51, 61), create_num_13855, *[i_13856, lineno_13858, col_offset_13860], **kwargs_13861)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 60), list_13853, create_num_call_result_13862)
        
        # Processing the call keyword arguments (line 51)
        kwargs_13863 = {}
        # Getting the type of 'functions_copy' (line 51)
        functions_copy_13850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'functions_copy', False)
        # Obtaining the member 'create_call' of a type (line 51)
        create_call_13851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 20), functions_copy_13850, 'create_call')
        # Calling create_call(args, kwargs) (line 51)
        create_call_call_result_13864 = invoke(stypy.reporting.localization.Localization(__file__, 51, 20), create_call_13851, *[getitem_att_13852, list_13853], **kwargs_13863)
        
        # Assigning a type to the variable 'item_call' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'item_call', create_call_call_result_13864)
        
        # Assigning a Call to a Tuple (line 52):
        
        # Assigning a Call to a Name:
        
        # Call to create_temp_Assign(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'item_call' (line 52)
        item_call_13867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 73), 'item_call', False)
        # Getting the type of 'node' (line 52)
        node_13868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 84), 'node', False)
        # Obtaining the member 'lineno' of a type (line 52)
        lineno_13869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 84), node_13868, 'lineno')
        # Getting the type of 'node' (line 52)
        node_13870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 97), 'node', False)
        # Obtaining the member 'col_offset' of a type (line 52)
        col_offset_13871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 97), node_13870, 'col_offset')
        
        # Call to format(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'id_str' (line 53)
        id_str_13874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 92), 'id_str', False)
        # Processing the call keyword arguments (line 53)
        kwargs_13875 = {}
        str_13872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 68), 'str', '{0}_assignment')
        # Obtaining the member 'format' of a type (line 53)
        format_13873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 68), str_13872, 'format')
        # Calling format(args, kwargs) (line 53)
        format_call_result_13876 = invoke(stypy.reporting.localization.Localization(__file__, 53, 68), format_13873, *[id_str_13874], **kwargs_13875)
        
        # Processing the call keyword arguments (line 52)
        kwargs_13877 = {}
        # Getting the type of 'stypy_functions_copy' (line 52)
        stypy_functions_copy_13865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 33), 'stypy_functions_copy', False)
        # Obtaining the member 'create_temp_Assign' of a type (line 52)
        create_temp_Assign_13866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 33), stypy_functions_copy_13865, 'create_temp_Assign')
        # Calling create_temp_Assign(args, kwargs) (line 52)
        create_temp_Assign_call_result_13878 = invoke(stypy.reporting.localization.Localization(__file__, 52, 33), create_temp_Assign_13866, *[item_call_13867, lineno_13869, col_offset_13871, format_call_result_13876], **kwargs_13877)
        
        # Assigning a type to the variable 'call_assignment_13775' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'call_assignment_13775', create_temp_Assign_call_result_13878)
        
        # Assigning a Call to a Name (line 52):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_13775' (line 52)
        call_assignment_13775_13879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'call_assignment_13775', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_13880 = stypy_get_value_from_tuple(call_assignment_13775_13879, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_13776' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'call_assignment_13776', stypy_get_value_from_tuple_call_result_13880)
        
        # Assigning a Name to a Name (line 52):
        # Getting the type of 'call_assignment_13776' (line 52)
        call_assignment_13776_13881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'call_assignment_13776')
        # Assigning a type to the variable 'temp_stmts' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'temp_stmts', call_assignment_13776_13881)
        
        # Assigning a Call to a Name (line 52):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_13775' (line 52)
        call_assignment_13775_13882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'call_assignment_13775', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_13883 = stypy_get_value_from_tuple(call_assignment_13775_13882, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_13777' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'call_assignment_13777', stypy_get_value_from_tuple_call_result_13883)
        
        # Assigning a Name to a Name (line 52):
        # Getting the type of 'call_assignment_13777' (line 52)
        call_assignment_13777_13884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'call_assignment_13777')
        # Assigning a type to the variable 'temp_value' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'temp_value', call_assignment_13777_13884)
        
        # Call to append(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'temp_stmts' (line 54)
        temp_stmts_13887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'temp_stmts', False)
        # Processing the call keyword arguments (line 54)
        kwargs_13888 = {}
        # Getting the type of 'assign_stmts' (line 54)
        assign_stmts_13885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'assign_stmts', False)
        # Obtaining the member 'append' of a type (line 54)
        append_13886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), assign_stmts_13885, 'append')
        # Calling append(args, kwargs) (line 54)
        append_call_result_13889 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), append_13886, *[temp_stmts_13887], **kwargs_13888)
        
        
        # Assigning a Call to a Name (line 56):
        
        # Assigning a Call to a Name (line 56):
        
        # Call to create_Assign(...): (line 56)
        # Processing the call arguments (line 56)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 56)
        i_13892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 66), 'i', False)
        # Getting the type of 'target' (line 56)
        target_13893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 54), 'target', False)
        # Obtaining the member 'elts' of a type (line 56)
        elts_13894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 54), target_13893, 'elts')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___13895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 54), elts_13894, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_13896 = invoke(stypy.reporting.localization.Localization(__file__, 56, 54), getitem___13895, i_13892)
        
        # Getting the type of 'temp_value' (line 56)
        temp_value_13897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 70), 'temp_value', False)
        # Processing the call keyword arguments (line 56)
        kwargs_13898 = {}
        # Getting the type of 'core_language_copy' (line 56)
        core_language_copy_13890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'core_language_copy', False)
        # Obtaining the member 'create_Assign' of a type (line 56)
        create_Assign_13891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 21), core_language_copy_13890, 'create_Assign')
        # Calling create_Assign(args, kwargs) (line 56)
        create_Assign_call_result_13899 = invoke(stypy.reporting.localization.Localization(__file__, 56, 21), create_Assign_13891, *[subscript_call_result_13896, temp_value_13897], **kwargs_13898)
        
        # Assigning a type to the variable 'temp_stmts' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'temp_stmts', create_Assign_call_result_13899)
        
        # Call to append(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'temp_stmts' (line 57)
        temp_stmts_13902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'temp_stmts', False)
        # Processing the call keyword arguments (line 57)
        kwargs_13903 = {}
        # Getting the type of 'assign_stmts' (line 57)
        assign_stmts_13900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'assign_stmts', False)
        # Obtaining the member 'append' of a type (line 57)
        append_13901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), assign_stmts_13900, 'append')
        # Calling append(args, kwargs) (line 57)
        append_call_result_13904 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), append_13901, *[temp_stmts_13902], **kwargs_13903)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'multiple_value_call_assignment_handler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'multiple_value_call_assignment_handler' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_13905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13905)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'multiple_value_call_assignment_handler'
    return stypy_return_type_13905

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

    str_13906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, (-1)), 'str', '\n    Code to handle assignments like a, b = c, d. This code is converted to:\n    a = c\n    b = d\n\n    Length of left and right part is cheched to make sure we are dealing with a valid assignment (an error is produced\n    otherwise)\n\n    :param target: tuple or list of variables\n    :param value:  tuple or list of variables\n    :param assign_stmts: statements holder list\n    :param node: current AST node\n    :param id_str: Type of assignment that we are processing (to create variables)\n    :return:\n    ')
    
    
    # Call to len(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'target' (line 76)
    target_13908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'target', False)
    # Obtaining the member 'elts' of a type (line 76)
    elts_13909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 11), target_13908, 'elts')
    # Processing the call keyword arguments (line 76)
    kwargs_13910 = {}
    # Getting the type of 'len' (line 76)
    len_13907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 7), 'len', False)
    # Calling len(args, kwargs) (line 76)
    len_call_result_13911 = invoke(stypy.reporting.localization.Localization(__file__, 76, 7), len_13907, *[elts_13909], **kwargs_13910)
    
    
    # Call to len(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'value' (line 76)
    value_13913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'value', False)
    # Obtaining the member 'elts' of a type (line 76)
    elts_13914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 31), value_13913, 'elts')
    # Processing the call keyword arguments (line 76)
    kwargs_13915 = {}
    # Getting the type of 'len' (line 76)
    len_13912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'len', False)
    # Calling len(args, kwargs) (line 76)
    len_call_result_13916 = invoke(stypy.reporting.localization.Localization(__file__, 76, 27), len_13912, *[elts_13914], **kwargs_13915)
    
    # Applying the binary operator '==' (line 76)
    result_eq_13917 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 7), '==', len_call_result_13911, len_call_result_13916)
    
    # Testing if the type of an if condition is none (line 76)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 4), result_eq_13917):
        
        # Call to TypeError(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Call to create_localization(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'node' (line 88)
        node_13993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 59), 'node', False)
        # Obtaining the member 'lineno' of a type (line 88)
        lineno_13994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 59), node_13993, 'lineno')
        # Getting the type of 'node' (line 88)
        node_13995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 72), 'node', False)
        # Obtaining the member 'col_offset' of a type (line 88)
        col_offset_13996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 72), node_13995, 'col_offset')
        # Processing the call keyword arguments (line 88)
        kwargs_13997 = {}
        # Getting the type of 'stypy_functions_copy' (line 88)
        stypy_functions_copy_13991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'stypy_functions_copy', False)
        # Obtaining the member 'create_localization' of a type (line 88)
        create_localization_13992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 18), stypy_functions_copy_13991, 'create_localization')
        # Calling create_localization(args, kwargs) (line 88)
        create_localization_call_result_13998 = invoke(stypy.reporting.localization.Localization(__file__, 88, 18), create_localization_13992, *[lineno_13994, col_offset_13996], **kwargs_13997)
        
        
        # Call to format(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'id_str' (line 90)
        id_str_14001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'id_str', False)
        # Processing the call keyword arguments (line 89)
        kwargs_14002 = {}
        str_13999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 18), 'str', 'Multi-value assignments with {0}s must have the same amount of elements on both assignment sides')
        # Obtaining the member 'format' of a type (line 89)
        format_14000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 18), str_13999, 'format')
        # Calling format(args, kwargs) (line 89)
        format_call_result_14003 = invoke(stypy.reporting.localization.Localization(__file__, 89, 18), format_14000, *[id_str_14001], **kwargs_14002)
        
        # Processing the call keyword arguments (line 88)
        kwargs_14004 = {}
        # Getting the type of 'TypeError' (line 88)
        TypeError_13990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 88)
        TypeError_call_result_14005 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), TypeError_13990, *[create_localization_call_result_13998, format_call_result_14003], **kwargs_14004)
        
    else:
        
        # Testing the type of an if condition (line 76)
        if_condition_13918 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 4), result_eq_13917)
        # Assigning a type to the variable 'if_condition_13918' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'if_condition_13918', if_condition_13918)
        # SSA begins for if statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 77):
        
        # Assigning a List to a Name (line 77):
        
        # Obtaining an instance of the builtin type 'list' (line 77)
        list_13919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 77)
        
        # Assigning a type to the variable 'temp_var_names' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'temp_var_names', list_13919)
        
        
        # Call to range(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Call to len(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'value' (line 79)
        value_13922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 27), 'value', False)
        # Obtaining the member 'elts' of a type (line 79)
        elts_13923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 27), value_13922, 'elts')
        # Processing the call keyword arguments (line 79)
        kwargs_13924 = {}
        # Getting the type of 'len' (line 79)
        len_13921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'len', False)
        # Calling len(args, kwargs) (line 79)
        len_call_result_13925 = invoke(stypy.reporting.localization.Localization(__file__, 79, 23), len_13921, *[elts_13923], **kwargs_13924)
        
        # Processing the call keyword arguments (line 79)
        kwargs_13926 = {}
        # Getting the type of 'range' (line 79)
        range_13920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'range', False)
        # Calling range(args, kwargs) (line 79)
        range_call_result_13927 = invoke(stypy.reporting.localization.Localization(__file__, 79, 17), range_13920, *[len_call_result_13925], **kwargs_13926)
        
        # Assigning a type to the variable 'range_call_result_13927' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'range_call_result_13927', range_call_result_13927)
        # Testing if the for loop is going to be iterated (line 79)
        # Testing the type of a for loop iterable (line 79)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 79, 8), range_call_result_13927)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 79, 8), range_call_result_13927):
            # Getting the type of the for loop variable (line 79)
            for_loop_var_13928 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 79, 8), range_call_result_13927)
            # Assigning a type to the variable 'i' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'i', for_loop_var_13928)
            # SSA begins for a for statement (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Tuple (line 80):
            
            # Assigning a Call to a Name:
            
            # Call to create_temp_Assign(...): (line 80)
            # Processing the call arguments (line 80)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 80)
            i_13931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 88), 'i', False)
            # Getting the type of 'value' (line 80)
            value_13932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 77), 'value', False)
            # Obtaining the member 'elts' of a type (line 80)
            elts_13933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 77), value_13932, 'elts')
            # Obtaining the member '__getitem__' of a type (line 80)
            getitem___13934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 77), elts_13933, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 80)
            subscript_call_result_13935 = invoke(stypy.reporting.localization.Localization(__file__, 80, 77), getitem___13934, i_13931)
            
            # Getting the type of 'node' (line 80)
            node_13936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 92), 'node', False)
            # Obtaining the member 'lineno' of a type (line 80)
            lineno_13937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 92), node_13936, 'lineno')
            # Getting the type of 'node' (line 80)
            node_13938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 105), 'node', False)
            # Obtaining the member 'col_offset' of a type (line 80)
            col_offset_13939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 105), node_13938, 'col_offset')
            
            # Call to format(...): (line 81)
            # Processing the call arguments (line 81)
            # Getting the type of 'id_str' (line 81)
            id_str_13942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 96), 'id_str', False)
            # Processing the call keyword arguments (line 81)
            kwargs_13943 = {}
            str_13940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 72), 'str', '{0}_assignment')
            # Obtaining the member 'format' of a type (line 81)
            format_13941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 72), str_13940, 'format')
            # Calling format(args, kwargs) (line 81)
            format_call_result_13944 = invoke(stypy.reporting.localization.Localization(__file__, 81, 72), format_13941, *[id_str_13942], **kwargs_13943)
            
            # Processing the call keyword arguments (line 80)
            kwargs_13945 = {}
            # Getting the type of 'stypy_functions_copy' (line 80)
            stypy_functions_copy_13929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 37), 'stypy_functions_copy', False)
            # Obtaining the member 'create_temp_Assign' of a type (line 80)
            create_temp_Assign_13930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 37), stypy_functions_copy_13929, 'create_temp_Assign')
            # Calling create_temp_Assign(args, kwargs) (line 80)
            create_temp_Assign_call_result_13946 = invoke(stypy.reporting.localization.Localization(__file__, 80, 37), create_temp_Assign_13930, *[subscript_call_result_13935, lineno_13937, col_offset_13939, format_call_result_13944], **kwargs_13945)
            
            # Assigning a type to the variable 'call_assignment_13778' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'call_assignment_13778', create_temp_Assign_call_result_13946)
            
            # Assigning a Call to a Name (line 80):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_13778' (line 80)
            call_assignment_13778_13947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'call_assignment_13778', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_13948 = stypy_get_value_from_tuple(call_assignment_13778_13947, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_13779' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'call_assignment_13779', stypy_get_value_from_tuple_call_result_13948)
            
            # Assigning a Name to a Name (line 80):
            # Getting the type of 'call_assignment_13779' (line 80)
            call_assignment_13779_13949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'call_assignment_13779')
            # Assigning a type to the variable 'temp_stmts' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'temp_stmts', call_assignment_13779_13949)
            
            # Assigning a Call to a Name (line 80):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_13778' (line 80)
            call_assignment_13778_13950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'call_assignment_13778', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_13951 = stypy_get_value_from_tuple(call_assignment_13778_13950, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_13780' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'call_assignment_13780', stypy_get_value_from_tuple_call_result_13951)
            
            # Assigning a Name to a Name (line 80):
            # Getting the type of 'call_assignment_13780' (line 80)
            call_assignment_13780_13952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'call_assignment_13780')
            # Assigning a type to the variable 'temp_value' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), 'temp_value', call_assignment_13780_13952)
            
            # Call to append(...): (line 82)
            # Processing the call arguments (line 82)
            # Getting the type of 'temp_stmts' (line 82)
            temp_stmts_13955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 32), 'temp_stmts', False)
            # Processing the call keyword arguments (line 82)
            kwargs_13956 = {}
            # Getting the type of 'assign_stmts' (line 82)
            assign_stmts_13953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'assign_stmts', False)
            # Obtaining the member 'append' of a type (line 82)
            append_13954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), assign_stmts_13953, 'append')
            # Calling append(args, kwargs) (line 82)
            append_call_result_13957 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), append_13954, *[temp_stmts_13955], **kwargs_13956)
            
            
            # Call to append(...): (line 83)
            # Processing the call arguments (line 83)
            # Getting the type of 'temp_value' (line 83)
            temp_value_13960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 34), 'temp_value', False)
            # Processing the call keyword arguments (line 83)
            kwargs_13961 = {}
            # Getting the type of 'temp_var_names' (line 83)
            temp_var_names_13958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'temp_var_names', False)
            # Obtaining the member 'append' of a type (line 83)
            append_13959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), temp_var_names_13958, 'append')
            # Calling append(args, kwargs) (line 83)
            append_call_result_13962 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), append_13959, *[temp_value_13960], **kwargs_13961)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to range(...): (line 84)
        # Processing the call arguments (line 84)
        
        # Call to len(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'target' (line 84)
        target_13965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'target', False)
        # Obtaining the member 'elts' of a type (line 84)
        elts_13966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 27), target_13965, 'elts')
        # Processing the call keyword arguments (line 84)
        kwargs_13967 = {}
        # Getting the type of 'len' (line 84)
        len_13964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'len', False)
        # Calling len(args, kwargs) (line 84)
        len_call_result_13968 = invoke(stypy.reporting.localization.Localization(__file__, 84, 23), len_13964, *[elts_13966], **kwargs_13967)
        
        # Processing the call keyword arguments (line 84)
        kwargs_13969 = {}
        # Getting the type of 'range' (line 84)
        range_13963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 17), 'range', False)
        # Calling range(args, kwargs) (line 84)
        range_call_result_13970 = invoke(stypy.reporting.localization.Localization(__file__, 84, 17), range_13963, *[len_call_result_13968], **kwargs_13969)
        
        # Assigning a type to the variable 'range_call_result_13970' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'range_call_result_13970', range_call_result_13970)
        # Testing if the for loop is going to be iterated (line 84)
        # Testing the type of a for loop iterable (line 84)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 84, 8), range_call_result_13970)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 84, 8), range_call_result_13970):
            # Getting the type of the for loop variable (line 84)
            for_loop_var_13971 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 84, 8), range_call_result_13970)
            # Assigning a type to the variable 'i' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'i', for_loop_var_13971)
            # SSA begins for a for statement (line 84)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 85):
            
            # Assigning a Call to a Name (line 85):
            
            # Call to create_Assign(...): (line 85)
            # Processing the call arguments (line 85)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 85)
            i_13974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 70), 'i', False)
            # Getting the type of 'target' (line 85)
            target_13975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 58), 'target', False)
            # Obtaining the member 'elts' of a type (line 85)
            elts_13976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 58), target_13975, 'elts')
            # Obtaining the member '__getitem__' of a type (line 85)
            getitem___13977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 58), elts_13976, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 85)
            subscript_call_result_13978 = invoke(stypy.reporting.localization.Localization(__file__, 85, 58), getitem___13977, i_13974)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 85)
            i_13979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 89), 'i', False)
            # Getting the type of 'temp_var_names' (line 85)
            temp_var_names_13980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 74), 'temp_var_names', False)
            # Obtaining the member '__getitem__' of a type (line 85)
            getitem___13981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 74), temp_var_names_13980, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 85)
            subscript_call_result_13982 = invoke(stypy.reporting.localization.Localization(__file__, 85, 74), getitem___13981, i_13979)
            
            # Processing the call keyword arguments (line 85)
            kwargs_13983 = {}
            # Getting the type of 'core_language_copy' (line 85)
            core_language_copy_13972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 25), 'core_language_copy', False)
            # Obtaining the member 'create_Assign' of a type (line 85)
            create_Assign_13973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 25), core_language_copy_13972, 'create_Assign')
            # Calling create_Assign(args, kwargs) (line 85)
            create_Assign_call_result_13984 = invoke(stypy.reporting.localization.Localization(__file__, 85, 25), create_Assign_13973, *[subscript_call_result_13978, subscript_call_result_13982], **kwargs_13983)
            
            # Assigning a type to the variable 'temp_stmts' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'temp_stmts', create_Assign_call_result_13984)
            
            # Call to append(...): (line 86)
            # Processing the call arguments (line 86)
            # Getting the type of 'temp_stmts' (line 86)
            temp_stmts_13987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 32), 'temp_stmts', False)
            # Processing the call keyword arguments (line 86)
            kwargs_13988 = {}
            # Getting the type of 'assign_stmts' (line 86)
            assign_stmts_13985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'assign_stmts', False)
            # Obtaining the member 'append' of a type (line 86)
            append_13986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), assign_stmts_13985, 'append')
            # Calling append(args, kwargs) (line 86)
            append_call_result_13989 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), append_13986, *[temp_stmts_13987], **kwargs_13988)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA branch for the else part of an if statement (line 76)
        module_type_store.open_ssa_branch('else')
        
        # Call to TypeError(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Call to create_localization(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'node' (line 88)
        node_13993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 59), 'node', False)
        # Obtaining the member 'lineno' of a type (line 88)
        lineno_13994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 59), node_13993, 'lineno')
        # Getting the type of 'node' (line 88)
        node_13995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 72), 'node', False)
        # Obtaining the member 'col_offset' of a type (line 88)
        col_offset_13996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 72), node_13995, 'col_offset')
        # Processing the call keyword arguments (line 88)
        kwargs_13997 = {}
        # Getting the type of 'stypy_functions_copy' (line 88)
        stypy_functions_copy_13991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 18), 'stypy_functions_copy', False)
        # Obtaining the member 'create_localization' of a type (line 88)
        create_localization_13992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 18), stypy_functions_copy_13991, 'create_localization')
        # Calling create_localization(args, kwargs) (line 88)
        create_localization_call_result_13998 = invoke(stypy.reporting.localization.Localization(__file__, 88, 18), create_localization_13992, *[lineno_13994, col_offset_13996], **kwargs_13997)
        
        
        # Call to format(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'id_str' (line 90)
        id_str_14001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'id_str', False)
        # Processing the call keyword arguments (line 89)
        kwargs_14002 = {}
        str_13999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 18), 'str', 'Multi-value assignments with {0}s must have the same amount of elements on both assignment sides')
        # Obtaining the member 'format' of a type (line 89)
        format_14000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 18), str_13999, 'format')
        # Calling format(args, kwargs) (line 89)
        format_call_result_14003 = invoke(stypy.reporting.localization.Localization(__file__, 89, 18), format_14000, *[id_str_14001], **kwargs_14002)
        
        # Processing the call keyword arguments (line 88)
        kwargs_14004 = {}
        # Getting the type of 'TypeError' (line 88)
        TypeError_13990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 88)
        TypeError_call_result_14005 = invoke(stypy.reporting.localization.Localization(__file__, 88, 8), TypeError_13990, *[create_localization_call_result_13998, format_call_result_14003], **kwargs_14004)
        
        # SSA join for if statement (line 76)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'multiple_value_assignment_handler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'multiple_value_assignment_handler' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_14006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14006)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'multiple_value_assignment_handler'
    return stypy_return_type_14006

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

    str_14007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, (-1)), 'str', '\n    Handles single statements for hte visitor. No change is produced in the code\n    :param target: Variable\n    :param value: Value to assign\n    :param assign_stmts: statements holder list\n    :param node: current AST node\n    :param id_str: Type of assignment that we are processing (to create variables)\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 103):
    
    # Assigning a Call to a Name (line 103):
    
    # Call to create_Assign(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'target' (line 103)
    target_14010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 50), 'target', False)
    # Getting the type of 'value' (line 103)
    value_14011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 58), 'value', False)
    # Processing the call keyword arguments (line 103)
    kwargs_14012 = {}
    # Getting the type of 'core_language_copy' (line 103)
    core_language_copy_14008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 17), 'core_language_copy', False)
    # Obtaining the member 'create_Assign' of a type (line 103)
    create_Assign_14009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 17), core_language_copy_14008, 'create_Assign')
    # Calling create_Assign(args, kwargs) (line 103)
    create_Assign_call_result_14013 = invoke(stypy.reporting.localization.Localization(__file__, 103, 17), create_Assign_14009, *[target_14010, value_14011], **kwargs_14012)
    
    # Assigning a type to the variable 'temp_stmts' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'temp_stmts', create_Assign_call_result_14013)
    
    # Call to append(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'temp_stmts' (line 104)
    temp_stmts_14016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'temp_stmts', False)
    # Processing the call keyword arguments (line 104)
    kwargs_14017 = {}
    # Getting the type of 'assign_stmts' (line 104)
    assign_stmts_14014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'assign_stmts', False)
    # Obtaining the member 'append' of a type (line 104)
    append_14015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 4), assign_stmts_14014, 'append')
    # Calling append(args, kwargs) (line 104)
    append_call_result_14018 = invoke(stypy.reporting.localization.Localization(__file__, 104, 4), append_14015, *[temp_stmts_14016], **kwargs_14017)
    
    
    # ################# End of 'single_assignment_handler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'single_assignment_handler' in the type store
    # Getting the type of 'stypy_return_type' (line 93)
    stypy_return_type_14019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14019)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'single_assignment_handler'
    return stypy_return_type_14019

# Assigning a type to the variable 'single_assignment_handler' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'single_assignment_handler', single_assignment_handler)
# Declaration of the 'MultipleAssignmentsDesugaringVisitor' class
# Getting the type of 'ast' (line 107)
ast_14020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 43), 'ast')
# Obtaining the member 'NodeTransformer' of a type (line 107)
NodeTransformer_14021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 43), ast_14020, 'NodeTransformer')

class MultipleAssignmentsDesugaringVisitor(NodeTransformer_14021, ):
    
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
        list_14022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 136)
        
        # Assigning a type to the variable 'assign_stmts' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'assign_stmts', list_14022)
        
        # Assigning a Attribute to a Name (line 137):
        
        # Assigning a Attribute to a Name (line 137):
        # Getting the type of 'node' (line 137)
        node_14023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'node')
        # Obtaining the member 'value' of a type (line 137)
        value_14024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 16), node_14023, 'value')
        # Assigning a type to the variable 'value' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'value', value_14024)
        
        # Assigning a Attribute to a Name (line 138):
        
        # Assigning a Attribute to a Name (line 138):
        # Getting the type of 'node' (line 138)
        node_14025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 27), 'node')
        # Obtaining the member 'targets' of a type (line 138)
        targets_14026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 27), node_14025, 'targets')
        # Assigning a type to the variable 'reversed_targets' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'reversed_targets', targets_14026)
        
        # Call to reverse(...): (line 139)
        # Processing the call keyword arguments (line 139)
        kwargs_14029 = {}
        # Getting the type of 'reversed_targets' (line 139)
        reversed_targets_14027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'reversed_targets', False)
        # Obtaining the member 'reverse' of a type (line 139)
        reverse_14028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), reversed_targets_14027, 'reverse')
        # Calling reverse(args, kwargs) (line 139)
        reverse_call_result_14030 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), reverse_14028, *[], **kwargs_14029)
        
        
        # Call to append(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Call to create_blank_line(...): (line 140)
        # Processing the call keyword arguments (line 140)
        kwargs_14035 = {}
        # Getting the type of 'stypy_functions_copy' (line 140)
        stypy_functions_copy_14033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 28), 'stypy_functions_copy', False)
        # Obtaining the member 'create_blank_line' of a type (line 140)
        create_blank_line_14034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 28), stypy_functions_copy_14033, 'create_blank_line')
        # Calling create_blank_line(args, kwargs) (line 140)
        create_blank_line_call_result_14036 = invoke(stypy.reporting.localization.Localization(__file__, 140, 28), create_blank_line_14034, *[], **kwargs_14035)
        
        # Processing the call keyword arguments (line 140)
        kwargs_14037 = {}
        # Getting the type of 'assign_stmts' (line 140)
        assign_stmts_14031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'assign_stmts', False)
        # Obtaining the member 'append' of a type (line 140)
        append_14032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), assign_stmts_14031, 'append')
        # Calling append(args, kwargs) (line 140)
        append_call_result_14038 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), append_14032, *[create_blank_line_call_result_14036], **kwargs_14037)
        
        
        
        # Call to len(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'reversed_targets' (line 141)
        reversed_targets_14040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'reversed_targets', False)
        # Processing the call keyword arguments (line 141)
        kwargs_14041 = {}
        # Getting the type of 'len' (line 141)
        len_14039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'len', False)
        # Calling len(args, kwargs) (line 141)
        len_call_result_14042 = invoke(stypy.reporting.localization.Localization(__file__, 141, 11), len_14039, *[reversed_targets_14040], **kwargs_14041)
        
        int_14043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 35), 'int')
        # Applying the binary operator '>' (line 141)
        result_gt_14044 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 11), '>', len_call_result_14042, int_14043)
        
        # Testing if the type of an if condition is none (line 141)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 141, 8), result_gt_14044):
            
            # Call to append(...): (line 145)
            # Processing the call arguments (line 145)
            
            # Call to create_src_comment(...): (line 145)
            # Processing the call arguments (line 145)
            
            # Call to format(...): (line 146)
            # Processing the call arguments (line 146)
            
            # Call to type(...): (line 146)
            # Processing the call arguments (line 146)
            
            # Obtaining the type of the subscript
            int_14069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 78), 'int')
            # Getting the type of 'reversed_targets' (line 146)
            reversed_targets_14070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 61), 'reversed_targets', False)
            # Obtaining the member '__getitem__' of a type (line 146)
            getitem___14071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 61), reversed_targets_14070, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 146)
            subscript_call_result_14072 = invoke(stypy.reporting.localization.Localization(__file__, 146, 61), getitem___14071, int_14069)
            
            # Processing the call keyword arguments (line 146)
            kwargs_14073 = {}
            # Getting the type of 'type' (line 146)
            type_14068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 56), 'type', False)
            # Calling type(args, kwargs) (line 146)
            type_call_result_14074 = invoke(stypy.reporting.localization.Localization(__file__, 146, 56), type_14068, *[subscript_call_result_14072], **kwargs_14073)
            
            # Obtaining the member '__name__' of a type (line 146)
            name___14075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 56), type_call_result_14074, '__name__')
            
            # Call to type(...): (line 147)
            # Processing the call arguments (line 147)
            # Getting the type of 'value' (line 147)
            value_14077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 61), 'value', False)
            # Processing the call keyword arguments (line 147)
            kwargs_14078 = {}
            # Getting the type of 'type' (line 147)
            type_14076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 56), 'type', False)
            # Calling type(args, kwargs) (line 147)
            type_call_result_14079 = invoke(stypy.reporting.localization.Localization(__file__, 147, 56), type_14076, *[value_14077], **kwargs_14078)
            
            # Obtaining the member '__name__' of a type (line 147)
            name___14080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 56), type_call_result_14079, '__name__')
            # Processing the call keyword arguments (line 146)
            kwargs_14081 = {}
            str_14066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 16), 'str', 'Assignment to a {0} from a {1}')
            # Obtaining the member 'format' of a type (line 146)
            format_14067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), str_14066, 'format')
            # Calling format(args, kwargs) (line 146)
            format_call_result_14082 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), format_14067, *[name___14075, name___14080], **kwargs_14081)
            
            # Processing the call keyword arguments (line 145)
            kwargs_14083 = {}
            # Getting the type of 'stypy_functions_copy' (line 145)
            stypy_functions_copy_14064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 32), 'stypy_functions_copy', False)
            # Obtaining the member 'create_src_comment' of a type (line 145)
            create_src_comment_14065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 32), stypy_functions_copy_14064, 'create_src_comment')
            # Calling create_src_comment(args, kwargs) (line 145)
            create_src_comment_call_result_14084 = invoke(stypy.reporting.localization.Localization(__file__, 145, 32), create_src_comment_14065, *[format_call_result_14082], **kwargs_14083)
            
            # Processing the call keyword arguments (line 145)
            kwargs_14085 = {}
            # Getting the type of 'assign_stmts' (line 145)
            assign_stmts_14062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'assign_stmts', False)
            # Obtaining the member 'append' of a type (line 145)
            append_14063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), assign_stmts_14062, 'append')
            # Calling append(args, kwargs) (line 145)
            append_call_result_14086 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), append_14063, *[create_src_comment_call_result_14084], **kwargs_14085)
            
        else:
            
            # Testing the type of an if condition (line 141)
            if_condition_14045 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 8), result_gt_14044)
            # Assigning a type to the variable 'if_condition_14045' (line 141)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'if_condition_14045', if_condition_14045)
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
            reversed_targets_14053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 105), 'reversed_targets', False)
            # Processing the call keyword arguments (line 143)
            kwargs_14054 = {}
            # Getting the type of 'len' (line 143)
            len_14052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 101), 'len', False)
            # Calling len(args, kwargs) (line 143)
            len_call_result_14055 = invoke(stypy.reporting.localization.Localization(__file__, 143, 101), len_14052, *[reversed_targets_14053], **kwargs_14054)
            
            # Processing the call keyword arguments (line 143)
            kwargs_14056 = {}
            str_14050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 56), 'str', 'Multiple assigment of {0} elements.')
            # Obtaining the member 'format' of a type (line 143)
            format_14051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 56), str_14050, 'format')
            # Calling format(args, kwargs) (line 143)
            format_call_result_14057 = invoke(stypy.reporting.localization.Localization(__file__, 143, 56), format_14051, *[len_call_result_14055], **kwargs_14056)
            
            # Processing the call keyword arguments (line 143)
            kwargs_14058 = {}
            # Getting the type of 'stypy_functions_copy' (line 143)
            stypy_functions_copy_14048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'stypy_functions_copy', False)
            # Obtaining the member 'create_src_comment' of a type (line 143)
            create_src_comment_14049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 16), stypy_functions_copy_14048, 'create_src_comment')
            # Calling create_src_comment(args, kwargs) (line 143)
            create_src_comment_call_result_14059 = invoke(stypy.reporting.localization.Localization(__file__, 143, 16), create_src_comment_14049, *[format_call_result_14057], **kwargs_14058)
            
            # Processing the call keyword arguments (line 142)
            kwargs_14060 = {}
            # Getting the type of 'assign_stmts' (line 142)
            assign_stmts_14046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'assign_stmts', False)
            # Obtaining the member 'append' of a type (line 142)
            append_14047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 12), assign_stmts_14046, 'append')
            # Calling append(args, kwargs) (line 142)
            append_call_result_14061 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), append_14047, *[create_src_comment_call_result_14059], **kwargs_14060)
            
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
            int_14069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 78), 'int')
            # Getting the type of 'reversed_targets' (line 146)
            reversed_targets_14070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 61), 'reversed_targets', False)
            # Obtaining the member '__getitem__' of a type (line 146)
            getitem___14071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 61), reversed_targets_14070, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 146)
            subscript_call_result_14072 = invoke(stypy.reporting.localization.Localization(__file__, 146, 61), getitem___14071, int_14069)
            
            # Processing the call keyword arguments (line 146)
            kwargs_14073 = {}
            # Getting the type of 'type' (line 146)
            type_14068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 56), 'type', False)
            # Calling type(args, kwargs) (line 146)
            type_call_result_14074 = invoke(stypy.reporting.localization.Localization(__file__, 146, 56), type_14068, *[subscript_call_result_14072], **kwargs_14073)
            
            # Obtaining the member '__name__' of a type (line 146)
            name___14075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 56), type_call_result_14074, '__name__')
            
            # Call to type(...): (line 147)
            # Processing the call arguments (line 147)
            # Getting the type of 'value' (line 147)
            value_14077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 61), 'value', False)
            # Processing the call keyword arguments (line 147)
            kwargs_14078 = {}
            # Getting the type of 'type' (line 147)
            type_14076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 56), 'type', False)
            # Calling type(args, kwargs) (line 147)
            type_call_result_14079 = invoke(stypy.reporting.localization.Localization(__file__, 147, 56), type_14076, *[value_14077], **kwargs_14078)
            
            # Obtaining the member '__name__' of a type (line 147)
            name___14080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 56), type_call_result_14079, '__name__')
            # Processing the call keyword arguments (line 146)
            kwargs_14081 = {}
            str_14066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 16), 'str', 'Assignment to a {0} from a {1}')
            # Obtaining the member 'format' of a type (line 146)
            format_14067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 16), str_14066, 'format')
            # Calling format(args, kwargs) (line 146)
            format_call_result_14082 = invoke(stypy.reporting.localization.Localization(__file__, 146, 16), format_14067, *[name___14075, name___14080], **kwargs_14081)
            
            # Processing the call keyword arguments (line 145)
            kwargs_14083 = {}
            # Getting the type of 'stypy_functions_copy' (line 145)
            stypy_functions_copy_14064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 32), 'stypy_functions_copy', False)
            # Obtaining the member 'create_src_comment' of a type (line 145)
            create_src_comment_14065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 32), stypy_functions_copy_14064, 'create_src_comment')
            # Calling create_src_comment(args, kwargs) (line 145)
            create_src_comment_call_result_14084 = invoke(stypy.reporting.localization.Localization(__file__, 145, 32), create_src_comment_14065, *[format_call_result_14082], **kwargs_14083)
            
            # Processing the call keyword arguments (line 145)
            kwargs_14085 = {}
            # Getting the type of 'assign_stmts' (line 145)
            assign_stmts_14062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'assign_stmts', False)
            # Obtaining the member 'append' of a type (line 145)
            append_14063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 12), assign_stmts_14062, 'append')
            # Calling append(args, kwargs) (line 145)
            append_call_result_14086 = invoke(stypy.reporting.localization.Localization(__file__, 145, 12), append_14063, *[create_src_comment_call_result_14084], **kwargs_14085)
            
            # SSA join for if statement (line 141)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to range(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Call to len(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'reversed_targets' (line 149)
        reversed_targets_14089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 36), 'reversed_targets', False)
        # Processing the call keyword arguments (line 149)
        kwargs_14090 = {}
        # Getting the type of 'len' (line 149)
        len_14088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 32), 'len', False)
        # Calling len(args, kwargs) (line 149)
        len_call_result_14091 = invoke(stypy.reporting.localization.Localization(__file__, 149, 32), len_14088, *[reversed_targets_14089], **kwargs_14090)
        
        # Processing the call keyword arguments (line 149)
        kwargs_14092 = {}
        # Getting the type of 'range' (line 149)
        range_14087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 26), 'range', False)
        # Calling range(args, kwargs) (line 149)
        range_call_result_14093 = invoke(stypy.reporting.localization.Localization(__file__, 149, 26), range_14087, *[len_call_result_14091], **kwargs_14092)
        
        # Assigning a type to the variable 'range_call_result_14093' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'range_call_result_14093', range_call_result_14093)
        # Testing if the for loop is going to be iterated (line 149)
        # Testing the type of a for loop iterable (line 149)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 149, 8), range_call_result_14093)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 149, 8), range_call_result_14093):
            # Getting the type of the for loop variable (line 149)
            for_loop_var_14094 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 149, 8), range_call_result_14093)
            # Assigning a type to the variable 'assign_num' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'assign_num', for_loop_var_14094)
            # SSA begins for a for statement (line 149)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 150):
            
            # Assigning a Subscript to a Name (line 150):
            
            # Obtaining the type of the subscript
            # Getting the type of 'assign_num' (line 150)
            assign_num_14095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 38), 'assign_num')
            # Getting the type of 'reversed_targets' (line 150)
            reversed_targets_14096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'reversed_targets')
            # Obtaining the member '__getitem__' of a type (line 150)
            getitem___14097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 21), reversed_targets_14096, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 150)
            subscript_call_result_14098 = invoke(stypy.reporting.localization.Localization(__file__, 150, 21), getitem___14097, assign_num_14095)
            
            # Assigning a type to the variable 'target' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'target', subscript_call_result_14098)
            
            # Getting the type of 'self' (line 152)
            self_14099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 38), 'self')
            # Obtaining the member '__assignment_handlers' of a type (line 152)
            assignment_handlers_14100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 38), self_14099, '__assignment_handlers')
            # Assigning a type to the variable 'assignment_handlers_14100' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'assignment_handlers_14100', assignment_handlers_14100)
            # Testing if the for loop is going to be iterated (line 152)
            # Testing the type of a for loop iterable (line 152)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 152, 12), assignment_handlers_14100)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 152, 12), assignment_handlers_14100):
                # Getting the type of the for loop variable (line 152)
                for_loop_var_14101 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 152, 12), assignment_handlers_14100)
                # Assigning a type to the variable 'handler_func_guard' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'handler_func_guard', for_loop_var_14101)
                # SSA begins for a for statement (line 152)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to handler_func_guard(...): (line 153)
                # Processing the call arguments (line 153)
                # Getting the type of 'target' (line 153)
                target_14103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 38), 'target', False)
                # Getting the type of 'value' (line 153)
                value_14104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 46), 'value', False)
                # Processing the call keyword arguments (line 153)
                kwargs_14105 = {}
                # Getting the type of 'handler_func_guard' (line 153)
                handler_func_guard_14102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 19), 'handler_func_guard', False)
                # Calling handler_func_guard(args, kwargs) (line 153)
                handler_func_guard_call_result_14106 = invoke(stypy.reporting.localization.Localization(__file__, 153, 19), handler_func_guard_14102, *[target_14103, value_14104], **kwargs_14105)
                
                # Testing if the type of an if condition is none (line 153)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 153, 16), handler_func_guard_call_result_14106):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 153)
                    if_condition_14107 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 16), handler_func_guard_call_result_14106)
                    # Assigning a type to the variable 'if_condition_14107' (line 153)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'if_condition_14107', if_condition_14107)
                    # SSA begins for if statement (line 153)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Subscript to a Tuple (line 154):
                    
                    # Assigning a Subscript to a Name (line 154):
                    
                    # Obtaining the type of the subscript
                    int_14108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 20), 'int')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'handler_func_guard' (line 154)
                    handler_func_guard_14109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 70), 'handler_func_guard')
                    # Getting the type of 'self' (line 154)
                    self_14110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 43), 'self')
                    # Obtaining the member '__assignment_handlers' of a type (line 154)
                    assignment_handlers_14111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 43), self_14110, '__assignment_handlers')
                    # Obtaining the member '__getitem__' of a type (line 154)
                    getitem___14112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 43), assignment_handlers_14111, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
                    subscript_call_result_14113 = invoke(stypy.reporting.localization.Localization(__file__, 154, 43), getitem___14112, handler_func_guard_14109)
                    
                    # Obtaining the member '__getitem__' of a type (line 154)
                    getitem___14114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 20), subscript_call_result_14113, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
                    subscript_call_result_14115 = invoke(stypy.reporting.localization.Localization(__file__, 154, 20), getitem___14114, int_14108)
                    
                    # Assigning a type to the variable 'tuple_var_assignment_13781' (line 154)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'tuple_var_assignment_13781', subscript_call_result_14115)
                    
                    # Assigning a Subscript to a Name (line 154):
                    
                    # Obtaining the type of the subscript
                    int_14116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 20), 'int')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'handler_func_guard' (line 154)
                    handler_func_guard_14117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 70), 'handler_func_guard')
                    # Getting the type of 'self' (line 154)
                    self_14118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 43), 'self')
                    # Obtaining the member '__assignment_handlers' of a type (line 154)
                    assignment_handlers_14119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 43), self_14118, '__assignment_handlers')
                    # Obtaining the member '__getitem__' of a type (line 154)
                    getitem___14120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 43), assignment_handlers_14119, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
                    subscript_call_result_14121 = invoke(stypy.reporting.localization.Localization(__file__, 154, 43), getitem___14120, handler_func_guard_14117)
                    
                    # Obtaining the member '__getitem__' of a type (line 154)
                    getitem___14122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 20), subscript_call_result_14121, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
                    subscript_call_result_14123 = invoke(stypy.reporting.localization.Localization(__file__, 154, 20), getitem___14122, int_14116)
                    
                    # Assigning a type to the variable 'tuple_var_assignment_13782' (line 154)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'tuple_var_assignment_13782', subscript_call_result_14123)
                    
                    # Assigning a Name to a Name (line 154):
                    # Getting the type of 'tuple_var_assignment_13781' (line 154)
                    tuple_var_assignment_13781_14124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'tuple_var_assignment_13781')
                    # Assigning a type to the variable 'id_str' (line 154)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'id_str', tuple_var_assignment_13781_14124)
                    
                    # Assigning a Name to a Name (line 154):
                    # Getting the type of 'tuple_var_assignment_13782' (line 154)
                    tuple_var_assignment_13782_14125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'tuple_var_assignment_13782')
                    # Assigning a type to the variable 'handler_func' (line 154)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 28), 'handler_func', tuple_var_assignment_13782_14125)
                    
                    # Call to handler_func(...): (line 155)
                    # Processing the call arguments (line 155)
                    # Getting the type of 'target' (line 155)
                    target_14127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 33), 'target', False)
                    # Getting the type of 'value' (line 155)
                    value_14128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 41), 'value', False)
                    # Getting the type of 'assign_stmts' (line 155)
                    assign_stmts_14129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 48), 'assign_stmts', False)
                    # Getting the type of 'node' (line 155)
                    node_14130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 62), 'node', False)
                    # Getting the type of 'id_str' (line 155)
                    id_str_14131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 68), 'id_str', False)
                    # Processing the call keyword arguments (line 155)
                    kwargs_14132 = {}
                    # Getting the type of 'handler_func' (line 155)
                    handler_func_14126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'handler_func', False)
                    # Calling handler_func(args, kwargs) (line 155)
                    handler_func_call_result_14133 = invoke(stypy.reporting.localization.Localization(__file__, 155, 20), handler_func_14126, *[target_14127, value_14128, assign_stmts_14129, node_14130, id_str_14131], **kwargs_14132)
                    
                    
                    # Assigning a Call to a Name (line 156):
                    
                    # Assigning a Call to a Name (line 156):
                    
                    # Call to flatten_lists(...): (line 156)
                    # Processing the call arguments (line 156)
                    # Getting the type of 'assign_stmts' (line 156)
                    assign_stmts_14136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 70), 'assign_stmts', False)
                    # Processing the call keyword arguments (line 156)
                    kwargs_14137 = {}
                    # Getting the type of 'stypy_functions_copy' (line 156)
                    stypy_functions_copy_14134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 35), 'stypy_functions_copy', False)
                    # Obtaining the member 'flatten_lists' of a type (line 156)
                    flatten_lists_14135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 35), stypy_functions_copy_14134, 'flatten_lists')
                    # Calling flatten_lists(args, kwargs) (line 156)
                    flatten_lists_call_result_14138 = invoke(stypy.reporting.localization.Localization(__file__, 156, 35), flatten_lists_14135, *[assign_stmts_14136], **kwargs_14137)
                    
                    # Assigning a type to the variable 'assign_stmts' (line 156)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 20), 'assign_stmts', flatten_lists_call_result_14138)
                    
                    # Assigning a Name to a Name (line 157):
                    
                    # Assigning a Name to a Name (line 157):
                    # Getting the type of 'target' (line 157)
                    target_14139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 28), 'target')
                    # Assigning a type to the variable 'value' (line 157)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'value', target_14139)
                    # SSA join for if statement (line 153)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 160)
        # Processing the call arguments (line 160)
        # Getting the type of 'assign_stmts' (line 160)
        assign_stmts_14141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), 'assign_stmts', False)
        # Processing the call keyword arguments (line 160)
        kwargs_14142 = {}
        # Getting the type of 'len' (line 160)
        len_14140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'len', False)
        # Calling len(args, kwargs) (line 160)
        len_call_result_14143 = invoke(stypy.reporting.localization.Localization(__file__, 160, 11), len_14140, *[assign_stmts_14141], **kwargs_14142)
        
        int_14144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 31), 'int')
        # Applying the binary operator '>' (line 160)
        result_gt_14145 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 11), '>', len_call_result_14143, int_14144)
        
        # Testing if the type of an if condition is none (line 160)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 160, 8), result_gt_14145):
            pass
        else:
            
            # Testing the type of an if condition (line 160)
            if_condition_14146 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 8), result_gt_14145)
            # Assigning a type to the variable 'if_condition_14146' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'if_condition_14146', if_condition_14146)
            # SSA begins for if statement (line 160)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'assign_stmts' (line 161)
            assign_stmts_14147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 19), 'assign_stmts')
            # Assigning a type to the variable 'stypy_return_type' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'stypy_return_type', assign_stmts_14147)
            # SSA join for if statement (line 160)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'node' (line 162)
        node_14148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'stypy_return_type', node_14148)
        
        # ################# End of 'visit_Assign(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Assign' in the type store
        # Getting the type of 'stypy_return_type' (line 135)
        stypy_return_type_14149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14149)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Assign'
        return stypy_return_type_14149


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
dict_14150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 28), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 111)
# Adding element type (key, value) (line 111)

@norecursion
def _stypy_temp_lambda_20(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_20'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_20', 112, 9, True)
    # Passed parameters checking function
    _stypy_temp_lambda_20.stypy_localization = localization
    _stypy_temp_lambda_20.stypy_type_of_self = None
    _stypy_temp_lambda_20.stypy_type_store = module_type_store
    _stypy_temp_lambda_20.stypy_function_name = '_stypy_temp_lambda_20'
    _stypy_temp_lambda_20.stypy_param_names_list = ['target', 'value']
    _stypy_temp_lambda_20.stypy_varargs_param_name = None
    _stypy_temp_lambda_20.stypy_kwargs_param_name = None
    _stypy_temp_lambda_20.stypy_call_defaults = defaults
    _stypy_temp_lambda_20.stypy_call_varargs = varargs
    _stypy_temp_lambda_20.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_20', ['target', 'value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_20', ['target', 'value'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'target' (line 112)
    target_14152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 42), 'target', False)
    # Getting the type of 'ast' (line 112)
    ast_14153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 50), 'ast', False)
    # Obtaining the member 'Tuple' of a type (line 112)
    Tuple_14154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 50), ast_14153, 'Tuple')
    # Processing the call keyword arguments (line 112)
    kwargs_14155 = {}
    # Getting the type of 'isinstance' (line 112)
    isinstance_14151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 112)
    isinstance_call_result_14156 = invoke(stypy.reporting.localization.Localization(__file__, 112, 31), isinstance_14151, *[target_14152, Tuple_14154], **kwargs_14155)
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'value' (line 112)
    value_14158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 77), 'value', False)
    # Getting the type of 'ast' (line 112)
    ast_14159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 84), 'ast', False)
    # Obtaining the member 'Tuple' of a type (line 112)
    Tuple_14160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 84), ast_14159, 'Tuple')
    # Processing the call keyword arguments (line 112)
    kwargs_14161 = {}
    # Getting the type of 'isinstance' (line 112)
    isinstance_14157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 66), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 112)
    isinstance_call_result_14162 = invoke(stypy.reporting.localization.Localization(__file__, 112, 66), isinstance_14157, *[value_14158, Tuple_14160], **kwargs_14161)
    
    
    # Call to isinstance(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'value' (line 113)
    value_14164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 77), 'value', False)
    # Getting the type of 'ast' (line 113)
    ast_14165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 84), 'ast', False)
    # Obtaining the member 'List' of a type (line 113)
    List_14166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 84), ast_14165, 'List')
    # Processing the call keyword arguments (line 113)
    kwargs_14167 = {}
    # Getting the type of 'isinstance' (line 113)
    isinstance_14163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 66), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 113)
    isinstance_call_result_14168 = invoke(stypy.reporting.localization.Localization(__file__, 113, 66), isinstance_14163, *[value_14164, List_14166], **kwargs_14167)
    
    # Applying the binary operator 'or' (line 112)
    result_or_keyword_14169 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 66), 'or', isinstance_call_result_14162, isinstance_call_result_14168)
    
    # Applying the binary operator 'and' (line 112)
    result_and_keyword_14170 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 31), 'and', isinstance_call_result_14156, result_or_keyword_14169)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 9), 'stypy_return_type', result_and_keyword_14170)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_20' in the type store
    # Getting the type of 'stypy_return_type' (line 112)
    stypy_return_type_14171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 9), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14171)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_20'
    return stypy_return_type_14171

# Assigning a type to the variable '_stypy_temp_lambda_20' (line 112)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 9), '_stypy_temp_lambda_20', _stypy_temp_lambda_20)
# Getting the type of '_stypy_temp_lambda_20' (line 112)
_stypy_temp_lambda_20_14172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 9), '_stypy_temp_lambda_20')

# Obtaining an instance of the builtin type 'tuple' (line 114)
tuple_14173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 114)
# Adding element type (line 114)
str_14174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'str', 'tuple')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 12), tuple_14173, str_14174)
# Adding element type (line 114)
# Getting the type of 'multiple_value_assignment_handler' (line 114)
multiple_value_assignment_handler_14175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 21), 'multiple_value_assignment_handler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 12), tuple_14173, multiple_value_assignment_handler_14175)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 28), dict_14150, (_stypy_temp_lambda_20_14172, tuple_14173))
# Adding element type (key, value) (line 111)

@norecursion
def _stypy_temp_lambda_21(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_21'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_21', 116, 9, True)
    # Passed parameters checking function
    _stypy_temp_lambda_21.stypy_localization = localization
    _stypy_temp_lambda_21.stypy_type_of_self = None
    _stypy_temp_lambda_21.stypy_type_store = module_type_store
    _stypy_temp_lambda_21.stypy_function_name = '_stypy_temp_lambda_21'
    _stypy_temp_lambda_21.stypy_param_names_list = ['target', 'value']
    _stypy_temp_lambda_21.stypy_varargs_param_name = None
    _stypy_temp_lambda_21.stypy_kwargs_param_name = None
    _stypy_temp_lambda_21.stypy_call_defaults = defaults
    _stypy_temp_lambda_21.stypy_call_varargs = varargs
    _stypy_temp_lambda_21.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_21', ['target', 'value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_21', ['target', 'value'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'target' (line 116)
    target_14177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 42), 'target', False)
    # Getting the type of 'ast' (line 116)
    ast_14178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 50), 'ast', False)
    # Obtaining the member 'List' of a type (line 116)
    List_14179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 50), ast_14178, 'List')
    # Processing the call keyword arguments (line 116)
    kwargs_14180 = {}
    # Getting the type of 'isinstance' (line 116)
    isinstance_14176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 116)
    isinstance_call_result_14181 = invoke(stypy.reporting.localization.Localization(__file__, 116, 31), isinstance_14176, *[target_14177, List_14179], **kwargs_14180)
    
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'value' (line 116)
    value_14183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 76), 'value', False)
    # Getting the type of 'ast' (line 116)
    ast_14184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 83), 'ast', False)
    # Obtaining the member 'Tuple' of a type (line 116)
    Tuple_14185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 83), ast_14184, 'Tuple')
    # Processing the call keyword arguments (line 116)
    kwargs_14186 = {}
    # Getting the type of 'isinstance' (line 116)
    isinstance_14182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 65), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 116)
    isinstance_call_result_14187 = invoke(stypy.reporting.localization.Localization(__file__, 116, 65), isinstance_14182, *[value_14183, Tuple_14185], **kwargs_14186)
    
    
    # Call to isinstance(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'value' (line 117)
    value_14189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 76), 'value', False)
    # Getting the type of 'ast' (line 117)
    ast_14190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 83), 'ast', False)
    # Obtaining the member 'List' of a type (line 117)
    List_14191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 83), ast_14190, 'List')
    # Processing the call keyword arguments (line 117)
    kwargs_14192 = {}
    # Getting the type of 'isinstance' (line 117)
    isinstance_14188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 65), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 117)
    isinstance_call_result_14193 = invoke(stypy.reporting.localization.Localization(__file__, 117, 65), isinstance_14188, *[value_14189, List_14191], **kwargs_14192)
    
    # Applying the binary operator 'or' (line 116)
    result_or_keyword_14194 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 65), 'or', isinstance_call_result_14187, isinstance_call_result_14193)
    
    # Applying the binary operator 'and' (line 116)
    result_and_keyword_14195 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 31), 'and', isinstance_call_result_14181, result_or_keyword_14194)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), 'stypy_return_type', result_and_keyword_14195)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_21' in the type store
    # Getting the type of 'stypy_return_type' (line 116)
    stypy_return_type_14196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14196)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_21'
    return stypy_return_type_14196

# Assigning a type to the variable '_stypy_temp_lambda_21' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), '_stypy_temp_lambda_21', _stypy_temp_lambda_21)
# Getting the type of '_stypy_temp_lambda_21' (line 116)
_stypy_temp_lambda_21_14197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), '_stypy_temp_lambda_21')

# Obtaining an instance of the builtin type 'tuple' (line 118)
tuple_14198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 118)
# Adding element type (line 118)
str_14199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 12), 'str', 'list')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 12), tuple_14198, str_14199)
# Adding element type (line 118)
# Getting the type of 'multiple_value_assignment_handler' (line 118)
multiple_value_assignment_handler_14200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'multiple_value_assignment_handler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 12), tuple_14198, multiple_value_assignment_handler_14200)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 28), dict_14150, (_stypy_temp_lambda_21_14197, tuple_14198))
# Adding element type (key, value) (line 111)

@norecursion
def _stypy_temp_lambda_22(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_22'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_22', 120, 9, True)
    # Passed parameters checking function
    _stypy_temp_lambda_22.stypy_localization = localization
    _stypy_temp_lambda_22.stypy_type_of_self = None
    _stypy_temp_lambda_22.stypy_type_store = module_type_store
    _stypy_temp_lambda_22.stypy_function_name = '_stypy_temp_lambda_22'
    _stypy_temp_lambda_22.stypy_param_names_list = ['target', 'value']
    _stypy_temp_lambda_22.stypy_varargs_param_name = None
    _stypy_temp_lambda_22.stypy_kwargs_param_name = None
    _stypy_temp_lambda_22.stypy_call_defaults = defaults
    _stypy_temp_lambda_22.stypy_call_varargs = varargs
    _stypy_temp_lambda_22.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_22', ['target', 'value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_22', ['target', 'value'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'target' (line 120)
    target_14202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 43), 'target', False)
    # Getting the type of 'ast' (line 120)
    ast_14203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 51), 'ast', False)
    # Obtaining the member 'List' of a type (line 120)
    List_14204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 51), ast_14203, 'List')
    # Processing the call keyword arguments (line 120)
    kwargs_14205 = {}
    # Getting the type of 'isinstance' (line 120)
    isinstance_14201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 32), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 120)
    isinstance_call_result_14206 = invoke(stypy.reporting.localization.Localization(__file__, 120, 32), isinstance_14201, *[target_14202, List_14204], **kwargs_14205)
    
    
    # Call to isinstance(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'target' (line 120)
    target_14208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 75), 'target', False)
    # Getting the type of 'ast' (line 120)
    ast_14209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 83), 'ast', False)
    # Obtaining the member 'Tuple' of a type (line 120)
    Tuple_14210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 83), ast_14209, 'Tuple')
    # Processing the call keyword arguments (line 120)
    kwargs_14211 = {}
    # Getting the type of 'isinstance' (line 120)
    isinstance_14207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 64), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 120)
    isinstance_call_result_14212 = invoke(stypy.reporting.localization.Localization(__file__, 120, 64), isinstance_14207, *[target_14208, Tuple_14210], **kwargs_14211)
    
    # Applying the binary operator 'or' (line 120)
    result_or_keyword_14213 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 32), 'or', isinstance_call_result_14206, isinstance_call_result_14212)
    
    
    # Call to isinstance(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'value' (line 121)
    value_14215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'value', False)
    # Getting the type of 'ast' (line 121)
    ast_14216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 30), 'ast', False)
    # Obtaining the member 'Call' of a type (line 121)
    Call_14217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 30), ast_14216, 'Call')
    # Processing the call keyword arguments (line 121)
    kwargs_14218 = {}
    # Getting the type of 'isinstance' (line 121)
    isinstance_14214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 121)
    isinstance_call_result_14219 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), isinstance_14214, *[value_14215, Call_14217], **kwargs_14218)
    
    # Applying the binary operator 'and' (line 120)
    result_and_keyword_14220 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 31), 'and', result_or_keyword_14213, isinstance_call_result_14219)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 9), 'stypy_return_type', result_and_keyword_14220)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_22' in the type store
    # Getting the type of 'stypy_return_type' (line 120)
    stypy_return_type_14221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 9), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14221)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_22'
    return stypy_return_type_14221

# Assigning a type to the variable '_stypy_temp_lambda_22' (line 120)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 9), '_stypy_temp_lambda_22', _stypy_temp_lambda_22)
# Getting the type of '_stypy_temp_lambda_22' (line 120)
_stypy_temp_lambda_22_14222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 9), '_stypy_temp_lambda_22')

# Obtaining an instance of the builtin type 'tuple' (line 121)
tuple_14223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 44), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 121)
# Adding element type (line 121)
str_14224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 44), 'str', 'call')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 44), tuple_14223, str_14224)
# Adding element type (line 121)
# Getting the type of 'multiple_value_call_assignment_handler' (line 121)
multiple_value_call_assignment_handler_14225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 52), 'multiple_value_call_assignment_handler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 44), tuple_14223, multiple_value_call_assignment_handler_14225)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 28), dict_14150, (_stypy_temp_lambda_22_14222, tuple_14223))
# Adding element type (key, value) (line 111)

@norecursion
def _stypy_temp_lambda_23(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_23'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_23', 123, 8, True)
    # Passed parameters checking function
    _stypy_temp_lambda_23.stypy_localization = localization
    _stypy_temp_lambda_23.stypy_type_of_self = None
    _stypy_temp_lambda_23.stypy_type_store = module_type_store
    _stypy_temp_lambda_23.stypy_function_name = '_stypy_temp_lambda_23'
    _stypy_temp_lambda_23.stypy_param_names_list = ['target', 'value']
    _stypy_temp_lambda_23.stypy_varargs_param_name = None
    _stypy_temp_lambda_23.stypy_kwargs_param_name = None
    _stypy_temp_lambda_23.stypy_call_defaults = defaults
    _stypy_temp_lambda_23.stypy_call_varargs = varargs
    _stypy_temp_lambda_23.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_23', ['target', 'value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_23', ['target', 'value'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to isinstance(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'target' (line 123)
    target_14227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 41), 'target', False)
    # Getting the type of 'ast' (line 123)
    ast_14228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 49), 'ast', False)
    # Obtaining the member 'Name' of a type (line 123)
    Name_14229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 49), ast_14228, 'Name')
    # Processing the call keyword arguments (line 123)
    kwargs_14230 = {}
    # Getting the type of 'isinstance' (line 123)
    isinstance_14226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 30), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 123)
    isinstance_call_result_14231 = invoke(stypy.reporting.localization.Localization(__file__, 123, 30), isinstance_14226, *[target_14227, Name_14229], **kwargs_14230)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type', isinstance_call_result_14231)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_23' in the type store
    # Getting the type of 'stypy_return_type' (line 123)
    stypy_return_type_14232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14232)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_23'
    return stypy_return_type_14232

# Assigning a type to the variable '_stypy_temp_lambda_23' (line 123)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), '_stypy_temp_lambda_23', _stypy_temp_lambda_23)
# Getting the type of '_stypy_temp_lambda_23' (line 123)
_stypy_temp_lambda_23_14233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), '_stypy_temp_lambda_23')

# Obtaining an instance of the builtin type 'tuple' (line 124)
tuple_14234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 124)
# Adding element type (line 124)
str_14235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 13), 'str', 'assignment')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 13), tuple_14234, str_14235)
# Adding element type (line 124)
# Getting the type of 'single_assignment_handler' (line 124)
single_assignment_handler_14236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'single_assignment_handler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 13), tuple_14234, single_assignment_handler_14236)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 28), dict_14150, (_stypy_temp_lambda_23_14233, tuple_14234))
# Adding element type (key, value) (line 111)

@norecursion
def _stypy_temp_lambda_24(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_24'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_24', 126, 8, True)
    # Passed parameters checking function
    _stypy_temp_lambda_24.stypy_localization = localization
    _stypy_temp_lambda_24.stypy_type_of_self = None
    _stypy_temp_lambda_24.stypy_type_store = module_type_store
    _stypy_temp_lambda_24.stypy_function_name = '_stypy_temp_lambda_24'
    _stypy_temp_lambda_24.stypy_param_names_list = ['target', 'value']
    _stypy_temp_lambda_24.stypy_varargs_param_name = None
    _stypy_temp_lambda_24.stypy_kwargs_param_name = None
    _stypy_temp_lambda_24.stypy_call_defaults = defaults
    _stypy_temp_lambda_24.stypy_call_varargs = varargs
    _stypy_temp_lambda_24.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_24', ['target', 'value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_24', ['target', 'value'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to isinstance(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'target' (line 126)
    target_14238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 41), 'target', False)
    # Getting the type of 'ast' (line 126)
    ast_14239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 49), 'ast', False)
    # Obtaining the member 'Subscript' of a type (line 126)
    Subscript_14240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 49), ast_14239, 'Subscript')
    # Processing the call keyword arguments (line 126)
    kwargs_14241 = {}
    # Getting the type of 'isinstance' (line 126)
    isinstance_14237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 30), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 126)
    isinstance_call_result_14242 = invoke(stypy.reporting.localization.Localization(__file__, 126, 30), isinstance_14237, *[target_14238, Subscript_14240], **kwargs_14241)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type', isinstance_call_result_14242)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_24' in the type store
    # Getting the type of 'stypy_return_type' (line 126)
    stypy_return_type_14243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14243)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_24'
    return stypy_return_type_14243

# Assigning a type to the variable '_stypy_temp_lambda_24' (line 126)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), '_stypy_temp_lambda_24', _stypy_temp_lambda_24)
# Getting the type of '_stypy_temp_lambda_24' (line 126)
_stypy_temp_lambda_24_14244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), '_stypy_temp_lambda_24')

# Obtaining an instance of the builtin type 'tuple' (line 127)
tuple_14245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 127)
# Adding element type (line 127)
str_14246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 13), 'str', 'assignment')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 13), tuple_14245, str_14246)
# Adding element type (line 127)
# Getting the type of 'single_assignment_handler' (line 127)
single_assignment_handler_14247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 27), 'single_assignment_handler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 13), tuple_14245, single_assignment_handler_14247)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 28), dict_14150, (_stypy_temp_lambda_24_14244, tuple_14245))
# Adding element type (key, value) (line 111)

@norecursion
def _stypy_temp_lambda_25(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_25'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_25', 129, 8, True)
    # Passed parameters checking function
    _stypy_temp_lambda_25.stypy_localization = localization
    _stypy_temp_lambda_25.stypy_type_of_self = None
    _stypy_temp_lambda_25.stypy_type_store = module_type_store
    _stypy_temp_lambda_25.stypy_function_name = '_stypy_temp_lambda_25'
    _stypy_temp_lambda_25.stypy_param_names_list = ['target', 'value']
    _stypy_temp_lambda_25.stypy_varargs_param_name = None
    _stypy_temp_lambda_25.stypy_kwargs_param_name = None
    _stypy_temp_lambda_25.stypy_call_defaults = defaults
    _stypy_temp_lambda_25.stypy_call_varargs = varargs
    _stypy_temp_lambda_25.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_25', ['target', 'value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_25', ['target', 'value'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to isinstance(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'target' (line 129)
    target_14249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 41), 'target', False)
    # Getting the type of 'ast' (line 129)
    ast_14250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 49), 'ast', False)
    # Obtaining the member 'Attribute' of a type (line 129)
    Attribute_14251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 49), ast_14250, 'Attribute')
    # Processing the call keyword arguments (line 129)
    kwargs_14252 = {}
    # Getting the type of 'isinstance' (line 129)
    isinstance_14248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 30), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 129)
    isinstance_call_result_14253 = invoke(stypy.reporting.localization.Localization(__file__, 129, 30), isinstance_14248, *[target_14249, Attribute_14251], **kwargs_14252)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type', isinstance_call_result_14253)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_25' in the type store
    # Getting the type of 'stypy_return_type' (line 129)
    stypy_return_type_14254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14254)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_25'
    return stypy_return_type_14254

# Assigning a type to the variable '_stypy_temp_lambda_25' (line 129)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), '_stypy_temp_lambda_25', _stypy_temp_lambda_25)
# Getting the type of '_stypy_temp_lambda_25' (line 129)
_stypy_temp_lambda_25_14255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), '_stypy_temp_lambda_25')

# Obtaining an instance of the builtin type 'tuple' (line 130)
tuple_14256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 130)
# Adding element type (line 130)
str_14257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 13), 'str', 'assignment')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 13), tuple_14256, str_14257)
# Adding element type (line 130)
# Getting the type of 'single_assignment_handler' (line 130)
single_assignment_handler_14258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 27), 'single_assignment_handler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 13), tuple_14256, single_assignment_handler_14258)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 28), dict_14150, (_stypy_temp_lambda_25_14255, tuple_14256))

# Getting the type of 'MultipleAssignmentsDesugaringVisitor'
MultipleAssignmentsDesugaringVisitor_14259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'MultipleAssignmentsDesugaringVisitor')
# Setting the type of the member '__assignment_handlers' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), MultipleAssignmentsDesugaringVisitor_14259, '__assignment_handlers', dict_14150)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
