
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import ast
2: import types
3: 
4: import stypy_functions_copy
5: import functions_copy
6: import core_language_copy
7: 
8: '''
9: Code that deals with various code idioms that can be optimized to better obtain the types of the variables used
10: on these idioms. The work with this file is unfinished, as not all the intended idioms are supported.
11: 
12: TODO: Finish this and its comments when idioms are fully implemented
13: '''
14: 
15: # Idiom constant names:
16: 
17: # Idiom identified, var of the type call, type
18: default_ret_tuple = False, None, None
19: 
20: may_be_type_func_name = "may_be_type"
21: may_not_be_type_func_name = "may_not_be_type"
22: may_be_var_name = "__may_be"
23: more_types_var_name = "__more_types_in_union"
24: 
25: 
26: def __has_call_to_type_builtin(test):
27:     if type(test) is ast.Call:
28:         if type(test.func) is ast.Name:
29:             if len(test.args) != 1:
30:                 return False
31:             if test.func.id == "type":
32:                 return True
33:     else:
34:         if hasattr(test, "left"):
35:             if type(test.left) is ast.Call:
36:                 if len(test.comparators) != 1:
37:                     return False
38:                 if type(test.left.func) is ast.Name:
39:                     if len(test.left.args) != 1:
40:                         return False
41:                     if test.left.func.id == "type":
42:                         return True
43:     return False
44: 
45: 
46: def __has_call_to_is(test):
47:     if len(test.ops) == 1:
48:         if type(test.ops[0]) is ast.Is:
49:             return True
50:     return False
51: 
52: 
53: def __is_type_name(test):
54:     if type(test) is ast.Name:
55:         name_id = test.id
56:         try:
57:             type_obj = eval(name_id)
58:             return type(type_obj) is types.TypeType
59:         except:
60:             return False
61:     return False
62: 
63: 
64: def type_is_idiom(test, visitor, context):
65:     '''
66:     Idiom "type is"
67:     :param test:
68:     :param visitor:
69:     :param context:
70:     :return:
71:     '''
72:     if type(test) is not ast.Compare:
73:         return default_ret_tuple
74: 
75:     if __has_call_to_type_builtin(test) and __has_call_to_is(test):
76:         if not (__has_call_to_type_builtin(test.comparators[0]) or __is_type_name(test.comparators[0])):
77:             return default_ret_tuple
78:         type_param = visitor.visit(test.left.args[0], context)
79:         if __is_type_name(test.comparators[0]):
80:             is_operator = visitor.visit(test.comparators[0], context)
81:         else:
82:             is_operator = visitor.visit(test.comparators[0].args[0], context)
83:             if not isinstance(is_operator[0], list):
84:                 is_operator = ([is_operator[0]], is_operator[1])
85: 
86:         return True, type_param, is_operator
87: 
88:     return default_ret_tuple
89: 
90: 
91: def not_type_is_idiom(test, visitor, context):
92:     '''
93:     Idiom "not type is"
94: 
95:     :param test:
96:     :param visitor:
97:     :param context:
98:     :return:
99:     '''
100:     if type(test) is not ast.UnaryOp:
101:         return default_ret_tuple
102:     if type(test.op) is not ast.Not:
103:         return default_ret_tuple
104: 
105:     return type_is_idiom(test.operand, visitor, context)
106: 
107: 
108: def __get_idiom_type_param(test):
109:     return test.left.args[0]
110: 
111: 
112: def __set_type_implementation(if_test, type_, lineno, col_offset):
113:     param = __get_idiom_type_param(if_test)
114:     if type(param) is ast.Name:
115:         return stypy_functions_copy.create_set_type_of(param.id, type_, lineno, col_offset)
116:     if type(param) is ast.Attribute:
117:         obj_type, obj_var = stypy_functions_copy.create_get_type_of(param.value.id, lineno, col_offset)
118:         set_member = stypy_functions_copy.create_set_type_of_member(obj_var, param.attr, type_, lineno, col_offset)
119:         return stypy_functions_copy.flatten_lists(obj_type, set_member)
120: 
121:     return []
122: 
123: 
124: def __remove_type_from_union_implementation(if_test, type_, lineno, col_offset):
125:     param = __get_idiom_type_param(if_test)
126:     if type(param) is ast.Name:
127:         obj_type, obj_var = stypy_functions_copy.create_get_type_of(param.id, lineno, col_offset)
128:         remove_type_call = functions_copy.create_call(core_language_copy.create_Name("remove_type_from_union"),
129:                                                  [obj_var, type_], line=lineno, column=col_offset)
130:         set_type = stypy_functions_copy.create_set_type_of(param.id, remove_type_call, lineno, col_offset)
131: 
132:         return stypy_functions_copy.flatten_lists(obj_type, set_type)
133:     if type(param) is ast.Attribute:
134:         # Get the owner of the attribute
135:         obj_type_stmts, obj_var = stypy_functions_copy.create_get_type_of(param.value.id, lineno, col_offset)
136:         # Get the current type of the owner of the attribute
137:         att_type_stmts, att_var = stypy_functions_copy.create_get_type_of_member(obj_var, param.attr, lineno, col_offset)
138:         remove_type_call = functions_copy.create_call(core_language_copy.create_Name("remove_type_from_union"),
139:                                                  [att_var, type_], line=lineno, column=col_offset)
140:         set_member = stypy_functions_copy.create_set_type_of_member(obj_var, param.attr, remove_type_call, lineno,
141:                                                                col_offset)
142:         return stypy_functions_copy.flatten_lists(obj_type_stmts, att_type_stmts, set_member)
143: 
144:     return []
145: 
146: 
147: def set_type_of_idiom_var(idiom_name, if_branch, if_test, type_, lineno, col_offset):
148:     if idiom_name == "type_is":
149:         if if_branch == "if":
150:             return __set_type_implementation(if_test, type_, lineno, col_offset)
151:         if if_branch == "else":
152:             return __remove_type_from_union_implementation(if_test, type_, lineno, col_offset)
153: 
154:     if idiom_name == "not_type_is":
155:         if_test = if_test.operand
156:         if if_branch == "if":
157:             return __remove_type_from_union_implementation(if_test, type_, lineno, col_offset)
158:         if if_branch == "else":
159:             return __set_type_implementation(if_test, type_, lineno, col_offset)
160: 
161:     return []
162: 
163: 
164: # Recognized idioms
165: recognized_idioms = {
166:     "type_is": type_is_idiom,
167:     "not_type_is": not_type_is_idiom,
168: }
169: 
170: # Implementation of recognized idioms
171: recognized_idioms_functions = {
172:     "type_is": may_be_type_func_name,
173:     "not_type_is": may_not_be_type_func_name,
174: }
175: 
176: 
177: def get_recognized_idiom_function(idiom_name):
178:     '''
179:     Gets the function that process an idiom once it has been recognized
180:     :param idiom_name: Idiom name
181:     :return:
182:     '''
183:     return recognized_idioms_functions[idiom_name]
184: 
185: 
186: def is_recognized_idiom(test, visitor, context):
187:     '''
188:     Check if the passed test can be considered an idioms
189: 
190:     :param test: Source code test
191:     :param visitor: Type inference visitor, to change generated instructions
192:     :param context: Context passed to the call
193:     :return: Tuple of values that identify if an idiom has been recognized and calculated data if it is been recognized
194:     '''
195:     for idiom in recognized_idioms:
196:         result = recognized_idioms[idiom](test, visitor, context)
197:         if result[0]:
198:             temp_list = list(result)
199:             temp_list.append(idiom)
200:             return tuple(temp_list)
201: 
202:     return False, None, None, None
203: 

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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import types' statement (line 2)
import types

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import stypy_functions_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_31726 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_functions_copy')

if (type(import_31726) is not StypyTypeError):

    if (import_31726 != 'pyd_module'):
        __import__(import_31726)
        sys_modules_31727 = sys.modules[import_31726]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_functions_copy', sys_modules_31727.module_type_store, module_type_store)
    else:
        import stypy_functions_copy

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_functions_copy', stypy_functions_copy, module_type_store)

else:
    # Assigning a type to the variable 'stypy_functions_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_functions_copy', import_31726)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import functions_copy' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_31728 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'functions_copy')

if (type(import_31728) is not StypyTypeError):

    if (import_31728 != 'pyd_module'):
        __import__(import_31728)
        sys_modules_31729 = sys.modules[import_31728]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'functions_copy', sys_modules_31729.module_type_store, module_type_store)
    else:
        import functions_copy

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'functions_copy', functions_copy, module_type_store)

else:
    # Assigning a type to the variable 'functions_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'functions_copy', import_31728)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import core_language_copy' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_31730 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'core_language_copy')

if (type(import_31730) is not StypyTypeError):

    if (import_31730 != 'pyd_module'):
        __import__(import_31730)
        sys_modules_31731 = sys.modules[import_31730]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'core_language_copy', sys_modules_31731.module_type_store, module_type_store)
    else:
        import core_language_copy

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'core_language_copy', core_language_copy, module_type_store)

else:
    # Assigning a type to the variable 'core_language_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'core_language_copy', import_31730)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

str_31732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'str', '\nCode that deals with various code idioms that can be optimized to better obtain the types of the variables used\non these idioms. The work with this file is unfinished, as not all the intended idioms are supported.\n\nTODO: Finish this and its comments when idioms are fully implemented\n')

# Assigning a Tuple to a Name (line 18):

# Assigning a Tuple to a Name (line 18):

# Obtaining an instance of the builtin type 'tuple' (line 18)
tuple_31733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 18)
# Adding element type (line 18)
# Getting the type of 'False' (line 18)
False_31734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 20), tuple_31733, False_31734)
# Adding element type (line 18)
# Getting the type of 'None' (line 18)
None_31735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 27), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 20), tuple_31733, None_31735)
# Adding element type (line 18)
# Getting the type of 'None' (line 18)
None_31736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 33), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 20), tuple_31733, None_31736)

# Assigning a type to the variable 'default_ret_tuple' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'default_ret_tuple', tuple_31733)

# Assigning a Str to a Name (line 20):

# Assigning a Str to a Name (line 20):
str_31737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'str', 'may_be_type')
# Assigning a type to the variable 'may_be_type_func_name' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'may_be_type_func_name', str_31737)

# Assigning a Str to a Name (line 21):

# Assigning a Str to a Name (line 21):
str_31738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'str', 'may_not_be_type')
# Assigning a type to the variable 'may_not_be_type_func_name' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'may_not_be_type_func_name', str_31738)

# Assigning a Str to a Name (line 22):

# Assigning a Str to a Name (line 22):
str_31739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'str', '__may_be')
# Assigning a type to the variable 'may_be_var_name' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'may_be_var_name', str_31739)

# Assigning a Str to a Name (line 23):

# Assigning a Str to a Name (line 23):
str_31740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 22), 'str', '__more_types_in_union')
# Assigning a type to the variable 'more_types_var_name' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'more_types_var_name', str_31740)

@norecursion
def __has_call_to_type_builtin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__has_call_to_type_builtin'
    module_type_store = module_type_store.open_function_context('__has_call_to_type_builtin', 26, 0, False)
    
    # Passed parameters checking function
    __has_call_to_type_builtin.stypy_localization = localization
    __has_call_to_type_builtin.stypy_type_of_self = None
    __has_call_to_type_builtin.stypy_type_store = module_type_store
    __has_call_to_type_builtin.stypy_function_name = '__has_call_to_type_builtin'
    __has_call_to_type_builtin.stypy_param_names_list = ['test']
    __has_call_to_type_builtin.stypy_varargs_param_name = None
    __has_call_to_type_builtin.stypy_kwargs_param_name = None
    __has_call_to_type_builtin.stypy_call_defaults = defaults
    __has_call_to_type_builtin.stypy_call_varargs = varargs
    __has_call_to_type_builtin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__has_call_to_type_builtin', ['test'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__has_call_to_type_builtin', localization, ['test'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__has_call_to_type_builtin(...)' code ##################

    
    
    # Call to type(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'test' (line 27)
    test_31742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'test', False)
    # Processing the call keyword arguments (line 27)
    kwargs_31743 = {}
    # Getting the type of 'type' (line 27)
    type_31741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 7), 'type', False)
    # Calling type(args, kwargs) (line 27)
    type_call_result_31744 = invoke(stypy.reporting.localization.Localization(__file__, 27, 7), type_31741, *[test_31742], **kwargs_31743)
    
    # Getting the type of 'ast' (line 27)
    ast_31745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'ast')
    # Obtaining the member 'Call' of a type (line 27)
    Call_31746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 21), ast_31745, 'Call')
    # Applying the binary operator 'is' (line 27)
    result_is__31747 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 7), 'is', type_call_result_31744, Call_31746)
    
    # Testing if the type of an if condition is none (line 27)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 27, 4), result_is__31747):
        
        # Type idiom detected: calculating its left and rigth part (line 34)
        str_31774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'str', 'left')
        # Getting the type of 'test' (line 34)
        test_31775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'test')
        
        (may_be_31776, more_types_in_union_31777) = may_provide_member(str_31774, test_31775)

        if may_be_31776:

            if more_types_in_union_31777:
                # Runtime conditional SSA (line 34)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'test' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'test', remove_not_member_provider_from_union(test_31775, 'left'))
            
            
            # Call to type(...): (line 35)
            # Processing the call arguments (line 35)
            # Getting the type of 'test' (line 35)
            test_31779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'test', False)
            # Obtaining the member 'left' of a type (line 35)
            left_31780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 20), test_31779, 'left')
            # Processing the call keyword arguments (line 35)
            kwargs_31781 = {}
            # Getting the type of 'type' (line 35)
            type_31778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'type', False)
            # Calling type(args, kwargs) (line 35)
            type_call_result_31782 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), type_31778, *[left_31780], **kwargs_31781)
            
            # Getting the type of 'ast' (line 35)
            ast_31783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'ast')
            # Obtaining the member 'Call' of a type (line 35)
            Call_31784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 34), ast_31783, 'Call')
            # Applying the binary operator 'is' (line 35)
            result_is__31785 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 15), 'is', type_call_result_31782, Call_31784)
            
            # Testing if the type of an if condition is none (line 35)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 35, 12), result_is__31785):
                pass
            else:
                
                # Testing the type of an if condition (line 35)
                if_condition_31786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 12), result_is__31785)
                # Assigning a type to the variable 'if_condition_31786' (line 35)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'if_condition_31786', if_condition_31786)
                # SSA begins for if statement (line 35)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Call to len(...): (line 36)
                # Processing the call arguments (line 36)
                # Getting the type of 'test' (line 36)
                test_31788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'test', False)
                # Obtaining the member 'comparators' of a type (line 36)
                comparators_31789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 23), test_31788, 'comparators')
                # Processing the call keyword arguments (line 36)
                kwargs_31790 = {}
                # Getting the type of 'len' (line 36)
                len_31787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'len', False)
                # Calling len(args, kwargs) (line 36)
                len_call_result_31791 = invoke(stypy.reporting.localization.Localization(__file__, 36, 19), len_31787, *[comparators_31789], **kwargs_31790)
                
                int_31792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 44), 'int')
                # Applying the binary operator '!=' (line 36)
                result_ne_31793 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 19), '!=', len_call_result_31791, int_31792)
                
                # Testing if the type of an if condition is none (line 36)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 36, 16), result_ne_31793):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 36)
                    if_condition_31794 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 16), result_ne_31793)
                    # Assigning a type to the variable 'if_condition_31794' (line 36)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'if_condition_31794', if_condition_31794)
                    # SSA begins for if statement (line 36)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 37)
                    False_31795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 37)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'stypy_return_type', False_31795)
                    # SSA join for if statement (line 36)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                
                # Call to type(...): (line 38)
                # Processing the call arguments (line 38)
                # Getting the type of 'test' (line 38)
                test_31797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'test', False)
                # Obtaining the member 'left' of a type (line 38)
                left_31798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 24), test_31797, 'left')
                # Obtaining the member 'func' of a type (line 38)
                func_31799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 24), left_31798, 'func')
                # Processing the call keyword arguments (line 38)
                kwargs_31800 = {}
                # Getting the type of 'type' (line 38)
                type_31796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'type', False)
                # Calling type(args, kwargs) (line 38)
                type_call_result_31801 = invoke(stypy.reporting.localization.Localization(__file__, 38, 19), type_31796, *[func_31799], **kwargs_31800)
                
                # Getting the type of 'ast' (line 38)
                ast_31802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 43), 'ast')
                # Obtaining the member 'Name' of a type (line 38)
                Name_31803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 43), ast_31802, 'Name')
                # Applying the binary operator 'is' (line 38)
                result_is__31804 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 19), 'is', type_call_result_31801, Name_31803)
                
                # Testing if the type of an if condition is none (line 38)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 38, 16), result_is__31804):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 38)
                    if_condition_31805 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 16), result_is__31804)
                    # Assigning a type to the variable 'if_condition_31805' (line 38)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'if_condition_31805', if_condition_31805)
                    # SSA begins for if statement (line 38)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    
                    # Call to len(...): (line 39)
                    # Processing the call arguments (line 39)
                    # Getting the type of 'test' (line 39)
                    test_31807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'test', False)
                    # Obtaining the member 'left' of a type (line 39)
                    left_31808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 27), test_31807, 'left')
                    # Obtaining the member 'args' of a type (line 39)
                    args_31809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 27), left_31808, 'args')
                    # Processing the call keyword arguments (line 39)
                    kwargs_31810 = {}
                    # Getting the type of 'len' (line 39)
                    len_31806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'len', False)
                    # Calling len(args, kwargs) (line 39)
                    len_call_result_31811 = invoke(stypy.reporting.localization.Localization(__file__, 39, 23), len_31806, *[args_31809], **kwargs_31810)
                    
                    int_31812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 46), 'int')
                    # Applying the binary operator '!=' (line 39)
                    result_ne_31813 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 23), '!=', len_call_result_31811, int_31812)
                    
                    # Testing if the type of an if condition is none (line 39)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 39, 20), result_ne_31813):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 39)
                        if_condition_31814 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 20), result_ne_31813)
                        # Assigning a type to the variable 'if_condition_31814' (line 39)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'if_condition_31814', if_condition_31814)
                        # SSA begins for if statement (line 39)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'False' (line 40)
                        False_31815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 31), 'False')
                        # Assigning a type to the variable 'stypy_return_type' (line 40)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'stypy_return_type', False_31815)
                        # SSA join for if statement (line 39)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Getting the type of 'test' (line 41)
                    test_31816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'test')
                    # Obtaining the member 'left' of a type (line 41)
                    left_31817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), test_31816, 'left')
                    # Obtaining the member 'func' of a type (line 41)
                    func_31818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), left_31817, 'func')
                    # Obtaining the member 'id' of a type (line 41)
                    id_31819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), func_31818, 'id')
                    str_31820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 44), 'str', 'type')
                    # Applying the binary operator '==' (line 41)
                    result_eq_31821 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 23), '==', id_31819, str_31820)
                    
                    # Testing if the type of an if condition is none (line 41)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 41, 20), result_eq_31821):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 41)
                        if_condition_31822 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 20), result_eq_31821)
                        # Assigning a type to the variable 'if_condition_31822' (line 41)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'if_condition_31822', if_condition_31822)
                        # SSA begins for if statement (line 41)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'True' (line 42)
                        True_31823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 31), 'True')
                        # Assigning a type to the variable 'stypy_return_type' (line 42)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'stypy_return_type', True_31823)
                        # SSA join for if statement (line 41)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 38)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 35)
                module_type_store = module_type_store.join_ssa_context()
                


            if more_types_in_union_31777:
                # SSA join for if statement (line 34)
                module_type_store = module_type_store.join_ssa_context()


        
    else:
        
        # Testing the type of an if condition (line 27)
        if_condition_31748 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 4), result_is__31747)
        # Assigning a type to the variable 'if_condition_31748' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'if_condition_31748', if_condition_31748)
        # SSA begins for if statement (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to type(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'test' (line 28)
        test_31750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'test', False)
        # Obtaining the member 'func' of a type (line 28)
        func_31751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), test_31750, 'func')
        # Processing the call keyword arguments (line 28)
        kwargs_31752 = {}
        # Getting the type of 'type' (line 28)
        type_31749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'type', False)
        # Calling type(args, kwargs) (line 28)
        type_call_result_31753 = invoke(stypy.reporting.localization.Localization(__file__, 28, 11), type_31749, *[func_31751], **kwargs_31752)
        
        # Getting the type of 'ast' (line 28)
        ast_31754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'ast')
        # Obtaining the member 'Name' of a type (line 28)
        Name_31755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 30), ast_31754, 'Name')
        # Applying the binary operator 'is' (line 28)
        result_is__31756 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 11), 'is', type_call_result_31753, Name_31755)
        
        # Testing if the type of an if condition is none (line 28)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 28, 8), result_is__31756):
            pass
        else:
            
            # Testing the type of an if condition (line 28)
            if_condition_31757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 8), result_is__31756)
            # Assigning a type to the variable 'if_condition_31757' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'if_condition_31757', if_condition_31757)
            # SSA begins for if statement (line 28)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to len(...): (line 29)
            # Processing the call arguments (line 29)
            # Getting the type of 'test' (line 29)
            test_31759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'test', False)
            # Obtaining the member 'args' of a type (line 29)
            args_31760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 19), test_31759, 'args')
            # Processing the call keyword arguments (line 29)
            kwargs_31761 = {}
            # Getting the type of 'len' (line 29)
            len_31758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'len', False)
            # Calling len(args, kwargs) (line 29)
            len_call_result_31762 = invoke(stypy.reporting.localization.Localization(__file__, 29, 15), len_31758, *[args_31760], **kwargs_31761)
            
            int_31763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 33), 'int')
            # Applying the binary operator '!=' (line 29)
            result_ne_31764 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 15), '!=', len_call_result_31762, int_31763)
            
            # Testing if the type of an if condition is none (line 29)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 29, 12), result_ne_31764):
                pass
            else:
                
                # Testing the type of an if condition (line 29)
                if_condition_31765 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 12), result_ne_31764)
                # Assigning a type to the variable 'if_condition_31765' (line 29)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'if_condition_31765', if_condition_31765)
                # SSA begins for if statement (line 29)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'False' (line 30)
                False_31766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 30)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'stypy_return_type', False_31766)
                # SSA join for if statement (line 29)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'test' (line 31)
            test_31767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'test')
            # Obtaining the member 'func' of a type (line 31)
            func_31768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 15), test_31767, 'func')
            # Obtaining the member 'id' of a type (line 31)
            id_31769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 15), func_31768, 'id')
            str_31770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'str', 'type')
            # Applying the binary operator '==' (line 31)
            result_eq_31771 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 15), '==', id_31769, str_31770)
            
            # Testing if the type of an if condition is none (line 31)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 31, 12), result_eq_31771):
                pass
            else:
                
                # Testing the type of an if condition (line 31)
                if_condition_31772 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 12), result_eq_31771)
                # Assigning a type to the variable 'if_condition_31772' (line 31)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'if_condition_31772', if_condition_31772)
                # SSA begins for if statement (line 31)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'True' (line 32)
                True_31773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'True')
                # Assigning a type to the variable 'stypy_return_type' (line 32)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'stypy_return_type', True_31773)
                # SSA join for if statement (line 31)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 28)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the else part of an if statement (line 27)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 34)
        str_31774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'str', 'left')
        # Getting the type of 'test' (line 34)
        test_31775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'test')
        
        (may_be_31776, more_types_in_union_31777) = may_provide_member(str_31774, test_31775)

        if may_be_31776:

            if more_types_in_union_31777:
                # Runtime conditional SSA (line 34)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'test' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'test', remove_not_member_provider_from_union(test_31775, 'left'))
            
            
            # Call to type(...): (line 35)
            # Processing the call arguments (line 35)
            # Getting the type of 'test' (line 35)
            test_31779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'test', False)
            # Obtaining the member 'left' of a type (line 35)
            left_31780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 20), test_31779, 'left')
            # Processing the call keyword arguments (line 35)
            kwargs_31781 = {}
            # Getting the type of 'type' (line 35)
            type_31778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'type', False)
            # Calling type(args, kwargs) (line 35)
            type_call_result_31782 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), type_31778, *[left_31780], **kwargs_31781)
            
            # Getting the type of 'ast' (line 35)
            ast_31783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'ast')
            # Obtaining the member 'Call' of a type (line 35)
            Call_31784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 34), ast_31783, 'Call')
            # Applying the binary operator 'is' (line 35)
            result_is__31785 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 15), 'is', type_call_result_31782, Call_31784)
            
            # Testing if the type of an if condition is none (line 35)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 35, 12), result_is__31785):
                pass
            else:
                
                # Testing the type of an if condition (line 35)
                if_condition_31786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 12), result_is__31785)
                # Assigning a type to the variable 'if_condition_31786' (line 35)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'if_condition_31786', if_condition_31786)
                # SSA begins for if statement (line 35)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Call to len(...): (line 36)
                # Processing the call arguments (line 36)
                # Getting the type of 'test' (line 36)
                test_31788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'test', False)
                # Obtaining the member 'comparators' of a type (line 36)
                comparators_31789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 23), test_31788, 'comparators')
                # Processing the call keyword arguments (line 36)
                kwargs_31790 = {}
                # Getting the type of 'len' (line 36)
                len_31787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'len', False)
                # Calling len(args, kwargs) (line 36)
                len_call_result_31791 = invoke(stypy.reporting.localization.Localization(__file__, 36, 19), len_31787, *[comparators_31789], **kwargs_31790)
                
                int_31792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 44), 'int')
                # Applying the binary operator '!=' (line 36)
                result_ne_31793 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 19), '!=', len_call_result_31791, int_31792)
                
                # Testing if the type of an if condition is none (line 36)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 36, 16), result_ne_31793):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 36)
                    if_condition_31794 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 16), result_ne_31793)
                    # Assigning a type to the variable 'if_condition_31794' (line 36)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'if_condition_31794', if_condition_31794)
                    # SSA begins for if statement (line 36)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 37)
                    False_31795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 37)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'stypy_return_type', False_31795)
                    # SSA join for if statement (line 36)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                
                # Call to type(...): (line 38)
                # Processing the call arguments (line 38)
                # Getting the type of 'test' (line 38)
                test_31797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'test', False)
                # Obtaining the member 'left' of a type (line 38)
                left_31798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 24), test_31797, 'left')
                # Obtaining the member 'func' of a type (line 38)
                func_31799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 24), left_31798, 'func')
                # Processing the call keyword arguments (line 38)
                kwargs_31800 = {}
                # Getting the type of 'type' (line 38)
                type_31796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'type', False)
                # Calling type(args, kwargs) (line 38)
                type_call_result_31801 = invoke(stypy.reporting.localization.Localization(__file__, 38, 19), type_31796, *[func_31799], **kwargs_31800)
                
                # Getting the type of 'ast' (line 38)
                ast_31802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 43), 'ast')
                # Obtaining the member 'Name' of a type (line 38)
                Name_31803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 43), ast_31802, 'Name')
                # Applying the binary operator 'is' (line 38)
                result_is__31804 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 19), 'is', type_call_result_31801, Name_31803)
                
                # Testing if the type of an if condition is none (line 38)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 38, 16), result_is__31804):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 38)
                    if_condition_31805 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 16), result_is__31804)
                    # Assigning a type to the variable 'if_condition_31805' (line 38)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'if_condition_31805', if_condition_31805)
                    # SSA begins for if statement (line 38)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    
                    # Call to len(...): (line 39)
                    # Processing the call arguments (line 39)
                    # Getting the type of 'test' (line 39)
                    test_31807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'test', False)
                    # Obtaining the member 'left' of a type (line 39)
                    left_31808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 27), test_31807, 'left')
                    # Obtaining the member 'args' of a type (line 39)
                    args_31809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 27), left_31808, 'args')
                    # Processing the call keyword arguments (line 39)
                    kwargs_31810 = {}
                    # Getting the type of 'len' (line 39)
                    len_31806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'len', False)
                    # Calling len(args, kwargs) (line 39)
                    len_call_result_31811 = invoke(stypy.reporting.localization.Localization(__file__, 39, 23), len_31806, *[args_31809], **kwargs_31810)
                    
                    int_31812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 46), 'int')
                    # Applying the binary operator '!=' (line 39)
                    result_ne_31813 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 23), '!=', len_call_result_31811, int_31812)
                    
                    # Testing if the type of an if condition is none (line 39)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 39, 20), result_ne_31813):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 39)
                        if_condition_31814 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 20), result_ne_31813)
                        # Assigning a type to the variable 'if_condition_31814' (line 39)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'if_condition_31814', if_condition_31814)
                        # SSA begins for if statement (line 39)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'False' (line 40)
                        False_31815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 31), 'False')
                        # Assigning a type to the variable 'stypy_return_type' (line 40)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'stypy_return_type', False_31815)
                        # SSA join for if statement (line 39)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Getting the type of 'test' (line 41)
                    test_31816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'test')
                    # Obtaining the member 'left' of a type (line 41)
                    left_31817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), test_31816, 'left')
                    # Obtaining the member 'func' of a type (line 41)
                    func_31818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), left_31817, 'func')
                    # Obtaining the member 'id' of a type (line 41)
                    id_31819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), func_31818, 'id')
                    str_31820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 44), 'str', 'type')
                    # Applying the binary operator '==' (line 41)
                    result_eq_31821 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 23), '==', id_31819, str_31820)
                    
                    # Testing if the type of an if condition is none (line 41)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 41, 20), result_eq_31821):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 41)
                        if_condition_31822 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 20), result_eq_31821)
                        # Assigning a type to the variable 'if_condition_31822' (line 41)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'if_condition_31822', if_condition_31822)
                        # SSA begins for if statement (line 41)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'True' (line 42)
                        True_31823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 31), 'True')
                        # Assigning a type to the variable 'stypy_return_type' (line 42)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'stypy_return_type', True_31823)
                        # SSA join for if statement (line 41)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 38)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 35)
                module_type_store = module_type_store.join_ssa_context()
                


            if more_types_in_union_31777:
                # SSA join for if statement (line 34)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 27)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'False' (line 43)
    False_31824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type', False_31824)
    
    # ################# End of '__has_call_to_type_builtin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__has_call_to_type_builtin' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_31825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31825)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__has_call_to_type_builtin'
    return stypy_return_type_31825

# Assigning a type to the variable '__has_call_to_type_builtin' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '__has_call_to_type_builtin', __has_call_to_type_builtin)

@norecursion
def __has_call_to_is(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__has_call_to_is'
    module_type_store = module_type_store.open_function_context('__has_call_to_is', 46, 0, False)
    
    # Passed parameters checking function
    __has_call_to_is.stypy_localization = localization
    __has_call_to_is.stypy_type_of_self = None
    __has_call_to_is.stypy_type_store = module_type_store
    __has_call_to_is.stypy_function_name = '__has_call_to_is'
    __has_call_to_is.stypy_param_names_list = ['test']
    __has_call_to_is.stypy_varargs_param_name = None
    __has_call_to_is.stypy_kwargs_param_name = None
    __has_call_to_is.stypy_call_defaults = defaults
    __has_call_to_is.stypy_call_varargs = varargs
    __has_call_to_is.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__has_call_to_is', ['test'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__has_call_to_is', localization, ['test'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__has_call_to_is(...)' code ##################

    
    
    # Call to len(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'test' (line 47)
    test_31827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'test', False)
    # Obtaining the member 'ops' of a type (line 47)
    ops_31828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), test_31827, 'ops')
    # Processing the call keyword arguments (line 47)
    kwargs_31829 = {}
    # Getting the type of 'len' (line 47)
    len_31826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 7), 'len', False)
    # Calling len(args, kwargs) (line 47)
    len_call_result_31830 = invoke(stypy.reporting.localization.Localization(__file__, 47, 7), len_31826, *[ops_31828], **kwargs_31829)
    
    int_31831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 24), 'int')
    # Applying the binary operator '==' (line 47)
    result_eq_31832 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 7), '==', len_call_result_31830, int_31831)
    
    # Testing if the type of an if condition is none (line 47)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 47, 4), result_eq_31832):
        pass
    else:
        
        # Testing the type of an if condition (line 47)
        if_condition_31833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 4), result_eq_31832)
        # Assigning a type to the variable 'if_condition_31833' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'if_condition_31833', if_condition_31833)
        # SSA begins for if statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to type(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Obtaining the type of the subscript
        int_31835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 25), 'int')
        # Getting the type of 'test' (line 48)
        test_31836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'test', False)
        # Obtaining the member 'ops' of a type (line 48)
        ops_31837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 16), test_31836, 'ops')
        # Obtaining the member '__getitem__' of a type (line 48)
        getitem___31838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 16), ops_31837, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 48)
        subscript_call_result_31839 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), getitem___31838, int_31835)
        
        # Processing the call keyword arguments (line 48)
        kwargs_31840 = {}
        # Getting the type of 'type' (line 48)
        type_31834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'type', False)
        # Calling type(args, kwargs) (line 48)
        type_call_result_31841 = invoke(stypy.reporting.localization.Localization(__file__, 48, 11), type_31834, *[subscript_call_result_31839], **kwargs_31840)
        
        # Getting the type of 'ast' (line 48)
        ast_31842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 32), 'ast')
        # Obtaining the member 'Is' of a type (line 48)
        Is_31843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 32), ast_31842, 'Is')
        # Applying the binary operator 'is' (line 48)
        result_is__31844 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 11), 'is', type_call_result_31841, Is_31843)
        
        # Testing if the type of an if condition is none (line 48)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 48, 8), result_is__31844):
            pass
        else:
            
            # Testing the type of an if condition (line 48)
            if_condition_31845 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 8), result_is__31844)
            # Assigning a type to the variable 'if_condition_31845' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'if_condition_31845', if_condition_31845)
            # SSA begins for if statement (line 48)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'True' (line 49)
            True_31846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'stypy_return_type', True_31846)
            # SSA join for if statement (line 48)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 47)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'False' (line 50)
    False_31847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type', False_31847)
    
    # ################# End of '__has_call_to_is(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__has_call_to_is' in the type store
    # Getting the type of 'stypy_return_type' (line 46)
    stypy_return_type_31848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31848)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__has_call_to_is'
    return stypy_return_type_31848

# Assigning a type to the variable '__has_call_to_is' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), '__has_call_to_is', __has_call_to_is)

@norecursion
def __is_type_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__is_type_name'
    module_type_store = module_type_store.open_function_context('__is_type_name', 53, 0, False)
    
    # Passed parameters checking function
    __is_type_name.stypy_localization = localization
    __is_type_name.stypy_type_of_self = None
    __is_type_name.stypy_type_store = module_type_store
    __is_type_name.stypy_function_name = '__is_type_name'
    __is_type_name.stypy_param_names_list = ['test']
    __is_type_name.stypy_varargs_param_name = None
    __is_type_name.stypy_kwargs_param_name = None
    __is_type_name.stypy_call_defaults = defaults
    __is_type_name.stypy_call_varargs = varargs
    __is_type_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__is_type_name', ['test'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__is_type_name', localization, ['test'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__is_type_name(...)' code ##################

    
    
    # Call to type(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'test' (line 54)
    test_31850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'test', False)
    # Processing the call keyword arguments (line 54)
    kwargs_31851 = {}
    # Getting the type of 'type' (line 54)
    type_31849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 7), 'type', False)
    # Calling type(args, kwargs) (line 54)
    type_call_result_31852 = invoke(stypy.reporting.localization.Localization(__file__, 54, 7), type_31849, *[test_31850], **kwargs_31851)
    
    # Getting the type of 'ast' (line 54)
    ast_31853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'ast')
    # Obtaining the member 'Name' of a type (line 54)
    Name_31854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 21), ast_31853, 'Name')
    # Applying the binary operator 'is' (line 54)
    result_is__31855 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 7), 'is', type_call_result_31852, Name_31854)
    
    # Testing if the type of an if condition is none (line 54)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 54, 4), result_is__31855):
        pass
    else:
        
        # Testing the type of an if condition (line 54)
        if_condition_31856 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 4), result_is__31855)
        # Assigning a type to the variable 'if_condition_31856' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'if_condition_31856', if_condition_31856)
        # SSA begins for if statement (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 55):
        
        # Assigning a Attribute to a Name (line 55):
        # Getting the type of 'test' (line 55)
        test_31857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 18), 'test')
        # Obtaining the member 'id' of a type (line 55)
        id_31858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 18), test_31857, 'id')
        # Assigning a type to the variable 'name_id' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'name_id', id_31858)
        
        
        # SSA begins for try-except statement (line 56)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 57):
        
        # Assigning a Call to a Name (line 57):
        
        # Call to eval(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'name_id' (line 57)
        name_id_31860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'name_id', False)
        # Processing the call keyword arguments (line 57)
        kwargs_31861 = {}
        # Getting the type of 'eval' (line 57)
        eval_31859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'eval', False)
        # Calling eval(args, kwargs) (line 57)
        eval_call_result_31862 = invoke(stypy.reporting.localization.Localization(__file__, 57, 23), eval_31859, *[name_id_31860], **kwargs_31861)
        
        # Assigning a type to the variable 'type_obj' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'type_obj', eval_call_result_31862)
        
        
        # Call to type(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'type_obj' (line 58)
        type_obj_31864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 24), 'type_obj', False)
        # Processing the call keyword arguments (line 58)
        kwargs_31865 = {}
        # Getting the type of 'type' (line 58)
        type_31863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'type', False)
        # Calling type(args, kwargs) (line 58)
        type_call_result_31866 = invoke(stypy.reporting.localization.Localization(__file__, 58, 19), type_31863, *[type_obj_31864], **kwargs_31865)
        
        # Getting the type of 'types' (line 58)
        types_31867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 37), 'types')
        # Obtaining the member 'TypeType' of a type (line 58)
        TypeType_31868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 37), types_31867, 'TypeType')
        # Applying the binary operator 'is' (line 58)
        result_is__31869 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 19), 'is', type_call_result_31866, TypeType_31868)
        
        # Assigning a type to the variable 'stypy_return_type' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'stypy_return_type', result_is__31869)
        # SSA branch for the except part of a try statement (line 56)
        # SSA branch for the except '<any exception>' branch of a try statement (line 56)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'False' (line 60)
        False_31870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'stypy_return_type', False_31870)
        # SSA join for try-except statement (line 56)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 54)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'False' (line 61)
    False_31871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type', False_31871)
    
    # ################# End of '__is_type_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__is_type_name' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_31872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31872)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__is_type_name'
    return stypy_return_type_31872

# Assigning a type to the variable '__is_type_name' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), '__is_type_name', __is_type_name)

@norecursion
def type_is_idiom(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'type_is_idiom'
    module_type_store = module_type_store.open_function_context('type_is_idiom', 64, 0, False)
    
    # Passed parameters checking function
    type_is_idiom.stypy_localization = localization
    type_is_idiom.stypy_type_of_self = None
    type_is_idiom.stypy_type_store = module_type_store
    type_is_idiom.stypy_function_name = 'type_is_idiom'
    type_is_idiom.stypy_param_names_list = ['test', 'visitor', 'context']
    type_is_idiom.stypy_varargs_param_name = None
    type_is_idiom.stypy_kwargs_param_name = None
    type_is_idiom.stypy_call_defaults = defaults
    type_is_idiom.stypy_call_varargs = varargs
    type_is_idiom.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'type_is_idiom', ['test', 'visitor', 'context'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'type_is_idiom', localization, ['test', 'visitor', 'context'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'type_is_idiom(...)' code ##################

    str_31873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, (-1)), 'str', '\n    Idiom "type is"\n    :param test:\n    :param visitor:\n    :param context:\n    :return:\n    ')
    
    
    # Call to type(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'test' (line 72)
    test_31875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'test', False)
    # Processing the call keyword arguments (line 72)
    kwargs_31876 = {}
    # Getting the type of 'type' (line 72)
    type_31874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 7), 'type', False)
    # Calling type(args, kwargs) (line 72)
    type_call_result_31877 = invoke(stypy.reporting.localization.Localization(__file__, 72, 7), type_31874, *[test_31875], **kwargs_31876)
    
    # Getting the type of 'ast' (line 72)
    ast_31878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'ast')
    # Obtaining the member 'Compare' of a type (line 72)
    Compare_31879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 25), ast_31878, 'Compare')
    # Applying the binary operator 'isnot' (line 72)
    result_is_not_31880 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 7), 'isnot', type_call_result_31877, Compare_31879)
    
    # Testing if the type of an if condition is none (line 72)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 4), result_is_not_31880):
        pass
    else:
        
        # Testing the type of an if condition (line 72)
        if_condition_31881 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 4), result_is_not_31880)
        # Assigning a type to the variable 'if_condition_31881' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'if_condition_31881', if_condition_31881)
        # SSA begins for if statement (line 72)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'default_ret_tuple' (line 73)
        default_ret_tuple_31882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'default_ret_tuple')
        # Assigning a type to the variable 'stypy_return_type' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'stypy_return_type', default_ret_tuple_31882)
        # SSA join for if statement (line 72)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Evaluating a boolean operation
    
    # Call to __has_call_to_type_builtin(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'test' (line 75)
    test_31884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 34), 'test', False)
    # Processing the call keyword arguments (line 75)
    kwargs_31885 = {}
    # Getting the type of '__has_call_to_type_builtin' (line 75)
    has_call_to_type_builtin_31883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 7), '__has_call_to_type_builtin', False)
    # Calling __has_call_to_type_builtin(args, kwargs) (line 75)
    has_call_to_type_builtin_call_result_31886 = invoke(stypy.reporting.localization.Localization(__file__, 75, 7), has_call_to_type_builtin_31883, *[test_31884], **kwargs_31885)
    
    
    # Call to __has_call_to_is(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'test' (line 75)
    test_31888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 61), 'test', False)
    # Processing the call keyword arguments (line 75)
    kwargs_31889 = {}
    # Getting the type of '__has_call_to_is' (line 75)
    has_call_to_is_31887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 44), '__has_call_to_is', False)
    # Calling __has_call_to_is(args, kwargs) (line 75)
    has_call_to_is_call_result_31890 = invoke(stypy.reporting.localization.Localization(__file__, 75, 44), has_call_to_is_31887, *[test_31888], **kwargs_31889)
    
    # Applying the binary operator 'and' (line 75)
    result_and_keyword_31891 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 7), 'and', has_call_to_type_builtin_call_result_31886, has_call_to_is_call_result_31890)
    
    # Testing if the type of an if condition is none (line 75)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 4), result_and_keyword_31891):
        pass
    else:
        
        # Testing the type of an if condition (line 75)
        if_condition_31892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 4), result_and_keyword_31891)
        # Assigning a type to the variable 'if_condition_31892' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'if_condition_31892', if_condition_31892)
        # SSA begins for if statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Call to __has_call_to_type_builtin(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Obtaining the type of the subscript
        int_31894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 60), 'int')
        # Getting the type of 'test' (line 76)
        test_31895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 43), 'test', False)
        # Obtaining the member 'comparators' of a type (line 76)
        comparators_31896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 43), test_31895, 'comparators')
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___31897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 43), comparators_31896, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_31898 = invoke(stypy.reporting.localization.Localization(__file__, 76, 43), getitem___31897, int_31894)
        
        # Processing the call keyword arguments (line 76)
        kwargs_31899 = {}
        # Getting the type of '__has_call_to_type_builtin' (line 76)
        has_call_to_type_builtin_31893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), '__has_call_to_type_builtin', False)
        # Calling __has_call_to_type_builtin(args, kwargs) (line 76)
        has_call_to_type_builtin_call_result_31900 = invoke(stypy.reporting.localization.Localization(__file__, 76, 16), has_call_to_type_builtin_31893, *[subscript_call_result_31898], **kwargs_31899)
        
        
        # Call to __is_type_name(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Obtaining the type of the subscript
        int_31902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 99), 'int')
        # Getting the type of 'test' (line 76)
        test_31903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 82), 'test', False)
        # Obtaining the member 'comparators' of a type (line 76)
        comparators_31904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 82), test_31903, 'comparators')
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___31905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 82), comparators_31904, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_31906 = invoke(stypy.reporting.localization.Localization(__file__, 76, 82), getitem___31905, int_31902)
        
        # Processing the call keyword arguments (line 76)
        kwargs_31907 = {}
        # Getting the type of '__is_type_name' (line 76)
        is_type_name_31901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 67), '__is_type_name', False)
        # Calling __is_type_name(args, kwargs) (line 76)
        is_type_name_call_result_31908 = invoke(stypy.reporting.localization.Localization(__file__, 76, 67), is_type_name_31901, *[subscript_call_result_31906], **kwargs_31907)
        
        # Applying the binary operator 'or' (line 76)
        result_or_keyword_31909 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 16), 'or', has_call_to_type_builtin_call_result_31900, is_type_name_call_result_31908)
        
        # Applying the 'not' unary operator (line 76)
        result_not__31910 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 11), 'not', result_or_keyword_31909)
        
        # Testing if the type of an if condition is none (line 76)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 8), result_not__31910):
            pass
        else:
            
            # Testing the type of an if condition (line 76)
            if_condition_31911 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 8), result_not__31910)
            # Assigning a type to the variable 'if_condition_31911' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'if_condition_31911', if_condition_31911)
            # SSA begins for if statement (line 76)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'default_ret_tuple' (line 77)
            default_ret_tuple_31912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'default_ret_tuple')
            # Assigning a type to the variable 'stypy_return_type' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'stypy_return_type', default_ret_tuple_31912)
            # SSA join for if statement (line 76)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 78):
        
        # Assigning a Call to a Name (line 78):
        
        # Call to visit(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Obtaining the type of the subscript
        int_31915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 50), 'int')
        # Getting the type of 'test' (line 78)
        test_31916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 35), 'test', False)
        # Obtaining the member 'left' of a type (line 78)
        left_31917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 35), test_31916, 'left')
        # Obtaining the member 'args' of a type (line 78)
        args_31918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 35), left_31917, 'args')
        # Obtaining the member '__getitem__' of a type (line 78)
        getitem___31919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 35), args_31918, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 78)
        subscript_call_result_31920 = invoke(stypy.reporting.localization.Localization(__file__, 78, 35), getitem___31919, int_31915)
        
        # Getting the type of 'context' (line 78)
        context_31921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 54), 'context', False)
        # Processing the call keyword arguments (line 78)
        kwargs_31922 = {}
        # Getting the type of 'visitor' (line 78)
        visitor_31913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 21), 'visitor', False)
        # Obtaining the member 'visit' of a type (line 78)
        visit_31914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 21), visitor_31913, 'visit')
        # Calling visit(args, kwargs) (line 78)
        visit_call_result_31923 = invoke(stypy.reporting.localization.Localization(__file__, 78, 21), visit_31914, *[subscript_call_result_31920, context_31921], **kwargs_31922)
        
        # Assigning a type to the variable 'type_param' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'type_param', visit_call_result_31923)
        
        # Call to __is_type_name(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Obtaining the type of the subscript
        int_31925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 43), 'int')
        # Getting the type of 'test' (line 79)
        test_31926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 26), 'test', False)
        # Obtaining the member 'comparators' of a type (line 79)
        comparators_31927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 26), test_31926, 'comparators')
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___31928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 26), comparators_31927, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_31929 = invoke(stypy.reporting.localization.Localization(__file__, 79, 26), getitem___31928, int_31925)
        
        # Processing the call keyword arguments (line 79)
        kwargs_31930 = {}
        # Getting the type of '__is_type_name' (line 79)
        is_type_name_31924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), '__is_type_name', False)
        # Calling __is_type_name(args, kwargs) (line 79)
        is_type_name_call_result_31931 = invoke(stypy.reporting.localization.Localization(__file__, 79, 11), is_type_name_31924, *[subscript_call_result_31929], **kwargs_31930)
        
        # Testing if the type of an if condition is none (line 79)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 79, 8), is_type_name_call_result_31931):
            
            # Assigning a Call to a Name (line 82):
            
            # Assigning a Call to a Name (line 82):
            
            # Call to visit(...): (line 82)
            # Processing the call arguments (line 82)
            
            # Obtaining the type of the subscript
            int_31945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 65), 'int')
            
            # Obtaining the type of the subscript
            int_31946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 57), 'int')
            # Getting the type of 'test' (line 82)
            test_31947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 40), 'test', False)
            # Obtaining the member 'comparators' of a type (line 82)
            comparators_31948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), test_31947, 'comparators')
            # Obtaining the member '__getitem__' of a type (line 82)
            getitem___31949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), comparators_31948, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 82)
            subscript_call_result_31950 = invoke(stypy.reporting.localization.Localization(__file__, 82, 40), getitem___31949, int_31946)
            
            # Obtaining the member 'args' of a type (line 82)
            args_31951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), subscript_call_result_31950, 'args')
            # Obtaining the member '__getitem__' of a type (line 82)
            getitem___31952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), args_31951, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 82)
            subscript_call_result_31953 = invoke(stypy.reporting.localization.Localization(__file__, 82, 40), getitem___31952, int_31945)
            
            # Getting the type of 'context' (line 82)
            context_31954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 69), 'context', False)
            # Processing the call keyword arguments (line 82)
            kwargs_31955 = {}
            # Getting the type of 'visitor' (line 82)
            visitor_31943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'visitor', False)
            # Obtaining the member 'visit' of a type (line 82)
            visit_31944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 26), visitor_31943, 'visit')
            # Calling visit(args, kwargs) (line 82)
            visit_call_result_31956 = invoke(stypy.reporting.localization.Localization(__file__, 82, 26), visit_31944, *[subscript_call_result_31953, context_31954], **kwargs_31955)
            
            # Assigning a type to the variable 'is_operator' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'is_operator', visit_call_result_31956)
            
            # Type idiom detected: calculating its left and rigth part (line 83)
            # Getting the type of 'list' (line 83)
            list_31957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 46), 'list')
            
            # Obtaining the type of the subscript
            int_31958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 42), 'int')
            # Getting the type of 'is_operator' (line 83)
            is_operator_31959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'is_operator')
            # Obtaining the member '__getitem__' of a type (line 83)
            getitem___31960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 30), is_operator_31959, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 83)
            subscript_call_result_31961 = invoke(stypy.reporting.localization.Localization(__file__, 83, 30), getitem___31960, int_31958)
            
            
            (may_be_31962, more_types_in_union_31963) = may_not_be_subtype(list_31957, subscript_call_result_31961)

            if may_be_31962:

                if more_types_in_union_31963:
                    # Runtime conditional SSA (line 83)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Tuple to a Name (line 84):
                
                # Assigning a Tuple to a Name (line 84):
                
                # Obtaining an instance of the builtin type 'tuple' (line 84)
                tuple_31964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 31), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 84)
                # Adding element type (line 84)
                
                # Obtaining an instance of the builtin type 'list' (line 84)
                list_31965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 31), 'list')
                # Adding type elements to the builtin type 'list' instance (line 84)
                # Adding element type (line 84)
                
                # Obtaining the type of the subscript
                int_31966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 44), 'int')
                # Getting the type of 'is_operator' (line 84)
                is_operator_31967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'is_operator')
                # Obtaining the member '__getitem__' of a type (line 84)
                getitem___31968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 32), is_operator_31967, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 84)
                subscript_call_result_31969 = invoke(stypy.reporting.localization.Localization(__file__, 84, 32), getitem___31968, int_31966)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 31), list_31965, subscript_call_result_31969)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 31), tuple_31964, list_31965)
                # Adding element type (line 84)
                
                # Obtaining the type of the subscript
                int_31970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 61), 'int')
                # Getting the type of 'is_operator' (line 84)
                is_operator_31971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 49), 'is_operator')
                # Obtaining the member '__getitem__' of a type (line 84)
                getitem___31972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 49), is_operator_31971, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 84)
                subscript_call_result_31973 = invoke(stypy.reporting.localization.Localization(__file__, 84, 49), getitem___31972, int_31970)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 31), tuple_31964, subscript_call_result_31973)
                
                # Assigning a type to the variable 'is_operator' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'is_operator', tuple_31964)

                if more_types_in_union_31963:
                    # SSA join for if statement (line 83)
                    module_type_store = module_type_store.join_ssa_context()


            
        else:
            
            # Testing the type of an if condition (line 79)
            if_condition_31932 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 8), is_type_name_call_result_31931)
            # Assigning a type to the variable 'if_condition_31932' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'if_condition_31932', if_condition_31932)
            # SSA begins for if statement (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 80):
            
            # Assigning a Call to a Name (line 80):
            
            # Call to visit(...): (line 80)
            # Processing the call arguments (line 80)
            
            # Obtaining the type of the subscript
            int_31935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 57), 'int')
            # Getting the type of 'test' (line 80)
            test_31936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 40), 'test', False)
            # Obtaining the member 'comparators' of a type (line 80)
            comparators_31937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 40), test_31936, 'comparators')
            # Obtaining the member '__getitem__' of a type (line 80)
            getitem___31938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 40), comparators_31937, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 80)
            subscript_call_result_31939 = invoke(stypy.reporting.localization.Localization(__file__, 80, 40), getitem___31938, int_31935)
            
            # Getting the type of 'context' (line 80)
            context_31940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 61), 'context', False)
            # Processing the call keyword arguments (line 80)
            kwargs_31941 = {}
            # Getting the type of 'visitor' (line 80)
            visitor_31933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 26), 'visitor', False)
            # Obtaining the member 'visit' of a type (line 80)
            visit_31934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 26), visitor_31933, 'visit')
            # Calling visit(args, kwargs) (line 80)
            visit_call_result_31942 = invoke(stypy.reporting.localization.Localization(__file__, 80, 26), visit_31934, *[subscript_call_result_31939, context_31940], **kwargs_31941)
            
            # Assigning a type to the variable 'is_operator' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'is_operator', visit_call_result_31942)
            # SSA branch for the else part of an if statement (line 79)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 82):
            
            # Assigning a Call to a Name (line 82):
            
            # Call to visit(...): (line 82)
            # Processing the call arguments (line 82)
            
            # Obtaining the type of the subscript
            int_31945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 65), 'int')
            
            # Obtaining the type of the subscript
            int_31946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 57), 'int')
            # Getting the type of 'test' (line 82)
            test_31947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 40), 'test', False)
            # Obtaining the member 'comparators' of a type (line 82)
            comparators_31948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), test_31947, 'comparators')
            # Obtaining the member '__getitem__' of a type (line 82)
            getitem___31949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), comparators_31948, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 82)
            subscript_call_result_31950 = invoke(stypy.reporting.localization.Localization(__file__, 82, 40), getitem___31949, int_31946)
            
            # Obtaining the member 'args' of a type (line 82)
            args_31951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), subscript_call_result_31950, 'args')
            # Obtaining the member '__getitem__' of a type (line 82)
            getitem___31952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), args_31951, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 82)
            subscript_call_result_31953 = invoke(stypy.reporting.localization.Localization(__file__, 82, 40), getitem___31952, int_31945)
            
            # Getting the type of 'context' (line 82)
            context_31954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 69), 'context', False)
            # Processing the call keyword arguments (line 82)
            kwargs_31955 = {}
            # Getting the type of 'visitor' (line 82)
            visitor_31943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'visitor', False)
            # Obtaining the member 'visit' of a type (line 82)
            visit_31944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 26), visitor_31943, 'visit')
            # Calling visit(args, kwargs) (line 82)
            visit_call_result_31956 = invoke(stypy.reporting.localization.Localization(__file__, 82, 26), visit_31944, *[subscript_call_result_31953, context_31954], **kwargs_31955)
            
            # Assigning a type to the variable 'is_operator' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'is_operator', visit_call_result_31956)
            
            # Type idiom detected: calculating its left and rigth part (line 83)
            # Getting the type of 'list' (line 83)
            list_31957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 46), 'list')
            
            # Obtaining the type of the subscript
            int_31958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 42), 'int')
            # Getting the type of 'is_operator' (line 83)
            is_operator_31959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'is_operator')
            # Obtaining the member '__getitem__' of a type (line 83)
            getitem___31960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 30), is_operator_31959, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 83)
            subscript_call_result_31961 = invoke(stypy.reporting.localization.Localization(__file__, 83, 30), getitem___31960, int_31958)
            
            
            (may_be_31962, more_types_in_union_31963) = may_not_be_subtype(list_31957, subscript_call_result_31961)

            if may_be_31962:

                if more_types_in_union_31963:
                    # Runtime conditional SSA (line 83)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Tuple to a Name (line 84):
                
                # Assigning a Tuple to a Name (line 84):
                
                # Obtaining an instance of the builtin type 'tuple' (line 84)
                tuple_31964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 31), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 84)
                # Adding element type (line 84)
                
                # Obtaining an instance of the builtin type 'list' (line 84)
                list_31965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 31), 'list')
                # Adding type elements to the builtin type 'list' instance (line 84)
                # Adding element type (line 84)
                
                # Obtaining the type of the subscript
                int_31966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 44), 'int')
                # Getting the type of 'is_operator' (line 84)
                is_operator_31967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'is_operator')
                # Obtaining the member '__getitem__' of a type (line 84)
                getitem___31968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 32), is_operator_31967, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 84)
                subscript_call_result_31969 = invoke(stypy.reporting.localization.Localization(__file__, 84, 32), getitem___31968, int_31966)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 31), list_31965, subscript_call_result_31969)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 31), tuple_31964, list_31965)
                # Adding element type (line 84)
                
                # Obtaining the type of the subscript
                int_31970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 61), 'int')
                # Getting the type of 'is_operator' (line 84)
                is_operator_31971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 49), 'is_operator')
                # Obtaining the member '__getitem__' of a type (line 84)
                getitem___31972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 49), is_operator_31971, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 84)
                subscript_call_result_31973 = invoke(stypy.reporting.localization.Localization(__file__, 84, 49), getitem___31972, int_31970)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 31), tuple_31964, subscript_call_result_31973)
                
                # Assigning a type to the variable 'is_operator' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'is_operator', tuple_31964)

                if more_types_in_union_31963:
                    # SSA join for if statement (line 83)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 79)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Obtaining an instance of the builtin type 'tuple' (line 86)
        tuple_31974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 86)
        # Adding element type (line 86)
        # Getting the type of 'True' (line 86)
        True_31975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 15), tuple_31974, True_31975)
        # Adding element type (line 86)
        # Getting the type of 'type_param' (line 86)
        type_param_31976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'type_param')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 15), tuple_31974, type_param_31976)
        # Adding element type (line 86)
        # Getting the type of 'is_operator' (line 86)
        is_operator_31977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 33), 'is_operator')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 15), tuple_31974, is_operator_31977)
        
        # Assigning a type to the variable 'stypy_return_type' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'stypy_return_type', tuple_31974)
        # SSA join for if statement (line 75)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'default_ret_tuple' (line 88)
    default_ret_tuple_31978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'default_ret_tuple')
    # Assigning a type to the variable 'stypy_return_type' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type', default_ret_tuple_31978)
    
    # ################# End of 'type_is_idiom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'type_is_idiom' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_31979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31979)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'type_is_idiom'
    return stypy_return_type_31979

# Assigning a type to the variable 'type_is_idiom' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'type_is_idiom', type_is_idiom)

@norecursion
def not_type_is_idiom(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'not_type_is_idiom'
    module_type_store = module_type_store.open_function_context('not_type_is_idiom', 91, 0, False)
    
    # Passed parameters checking function
    not_type_is_idiom.stypy_localization = localization
    not_type_is_idiom.stypy_type_of_self = None
    not_type_is_idiom.stypy_type_store = module_type_store
    not_type_is_idiom.stypy_function_name = 'not_type_is_idiom'
    not_type_is_idiom.stypy_param_names_list = ['test', 'visitor', 'context']
    not_type_is_idiom.stypy_varargs_param_name = None
    not_type_is_idiom.stypy_kwargs_param_name = None
    not_type_is_idiom.stypy_call_defaults = defaults
    not_type_is_idiom.stypy_call_varargs = varargs
    not_type_is_idiom.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'not_type_is_idiom', ['test', 'visitor', 'context'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'not_type_is_idiom', localization, ['test', 'visitor', 'context'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'not_type_is_idiom(...)' code ##################

    str_31980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, (-1)), 'str', '\n    Idiom "not type is"\n\n    :param test:\n    :param visitor:\n    :param context:\n    :return:\n    ')
    
    
    # Call to type(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'test' (line 100)
    test_31982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'test', False)
    # Processing the call keyword arguments (line 100)
    kwargs_31983 = {}
    # Getting the type of 'type' (line 100)
    type_31981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 7), 'type', False)
    # Calling type(args, kwargs) (line 100)
    type_call_result_31984 = invoke(stypy.reporting.localization.Localization(__file__, 100, 7), type_31981, *[test_31982], **kwargs_31983)
    
    # Getting the type of 'ast' (line 100)
    ast_31985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'ast')
    # Obtaining the member 'UnaryOp' of a type (line 100)
    UnaryOp_31986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 25), ast_31985, 'UnaryOp')
    # Applying the binary operator 'isnot' (line 100)
    result_is_not_31987 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 7), 'isnot', type_call_result_31984, UnaryOp_31986)
    
    # Testing if the type of an if condition is none (line 100)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 100, 4), result_is_not_31987):
        pass
    else:
        
        # Testing the type of an if condition (line 100)
        if_condition_31988 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 4), result_is_not_31987)
        # Assigning a type to the variable 'if_condition_31988' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'if_condition_31988', if_condition_31988)
        # SSA begins for if statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'default_ret_tuple' (line 101)
        default_ret_tuple_31989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'default_ret_tuple')
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'stypy_return_type', default_ret_tuple_31989)
        # SSA join for if statement (line 100)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to type(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'test' (line 102)
    test_31991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'test', False)
    # Obtaining the member 'op' of a type (line 102)
    op_31992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), test_31991, 'op')
    # Processing the call keyword arguments (line 102)
    kwargs_31993 = {}
    # Getting the type of 'type' (line 102)
    type_31990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 7), 'type', False)
    # Calling type(args, kwargs) (line 102)
    type_call_result_31994 = invoke(stypy.reporting.localization.Localization(__file__, 102, 7), type_31990, *[op_31992], **kwargs_31993)
    
    # Getting the type of 'ast' (line 102)
    ast_31995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'ast')
    # Obtaining the member 'Not' of a type (line 102)
    Not_31996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 28), ast_31995, 'Not')
    # Applying the binary operator 'isnot' (line 102)
    result_is_not_31997 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 7), 'isnot', type_call_result_31994, Not_31996)
    
    # Testing if the type of an if condition is none (line 102)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 102, 4), result_is_not_31997):
        pass
    else:
        
        # Testing the type of an if condition (line 102)
        if_condition_31998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 4), result_is_not_31997)
        # Assigning a type to the variable 'if_condition_31998' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'if_condition_31998', if_condition_31998)
        # SSA begins for if statement (line 102)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'default_ret_tuple' (line 103)
        default_ret_tuple_31999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'default_ret_tuple')
        # Assigning a type to the variable 'stypy_return_type' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'stypy_return_type', default_ret_tuple_31999)
        # SSA join for if statement (line 102)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to type_is_idiom(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'test' (line 105)
    test_32001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 25), 'test', False)
    # Obtaining the member 'operand' of a type (line 105)
    operand_32002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 25), test_32001, 'operand')
    # Getting the type of 'visitor' (line 105)
    visitor_32003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 39), 'visitor', False)
    # Getting the type of 'context' (line 105)
    context_32004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 48), 'context', False)
    # Processing the call keyword arguments (line 105)
    kwargs_32005 = {}
    # Getting the type of 'type_is_idiom' (line 105)
    type_is_idiom_32000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'type_is_idiom', False)
    # Calling type_is_idiom(args, kwargs) (line 105)
    type_is_idiom_call_result_32006 = invoke(stypy.reporting.localization.Localization(__file__, 105, 11), type_is_idiom_32000, *[operand_32002, visitor_32003, context_32004], **kwargs_32005)
    
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', type_is_idiom_call_result_32006)
    
    # ################# End of 'not_type_is_idiom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'not_type_is_idiom' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_32007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32007)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'not_type_is_idiom'
    return stypy_return_type_32007

# Assigning a type to the variable 'not_type_is_idiom' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'not_type_is_idiom', not_type_is_idiom)

@norecursion
def __get_idiom_type_param(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__get_idiom_type_param'
    module_type_store = module_type_store.open_function_context('__get_idiom_type_param', 108, 0, False)
    
    # Passed parameters checking function
    __get_idiom_type_param.stypy_localization = localization
    __get_idiom_type_param.stypy_type_of_self = None
    __get_idiom_type_param.stypy_type_store = module_type_store
    __get_idiom_type_param.stypy_function_name = '__get_idiom_type_param'
    __get_idiom_type_param.stypy_param_names_list = ['test']
    __get_idiom_type_param.stypy_varargs_param_name = None
    __get_idiom_type_param.stypy_kwargs_param_name = None
    __get_idiom_type_param.stypy_call_defaults = defaults
    __get_idiom_type_param.stypy_call_varargs = varargs
    __get_idiom_type_param.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__get_idiom_type_param', ['test'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__get_idiom_type_param', localization, ['test'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__get_idiom_type_param(...)' code ##################

    
    # Obtaining the type of the subscript
    int_32008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 26), 'int')
    # Getting the type of 'test' (line 109)
    test_32009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'test')
    # Obtaining the member 'left' of a type (line 109)
    left_32010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 11), test_32009, 'left')
    # Obtaining the member 'args' of a type (line 109)
    args_32011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 11), left_32010, 'args')
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___32012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 11), args_32011, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_32013 = invoke(stypy.reporting.localization.Localization(__file__, 109, 11), getitem___32012, int_32008)
    
    # Assigning a type to the variable 'stypy_return_type' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type', subscript_call_result_32013)
    
    # ################# End of '__get_idiom_type_param(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__get_idiom_type_param' in the type store
    # Getting the type of 'stypy_return_type' (line 108)
    stypy_return_type_32014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32014)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__get_idiom_type_param'
    return stypy_return_type_32014

# Assigning a type to the variable '__get_idiom_type_param' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), '__get_idiom_type_param', __get_idiom_type_param)

@norecursion
def __set_type_implementation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__set_type_implementation'
    module_type_store = module_type_store.open_function_context('__set_type_implementation', 112, 0, False)
    
    # Passed parameters checking function
    __set_type_implementation.stypy_localization = localization
    __set_type_implementation.stypy_type_of_self = None
    __set_type_implementation.stypy_type_store = module_type_store
    __set_type_implementation.stypy_function_name = '__set_type_implementation'
    __set_type_implementation.stypy_param_names_list = ['if_test', 'type_', 'lineno', 'col_offset']
    __set_type_implementation.stypy_varargs_param_name = None
    __set_type_implementation.stypy_kwargs_param_name = None
    __set_type_implementation.stypy_call_defaults = defaults
    __set_type_implementation.stypy_call_varargs = varargs
    __set_type_implementation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__set_type_implementation', ['if_test', 'type_', 'lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__set_type_implementation', localization, ['if_test', 'type_', 'lineno', 'col_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__set_type_implementation(...)' code ##################

    
    # Assigning a Call to a Name (line 113):
    
    # Assigning a Call to a Name (line 113):
    
    # Call to __get_idiom_type_param(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'if_test' (line 113)
    if_test_32016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 35), 'if_test', False)
    # Processing the call keyword arguments (line 113)
    kwargs_32017 = {}
    # Getting the type of '__get_idiom_type_param' (line 113)
    get_idiom_type_param_32015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), '__get_idiom_type_param', False)
    # Calling __get_idiom_type_param(args, kwargs) (line 113)
    get_idiom_type_param_call_result_32018 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), get_idiom_type_param_32015, *[if_test_32016], **kwargs_32017)
    
    # Assigning a type to the variable 'param' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'param', get_idiom_type_param_call_result_32018)
    
    
    # Call to type(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'param' (line 114)
    param_32020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'param', False)
    # Processing the call keyword arguments (line 114)
    kwargs_32021 = {}
    # Getting the type of 'type' (line 114)
    type_32019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 7), 'type', False)
    # Calling type(args, kwargs) (line 114)
    type_call_result_32022 = invoke(stypy.reporting.localization.Localization(__file__, 114, 7), type_32019, *[param_32020], **kwargs_32021)
    
    # Getting the type of 'ast' (line 114)
    ast_32023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'ast')
    # Obtaining the member 'Name' of a type (line 114)
    Name_32024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 22), ast_32023, 'Name')
    # Applying the binary operator 'is' (line 114)
    result_is__32025 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 7), 'is', type_call_result_32022, Name_32024)
    
    # Testing if the type of an if condition is none (line 114)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 114, 4), result_is__32025):
        pass
    else:
        
        # Testing the type of an if condition (line 114)
        if_condition_32026 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 4), result_is__32025)
        # Assigning a type to the variable 'if_condition_32026' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'if_condition_32026', if_condition_32026)
        # SSA begins for if statement (line 114)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to create_set_type_of(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'param' (line 115)
        param_32029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 55), 'param', False)
        # Obtaining the member 'id' of a type (line 115)
        id_32030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 55), param_32029, 'id')
        # Getting the type of 'type_' (line 115)
        type__32031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 65), 'type_', False)
        # Getting the type of 'lineno' (line 115)
        lineno_32032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 72), 'lineno', False)
        # Getting the type of 'col_offset' (line 115)
        col_offset_32033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 80), 'col_offset', False)
        # Processing the call keyword arguments (line 115)
        kwargs_32034 = {}
        # Getting the type of 'stypy_functions_copy' (line 115)
        stypy_functions_copy_32027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 'stypy_functions_copy', False)
        # Obtaining the member 'create_set_type_of' of a type (line 115)
        create_set_type_of_32028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 15), stypy_functions_copy_32027, 'create_set_type_of')
        # Calling create_set_type_of(args, kwargs) (line 115)
        create_set_type_of_call_result_32035 = invoke(stypy.reporting.localization.Localization(__file__, 115, 15), create_set_type_of_32028, *[id_32030, type__32031, lineno_32032, col_offset_32033], **kwargs_32034)
        
        # Assigning a type to the variable 'stypy_return_type' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'stypy_return_type', create_set_type_of_call_result_32035)
        # SSA join for if statement (line 114)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to type(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'param' (line 116)
    param_32037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'param', False)
    # Processing the call keyword arguments (line 116)
    kwargs_32038 = {}
    # Getting the type of 'type' (line 116)
    type_32036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 7), 'type', False)
    # Calling type(args, kwargs) (line 116)
    type_call_result_32039 = invoke(stypy.reporting.localization.Localization(__file__, 116, 7), type_32036, *[param_32037], **kwargs_32038)
    
    # Getting the type of 'ast' (line 116)
    ast_32040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 22), 'ast')
    # Obtaining the member 'Attribute' of a type (line 116)
    Attribute_32041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 22), ast_32040, 'Attribute')
    # Applying the binary operator 'is' (line 116)
    result_is__32042 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 7), 'is', type_call_result_32039, Attribute_32041)
    
    # Testing if the type of an if condition is none (line 116)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 116, 4), result_is__32042):
        pass
    else:
        
        # Testing the type of an if condition (line 116)
        if_condition_32043 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 4), result_is__32042)
        # Assigning a type to the variable 'if_condition_32043' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'if_condition_32043', if_condition_32043)
        # SSA begins for if statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 117):
        
        # Assigning a Call to a Name:
        
        # Call to create_get_type_of(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'param' (line 117)
        param_32046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 68), 'param', False)
        # Obtaining the member 'value' of a type (line 117)
        value_32047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 68), param_32046, 'value')
        # Obtaining the member 'id' of a type (line 117)
        id_32048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 68), value_32047, 'id')
        # Getting the type of 'lineno' (line 117)
        lineno_32049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 84), 'lineno', False)
        # Getting the type of 'col_offset' (line 117)
        col_offset_32050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 92), 'col_offset', False)
        # Processing the call keyword arguments (line 117)
        kwargs_32051 = {}
        # Getting the type of 'stypy_functions_copy' (line 117)
        stypy_functions_copy_32044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 28), 'stypy_functions_copy', False)
        # Obtaining the member 'create_get_type_of' of a type (line 117)
        create_get_type_of_32045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 28), stypy_functions_copy_32044, 'create_get_type_of')
        # Calling create_get_type_of(args, kwargs) (line 117)
        create_get_type_of_call_result_32052 = invoke(stypy.reporting.localization.Localization(__file__, 117, 28), create_get_type_of_32045, *[id_32048, lineno_32049, col_offset_32050], **kwargs_32051)
        
        # Assigning a type to the variable 'call_assignment_31714' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_31714', create_get_type_of_call_result_32052)
        
        # Assigning a Call to a Name (line 117):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_31714' (line 117)
        call_assignment_31714_32053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_31714', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_32054 = stypy_get_value_from_tuple(call_assignment_31714_32053, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_31715' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_31715', stypy_get_value_from_tuple_call_result_32054)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'call_assignment_31715' (line 117)
        call_assignment_31715_32055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_31715')
        # Assigning a type to the variable 'obj_type' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'obj_type', call_assignment_31715_32055)
        
        # Assigning a Call to a Name (line 117):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_31714' (line 117)
        call_assignment_31714_32056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_31714', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_32057 = stypy_get_value_from_tuple(call_assignment_31714_32056, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_31716' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_31716', stypy_get_value_from_tuple_call_result_32057)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'call_assignment_31716' (line 117)
        call_assignment_31716_32058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_31716')
        # Assigning a type to the variable 'obj_var' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 18), 'obj_var', call_assignment_31716_32058)
        
        # Assigning a Call to a Name (line 118):
        
        # Assigning a Call to a Name (line 118):
        
        # Call to create_set_type_of_member(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'obj_var' (line 118)
        obj_var_32061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 68), 'obj_var', False)
        # Getting the type of 'param' (line 118)
        param_32062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 77), 'param', False)
        # Obtaining the member 'attr' of a type (line 118)
        attr_32063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 77), param_32062, 'attr')
        # Getting the type of 'type_' (line 118)
        type__32064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 89), 'type_', False)
        # Getting the type of 'lineno' (line 118)
        lineno_32065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 96), 'lineno', False)
        # Getting the type of 'col_offset' (line 118)
        col_offset_32066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 104), 'col_offset', False)
        # Processing the call keyword arguments (line 118)
        kwargs_32067 = {}
        # Getting the type of 'stypy_functions_copy' (line 118)
        stypy_functions_copy_32059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'stypy_functions_copy', False)
        # Obtaining the member 'create_set_type_of_member' of a type (line 118)
        create_set_type_of_member_32060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 21), stypy_functions_copy_32059, 'create_set_type_of_member')
        # Calling create_set_type_of_member(args, kwargs) (line 118)
        create_set_type_of_member_call_result_32068 = invoke(stypy.reporting.localization.Localization(__file__, 118, 21), create_set_type_of_member_32060, *[obj_var_32061, attr_32063, type__32064, lineno_32065, col_offset_32066], **kwargs_32067)
        
        # Assigning a type to the variable 'set_member' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'set_member', create_set_type_of_member_call_result_32068)
        
        # Call to flatten_lists(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'obj_type' (line 119)
        obj_type_32071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 50), 'obj_type', False)
        # Getting the type of 'set_member' (line 119)
        set_member_32072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 60), 'set_member', False)
        # Processing the call keyword arguments (line 119)
        kwargs_32073 = {}
        # Getting the type of 'stypy_functions_copy' (line 119)
        stypy_functions_copy_32069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'stypy_functions_copy', False)
        # Obtaining the member 'flatten_lists' of a type (line 119)
        flatten_lists_32070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 15), stypy_functions_copy_32069, 'flatten_lists')
        # Calling flatten_lists(args, kwargs) (line 119)
        flatten_lists_call_result_32074 = invoke(stypy.reporting.localization.Localization(__file__, 119, 15), flatten_lists_32070, *[obj_type_32071, set_member_32072], **kwargs_32073)
        
        # Assigning a type to the variable 'stypy_return_type' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'stypy_return_type', flatten_lists_call_result_32074)
        # SSA join for if statement (line 116)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Obtaining an instance of the builtin type 'list' (line 121)
    list_32075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 121)
    
    # Assigning a type to the variable 'stypy_return_type' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type', list_32075)
    
    # ################# End of '__set_type_implementation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__set_type_implementation' in the type store
    # Getting the type of 'stypy_return_type' (line 112)
    stypy_return_type_32076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32076)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__set_type_implementation'
    return stypy_return_type_32076

# Assigning a type to the variable '__set_type_implementation' (line 112)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), '__set_type_implementation', __set_type_implementation)

@norecursion
def __remove_type_from_union_implementation(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__remove_type_from_union_implementation'
    module_type_store = module_type_store.open_function_context('__remove_type_from_union_implementation', 124, 0, False)
    
    # Passed parameters checking function
    __remove_type_from_union_implementation.stypy_localization = localization
    __remove_type_from_union_implementation.stypy_type_of_self = None
    __remove_type_from_union_implementation.stypy_type_store = module_type_store
    __remove_type_from_union_implementation.stypy_function_name = '__remove_type_from_union_implementation'
    __remove_type_from_union_implementation.stypy_param_names_list = ['if_test', 'type_', 'lineno', 'col_offset']
    __remove_type_from_union_implementation.stypy_varargs_param_name = None
    __remove_type_from_union_implementation.stypy_kwargs_param_name = None
    __remove_type_from_union_implementation.stypy_call_defaults = defaults
    __remove_type_from_union_implementation.stypy_call_varargs = varargs
    __remove_type_from_union_implementation.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__remove_type_from_union_implementation', ['if_test', 'type_', 'lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__remove_type_from_union_implementation', localization, ['if_test', 'type_', 'lineno', 'col_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__remove_type_from_union_implementation(...)' code ##################

    
    # Assigning a Call to a Name (line 125):
    
    # Assigning a Call to a Name (line 125):
    
    # Call to __get_idiom_type_param(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'if_test' (line 125)
    if_test_32078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 35), 'if_test', False)
    # Processing the call keyword arguments (line 125)
    kwargs_32079 = {}
    # Getting the type of '__get_idiom_type_param' (line 125)
    get_idiom_type_param_32077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), '__get_idiom_type_param', False)
    # Calling __get_idiom_type_param(args, kwargs) (line 125)
    get_idiom_type_param_call_result_32080 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), get_idiom_type_param_32077, *[if_test_32078], **kwargs_32079)
    
    # Assigning a type to the variable 'param' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'param', get_idiom_type_param_call_result_32080)
    
    
    # Call to type(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'param' (line 126)
    param_32082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'param', False)
    # Processing the call keyword arguments (line 126)
    kwargs_32083 = {}
    # Getting the type of 'type' (line 126)
    type_32081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 7), 'type', False)
    # Calling type(args, kwargs) (line 126)
    type_call_result_32084 = invoke(stypy.reporting.localization.Localization(__file__, 126, 7), type_32081, *[param_32082], **kwargs_32083)
    
    # Getting the type of 'ast' (line 126)
    ast_32085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'ast')
    # Obtaining the member 'Name' of a type (line 126)
    Name_32086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 22), ast_32085, 'Name')
    # Applying the binary operator 'is' (line 126)
    result_is__32087 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 7), 'is', type_call_result_32084, Name_32086)
    
    # Testing if the type of an if condition is none (line 126)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 126, 4), result_is__32087):
        pass
    else:
        
        # Testing the type of an if condition (line 126)
        if_condition_32088 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 4), result_is__32087)
        # Assigning a type to the variable 'if_condition_32088' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'if_condition_32088', if_condition_32088)
        # SSA begins for if statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 127):
        
        # Assigning a Call to a Name:
        
        # Call to create_get_type_of(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'param' (line 127)
        param_32091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 68), 'param', False)
        # Obtaining the member 'id' of a type (line 127)
        id_32092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 68), param_32091, 'id')
        # Getting the type of 'lineno' (line 127)
        lineno_32093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 78), 'lineno', False)
        # Getting the type of 'col_offset' (line 127)
        col_offset_32094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 86), 'col_offset', False)
        # Processing the call keyword arguments (line 127)
        kwargs_32095 = {}
        # Getting the type of 'stypy_functions_copy' (line 127)
        stypy_functions_copy_32089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 28), 'stypy_functions_copy', False)
        # Obtaining the member 'create_get_type_of' of a type (line 127)
        create_get_type_of_32090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 28), stypy_functions_copy_32089, 'create_get_type_of')
        # Calling create_get_type_of(args, kwargs) (line 127)
        create_get_type_of_call_result_32096 = invoke(stypy.reporting.localization.Localization(__file__, 127, 28), create_get_type_of_32090, *[id_32092, lineno_32093, col_offset_32094], **kwargs_32095)
        
        # Assigning a type to the variable 'call_assignment_31717' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'call_assignment_31717', create_get_type_of_call_result_32096)
        
        # Assigning a Call to a Name (line 127):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_31717' (line 127)
        call_assignment_31717_32097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'call_assignment_31717', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_32098 = stypy_get_value_from_tuple(call_assignment_31717_32097, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_31718' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'call_assignment_31718', stypy_get_value_from_tuple_call_result_32098)
        
        # Assigning a Name to a Name (line 127):
        # Getting the type of 'call_assignment_31718' (line 127)
        call_assignment_31718_32099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'call_assignment_31718')
        # Assigning a type to the variable 'obj_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'obj_type', call_assignment_31718_32099)
        
        # Assigning a Call to a Name (line 127):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_31717' (line 127)
        call_assignment_31717_32100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'call_assignment_31717', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_32101 = stypy_get_value_from_tuple(call_assignment_31717_32100, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_31719' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'call_assignment_31719', stypy_get_value_from_tuple_call_result_32101)
        
        # Assigning a Name to a Name (line 127):
        # Getting the type of 'call_assignment_31719' (line 127)
        call_assignment_31719_32102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'call_assignment_31719')
        # Assigning a type to the variable 'obj_var' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 18), 'obj_var', call_assignment_31719_32102)
        
        # Assigning a Call to a Name (line 128):
        
        # Assigning a Call to a Name (line 128):
        
        # Call to create_call(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Call to create_Name(...): (line 128)
        # Processing the call arguments (line 128)
        str_32107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 85), 'str', 'remove_type_from_union')
        # Processing the call keyword arguments (line 128)
        kwargs_32108 = {}
        # Getting the type of 'core_language_copy' (line 128)
        core_language_copy_32105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 54), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 128)
        create_Name_32106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 54), core_language_copy_32105, 'create_Name')
        # Calling create_Name(args, kwargs) (line 128)
        create_Name_call_result_32109 = invoke(stypy.reporting.localization.Localization(__file__, 128, 54), create_Name_32106, *[str_32107], **kwargs_32108)
        
        
        # Obtaining an instance of the builtin type 'list' (line 129)
        list_32110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 129)
        # Adding element type (line 129)
        # Getting the type of 'obj_var' (line 129)
        obj_var_32111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 50), 'obj_var', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 49), list_32110, obj_var_32111)
        # Adding element type (line 129)
        # Getting the type of 'type_' (line 129)
        type__32112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 59), 'type_', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 49), list_32110, type__32112)
        
        # Processing the call keyword arguments (line 128)
        # Getting the type of 'lineno' (line 129)
        lineno_32113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 72), 'lineno', False)
        keyword_32114 = lineno_32113
        # Getting the type of 'col_offset' (line 129)
        col_offset_32115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 87), 'col_offset', False)
        keyword_32116 = col_offset_32115
        kwargs_32117 = {'column': keyword_32116, 'line': keyword_32114}
        # Getting the type of 'functions_copy' (line 128)
        functions_copy_32103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 27), 'functions_copy', False)
        # Obtaining the member 'create_call' of a type (line 128)
        create_call_32104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 27), functions_copy_32103, 'create_call')
        # Calling create_call(args, kwargs) (line 128)
        create_call_call_result_32118 = invoke(stypy.reporting.localization.Localization(__file__, 128, 27), create_call_32104, *[create_Name_call_result_32109, list_32110], **kwargs_32117)
        
        # Assigning a type to the variable 'remove_type_call' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'remove_type_call', create_call_call_result_32118)
        
        # Assigning a Call to a Name (line 130):
        
        # Assigning a Call to a Name (line 130):
        
        # Call to create_set_type_of(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'param' (line 130)
        param_32121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 59), 'param', False)
        # Obtaining the member 'id' of a type (line 130)
        id_32122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 59), param_32121, 'id')
        # Getting the type of 'remove_type_call' (line 130)
        remove_type_call_32123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 69), 'remove_type_call', False)
        # Getting the type of 'lineno' (line 130)
        lineno_32124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 87), 'lineno', False)
        # Getting the type of 'col_offset' (line 130)
        col_offset_32125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 95), 'col_offset', False)
        # Processing the call keyword arguments (line 130)
        kwargs_32126 = {}
        # Getting the type of 'stypy_functions_copy' (line 130)
        stypy_functions_copy_32119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 19), 'stypy_functions_copy', False)
        # Obtaining the member 'create_set_type_of' of a type (line 130)
        create_set_type_of_32120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 19), stypy_functions_copy_32119, 'create_set_type_of')
        # Calling create_set_type_of(args, kwargs) (line 130)
        create_set_type_of_call_result_32127 = invoke(stypy.reporting.localization.Localization(__file__, 130, 19), create_set_type_of_32120, *[id_32122, remove_type_call_32123, lineno_32124, col_offset_32125], **kwargs_32126)
        
        # Assigning a type to the variable 'set_type' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'set_type', create_set_type_of_call_result_32127)
        
        # Call to flatten_lists(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'obj_type' (line 132)
        obj_type_32130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 50), 'obj_type', False)
        # Getting the type of 'set_type' (line 132)
        set_type_32131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 60), 'set_type', False)
        # Processing the call keyword arguments (line 132)
        kwargs_32132 = {}
        # Getting the type of 'stypy_functions_copy' (line 132)
        stypy_functions_copy_32128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'stypy_functions_copy', False)
        # Obtaining the member 'flatten_lists' of a type (line 132)
        flatten_lists_32129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 15), stypy_functions_copy_32128, 'flatten_lists')
        # Calling flatten_lists(args, kwargs) (line 132)
        flatten_lists_call_result_32133 = invoke(stypy.reporting.localization.Localization(__file__, 132, 15), flatten_lists_32129, *[obj_type_32130, set_type_32131], **kwargs_32132)
        
        # Assigning a type to the variable 'stypy_return_type' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'stypy_return_type', flatten_lists_call_result_32133)
        # SSA join for if statement (line 126)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to type(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'param' (line 133)
    param_32135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'param', False)
    # Processing the call keyword arguments (line 133)
    kwargs_32136 = {}
    # Getting the type of 'type' (line 133)
    type_32134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 7), 'type', False)
    # Calling type(args, kwargs) (line 133)
    type_call_result_32137 = invoke(stypy.reporting.localization.Localization(__file__, 133, 7), type_32134, *[param_32135], **kwargs_32136)
    
    # Getting the type of 'ast' (line 133)
    ast_32138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'ast')
    # Obtaining the member 'Attribute' of a type (line 133)
    Attribute_32139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 22), ast_32138, 'Attribute')
    # Applying the binary operator 'is' (line 133)
    result_is__32140 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 7), 'is', type_call_result_32137, Attribute_32139)
    
    # Testing if the type of an if condition is none (line 133)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 133, 4), result_is__32140):
        pass
    else:
        
        # Testing the type of an if condition (line 133)
        if_condition_32141 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 4), result_is__32140)
        # Assigning a type to the variable 'if_condition_32141' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'if_condition_32141', if_condition_32141)
        # SSA begins for if statement (line 133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 135):
        
        # Assigning a Call to a Name:
        
        # Call to create_get_type_of(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'param' (line 135)
        param_32144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 74), 'param', False)
        # Obtaining the member 'value' of a type (line 135)
        value_32145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 74), param_32144, 'value')
        # Obtaining the member 'id' of a type (line 135)
        id_32146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 74), value_32145, 'id')
        # Getting the type of 'lineno' (line 135)
        lineno_32147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 90), 'lineno', False)
        # Getting the type of 'col_offset' (line 135)
        col_offset_32148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 98), 'col_offset', False)
        # Processing the call keyword arguments (line 135)
        kwargs_32149 = {}
        # Getting the type of 'stypy_functions_copy' (line 135)
        stypy_functions_copy_32142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 34), 'stypy_functions_copy', False)
        # Obtaining the member 'create_get_type_of' of a type (line 135)
        create_get_type_of_32143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 34), stypy_functions_copy_32142, 'create_get_type_of')
        # Calling create_get_type_of(args, kwargs) (line 135)
        create_get_type_of_call_result_32150 = invoke(stypy.reporting.localization.Localization(__file__, 135, 34), create_get_type_of_32143, *[id_32146, lineno_32147, col_offset_32148], **kwargs_32149)
        
        # Assigning a type to the variable 'call_assignment_31720' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'call_assignment_31720', create_get_type_of_call_result_32150)
        
        # Assigning a Call to a Name (line 135):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_31720' (line 135)
        call_assignment_31720_32151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'call_assignment_31720', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_32152 = stypy_get_value_from_tuple(call_assignment_31720_32151, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_31721' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'call_assignment_31721', stypy_get_value_from_tuple_call_result_32152)
        
        # Assigning a Name to a Name (line 135):
        # Getting the type of 'call_assignment_31721' (line 135)
        call_assignment_31721_32153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'call_assignment_31721')
        # Assigning a type to the variable 'obj_type_stmts' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'obj_type_stmts', call_assignment_31721_32153)
        
        # Assigning a Call to a Name (line 135):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_31720' (line 135)
        call_assignment_31720_32154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'call_assignment_31720', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_32155 = stypy_get_value_from_tuple(call_assignment_31720_32154, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_31722' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'call_assignment_31722', stypy_get_value_from_tuple_call_result_32155)
        
        # Assigning a Name to a Name (line 135):
        # Getting the type of 'call_assignment_31722' (line 135)
        call_assignment_31722_32156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'call_assignment_31722')
        # Assigning a type to the variable 'obj_var' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'obj_var', call_assignment_31722_32156)
        
        # Assigning a Call to a Tuple (line 137):
        
        # Assigning a Call to a Name:
        
        # Call to create_get_type_of_member(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'obj_var' (line 137)
        obj_var_32159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 81), 'obj_var', False)
        # Getting the type of 'param' (line 137)
        param_32160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 90), 'param', False)
        # Obtaining the member 'attr' of a type (line 137)
        attr_32161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 90), param_32160, 'attr')
        # Getting the type of 'lineno' (line 137)
        lineno_32162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 102), 'lineno', False)
        # Getting the type of 'col_offset' (line 137)
        col_offset_32163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 110), 'col_offset', False)
        # Processing the call keyword arguments (line 137)
        kwargs_32164 = {}
        # Getting the type of 'stypy_functions_copy' (line 137)
        stypy_functions_copy_32157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 34), 'stypy_functions_copy', False)
        # Obtaining the member 'create_get_type_of_member' of a type (line 137)
        create_get_type_of_member_32158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 34), stypy_functions_copy_32157, 'create_get_type_of_member')
        # Calling create_get_type_of_member(args, kwargs) (line 137)
        create_get_type_of_member_call_result_32165 = invoke(stypy.reporting.localization.Localization(__file__, 137, 34), create_get_type_of_member_32158, *[obj_var_32159, attr_32161, lineno_32162, col_offset_32163], **kwargs_32164)
        
        # Assigning a type to the variable 'call_assignment_31723' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'call_assignment_31723', create_get_type_of_member_call_result_32165)
        
        # Assigning a Call to a Name (line 137):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_31723' (line 137)
        call_assignment_31723_32166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'call_assignment_31723', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_32167 = stypy_get_value_from_tuple(call_assignment_31723_32166, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_31724' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'call_assignment_31724', stypy_get_value_from_tuple_call_result_32167)
        
        # Assigning a Name to a Name (line 137):
        # Getting the type of 'call_assignment_31724' (line 137)
        call_assignment_31724_32168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'call_assignment_31724')
        # Assigning a type to the variable 'att_type_stmts' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'att_type_stmts', call_assignment_31724_32168)
        
        # Assigning a Call to a Name (line 137):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_31723' (line 137)
        call_assignment_31723_32169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'call_assignment_31723', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_32170 = stypy_get_value_from_tuple(call_assignment_31723_32169, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_31725' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'call_assignment_31725', stypy_get_value_from_tuple_call_result_32170)
        
        # Assigning a Name to a Name (line 137):
        # Getting the type of 'call_assignment_31725' (line 137)
        call_assignment_31725_32171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'call_assignment_31725')
        # Assigning a type to the variable 'att_var' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 24), 'att_var', call_assignment_31725_32171)
        
        # Assigning a Call to a Name (line 138):
        
        # Assigning a Call to a Name (line 138):
        
        # Call to create_call(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Call to create_Name(...): (line 138)
        # Processing the call arguments (line 138)
        str_32176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 85), 'str', 'remove_type_from_union')
        # Processing the call keyword arguments (line 138)
        kwargs_32177 = {}
        # Getting the type of 'core_language_copy' (line 138)
        core_language_copy_32174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 54), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 138)
        create_Name_32175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 54), core_language_copy_32174, 'create_Name')
        # Calling create_Name(args, kwargs) (line 138)
        create_Name_call_result_32178 = invoke(stypy.reporting.localization.Localization(__file__, 138, 54), create_Name_32175, *[str_32176], **kwargs_32177)
        
        
        # Obtaining an instance of the builtin type 'list' (line 139)
        list_32179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 139)
        # Adding element type (line 139)
        # Getting the type of 'att_var' (line 139)
        att_var_32180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 50), 'att_var', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 49), list_32179, att_var_32180)
        # Adding element type (line 139)
        # Getting the type of 'type_' (line 139)
        type__32181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 59), 'type_', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 49), list_32179, type__32181)
        
        # Processing the call keyword arguments (line 138)
        # Getting the type of 'lineno' (line 139)
        lineno_32182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 72), 'lineno', False)
        keyword_32183 = lineno_32182
        # Getting the type of 'col_offset' (line 139)
        col_offset_32184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 87), 'col_offset', False)
        keyword_32185 = col_offset_32184
        kwargs_32186 = {'column': keyword_32185, 'line': keyword_32183}
        # Getting the type of 'functions_copy' (line 138)
        functions_copy_32172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 27), 'functions_copy', False)
        # Obtaining the member 'create_call' of a type (line 138)
        create_call_32173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 27), functions_copy_32172, 'create_call')
        # Calling create_call(args, kwargs) (line 138)
        create_call_call_result_32187 = invoke(stypy.reporting.localization.Localization(__file__, 138, 27), create_call_32173, *[create_Name_call_result_32178, list_32179], **kwargs_32186)
        
        # Assigning a type to the variable 'remove_type_call' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'remove_type_call', create_call_call_result_32187)
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to create_set_type_of_member(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'obj_var' (line 140)
        obj_var_32190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 68), 'obj_var', False)
        # Getting the type of 'param' (line 140)
        param_32191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 77), 'param', False)
        # Obtaining the member 'attr' of a type (line 140)
        attr_32192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 77), param_32191, 'attr')
        # Getting the type of 'remove_type_call' (line 140)
        remove_type_call_32193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 89), 'remove_type_call', False)
        # Getting the type of 'lineno' (line 140)
        lineno_32194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 107), 'lineno', False)
        # Getting the type of 'col_offset' (line 141)
        col_offset_32195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 63), 'col_offset', False)
        # Processing the call keyword arguments (line 140)
        kwargs_32196 = {}
        # Getting the type of 'stypy_functions_copy' (line 140)
        stypy_functions_copy_32188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 21), 'stypy_functions_copy', False)
        # Obtaining the member 'create_set_type_of_member' of a type (line 140)
        create_set_type_of_member_32189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 21), stypy_functions_copy_32188, 'create_set_type_of_member')
        # Calling create_set_type_of_member(args, kwargs) (line 140)
        create_set_type_of_member_call_result_32197 = invoke(stypy.reporting.localization.Localization(__file__, 140, 21), create_set_type_of_member_32189, *[obj_var_32190, attr_32192, remove_type_call_32193, lineno_32194, col_offset_32195], **kwargs_32196)
        
        # Assigning a type to the variable 'set_member' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'set_member', create_set_type_of_member_call_result_32197)
        
        # Call to flatten_lists(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'obj_type_stmts' (line 142)
        obj_type_stmts_32200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 50), 'obj_type_stmts', False)
        # Getting the type of 'att_type_stmts' (line 142)
        att_type_stmts_32201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 66), 'att_type_stmts', False)
        # Getting the type of 'set_member' (line 142)
        set_member_32202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 82), 'set_member', False)
        # Processing the call keyword arguments (line 142)
        kwargs_32203 = {}
        # Getting the type of 'stypy_functions_copy' (line 142)
        stypy_functions_copy_32198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'stypy_functions_copy', False)
        # Obtaining the member 'flatten_lists' of a type (line 142)
        flatten_lists_32199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 15), stypy_functions_copy_32198, 'flatten_lists')
        # Calling flatten_lists(args, kwargs) (line 142)
        flatten_lists_call_result_32204 = invoke(stypy.reporting.localization.Localization(__file__, 142, 15), flatten_lists_32199, *[obj_type_stmts_32200, att_type_stmts_32201, set_member_32202], **kwargs_32203)
        
        # Assigning a type to the variable 'stypy_return_type' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'stypy_return_type', flatten_lists_call_result_32204)
        # SSA join for if statement (line 133)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Obtaining an instance of the builtin type 'list' (line 144)
    list_32205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 144)
    
    # Assigning a type to the variable 'stypy_return_type' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type', list_32205)
    
    # ################# End of '__remove_type_from_union_implementation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__remove_type_from_union_implementation' in the type store
    # Getting the type of 'stypy_return_type' (line 124)
    stypy_return_type_32206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32206)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__remove_type_from_union_implementation'
    return stypy_return_type_32206

# Assigning a type to the variable '__remove_type_from_union_implementation' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), '__remove_type_from_union_implementation', __remove_type_from_union_implementation)

@norecursion
def set_type_of_idiom_var(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'set_type_of_idiom_var'
    module_type_store = module_type_store.open_function_context('set_type_of_idiom_var', 147, 0, False)
    
    # Passed parameters checking function
    set_type_of_idiom_var.stypy_localization = localization
    set_type_of_idiom_var.stypy_type_of_self = None
    set_type_of_idiom_var.stypy_type_store = module_type_store
    set_type_of_idiom_var.stypy_function_name = 'set_type_of_idiom_var'
    set_type_of_idiom_var.stypy_param_names_list = ['idiom_name', 'if_branch', 'if_test', 'type_', 'lineno', 'col_offset']
    set_type_of_idiom_var.stypy_varargs_param_name = None
    set_type_of_idiom_var.stypy_kwargs_param_name = None
    set_type_of_idiom_var.stypy_call_defaults = defaults
    set_type_of_idiom_var.stypy_call_varargs = varargs
    set_type_of_idiom_var.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'set_type_of_idiom_var', ['idiom_name', 'if_branch', 'if_test', 'type_', 'lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'set_type_of_idiom_var', localization, ['idiom_name', 'if_branch', 'if_test', 'type_', 'lineno', 'col_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'set_type_of_idiom_var(...)' code ##################

    
    # Getting the type of 'idiom_name' (line 148)
    idiom_name_32207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 7), 'idiom_name')
    str_32208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 21), 'str', 'type_is')
    # Applying the binary operator '==' (line 148)
    result_eq_32209 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 7), '==', idiom_name_32207, str_32208)
    
    # Testing if the type of an if condition is none (line 148)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 148, 4), result_eq_32209):
        pass
    else:
        
        # Testing the type of an if condition (line 148)
        if_condition_32210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 4), result_eq_32209)
        # Assigning a type to the variable 'if_condition_32210' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'if_condition_32210', if_condition_32210)
        # SSA begins for if statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'if_branch' (line 149)
        if_branch_32211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'if_branch')
        str_32212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 24), 'str', 'if')
        # Applying the binary operator '==' (line 149)
        result_eq_32213 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 11), '==', if_branch_32211, str_32212)
        
        # Testing if the type of an if condition is none (line 149)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 149, 8), result_eq_32213):
            pass
        else:
            
            # Testing the type of an if condition (line 149)
            if_condition_32214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 8), result_eq_32213)
            # Assigning a type to the variable 'if_condition_32214' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'if_condition_32214', if_condition_32214)
            # SSA begins for if statement (line 149)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __set_type_implementation(...): (line 150)
            # Processing the call arguments (line 150)
            # Getting the type of 'if_test' (line 150)
            if_test_32216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 45), 'if_test', False)
            # Getting the type of 'type_' (line 150)
            type__32217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 54), 'type_', False)
            # Getting the type of 'lineno' (line 150)
            lineno_32218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 61), 'lineno', False)
            # Getting the type of 'col_offset' (line 150)
            col_offset_32219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 69), 'col_offset', False)
            # Processing the call keyword arguments (line 150)
            kwargs_32220 = {}
            # Getting the type of '__set_type_implementation' (line 150)
            set_type_implementation_32215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), '__set_type_implementation', False)
            # Calling __set_type_implementation(args, kwargs) (line 150)
            set_type_implementation_call_result_32221 = invoke(stypy.reporting.localization.Localization(__file__, 150, 19), set_type_implementation_32215, *[if_test_32216, type__32217, lineno_32218, col_offset_32219], **kwargs_32220)
            
            # Assigning a type to the variable 'stypy_return_type' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'stypy_return_type', set_type_implementation_call_result_32221)
            # SSA join for if statement (line 149)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'if_branch' (line 151)
        if_branch_32222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'if_branch')
        str_32223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 24), 'str', 'else')
        # Applying the binary operator '==' (line 151)
        result_eq_32224 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 11), '==', if_branch_32222, str_32223)
        
        # Testing if the type of an if condition is none (line 151)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 151, 8), result_eq_32224):
            pass
        else:
            
            # Testing the type of an if condition (line 151)
            if_condition_32225 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 8), result_eq_32224)
            # Assigning a type to the variable 'if_condition_32225' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'if_condition_32225', if_condition_32225)
            # SSA begins for if statement (line 151)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __remove_type_from_union_implementation(...): (line 152)
            # Processing the call arguments (line 152)
            # Getting the type of 'if_test' (line 152)
            if_test_32227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 59), 'if_test', False)
            # Getting the type of 'type_' (line 152)
            type__32228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 68), 'type_', False)
            # Getting the type of 'lineno' (line 152)
            lineno_32229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 75), 'lineno', False)
            # Getting the type of 'col_offset' (line 152)
            col_offset_32230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 83), 'col_offset', False)
            # Processing the call keyword arguments (line 152)
            kwargs_32231 = {}
            # Getting the type of '__remove_type_from_union_implementation' (line 152)
            remove_type_from_union_implementation_32226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 19), '__remove_type_from_union_implementation', False)
            # Calling __remove_type_from_union_implementation(args, kwargs) (line 152)
            remove_type_from_union_implementation_call_result_32232 = invoke(stypy.reporting.localization.Localization(__file__, 152, 19), remove_type_from_union_implementation_32226, *[if_test_32227, type__32228, lineno_32229, col_offset_32230], **kwargs_32231)
            
            # Assigning a type to the variable 'stypy_return_type' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'stypy_return_type', remove_type_from_union_implementation_call_result_32232)
            # SSA join for if statement (line 151)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 148)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'idiom_name' (line 154)
    idiom_name_32233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 7), 'idiom_name')
    str_32234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 21), 'str', 'not_type_is')
    # Applying the binary operator '==' (line 154)
    result_eq_32235 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 7), '==', idiom_name_32233, str_32234)
    
    # Testing if the type of an if condition is none (line 154)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 154, 4), result_eq_32235):
        pass
    else:
        
        # Testing the type of an if condition (line 154)
        if_condition_32236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 4), result_eq_32235)
        # Assigning a type to the variable 'if_condition_32236' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'if_condition_32236', if_condition_32236)
        # SSA begins for if statement (line 154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 155):
        
        # Assigning a Attribute to a Name (line 155):
        # Getting the type of 'if_test' (line 155)
        if_test_32237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 18), 'if_test')
        # Obtaining the member 'operand' of a type (line 155)
        operand_32238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 18), if_test_32237, 'operand')
        # Assigning a type to the variable 'if_test' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'if_test', operand_32238)
        
        # Getting the type of 'if_branch' (line 156)
        if_branch_32239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 11), 'if_branch')
        str_32240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 24), 'str', 'if')
        # Applying the binary operator '==' (line 156)
        result_eq_32241 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 11), '==', if_branch_32239, str_32240)
        
        # Testing if the type of an if condition is none (line 156)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 156, 8), result_eq_32241):
            pass
        else:
            
            # Testing the type of an if condition (line 156)
            if_condition_32242 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 8), result_eq_32241)
            # Assigning a type to the variable 'if_condition_32242' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'if_condition_32242', if_condition_32242)
            # SSA begins for if statement (line 156)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __remove_type_from_union_implementation(...): (line 157)
            # Processing the call arguments (line 157)
            # Getting the type of 'if_test' (line 157)
            if_test_32244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 59), 'if_test', False)
            # Getting the type of 'type_' (line 157)
            type__32245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 68), 'type_', False)
            # Getting the type of 'lineno' (line 157)
            lineno_32246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 75), 'lineno', False)
            # Getting the type of 'col_offset' (line 157)
            col_offset_32247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 83), 'col_offset', False)
            # Processing the call keyword arguments (line 157)
            kwargs_32248 = {}
            # Getting the type of '__remove_type_from_union_implementation' (line 157)
            remove_type_from_union_implementation_32243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 19), '__remove_type_from_union_implementation', False)
            # Calling __remove_type_from_union_implementation(args, kwargs) (line 157)
            remove_type_from_union_implementation_call_result_32249 = invoke(stypy.reporting.localization.Localization(__file__, 157, 19), remove_type_from_union_implementation_32243, *[if_test_32244, type__32245, lineno_32246, col_offset_32247], **kwargs_32248)
            
            # Assigning a type to the variable 'stypy_return_type' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'stypy_return_type', remove_type_from_union_implementation_call_result_32249)
            # SSA join for if statement (line 156)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'if_branch' (line 158)
        if_branch_32250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'if_branch')
        str_32251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 24), 'str', 'else')
        # Applying the binary operator '==' (line 158)
        result_eq_32252 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 11), '==', if_branch_32250, str_32251)
        
        # Testing if the type of an if condition is none (line 158)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 158, 8), result_eq_32252):
            pass
        else:
            
            # Testing the type of an if condition (line 158)
            if_condition_32253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 8), result_eq_32252)
            # Assigning a type to the variable 'if_condition_32253' (line 158)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'if_condition_32253', if_condition_32253)
            # SSA begins for if statement (line 158)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __set_type_implementation(...): (line 159)
            # Processing the call arguments (line 159)
            # Getting the type of 'if_test' (line 159)
            if_test_32255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 45), 'if_test', False)
            # Getting the type of 'type_' (line 159)
            type__32256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 54), 'type_', False)
            # Getting the type of 'lineno' (line 159)
            lineno_32257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 61), 'lineno', False)
            # Getting the type of 'col_offset' (line 159)
            col_offset_32258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 69), 'col_offset', False)
            # Processing the call keyword arguments (line 159)
            kwargs_32259 = {}
            # Getting the type of '__set_type_implementation' (line 159)
            set_type_implementation_32254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), '__set_type_implementation', False)
            # Calling __set_type_implementation(args, kwargs) (line 159)
            set_type_implementation_call_result_32260 = invoke(stypy.reporting.localization.Localization(__file__, 159, 19), set_type_implementation_32254, *[if_test_32255, type__32256, lineno_32257, col_offset_32258], **kwargs_32259)
            
            # Assigning a type to the variable 'stypy_return_type' (line 159)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'stypy_return_type', set_type_implementation_call_result_32260)
            # SSA join for if statement (line 158)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 154)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Obtaining an instance of the builtin type 'list' (line 161)
    list_32261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 161)
    
    # Assigning a type to the variable 'stypy_return_type' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type', list_32261)
    
    # ################# End of 'set_type_of_idiom_var(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'set_type_of_idiom_var' in the type store
    # Getting the type of 'stypy_return_type' (line 147)
    stypy_return_type_32262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32262)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'set_type_of_idiom_var'
    return stypy_return_type_32262

# Assigning a type to the variable 'set_type_of_idiom_var' (line 147)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'set_type_of_idiom_var', set_type_of_idiom_var)

# Assigning a Dict to a Name (line 165):

# Assigning a Dict to a Name (line 165):

# Obtaining an instance of the builtin type 'dict' (line 165)
dict_32263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 20), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 165)
# Adding element type (key, value) (line 165)
str_32264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 4), 'str', 'type_is')
# Getting the type of 'type_is_idiom' (line 166)
type_is_idiom_32265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'type_is_idiom')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 20), dict_32263, (str_32264, type_is_idiom_32265))
# Adding element type (key, value) (line 165)
str_32266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 4), 'str', 'not_type_is')
# Getting the type of 'not_type_is_idiom' (line 167)
not_type_is_idiom_32267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 19), 'not_type_is_idiom')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 20), dict_32263, (str_32266, not_type_is_idiom_32267))

# Assigning a type to the variable 'recognized_idioms' (line 165)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'recognized_idioms', dict_32263)

# Assigning a Dict to a Name (line 171):

# Assigning a Dict to a Name (line 171):

# Obtaining an instance of the builtin type 'dict' (line 171)
dict_32268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 30), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 171)
# Adding element type (key, value) (line 171)
str_32269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 4), 'str', 'type_is')
# Getting the type of 'may_be_type_func_name' (line 172)
may_be_type_func_name_32270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'may_be_type_func_name')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 30), dict_32268, (str_32269, may_be_type_func_name_32270))
# Adding element type (key, value) (line 171)
str_32271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 4), 'str', 'not_type_is')
# Getting the type of 'may_not_be_type_func_name' (line 173)
may_not_be_type_func_name_32272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'may_not_be_type_func_name')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 30), dict_32268, (str_32271, may_not_be_type_func_name_32272))

# Assigning a type to the variable 'recognized_idioms_functions' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'recognized_idioms_functions', dict_32268)

@norecursion
def get_recognized_idiom_function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_recognized_idiom_function'
    module_type_store = module_type_store.open_function_context('get_recognized_idiom_function', 177, 0, False)
    
    # Passed parameters checking function
    get_recognized_idiom_function.stypy_localization = localization
    get_recognized_idiom_function.stypy_type_of_self = None
    get_recognized_idiom_function.stypy_type_store = module_type_store
    get_recognized_idiom_function.stypy_function_name = 'get_recognized_idiom_function'
    get_recognized_idiom_function.stypy_param_names_list = ['idiom_name']
    get_recognized_idiom_function.stypy_varargs_param_name = None
    get_recognized_idiom_function.stypy_kwargs_param_name = None
    get_recognized_idiom_function.stypy_call_defaults = defaults
    get_recognized_idiom_function.stypy_call_varargs = varargs
    get_recognized_idiom_function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_recognized_idiom_function', ['idiom_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_recognized_idiom_function', localization, ['idiom_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_recognized_idiom_function(...)' code ##################

    str_32273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, (-1)), 'str', '\n    Gets the function that process an idiom once it has been recognized\n    :param idiom_name: Idiom name\n    :return:\n    ')
    
    # Obtaining the type of the subscript
    # Getting the type of 'idiom_name' (line 183)
    idiom_name_32274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 39), 'idiom_name')
    # Getting the type of 'recognized_idioms_functions' (line 183)
    recognized_idioms_functions_32275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 11), 'recognized_idioms_functions')
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___32276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 11), recognized_idioms_functions_32275, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_32277 = invoke(stypy.reporting.localization.Localization(__file__, 183, 11), getitem___32276, idiom_name_32274)
    
    # Assigning a type to the variable 'stypy_return_type' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type', subscript_call_result_32277)
    
    # ################# End of 'get_recognized_idiom_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_recognized_idiom_function' in the type store
    # Getting the type of 'stypy_return_type' (line 177)
    stypy_return_type_32278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32278)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_recognized_idiom_function'
    return stypy_return_type_32278

# Assigning a type to the variable 'get_recognized_idiom_function' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'get_recognized_idiom_function', get_recognized_idiom_function)

@norecursion
def is_recognized_idiom(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_recognized_idiom'
    module_type_store = module_type_store.open_function_context('is_recognized_idiom', 186, 0, False)
    
    # Passed parameters checking function
    is_recognized_idiom.stypy_localization = localization
    is_recognized_idiom.stypy_type_of_self = None
    is_recognized_idiom.stypy_type_store = module_type_store
    is_recognized_idiom.stypy_function_name = 'is_recognized_idiom'
    is_recognized_idiom.stypy_param_names_list = ['test', 'visitor', 'context']
    is_recognized_idiom.stypy_varargs_param_name = None
    is_recognized_idiom.stypy_kwargs_param_name = None
    is_recognized_idiom.stypy_call_defaults = defaults
    is_recognized_idiom.stypy_call_varargs = varargs
    is_recognized_idiom.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_recognized_idiom', ['test', 'visitor', 'context'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_recognized_idiom', localization, ['test', 'visitor', 'context'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_recognized_idiom(...)' code ##################

    str_32279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, (-1)), 'str', '\n    Check if the passed test can be considered an idioms\n\n    :param test: Source code test\n    :param visitor: Type inference visitor, to change generated instructions\n    :param context: Context passed to the call\n    :return: Tuple of values that identify if an idiom has been recognized and calculated data if it is been recognized\n    ')
    
    # Getting the type of 'recognized_idioms' (line 195)
    recognized_idioms_32280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 17), 'recognized_idioms')
    # Assigning a type to the variable 'recognized_idioms_32280' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'recognized_idioms_32280', recognized_idioms_32280)
    # Testing if the for loop is going to be iterated (line 195)
    # Testing the type of a for loop iterable (line 195)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 195, 4), recognized_idioms_32280)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 195, 4), recognized_idioms_32280):
        # Getting the type of the for loop variable (line 195)
        for_loop_var_32281 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 195, 4), recognized_idioms_32280)
        # Assigning a type to the variable 'idiom' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'idiom', for_loop_var_32281)
        # SSA begins for a for statement (line 195)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to (...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'test' (line 196)
        test_32286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 42), 'test', False)
        # Getting the type of 'visitor' (line 196)
        visitor_32287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 48), 'visitor', False)
        # Getting the type of 'context' (line 196)
        context_32288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 57), 'context', False)
        # Processing the call keyword arguments (line 196)
        kwargs_32289 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'idiom' (line 196)
        idiom_32282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 35), 'idiom', False)
        # Getting the type of 'recognized_idioms' (line 196)
        recognized_idioms_32283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 17), 'recognized_idioms', False)
        # Obtaining the member '__getitem__' of a type (line 196)
        getitem___32284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 17), recognized_idioms_32283, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 196)
        subscript_call_result_32285 = invoke(stypy.reporting.localization.Localization(__file__, 196, 17), getitem___32284, idiom_32282)
        
        # Calling (args, kwargs) (line 196)
        _call_result_32290 = invoke(stypy.reporting.localization.Localization(__file__, 196, 17), subscript_call_result_32285, *[test_32286, visitor_32287, context_32288], **kwargs_32289)
        
        # Assigning a type to the variable 'result' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'result', _call_result_32290)
        
        # Obtaining the type of the subscript
        int_32291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 18), 'int')
        # Getting the type of 'result' (line 197)
        result_32292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 11), 'result')
        # Obtaining the member '__getitem__' of a type (line 197)
        getitem___32293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 11), result_32292, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 197)
        subscript_call_result_32294 = invoke(stypy.reporting.localization.Localization(__file__, 197, 11), getitem___32293, int_32291)
        
        # Testing if the type of an if condition is none (line 197)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 197, 8), subscript_call_result_32294):
            pass
        else:
            
            # Testing the type of an if condition (line 197)
            if_condition_32295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 8), subscript_call_result_32294)
            # Assigning a type to the variable 'if_condition_32295' (line 197)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'if_condition_32295', if_condition_32295)
            # SSA begins for if statement (line 197)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 198):
            
            # Assigning a Call to a Name (line 198):
            
            # Call to list(...): (line 198)
            # Processing the call arguments (line 198)
            # Getting the type of 'result' (line 198)
            result_32297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 29), 'result', False)
            # Processing the call keyword arguments (line 198)
            kwargs_32298 = {}
            # Getting the type of 'list' (line 198)
            list_32296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 24), 'list', False)
            # Calling list(args, kwargs) (line 198)
            list_call_result_32299 = invoke(stypy.reporting.localization.Localization(__file__, 198, 24), list_32296, *[result_32297], **kwargs_32298)
            
            # Assigning a type to the variable 'temp_list' (line 198)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'temp_list', list_call_result_32299)
            
            # Call to append(...): (line 199)
            # Processing the call arguments (line 199)
            # Getting the type of 'idiom' (line 199)
            idiom_32302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 29), 'idiom', False)
            # Processing the call keyword arguments (line 199)
            kwargs_32303 = {}
            # Getting the type of 'temp_list' (line 199)
            temp_list_32300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'temp_list', False)
            # Obtaining the member 'append' of a type (line 199)
            append_32301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 12), temp_list_32300, 'append')
            # Calling append(args, kwargs) (line 199)
            append_call_result_32304 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), append_32301, *[idiom_32302], **kwargs_32303)
            
            
            # Call to tuple(...): (line 200)
            # Processing the call arguments (line 200)
            # Getting the type of 'temp_list' (line 200)
            temp_list_32306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 25), 'temp_list', False)
            # Processing the call keyword arguments (line 200)
            kwargs_32307 = {}
            # Getting the type of 'tuple' (line 200)
            tuple_32305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 19), 'tuple', False)
            # Calling tuple(args, kwargs) (line 200)
            tuple_call_result_32308 = invoke(stypy.reporting.localization.Localization(__file__, 200, 19), tuple_32305, *[temp_list_32306], **kwargs_32307)
            
            # Assigning a type to the variable 'stypy_return_type' (line 200)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'stypy_return_type', tuple_call_result_32308)
            # SSA join for if statement (line 197)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 202)
    tuple_32309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 202)
    # Adding element type (line 202)
    # Getting the type of 'False' (line 202)
    False_32310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 11), tuple_32309, False_32310)
    # Adding element type (line 202)
    # Getting the type of 'None' (line 202)
    None_32311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 11), tuple_32309, None_32311)
    # Adding element type (line 202)
    # Getting the type of 'None' (line 202)
    None_32312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 24), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 11), tuple_32309, None_32312)
    # Adding element type (line 202)
    # Getting the type of 'None' (line 202)
    None_32313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 30), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 11), tuple_32309, None_32313)
    
    # Assigning a type to the variable 'stypy_return_type' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stypy_return_type', tuple_32309)
    
    # ################# End of 'is_recognized_idiom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_recognized_idiom' in the type store
    # Getting the type of 'stypy_return_type' (line 186)
    stypy_return_type_32314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32314)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_recognized_idiom'
    return stypy_return_type_32314

# Assigning a type to the variable 'is_recognized_idiom' (line 186)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'is_recognized_idiom', is_recognized_idiom)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
