
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
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_16161 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_functions_copy')

if (type(import_16161) is not StypyTypeError):

    if (import_16161 != 'pyd_module'):
        __import__(import_16161)
        sys_modules_16162 = sys.modules[import_16161]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_functions_copy', sys_modules_16162.module_type_store, module_type_store)
    else:
        import stypy_functions_copy

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_functions_copy', stypy_functions_copy, module_type_store)

else:
    # Assigning a type to the variable 'stypy_functions_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_functions_copy', import_16161)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import functions_copy' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_16163 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'functions_copy')

if (type(import_16163) is not StypyTypeError):

    if (import_16163 != 'pyd_module'):
        __import__(import_16163)
        sys_modules_16164 = sys.modules[import_16163]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'functions_copy', sys_modules_16164.module_type_store, module_type_store)
    else:
        import functions_copy

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'functions_copy', functions_copy, module_type_store)

else:
    # Assigning a type to the variable 'functions_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'functions_copy', import_16163)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import core_language_copy' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_16165 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'core_language_copy')

if (type(import_16165) is not StypyTypeError):

    if (import_16165 != 'pyd_module'):
        __import__(import_16165)
        sys_modules_16166 = sys.modules[import_16165]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'core_language_copy', sys_modules_16166.module_type_store, module_type_store)
    else:
        import core_language_copy

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'core_language_copy', core_language_copy, module_type_store)

else:
    # Assigning a type to the variable 'core_language_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'core_language_copy', import_16165)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

str_16167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'str', '\nCode that deals with various code idioms that can be optimized to better obtain the types of the variables used\non these idioms. The work with this file is unfinished, as not all the intended idioms are supported.\n\nTODO: Finish this and its comments when idioms are fully implemented\n')

# Assigning a Tuple to a Name (line 18):

# Assigning a Tuple to a Name (line 18):

# Obtaining an instance of the builtin type 'tuple' (line 18)
tuple_16168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 20), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 18)
# Adding element type (line 18)
# Getting the type of 'False' (line 18)
False_16169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'False')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 20), tuple_16168, False_16169)
# Adding element type (line 18)
# Getting the type of 'None' (line 18)
None_16170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 27), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 20), tuple_16168, None_16170)
# Adding element type (line 18)
# Getting the type of 'None' (line 18)
None_16171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 33), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 20), tuple_16168, None_16171)

# Assigning a type to the variable 'default_ret_tuple' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'default_ret_tuple', tuple_16168)

# Assigning a Str to a Name (line 20):

# Assigning a Str to a Name (line 20):
str_16172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 24), 'str', 'may_be_type')
# Assigning a type to the variable 'may_be_type_func_name' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'may_be_type_func_name', str_16172)

# Assigning a Str to a Name (line 21):

# Assigning a Str to a Name (line 21):
str_16173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'str', 'may_not_be_type')
# Assigning a type to the variable 'may_not_be_type_func_name' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'may_not_be_type_func_name', str_16173)

# Assigning a Str to a Name (line 22):

# Assigning a Str to a Name (line 22):
str_16174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'str', '__may_be')
# Assigning a type to the variable 'may_be_var_name' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'may_be_var_name', str_16174)

# Assigning a Str to a Name (line 23):

# Assigning a Str to a Name (line 23):
str_16175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 22), 'str', '__more_types_in_union')
# Assigning a type to the variable 'more_types_var_name' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'more_types_var_name', str_16175)

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
    test_16177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'test', False)
    # Processing the call keyword arguments (line 27)
    kwargs_16178 = {}
    # Getting the type of 'type' (line 27)
    type_16176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 7), 'type', False)
    # Calling type(args, kwargs) (line 27)
    type_call_result_16179 = invoke(stypy.reporting.localization.Localization(__file__, 27, 7), type_16176, *[test_16177], **kwargs_16178)
    
    # Getting the type of 'ast' (line 27)
    ast_16180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'ast')
    # Obtaining the member 'Call' of a type (line 27)
    Call_16181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 21), ast_16180, 'Call')
    # Applying the binary operator 'is' (line 27)
    result_is__16182 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 7), 'is', type_call_result_16179, Call_16181)
    
    # Testing if the type of an if condition is none (line 27)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 27, 4), result_is__16182):
        
        # Type idiom detected: calculating its left and rigth part (line 34)
        str_16209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'str', 'left')
        # Getting the type of 'test' (line 34)
        test_16210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'test')
        
        (may_be_16211, more_types_in_union_16212) = may_provide_member(str_16209, test_16210)

        if may_be_16211:

            if more_types_in_union_16212:
                # Runtime conditional SSA (line 34)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'test' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'test', remove_not_member_provider_from_union(test_16210, 'left'))
            
            
            # Call to type(...): (line 35)
            # Processing the call arguments (line 35)
            # Getting the type of 'test' (line 35)
            test_16214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'test', False)
            # Obtaining the member 'left' of a type (line 35)
            left_16215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 20), test_16214, 'left')
            # Processing the call keyword arguments (line 35)
            kwargs_16216 = {}
            # Getting the type of 'type' (line 35)
            type_16213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'type', False)
            # Calling type(args, kwargs) (line 35)
            type_call_result_16217 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), type_16213, *[left_16215], **kwargs_16216)
            
            # Getting the type of 'ast' (line 35)
            ast_16218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'ast')
            # Obtaining the member 'Call' of a type (line 35)
            Call_16219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 34), ast_16218, 'Call')
            # Applying the binary operator 'is' (line 35)
            result_is__16220 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 15), 'is', type_call_result_16217, Call_16219)
            
            # Testing if the type of an if condition is none (line 35)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 35, 12), result_is__16220):
                pass
            else:
                
                # Testing the type of an if condition (line 35)
                if_condition_16221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 12), result_is__16220)
                # Assigning a type to the variable 'if_condition_16221' (line 35)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'if_condition_16221', if_condition_16221)
                # SSA begins for if statement (line 35)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Call to len(...): (line 36)
                # Processing the call arguments (line 36)
                # Getting the type of 'test' (line 36)
                test_16223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'test', False)
                # Obtaining the member 'comparators' of a type (line 36)
                comparators_16224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 23), test_16223, 'comparators')
                # Processing the call keyword arguments (line 36)
                kwargs_16225 = {}
                # Getting the type of 'len' (line 36)
                len_16222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'len', False)
                # Calling len(args, kwargs) (line 36)
                len_call_result_16226 = invoke(stypy.reporting.localization.Localization(__file__, 36, 19), len_16222, *[comparators_16224], **kwargs_16225)
                
                int_16227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 44), 'int')
                # Applying the binary operator '!=' (line 36)
                result_ne_16228 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 19), '!=', len_call_result_16226, int_16227)
                
                # Testing if the type of an if condition is none (line 36)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 36, 16), result_ne_16228):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 36)
                    if_condition_16229 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 16), result_ne_16228)
                    # Assigning a type to the variable 'if_condition_16229' (line 36)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'if_condition_16229', if_condition_16229)
                    # SSA begins for if statement (line 36)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 37)
                    False_16230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 37)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'stypy_return_type', False_16230)
                    # SSA join for if statement (line 36)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                
                # Call to type(...): (line 38)
                # Processing the call arguments (line 38)
                # Getting the type of 'test' (line 38)
                test_16232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'test', False)
                # Obtaining the member 'left' of a type (line 38)
                left_16233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 24), test_16232, 'left')
                # Obtaining the member 'func' of a type (line 38)
                func_16234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 24), left_16233, 'func')
                # Processing the call keyword arguments (line 38)
                kwargs_16235 = {}
                # Getting the type of 'type' (line 38)
                type_16231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'type', False)
                # Calling type(args, kwargs) (line 38)
                type_call_result_16236 = invoke(stypy.reporting.localization.Localization(__file__, 38, 19), type_16231, *[func_16234], **kwargs_16235)
                
                # Getting the type of 'ast' (line 38)
                ast_16237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 43), 'ast')
                # Obtaining the member 'Name' of a type (line 38)
                Name_16238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 43), ast_16237, 'Name')
                # Applying the binary operator 'is' (line 38)
                result_is__16239 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 19), 'is', type_call_result_16236, Name_16238)
                
                # Testing if the type of an if condition is none (line 38)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 38, 16), result_is__16239):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 38)
                    if_condition_16240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 16), result_is__16239)
                    # Assigning a type to the variable 'if_condition_16240' (line 38)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'if_condition_16240', if_condition_16240)
                    # SSA begins for if statement (line 38)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    
                    # Call to len(...): (line 39)
                    # Processing the call arguments (line 39)
                    # Getting the type of 'test' (line 39)
                    test_16242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'test', False)
                    # Obtaining the member 'left' of a type (line 39)
                    left_16243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 27), test_16242, 'left')
                    # Obtaining the member 'args' of a type (line 39)
                    args_16244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 27), left_16243, 'args')
                    # Processing the call keyword arguments (line 39)
                    kwargs_16245 = {}
                    # Getting the type of 'len' (line 39)
                    len_16241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'len', False)
                    # Calling len(args, kwargs) (line 39)
                    len_call_result_16246 = invoke(stypy.reporting.localization.Localization(__file__, 39, 23), len_16241, *[args_16244], **kwargs_16245)
                    
                    int_16247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 46), 'int')
                    # Applying the binary operator '!=' (line 39)
                    result_ne_16248 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 23), '!=', len_call_result_16246, int_16247)
                    
                    # Testing if the type of an if condition is none (line 39)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 39, 20), result_ne_16248):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 39)
                        if_condition_16249 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 20), result_ne_16248)
                        # Assigning a type to the variable 'if_condition_16249' (line 39)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'if_condition_16249', if_condition_16249)
                        # SSA begins for if statement (line 39)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'False' (line 40)
                        False_16250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 31), 'False')
                        # Assigning a type to the variable 'stypy_return_type' (line 40)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'stypy_return_type', False_16250)
                        # SSA join for if statement (line 39)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Getting the type of 'test' (line 41)
                    test_16251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'test')
                    # Obtaining the member 'left' of a type (line 41)
                    left_16252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), test_16251, 'left')
                    # Obtaining the member 'func' of a type (line 41)
                    func_16253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), left_16252, 'func')
                    # Obtaining the member 'id' of a type (line 41)
                    id_16254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), func_16253, 'id')
                    str_16255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 44), 'str', 'type')
                    # Applying the binary operator '==' (line 41)
                    result_eq_16256 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 23), '==', id_16254, str_16255)
                    
                    # Testing if the type of an if condition is none (line 41)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 41, 20), result_eq_16256):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 41)
                        if_condition_16257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 20), result_eq_16256)
                        # Assigning a type to the variable 'if_condition_16257' (line 41)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'if_condition_16257', if_condition_16257)
                        # SSA begins for if statement (line 41)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'True' (line 42)
                        True_16258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 31), 'True')
                        # Assigning a type to the variable 'stypy_return_type' (line 42)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'stypy_return_type', True_16258)
                        # SSA join for if statement (line 41)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 38)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 35)
                module_type_store = module_type_store.join_ssa_context()
                


            if more_types_in_union_16212:
                # SSA join for if statement (line 34)
                module_type_store = module_type_store.join_ssa_context()


        
    else:
        
        # Testing the type of an if condition (line 27)
        if_condition_16183 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 4), result_is__16182)
        # Assigning a type to the variable 'if_condition_16183' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'if_condition_16183', if_condition_16183)
        # SSA begins for if statement (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to type(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'test' (line 28)
        test_16185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'test', False)
        # Obtaining the member 'func' of a type (line 28)
        func_16186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), test_16185, 'func')
        # Processing the call keyword arguments (line 28)
        kwargs_16187 = {}
        # Getting the type of 'type' (line 28)
        type_16184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'type', False)
        # Calling type(args, kwargs) (line 28)
        type_call_result_16188 = invoke(stypy.reporting.localization.Localization(__file__, 28, 11), type_16184, *[func_16186], **kwargs_16187)
        
        # Getting the type of 'ast' (line 28)
        ast_16189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'ast')
        # Obtaining the member 'Name' of a type (line 28)
        Name_16190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 30), ast_16189, 'Name')
        # Applying the binary operator 'is' (line 28)
        result_is__16191 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 11), 'is', type_call_result_16188, Name_16190)
        
        # Testing if the type of an if condition is none (line 28)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 28, 8), result_is__16191):
            pass
        else:
            
            # Testing the type of an if condition (line 28)
            if_condition_16192 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 8), result_is__16191)
            # Assigning a type to the variable 'if_condition_16192' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'if_condition_16192', if_condition_16192)
            # SSA begins for if statement (line 28)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to len(...): (line 29)
            # Processing the call arguments (line 29)
            # Getting the type of 'test' (line 29)
            test_16194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'test', False)
            # Obtaining the member 'args' of a type (line 29)
            args_16195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 19), test_16194, 'args')
            # Processing the call keyword arguments (line 29)
            kwargs_16196 = {}
            # Getting the type of 'len' (line 29)
            len_16193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'len', False)
            # Calling len(args, kwargs) (line 29)
            len_call_result_16197 = invoke(stypy.reporting.localization.Localization(__file__, 29, 15), len_16193, *[args_16195], **kwargs_16196)
            
            int_16198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 33), 'int')
            # Applying the binary operator '!=' (line 29)
            result_ne_16199 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 15), '!=', len_call_result_16197, int_16198)
            
            # Testing if the type of an if condition is none (line 29)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 29, 12), result_ne_16199):
                pass
            else:
                
                # Testing the type of an if condition (line 29)
                if_condition_16200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 12), result_ne_16199)
                # Assigning a type to the variable 'if_condition_16200' (line 29)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'if_condition_16200', if_condition_16200)
                # SSA begins for if statement (line 29)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'False' (line 30)
                False_16201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 30)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'stypy_return_type', False_16201)
                # SSA join for if statement (line 29)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'test' (line 31)
            test_16202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'test')
            # Obtaining the member 'func' of a type (line 31)
            func_16203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 15), test_16202, 'func')
            # Obtaining the member 'id' of a type (line 31)
            id_16204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 15), func_16203, 'id')
            str_16205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'str', 'type')
            # Applying the binary operator '==' (line 31)
            result_eq_16206 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 15), '==', id_16204, str_16205)
            
            # Testing if the type of an if condition is none (line 31)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 31, 12), result_eq_16206):
                pass
            else:
                
                # Testing the type of an if condition (line 31)
                if_condition_16207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 12), result_eq_16206)
                # Assigning a type to the variable 'if_condition_16207' (line 31)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'if_condition_16207', if_condition_16207)
                # SSA begins for if statement (line 31)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'True' (line 32)
                True_16208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'True')
                # Assigning a type to the variable 'stypy_return_type' (line 32)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'stypy_return_type', True_16208)
                # SSA join for if statement (line 31)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 28)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the else part of an if statement (line 27)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 34)
        str_16209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 25), 'str', 'left')
        # Getting the type of 'test' (line 34)
        test_16210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'test')
        
        (may_be_16211, more_types_in_union_16212) = may_provide_member(str_16209, test_16210)

        if may_be_16211:

            if more_types_in_union_16212:
                # Runtime conditional SSA (line 34)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'test' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'test', remove_not_member_provider_from_union(test_16210, 'left'))
            
            
            # Call to type(...): (line 35)
            # Processing the call arguments (line 35)
            # Getting the type of 'test' (line 35)
            test_16214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'test', False)
            # Obtaining the member 'left' of a type (line 35)
            left_16215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 20), test_16214, 'left')
            # Processing the call keyword arguments (line 35)
            kwargs_16216 = {}
            # Getting the type of 'type' (line 35)
            type_16213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'type', False)
            # Calling type(args, kwargs) (line 35)
            type_call_result_16217 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), type_16213, *[left_16215], **kwargs_16216)
            
            # Getting the type of 'ast' (line 35)
            ast_16218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'ast')
            # Obtaining the member 'Call' of a type (line 35)
            Call_16219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 34), ast_16218, 'Call')
            # Applying the binary operator 'is' (line 35)
            result_is__16220 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 15), 'is', type_call_result_16217, Call_16219)
            
            # Testing if the type of an if condition is none (line 35)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 35, 12), result_is__16220):
                pass
            else:
                
                # Testing the type of an if condition (line 35)
                if_condition_16221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 12), result_is__16220)
                # Assigning a type to the variable 'if_condition_16221' (line 35)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'if_condition_16221', if_condition_16221)
                # SSA begins for if statement (line 35)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Call to len(...): (line 36)
                # Processing the call arguments (line 36)
                # Getting the type of 'test' (line 36)
                test_16223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'test', False)
                # Obtaining the member 'comparators' of a type (line 36)
                comparators_16224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 23), test_16223, 'comparators')
                # Processing the call keyword arguments (line 36)
                kwargs_16225 = {}
                # Getting the type of 'len' (line 36)
                len_16222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'len', False)
                # Calling len(args, kwargs) (line 36)
                len_call_result_16226 = invoke(stypy.reporting.localization.Localization(__file__, 36, 19), len_16222, *[comparators_16224], **kwargs_16225)
                
                int_16227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 44), 'int')
                # Applying the binary operator '!=' (line 36)
                result_ne_16228 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 19), '!=', len_call_result_16226, int_16227)
                
                # Testing if the type of an if condition is none (line 36)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 36, 16), result_ne_16228):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 36)
                    if_condition_16229 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 16), result_ne_16228)
                    # Assigning a type to the variable 'if_condition_16229' (line 36)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'if_condition_16229', if_condition_16229)
                    # SSA begins for if statement (line 36)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'False' (line 37)
                    False_16230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 27), 'False')
                    # Assigning a type to the variable 'stypy_return_type' (line 37)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'stypy_return_type', False_16230)
                    # SSA join for if statement (line 36)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                
                # Call to type(...): (line 38)
                # Processing the call arguments (line 38)
                # Getting the type of 'test' (line 38)
                test_16232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'test', False)
                # Obtaining the member 'left' of a type (line 38)
                left_16233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 24), test_16232, 'left')
                # Obtaining the member 'func' of a type (line 38)
                func_16234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 24), left_16233, 'func')
                # Processing the call keyword arguments (line 38)
                kwargs_16235 = {}
                # Getting the type of 'type' (line 38)
                type_16231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'type', False)
                # Calling type(args, kwargs) (line 38)
                type_call_result_16236 = invoke(stypy.reporting.localization.Localization(__file__, 38, 19), type_16231, *[func_16234], **kwargs_16235)
                
                # Getting the type of 'ast' (line 38)
                ast_16237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 43), 'ast')
                # Obtaining the member 'Name' of a type (line 38)
                Name_16238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 43), ast_16237, 'Name')
                # Applying the binary operator 'is' (line 38)
                result_is__16239 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 19), 'is', type_call_result_16236, Name_16238)
                
                # Testing if the type of an if condition is none (line 38)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 38, 16), result_is__16239):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 38)
                    if_condition_16240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 16), result_is__16239)
                    # Assigning a type to the variable 'if_condition_16240' (line 38)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'if_condition_16240', if_condition_16240)
                    # SSA begins for if statement (line 38)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    
                    # Call to len(...): (line 39)
                    # Processing the call arguments (line 39)
                    # Getting the type of 'test' (line 39)
                    test_16242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'test', False)
                    # Obtaining the member 'left' of a type (line 39)
                    left_16243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 27), test_16242, 'left')
                    # Obtaining the member 'args' of a type (line 39)
                    args_16244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 27), left_16243, 'args')
                    # Processing the call keyword arguments (line 39)
                    kwargs_16245 = {}
                    # Getting the type of 'len' (line 39)
                    len_16241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'len', False)
                    # Calling len(args, kwargs) (line 39)
                    len_call_result_16246 = invoke(stypy.reporting.localization.Localization(__file__, 39, 23), len_16241, *[args_16244], **kwargs_16245)
                    
                    int_16247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 46), 'int')
                    # Applying the binary operator '!=' (line 39)
                    result_ne_16248 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 23), '!=', len_call_result_16246, int_16247)
                    
                    # Testing if the type of an if condition is none (line 39)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 39, 20), result_ne_16248):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 39)
                        if_condition_16249 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 20), result_ne_16248)
                        # Assigning a type to the variable 'if_condition_16249' (line 39)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'if_condition_16249', if_condition_16249)
                        # SSA begins for if statement (line 39)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'False' (line 40)
                        False_16250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 31), 'False')
                        # Assigning a type to the variable 'stypy_return_type' (line 40)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'stypy_return_type', False_16250)
                        # SSA join for if statement (line 39)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Getting the type of 'test' (line 41)
                    test_16251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'test')
                    # Obtaining the member 'left' of a type (line 41)
                    left_16252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), test_16251, 'left')
                    # Obtaining the member 'func' of a type (line 41)
                    func_16253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), left_16252, 'func')
                    # Obtaining the member 'id' of a type (line 41)
                    id_16254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), func_16253, 'id')
                    str_16255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 44), 'str', 'type')
                    # Applying the binary operator '==' (line 41)
                    result_eq_16256 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 23), '==', id_16254, str_16255)
                    
                    # Testing if the type of an if condition is none (line 41)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 41, 20), result_eq_16256):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 41)
                        if_condition_16257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 20), result_eq_16256)
                        # Assigning a type to the variable 'if_condition_16257' (line 41)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'if_condition_16257', if_condition_16257)
                        # SSA begins for if statement (line 41)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'True' (line 42)
                        True_16258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 31), 'True')
                        # Assigning a type to the variable 'stypy_return_type' (line 42)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'stypy_return_type', True_16258)
                        # SSA join for if statement (line 41)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 38)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 35)
                module_type_store = module_type_store.join_ssa_context()
                


            if more_types_in_union_16212:
                # SSA join for if statement (line 34)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 27)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'False' (line 43)
    False_16259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type', False_16259)
    
    # ################# End of '__has_call_to_type_builtin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__has_call_to_type_builtin' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_16260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16260)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__has_call_to_type_builtin'
    return stypy_return_type_16260

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
    test_16262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'test', False)
    # Obtaining the member 'ops' of a type (line 47)
    ops_16263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 11), test_16262, 'ops')
    # Processing the call keyword arguments (line 47)
    kwargs_16264 = {}
    # Getting the type of 'len' (line 47)
    len_16261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 7), 'len', False)
    # Calling len(args, kwargs) (line 47)
    len_call_result_16265 = invoke(stypy.reporting.localization.Localization(__file__, 47, 7), len_16261, *[ops_16263], **kwargs_16264)
    
    int_16266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 24), 'int')
    # Applying the binary operator '==' (line 47)
    result_eq_16267 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 7), '==', len_call_result_16265, int_16266)
    
    # Testing if the type of an if condition is none (line 47)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 47, 4), result_eq_16267):
        pass
    else:
        
        # Testing the type of an if condition (line 47)
        if_condition_16268 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 4), result_eq_16267)
        # Assigning a type to the variable 'if_condition_16268' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'if_condition_16268', if_condition_16268)
        # SSA begins for if statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to type(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Obtaining the type of the subscript
        int_16270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 25), 'int')
        # Getting the type of 'test' (line 48)
        test_16271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'test', False)
        # Obtaining the member 'ops' of a type (line 48)
        ops_16272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 16), test_16271, 'ops')
        # Obtaining the member '__getitem__' of a type (line 48)
        getitem___16273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 16), ops_16272, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 48)
        subscript_call_result_16274 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), getitem___16273, int_16270)
        
        # Processing the call keyword arguments (line 48)
        kwargs_16275 = {}
        # Getting the type of 'type' (line 48)
        type_16269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'type', False)
        # Calling type(args, kwargs) (line 48)
        type_call_result_16276 = invoke(stypy.reporting.localization.Localization(__file__, 48, 11), type_16269, *[subscript_call_result_16274], **kwargs_16275)
        
        # Getting the type of 'ast' (line 48)
        ast_16277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 32), 'ast')
        # Obtaining the member 'Is' of a type (line 48)
        Is_16278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 32), ast_16277, 'Is')
        # Applying the binary operator 'is' (line 48)
        result_is__16279 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 11), 'is', type_call_result_16276, Is_16278)
        
        # Testing if the type of an if condition is none (line 48)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 48, 8), result_is__16279):
            pass
        else:
            
            # Testing the type of an if condition (line 48)
            if_condition_16280 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 8), result_is__16279)
            # Assigning a type to the variable 'if_condition_16280' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'if_condition_16280', if_condition_16280)
            # SSA begins for if statement (line 48)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'True' (line 49)
            True_16281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'stypy_return_type', True_16281)
            # SSA join for if statement (line 48)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 47)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'False' (line 50)
    False_16282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type', False_16282)
    
    # ################# End of '__has_call_to_is(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__has_call_to_is' in the type store
    # Getting the type of 'stypy_return_type' (line 46)
    stypy_return_type_16283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16283)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__has_call_to_is'
    return stypy_return_type_16283

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
    test_16285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'test', False)
    # Processing the call keyword arguments (line 54)
    kwargs_16286 = {}
    # Getting the type of 'type' (line 54)
    type_16284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 7), 'type', False)
    # Calling type(args, kwargs) (line 54)
    type_call_result_16287 = invoke(stypy.reporting.localization.Localization(__file__, 54, 7), type_16284, *[test_16285], **kwargs_16286)
    
    # Getting the type of 'ast' (line 54)
    ast_16288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'ast')
    # Obtaining the member 'Name' of a type (line 54)
    Name_16289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 21), ast_16288, 'Name')
    # Applying the binary operator 'is' (line 54)
    result_is__16290 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 7), 'is', type_call_result_16287, Name_16289)
    
    # Testing if the type of an if condition is none (line 54)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 54, 4), result_is__16290):
        pass
    else:
        
        # Testing the type of an if condition (line 54)
        if_condition_16291 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 4), result_is__16290)
        # Assigning a type to the variable 'if_condition_16291' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'if_condition_16291', if_condition_16291)
        # SSA begins for if statement (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 55):
        
        # Assigning a Attribute to a Name (line 55):
        # Getting the type of 'test' (line 55)
        test_16292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 18), 'test')
        # Obtaining the member 'id' of a type (line 55)
        id_16293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 18), test_16292, 'id')
        # Assigning a type to the variable 'name_id' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'name_id', id_16293)
        
        
        # SSA begins for try-except statement (line 56)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 57):
        
        # Assigning a Call to a Name (line 57):
        
        # Call to eval(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'name_id' (line 57)
        name_id_16295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'name_id', False)
        # Processing the call keyword arguments (line 57)
        kwargs_16296 = {}
        # Getting the type of 'eval' (line 57)
        eval_16294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'eval', False)
        # Calling eval(args, kwargs) (line 57)
        eval_call_result_16297 = invoke(stypy.reporting.localization.Localization(__file__, 57, 23), eval_16294, *[name_id_16295], **kwargs_16296)
        
        # Assigning a type to the variable 'type_obj' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'type_obj', eval_call_result_16297)
        
        
        # Call to type(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'type_obj' (line 58)
        type_obj_16299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 24), 'type_obj', False)
        # Processing the call keyword arguments (line 58)
        kwargs_16300 = {}
        # Getting the type of 'type' (line 58)
        type_16298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'type', False)
        # Calling type(args, kwargs) (line 58)
        type_call_result_16301 = invoke(stypy.reporting.localization.Localization(__file__, 58, 19), type_16298, *[type_obj_16299], **kwargs_16300)
        
        # Getting the type of 'types' (line 58)
        types_16302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 37), 'types')
        # Obtaining the member 'TypeType' of a type (line 58)
        TypeType_16303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 37), types_16302, 'TypeType')
        # Applying the binary operator 'is' (line 58)
        result_is__16304 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 19), 'is', type_call_result_16301, TypeType_16303)
        
        # Assigning a type to the variable 'stypy_return_type' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'stypy_return_type', result_is__16304)
        # SSA branch for the except part of a try statement (line 56)
        # SSA branch for the except '<any exception>' branch of a try statement (line 56)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'False' (line 60)
        False_16305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'stypy_return_type', False_16305)
        # SSA join for try-except statement (line 56)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 54)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'False' (line 61)
    False_16306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type', False_16306)
    
    # ################# End of '__is_type_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__is_type_name' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_16307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16307)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__is_type_name'
    return stypy_return_type_16307

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

    str_16308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, (-1)), 'str', '\n    Idiom "type is"\n    :param test:\n    :param visitor:\n    :param context:\n    :return:\n    ')
    
    
    # Call to type(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'test' (line 72)
    test_16310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'test', False)
    # Processing the call keyword arguments (line 72)
    kwargs_16311 = {}
    # Getting the type of 'type' (line 72)
    type_16309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 7), 'type', False)
    # Calling type(args, kwargs) (line 72)
    type_call_result_16312 = invoke(stypy.reporting.localization.Localization(__file__, 72, 7), type_16309, *[test_16310], **kwargs_16311)
    
    # Getting the type of 'ast' (line 72)
    ast_16313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'ast')
    # Obtaining the member 'Compare' of a type (line 72)
    Compare_16314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 25), ast_16313, 'Compare')
    # Applying the binary operator 'isnot' (line 72)
    result_is_not_16315 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 7), 'isnot', type_call_result_16312, Compare_16314)
    
    # Testing if the type of an if condition is none (line 72)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 4), result_is_not_16315):
        pass
    else:
        
        # Testing the type of an if condition (line 72)
        if_condition_16316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 4), result_is_not_16315)
        # Assigning a type to the variable 'if_condition_16316' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'if_condition_16316', if_condition_16316)
        # SSA begins for if statement (line 72)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'default_ret_tuple' (line 73)
        default_ret_tuple_16317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'default_ret_tuple')
        # Assigning a type to the variable 'stypy_return_type' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'stypy_return_type', default_ret_tuple_16317)
        # SSA join for if statement (line 72)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Evaluating a boolean operation
    
    # Call to __has_call_to_type_builtin(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'test' (line 75)
    test_16319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 34), 'test', False)
    # Processing the call keyword arguments (line 75)
    kwargs_16320 = {}
    # Getting the type of '__has_call_to_type_builtin' (line 75)
    has_call_to_type_builtin_16318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 7), '__has_call_to_type_builtin', False)
    # Calling __has_call_to_type_builtin(args, kwargs) (line 75)
    has_call_to_type_builtin_call_result_16321 = invoke(stypy.reporting.localization.Localization(__file__, 75, 7), has_call_to_type_builtin_16318, *[test_16319], **kwargs_16320)
    
    
    # Call to __has_call_to_is(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'test' (line 75)
    test_16323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 61), 'test', False)
    # Processing the call keyword arguments (line 75)
    kwargs_16324 = {}
    # Getting the type of '__has_call_to_is' (line 75)
    has_call_to_is_16322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 44), '__has_call_to_is', False)
    # Calling __has_call_to_is(args, kwargs) (line 75)
    has_call_to_is_call_result_16325 = invoke(stypy.reporting.localization.Localization(__file__, 75, 44), has_call_to_is_16322, *[test_16323], **kwargs_16324)
    
    # Applying the binary operator 'and' (line 75)
    result_and_keyword_16326 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 7), 'and', has_call_to_type_builtin_call_result_16321, has_call_to_is_call_result_16325)
    
    # Testing if the type of an if condition is none (line 75)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 4), result_and_keyword_16326):
        pass
    else:
        
        # Testing the type of an if condition (line 75)
        if_condition_16327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 4), result_and_keyword_16326)
        # Assigning a type to the variable 'if_condition_16327' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'if_condition_16327', if_condition_16327)
        # SSA begins for if statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Evaluating a boolean operation
        
        # Call to __has_call_to_type_builtin(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Obtaining the type of the subscript
        int_16329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 60), 'int')
        # Getting the type of 'test' (line 76)
        test_16330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 43), 'test', False)
        # Obtaining the member 'comparators' of a type (line 76)
        comparators_16331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 43), test_16330, 'comparators')
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___16332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 43), comparators_16331, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_16333 = invoke(stypy.reporting.localization.Localization(__file__, 76, 43), getitem___16332, int_16329)
        
        # Processing the call keyword arguments (line 76)
        kwargs_16334 = {}
        # Getting the type of '__has_call_to_type_builtin' (line 76)
        has_call_to_type_builtin_16328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), '__has_call_to_type_builtin', False)
        # Calling __has_call_to_type_builtin(args, kwargs) (line 76)
        has_call_to_type_builtin_call_result_16335 = invoke(stypy.reporting.localization.Localization(__file__, 76, 16), has_call_to_type_builtin_16328, *[subscript_call_result_16333], **kwargs_16334)
        
        
        # Call to __is_type_name(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Obtaining the type of the subscript
        int_16337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 99), 'int')
        # Getting the type of 'test' (line 76)
        test_16338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 82), 'test', False)
        # Obtaining the member 'comparators' of a type (line 76)
        comparators_16339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 82), test_16338, 'comparators')
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___16340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 82), comparators_16339, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_16341 = invoke(stypy.reporting.localization.Localization(__file__, 76, 82), getitem___16340, int_16337)
        
        # Processing the call keyword arguments (line 76)
        kwargs_16342 = {}
        # Getting the type of '__is_type_name' (line 76)
        is_type_name_16336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 67), '__is_type_name', False)
        # Calling __is_type_name(args, kwargs) (line 76)
        is_type_name_call_result_16343 = invoke(stypy.reporting.localization.Localization(__file__, 76, 67), is_type_name_16336, *[subscript_call_result_16341], **kwargs_16342)
        
        # Applying the binary operator 'or' (line 76)
        result_or_keyword_16344 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 16), 'or', has_call_to_type_builtin_call_result_16335, is_type_name_call_result_16343)
        
        # Applying the 'not' unary operator (line 76)
        result_not__16345 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 11), 'not', result_or_keyword_16344)
        
        # Testing if the type of an if condition is none (line 76)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 8), result_not__16345):
            pass
        else:
            
            # Testing the type of an if condition (line 76)
            if_condition_16346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 8), result_not__16345)
            # Assigning a type to the variable 'if_condition_16346' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'if_condition_16346', if_condition_16346)
            # SSA begins for if statement (line 76)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'default_ret_tuple' (line 77)
            default_ret_tuple_16347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'default_ret_tuple')
            # Assigning a type to the variable 'stypy_return_type' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'stypy_return_type', default_ret_tuple_16347)
            # SSA join for if statement (line 76)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 78):
        
        # Assigning a Call to a Name (line 78):
        
        # Call to visit(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Obtaining the type of the subscript
        int_16350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 50), 'int')
        # Getting the type of 'test' (line 78)
        test_16351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 35), 'test', False)
        # Obtaining the member 'left' of a type (line 78)
        left_16352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 35), test_16351, 'left')
        # Obtaining the member 'args' of a type (line 78)
        args_16353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 35), left_16352, 'args')
        # Obtaining the member '__getitem__' of a type (line 78)
        getitem___16354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 35), args_16353, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 78)
        subscript_call_result_16355 = invoke(stypy.reporting.localization.Localization(__file__, 78, 35), getitem___16354, int_16350)
        
        # Getting the type of 'context' (line 78)
        context_16356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 54), 'context', False)
        # Processing the call keyword arguments (line 78)
        kwargs_16357 = {}
        # Getting the type of 'visitor' (line 78)
        visitor_16348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 21), 'visitor', False)
        # Obtaining the member 'visit' of a type (line 78)
        visit_16349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 21), visitor_16348, 'visit')
        # Calling visit(args, kwargs) (line 78)
        visit_call_result_16358 = invoke(stypy.reporting.localization.Localization(__file__, 78, 21), visit_16349, *[subscript_call_result_16355, context_16356], **kwargs_16357)
        
        # Assigning a type to the variable 'type_param' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'type_param', visit_call_result_16358)
        
        # Call to __is_type_name(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Obtaining the type of the subscript
        int_16360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 43), 'int')
        # Getting the type of 'test' (line 79)
        test_16361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 26), 'test', False)
        # Obtaining the member 'comparators' of a type (line 79)
        comparators_16362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 26), test_16361, 'comparators')
        # Obtaining the member '__getitem__' of a type (line 79)
        getitem___16363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 26), comparators_16362, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 79)
        subscript_call_result_16364 = invoke(stypy.reporting.localization.Localization(__file__, 79, 26), getitem___16363, int_16360)
        
        # Processing the call keyword arguments (line 79)
        kwargs_16365 = {}
        # Getting the type of '__is_type_name' (line 79)
        is_type_name_16359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), '__is_type_name', False)
        # Calling __is_type_name(args, kwargs) (line 79)
        is_type_name_call_result_16366 = invoke(stypy.reporting.localization.Localization(__file__, 79, 11), is_type_name_16359, *[subscript_call_result_16364], **kwargs_16365)
        
        # Testing if the type of an if condition is none (line 79)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 79, 8), is_type_name_call_result_16366):
            
            # Assigning a Call to a Name (line 82):
            
            # Assigning a Call to a Name (line 82):
            
            # Call to visit(...): (line 82)
            # Processing the call arguments (line 82)
            
            # Obtaining the type of the subscript
            int_16380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 65), 'int')
            
            # Obtaining the type of the subscript
            int_16381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 57), 'int')
            # Getting the type of 'test' (line 82)
            test_16382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 40), 'test', False)
            # Obtaining the member 'comparators' of a type (line 82)
            comparators_16383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), test_16382, 'comparators')
            # Obtaining the member '__getitem__' of a type (line 82)
            getitem___16384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), comparators_16383, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 82)
            subscript_call_result_16385 = invoke(stypy.reporting.localization.Localization(__file__, 82, 40), getitem___16384, int_16381)
            
            # Obtaining the member 'args' of a type (line 82)
            args_16386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), subscript_call_result_16385, 'args')
            # Obtaining the member '__getitem__' of a type (line 82)
            getitem___16387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), args_16386, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 82)
            subscript_call_result_16388 = invoke(stypy.reporting.localization.Localization(__file__, 82, 40), getitem___16387, int_16380)
            
            # Getting the type of 'context' (line 82)
            context_16389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 69), 'context', False)
            # Processing the call keyword arguments (line 82)
            kwargs_16390 = {}
            # Getting the type of 'visitor' (line 82)
            visitor_16378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'visitor', False)
            # Obtaining the member 'visit' of a type (line 82)
            visit_16379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 26), visitor_16378, 'visit')
            # Calling visit(args, kwargs) (line 82)
            visit_call_result_16391 = invoke(stypy.reporting.localization.Localization(__file__, 82, 26), visit_16379, *[subscript_call_result_16388, context_16389], **kwargs_16390)
            
            # Assigning a type to the variable 'is_operator' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'is_operator', visit_call_result_16391)
            
            # Type idiom detected: calculating its left and rigth part (line 83)
            # Getting the type of 'list' (line 83)
            list_16392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 46), 'list')
            
            # Obtaining the type of the subscript
            int_16393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 42), 'int')
            # Getting the type of 'is_operator' (line 83)
            is_operator_16394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'is_operator')
            # Obtaining the member '__getitem__' of a type (line 83)
            getitem___16395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 30), is_operator_16394, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 83)
            subscript_call_result_16396 = invoke(stypy.reporting.localization.Localization(__file__, 83, 30), getitem___16395, int_16393)
            
            
            (may_be_16397, more_types_in_union_16398) = may_not_be_subtype(list_16392, subscript_call_result_16396)

            if may_be_16397:

                if more_types_in_union_16398:
                    # Runtime conditional SSA (line 83)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Tuple to a Name (line 84):
                
                # Assigning a Tuple to a Name (line 84):
                
                # Obtaining an instance of the builtin type 'tuple' (line 84)
                tuple_16399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 31), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 84)
                # Adding element type (line 84)
                
                # Obtaining an instance of the builtin type 'list' (line 84)
                list_16400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 31), 'list')
                # Adding type elements to the builtin type 'list' instance (line 84)
                # Adding element type (line 84)
                
                # Obtaining the type of the subscript
                int_16401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 44), 'int')
                # Getting the type of 'is_operator' (line 84)
                is_operator_16402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'is_operator')
                # Obtaining the member '__getitem__' of a type (line 84)
                getitem___16403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 32), is_operator_16402, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 84)
                subscript_call_result_16404 = invoke(stypy.reporting.localization.Localization(__file__, 84, 32), getitem___16403, int_16401)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 31), list_16400, subscript_call_result_16404)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 31), tuple_16399, list_16400)
                # Adding element type (line 84)
                
                # Obtaining the type of the subscript
                int_16405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 61), 'int')
                # Getting the type of 'is_operator' (line 84)
                is_operator_16406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 49), 'is_operator')
                # Obtaining the member '__getitem__' of a type (line 84)
                getitem___16407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 49), is_operator_16406, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 84)
                subscript_call_result_16408 = invoke(stypy.reporting.localization.Localization(__file__, 84, 49), getitem___16407, int_16405)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 31), tuple_16399, subscript_call_result_16408)
                
                # Assigning a type to the variable 'is_operator' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'is_operator', tuple_16399)

                if more_types_in_union_16398:
                    # SSA join for if statement (line 83)
                    module_type_store = module_type_store.join_ssa_context()


            
        else:
            
            # Testing the type of an if condition (line 79)
            if_condition_16367 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 8), is_type_name_call_result_16366)
            # Assigning a type to the variable 'if_condition_16367' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'if_condition_16367', if_condition_16367)
            # SSA begins for if statement (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 80):
            
            # Assigning a Call to a Name (line 80):
            
            # Call to visit(...): (line 80)
            # Processing the call arguments (line 80)
            
            # Obtaining the type of the subscript
            int_16370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 57), 'int')
            # Getting the type of 'test' (line 80)
            test_16371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 40), 'test', False)
            # Obtaining the member 'comparators' of a type (line 80)
            comparators_16372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 40), test_16371, 'comparators')
            # Obtaining the member '__getitem__' of a type (line 80)
            getitem___16373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 40), comparators_16372, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 80)
            subscript_call_result_16374 = invoke(stypy.reporting.localization.Localization(__file__, 80, 40), getitem___16373, int_16370)
            
            # Getting the type of 'context' (line 80)
            context_16375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 61), 'context', False)
            # Processing the call keyword arguments (line 80)
            kwargs_16376 = {}
            # Getting the type of 'visitor' (line 80)
            visitor_16368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 26), 'visitor', False)
            # Obtaining the member 'visit' of a type (line 80)
            visit_16369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 26), visitor_16368, 'visit')
            # Calling visit(args, kwargs) (line 80)
            visit_call_result_16377 = invoke(stypy.reporting.localization.Localization(__file__, 80, 26), visit_16369, *[subscript_call_result_16374, context_16375], **kwargs_16376)
            
            # Assigning a type to the variable 'is_operator' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'is_operator', visit_call_result_16377)
            # SSA branch for the else part of an if statement (line 79)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 82):
            
            # Assigning a Call to a Name (line 82):
            
            # Call to visit(...): (line 82)
            # Processing the call arguments (line 82)
            
            # Obtaining the type of the subscript
            int_16380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 65), 'int')
            
            # Obtaining the type of the subscript
            int_16381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 57), 'int')
            # Getting the type of 'test' (line 82)
            test_16382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 40), 'test', False)
            # Obtaining the member 'comparators' of a type (line 82)
            comparators_16383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), test_16382, 'comparators')
            # Obtaining the member '__getitem__' of a type (line 82)
            getitem___16384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), comparators_16383, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 82)
            subscript_call_result_16385 = invoke(stypy.reporting.localization.Localization(__file__, 82, 40), getitem___16384, int_16381)
            
            # Obtaining the member 'args' of a type (line 82)
            args_16386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), subscript_call_result_16385, 'args')
            # Obtaining the member '__getitem__' of a type (line 82)
            getitem___16387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), args_16386, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 82)
            subscript_call_result_16388 = invoke(stypy.reporting.localization.Localization(__file__, 82, 40), getitem___16387, int_16380)
            
            # Getting the type of 'context' (line 82)
            context_16389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 69), 'context', False)
            # Processing the call keyword arguments (line 82)
            kwargs_16390 = {}
            # Getting the type of 'visitor' (line 82)
            visitor_16378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'visitor', False)
            # Obtaining the member 'visit' of a type (line 82)
            visit_16379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 26), visitor_16378, 'visit')
            # Calling visit(args, kwargs) (line 82)
            visit_call_result_16391 = invoke(stypy.reporting.localization.Localization(__file__, 82, 26), visit_16379, *[subscript_call_result_16388, context_16389], **kwargs_16390)
            
            # Assigning a type to the variable 'is_operator' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'is_operator', visit_call_result_16391)
            
            # Type idiom detected: calculating its left and rigth part (line 83)
            # Getting the type of 'list' (line 83)
            list_16392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 46), 'list')
            
            # Obtaining the type of the subscript
            int_16393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 42), 'int')
            # Getting the type of 'is_operator' (line 83)
            is_operator_16394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'is_operator')
            # Obtaining the member '__getitem__' of a type (line 83)
            getitem___16395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 30), is_operator_16394, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 83)
            subscript_call_result_16396 = invoke(stypy.reporting.localization.Localization(__file__, 83, 30), getitem___16395, int_16393)
            
            
            (may_be_16397, more_types_in_union_16398) = may_not_be_subtype(list_16392, subscript_call_result_16396)

            if may_be_16397:

                if more_types_in_union_16398:
                    # Runtime conditional SSA (line 83)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Tuple to a Name (line 84):
                
                # Assigning a Tuple to a Name (line 84):
                
                # Obtaining an instance of the builtin type 'tuple' (line 84)
                tuple_16399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 31), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 84)
                # Adding element type (line 84)
                
                # Obtaining an instance of the builtin type 'list' (line 84)
                list_16400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 31), 'list')
                # Adding type elements to the builtin type 'list' instance (line 84)
                # Adding element type (line 84)
                
                # Obtaining the type of the subscript
                int_16401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 44), 'int')
                # Getting the type of 'is_operator' (line 84)
                is_operator_16402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'is_operator')
                # Obtaining the member '__getitem__' of a type (line 84)
                getitem___16403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 32), is_operator_16402, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 84)
                subscript_call_result_16404 = invoke(stypy.reporting.localization.Localization(__file__, 84, 32), getitem___16403, int_16401)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 31), list_16400, subscript_call_result_16404)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 31), tuple_16399, list_16400)
                # Adding element type (line 84)
                
                # Obtaining the type of the subscript
                int_16405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 61), 'int')
                # Getting the type of 'is_operator' (line 84)
                is_operator_16406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 49), 'is_operator')
                # Obtaining the member '__getitem__' of a type (line 84)
                getitem___16407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 49), is_operator_16406, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 84)
                subscript_call_result_16408 = invoke(stypy.reporting.localization.Localization(__file__, 84, 49), getitem___16407, int_16405)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 31), tuple_16399, subscript_call_result_16408)
                
                # Assigning a type to the variable 'is_operator' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'is_operator', tuple_16399)

                if more_types_in_union_16398:
                    # SSA join for if statement (line 83)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 79)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Obtaining an instance of the builtin type 'tuple' (line 86)
        tuple_16409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 86)
        # Adding element type (line 86)
        # Getting the type of 'True' (line 86)
        True_16410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 15), tuple_16409, True_16410)
        # Adding element type (line 86)
        # Getting the type of 'type_param' (line 86)
        type_param_16411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 21), 'type_param')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 15), tuple_16409, type_param_16411)
        # Adding element type (line 86)
        # Getting the type of 'is_operator' (line 86)
        is_operator_16412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 33), 'is_operator')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 15), tuple_16409, is_operator_16412)
        
        # Assigning a type to the variable 'stypy_return_type' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'stypy_return_type', tuple_16409)
        # SSA join for if statement (line 75)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'default_ret_tuple' (line 88)
    default_ret_tuple_16413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'default_ret_tuple')
    # Assigning a type to the variable 'stypy_return_type' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type', default_ret_tuple_16413)
    
    # ################# End of 'type_is_idiom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'type_is_idiom' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_16414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16414)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'type_is_idiom'
    return stypy_return_type_16414

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

    str_16415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, (-1)), 'str', '\n    Idiom "not type is"\n\n    :param test:\n    :param visitor:\n    :param context:\n    :return:\n    ')
    
    
    # Call to type(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'test' (line 100)
    test_16417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'test', False)
    # Processing the call keyword arguments (line 100)
    kwargs_16418 = {}
    # Getting the type of 'type' (line 100)
    type_16416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 7), 'type', False)
    # Calling type(args, kwargs) (line 100)
    type_call_result_16419 = invoke(stypy.reporting.localization.Localization(__file__, 100, 7), type_16416, *[test_16417], **kwargs_16418)
    
    # Getting the type of 'ast' (line 100)
    ast_16420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'ast')
    # Obtaining the member 'UnaryOp' of a type (line 100)
    UnaryOp_16421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 25), ast_16420, 'UnaryOp')
    # Applying the binary operator 'isnot' (line 100)
    result_is_not_16422 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 7), 'isnot', type_call_result_16419, UnaryOp_16421)
    
    # Testing if the type of an if condition is none (line 100)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 100, 4), result_is_not_16422):
        pass
    else:
        
        # Testing the type of an if condition (line 100)
        if_condition_16423 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 4), result_is_not_16422)
        # Assigning a type to the variable 'if_condition_16423' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'if_condition_16423', if_condition_16423)
        # SSA begins for if statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'default_ret_tuple' (line 101)
        default_ret_tuple_16424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'default_ret_tuple')
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'stypy_return_type', default_ret_tuple_16424)
        # SSA join for if statement (line 100)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to type(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'test' (line 102)
    test_16426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'test', False)
    # Obtaining the member 'op' of a type (line 102)
    op_16427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), test_16426, 'op')
    # Processing the call keyword arguments (line 102)
    kwargs_16428 = {}
    # Getting the type of 'type' (line 102)
    type_16425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 7), 'type', False)
    # Calling type(args, kwargs) (line 102)
    type_call_result_16429 = invoke(stypy.reporting.localization.Localization(__file__, 102, 7), type_16425, *[op_16427], **kwargs_16428)
    
    # Getting the type of 'ast' (line 102)
    ast_16430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'ast')
    # Obtaining the member 'Not' of a type (line 102)
    Not_16431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 28), ast_16430, 'Not')
    # Applying the binary operator 'isnot' (line 102)
    result_is_not_16432 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 7), 'isnot', type_call_result_16429, Not_16431)
    
    # Testing if the type of an if condition is none (line 102)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 102, 4), result_is_not_16432):
        pass
    else:
        
        # Testing the type of an if condition (line 102)
        if_condition_16433 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 4), result_is_not_16432)
        # Assigning a type to the variable 'if_condition_16433' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'if_condition_16433', if_condition_16433)
        # SSA begins for if statement (line 102)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'default_ret_tuple' (line 103)
        default_ret_tuple_16434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'default_ret_tuple')
        # Assigning a type to the variable 'stypy_return_type' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'stypy_return_type', default_ret_tuple_16434)
        # SSA join for if statement (line 102)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to type_is_idiom(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'test' (line 105)
    test_16436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 25), 'test', False)
    # Obtaining the member 'operand' of a type (line 105)
    operand_16437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 25), test_16436, 'operand')
    # Getting the type of 'visitor' (line 105)
    visitor_16438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 39), 'visitor', False)
    # Getting the type of 'context' (line 105)
    context_16439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 48), 'context', False)
    # Processing the call keyword arguments (line 105)
    kwargs_16440 = {}
    # Getting the type of 'type_is_idiom' (line 105)
    type_is_idiom_16435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'type_is_idiom', False)
    # Calling type_is_idiom(args, kwargs) (line 105)
    type_is_idiom_call_result_16441 = invoke(stypy.reporting.localization.Localization(__file__, 105, 11), type_is_idiom_16435, *[operand_16437, visitor_16438, context_16439], **kwargs_16440)
    
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', type_is_idiom_call_result_16441)
    
    # ################# End of 'not_type_is_idiom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'not_type_is_idiom' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_16442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16442)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'not_type_is_idiom'
    return stypy_return_type_16442

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
    int_16443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 26), 'int')
    # Getting the type of 'test' (line 109)
    test_16444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'test')
    # Obtaining the member 'left' of a type (line 109)
    left_16445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 11), test_16444, 'left')
    # Obtaining the member 'args' of a type (line 109)
    args_16446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 11), left_16445, 'args')
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___16447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 11), args_16446, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_16448 = invoke(stypy.reporting.localization.Localization(__file__, 109, 11), getitem___16447, int_16443)
    
    # Assigning a type to the variable 'stypy_return_type' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type', subscript_call_result_16448)
    
    # ################# End of '__get_idiom_type_param(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__get_idiom_type_param' in the type store
    # Getting the type of 'stypy_return_type' (line 108)
    stypy_return_type_16449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16449)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__get_idiom_type_param'
    return stypy_return_type_16449

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
    if_test_16451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 35), 'if_test', False)
    # Processing the call keyword arguments (line 113)
    kwargs_16452 = {}
    # Getting the type of '__get_idiom_type_param' (line 113)
    get_idiom_type_param_16450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), '__get_idiom_type_param', False)
    # Calling __get_idiom_type_param(args, kwargs) (line 113)
    get_idiom_type_param_call_result_16453 = invoke(stypy.reporting.localization.Localization(__file__, 113, 12), get_idiom_type_param_16450, *[if_test_16451], **kwargs_16452)
    
    # Assigning a type to the variable 'param' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'param', get_idiom_type_param_call_result_16453)
    
    
    # Call to type(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'param' (line 114)
    param_16455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'param', False)
    # Processing the call keyword arguments (line 114)
    kwargs_16456 = {}
    # Getting the type of 'type' (line 114)
    type_16454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 7), 'type', False)
    # Calling type(args, kwargs) (line 114)
    type_call_result_16457 = invoke(stypy.reporting.localization.Localization(__file__, 114, 7), type_16454, *[param_16455], **kwargs_16456)
    
    # Getting the type of 'ast' (line 114)
    ast_16458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'ast')
    # Obtaining the member 'Name' of a type (line 114)
    Name_16459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 22), ast_16458, 'Name')
    # Applying the binary operator 'is' (line 114)
    result_is__16460 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 7), 'is', type_call_result_16457, Name_16459)
    
    # Testing if the type of an if condition is none (line 114)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 114, 4), result_is__16460):
        pass
    else:
        
        # Testing the type of an if condition (line 114)
        if_condition_16461 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 4), result_is__16460)
        # Assigning a type to the variable 'if_condition_16461' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'if_condition_16461', if_condition_16461)
        # SSA begins for if statement (line 114)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to create_set_type_of(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'param' (line 115)
        param_16464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 55), 'param', False)
        # Obtaining the member 'id' of a type (line 115)
        id_16465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 55), param_16464, 'id')
        # Getting the type of 'type_' (line 115)
        type__16466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 65), 'type_', False)
        # Getting the type of 'lineno' (line 115)
        lineno_16467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 72), 'lineno', False)
        # Getting the type of 'col_offset' (line 115)
        col_offset_16468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 80), 'col_offset', False)
        # Processing the call keyword arguments (line 115)
        kwargs_16469 = {}
        # Getting the type of 'stypy_functions_copy' (line 115)
        stypy_functions_copy_16462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 'stypy_functions_copy', False)
        # Obtaining the member 'create_set_type_of' of a type (line 115)
        create_set_type_of_16463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 15), stypy_functions_copy_16462, 'create_set_type_of')
        # Calling create_set_type_of(args, kwargs) (line 115)
        create_set_type_of_call_result_16470 = invoke(stypy.reporting.localization.Localization(__file__, 115, 15), create_set_type_of_16463, *[id_16465, type__16466, lineno_16467, col_offset_16468], **kwargs_16469)
        
        # Assigning a type to the variable 'stypy_return_type' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'stypy_return_type', create_set_type_of_call_result_16470)
        # SSA join for if statement (line 114)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to type(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'param' (line 116)
    param_16472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'param', False)
    # Processing the call keyword arguments (line 116)
    kwargs_16473 = {}
    # Getting the type of 'type' (line 116)
    type_16471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 7), 'type', False)
    # Calling type(args, kwargs) (line 116)
    type_call_result_16474 = invoke(stypy.reporting.localization.Localization(__file__, 116, 7), type_16471, *[param_16472], **kwargs_16473)
    
    # Getting the type of 'ast' (line 116)
    ast_16475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 22), 'ast')
    # Obtaining the member 'Attribute' of a type (line 116)
    Attribute_16476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 22), ast_16475, 'Attribute')
    # Applying the binary operator 'is' (line 116)
    result_is__16477 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 7), 'is', type_call_result_16474, Attribute_16476)
    
    # Testing if the type of an if condition is none (line 116)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 116, 4), result_is__16477):
        pass
    else:
        
        # Testing the type of an if condition (line 116)
        if_condition_16478 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 4), result_is__16477)
        # Assigning a type to the variable 'if_condition_16478' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'if_condition_16478', if_condition_16478)
        # SSA begins for if statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 117):
        
        # Assigning a Call to a Name:
        
        # Call to create_get_type_of(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'param' (line 117)
        param_16481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 68), 'param', False)
        # Obtaining the member 'value' of a type (line 117)
        value_16482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 68), param_16481, 'value')
        # Obtaining the member 'id' of a type (line 117)
        id_16483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 68), value_16482, 'id')
        # Getting the type of 'lineno' (line 117)
        lineno_16484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 84), 'lineno', False)
        # Getting the type of 'col_offset' (line 117)
        col_offset_16485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 92), 'col_offset', False)
        # Processing the call keyword arguments (line 117)
        kwargs_16486 = {}
        # Getting the type of 'stypy_functions_copy' (line 117)
        stypy_functions_copy_16479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 28), 'stypy_functions_copy', False)
        # Obtaining the member 'create_get_type_of' of a type (line 117)
        create_get_type_of_16480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 28), stypy_functions_copy_16479, 'create_get_type_of')
        # Calling create_get_type_of(args, kwargs) (line 117)
        create_get_type_of_call_result_16487 = invoke(stypy.reporting.localization.Localization(__file__, 117, 28), create_get_type_of_16480, *[id_16483, lineno_16484, col_offset_16485], **kwargs_16486)
        
        # Assigning a type to the variable 'call_assignment_16149' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_16149', create_get_type_of_call_result_16487)
        
        # Assigning a Call to a Name (line 117):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_16149' (line 117)
        call_assignment_16149_16488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_16149', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_16489 = stypy_get_value_from_tuple(call_assignment_16149_16488, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_16150' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_16150', stypy_get_value_from_tuple_call_result_16489)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'call_assignment_16150' (line 117)
        call_assignment_16150_16490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_16150')
        # Assigning a type to the variable 'obj_type' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'obj_type', call_assignment_16150_16490)
        
        # Assigning a Call to a Name (line 117):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_16149' (line 117)
        call_assignment_16149_16491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_16149', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_16492 = stypy_get_value_from_tuple(call_assignment_16149_16491, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_16151' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_16151', stypy_get_value_from_tuple_call_result_16492)
        
        # Assigning a Name to a Name (line 117):
        # Getting the type of 'call_assignment_16151' (line 117)
        call_assignment_16151_16493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'call_assignment_16151')
        # Assigning a type to the variable 'obj_var' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 18), 'obj_var', call_assignment_16151_16493)
        
        # Assigning a Call to a Name (line 118):
        
        # Assigning a Call to a Name (line 118):
        
        # Call to create_set_type_of_member(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'obj_var' (line 118)
        obj_var_16496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 68), 'obj_var', False)
        # Getting the type of 'param' (line 118)
        param_16497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 77), 'param', False)
        # Obtaining the member 'attr' of a type (line 118)
        attr_16498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 77), param_16497, 'attr')
        # Getting the type of 'type_' (line 118)
        type__16499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 89), 'type_', False)
        # Getting the type of 'lineno' (line 118)
        lineno_16500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 96), 'lineno', False)
        # Getting the type of 'col_offset' (line 118)
        col_offset_16501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 104), 'col_offset', False)
        # Processing the call keyword arguments (line 118)
        kwargs_16502 = {}
        # Getting the type of 'stypy_functions_copy' (line 118)
        stypy_functions_copy_16494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'stypy_functions_copy', False)
        # Obtaining the member 'create_set_type_of_member' of a type (line 118)
        create_set_type_of_member_16495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 21), stypy_functions_copy_16494, 'create_set_type_of_member')
        # Calling create_set_type_of_member(args, kwargs) (line 118)
        create_set_type_of_member_call_result_16503 = invoke(stypy.reporting.localization.Localization(__file__, 118, 21), create_set_type_of_member_16495, *[obj_var_16496, attr_16498, type__16499, lineno_16500, col_offset_16501], **kwargs_16502)
        
        # Assigning a type to the variable 'set_member' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'set_member', create_set_type_of_member_call_result_16503)
        
        # Call to flatten_lists(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'obj_type' (line 119)
        obj_type_16506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 50), 'obj_type', False)
        # Getting the type of 'set_member' (line 119)
        set_member_16507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 60), 'set_member', False)
        # Processing the call keyword arguments (line 119)
        kwargs_16508 = {}
        # Getting the type of 'stypy_functions_copy' (line 119)
        stypy_functions_copy_16504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'stypy_functions_copy', False)
        # Obtaining the member 'flatten_lists' of a type (line 119)
        flatten_lists_16505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 15), stypy_functions_copy_16504, 'flatten_lists')
        # Calling flatten_lists(args, kwargs) (line 119)
        flatten_lists_call_result_16509 = invoke(stypy.reporting.localization.Localization(__file__, 119, 15), flatten_lists_16505, *[obj_type_16506, set_member_16507], **kwargs_16508)
        
        # Assigning a type to the variable 'stypy_return_type' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'stypy_return_type', flatten_lists_call_result_16509)
        # SSA join for if statement (line 116)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Obtaining an instance of the builtin type 'list' (line 121)
    list_16510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 121)
    
    # Assigning a type to the variable 'stypy_return_type' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type', list_16510)
    
    # ################# End of '__set_type_implementation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__set_type_implementation' in the type store
    # Getting the type of 'stypy_return_type' (line 112)
    stypy_return_type_16511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16511)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__set_type_implementation'
    return stypy_return_type_16511

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
    if_test_16513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 35), 'if_test', False)
    # Processing the call keyword arguments (line 125)
    kwargs_16514 = {}
    # Getting the type of '__get_idiom_type_param' (line 125)
    get_idiom_type_param_16512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), '__get_idiom_type_param', False)
    # Calling __get_idiom_type_param(args, kwargs) (line 125)
    get_idiom_type_param_call_result_16515 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), get_idiom_type_param_16512, *[if_test_16513], **kwargs_16514)
    
    # Assigning a type to the variable 'param' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'param', get_idiom_type_param_call_result_16515)
    
    
    # Call to type(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'param' (line 126)
    param_16517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'param', False)
    # Processing the call keyword arguments (line 126)
    kwargs_16518 = {}
    # Getting the type of 'type' (line 126)
    type_16516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 7), 'type', False)
    # Calling type(args, kwargs) (line 126)
    type_call_result_16519 = invoke(stypy.reporting.localization.Localization(__file__, 126, 7), type_16516, *[param_16517], **kwargs_16518)
    
    # Getting the type of 'ast' (line 126)
    ast_16520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'ast')
    # Obtaining the member 'Name' of a type (line 126)
    Name_16521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 22), ast_16520, 'Name')
    # Applying the binary operator 'is' (line 126)
    result_is__16522 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 7), 'is', type_call_result_16519, Name_16521)
    
    # Testing if the type of an if condition is none (line 126)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 126, 4), result_is__16522):
        pass
    else:
        
        # Testing the type of an if condition (line 126)
        if_condition_16523 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 4), result_is__16522)
        # Assigning a type to the variable 'if_condition_16523' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'if_condition_16523', if_condition_16523)
        # SSA begins for if statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 127):
        
        # Assigning a Call to a Name:
        
        # Call to create_get_type_of(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'param' (line 127)
        param_16526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 68), 'param', False)
        # Obtaining the member 'id' of a type (line 127)
        id_16527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 68), param_16526, 'id')
        # Getting the type of 'lineno' (line 127)
        lineno_16528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 78), 'lineno', False)
        # Getting the type of 'col_offset' (line 127)
        col_offset_16529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 86), 'col_offset', False)
        # Processing the call keyword arguments (line 127)
        kwargs_16530 = {}
        # Getting the type of 'stypy_functions_copy' (line 127)
        stypy_functions_copy_16524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 28), 'stypy_functions_copy', False)
        # Obtaining the member 'create_get_type_of' of a type (line 127)
        create_get_type_of_16525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 28), stypy_functions_copy_16524, 'create_get_type_of')
        # Calling create_get_type_of(args, kwargs) (line 127)
        create_get_type_of_call_result_16531 = invoke(stypy.reporting.localization.Localization(__file__, 127, 28), create_get_type_of_16525, *[id_16527, lineno_16528, col_offset_16529], **kwargs_16530)
        
        # Assigning a type to the variable 'call_assignment_16152' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'call_assignment_16152', create_get_type_of_call_result_16531)
        
        # Assigning a Call to a Name (line 127):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_16152' (line 127)
        call_assignment_16152_16532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'call_assignment_16152', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_16533 = stypy_get_value_from_tuple(call_assignment_16152_16532, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_16153' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'call_assignment_16153', stypy_get_value_from_tuple_call_result_16533)
        
        # Assigning a Name to a Name (line 127):
        # Getting the type of 'call_assignment_16153' (line 127)
        call_assignment_16153_16534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'call_assignment_16153')
        # Assigning a type to the variable 'obj_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'obj_type', call_assignment_16153_16534)
        
        # Assigning a Call to a Name (line 127):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_16152' (line 127)
        call_assignment_16152_16535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'call_assignment_16152', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_16536 = stypy_get_value_from_tuple(call_assignment_16152_16535, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_16154' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'call_assignment_16154', stypy_get_value_from_tuple_call_result_16536)
        
        # Assigning a Name to a Name (line 127):
        # Getting the type of 'call_assignment_16154' (line 127)
        call_assignment_16154_16537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'call_assignment_16154')
        # Assigning a type to the variable 'obj_var' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 18), 'obj_var', call_assignment_16154_16537)
        
        # Assigning a Call to a Name (line 128):
        
        # Assigning a Call to a Name (line 128):
        
        # Call to create_call(...): (line 128)
        # Processing the call arguments (line 128)
        
        # Call to create_Name(...): (line 128)
        # Processing the call arguments (line 128)
        str_16542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 85), 'str', 'remove_type_from_union')
        # Processing the call keyword arguments (line 128)
        kwargs_16543 = {}
        # Getting the type of 'core_language_copy' (line 128)
        core_language_copy_16540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 54), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 128)
        create_Name_16541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 54), core_language_copy_16540, 'create_Name')
        # Calling create_Name(args, kwargs) (line 128)
        create_Name_call_result_16544 = invoke(stypy.reporting.localization.Localization(__file__, 128, 54), create_Name_16541, *[str_16542], **kwargs_16543)
        
        
        # Obtaining an instance of the builtin type 'list' (line 129)
        list_16545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 129)
        # Adding element type (line 129)
        # Getting the type of 'obj_var' (line 129)
        obj_var_16546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 50), 'obj_var', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 49), list_16545, obj_var_16546)
        # Adding element type (line 129)
        # Getting the type of 'type_' (line 129)
        type__16547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 59), 'type_', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 49), list_16545, type__16547)
        
        # Processing the call keyword arguments (line 128)
        # Getting the type of 'lineno' (line 129)
        lineno_16548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 72), 'lineno', False)
        keyword_16549 = lineno_16548
        # Getting the type of 'col_offset' (line 129)
        col_offset_16550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 87), 'col_offset', False)
        keyword_16551 = col_offset_16550
        kwargs_16552 = {'column': keyword_16551, 'line': keyword_16549}
        # Getting the type of 'functions_copy' (line 128)
        functions_copy_16538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 27), 'functions_copy', False)
        # Obtaining the member 'create_call' of a type (line 128)
        create_call_16539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 27), functions_copy_16538, 'create_call')
        # Calling create_call(args, kwargs) (line 128)
        create_call_call_result_16553 = invoke(stypy.reporting.localization.Localization(__file__, 128, 27), create_call_16539, *[create_Name_call_result_16544, list_16545], **kwargs_16552)
        
        # Assigning a type to the variable 'remove_type_call' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'remove_type_call', create_call_call_result_16553)
        
        # Assigning a Call to a Name (line 130):
        
        # Assigning a Call to a Name (line 130):
        
        # Call to create_set_type_of(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'param' (line 130)
        param_16556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 59), 'param', False)
        # Obtaining the member 'id' of a type (line 130)
        id_16557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 59), param_16556, 'id')
        # Getting the type of 'remove_type_call' (line 130)
        remove_type_call_16558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 69), 'remove_type_call', False)
        # Getting the type of 'lineno' (line 130)
        lineno_16559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 87), 'lineno', False)
        # Getting the type of 'col_offset' (line 130)
        col_offset_16560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 95), 'col_offset', False)
        # Processing the call keyword arguments (line 130)
        kwargs_16561 = {}
        # Getting the type of 'stypy_functions_copy' (line 130)
        stypy_functions_copy_16554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 19), 'stypy_functions_copy', False)
        # Obtaining the member 'create_set_type_of' of a type (line 130)
        create_set_type_of_16555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 19), stypy_functions_copy_16554, 'create_set_type_of')
        # Calling create_set_type_of(args, kwargs) (line 130)
        create_set_type_of_call_result_16562 = invoke(stypy.reporting.localization.Localization(__file__, 130, 19), create_set_type_of_16555, *[id_16557, remove_type_call_16558, lineno_16559, col_offset_16560], **kwargs_16561)
        
        # Assigning a type to the variable 'set_type' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'set_type', create_set_type_of_call_result_16562)
        
        # Call to flatten_lists(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'obj_type' (line 132)
        obj_type_16565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 50), 'obj_type', False)
        # Getting the type of 'set_type' (line 132)
        set_type_16566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 60), 'set_type', False)
        # Processing the call keyword arguments (line 132)
        kwargs_16567 = {}
        # Getting the type of 'stypy_functions_copy' (line 132)
        stypy_functions_copy_16563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 15), 'stypy_functions_copy', False)
        # Obtaining the member 'flatten_lists' of a type (line 132)
        flatten_lists_16564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 15), stypy_functions_copy_16563, 'flatten_lists')
        # Calling flatten_lists(args, kwargs) (line 132)
        flatten_lists_call_result_16568 = invoke(stypy.reporting.localization.Localization(__file__, 132, 15), flatten_lists_16564, *[obj_type_16565, set_type_16566], **kwargs_16567)
        
        # Assigning a type to the variable 'stypy_return_type' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'stypy_return_type', flatten_lists_call_result_16568)
        # SSA join for if statement (line 126)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to type(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'param' (line 133)
    param_16570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'param', False)
    # Processing the call keyword arguments (line 133)
    kwargs_16571 = {}
    # Getting the type of 'type' (line 133)
    type_16569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 7), 'type', False)
    # Calling type(args, kwargs) (line 133)
    type_call_result_16572 = invoke(stypy.reporting.localization.Localization(__file__, 133, 7), type_16569, *[param_16570], **kwargs_16571)
    
    # Getting the type of 'ast' (line 133)
    ast_16573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'ast')
    # Obtaining the member 'Attribute' of a type (line 133)
    Attribute_16574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 22), ast_16573, 'Attribute')
    # Applying the binary operator 'is' (line 133)
    result_is__16575 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 7), 'is', type_call_result_16572, Attribute_16574)
    
    # Testing if the type of an if condition is none (line 133)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 133, 4), result_is__16575):
        pass
    else:
        
        # Testing the type of an if condition (line 133)
        if_condition_16576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 4), result_is__16575)
        # Assigning a type to the variable 'if_condition_16576' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'if_condition_16576', if_condition_16576)
        # SSA begins for if statement (line 133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 135):
        
        # Assigning a Call to a Name:
        
        # Call to create_get_type_of(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'param' (line 135)
        param_16579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 74), 'param', False)
        # Obtaining the member 'value' of a type (line 135)
        value_16580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 74), param_16579, 'value')
        # Obtaining the member 'id' of a type (line 135)
        id_16581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 74), value_16580, 'id')
        # Getting the type of 'lineno' (line 135)
        lineno_16582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 90), 'lineno', False)
        # Getting the type of 'col_offset' (line 135)
        col_offset_16583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 98), 'col_offset', False)
        # Processing the call keyword arguments (line 135)
        kwargs_16584 = {}
        # Getting the type of 'stypy_functions_copy' (line 135)
        stypy_functions_copy_16577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 34), 'stypy_functions_copy', False)
        # Obtaining the member 'create_get_type_of' of a type (line 135)
        create_get_type_of_16578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 34), stypy_functions_copy_16577, 'create_get_type_of')
        # Calling create_get_type_of(args, kwargs) (line 135)
        create_get_type_of_call_result_16585 = invoke(stypy.reporting.localization.Localization(__file__, 135, 34), create_get_type_of_16578, *[id_16581, lineno_16582, col_offset_16583], **kwargs_16584)
        
        # Assigning a type to the variable 'call_assignment_16155' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'call_assignment_16155', create_get_type_of_call_result_16585)
        
        # Assigning a Call to a Name (line 135):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_16155' (line 135)
        call_assignment_16155_16586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'call_assignment_16155', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_16587 = stypy_get_value_from_tuple(call_assignment_16155_16586, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_16156' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'call_assignment_16156', stypy_get_value_from_tuple_call_result_16587)
        
        # Assigning a Name to a Name (line 135):
        # Getting the type of 'call_assignment_16156' (line 135)
        call_assignment_16156_16588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'call_assignment_16156')
        # Assigning a type to the variable 'obj_type_stmts' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'obj_type_stmts', call_assignment_16156_16588)
        
        # Assigning a Call to a Name (line 135):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_16155' (line 135)
        call_assignment_16155_16589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'call_assignment_16155', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_16590 = stypy_get_value_from_tuple(call_assignment_16155_16589, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_16157' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'call_assignment_16157', stypy_get_value_from_tuple_call_result_16590)
        
        # Assigning a Name to a Name (line 135):
        # Getting the type of 'call_assignment_16157' (line 135)
        call_assignment_16157_16591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'call_assignment_16157')
        # Assigning a type to the variable 'obj_var' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'obj_var', call_assignment_16157_16591)
        
        # Assigning a Call to a Tuple (line 137):
        
        # Assigning a Call to a Name:
        
        # Call to create_get_type_of_member(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'obj_var' (line 137)
        obj_var_16594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 81), 'obj_var', False)
        # Getting the type of 'param' (line 137)
        param_16595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 90), 'param', False)
        # Obtaining the member 'attr' of a type (line 137)
        attr_16596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 90), param_16595, 'attr')
        # Getting the type of 'lineno' (line 137)
        lineno_16597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 102), 'lineno', False)
        # Getting the type of 'col_offset' (line 137)
        col_offset_16598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 110), 'col_offset', False)
        # Processing the call keyword arguments (line 137)
        kwargs_16599 = {}
        # Getting the type of 'stypy_functions_copy' (line 137)
        stypy_functions_copy_16592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 34), 'stypy_functions_copy', False)
        # Obtaining the member 'create_get_type_of_member' of a type (line 137)
        create_get_type_of_member_16593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 34), stypy_functions_copy_16592, 'create_get_type_of_member')
        # Calling create_get_type_of_member(args, kwargs) (line 137)
        create_get_type_of_member_call_result_16600 = invoke(stypy.reporting.localization.Localization(__file__, 137, 34), create_get_type_of_member_16593, *[obj_var_16594, attr_16596, lineno_16597, col_offset_16598], **kwargs_16599)
        
        # Assigning a type to the variable 'call_assignment_16158' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'call_assignment_16158', create_get_type_of_member_call_result_16600)
        
        # Assigning a Call to a Name (line 137):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_16158' (line 137)
        call_assignment_16158_16601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'call_assignment_16158', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_16602 = stypy_get_value_from_tuple(call_assignment_16158_16601, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_16159' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'call_assignment_16159', stypy_get_value_from_tuple_call_result_16602)
        
        # Assigning a Name to a Name (line 137):
        # Getting the type of 'call_assignment_16159' (line 137)
        call_assignment_16159_16603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'call_assignment_16159')
        # Assigning a type to the variable 'att_type_stmts' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'att_type_stmts', call_assignment_16159_16603)
        
        # Assigning a Call to a Name (line 137):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_16158' (line 137)
        call_assignment_16158_16604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'call_assignment_16158', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_16605 = stypy_get_value_from_tuple(call_assignment_16158_16604, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_16160' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'call_assignment_16160', stypy_get_value_from_tuple_call_result_16605)
        
        # Assigning a Name to a Name (line 137):
        # Getting the type of 'call_assignment_16160' (line 137)
        call_assignment_16160_16606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'call_assignment_16160')
        # Assigning a type to the variable 'att_var' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 24), 'att_var', call_assignment_16160_16606)
        
        # Assigning a Call to a Name (line 138):
        
        # Assigning a Call to a Name (line 138):
        
        # Call to create_call(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Call to create_Name(...): (line 138)
        # Processing the call arguments (line 138)
        str_16611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 85), 'str', 'remove_type_from_union')
        # Processing the call keyword arguments (line 138)
        kwargs_16612 = {}
        # Getting the type of 'core_language_copy' (line 138)
        core_language_copy_16609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 54), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 138)
        create_Name_16610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 54), core_language_copy_16609, 'create_Name')
        # Calling create_Name(args, kwargs) (line 138)
        create_Name_call_result_16613 = invoke(stypy.reporting.localization.Localization(__file__, 138, 54), create_Name_16610, *[str_16611], **kwargs_16612)
        
        
        # Obtaining an instance of the builtin type 'list' (line 139)
        list_16614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 49), 'list')
        # Adding type elements to the builtin type 'list' instance (line 139)
        # Adding element type (line 139)
        # Getting the type of 'att_var' (line 139)
        att_var_16615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 50), 'att_var', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 49), list_16614, att_var_16615)
        # Adding element type (line 139)
        # Getting the type of 'type_' (line 139)
        type__16616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 59), 'type_', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 49), list_16614, type__16616)
        
        # Processing the call keyword arguments (line 138)
        # Getting the type of 'lineno' (line 139)
        lineno_16617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 72), 'lineno', False)
        keyword_16618 = lineno_16617
        # Getting the type of 'col_offset' (line 139)
        col_offset_16619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 87), 'col_offset', False)
        keyword_16620 = col_offset_16619
        kwargs_16621 = {'column': keyword_16620, 'line': keyword_16618}
        # Getting the type of 'functions_copy' (line 138)
        functions_copy_16607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 27), 'functions_copy', False)
        # Obtaining the member 'create_call' of a type (line 138)
        create_call_16608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 27), functions_copy_16607, 'create_call')
        # Calling create_call(args, kwargs) (line 138)
        create_call_call_result_16622 = invoke(stypy.reporting.localization.Localization(__file__, 138, 27), create_call_16608, *[create_Name_call_result_16613, list_16614], **kwargs_16621)
        
        # Assigning a type to the variable 'remove_type_call' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'remove_type_call', create_call_call_result_16622)
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to create_set_type_of_member(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'obj_var' (line 140)
        obj_var_16625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 68), 'obj_var', False)
        # Getting the type of 'param' (line 140)
        param_16626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 77), 'param', False)
        # Obtaining the member 'attr' of a type (line 140)
        attr_16627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 77), param_16626, 'attr')
        # Getting the type of 'remove_type_call' (line 140)
        remove_type_call_16628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 89), 'remove_type_call', False)
        # Getting the type of 'lineno' (line 140)
        lineno_16629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 107), 'lineno', False)
        # Getting the type of 'col_offset' (line 141)
        col_offset_16630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 63), 'col_offset', False)
        # Processing the call keyword arguments (line 140)
        kwargs_16631 = {}
        # Getting the type of 'stypy_functions_copy' (line 140)
        stypy_functions_copy_16623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 21), 'stypy_functions_copy', False)
        # Obtaining the member 'create_set_type_of_member' of a type (line 140)
        create_set_type_of_member_16624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 21), stypy_functions_copy_16623, 'create_set_type_of_member')
        # Calling create_set_type_of_member(args, kwargs) (line 140)
        create_set_type_of_member_call_result_16632 = invoke(stypy.reporting.localization.Localization(__file__, 140, 21), create_set_type_of_member_16624, *[obj_var_16625, attr_16627, remove_type_call_16628, lineno_16629, col_offset_16630], **kwargs_16631)
        
        # Assigning a type to the variable 'set_member' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'set_member', create_set_type_of_member_call_result_16632)
        
        # Call to flatten_lists(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'obj_type_stmts' (line 142)
        obj_type_stmts_16635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 50), 'obj_type_stmts', False)
        # Getting the type of 'att_type_stmts' (line 142)
        att_type_stmts_16636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 66), 'att_type_stmts', False)
        # Getting the type of 'set_member' (line 142)
        set_member_16637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 82), 'set_member', False)
        # Processing the call keyword arguments (line 142)
        kwargs_16638 = {}
        # Getting the type of 'stypy_functions_copy' (line 142)
        stypy_functions_copy_16633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'stypy_functions_copy', False)
        # Obtaining the member 'flatten_lists' of a type (line 142)
        flatten_lists_16634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 15), stypy_functions_copy_16633, 'flatten_lists')
        # Calling flatten_lists(args, kwargs) (line 142)
        flatten_lists_call_result_16639 = invoke(stypy.reporting.localization.Localization(__file__, 142, 15), flatten_lists_16634, *[obj_type_stmts_16635, att_type_stmts_16636, set_member_16637], **kwargs_16638)
        
        # Assigning a type to the variable 'stypy_return_type' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'stypy_return_type', flatten_lists_call_result_16639)
        # SSA join for if statement (line 133)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Obtaining an instance of the builtin type 'list' (line 144)
    list_16640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 144)
    
    # Assigning a type to the variable 'stypy_return_type' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type', list_16640)
    
    # ################# End of '__remove_type_from_union_implementation(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__remove_type_from_union_implementation' in the type store
    # Getting the type of 'stypy_return_type' (line 124)
    stypy_return_type_16641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16641)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__remove_type_from_union_implementation'
    return stypy_return_type_16641

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
    idiom_name_16642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 7), 'idiom_name')
    str_16643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 21), 'str', 'type_is')
    # Applying the binary operator '==' (line 148)
    result_eq_16644 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 7), '==', idiom_name_16642, str_16643)
    
    # Testing if the type of an if condition is none (line 148)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 148, 4), result_eq_16644):
        pass
    else:
        
        # Testing the type of an if condition (line 148)
        if_condition_16645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 4), result_eq_16644)
        # Assigning a type to the variable 'if_condition_16645' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'if_condition_16645', if_condition_16645)
        # SSA begins for if statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'if_branch' (line 149)
        if_branch_16646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'if_branch')
        str_16647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 24), 'str', 'if')
        # Applying the binary operator '==' (line 149)
        result_eq_16648 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 11), '==', if_branch_16646, str_16647)
        
        # Testing if the type of an if condition is none (line 149)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 149, 8), result_eq_16648):
            pass
        else:
            
            # Testing the type of an if condition (line 149)
            if_condition_16649 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 8), result_eq_16648)
            # Assigning a type to the variable 'if_condition_16649' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'if_condition_16649', if_condition_16649)
            # SSA begins for if statement (line 149)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __set_type_implementation(...): (line 150)
            # Processing the call arguments (line 150)
            # Getting the type of 'if_test' (line 150)
            if_test_16651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 45), 'if_test', False)
            # Getting the type of 'type_' (line 150)
            type__16652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 54), 'type_', False)
            # Getting the type of 'lineno' (line 150)
            lineno_16653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 61), 'lineno', False)
            # Getting the type of 'col_offset' (line 150)
            col_offset_16654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 69), 'col_offset', False)
            # Processing the call keyword arguments (line 150)
            kwargs_16655 = {}
            # Getting the type of '__set_type_implementation' (line 150)
            set_type_implementation_16650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), '__set_type_implementation', False)
            # Calling __set_type_implementation(args, kwargs) (line 150)
            set_type_implementation_call_result_16656 = invoke(stypy.reporting.localization.Localization(__file__, 150, 19), set_type_implementation_16650, *[if_test_16651, type__16652, lineno_16653, col_offset_16654], **kwargs_16655)
            
            # Assigning a type to the variable 'stypy_return_type' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'stypy_return_type', set_type_implementation_call_result_16656)
            # SSA join for if statement (line 149)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'if_branch' (line 151)
        if_branch_16657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'if_branch')
        str_16658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 24), 'str', 'else')
        # Applying the binary operator '==' (line 151)
        result_eq_16659 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 11), '==', if_branch_16657, str_16658)
        
        # Testing if the type of an if condition is none (line 151)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 151, 8), result_eq_16659):
            pass
        else:
            
            # Testing the type of an if condition (line 151)
            if_condition_16660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 8), result_eq_16659)
            # Assigning a type to the variable 'if_condition_16660' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'if_condition_16660', if_condition_16660)
            # SSA begins for if statement (line 151)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __remove_type_from_union_implementation(...): (line 152)
            # Processing the call arguments (line 152)
            # Getting the type of 'if_test' (line 152)
            if_test_16662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 59), 'if_test', False)
            # Getting the type of 'type_' (line 152)
            type__16663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 68), 'type_', False)
            # Getting the type of 'lineno' (line 152)
            lineno_16664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 75), 'lineno', False)
            # Getting the type of 'col_offset' (line 152)
            col_offset_16665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 83), 'col_offset', False)
            # Processing the call keyword arguments (line 152)
            kwargs_16666 = {}
            # Getting the type of '__remove_type_from_union_implementation' (line 152)
            remove_type_from_union_implementation_16661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 19), '__remove_type_from_union_implementation', False)
            # Calling __remove_type_from_union_implementation(args, kwargs) (line 152)
            remove_type_from_union_implementation_call_result_16667 = invoke(stypy.reporting.localization.Localization(__file__, 152, 19), remove_type_from_union_implementation_16661, *[if_test_16662, type__16663, lineno_16664, col_offset_16665], **kwargs_16666)
            
            # Assigning a type to the variable 'stypy_return_type' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'stypy_return_type', remove_type_from_union_implementation_call_result_16667)
            # SSA join for if statement (line 151)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 148)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'idiom_name' (line 154)
    idiom_name_16668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 7), 'idiom_name')
    str_16669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 21), 'str', 'not_type_is')
    # Applying the binary operator '==' (line 154)
    result_eq_16670 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 7), '==', idiom_name_16668, str_16669)
    
    # Testing if the type of an if condition is none (line 154)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 154, 4), result_eq_16670):
        pass
    else:
        
        # Testing the type of an if condition (line 154)
        if_condition_16671 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 4), result_eq_16670)
        # Assigning a type to the variable 'if_condition_16671' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'if_condition_16671', if_condition_16671)
        # SSA begins for if statement (line 154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 155):
        
        # Assigning a Attribute to a Name (line 155):
        # Getting the type of 'if_test' (line 155)
        if_test_16672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 18), 'if_test')
        # Obtaining the member 'operand' of a type (line 155)
        operand_16673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 18), if_test_16672, 'operand')
        # Assigning a type to the variable 'if_test' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'if_test', operand_16673)
        
        # Getting the type of 'if_branch' (line 156)
        if_branch_16674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 11), 'if_branch')
        str_16675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 24), 'str', 'if')
        # Applying the binary operator '==' (line 156)
        result_eq_16676 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 11), '==', if_branch_16674, str_16675)
        
        # Testing if the type of an if condition is none (line 156)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 156, 8), result_eq_16676):
            pass
        else:
            
            # Testing the type of an if condition (line 156)
            if_condition_16677 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 8), result_eq_16676)
            # Assigning a type to the variable 'if_condition_16677' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'if_condition_16677', if_condition_16677)
            # SSA begins for if statement (line 156)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __remove_type_from_union_implementation(...): (line 157)
            # Processing the call arguments (line 157)
            # Getting the type of 'if_test' (line 157)
            if_test_16679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 59), 'if_test', False)
            # Getting the type of 'type_' (line 157)
            type__16680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 68), 'type_', False)
            # Getting the type of 'lineno' (line 157)
            lineno_16681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 75), 'lineno', False)
            # Getting the type of 'col_offset' (line 157)
            col_offset_16682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 83), 'col_offset', False)
            # Processing the call keyword arguments (line 157)
            kwargs_16683 = {}
            # Getting the type of '__remove_type_from_union_implementation' (line 157)
            remove_type_from_union_implementation_16678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 19), '__remove_type_from_union_implementation', False)
            # Calling __remove_type_from_union_implementation(args, kwargs) (line 157)
            remove_type_from_union_implementation_call_result_16684 = invoke(stypy.reporting.localization.Localization(__file__, 157, 19), remove_type_from_union_implementation_16678, *[if_test_16679, type__16680, lineno_16681, col_offset_16682], **kwargs_16683)
            
            # Assigning a type to the variable 'stypy_return_type' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'stypy_return_type', remove_type_from_union_implementation_call_result_16684)
            # SSA join for if statement (line 156)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'if_branch' (line 158)
        if_branch_16685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'if_branch')
        str_16686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 24), 'str', 'else')
        # Applying the binary operator '==' (line 158)
        result_eq_16687 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 11), '==', if_branch_16685, str_16686)
        
        # Testing if the type of an if condition is none (line 158)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 158, 8), result_eq_16687):
            pass
        else:
            
            # Testing the type of an if condition (line 158)
            if_condition_16688 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 8), result_eq_16687)
            # Assigning a type to the variable 'if_condition_16688' (line 158)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'if_condition_16688', if_condition_16688)
            # SSA begins for if statement (line 158)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __set_type_implementation(...): (line 159)
            # Processing the call arguments (line 159)
            # Getting the type of 'if_test' (line 159)
            if_test_16690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 45), 'if_test', False)
            # Getting the type of 'type_' (line 159)
            type__16691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 54), 'type_', False)
            # Getting the type of 'lineno' (line 159)
            lineno_16692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 61), 'lineno', False)
            # Getting the type of 'col_offset' (line 159)
            col_offset_16693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 69), 'col_offset', False)
            # Processing the call keyword arguments (line 159)
            kwargs_16694 = {}
            # Getting the type of '__set_type_implementation' (line 159)
            set_type_implementation_16689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), '__set_type_implementation', False)
            # Calling __set_type_implementation(args, kwargs) (line 159)
            set_type_implementation_call_result_16695 = invoke(stypy.reporting.localization.Localization(__file__, 159, 19), set_type_implementation_16689, *[if_test_16690, type__16691, lineno_16692, col_offset_16693], **kwargs_16694)
            
            # Assigning a type to the variable 'stypy_return_type' (line 159)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'stypy_return_type', set_type_implementation_call_result_16695)
            # SSA join for if statement (line 158)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 154)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Obtaining an instance of the builtin type 'list' (line 161)
    list_16696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 161)
    
    # Assigning a type to the variable 'stypy_return_type' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type', list_16696)
    
    # ################# End of 'set_type_of_idiom_var(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'set_type_of_idiom_var' in the type store
    # Getting the type of 'stypy_return_type' (line 147)
    stypy_return_type_16697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16697)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'set_type_of_idiom_var'
    return stypy_return_type_16697

# Assigning a type to the variable 'set_type_of_idiom_var' (line 147)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'set_type_of_idiom_var', set_type_of_idiom_var)

# Assigning a Dict to a Name (line 165):

# Assigning a Dict to a Name (line 165):

# Obtaining an instance of the builtin type 'dict' (line 165)
dict_16698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 20), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 165)
# Adding element type (key, value) (line 165)
str_16699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 4), 'str', 'type_is')
# Getting the type of 'type_is_idiom' (line 166)
type_is_idiom_16700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'type_is_idiom')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 20), dict_16698, (str_16699, type_is_idiom_16700))
# Adding element type (key, value) (line 165)
str_16701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 4), 'str', 'not_type_is')
# Getting the type of 'not_type_is_idiom' (line 167)
not_type_is_idiom_16702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 19), 'not_type_is_idiom')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 20), dict_16698, (str_16701, not_type_is_idiom_16702))

# Assigning a type to the variable 'recognized_idioms' (line 165)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'recognized_idioms', dict_16698)

# Assigning a Dict to a Name (line 171):

# Assigning a Dict to a Name (line 171):

# Obtaining an instance of the builtin type 'dict' (line 171)
dict_16703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 30), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 171)
# Adding element type (key, value) (line 171)
str_16704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 4), 'str', 'type_is')
# Getting the type of 'may_be_type_func_name' (line 172)
may_be_type_func_name_16705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'may_be_type_func_name')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 30), dict_16703, (str_16704, may_be_type_func_name_16705))
# Adding element type (key, value) (line 171)
str_16706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 4), 'str', 'not_type_is')
# Getting the type of 'may_not_be_type_func_name' (line 173)
may_not_be_type_func_name_16707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'may_not_be_type_func_name')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 30), dict_16703, (str_16706, may_not_be_type_func_name_16707))

# Assigning a type to the variable 'recognized_idioms_functions' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'recognized_idioms_functions', dict_16703)

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

    str_16708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, (-1)), 'str', '\n    Gets the function that process an idiom once it has been recognized\n    :param idiom_name: Idiom name\n    :return:\n    ')
    
    # Obtaining the type of the subscript
    # Getting the type of 'idiom_name' (line 183)
    idiom_name_16709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 39), 'idiom_name')
    # Getting the type of 'recognized_idioms_functions' (line 183)
    recognized_idioms_functions_16710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 11), 'recognized_idioms_functions')
    # Obtaining the member '__getitem__' of a type (line 183)
    getitem___16711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 11), recognized_idioms_functions_16710, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 183)
    subscript_call_result_16712 = invoke(stypy.reporting.localization.Localization(__file__, 183, 11), getitem___16711, idiom_name_16709)
    
    # Assigning a type to the variable 'stypy_return_type' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type', subscript_call_result_16712)
    
    # ################# End of 'get_recognized_idiom_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_recognized_idiom_function' in the type store
    # Getting the type of 'stypy_return_type' (line 177)
    stypy_return_type_16713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16713)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_recognized_idiom_function'
    return stypy_return_type_16713

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

    str_16714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, (-1)), 'str', '\n    Check if the passed test can be considered an idioms\n\n    :param test: Source code test\n    :param visitor: Type inference visitor, to change generated instructions\n    :param context: Context passed to the call\n    :return: Tuple of values that identify if an idiom has been recognized and calculated data if it is been recognized\n    ')
    
    # Getting the type of 'recognized_idioms' (line 195)
    recognized_idioms_16715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 17), 'recognized_idioms')
    # Assigning a type to the variable 'recognized_idioms_16715' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'recognized_idioms_16715', recognized_idioms_16715)
    # Testing if the for loop is going to be iterated (line 195)
    # Testing the type of a for loop iterable (line 195)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 195, 4), recognized_idioms_16715)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 195, 4), recognized_idioms_16715):
        # Getting the type of the for loop variable (line 195)
        for_loop_var_16716 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 195, 4), recognized_idioms_16715)
        # Assigning a type to the variable 'idiom' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'idiom', for_loop_var_16716)
        # SSA begins for a for statement (line 195)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to (...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'test' (line 196)
        test_16721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 42), 'test', False)
        # Getting the type of 'visitor' (line 196)
        visitor_16722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 48), 'visitor', False)
        # Getting the type of 'context' (line 196)
        context_16723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 57), 'context', False)
        # Processing the call keyword arguments (line 196)
        kwargs_16724 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'idiom' (line 196)
        idiom_16717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 35), 'idiom', False)
        # Getting the type of 'recognized_idioms' (line 196)
        recognized_idioms_16718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 17), 'recognized_idioms', False)
        # Obtaining the member '__getitem__' of a type (line 196)
        getitem___16719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 17), recognized_idioms_16718, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 196)
        subscript_call_result_16720 = invoke(stypy.reporting.localization.Localization(__file__, 196, 17), getitem___16719, idiom_16717)
        
        # Calling (args, kwargs) (line 196)
        _call_result_16725 = invoke(stypy.reporting.localization.Localization(__file__, 196, 17), subscript_call_result_16720, *[test_16721, visitor_16722, context_16723], **kwargs_16724)
        
        # Assigning a type to the variable 'result' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'result', _call_result_16725)
        
        # Obtaining the type of the subscript
        int_16726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 18), 'int')
        # Getting the type of 'result' (line 197)
        result_16727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 11), 'result')
        # Obtaining the member '__getitem__' of a type (line 197)
        getitem___16728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 11), result_16727, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 197)
        subscript_call_result_16729 = invoke(stypy.reporting.localization.Localization(__file__, 197, 11), getitem___16728, int_16726)
        
        # Testing if the type of an if condition is none (line 197)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 197, 8), subscript_call_result_16729):
            pass
        else:
            
            # Testing the type of an if condition (line 197)
            if_condition_16730 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 8), subscript_call_result_16729)
            # Assigning a type to the variable 'if_condition_16730' (line 197)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'if_condition_16730', if_condition_16730)
            # SSA begins for if statement (line 197)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 198):
            
            # Assigning a Call to a Name (line 198):
            
            # Call to list(...): (line 198)
            # Processing the call arguments (line 198)
            # Getting the type of 'result' (line 198)
            result_16732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 29), 'result', False)
            # Processing the call keyword arguments (line 198)
            kwargs_16733 = {}
            # Getting the type of 'list' (line 198)
            list_16731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 24), 'list', False)
            # Calling list(args, kwargs) (line 198)
            list_call_result_16734 = invoke(stypy.reporting.localization.Localization(__file__, 198, 24), list_16731, *[result_16732], **kwargs_16733)
            
            # Assigning a type to the variable 'temp_list' (line 198)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'temp_list', list_call_result_16734)
            
            # Call to append(...): (line 199)
            # Processing the call arguments (line 199)
            # Getting the type of 'idiom' (line 199)
            idiom_16737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 29), 'idiom', False)
            # Processing the call keyword arguments (line 199)
            kwargs_16738 = {}
            # Getting the type of 'temp_list' (line 199)
            temp_list_16735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'temp_list', False)
            # Obtaining the member 'append' of a type (line 199)
            append_16736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 12), temp_list_16735, 'append')
            # Calling append(args, kwargs) (line 199)
            append_call_result_16739 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), append_16736, *[idiom_16737], **kwargs_16738)
            
            
            # Call to tuple(...): (line 200)
            # Processing the call arguments (line 200)
            # Getting the type of 'temp_list' (line 200)
            temp_list_16741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 25), 'temp_list', False)
            # Processing the call keyword arguments (line 200)
            kwargs_16742 = {}
            # Getting the type of 'tuple' (line 200)
            tuple_16740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 19), 'tuple', False)
            # Calling tuple(args, kwargs) (line 200)
            tuple_call_result_16743 = invoke(stypy.reporting.localization.Localization(__file__, 200, 19), tuple_16740, *[temp_list_16741], **kwargs_16742)
            
            # Assigning a type to the variable 'stypy_return_type' (line 200)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'stypy_return_type', tuple_call_result_16743)
            # SSA join for if statement (line 197)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 202)
    tuple_16744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 202)
    # Adding element type (line 202)
    # Getting the type of 'False' (line 202)
    False_16745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 11), tuple_16744, False_16745)
    # Adding element type (line 202)
    # Getting the type of 'None' (line 202)
    None_16746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 18), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 11), tuple_16744, None_16746)
    # Adding element type (line 202)
    # Getting the type of 'None' (line 202)
    None_16747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 24), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 11), tuple_16744, None_16747)
    # Adding element type (line 202)
    # Getting the type of 'None' (line 202)
    None_16748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 30), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 11), tuple_16744, None_16748)
    
    # Assigning a type to the variable 'stypy_return_type' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stypy_return_type', tuple_16744)
    
    # ################# End of 'is_recognized_idiom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_recognized_idiom' in the type store
    # Getting the type of 'stypy_return_type' (line 186)
    stypy_return_type_16749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16749)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_recognized_idiom'
    return stypy_return_type_16749

# Assigning a type to the variable 'is_recognized_idiom' (line 186)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'is_recognized_idiom', is_recognized_idiom)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
