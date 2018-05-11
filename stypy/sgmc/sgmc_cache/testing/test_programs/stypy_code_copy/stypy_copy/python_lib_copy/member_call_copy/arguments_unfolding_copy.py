
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from ..python_types_copy.type_inference_copy import union_type_copy
2: #
3: 
4: # TODO: Remove?
5: # def __get_type(arg):
6: #     # if isinstance(arg, type.Type):
7: #     #     return arg.get_python_type()
8: #     return arg
9: #
10: #
11: # def get_arg_types(args):
12: #     return map(lambda arg: __get_type(arg), args)
13: #
14: #
15: # def get_kwarg_types(kwargs):
16: #     kwargs_types = {}
17: #     for value in kwargs:
18: #         kwargs_types[value] = __get_type(kwargs[value])
19: #
20: #     return kwargs_types
21: 
22: 
23: # ############################# UNFOLDING POSSIBLE UNION TYPES  ######################################
24: 
25: 
26: def __has_union_types(type_list):
27:     '''
28:     Determines if a list of types has union types inside it
29:     :param type_list: List of types
30:     :return: bool
31:     '''
32:     return len(filter(lambda elem: isinstance(elem, union_type_copy.UnionType), type_list)) > 0
33: 
34: 
35: def __is_union_type(obj):
36:     '''
37:     Determines if an object is a union type
38:     :param obj: Any Python object
39:     :return: bool
40:     '''
41:     return isinstance(obj, union_type_copy.UnionType)
42: 
43: 
44: def clone_list(list_):
45:     '''
46:     Shallow copy of a list.
47:     :param list_:
48:     :return:
49:     '''
50:     result = []
51:     for elem in list_:
52:         result.append(elem)
53: 
54:     return result
55: 
56: 
57: def clone_dict(dict_):
58:     '''
59:     Shallow copy of a dict
60:     :param dict_:
61:     :return:
62:     '''
63:     result = {}
64:     for elem in dict_:
65:         result[elem] = dict_[elem]
66: 
67:     return result
68: 
69: 
70: def __unfold_union_types_from_args(argument_list, possible_argument_combinations_list):
71:     '''
72:     Helper for the following function
73:     :param argument_list:
74:     :param possible_argument_combinations_list:
75:     :return:
76:     '''
77:     if not __has_union_types(argument_list):
78:         if argument_list not in possible_argument_combinations_list:
79:             possible_argument_combinations_list.append(argument_list)
80:         return
81:     for cont in range(len(argument_list)):
82:         arg = argument_list[cont]
83:         # For each union type, make type checks using each of their contained types
84:         if __is_union_type(arg):
85:             for t in arg.types:
86:                 clone = clone_list(argument_list)
87:                 clone[cont] = t
88:                 __unfold_union_types_from_args(clone, possible_argument_combinations_list)
89: 
90: 
91: def unfold_union_types_from_args(argument_list):
92:     '''
93:     Turns [(int \/ long \/ str), str] into:
94:     [
95:         (int, str),
96:         (long, str),
97:         (str, str),
98:     ]
99:     Note that if multiple union types are present, all are used to create combinations. This function is recursive.
100:     :param argument_list:
101:     :return:
102:     '''
103:     list_of_possible_args = []
104:     if __has_union_types(argument_list):
105:         __unfold_union_types_from_args(argument_list, list_of_possible_args)
106:         return list_of_possible_args
107:     else:
108:         return [argument_list]
109: 
110: 
111: def __unfold_union_types_from_kwargs(keyword_arguments_dict, possible_argument_combinations_list):
112:     '''
113:     Helper for the following function
114:     :param keyword_arguments_dict:
115:     :param possible_argument_combinations_list:
116:     :return:
117:     '''
118:     if not __has_union_types(keyword_arguments_dict.values()):
119:         if keyword_arguments_dict not in possible_argument_combinations_list:
120:             possible_argument_combinations_list.append(keyword_arguments_dict)
121:         return
122:     for elem in keyword_arguments_dict:
123:         arg = keyword_arguments_dict[elem]
124:         # For each union type, make type checks using each of their contained types
125:         if __is_union_type(arg):
126:             for t in arg.types:
127:                 clone = clone_dict(keyword_arguments_dict)
128:                 clone[elem] = t
129:                 __unfold_union_types_from_kwargs(clone, possible_argument_combinations_list)
130: 
131: 
132: def unfold_union_types_from_kwargs(keyword_argument_dict):
133:     '''
134:     Recursive function that does the same as its args-dealing equivalent, but with keyword arguments
135:     :param keyword_argument_dict:
136:     :return:
137:     '''
138:     list_of_possible_kwargs = []
139:     if __has_union_types(keyword_argument_dict.values()):
140:         __unfold_union_types_from_kwargs(keyword_argument_dict, list_of_possible_kwargs)
141:         return list_of_possible_kwargs
142:     else:
143:         return [keyword_argument_dict]
144: 
145: 
146: def unfold_arguments(*args, **kwargs):
147:     '''
148:     Turns parameter lists with union types into a a list of tuples. Each tuple contains a single type of every
149:      union type present in the original parameter list. Each tuple contains a different type of some of its union types
150:       from the other ones, so in the end all the possible combinations are generated and
151:      no union types are present in the result list. This is also done with keyword arguments. Note that if multiple
152:      union types with lots of contained types are present in the original parameter list, the result of this function
153:      may be very big. As later on every list returned by this function will be checked by a call handler, the
154:      performance of the type inference checking may suffer. However, we cannot check the types of Python library
155:      functions using other approaches, as union types cannot be properly expressed in type rules nor converted to a
156:      single Python value.
157:     :param args: Call arguments
158:     :param kwargs: Call keyword arguments
159:     :return:
160:     '''
161: 
162:     # Decompose union types among arguments
163:     unfolded_arguments = unfold_union_types_from_args(args)
164:     # Decompose union types among keyword arguments
165:     unfolded_keyword_arguments = unfold_union_types_from_kwargs(kwargs)
166:     result_arg_kwarg_tuples = []
167: 
168:     # Only keyword arguments are passed? return and empty list with each dictionary
169:     if len(unfolded_arguments) == 0:
170:         if len(unfolded_keyword_arguments) > 0:
171:             for kwarg in unfolded_keyword_arguments:
172:                 result_arg_kwarg_tuples.append(([], kwarg))
173:         else:
174:             # 0-argument call
175:             result_arg_kwarg_tuples.append(([], {}))
176:     else:
177:         # Combine each argument list returned with each keyword arguments dictionary returned, so we obtain all the
178:         # possible args, kwargs combinations.
179:         for arg in unfolded_arguments:
180:             if len(unfolded_keyword_arguments) > 0:
181:                 for kwarg in unfolded_keyword_arguments:
182:                     result_arg_kwarg_tuples.append((arg, kwarg))
183:             else:
184:                 result_arg_kwarg_tuples.append((arg, {}))
185: 
186:     return result_arg_kwarg_tuples
187: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_4926 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_4926) is not StypyTypeError):

    if (import_4926 != 'pyd_module'):
        __import__(import_4926)
        sys_modules_4927 = sys.modules[import_4926]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_4927.module_type_store, module_type_store, ['union_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_4927, sys_modules_4927.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['union_type_copy'], [union_type_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_4926)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')


@norecursion
def __has_union_types(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__has_union_types'
    module_type_store = module_type_store.open_function_context('__has_union_types', 26, 0, False)
    
    # Passed parameters checking function
    __has_union_types.stypy_localization = localization
    __has_union_types.stypy_type_of_self = None
    __has_union_types.stypy_type_store = module_type_store
    __has_union_types.stypy_function_name = '__has_union_types'
    __has_union_types.stypy_param_names_list = ['type_list']
    __has_union_types.stypy_varargs_param_name = None
    __has_union_types.stypy_kwargs_param_name = None
    __has_union_types.stypy_call_defaults = defaults
    __has_union_types.stypy_call_varargs = varargs
    __has_union_types.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__has_union_types', ['type_list'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__has_union_types', localization, ['type_list'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__has_union_types(...)' code ##################

    str_4928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'str', '\n    Determines if a list of types has union types inside it\n    :param type_list: List of types\n    :return: bool\n    ')
    
    
    # Call to len(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Call to filter(...): (line 32)
    # Processing the call arguments (line 32)

    @norecursion
    def _stypy_temp_lambda_6(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_6'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_6', 32, 22, True)
        # Passed parameters checking function
        _stypy_temp_lambda_6.stypy_localization = localization
        _stypy_temp_lambda_6.stypy_type_of_self = None
        _stypy_temp_lambda_6.stypy_type_store = module_type_store
        _stypy_temp_lambda_6.stypy_function_name = '_stypy_temp_lambda_6'
        _stypy_temp_lambda_6.stypy_param_names_list = ['elem']
        _stypy_temp_lambda_6.stypy_varargs_param_name = None
        _stypy_temp_lambda_6.stypy_kwargs_param_name = None
        _stypy_temp_lambda_6.stypy_call_defaults = defaults
        _stypy_temp_lambda_6.stypy_call_varargs = varargs
        _stypy_temp_lambda_6.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_6', ['elem'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_6', ['elem'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to isinstance(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'elem' (line 32)
        elem_4932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 46), 'elem', False)
        # Getting the type of 'union_type_copy' (line 32)
        union_type_copy_4933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 52), 'union_type_copy', False)
        # Obtaining the member 'UnionType' of a type (line 32)
        UnionType_4934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 52), union_type_copy_4933, 'UnionType')
        # Processing the call keyword arguments (line 32)
        kwargs_4935 = {}
        # Getting the type of 'isinstance' (line 32)
        isinstance_4931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 35), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 32)
        isinstance_call_result_4936 = invoke(stypy.reporting.localization.Localization(__file__, 32, 35), isinstance_4931, *[elem_4932, UnionType_4934], **kwargs_4935)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), 'stypy_return_type', isinstance_call_result_4936)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_6' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_4937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4937)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_6'
        return stypy_return_type_4937

    # Assigning a type to the variable '_stypy_temp_lambda_6' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), '_stypy_temp_lambda_6', _stypy_temp_lambda_6)
    # Getting the type of '_stypy_temp_lambda_6' (line 32)
    _stypy_temp_lambda_6_4938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), '_stypy_temp_lambda_6')
    # Getting the type of 'type_list' (line 32)
    type_list_4939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 80), 'type_list', False)
    # Processing the call keyword arguments (line 32)
    kwargs_4940 = {}
    # Getting the type of 'filter' (line 32)
    filter_4930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'filter', False)
    # Calling filter(args, kwargs) (line 32)
    filter_call_result_4941 = invoke(stypy.reporting.localization.Localization(__file__, 32, 15), filter_4930, *[_stypy_temp_lambda_6_4938, type_list_4939], **kwargs_4940)
    
    # Processing the call keyword arguments (line 32)
    kwargs_4942 = {}
    # Getting the type of 'len' (line 32)
    len_4929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'len', False)
    # Calling len(args, kwargs) (line 32)
    len_call_result_4943 = invoke(stypy.reporting.localization.Localization(__file__, 32, 11), len_4929, *[filter_call_result_4941], **kwargs_4942)
    
    int_4944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 94), 'int')
    # Applying the binary operator '>' (line 32)
    result_gt_4945 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 11), '>', len_call_result_4943, int_4944)
    
    # Assigning a type to the variable 'stypy_return_type' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type', result_gt_4945)
    
    # ################# End of '__has_union_types(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__has_union_types' in the type store
    # Getting the type of 'stypy_return_type' (line 26)
    stypy_return_type_4946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4946)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__has_union_types'
    return stypy_return_type_4946

# Assigning a type to the variable '__has_union_types' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '__has_union_types', __has_union_types)

@norecursion
def __is_union_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__is_union_type'
    module_type_store = module_type_store.open_function_context('__is_union_type', 35, 0, False)
    
    # Passed parameters checking function
    __is_union_type.stypy_localization = localization
    __is_union_type.stypy_type_of_self = None
    __is_union_type.stypy_type_store = module_type_store
    __is_union_type.stypy_function_name = '__is_union_type'
    __is_union_type.stypy_param_names_list = ['obj']
    __is_union_type.stypy_varargs_param_name = None
    __is_union_type.stypy_kwargs_param_name = None
    __is_union_type.stypy_call_defaults = defaults
    __is_union_type.stypy_call_varargs = varargs
    __is_union_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__is_union_type', ['obj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__is_union_type', localization, ['obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__is_union_type(...)' code ##################

    str_4947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'str', '\n    Determines if an object is a union type\n    :param obj: Any Python object\n    :return: bool\n    ')
    
    # Call to isinstance(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'obj' (line 41)
    obj_4949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'obj', False)
    # Getting the type of 'union_type_copy' (line 41)
    union_type_copy_4950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 27), 'union_type_copy', False)
    # Obtaining the member 'UnionType' of a type (line 41)
    UnionType_4951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 27), union_type_copy_4950, 'UnionType')
    # Processing the call keyword arguments (line 41)
    kwargs_4952 = {}
    # Getting the type of 'isinstance' (line 41)
    isinstance_4948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 41)
    isinstance_call_result_4953 = invoke(stypy.reporting.localization.Localization(__file__, 41, 11), isinstance_4948, *[obj_4949, UnionType_4951], **kwargs_4952)
    
    # Assigning a type to the variable 'stypy_return_type' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type', isinstance_call_result_4953)
    
    # ################# End of '__is_union_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__is_union_type' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_4954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4954)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__is_union_type'
    return stypy_return_type_4954

# Assigning a type to the variable '__is_union_type' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), '__is_union_type', __is_union_type)

@norecursion
def clone_list(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'clone_list'
    module_type_store = module_type_store.open_function_context('clone_list', 44, 0, False)
    
    # Passed parameters checking function
    clone_list.stypy_localization = localization
    clone_list.stypy_type_of_self = None
    clone_list.stypy_type_store = module_type_store
    clone_list.stypy_function_name = 'clone_list'
    clone_list.stypy_param_names_list = ['list_']
    clone_list.stypy_varargs_param_name = None
    clone_list.stypy_kwargs_param_name = None
    clone_list.stypy_call_defaults = defaults
    clone_list.stypy_call_varargs = varargs
    clone_list.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'clone_list', ['list_'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'clone_list', localization, ['list_'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'clone_list(...)' code ##################

    str_4955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, (-1)), 'str', '\n    Shallow copy of a list.\n    :param list_:\n    :return:\n    ')
    
    # Assigning a List to a Name (line 50):
    
    # Obtaining an instance of the builtin type 'list' (line 50)
    list_4956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 50)
    
    # Assigning a type to the variable 'result' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'result', list_4956)
    
    # Getting the type of 'list_' (line 51)
    list__4957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'list_')
    # Assigning a type to the variable 'list__4957' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'list__4957', list__4957)
    # Testing if the for loop is going to be iterated (line 51)
    # Testing the type of a for loop iterable (line 51)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 51, 4), list__4957)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 51, 4), list__4957):
        # Getting the type of the for loop variable (line 51)
        for_loop_var_4958 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 51, 4), list__4957)
        # Assigning a type to the variable 'elem' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'elem', for_loop_var_4958)
        # SSA begins for a for statement (line 51)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'elem' (line 52)
        elem_4961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 22), 'elem', False)
        # Processing the call keyword arguments (line 52)
        kwargs_4962 = {}
        # Getting the type of 'result' (line 52)
        result_4959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'result', False)
        # Obtaining the member 'append' of a type (line 52)
        append_4960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), result_4959, 'append')
        # Calling append(args, kwargs) (line 52)
        append_call_result_4963 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), append_4960, *[elem_4961], **kwargs_4962)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'result' (line 54)
    result_4964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type', result_4964)
    
    # ################# End of 'clone_list(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'clone_list' in the type store
    # Getting the type of 'stypy_return_type' (line 44)
    stypy_return_type_4965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4965)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'clone_list'
    return stypy_return_type_4965

# Assigning a type to the variable 'clone_list' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'clone_list', clone_list)

@norecursion
def clone_dict(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'clone_dict'
    module_type_store = module_type_store.open_function_context('clone_dict', 57, 0, False)
    
    # Passed parameters checking function
    clone_dict.stypy_localization = localization
    clone_dict.stypy_type_of_self = None
    clone_dict.stypy_type_store = module_type_store
    clone_dict.stypy_function_name = 'clone_dict'
    clone_dict.stypy_param_names_list = ['dict_']
    clone_dict.stypy_varargs_param_name = None
    clone_dict.stypy_kwargs_param_name = None
    clone_dict.stypy_call_defaults = defaults
    clone_dict.stypy_call_varargs = varargs
    clone_dict.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'clone_dict', ['dict_'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'clone_dict', localization, ['dict_'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'clone_dict(...)' code ##################

    str_4966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'str', '\n    Shallow copy of a dict\n    :param dict_:\n    :return:\n    ')
    
    # Assigning a Dict to a Name (line 63):
    
    # Obtaining an instance of the builtin type 'dict' (line 63)
    dict_4967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 63)
    
    # Assigning a type to the variable 'result' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'result', dict_4967)
    
    # Getting the type of 'dict_' (line 64)
    dict__4968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'dict_')
    # Assigning a type to the variable 'dict__4968' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'dict__4968', dict__4968)
    # Testing if the for loop is going to be iterated (line 64)
    # Testing the type of a for loop iterable (line 64)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 64, 4), dict__4968)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 64, 4), dict__4968):
        # Getting the type of the for loop variable (line 64)
        for_loop_var_4969 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 64, 4), dict__4968)
        # Assigning a type to the variable 'elem' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'elem', for_loop_var_4969)
        # SSA begins for a for statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Subscript (line 65):
        
        # Obtaining the type of the subscript
        # Getting the type of 'elem' (line 65)
        elem_4970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 29), 'elem')
        # Getting the type of 'dict_' (line 65)
        dict__4971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 23), 'dict_')
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___4972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 23), dict__4971, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_4973 = invoke(stypy.reporting.localization.Localization(__file__, 65, 23), getitem___4972, elem_4970)
        
        # Getting the type of 'result' (line 65)
        result_4974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'result')
        # Getting the type of 'elem' (line 65)
        elem_4975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'elem')
        # Storing an element on a container (line 65)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 8), result_4974, (elem_4975, subscript_call_result_4973))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'result' (line 67)
    result_4976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type', result_4976)
    
    # ################# End of 'clone_dict(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'clone_dict' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_4977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4977)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'clone_dict'
    return stypy_return_type_4977

# Assigning a type to the variable 'clone_dict' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'clone_dict', clone_dict)

@norecursion
def __unfold_union_types_from_args(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__unfold_union_types_from_args'
    module_type_store = module_type_store.open_function_context('__unfold_union_types_from_args', 70, 0, False)
    
    # Passed parameters checking function
    __unfold_union_types_from_args.stypy_localization = localization
    __unfold_union_types_from_args.stypy_type_of_self = None
    __unfold_union_types_from_args.stypy_type_store = module_type_store
    __unfold_union_types_from_args.stypy_function_name = '__unfold_union_types_from_args'
    __unfold_union_types_from_args.stypy_param_names_list = ['argument_list', 'possible_argument_combinations_list']
    __unfold_union_types_from_args.stypy_varargs_param_name = None
    __unfold_union_types_from_args.stypy_kwargs_param_name = None
    __unfold_union_types_from_args.stypy_call_defaults = defaults
    __unfold_union_types_from_args.stypy_call_varargs = varargs
    __unfold_union_types_from_args.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__unfold_union_types_from_args', ['argument_list', 'possible_argument_combinations_list'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__unfold_union_types_from_args', localization, ['argument_list', 'possible_argument_combinations_list'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__unfold_union_types_from_args(...)' code ##################

    str_4978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, (-1)), 'str', '\n    Helper for the following function\n    :param argument_list:\n    :param possible_argument_combinations_list:\n    :return:\n    ')
    
    
    # Call to __has_union_types(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'argument_list' (line 77)
    argument_list_4980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'argument_list', False)
    # Processing the call keyword arguments (line 77)
    kwargs_4981 = {}
    # Getting the type of '__has_union_types' (line 77)
    has_union_types_4979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), '__has_union_types', False)
    # Calling __has_union_types(args, kwargs) (line 77)
    has_union_types_call_result_4982 = invoke(stypy.reporting.localization.Localization(__file__, 77, 11), has_union_types_4979, *[argument_list_4980], **kwargs_4981)
    
    # Applying the 'not' unary operator (line 77)
    result_not__4983 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 7), 'not', has_union_types_call_result_4982)
    
    # Testing if the type of an if condition is none (line 77)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 77, 4), result_not__4983):
        pass
    else:
        
        # Testing the type of an if condition (line 77)
        if_condition_4984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 4), result_not__4983)
        # Assigning a type to the variable 'if_condition_4984' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'if_condition_4984', if_condition_4984)
        # SSA begins for if statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'argument_list' (line 78)
        argument_list_4985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'argument_list')
        # Getting the type of 'possible_argument_combinations_list' (line 78)
        possible_argument_combinations_list_4986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 32), 'possible_argument_combinations_list')
        # Applying the binary operator 'notin' (line 78)
        result_contains_4987 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 11), 'notin', argument_list_4985, possible_argument_combinations_list_4986)
        
        # Testing if the type of an if condition is none (line 78)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 78, 8), result_contains_4987):
            pass
        else:
            
            # Testing the type of an if condition (line 78)
            if_condition_4988 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 8), result_contains_4987)
            # Assigning a type to the variable 'if_condition_4988' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'if_condition_4988', if_condition_4988)
            # SSA begins for if statement (line 78)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 79)
            # Processing the call arguments (line 79)
            # Getting the type of 'argument_list' (line 79)
            argument_list_4991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 55), 'argument_list', False)
            # Processing the call keyword arguments (line 79)
            kwargs_4992 = {}
            # Getting the type of 'possible_argument_combinations_list' (line 79)
            possible_argument_combinations_list_4989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'possible_argument_combinations_list', False)
            # Obtaining the member 'append' of a type (line 79)
            append_4990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 12), possible_argument_combinations_list_4989, 'append')
            # Calling append(args, kwargs) (line 79)
            append_call_result_4993 = invoke(stypy.reporting.localization.Localization(__file__, 79, 12), append_4990, *[argument_list_4991], **kwargs_4992)
            
            # SSA join for if statement (line 78)
            module_type_store = module_type_store.join_ssa_context()
            

        # Assigning a type to the variable 'stypy_return_type' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 77)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to range(...): (line 81)
    # Processing the call arguments (line 81)
    
    # Call to len(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'argument_list' (line 81)
    argument_list_4996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 26), 'argument_list', False)
    # Processing the call keyword arguments (line 81)
    kwargs_4997 = {}
    # Getting the type of 'len' (line 81)
    len_4995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'len', False)
    # Calling len(args, kwargs) (line 81)
    len_call_result_4998 = invoke(stypy.reporting.localization.Localization(__file__, 81, 22), len_4995, *[argument_list_4996], **kwargs_4997)
    
    # Processing the call keyword arguments (line 81)
    kwargs_4999 = {}
    # Getting the type of 'range' (line 81)
    range_4994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'range', False)
    # Calling range(args, kwargs) (line 81)
    range_call_result_5000 = invoke(stypy.reporting.localization.Localization(__file__, 81, 16), range_4994, *[len_call_result_4998], **kwargs_4999)
    
    # Assigning a type to the variable 'range_call_result_5000' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'range_call_result_5000', range_call_result_5000)
    # Testing if the for loop is going to be iterated (line 81)
    # Testing the type of a for loop iterable (line 81)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 81, 4), range_call_result_5000)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 81, 4), range_call_result_5000):
        # Getting the type of the for loop variable (line 81)
        for_loop_var_5001 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 81, 4), range_call_result_5000)
        # Assigning a type to the variable 'cont' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'cont', for_loop_var_5001)
        # SSA begins for a for statement (line 81)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 82):
        
        # Obtaining the type of the subscript
        # Getting the type of 'cont' (line 82)
        cont_5002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 28), 'cont')
        # Getting the type of 'argument_list' (line 82)
        argument_list_5003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 14), 'argument_list')
        # Obtaining the member '__getitem__' of a type (line 82)
        getitem___5004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 14), argument_list_5003, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 82)
        subscript_call_result_5005 = invoke(stypy.reporting.localization.Localization(__file__, 82, 14), getitem___5004, cont_5002)
        
        # Assigning a type to the variable 'arg' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'arg', subscript_call_result_5005)
        
        # Call to __is_union_type(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'arg' (line 84)
        arg_5007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'arg', False)
        # Processing the call keyword arguments (line 84)
        kwargs_5008 = {}
        # Getting the type of '__is_union_type' (line 84)
        is_union_type_5006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), '__is_union_type', False)
        # Calling __is_union_type(args, kwargs) (line 84)
        is_union_type_call_result_5009 = invoke(stypy.reporting.localization.Localization(__file__, 84, 11), is_union_type_5006, *[arg_5007], **kwargs_5008)
        
        # Testing if the type of an if condition is none (line 84)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 84, 8), is_union_type_call_result_5009):
            pass
        else:
            
            # Testing the type of an if condition (line 84)
            if_condition_5010 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 8), is_union_type_call_result_5009)
            # Assigning a type to the variable 'if_condition_5010' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'if_condition_5010', if_condition_5010)
            # SSA begins for if statement (line 84)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'arg' (line 85)
            arg_5011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'arg')
            # Obtaining the member 'types' of a type (line 85)
            types_5012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 21), arg_5011, 'types')
            # Assigning a type to the variable 'types_5012' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'types_5012', types_5012)
            # Testing if the for loop is going to be iterated (line 85)
            # Testing the type of a for loop iterable (line 85)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 85, 12), types_5012)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 85, 12), types_5012):
                # Getting the type of the for loop variable (line 85)
                for_loop_var_5013 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 85, 12), types_5012)
                # Assigning a type to the variable 't' (line 85)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 't', for_loop_var_5013)
                # SSA begins for a for statement (line 85)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 86):
                
                # Call to clone_list(...): (line 86)
                # Processing the call arguments (line 86)
                # Getting the type of 'argument_list' (line 86)
                argument_list_5015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 35), 'argument_list', False)
                # Processing the call keyword arguments (line 86)
                kwargs_5016 = {}
                # Getting the type of 'clone_list' (line 86)
                clone_list_5014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'clone_list', False)
                # Calling clone_list(args, kwargs) (line 86)
                clone_list_call_result_5017 = invoke(stypy.reporting.localization.Localization(__file__, 86, 24), clone_list_5014, *[argument_list_5015], **kwargs_5016)
                
                # Assigning a type to the variable 'clone' (line 86)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'clone', clone_list_call_result_5017)
                
                # Assigning a Name to a Subscript (line 87):
                # Getting the type of 't' (line 87)
                t_5018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 30), 't')
                # Getting the type of 'clone' (line 87)
                clone_5019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'clone')
                # Getting the type of 'cont' (line 87)
                cont_5020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 22), 'cont')
                # Storing an element on a container (line 87)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 16), clone_5019, (cont_5020, t_5018))
                
                # Call to __unfold_union_types_from_args(...): (line 88)
                # Processing the call arguments (line 88)
                # Getting the type of 'clone' (line 88)
                clone_5022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 47), 'clone', False)
                # Getting the type of 'possible_argument_combinations_list' (line 88)
                possible_argument_combinations_list_5023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 54), 'possible_argument_combinations_list', False)
                # Processing the call keyword arguments (line 88)
                kwargs_5024 = {}
                # Getting the type of '__unfold_union_types_from_args' (line 88)
                unfold_union_types_from_args_5021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), '__unfold_union_types_from_args', False)
                # Calling __unfold_union_types_from_args(args, kwargs) (line 88)
                unfold_union_types_from_args_call_result_5025 = invoke(stypy.reporting.localization.Localization(__file__, 88, 16), unfold_union_types_from_args_5021, *[clone_5022, possible_argument_combinations_list_5023], **kwargs_5024)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 84)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of '__unfold_union_types_from_args(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__unfold_union_types_from_args' in the type store
    # Getting the type of 'stypy_return_type' (line 70)
    stypy_return_type_5026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5026)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__unfold_union_types_from_args'
    return stypy_return_type_5026

# Assigning a type to the variable '__unfold_union_types_from_args' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), '__unfold_union_types_from_args', __unfold_union_types_from_args)

@norecursion
def unfold_union_types_from_args(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'unfold_union_types_from_args'
    module_type_store = module_type_store.open_function_context('unfold_union_types_from_args', 91, 0, False)
    
    # Passed parameters checking function
    unfold_union_types_from_args.stypy_localization = localization
    unfold_union_types_from_args.stypy_type_of_self = None
    unfold_union_types_from_args.stypy_type_store = module_type_store
    unfold_union_types_from_args.stypy_function_name = 'unfold_union_types_from_args'
    unfold_union_types_from_args.stypy_param_names_list = ['argument_list']
    unfold_union_types_from_args.stypy_varargs_param_name = None
    unfold_union_types_from_args.stypy_kwargs_param_name = None
    unfold_union_types_from_args.stypy_call_defaults = defaults
    unfold_union_types_from_args.stypy_call_varargs = varargs
    unfold_union_types_from_args.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unfold_union_types_from_args', ['argument_list'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unfold_union_types_from_args', localization, ['argument_list'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unfold_union_types_from_args(...)' code ##################

    str_5027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, (-1)), 'str', '\n    Turns [(int \\/ long \\/ str), str] into:\n    [\n        (int, str),\n        (long, str),\n        (str, str),\n    ]\n    Note that if multiple union types are present, all are used to create combinations. This function is recursive.\n    :param argument_list:\n    :return:\n    ')
    
    # Assigning a List to a Name (line 103):
    
    # Obtaining an instance of the builtin type 'list' (line 103)
    list_5028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 103)
    
    # Assigning a type to the variable 'list_of_possible_args' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'list_of_possible_args', list_5028)
    
    # Call to __has_union_types(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'argument_list' (line 104)
    argument_list_5030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 25), 'argument_list', False)
    # Processing the call keyword arguments (line 104)
    kwargs_5031 = {}
    # Getting the type of '__has_union_types' (line 104)
    has_union_types_5029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 7), '__has_union_types', False)
    # Calling __has_union_types(args, kwargs) (line 104)
    has_union_types_call_result_5032 = invoke(stypy.reporting.localization.Localization(__file__, 104, 7), has_union_types_5029, *[argument_list_5030], **kwargs_5031)
    
    # Testing if the type of an if condition is none (line 104)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 104, 4), has_union_types_call_result_5032):
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_5040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        # Getting the type of 'argument_list' (line 108)
        argument_list_5041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'argument_list')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), list_5040, argument_list_5041)
        
        # Assigning a type to the variable 'stypy_return_type' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'stypy_return_type', list_5040)
    else:
        
        # Testing the type of an if condition (line 104)
        if_condition_5033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 4), has_union_types_call_result_5032)
        # Assigning a type to the variable 'if_condition_5033' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'if_condition_5033', if_condition_5033)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __unfold_union_types_from_args(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'argument_list' (line 105)
        argument_list_5035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 39), 'argument_list', False)
        # Getting the type of 'list_of_possible_args' (line 105)
        list_of_possible_args_5036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 54), 'list_of_possible_args', False)
        # Processing the call keyword arguments (line 105)
        kwargs_5037 = {}
        # Getting the type of '__unfold_union_types_from_args' (line 105)
        unfold_union_types_from_args_5034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), '__unfold_union_types_from_args', False)
        # Calling __unfold_union_types_from_args(args, kwargs) (line 105)
        unfold_union_types_from_args_call_result_5038 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), unfold_union_types_from_args_5034, *[argument_list_5035, list_of_possible_args_5036], **kwargs_5037)
        
        # Getting the type of 'list_of_possible_args' (line 106)
        list_of_possible_args_5039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'list_of_possible_args')
        # Assigning a type to the variable 'stypy_return_type' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'stypy_return_type', list_of_possible_args_5039)
        # SSA branch for the else part of an if statement (line 104)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_5040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        # Getting the type of 'argument_list' (line 108)
        argument_list_5041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'argument_list')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), list_5040, argument_list_5041)
        
        # Assigning a type to the variable 'stypy_return_type' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'stypy_return_type', list_5040)
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'unfold_union_types_from_args(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unfold_union_types_from_args' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_5042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5042)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unfold_union_types_from_args'
    return stypy_return_type_5042

# Assigning a type to the variable 'unfold_union_types_from_args' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'unfold_union_types_from_args', unfold_union_types_from_args)

@norecursion
def __unfold_union_types_from_kwargs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__unfold_union_types_from_kwargs'
    module_type_store = module_type_store.open_function_context('__unfold_union_types_from_kwargs', 111, 0, False)
    
    # Passed parameters checking function
    __unfold_union_types_from_kwargs.stypy_localization = localization
    __unfold_union_types_from_kwargs.stypy_type_of_self = None
    __unfold_union_types_from_kwargs.stypy_type_store = module_type_store
    __unfold_union_types_from_kwargs.stypy_function_name = '__unfold_union_types_from_kwargs'
    __unfold_union_types_from_kwargs.stypy_param_names_list = ['keyword_arguments_dict', 'possible_argument_combinations_list']
    __unfold_union_types_from_kwargs.stypy_varargs_param_name = None
    __unfold_union_types_from_kwargs.stypy_kwargs_param_name = None
    __unfold_union_types_from_kwargs.stypy_call_defaults = defaults
    __unfold_union_types_from_kwargs.stypy_call_varargs = varargs
    __unfold_union_types_from_kwargs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__unfold_union_types_from_kwargs', ['keyword_arguments_dict', 'possible_argument_combinations_list'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__unfold_union_types_from_kwargs', localization, ['keyword_arguments_dict', 'possible_argument_combinations_list'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__unfold_union_types_from_kwargs(...)' code ##################

    str_5043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', '\n    Helper for the following function\n    :param keyword_arguments_dict:\n    :param possible_argument_combinations_list:\n    :return:\n    ')
    
    
    # Call to __has_union_types(...): (line 118)
    # Processing the call arguments (line 118)
    
    # Call to values(...): (line 118)
    # Processing the call keyword arguments (line 118)
    kwargs_5047 = {}
    # Getting the type of 'keyword_arguments_dict' (line 118)
    keyword_arguments_dict_5045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 29), 'keyword_arguments_dict', False)
    # Obtaining the member 'values' of a type (line 118)
    values_5046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 29), keyword_arguments_dict_5045, 'values')
    # Calling values(args, kwargs) (line 118)
    values_call_result_5048 = invoke(stypy.reporting.localization.Localization(__file__, 118, 29), values_5046, *[], **kwargs_5047)
    
    # Processing the call keyword arguments (line 118)
    kwargs_5049 = {}
    # Getting the type of '__has_union_types' (line 118)
    has_union_types_5044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 11), '__has_union_types', False)
    # Calling __has_union_types(args, kwargs) (line 118)
    has_union_types_call_result_5050 = invoke(stypy.reporting.localization.Localization(__file__, 118, 11), has_union_types_5044, *[values_call_result_5048], **kwargs_5049)
    
    # Applying the 'not' unary operator (line 118)
    result_not__5051 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 7), 'not', has_union_types_call_result_5050)
    
    # Testing if the type of an if condition is none (line 118)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 118, 4), result_not__5051):
        pass
    else:
        
        # Testing the type of an if condition (line 118)
        if_condition_5052 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 4), result_not__5051)
        # Assigning a type to the variable 'if_condition_5052' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'if_condition_5052', if_condition_5052)
        # SSA begins for if statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'keyword_arguments_dict' (line 119)
        keyword_arguments_dict_5053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'keyword_arguments_dict')
        # Getting the type of 'possible_argument_combinations_list' (line 119)
        possible_argument_combinations_list_5054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 41), 'possible_argument_combinations_list')
        # Applying the binary operator 'notin' (line 119)
        result_contains_5055 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 11), 'notin', keyword_arguments_dict_5053, possible_argument_combinations_list_5054)
        
        # Testing if the type of an if condition is none (line 119)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 119, 8), result_contains_5055):
            pass
        else:
            
            # Testing the type of an if condition (line 119)
            if_condition_5056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 8), result_contains_5055)
            # Assigning a type to the variable 'if_condition_5056' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'if_condition_5056', if_condition_5056)
            # SSA begins for if statement (line 119)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 120)
            # Processing the call arguments (line 120)
            # Getting the type of 'keyword_arguments_dict' (line 120)
            keyword_arguments_dict_5059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 55), 'keyword_arguments_dict', False)
            # Processing the call keyword arguments (line 120)
            kwargs_5060 = {}
            # Getting the type of 'possible_argument_combinations_list' (line 120)
            possible_argument_combinations_list_5057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'possible_argument_combinations_list', False)
            # Obtaining the member 'append' of a type (line 120)
            append_5058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), possible_argument_combinations_list_5057, 'append')
            # Calling append(args, kwargs) (line 120)
            append_call_result_5061 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), append_5058, *[keyword_arguments_dict_5059], **kwargs_5060)
            
            # SSA join for if statement (line 119)
            module_type_store = module_type_store.join_ssa_context()
            

        # Assigning a type to the variable 'stypy_return_type' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 118)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'keyword_arguments_dict' (line 122)
    keyword_arguments_dict_5062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'keyword_arguments_dict')
    # Assigning a type to the variable 'keyword_arguments_dict_5062' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'keyword_arguments_dict_5062', keyword_arguments_dict_5062)
    # Testing if the for loop is going to be iterated (line 122)
    # Testing the type of a for loop iterable (line 122)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 122, 4), keyword_arguments_dict_5062)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 122, 4), keyword_arguments_dict_5062):
        # Getting the type of the for loop variable (line 122)
        for_loop_var_5063 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 122, 4), keyword_arguments_dict_5062)
        # Assigning a type to the variable 'elem' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'elem', for_loop_var_5063)
        # SSA begins for a for statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 123):
        
        # Obtaining the type of the subscript
        # Getting the type of 'elem' (line 123)
        elem_5064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 37), 'elem')
        # Getting the type of 'keyword_arguments_dict' (line 123)
        keyword_arguments_dict_5065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 14), 'keyword_arguments_dict')
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___5066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 14), keyword_arguments_dict_5065, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_5067 = invoke(stypy.reporting.localization.Localization(__file__, 123, 14), getitem___5066, elem_5064)
        
        # Assigning a type to the variable 'arg' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'arg', subscript_call_result_5067)
        
        # Call to __is_union_type(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'arg' (line 125)
        arg_5069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'arg', False)
        # Processing the call keyword arguments (line 125)
        kwargs_5070 = {}
        # Getting the type of '__is_union_type' (line 125)
        is_union_type_5068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), '__is_union_type', False)
        # Calling __is_union_type(args, kwargs) (line 125)
        is_union_type_call_result_5071 = invoke(stypy.reporting.localization.Localization(__file__, 125, 11), is_union_type_5068, *[arg_5069], **kwargs_5070)
        
        # Testing if the type of an if condition is none (line 125)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 8), is_union_type_call_result_5071):
            pass
        else:
            
            # Testing the type of an if condition (line 125)
            if_condition_5072 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), is_union_type_call_result_5071)
            # Assigning a type to the variable 'if_condition_5072' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'if_condition_5072', if_condition_5072)
            # SSA begins for if statement (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'arg' (line 126)
            arg_5073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 21), 'arg')
            # Obtaining the member 'types' of a type (line 126)
            types_5074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 21), arg_5073, 'types')
            # Assigning a type to the variable 'types_5074' (line 126)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'types_5074', types_5074)
            # Testing if the for loop is going to be iterated (line 126)
            # Testing the type of a for loop iterable (line 126)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 126, 12), types_5074)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 126, 12), types_5074):
                # Getting the type of the for loop variable (line 126)
                for_loop_var_5075 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 126, 12), types_5074)
                # Assigning a type to the variable 't' (line 126)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 't', for_loop_var_5075)
                # SSA begins for a for statement (line 126)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 127):
                
                # Call to clone_dict(...): (line 127)
                # Processing the call arguments (line 127)
                # Getting the type of 'keyword_arguments_dict' (line 127)
                keyword_arguments_dict_5077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 35), 'keyword_arguments_dict', False)
                # Processing the call keyword arguments (line 127)
                kwargs_5078 = {}
                # Getting the type of 'clone_dict' (line 127)
                clone_dict_5076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'clone_dict', False)
                # Calling clone_dict(args, kwargs) (line 127)
                clone_dict_call_result_5079 = invoke(stypy.reporting.localization.Localization(__file__, 127, 24), clone_dict_5076, *[keyword_arguments_dict_5077], **kwargs_5078)
                
                # Assigning a type to the variable 'clone' (line 127)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'clone', clone_dict_call_result_5079)
                
                # Assigning a Name to a Subscript (line 128):
                # Getting the type of 't' (line 128)
                t_5080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 30), 't')
                # Getting the type of 'clone' (line 128)
                clone_5081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'clone')
                # Getting the type of 'elem' (line 128)
                elem_5082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 22), 'elem')
                # Storing an element on a container (line 128)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 16), clone_5081, (elem_5082, t_5080))
                
                # Call to __unfold_union_types_from_kwargs(...): (line 129)
                # Processing the call arguments (line 129)
                # Getting the type of 'clone' (line 129)
                clone_5084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 49), 'clone', False)
                # Getting the type of 'possible_argument_combinations_list' (line 129)
                possible_argument_combinations_list_5085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 56), 'possible_argument_combinations_list', False)
                # Processing the call keyword arguments (line 129)
                kwargs_5086 = {}
                # Getting the type of '__unfold_union_types_from_kwargs' (line 129)
                unfold_union_types_from_kwargs_5083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), '__unfold_union_types_from_kwargs', False)
                # Calling __unfold_union_types_from_kwargs(args, kwargs) (line 129)
                unfold_union_types_from_kwargs_call_result_5087 = invoke(stypy.reporting.localization.Localization(__file__, 129, 16), unfold_union_types_from_kwargs_5083, *[clone_5084, possible_argument_combinations_list_5085], **kwargs_5086)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 125)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of '__unfold_union_types_from_kwargs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__unfold_union_types_from_kwargs' in the type store
    # Getting the type of 'stypy_return_type' (line 111)
    stypy_return_type_5088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5088)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__unfold_union_types_from_kwargs'
    return stypy_return_type_5088

# Assigning a type to the variable '__unfold_union_types_from_kwargs' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), '__unfold_union_types_from_kwargs', __unfold_union_types_from_kwargs)

@norecursion
def unfold_union_types_from_kwargs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'unfold_union_types_from_kwargs'
    module_type_store = module_type_store.open_function_context('unfold_union_types_from_kwargs', 132, 0, False)
    
    # Passed parameters checking function
    unfold_union_types_from_kwargs.stypy_localization = localization
    unfold_union_types_from_kwargs.stypy_type_of_self = None
    unfold_union_types_from_kwargs.stypy_type_store = module_type_store
    unfold_union_types_from_kwargs.stypy_function_name = 'unfold_union_types_from_kwargs'
    unfold_union_types_from_kwargs.stypy_param_names_list = ['keyword_argument_dict']
    unfold_union_types_from_kwargs.stypy_varargs_param_name = None
    unfold_union_types_from_kwargs.stypy_kwargs_param_name = None
    unfold_union_types_from_kwargs.stypy_call_defaults = defaults
    unfold_union_types_from_kwargs.stypy_call_varargs = varargs
    unfold_union_types_from_kwargs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unfold_union_types_from_kwargs', ['keyword_argument_dict'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unfold_union_types_from_kwargs', localization, ['keyword_argument_dict'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unfold_union_types_from_kwargs(...)' code ##################

    str_5089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, (-1)), 'str', '\n    Recursive function that does the same as its args-dealing equivalent, but with keyword arguments\n    :param keyword_argument_dict:\n    :return:\n    ')
    
    # Assigning a List to a Name (line 138):
    
    # Obtaining an instance of the builtin type 'list' (line 138)
    list_5090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 138)
    
    # Assigning a type to the variable 'list_of_possible_kwargs' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'list_of_possible_kwargs', list_5090)
    
    # Call to __has_union_types(...): (line 139)
    # Processing the call arguments (line 139)
    
    # Call to values(...): (line 139)
    # Processing the call keyword arguments (line 139)
    kwargs_5094 = {}
    # Getting the type of 'keyword_argument_dict' (line 139)
    keyword_argument_dict_5092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 25), 'keyword_argument_dict', False)
    # Obtaining the member 'values' of a type (line 139)
    values_5093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 25), keyword_argument_dict_5092, 'values')
    # Calling values(args, kwargs) (line 139)
    values_call_result_5095 = invoke(stypy.reporting.localization.Localization(__file__, 139, 25), values_5093, *[], **kwargs_5094)
    
    # Processing the call keyword arguments (line 139)
    kwargs_5096 = {}
    # Getting the type of '__has_union_types' (line 139)
    has_union_types_5091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 7), '__has_union_types', False)
    # Calling __has_union_types(args, kwargs) (line 139)
    has_union_types_call_result_5097 = invoke(stypy.reporting.localization.Localization(__file__, 139, 7), has_union_types_5091, *[values_call_result_5095], **kwargs_5096)
    
    # Testing if the type of an if condition is none (line 139)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 139, 4), has_union_types_call_result_5097):
        
        # Obtaining an instance of the builtin type 'list' (line 143)
        list_5105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 143)
        # Adding element type (line 143)
        # Getting the type of 'keyword_argument_dict' (line 143)
        keyword_argument_dict_5106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'keyword_argument_dict')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 15), list_5105, keyword_argument_dict_5106)
        
        # Assigning a type to the variable 'stypy_return_type' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'stypy_return_type', list_5105)
    else:
        
        # Testing the type of an if condition (line 139)
        if_condition_5098 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 4), has_union_types_call_result_5097)
        # Assigning a type to the variable 'if_condition_5098' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'if_condition_5098', if_condition_5098)
        # SSA begins for if statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __unfold_union_types_from_kwargs(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'keyword_argument_dict' (line 140)
        keyword_argument_dict_5100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 41), 'keyword_argument_dict', False)
        # Getting the type of 'list_of_possible_kwargs' (line 140)
        list_of_possible_kwargs_5101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 64), 'list_of_possible_kwargs', False)
        # Processing the call keyword arguments (line 140)
        kwargs_5102 = {}
        # Getting the type of '__unfold_union_types_from_kwargs' (line 140)
        unfold_union_types_from_kwargs_5099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), '__unfold_union_types_from_kwargs', False)
        # Calling __unfold_union_types_from_kwargs(args, kwargs) (line 140)
        unfold_union_types_from_kwargs_call_result_5103 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), unfold_union_types_from_kwargs_5099, *[keyword_argument_dict_5100, list_of_possible_kwargs_5101], **kwargs_5102)
        
        # Getting the type of 'list_of_possible_kwargs' (line 141)
        list_of_possible_kwargs_5104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'list_of_possible_kwargs')
        # Assigning a type to the variable 'stypy_return_type' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'stypy_return_type', list_of_possible_kwargs_5104)
        # SSA branch for the else part of an if statement (line 139)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining an instance of the builtin type 'list' (line 143)
        list_5105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 143)
        # Adding element type (line 143)
        # Getting the type of 'keyword_argument_dict' (line 143)
        keyword_argument_dict_5106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'keyword_argument_dict')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 15), list_5105, keyword_argument_dict_5106)
        
        # Assigning a type to the variable 'stypy_return_type' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'stypy_return_type', list_5105)
        # SSA join for if statement (line 139)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'unfold_union_types_from_kwargs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unfold_union_types_from_kwargs' in the type store
    # Getting the type of 'stypy_return_type' (line 132)
    stypy_return_type_5107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5107)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unfold_union_types_from_kwargs'
    return stypy_return_type_5107

# Assigning a type to the variable 'unfold_union_types_from_kwargs' (line 132)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'unfold_union_types_from_kwargs', unfold_union_types_from_kwargs)

@norecursion
def unfold_arguments(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'unfold_arguments'
    module_type_store = module_type_store.open_function_context('unfold_arguments', 146, 0, False)
    
    # Passed parameters checking function
    unfold_arguments.stypy_localization = localization
    unfold_arguments.stypy_type_of_self = None
    unfold_arguments.stypy_type_store = module_type_store
    unfold_arguments.stypy_function_name = 'unfold_arguments'
    unfold_arguments.stypy_param_names_list = []
    unfold_arguments.stypy_varargs_param_name = 'args'
    unfold_arguments.stypy_kwargs_param_name = 'kwargs'
    unfold_arguments.stypy_call_defaults = defaults
    unfold_arguments.stypy_call_varargs = varargs
    unfold_arguments.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unfold_arguments', [], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unfold_arguments', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unfold_arguments(...)' code ##################

    str_5108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, (-1)), 'str', '\n    Turns parameter lists with union types into a a list of tuples. Each tuple contains a single type of every\n     union type present in the original parameter list. Each tuple contains a different type of some of its union types\n      from the other ones, so in the end all the possible combinations are generated and\n     no union types are present in the result list. This is also done with keyword arguments. Note that if multiple\n     union types with lots of contained types are present in the original parameter list, the result of this function\n     may be very big. As later on every list returned by this function will be checked by a call handler, the\n     performance of the type inference checking may suffer. However, we cannot check the types of Python library\n     functions using other approaches, as union types cannot be properly expressed in type rules nor converted to a\n     single Python value.\n    :param args: Call arguments\n    :param kwargs: Call keyword arguments\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 163):
    
    # Call to unfold_union_types_from_args(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'args' (line 163)
    args_5110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 54), 'args', False)
    # Processing the call keyword arguments (line 163)
    kwargs_5111 = {}
    # Getting the type of 'unfold_union_types_from_args' (line 163)
    unfold_union_types_from_args_5109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 25), 'unfold_union_types_from_args', False)
    # Calling unfold_union_types_from_args(args, kwargs) (line 163)
    unfold_union_types_from_args_call_result_5112 = invoke(stypy.reporting.localization.Localization(__file__, 163, 25), unfold_union_types_from_args_5109, *[args_5110], **kwargs_5111)
    
    # Assigning a type to the variable 'unfolded_arguments' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'unfolded_arguments', unfold_union_types_from_args_call_result_5112)
    
    # Assigning a Call to a Name (line 165):
    
    # Call to unfold_union_types_from_kwargs(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'kwargs' (line 165)
    kwargs_5114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 64), 'kwargs', False)
    # Processing the call keyword arguments (line 165)
    kwargs_5115 = {}
    # Getting the type of 'unfold_union_types_from_kwargs' (line 165)
    unfold_union_types_from_kwargs_5113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 33), 'unfold_union_types_from_kwargs', False)
    # Calling unfold_union_types_from_kwargs(args, kwargs) (line 165)
    unfold_union_types_from_kwargs_call_result_5116 = invoke(stypy.reporting.localization.Localization(__file__, 165, 33), unfold_union_types_from_kwargs_5113, *[kwargs_5114], **kwargs_5115)
    
    # Assigning a type to the variable 'unfolded_keyword_arguments' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'unfolded_keyword_arguments', unfold_union_types_from_kwargs_call_result_5116)
    
    # Assigning a List to a Name (line 166):
    
    # Obtaining an instance of the builtin type 'list' (line 166)
    list_5117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 166)
    
    # Assigning a type to the variable 'result_arg_kwarg_tuples' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'result_arg_kwarg_tuples', list_5117)
    
    
    # Call to len(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'unfolded_arguments' (line 169)
    unfolded_arguments_5119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 11), 'unfolded_arguments', False)
    # Processing the call keyword arguments (line 169)
    kwargs_5120 = {}
    # Getting the type of 'len' (line 169)
    len_5118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 7), 'len', False)
    # Calling len(args, kwargs) (line 169)
    len_call_result_5121 = invoke(stypy.reporting.localization.Localization(__file__, 169, 7), len_5118, *[unfolded_arguments_5119], **kwargs_5120)
    
    int_5122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 34), 'int')
    # Applying the binary operator '==' (line 169)
    result_eq_5123 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 7), '==', len_call_result_5121, int_5122)
    
    # Testing if the type of an if condition is none (line 169)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 169, 4), result_eq_5123):
        
        # Getting the type of 'unfolded_arguments' (line 179)
        unfolded_arguments_5148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 19), 'unfolded_arguments')
        # Assigning a type to the variable 'unfolded_arguments_5148' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'unfolded_arguments_5148', unfolded_arguments_5148)
        # Testing if the for loop is going to be iterated (line 179)
        # Testing the type of a for loop iterable (line 179)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 179, 8), unfolded_arguments_5148)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 179, 8), unfolded_arguments_5148):
            # Getting the type of the for loop variable (line 179)
            for_loop_var_5149 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 179, 8), unfolded_arguments_5148)
            # Assigning a type to the variable 'arg' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'arg', for_loop_var_5149)
            # SSA begins for a for statement (line 179)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to len(...): (line 180)
            # Processing the call arguments (line 180)
            # Getting the type of 'unfolded_keyword_arguments' (line 180)
            unfolded_keyword_arguments_5151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'unfolded_keyword_arguments', False)
            # Processing the call keyword arguments (line 180)
            kwargs_5152 = {}
            # Getting the type of 'len' (line 180)
            len_5150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 15), 'len', False)
            # Calling len(args, kwargs) (line 180)
            len_call_result_5153 = invoke(stypy.reporting.localization.Localization(__file__, 180, 15), len_5150, *[unfolded_keyword_arguments_5151], **kwargs_5152)
            
            int_5154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 49), 'int')
            # Applying the binary operator '>' (line 180)
            result_gt_5155 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 15), '>', len_call_result_5153, int_5154)
            
            # Testing if the type of an if condition is none (line 180)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 180, 12), result_gt_5155):
                
                # Call to append(...): (line 184)
                # Processing the call arguments (line 184)
                
                # Obtaining an instance of the builtin type 'tuple' (line 184)
                tuple_5168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 48), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 184)
                # Adding element type (line 184)
                # Getting the type of 'arg' (line 184)
                arg_5169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 48), 'arg', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 48), tuple_5168, arg_5169)
                # Adding element type (line 184)
                
                # Obtaining an instance of the builtin type 'dict' (line 184)
                dict_5170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 53), 'dict')
                # Adding type elements to the builtin type 'dict' instance (line 184)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 48), tuple_5168, dict_5170)
                
                # Processing the call keyword arguments (line 184)
                kwargs_5171 = {}
                # Getting the type of 'result_arg_kwarg_tuples' (line 184)
                result_arg_kwarg_tuples_5166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'result_arg_kwarg_tuples', False)
                # Obtaining the member 'append' of a type (line 184)
                append_5167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 16), result_arg_kwarg_tuples_5166, 'append')
                # Calling append(args, kwargs) (line 184)
                append_call_result_5172 = invoke(stypy.reporting.localization.Localization(__file__, 184, 16), append_5167, *[tuple_5168], **kwargs_5171)
                
            else:
                
                # Testing the type of an if condition (line 180)
                if_condition_5156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 12), result_gt_5155)
                # Assigning a type to the variable 'if_condition_5156' (line 180)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'if_condition_5156', if_condition_5156)
                # SSA begins for if statement (line 180)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'unfolded_keyword_arguments' (line 181)
                unfolded_keyword_arguments_5157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 29), 'unfolded_keyword_arguments')
                # Assigning a type to the variable 'unfolded_keyword_arguments_5157' (line 181)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'unfolded_keyword_arguments_5157', unfolded_keyword_arguments_5157)
                # Testing if the for loop is going to be iterated (line 181)
                # Testing the type of a for loop iterable (line 181)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 181, 16), unfolded_keyword_arguments_5157)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 181, 16), unfolded_keyword_arguments_5157):
                    # Getting the type of the for loop variable (line 181)
                    for_loop_var_5158 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 181, 16), unfolded_keyword_arguments_5157)
                    # Assigning a type to the variable 'kwarg' (line 181)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'kwarg', for_loop_var_5158)
                    # SSA begins for a for statement (line 181)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to append(...): (line 182)
                    # Processing the call arguments (line 182)
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 182)
                    tuple_5161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 52), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 182)
                    # Adding element type (line 182)
                    # Getting the type of 'arg' (line 182)
                    arg_5162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 52), 'arg', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 52), tuple_5161, arg_5162)
                    # Adding element type (line 182)
                    # Getting the type of 'kwarg' (line 182)
                    kwarg_5163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 57), 'kwarg', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 52), tuple_5161, kwarg_5163)
                    
                    # Processing the call keyword arguments (line 182)
                    kwargs_5164 = {}
                    # Getting the type of 'result_arg_kwarg_tuples' (line 182)
                    result_arg_kwarg_tuples_5159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 20), 'result_arg_kwarg_tuples', False)
                    # Obtaining the member 'append' of a type (line 182)
                    append_5160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 20), result_arg_kwarg_tuples_5159, 'append')
                    # Calling append(args, kwargs) (line 182)
                    append_call_result_5165 = invoke(stypy.reporting.localization.Localization(__file__, 182, 20), append_5160, *[tuple_5161], **kwargs_5164)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA branch for the else part of an if statement (line 180)
                module_type_store.open_ssa_branch('else')
                
                # Call to append(...): (line 184)
                # Processing the call arguments (line 184)
                
                # Obtaining an instance of the builtin type 'tuple' (line 184)
                tuple_5168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 48), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 184)
                # Adding element type (line 184)
                # Getting the type of 'arg' (line 184)
                arg_5169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 48), 'arg', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 48), tuple_5168, arg_5169)
                # Adding element type (line 184)
                
                # Obtaining an instance of the builtin type 'dict' (line 184)
                dict_5170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 53), 'dict')
                # Adding type elements to the builtin type 'dict' instance (line 184)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 48), tuple_5168, dict_5170)
                
                # Processing the call keyword arguments (line 184)
                kwargs_5171 = {}
                # Getting the type of 'result_arg_kwarg_tuples' (line 184)
                result_arg_kwarg_tuples_5166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'result_arg_kwarg_tuples', False)
                # Obtaining the member 'append' of a type (line 184)
                append_5167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 16), result_arg_kwarg_tuples_5166, 'append')
                # Calling append(args, kwargs) (line 184)
                append_call_result_5172 = invoke(stypy.reporting.localization.Localization(__file__, 184, 16), append_5167, *[tuple_5168], **kwargs_5171)
                
                # SSA join for if statement (line 180)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
    else:
        
        # Testing the type of an if condition (line 169)
        if_condition_5124 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 4), result_eq_5123)
        # Assigning a type to the variable 'if_condition_5124' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'if_condition_5124', if_condition_5124)
        # SSA begins for if statement (line 169)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to len(...): (line 170)
        # Processing the call arguments (line 170)
        # Getting the type of 'unfolded_keyword_arguments' (line 170)
        unfolded_keyword_arguments_5126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 15), 'unfolded_keyword_arguments', False)
        # Processing the call keyword arguments (line 170)
        kwargs_5127 = {}
        # Getting the type of 'len' (line 170)
        len_5125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'len', False)
        # Calling len(args, kwargs) (line 170)
        len_call_result_5128 = invoke(stypy.reporting.localization.Localization(__file__, 170, 11), len_5125, *[unfolded_keyword_arguments_5126], **kwargs_5127)
        
        int_5129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 45), 'int')
        # Applying the binary operator '>' (line 170)
        result_gt_5130 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 11), '>', len_call_result_5128, int_5129)
        
        # Testing if the type of an if condition is none (line 170)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 170, 8), result_gt_5130):
            
            # Call to append(...): (line 175)
            # Processing the call arguments (line 175)
            
            # Obtaining an instance of the builtin type 'tuple' (line 175)
            tuple_5143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 44), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 175)
            # Adding element type (line 175)
            
            # Obtaining an instance of the builtin type 'list' (line 175)
            list_5144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 44), 'list')
            # Adding type elements to the builtin type 'list' instance (line 175)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 44), tuple_5143, list_5144)
            # Adding element type (line 175)
            
            # Obtaining an instance of the builtin type 'dict' (line 175)
            dict_5145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 48), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 175)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 44), tuple_5143, dict_5145)
            
            # Processing the call keyword arguments (line 175)
            kwargs_5146 = {}
            # Getting the type of 'result_arg_kwarg_tuples' (line 175)
            result_arg_kwarg_tuples_5141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'result_arg_kwarg_tuples', False)
            # Obtaining the member 'append' of a type (line 175)
            append_5142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), result_arg_kwarg_tuples_5141, 'append')
            # Calling append(args, kwargs) (line 175)
            append_call_result_5147 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), append_5142, *[tuple_5143], **kwargs_5146)
            
        else:
            
            # Testing the type of an if condition (line 170)
            if_condition_5131 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 8), result_gt_5130)
            # Assigning a type to the variable 'if_condition_5131' (line 170)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'if_condition_5131', if_condition_5131)
            # SSA begins for if statement (line 170)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'unfolded_keyword_arguments' (line 171)
            unfolded_keyword_arguments_5132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 25), 'unfolded_keyword_arguments')
            # Assigning a type to the variable 'unfolded_keyword_arguments_5132' (line 171)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'unfolded_keyword_arguments_5132', unfolded_keyword_arguments_5132)
            # Testing if the for loop is going to be iterated (line 171)
            # Testing the type of a for loop iterable (line 171)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 171, 12), unfolded_keyword_arguments_5132)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 171, 12), unfolded_keyword_arguments_5132):
                # Getting the type of the for loop variable (line 171)
                for_loop_var_5133 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 171, 12), unfolded_keyword_arguments_5132)
                # Assigning a type to the variable 'kwarg' (line 171)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'kwarg', for_loop_var_5133)
                # SSA begins for a for statement (line 171)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to append(...): (line 172)
                # Processing the call arguments (line 172)
                
                # Obtaining an instance of the builtin type 'tuple' (line 172)
                tuple_5136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 48), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 172)
                # Adding element type (line 172)
                
                # Obtaining an instance of the builtin type 'list' (line 172)
                list_5137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 48), 'list')
                # Adding type elements to the builtin type 'list' instance (line 172)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 48), tuple_5136, list_5137)
                # Adding element type (line 172)
                # Getting the type of 'kwarg' (line 172)
                kwarg_5138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 52), 'kwarg', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 48), tuple_5136, kwarg_5138)
                
                # Processing the call keyword arguments (line 172)
                kwargs_5139 = {}
                # Getting the type of 'result_arg_kwarg_tuples' (line 172)
                result_arg_kwarg_tuples_5134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'result_arg_kwarg_tuples', False)
                # Obtaining the member 'append' of a type (line 172)
                append_5135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 16), result_arg_kwarg_tuples_5134, 'append')
                # Calling append(args, kwargs) (line 172)
                append_call_result_5140 = invoke(stypy.reporting.localization.Localization(__file__, 172, 16), append_5135, *[tuple_5136], **kwargs_5139)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA branch for the else part of an if statement (line 170)
            module_type_store.open_ssa_branch('else')
            
            # Call to append(...): (line 175)
            # Processing the call arguments (line 175)
            
            # Obtaining an instance of the builtin type 'tuple' (line 175)
            tuple_5143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 44), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 175)
            # Adding element type (line 175)
            
            # Obtaining an instance of the builtin type 'list' (line 175)
            list_5144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 44), 'list')
            # Adding type elements to the builtin type 'list' instance (line 175)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 44), tuple_5143, list_5144)
            # Adding element type (line 175)
            
            # Obtaining an instance of the builtin type 'dict' (line 175)
            dict_5145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 48), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 175)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 44), tuple_5143, dict_5145)
            
            # Processing the call keyword arguments (line 175)
            kwargs_5146 = {}
            # Getting the type of 'result_arg_kwarg_tuples' (line 175)
            result_arg_kwarg_tuples_5141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'result_arg_kwarg_tuples', False)
            # Obtaining the member 'append' of a type (line 175)
            append_5142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), result_arg_kwarg_tuples_5141, 'append')
            # Calling append(args, kwargs) (line 175)
            append_call_result_5147 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), append_5142, *[tuple_5143], **kwargs_5146)
            
            # SSA join for if statement (line 170)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA branch for the else part of an if statement (line 169)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'unfolded_arguments' (line 179)
        unfolded_arguments_5148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 19), 'unfolded_arguments')
        # Assigning a type to the variable 'unfolded_arguments_5148' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'unfolded_arguments_5148', unfolded_arguments_5148)
        # Testing if the for loop is going to be iterated (line 179)
        # Testing the type of a for loop iterable (line 179)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 179, 8), unfolded_arguments_5148)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 179, 8), unfolded_arguments_5148):
            # Getting the type of the for loop variable (line 179)
            for_loop_var_5149 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 179, 8), unfolded_arguments_5148)
            # Assigning a type to the variable 'arg' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'arg', for_loop_var_5149)
            # SSA begins for a for statement (line 179)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to len(...): (line 180)
            # Processing the call arguments (line 180)
            # Getting the type of 'unfolded_keyword_arguments' (line 180)
            unfolded_keyword_arguments_5151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'unfolded_keyword_arguments', False)
            # Processing the call keyword arguments (line 180)
            kwargs_5152 = {}
            # Getting the type of 'len' (line 180)
            len_5150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 15), 'len', False)
            # Calling len(args, kwargs) (line 180)
            len_call_result_5153 = invoke(stypy.reporting.localization.Localization(__file__, 180, 15), len_5150, *[unfolded_keyword_arguments_5151], **kwargs_5152)
            
            int_5154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 49), 'int')
            # Applying the binary operator '>' (line 180)
            result_gt_5155 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 15), '>', len_call_result_5153, int_5154)
            
            # Testing if the type of an if condition is none (line 180)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 180, 12), result_gt_5155):
                
                # Call to append(...): (line 184)
                # Processing the call arguments (line 184)
                
                # Obtaining an instance of the builtin type 'tuple' (line 184)
                tuple_5168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 48), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 184)
                # Adding element type (line 184)
                # Getting the type of 'arg' (line 184)
                arg_5169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 48), 'arg', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 48), tuple_5168, arg_5169)
                # Adding element type (line 184)
                
                # Obtaining an instance of the builtin type 'dict' (line 184)
                dict_5170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 53), 'dict')
                # Adding type elements to the builtin type 'dict' instance (line 184)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 48), tuple_5168, dict_5170)
                
                # Processing the call keyword arguments (line 184)
                kwargs_5171 = {}
                # Getting the type of 'result_arg_kwarg_tuples' (line 184)
                result_arg_kwarg_tuples_5166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'result_arg_kwarg_tuples', False)
                # Obtaining the member 'append' of a type (line 184)
                append_5167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 16), result_arg_kwarg_tuples_5166, 'append')
                # Calling append(args, kwargs) (line 184)
                append_call_result_5172 = invoke(stypy.reporting.localization.Localization(__file__, 184, 16), append_5167, *[tuple_5168], **kwargs_5171)
                
            else:
                
                # Testing the type of an if condition (line 180)
                if_condition_5156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 12), result_gt_5155)
                # Assigning a type to the variable 'if_condition_5156' (line 180)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'if_condition_5156', if_condition_5156)
                # SSA begins for if statement (line 180)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'unfolded_keyword_arguments' (line 181)
                unfolded_keyword_arguments_5157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 29), 'unfolded_keyword_arguments')
                # Assigning a type to the variable 'unfolded_keyword_arguments_5157' (line 181)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'unfolded_keyword_arguments_5157', unfolded_keyword_arguments_5157)
                # Testing if the for loop is going to be iterated (line 181)
                # Testing the type of a for loop iterable (line 181)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 181, 16), unfolded_keyword_arguments_5157)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 181, 16), unfolded_keyword_arguments_5157):
                    # Getting the type of the for loop variable (line 181)
                    for_loop_var_5158 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 181, 16), unfolded_keyword_arguments_5157)
                    # Assigning a type to the variable 'kwarg' (line 181)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'kwarg', for_loop_var_5158)
                    # SSA begins for a for statement (line 181)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to append(...): (line 182)
                    # Processing the call arguments (line 182)
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 182)
                    tuple_5161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 52), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 182)
                    # Adding element type (line 182)
                    # Getting the type of 'arg' (line 182)
                    arg_5162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 52), 'arg', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 52), tuple_5161, arg_5162)
                    # Adding element type (line 182)
                    # Getting the type of 'kwarg' (line 182)
                    kwarg_5163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 57), 'kwarg', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 52), tuple_5161, kwarg_5163)
                    
                    # Processing the call keyword arguments (line 182)
                    kwargs_5164 = {}
                    # Getting the type of 'result_arg_kwarg_tuples' (line 182)
                    result_arg_kwarg_tuples_5159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 20), 'result_arg_kwarg_tuples', False)
                    # Obtaining the member 'append' of a type (line 182)
                    append_5160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 20), result_arg_kwarg_tuples_5159, 'append')
                    # Calling append(args, kwargs) (line 182)
                    append_call_result_5165 = invoke(stypy.reporting.localization.Localization(__file__, 182, 20), append_5160, *[tuple_5161], **kwargs_5164)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA branch for the else part of an if statement (line 180)
                module_type_store.open_ssa_branch('else')
                
                # Call to append(...): (line 184)
                # Processing the call arguments (line 184)
                
                # Obtaining an instance of the builtin type 'tuple' (line 184)
                tuple_5168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 48), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 184)
                # Adding element type (line 184)
                # Getting the type of 'arg' (line 184)
                arg_5169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 48), 'arg', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 48), tuple_5168, arg_5169)
                # Adding element type (line 184)
                
                # Obtaining an instance of the builtin type 'dict' (line 184)
                dict_5170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 53), 'dict')
                # Adding type elements to the builtin type 'dict' instance (line 184)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 48), tuple_5168, dict_5170)
                
                # Processing the call keyword arguments (line 184)
                kwargs_5171 = {}
                # Getting the type of 'result_arg_kwarg_tuples' (line 184)
                result_arg_kwarg_tuples_5166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'result_arg_kwarg_tuples', False)
                # Obtaining the member 'append' of a type (line 184)
                append_5167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 16), result_arg_kwarg_tuples_5166, 'append')
                # Calling append(args, kwargs) (line 184)
                append_call_result_5172 = invoke(stypy.reporting.localization.Localization(__file__, 184, 16), append_5167, *[tuple_5168], **kwargs_5171)
                
                # SSA join for if statement (line 180)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 169)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'result_arg_kwarg_tuples' (line 186)
    result_arg_kwarg_tuples_5173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 11), 'result_arg_kwarg_tuples')
    # Assigning a type to the variable 'stypy_return_type' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type', result_arg_kwarg_tuples_5173)
    
    # ################# End of 'unfold_arguments(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unfold_arguments' in the type store
    # Getting the type of 'stypy_return_type' (line 146)
    stypy_return_type_5174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5174)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unfold_arguments'
    return stypy_return_type_5174

# Assigning a type to the variable 'unfold_arguments' (line 146)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'unfold_arguments', unfold_arguments)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
