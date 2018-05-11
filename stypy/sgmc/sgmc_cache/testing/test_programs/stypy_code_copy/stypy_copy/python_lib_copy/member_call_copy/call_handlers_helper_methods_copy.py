
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from ...errors_copy.type_error_copy import TypeError
2: from ...errors_copy.type_warning_copy import TypeWarning
3: from ...python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType
4: from ...python_lib_copy.python_types_copy.type_inference_copy import union_type_copy
5: from ...reporting_copy import print_utils_copy, module_line_numbering_copy
6: 
7: '''
8: Several functions that help call handler management in various ways. Moved here to limit the size of Python files.
9: '''
10: 
11: 
12: def exist_a_type_error_within_parameters(*arg_types, **kwargs_types):
13:     '''
14:     Is there at least a type error among the call parameters?
15:     :param arg_types: Call arguments
16:     :param kwargs_types: Call keyword arguments
17:     :return: bool
18:     '''
19:     t_e_args = filter(lambda elem: isinstance(elem, TypeError), arg_types)
20:     if len(t_e_args) > 0:
21:         return True
22: 
23:     t_e_kwargs = filter(lambda elem: isinstance(elem, TypeError), kwargs_types.values())
24:     if len(t_e_kwargs) > 0:
25:         return True
26: 
27:     return False
28: 
29: 
30: def strip_undefined_type_from_union_type(union):
31:     '''
32:     Remove undefined types from a union type
33:     :param union:
34:     :return:
35:     '''
36:     ret_union = None
37: 
38:     for type_ in union.types:
39:         if not isinstance(type_, UndefinedType):
40:             ret_union = union_type_copy.UnionType.add(ret_union, type_)
41: 
42:     return ret_union
43: 
44: 
45: def check_undefined_type_within_parameters(localization, call_description, *arg_types, **kwargs_types):
46:     '''
47:     When calling a callable element, the type of some parameters might be undefined (not initialized
48:     to any value in the preceding code). This function check this fact and substitute the Undefined
49:     parameters by suitable type errors. It also creates warnings if the undefined type is inside a
50:     union type, removing the undefined type from the union afterwards. It does the same with keyword arguments.
51: 
52:     :param localization: Caller information
53:     :param call_description: A textual description of the call (to generate errors)
54:     :param arg_types: Call arguments
55:     :param kwargs_types: Call keyword arguments
56:     :return: arguments, keyword arguments tuple with the undefined types removed or substituted by TypeErrors depending
57:     on if they are into union types or not
58:     '''
59:     arg_types_list = list(arg_types)
60: 
61:     # Process arguments
62:     for i in range(len(arg_types_list)):
63:         if isinstance(arg_types_list[i], union_type_copy.UnionType):
64:             # Is an undefined type inside this union type?
65:             exist_undefined = len(filter(lambda elem: isinstance(elem, UndefinedType), arg_types[i].types)) > 0
66:             if exist_undefined:
67:                 # Compose a type warning with the full description of the problem.
68:                 offset = print_utils_copy.get_param_position(
69:                     module_line_numbering_copy.ModuleLineNumbering.get_line_from_module_code(
70:                         localization.file_name, localization.line), i)
71:                 if offset is not -1:  # Sometimes offsets of the offending parameters cannot be obtained
72:                     clone_loc = localization.clone()
73:                     clone_loc.column = offset
74:                 else:
75:                     clone_loc = localization
76:                 TypeWarning.instance(clone_loc, "{0}: Argument {1} could be undefined".format(call_description,
77:                                                                                               i + 1))
78:             # Remove undefined type from the union type
79:             arg_types_list[i] = strip_undefined_type_from_union_type(arg_types[i])
80:             continue
81:         else:
82:             # Undefined types outside union types are treated as Type errors.
83:             if isinstance(arg_types[i], UndefinedType):
84:                 offset = print_utils_copy.get_param_position(
85:                     module_line_numbering_copy.ModuleLineNumbering.get_line_from_module_code(
86:                         localization.file_name, localization.line), i)
87:                 if offset is not -1:  # Sometimes offsets of the offending parameters cannot be obtained
88:                     clone_loc = localization.clone()
89:                     clone_loc.column = offset
90:                 else:
91:                     clone_loc = localization
92: 
93:                 arg_types_list[i] = TypeError(clone_loc, "{0}: Argument {1} is not defined".format(call_description,
94:                                                                                                    i + 1))
95:                 continue
96: 
97:         arg_types_list[i] = arg_types[i]
98: 
99:     # Process keyword arguments (the same processing as argument lists)
100:     final_kwargs = {}
101:     for key, value in kwargs_types.items():
102:         if isinstance(value, union_type_copy.UnionType):
103:             exist_undefined = filter(lambda elem: isinstance(elem, UndefinedType), value.types)
104:             if exist_undefined:
105:                 TypeWarning.instance(localization,
106:                                      "{0}: Keyword argument {1} could be undefined".format(call_description,
107:                                                                                            key))
108:             final_kwargs[key] = strip_undefined_type_from_union_type(value)
109:             continue
110:         else:
111:             if isinstance(value, UndefinedType):
112:                 final_kwargs[key] = TypeError(localization,
113:                                               "{0}: Keyword argument {1} is not defined".format(call_description,
114:                                                                                                 key))
115:                 continue
116:         final_kwargs[key] = value
117: 
118:     return tuple(arg_types_list), final_kwargs
119: 
120: 
121: # ########################################## PRETTY-PRINTING FUNCTIONS ##########################################
122: 
123: 
124: def __type_error_str(arg):
125:     '''
126:     Helper function of the following one.
127:     If arg is a type error, this avoids printing all the TypeError information and only prints the name. This is
128:     convenient when pretty-printing calls and its passed parameters to report errors, because if we print the full
129:     error information (the same one that is returned by stypy at the end) the message will be unclear.
130:     :param arg:
131:     :return:
132:     '''
133:     if isinstance(arg, TypeError):
134:         return "TypeError"
135:     else:
136:         return str(arg)
137: 
138: 
139: def __format_type_list(*arg_types, **kwargs_types):
140:     '''
141:     Pretty-print passed parameter list
142:     :param arg_types:
143:     :param kwargs_types:
144:     :return:
145:     '''
146:     arg_str_list = map(lambda elem: __type_error_str(elem), arg_types[0])
147:     arg_str = ""
148:     for arg in arg_str_list:
149:         arg_str += arg + ", "
150: 
151:     if len(arg_str) > 0:
152:         arg_str = arg_str[:-2]
153: 
154:     kwarg_str_list = map(lambda elem: __type_error_str(elem), kwargs_types)
155:     kwarg_str = ""
156:     for arg in kwarg_str_list:
157:         kwarg_str += arg + ", "
158: 
159:     if len(kwarg_str) > 0:
160:         kwarg_str = kwarg_str[:-1]
161:         kwarg_str = '{' + kwarg_str + '}'
162: 
163:     return arg_str, kwarg_str
164: 
165: 
166: def __format_callable(callable_):
167:     '''
168:     Pretty-print a callable entity
169:     :param callable_:
170:     :return:
171:     '''
172:     if hasattr(callable_, "__name__"):
173:         return callable_.__name__
174:     else:
175:         return str(callable_)
176: 
177: 
178: def format_call(callable_, arg_types, kwarg_types):
179:     '''
180:     Pretty-print calls and its passed parameters, for error reporting, using the previously defined functions
181:     :param callable_:
182:     :param arg_types:
183:     :param kwarg_types:
184:     :return:
185:     '''
186:     arg_str, kwarg_str = __format_type_list(arg_types, kwarg_types.values())
187:     callable_str = __format_callable(callable_)
188:     if len(kwarg_str) == 0:
189:         return "\t" + callable_str + "(" + arg_str + ")"
190:     else:
191:         return "\t" + callable_str + "(" + arg_str + ", " + kwarg_str + ")"
192: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_5515 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy')

if (type(import_5515) is not StypyTypeError):

    if (import_5515 != 'pyd_module'):
        __import__(import_5515)
        sys_modules_5516 = sys.modules[import_5515]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', sys_modules_5516.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_5516, sys_modules_5516.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', import_5515)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy import TypeWarning' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_5517 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy')

if (type(import_5517) is not StypyTypeError):

    if (import_5517 != 'pyd_module'):
        __import__(import_5517)
        sys_modules_5518 = sys.modules[import_5517]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', sys_modules_5518.module_type_store, module_type_store, ['TypeWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_5518, sys_modules_5518.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy import TypeWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', None, module_type_store, ['TypeWarning'], [TypeWarning])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', import_5517)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_5519 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy')

if (type(import_5519) is not StypyTypeError):

    if (import_5519 != 'pyd_module'):
        __import__(import_5519)
        sys_modules_5520 = sys.modules[import_5519]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', sys_modules_5520.module_type_store, module_type_store, ['UndefinedType'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_5520, sys_modules_5520.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', None, module_type_store, ['UndefinedType'], [UndefinedType])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', import_5519)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_5521 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_5521) is not StypyTypeError):

    if (import_5521 != 'pyd_module'):
        __import__(import_5521)
        sys_modules_5522 = sys.modules[import_5521]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_5522.module_type_store, module_type_store, ['union_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_5522, sys_modules_5522.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['union_type_copy'], [union_type_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_5521)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy import print_utils_copy, module_line_numbering_copy' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_5523 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy')

if (type(import_5523) is not StypyTypeError):

    if (import_5523 != 'pyd_module'):
        __import__(import_5523)
        sys_modules_5524 = sys.modules[import_5523]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy', sys_modules_5524.module_type_store, module_type_store, ['print_utils_copy', 'module_line_numbering_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_5524, sys_modules_5524.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy import print_utils_copy, module_line_numbering_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy', None, module_type_store, ['print_utils_copy', 'module_line_numbering_copy'], [print_utils_copy, module_line_numbering_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy', import_5523)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

str_5525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, (-1)), 'str', '\nSeveral functions that help call handler management in various ways. Moved here to limit the size of Python files.\n')

@norecursion
def exist_a_type_error_within_parameters(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'exist_a_type_error_within_parameters'
    module_type_store = module_type_store.open_function_context('exist_a_type_error_within_parameters', 12, 0, False)
    
    # Passed parameters checking function
    exist_a_type_error_within_parameters.stypy_localization = localization
    exist_a_type_error_within_parameters.stypy_type_of_self = None
    exist_a_type_error_within_parameters.stypy_type_store = module_type_store
    exist_a_type_error_within_parameters.stypy_function_name = 'exist_a_type_error_within_parameters'
    exist_a_type_error_within_parameters.stypy_param_names_list = []
    exist_a_type_error_within_parameters.stypy_varargs_param_name = 'arg_types'
    exist_a_type_error_within_parameters.stypy_kwargs_param_name = 'kwargs_types'
    exist_a_type_error_within_parameters.stypy_call_defaults = defaults
    exist_a_type_error_within_parameters.stypy_call_varargs = varargs
    exist_a_type_error_within_parameters.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'exist_a_type_error_within_parameters', [], 'arg_types', 'kwargs_types', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'exist_a_type_error_within_parameters', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'exist_a_type_error_within_parameters(...)' code ##################

    str_5526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, (-1)), 'str', '\n    Is there at least a type error among the call parameters?\n    :param arg_types: Call arguments\n    :param kwargs_types: Call keyword arguments\n    :return: bool\n    ')
    
    # Assigning a Call to a Name (line 19):
    
    # Assigning a Call to a Name (line 19):
    
    # Call to filter(...): (line 19)
    # Processing the call arguments (line 19)

    @norecursion
    def _stypy_temp_lambda_7(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_7'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_7', 19, 22, True)
        # Passed parameters checking function
        _stypy_temp_lambda_7.stypy_localization = localization
        _stypy_temp_lambda_7.stypy_type_of_self = None
        _stypy_temp_lambda_7.stypy_type_store = module_type_store
        _stypy_temp_lambda_7.stypy_function_name = '_stypy_temp_lambda_7'
        _stypy_temp_lambda_7.stypy_param_names_list = ['elem']
        _stypy_temp_lambda_7.stypy_varargs_param_name = None
        _stypy_temp_lambda_7.stypy_kwargs_param_name = None
        _stypy_temp_lambda_7.stypy_call_defaults = defaults
        _stypy_temp_lambda_7.stypy_call_varargs = varargs
        _stypy_temp_lambda_7.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_7', ['elem'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_7', ['elem'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to isinstance(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'elem' (line 19)
        elem_5529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 46), 'elem', False)
        # Getting the type of 'TypeError' (line 19)
        TypeError_5530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 52), 'TypeError', False)
        # Processing the call keyword arguments (line 19)
        kwargs_5531 = {}
        # Getting the type of 'isinstance' (line 19)
        isinstance_5528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 35), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 19)
        isinstance_call_result_5532 = invoke(stypy.reporting.localization.Localization(__file__, 19, 35), isinstance_5528, *[elem_5529, TypeError_5530], **kwargs_5531)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'stypy_return_type', isinstance_call_result_5532)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_7' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_5533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5533)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_7'
        return stypy_return_type_5533

    # Assigning a type to the variable '_stypy_temp_lambda_7' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), '_stypy_temp_lambda_7', _stypy_temp_lambda_7)
    # Getting the type of '_stypy_temp_lambda_7' (line 19)
    _stypy_temp_lambda_7_5534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), '_stypy_temp_lambda_7')
    # Getting the type of 'arg_types' (line 19)
    arg_types_5535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 64), 'arg_types', False)
    # Processing the call keyword arguments (line 19)
    kwargs_5536 = {}
    # Getting the type of 'filter' (line 19)
    filter_5527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'filter', False)
    # Calling filter(args, kwargs) (line 19)
    filter_call_result_5537 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), filter_5527, *[_stypy_temp_lambda_7_5534, arg_types_5535], **kwargs_5536)
    
    # Assigning a type to the variable 't_e_args' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 't_e_args', filter_call_result_5537)
    
    
    # Call to len(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 't_e_args' (line 20)
    t_e_args_5539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 't_e_args', False)
    # Processing the call keyword arguments (line 20)
    kwargs_5540 = {}
    # Getting the type of 'len' (line 20)
    len_5538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 7), 'len', False)
    # Calling len(args, kwargs) (line 20)
    len_call_result_5541 = invoke(stypy.reporting.localization.Localization(__file__, 20, 7), len_5538, *[t_e_args_5539], **kwargs_5540)
    
    int_5542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'int')
    # Applying the binary operator '>' (line 20)
    result_gt_5543 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 7), '>', len_call_result_5541, int_5542)
    
    # Testing if the type of an if condition is none (line 20)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 20, 4), result_gt_5543):
        pass
    else:
        
        # Testing the type of an if condition (line 20)
        if_condition_5544 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 4), result_gt_5543)
        # Assigning a type to the variable 'if_condition_5544' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'if_condition_5544', if_condition_5544)
        # SSA begins for if statement (line 20)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 21)
        True_5545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type', True_5545)
        # SSA join for if statement (line 20)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 23):
    
    # Assigning a Call to a Name (line 23):
    
    # Call to filter(...): (line 23)
    # Processing the call arguments (line 23)

    @norecursion
    def _stypy_temp_lambda_8(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_8'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_8', 23, 24, True)
        # Passed parameters checking function
        _stypy_temp_lambda_8.stypy_localization = localization
        _stypy_temp_lambda_8.stypy_type_of_self = None
        _stypy_temp_lambda_8.stypy_type_store = module_type_store
        _stypy_temp_lambda_8.stypy_function_name = '_stypy_temp_lambda_8'
        _stypy_temp_lambda_8.stypy_param_names_list = ['elem']
        _stypy_temp_lambda_8.stypy_varargs_param_name = None
        _stypy_temp_lambda_8.stypy_kwargs_param_name = None
        _stypy_temp_lambda_8.stypy_call_defaults = defaults
        _stypy_temp_lambda_8.stypy_call_varargs = varargs
        _stypy_temp_lambda_8.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_8', ['elem'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_8', ['elem'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to isinstance(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'elem' (line 23)
        elem_5548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 48), 'elem', False)
        # Getting the type of 'TypeError' (line 23)
        TypeError_5549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 54), 'TypeError', False)
        # Processing the call keyword arguments (line 23)
        kwargs_5550 = {}
        # Getting the type of 'isinstance' (line 23)
        isinstance_5547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 37), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 23)
        isinstance_call_result_5551 = invoke(stypy.reporting.localization.Localization(__file__, 23, 37), isinstance_5547, *[elem_5548, TypeError_5549], **kwargs_5550)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'stypy_return_type', isinstance_call_result_5551)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_8' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_5552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5552)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_8'
        return stypy_return_type_5552

    # Assigning a type to the variable '_stypy_temp_lambda_8' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), '_stypy_temp_lambda_8', _stypy_temp_lambda_8)
    # Getting the type of '_stypy_temp_lambda_8' (line 23)
    _stypy_temp_lambda_8_5553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), '_stypy_temp_lambda_8')
    
    # Call to values(...): (line 23)
    # Processing the call keyword arguments (line 23)
    kwargs_5556 = {}
    # Getting the type of 'kwargs_types' (line 23)
    kwargs_types_5554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 66), 'kwargs_types', False)
    # Obtaining the member 'values' of a type (line 23)
    values_5555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 66), kwargs_types_5554, 'values')
    # Calling values(args, kwargs) (line 23)
    values_call_result_5557 = invoke(stypy.reporting.localization.Localization(__file__, 23, 66), values_5555, *[], **kwargs_5556)
    
    # Processing the call keyword arguments (line 23)
    kwargs_5558 = {}
    # Getting the type of 'filter' (line 23)
    filter_5546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 17), 'filter', False)
    # Calling filter(args, kwargs) (line 23)
    filter_call_result_5559 = invoke(stypy.reporting.localization.Localization(__file__, 23, 17), filter_5546, *[_stypy_temp_lambda_8_5553, values_call_result_5557], **kwargs_5558)
    
    # Assigning a type to the variable 't_e_kwargs' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 't_e_kwargs', filter_call_result_5559)
    
    
    # Call to len(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 't_e_kwargs' (line 24)
    t_e_kwargs_5561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 't_e_kwargs', False)
    # Processing the call keyword arguments (line 24)
    kwargs_5562 = {}
    # Getting the type of 'len' (line 24)
    len_5560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 7), 'len', False)
    # Calling len(args, kwargs) (line 24)
    len_call_result_5563 = invoke(stypy.reporting.localization.Localization(__file__, 24, 7), len_5560, *[t_e_kwargs_5561], **kwargs_5562)
    
    int_5564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'int')
    # Applying the binary operator '>' (line 24)
    result_gt_5565 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 7), '>', len_call_result_5563, int_5564)
    
    # Testing if the type of an if condition is none (line 24)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 24, 4), result_gt_5565):
        pass
    else:
        
        # Testing the type of an if condition (line 24)
        if_condition_5566 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 4), result_gt_5565)
        # Assigning a type to the variable 'if_condition_5566' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'if_condition_5566', if_condition_5566)
        # SSA begins for if statement (line 24)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 25)
        True_5567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', True_5567)
        # SSA join for if statement (line 24)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'False' (line 27)
    False_5568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type', False_5568)
    
    # ################# End of 'exist_a_type_error_within_parameters(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'exist_a_type_error_within_parameters' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_5569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5569)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'exist_a_type_error_within_parameters'
    return stypy_return_type_5569

# Assigning a type to the variable 'exist_a_type_error_within_parameters' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'exist_a_type_error_within_parameters', exist_a_type_error_within_parameters)

@norecursion
def strip_undefined_type_from_union_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'strip_undefined_type_from_union_type'
    module_type_store = module_type_store.open_function_context('strip_undefined_type_from_union_type', 30, 0, False)
    
    # Passed parameters checking function
    strip_undefined_type_from_union_type.stypy_localization = localization
    strip_undefined_type_from_union_type.stypy_type_of_self = None
    strip_undefined_type_from_union_type.stypy_type_store = module_type_store
    strip_undefined_type_from_union_type.stypy_function_name = 'strip_undefined_type_from_union_type'
    strip_undefined_type_from_union_type.stypy_param_names_list = ['union']
    strip_undefined_type_from_union_type.stypy_varargs_param_name = None
    strip_undefined_type_from_union_type.stypy_kwargs_param_name = None
    strip_undefined_type_from_union_type.stypy_call_defaults = defaults
    strip_undefined_type_from_union_type.stypy_call_varargs = varargs
    strip_undefined_type_from_union_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'strip_undefined_type_from_union_type', ['union'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'strip_undefined_type_from_union_type', localization, ['union'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'strip_undefined_type_from_union_type(...)' code ##################

    str_5570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'str', '\n    Remove undefined types from a union type\n    :param union:\n    :return:\n    ')
    
    # Assigning a Name to a Name (line 36):
    
    # Assigning a Name to a Name (line 36):
    # Getting the type of 'None' (line 36)
    None_5571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'None')
    # Assigning a type to the variable 'ret_union' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'ret_union', None_5571)
    
    # Getting the type of 'union' (line 38)
    union_5572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'union')
    # Obtaining the member 'types' of a type (line 38)
    types_5573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 17), union_5572, 'types')
    # Assigning a type to the variable 'types_5573' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'types_5573', types_5573)
    # Testing if the for loop is going to be iterated (line 38)
    # Testing the type of a for loop iterable (line 38)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 4), types_5573)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 38, 4), types_5573):
        # Getting the type of the for loop variable (line 38)
        for_loop_var_5574 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 4), types_5573)
        # Assigning a type to the variable 'type_' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'type_', for_loop_var_5574)
        # SSA begins for a for statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to isinstance(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'type_' (line 39)
        type__5576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 26), 'type_', False)
        # Getting the type of 'UndefinedType' (line 39)
        UndefinedType_5577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 33), 'UndefinedType', False)
        # Processing the call keyword arguments (line 39)
        kwargs_5578 = {}
        # Getting the type of 'isinstance' (line 39)
        isinstance_5575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 39)
        isinstance_call_result_5579 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), isinstance_5575, *[type__5576, UndefinedType_5577], **kwargs_5578)
        
        # Applying the 'not' unary operator (line 39)
        result_not__5580 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 11), 'not', isinstance_call_result_5579)
        
        # Testing if the type of an if condition is none (line 39)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 39, 8), result_not__5580):
            pass
        else:
            
            # Testing the type of an if condition (line 39)
            if_condition_5581 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 8), result_not__5580)
            # Assigning a type to the variable 'if_condition_5581' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'if_condition_5581', if_condition_5581)
            # SSA begins for if statement (line 39)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 40):
            
            # Assigning a Call to a Name (line 40):
            
            # Call to add(...): (line 40)
            # Processing the call arguments (line 40)
            # Getting the type of 'ret_union' (line 40)
            ret_union_5585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 54), 'ret_union', False)
            # Getting the type of 'type_' (line 40)
            type__5586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 65), 'type_', False)
            # Processing the call keyword arguments (line 40)
            kwargs_5587 = {}
            # Getting the type of 'union_type_copy' (line 40)
            union_type_copy_5582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'union_type_copy', False)
            # Obtaining the member 'UnionType' of a type (line 40)
            UnionType_5583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 24), union_type_copy_5582, 'UnionType')
            # Obtaining the member 'add' of a type (line 40)
            add_5584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 24), UnionType_5583, 'add')
            # Calling add(args, kwargs) (line 40)
            add_call_result_5588 = invoke(stypy.reporting.localization.Localization(__file__, 40, 24), add_5584, *[ret_union_5585, type__5586], **kwargs_5587)
            
            # Assigning a type to the variable 'ret_union' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'ret_union', add_call_result_5588)
            # SSA join for if statement (line 39)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'ret_union' (line 42)
    ret_union_5589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'ret_union')
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type', ret_union_5589)
    
    # ################# End of 'strip_undefined_type_from_union_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'strip_undefined_type_from_union_type' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_5590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5590)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'strip_undefined_type_from_union_type'
    return stypy_return_type_5590

# Assigning a type to the variable 'strip_undefined_type_from_union_type' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'strip_undefined_type_from_union_type', strip_undefined_type_from_union_type)

@norecursion
def check_undefined_type_within_parameters(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_undefined_type_within_parameters'
    module_type_store = module_type_store.open_function_context('check_undefined_type_within_parameters', 45, 0, False)
    
    # Passed parameters checking function
    check_undefined_type_within_parameters.stypy_localization = localization
    check_undefined_type_within_parameters.stypy_type_of_self = None
    check_undefined_type_within_parameters.stypy_type_store = module_type_store
    check_undefined_type_within_parameters.stypy_function_name = 'check_undefined_type_within_parameters'
    check_undefined_type_within_parameters.stypy_param_names_list = ['localization', 'call_description']
    check_undefined_type_within_parameters.stypy_varargs_param_name = 'arg_types'
    check_undefined_type_within_parameters.stypy_kwargs_param_name = 'kwargs_types'
    check_undefined_type_within_parameters.stypy_call_defaults = defaults
    check_undefined_type_within_parameters.stypy_call_varargs = varargs
    check_undefined_type_within_parameters.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_undefined_type_within_parameters', ['localization', 'call_description'], 'arg_types', 'kwargs_types', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_undefined_type_within_parameters', localization, ['localization', 'call_description'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_undefined_type_within_parameters(...)' code ##################

    str_5591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, (-1)), 'str', '\n    When calling a callable element, the type of some parameters might be undefined (not initialized\n    to any value in the preceding code). This function check this fact and substitute the Undefined\n    parameters by suitable type errors. It also creates warnings if the undefined type is inside a\n    union type, removing the undefined type from the union afterwards. It does the same with keyword arguments.\n\n    :param localization: Caller information\n    :param call_description: A textual description of the call (to generate errors)\n    :param arg_types: Call arguments\n    :param kwargs_types: Call keyword arguments\n    :return: arguments, keyword arguments tuple with the undefined types removed or substituted by TypeErrors depending\n    on if they are into union types or not\n    ')
    
    # Assigning a Call to a Name (line 59):
    
    # Assigning a Call to a Name (line 59):
    
    # Call to list(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'arg_types' (line 59)
    arg_types_5593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 26), 'arg_types', False)
    # Processing the call keyword arguments (line 59)
    kwargs_5594 = {}
    # Getting the type of 'list' (line 59)
    list_5592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'list', False)
    # Calling list(args, kwargs) (line 59)
    list_call_result_5595 = invoke(stypy.reporting.localization.Localization(__file__, 59, 21), list_5592, *[arg_types_5593], **kwargs_5594)
    
    # Assigning a type to the variable 'arg_types_list' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'arg_types_list', list_call_result_5595)
    
    
    # Call to range(...): (line 62)
    # Processing the call arguments (line 62)
    
    # Call to len(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'arg_types_list' (line 62)
    arg_types_list_5598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'arg_types_list', False)
    # Processing the call keyword arguments (line 62)
    kwargs_5599 = {}
    # Getting the type of 'len' (line 62)
    len_5597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 19), 'len', False)
    # Calling len(args, kwargs) (line 62)
    len_call_result_5600 = invoke(stypy.reporting.localization.Localization(__file__, 62, 19), len_5597, *[arg_types_list_5598], **kwargs_5599)
    
    # Processing the call keyword arguments (line 62)
    kwargs_5601 = {}
    # Getting the type of 'range' (line 62)
    range_5596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 13), 'range', False)
    # Calling range(args, kwargs) (line 62)
    range_call_result_5602 = invoke(stypy.reporting.localization.Localization(__file__, 62, 13), range_5596, *[len_call_result_5600], **kwargs_5601)
    
    # Assigning a type to the variable 'range_call_result_5602' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'range_call_result_5602', range_call_result_5602)
    # Testing if the for loop is going to be iterated (line 62)
    # Testing the type of a for loop iterable (line 62)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 62, 4), range_call_result_5602)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 62, 4), range_call_result_5602):
        # Getting the type of the for loop variable (line 62)
        for_loop_var_5603 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 62, 4), range_call_result_5602)
        # Assigning a type to the variable 'i' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'i', for_loop_var_5603)
        # SSA begins for a for statement (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to isinstance(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 63)
        i_5605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 37), 'i', False)
        # Getting the type of 'arg_types_list' (line 63)
        arg_types_list_5606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'arg_types_list', False)
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___5607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 22), arg_types_list_5606, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_5608 = invoke(stypy.reporting.localization.Localization(__file__, 63, 22), getitem___5607, i_5605)
        
        # Getting the type of 'union_type_copy' (line 63)
        union_type_copy_5609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 41), 'union_type_copy', False)
        # Obtaining the member 'UnionType' of a type (line 63)
        UnionType_5610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 41), union_type_copy_5609, 'UnionType')
        # Processing the call keyword arguments (line 63)
        kwargs_5611 = {}
        # Getting the type of 'isinstance' (line 63)
        isinstance_5604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 63)
        isinstance_call_result_5612 = invoke(stypy.reporting.localization.Localization(__file__, 63, 11), isinstance_5604, *[subscript_call_result_5608, UnionType_5610], **kwargs_5611)
        
        # Testing if the type of an if condition is none (line 63)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 63, 8), isinstance_call_result_5612):
            
            # Call to isinstance(...): (line 83)
            # Processing the call arguments (line 83)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 83)
            i_5684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 36), 'i', False)
            # Getting the type of 'arg_types' (line 83)
            arg_types_5685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 26), 'arg_types', False)
            # Obtaining the member '__getitem__' of a type (line 83)
            getitem___5686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 26), arg_types_5685, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 83)
            subscript_call_result_5687 = invoke(stypy.reporting.localization.Localization(__file__, 83, 26), getitem___5686, i_5684)
            
            # Getting the type of 'UndefinedType' (line 83)
            UndefinedType_5688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 40), 'UndefinedType', False)
            # Processing the call keyword arguments (line 83)
            kwargs_5689 = {}
            # Getting the type of 'isinstance' (line 83)
            isinstance_5683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 83)
            isinstance_call_result_5690 = invoke(stypy.reporting.localization.Localization(__file__, 83, 15), isinstance_5683, *[subscript_call_result_5687, UndefinedType_5688], **kwargs_5689)
            
            # Testing if the type of an if condition is none (line 83)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 83, 12), isinstance_call_result_5690):
                pass
            else:
                
                # Testing the type of an if condition (line 83)
                if_condition_5691 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 12), isinstance_call_result_5690)
                # Assigning a type to the variable 'if_condition_5691' (line 83)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'if_condition_5691', if_condition_5691)
                # SSA begins for if statement (line 83)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 84):
                
                # Assigning a Call to a Name (line 84):
                
                # Call to get_param_position(...): (line 84)
                # Processing the call arguments (line 84)
                
                # Call to get_line_from_module_code(...): (line 85)
                # Processing the call arguments (line 85)
                # Getting the type of 'localization' (line 86)
                localization_5697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'localization', False)
                # Obtaining the member 'file_name' of a type (line 86)
                file_name_5698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 24), localization_5697, 'file_name')
                # Getting the type of 'localization' (line 86)
                localization_5699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 48), 'localization', False)
                # Obtaining the member 'line' of a type (line 86)
                line_5700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 48), localization_5699, 'line')
                # Processing the call keyword arguments (line 85)
                kwargs_5701 = {}
                # Getting the type of 'module_line_numbering_copy' (line 85)
                module_line_numbering_copy_5694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'module_line_numbering_copy', False)
                # Obtaining the member 'ModuleLineNumbering' of a type (line 85)
                ModuleLineNumbering_5695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 20), module_line_numbering_copy_5694, 'ModuleLineNumbering')
                # Obtaining the member 'get_line_from_module_code' of a type (line 85)
                get_line_from_module_code_5696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 20), ModuleLineNumbering_5695, 'get_line_from_module_code')
                # Calling get_line_from_module_code(args, kwargs) (line 85)
                get_line_from_module_code_call_result_5702 = invoke(stypy.reporting.localization.Localization(__file__, 85, 20), get_line_from_module_code_5696, *[file_name_5698, line_5700], **kwargs_5701)
                
                # Getting the type of 'i' (line 86)
                i_5703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 68), 'i', False)
                # Processing the call keyword arguments (line 84)
                kwargs_5704 = {}
                # Getting the type of 'print_utils_copy' (line 84)
                print_utils_copy_5692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 25), 'print_utils_copy', False)
                # Obtaining the member 'get_param_position' of a type (line 84)
                get_param_position_5693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 25), print_utils_copy_5692, 'get_param_position')
                # Calling get_param_position(args, kwargs) (line 84)
                get_param_position_call_result_5705 = invoke(stypy.reporting.localization.Localization(__file__, 84, 25), get_param_position_5693, *[get_line_from_module_code_call_result_5702, i_5703], **kwargs_5704)
                
                # Assigning a type to the variable 'offset' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'offset', get_param_position_call_result_5705)
                
                # Getting the type of 'offset' (line 87)
                offset_5706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'offset')
                int_5707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 33), 'int')
                # Applying the binary operator 'isnot' (line 87)
                result_is_not_5708 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 19), 'isnot', offset_5706, int_5707)
                
                # Testing if the type of an if condition is none (line 87)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 87, 16), result_is_not_5708):
                    
                    # Assigning a Name to a Name (line 91):
                    
                    # Assigning a Name to a Name (line 91):
                    # Getting the type of 'localization' (line 91)
                    localization_5716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'localization')
                    # Assigning a type to the variable 'clone_loc' (line 91)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'clone_loc', localization_5716)
                else:
                    
                    # Testing the type of an if condition (line 87)
                    if_condition_5709 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 16), result_is_not_5708)
                    # Assigning a type to the variable 'if_condition_5709' (line 87)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'if_condition_5709', if_condition_5709)
                    # SSA begins for if statement (line 87)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 88):
                    
                    # Assigning a Call to a Name (line 88):
                    
                    # Call to clone(...): (line 88)
                    # Processing the call keyword arguments (line 88)
                    kwargs_5712 = {}
                    # Getting the type of 'localization' (line 88)
                    localization_5710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 32), 'localization', False)
                    # Obtaining the member 'clone' of a type (line 88)
                    clone_5711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 32), localization_5710, 'clone')
                    # Calling clone(args, kwargs) (line 88)
                    clone_call_result_5713 = invoke(stypy.reporting.localization.Localization(__file__, 88, 32), clone_5711, *[], **kwargs_5712)
                    
                    # Assigning a type to the variable 'clone_loc' (line 88)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'clone_loc', clone_call_result_5713)
                    
                    # Assigning a Name to a Attribute (line 89):
                    
                    # Assigning a Name to a Attribute (line 89):
                    # Getting the type of 'offset' (line 89)
                    offset_5714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 39), 'offset')
                    # Getting the type of 'clone_loc' (line 89)
                    clone_loc_5715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'clone_loc')
                    # Setting the type of the member 'column' of a type (line 89)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 20), clone_loc_5715, 'column', offset_5714)
                    # SSA branch for the else part of an if statement (line 87)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Name to a Name (line 91):
                    
                    # Assigning a Name to a Name (line 91):
                    # Getting the type of 'localization' (line 91)
                    localization_5716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'localization')
                    # Assigning a type to the variable 'clone_loc' (line 91)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'clone_loc', localization_5716)
                    # SSA join for if statement (line 87)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Subscript (line 93):
                
                # Assigning a Call to a Subscript (line 93):
                
                # Call to TypeError(...): (line 93)
                # Processing the call arguments (line 93)
                # Getting the type of 'clone_loc' (line 93)
                clone_loc_5718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 46), 'clone_loc', False)
                
                # Call to format(...): (line 93)
                # Processing the call arguments (line 93)
                # Getting the type of 'call_description' (line 93)
                call_description_5721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 99), 'call_description', False)
                # Getting the type of 'i' (line 94)
                i_5722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 99), 'i', False)
                int_5723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 103), 'int')
                # Applying the binary operator '+' (line 94)
                result_add_5724 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 99), '+', i_5722, int_5723)
                
                # Processing the call keyword arguments (line 93)
                kwargs_5725 = {}
                str_5719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 57), 'str', '{0}: Argument {1} is not defined')
                # Obtaining the member 'format' of a type (line 93)
                format_5720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 57), str_5719, 'format')
                # Calling format(args, kwargs) (line 93)
                format_call_result_5726 = invoke(stypy.reporting.localization.Localization(__file__, 93, 57), format_5720, *[call_description_5721, result_add_5724], **kwargs_5725)
                
                # Processing the call keyword arguments (line 93)
                kwargs_5727 = {}
                # Getting the type of 'TypeError' (line 93)
                TypeError_5717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 36), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 93)
                TypeError_call_result_5728 = invoke(stypy.reporting.localization.Localization(__file__, 93, 36), TypeError_5717, *[clone_loc_5718, format_call_result_5726], **kwargs_5727)
                
                # Getting the type of 'arg_types_list' (line 93)
                arg_types_list_5729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'arg_types_list')
                # Getting the type of 'i' (line 93)
                i_5730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'i')
                # Storing an element on a container (line 93)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 16), arg_types_list_5729, (i_5730, TypeError_call_result_5728))
                # SSA join for if statement (line 83)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 63)
            if_condition_5613 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 8), isinstance_call_result_5612)
            # Assigning a type to the variable 'if_condition_5613' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'if_condition_5613', if_condition_5613)
            # SSA begins for if statement (line 63)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Compare to a Name (line 65):
            
            # Assigning a Compare to a Name (line 65):
            
            
            # Call to len(...): (line 65)
            # Processing the call arguments (line 65)
            
            # Call to filter(...): (line 65)
            # Processing the call arguments (line 65)

            @norecursion
            def _stypy_temp_lambda_9(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_9'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_9', 65, 41, True)
                # Passed parameters checking function
                _stypy_temp_lambda_9.stypy_localization = localization
                _stypy_temp_lambda_9.stypy_type_of_self = None
                _stypy_temp_lambda_9.stypy_type_store = module_type_store
                _stypy_temp_lambda_9.stypy_function_name = '_stypy_temp_lambda_9'
                _stypy_temp_lambda_9.stypy_param_names_list = ['elem']
                _stypy_temp_lambda_9.stypy_varargs_param_name = None
                _stypy_temp_lambda_9.stypy_kwargs_param_name = None
                _stypy_temp_lambda_9.stypy_call_defaults = defaults
                _stypy_temp_lambda_9.stypy_call_varargs = varargs
                _stypy_temp_lambda_9.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_9', ['elem'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_9', ['elem'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to isinstance(...): (line 65)
                # Processing the call arguments (line 65)
                # Getting the type of 'elem' (line 65)
                elem_5617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 65), 'elem', False)
                # Getting the type of 'UndefinedType' (line 65)
                UndefinedType_5618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 71), 'UndefinedType', False)
                # Processing the call keyword arguments (line 65)
                kwargs_5619 = {}
                # Getting the type of 'isinstance' (line 65)
                isinstance_5616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 54), 'isinstance', False)
                # Calling isinstance(args, kwargs) (line 65)
                isinstance_call_result_5620 = invoke(stypy.reporting.localization.Localization(__file__, 65, 54), isinstance_5616, *[elem_5617, UndefinedType_5618], **kwargs_5619)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 65)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 41), 'stypy_return_type', isinstance_call_result_5620)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_9' in the type store
                # Getting the type of 'stypy_return_type' (line 65)
                stypy_return_type_5621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 41), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_5621)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_9'
                return stypy_return_type_5621

            # Assigning a type to the variable '_stypy_temp_lambda_9' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 41), '_stypy_temp_lambda_9', _stypy_temp_lambda_9)
            # Getting the type of '_stypy_temp_lambda_9' (line 65)
            _stypy_temp_lambda_9_5622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 41), '_stypy_temp_lambda_9')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 65)
            i_5623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 97), 'i', False)
            # Getting the type of 'arg_types' (line 65)
            arg_types_5624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 87), 'arg_types', False)
            # Obtaining the member '__getitem__' of a type (line 65)
            getitem___5625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 87), arg_types_5624, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 65)
            subscript_call_result_5626 = invoke(stypy.reporting.localization.Localization(__file__, 65, 87), getitem___5625, i_5623)
            
            # Obtaining the member 'types' of a type (line 65)
            types_5627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 87), subscript_call_result_5626, 'types')
            # Processing the call keyword arguments (line 65)
            kwargs_5628 = {}
            # Getting the type of 'filter' (line 65)
            filter_5615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 34), 'filter', False)
            # Calling filter(args, kwargs) (line 65)
            filter_call_result_5629 = invoke(stypy.reporting.localization.Localization(__file__, 65, 34), filter_5615, *[_stypy_temp_lambda_9_5622, types_5627], **kwargs_5628)
            
            # Processing the call keyword arguments (line 65)
            kwargs_5630 = {}
            # Getting the type of 'len' (line 65)
            len_5614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'len', False)
            # Calling len(args, kwargs) (line 65)
            len_call_result_5631 = invoke(stypy.reporting.localization.Localization(__file__, 65, 30), len_5614, *[filter_call_result_5629], **kwargs_5630)
            
            int_5632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 110), 'int')
            # Applying the binary operator '>' (line 65)
            result_gt_5633 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 30), '>', len_call_result_5631, int_5632)
            
            # Assigning a type to the variable 'exist_undefined' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'exist_undefined', result_gt_5633)
            # Getting the type of 'exist_undefined' (line 66)
            exist_undefined_5634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 15), 'exist_undefined')
            # Testing if the type of an if condition is none (line 66)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 66, 12), exist_undefined_5634):
                pass
            else:
                
                # Testing the type of an if condition (line 66)
                if_condition_5635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 12), exist_undefined_5634)
                # Assigning a type to the variable 'if_condition_5635' (line 66)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'if_condition_5635', if_condition_5635)
                # SSA begins for if statement (line 66)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 68):
                
                # Assigning a Call to a Name (line 68):
                
                # Call to get_param_position(...): (line 68)
                # Processing the call arguments (line 68)
                
                # Call to get_line_from_module_code(...): (line 69)
                # Processing the call arguments (line 69)
                # Getting the type of 'localization' (line 70)
                localization_5641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 24), 'localization', False)
                # Obtaining the member 'file_name' of a type (line 70)
                file_name_5642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 24), localization_5641, 'file_name')
                # Getting the type of 'localization' (line 70)
                localization_5643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 48), 'localization', False)
                # Obtaining the member 'line' of a type (line 70)
                line_5644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 48), localization_5643, 'line')
                # Processing the call keyword arguments (line 69)
                kwargs_5645 = {}
                # Getting the type of 'module_line_numbering_copy' (line 69)
                module_line_numbering_copy_5638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'module_line_numbering_copy', False)
                # Obtaining the member 'ModuleLineNumbering' of a type (line 69)
                ModuleLineNumbering_5639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 20), module_line_numbering_copy_5638, 'ModuleLineNumbering')
                # Obtaining the member 'get_line_from_module_code' of a type (line 69)
                get_line_from_module_code_5640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 20), ModuleLineNumbering_5639, 'get_line_from_module_code')
                # Calling get_line_from_module_code(args, kwargs) (line 69)
                get_line_from_module_code_call_result_5646 = invoke(stypy.reporting.localization.Localization(__file__, 69, 20), get_line_from_module_code_5640, *[file_name_5642, line_5644], **kwargs_5645)
                
                # Getting the type of 'i' (line 70)
                i_5647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 68), 'i', False)
                # Processing the call keyword arguments (line 68)
                kwargs_5648 = {}
                # Getting the type of 'print_utils_copy' (line 68)
                print_utils_copy_5636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 'print_utils_copy', False)
                # Obtaining the member 'get_param_position' of a type (line 68)
                get_param_position_5637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 25), print_utils_copy_5636, 'get_param_position')
                # Calling get_param_position(args, kwargs) (line 68)
                get_param_position_call_result_5649 = invoke(stypy.reporting.localization.Localization(__file__, 68, 25), get_param_position_5637, *[get_line_from_module_code_call_result_5646, i_5647], **kwargs_5648)
                
                # Assigning a type to the variable 'offset' (line 68)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'offset', get_param_position_call_result_5649)
                
                # Getting the type of 'offset' (line 71)
                offset_5650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'offset')
                int_5651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 33), 'int')
                # Applying the binary operator 'isnot' (line 71)
                result_is_not_5652 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 19), 'isnot', offset_5650, int_5651)
                
                # Testing if the type of an if condition is none (line 71)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 71, 16), result_is_not_5652):
                    
                    # Assigning a Name to a Name (line 75):
                    
                    # Assigning a Name to a Name (line 75):
                    # Getting the type of 'localization' (line 75)
                    localization_5660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'localization')
                    # Assigning a type to the variable 'clone_loc' (line 75)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'clone_loc', localization_5660)
                else:
                    
                    # Testing the type of an if condition (line 71)
                    if_condition_5653 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 16), result_is_not_5652)
                    # Assigning a type to the variable 'if_condition_5653' (line 71)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'if_condition_5653', if_condition_5653)
                    # SSA begins for if statement (line 71)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 72):
                    
                    # Assigning a Call to a Name (line 72):
                    
                    # Call to clone(...): (line 72)
                    # Processing the call keyword arguments (line 72)
                    kwargs_5656 = {}
                    # Getting the type of 'localization' (line 72)
                    localization_5654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'localization', False)
                    # Obtaining the member 'clone' of a type (line 72)
                    clone_5655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 32), localization_5654, 'clone')
                    # Calling clone(args, kwargs) (line 72)
                    clone_call_result_5657 = invoke(stypy.reporting.localization.Localization(__file__, 72, 32), clone_5655, *[], **kwargs_5656)
                    
                    # Assigning a type to the variable 'clone_loc' (line 72)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'clone_loc', clone_call_result_5657)
                    
                    # Assigning a Name to a Attribute (line 73):
                    
                    # Assigning a Name to a Attribute (line 73):
                    # Getting the type of 'offset' (line 73)
                    offset_5658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 39), 'offset')
                    # Getting the type of 'clone_loc' (line 73)
                    clone_loc_5659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'clone_loc')
                    # Setting the type of the member 'column' of a type (line 73)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 20), clone_loc_5659, 'column', offset_5658)
                    # SSA branch for the else part of an if statement (line 71)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Name to a Name (line 75):
                    
                    # Assigning a Name to a Name (line 75):
                    # Getting the type of 'localization' (line 75)
                    localization_5660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'localization')
                    # Assigning a type to the variable 'clone_loc' (line 75)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'clone_loc', localization_5660)
                    # SSA join for if statement (line 71)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Call to instance(...): (line 76)
                # Processing the call arguments (line 76)
                # Getting the type of 'clone_loc' (line 76)
                clone_loc_5663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 37), 'clone_loc', False)
                
                # Call to format(...): (line 76)
                # Processing the call arguments (line 76)
                # Getting the type of 'call_description' (line 76)
                call_description_5666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 94), 'call_description', False)
                # Getting the type of 'i' (line 77)
                i_5667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 94), 'i', False)
                int_5668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 98), 'int')
                # Applying the binary operator '+' (line 77)
                result_add_5669 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 94), '+', i_5667, int_5668)
                
                # Processing the call keyword arguments (line 76)
                kwargs_5670 = {}
                str_5664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 48), 'str', '{0}: Argument {1} could be undefined')
                # Obtaining the member 'format' of a type (line 76)
                format_5665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 48), str_5664, 'format')
                # Calling format(args, kwargs) (line 76)
                format_call_result_5671 = invoke(stypy.reporting.localization.Localization(__file__, 76, 48), format_5665, *[call_description_5666, result_add_5669], **kwargs_5670)
                
                # Processing the call keyword arguments (line 76)
                kwargs_5672 = {}
                # Getting the type of 'TypeWarning' (line 76)
                TypeWarning_5661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'TypeWarning', False)
                # Obtaining the member 'instance' of a type (line 76)
                instance_5662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 16), TypeWarning_5661, 'instance')
                # Calling instance(args, kwargs) (line 76)
                instance_call_result_5673 = invoke(stypy.reporting.localization.Localization(__file__, 76, 16), instance_5662, *[clone_loc_5663, format_call_result_5671], **kwargs_5672)
                
                # SSA join for if statement (line 66)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Subscript (line 79):
            
            # Assigning a Call to a Subscript (line 79):
            
            # Call to strip_undefined_type_from_union_type(...): (line 79)
            # Processing the call arguments (line 79)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 79)
            i_5675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 79), 'i', False)
            # Getting the type of 'arg_types' (line 79)
            arg_types_5676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 69), 'arg_types', False)
            # Obtaining the member '__getitem__' of a type (line 79)
            getitem___5677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 69), arg_types_5676, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 79)
            subscript_call_result_5678 = invoke(stypy.reporting.localization.Localization(__file__, 79, 69), getitem___5677, i_5675)
            
            # Processing the call keyword arguments (line 79)
            kwargs_5679 = {}
            # Getting the type of 'strip_undefined_type_from_union_type' (line 79)
            strip_undefined_type_from_union_type_5674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 32), 'strip_undefined_type_from_union_type', False)
            # Calling strip_undefined_type_from_union_type(args, kwargs) (line 79)
            strip_undefined_type_from_union_type_call_result_5680 = invoke(stypy.reporting.localization.Localization(__file__, 79, 32), strip_undefined_type_from_union_type_5674, *[subscript_call_result_5678], **kwargs_5679)
            
            # Getting the type of 'arg_types_list' (line 79)
            arg_types_list_5681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'arg_types_list')
            # Getting the type of 'i' (line 79)
            i_5682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 27), 'i')
            # Storing an element on a container (line 79)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 12), arg_types_list_5681, (i_5682, strip_undefined_type_from_union_type_call_result_5680))
            # SSA branch for the else part of an if statement (line 63)
            module_type_store.open_ssa_branch('else')
            
            # Call to isinstance(...): (line 83)
            # Processing the call arguments (line 83)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 83)
            i_5684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 36), 'i', False)
            # Getting the type of 'arg_types' (line 83)
            arg_types_5685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 26), 'arg_types', False)
            # Obtaining the member '__getitem__' of a type (line 83)
            getitem___5686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 26), arg_types_5685, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 83)
            subscript_call_result_5687 = invoke(stypy.reporting.localization.Localization(__file__, 83, 26), getitem___5686, i_5684)
            
            # Getting the type of 'UndefinedType' (line 83)
            UndefinedType_5688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 40), 'UndefinedType', False)
            # Processing the call keyword arguments (line 83)
            kwargs_5689 = {}
            # Getting the type of 'isinstance' (line 83)
            isinstance_5683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 83)
            isinstance_call_result_5690 = invoke(stypy.reporting.localization.Localization(__file__, 83, 15), isinstance_5683, *[subscript_call_result_5687, UndefinedType_5688], **kwargs_5689)
            
            # Testing if the type of an if condition is none (line 83)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 83, 12), isinstance_call_result_5690):
                pass
            else:
                
                # Testing the type of an if condition (line 83)
                if_condition_5691 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 12), isinstance_call_result_5690)
                # Assigning a type to the variable 'if_condition_5691' (line 83)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'if_condition_5691', if_condition_5691)
                # SSA begins for if statement (line 83)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 84):
                
                # Assigning a Call to a Name (line 84):
                
                # Call to get_param_position(...): (line 84)
                # Processing the call arguments (line 84)
                
                # Call to get_line_from_module_code(...): (line 85)
                # Processing the call arguments (line 85)
                # Getting the type of 'localization' (line 86)
                localization_5697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 24), 'localization', False)
                # Obtaining the member 'file_name' of a type (line 86)
                file_name_5698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 24), localization_5697, 'file_name')
                # Getting the type of 'localization' (line 86)
                localization_5699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 48), 'localization', False)
                # Obtaining the member 'line' of a type (line 86)
                line_5700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 48), localization_5699, 'line')
                # Processing the call keyword arguments (line 85)
                kwargs_5701 = {}
                # Getting the type of 'module_line_numbering_copy' (line 85)
                module_line_numbering_copy_5694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'module_line_numbering_copy', False)
                # Obtaining the member 'ModuleLineNumbering' of a type (line 85)
                ModuleLineNumbering_5695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 20), module_line_numbering_copy_5694, 'ModuleLineNumbering')
                # Obtaining the member 'get_line_from_module_code' of a type (line 85)
                get_line_from_module_code_5696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 20), ModuleLineNumbering_5695, 'get_line_from_module_code')
                # Calling get_line_from_module_code(args, kwargs) (line 85)
                get_line_from_module_code_call_result_5702 = invoke(stypy.reporting.localization.Localization(__file__, 85, 20), get_line_from_module_code_5696, *[file_name_5698, line_5700], **kwargs_5701)
                
                # Getting the type of 'i' (line 86)
                i_5703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 68), 'i', False)
                # Processing the call keyword arguments (line 84)
                kwargs_5704 = {}
                # Getting the type of 'print_utils_copy' (line 84)
                print_utils_copy_5692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 25), 'print_utils_copy', False)
                # Obtaining the member 'get_param_position' of a type (line 84)
                get_param_position_5693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 25), print_utils_copy_5692, 'get_param_position')
                # Calling get_param_position(args, kwargs) (line 84)
                get_param_position_call_result_5705 = invoke(stypy.reporting.localization.Localization(__file__, 84, 25), get_param_position_5693, *[get_line_from_module_code_call_result_5702, i_5703], **kwargs_5704)
                
                # Assigning a type to the variable 'offset' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'offset', get_param_position_call_result_5705)
                
                # Getting the type of 'offset' (line 87)
                offset_5706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'offset')
                int_5707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 33), 'int')
                # Applying the binary operator 'isnot' (line 87)
                result_is_not_5708 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 19), 'isnot', offset_5706, int_5707)
                
                # Testing if the type of an if condition is none (line 87)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 87, 16), result_is_not_5708):
                    
                    # Assigning a Name to a Name (line 91):
                    
                    # Assigning a Name to a Name (line 91):
                    # Getting the type of 'localization' (line 91)
                    localization_5716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'localization')
                    # Assigning a type to the variable 'clone_loc' (line 91)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'clone_loc', localization_5716)
                else:
                    
                    # Testing the type of an if condition (line 87)
                    if_condition_5709 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 16), result_is_not_5708)
                    # Assigning a type to the variable 'if_condition_5709' (line 87)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'if_condition_5709', if_condition_5709)
                    # SSA begins for if statement (line 87)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 88):
                    
                    # Assigning a Call to a Name (line 88):
                    
                    # Call to clone(...): (line 88)
                    # Processing the call keyword arguments (line 88)
                    kwargs_5712 = {}
                    # Getting the type of 'localization' (line 88)
                    localization_5710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 32), 'localization', False)
                    # Obtaining the member 'clone' of a type (line 88)
                    clone_5711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 32), localization_5710, 'clone')
                    # Calling clone(args, kwargs) (line 88)
                    clone_call_result_5713 = invoke(stypy.reporting.localization.Localization(__file__, 88, 32), clone_5711, *[], **kwargs_5712)
                    
                    # Assigning a type to the variable 'clone_loc' (line 88)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'clone_loc', clone_call_result_5713)
                    
                    # Assigning a Name to a Attribute (line 89):
                    
                    # Assigning a Name to a Attribute (line 89):
                    # Getting the type of 'offset' (line 89)
                    offset_5714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 39), 'offset')
                    # Getting the type of 'clone_loc' (line 89)
                    clone_loc_5715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'clone_loc')
                    # Setting the type of the member 'column' of a type (line 89)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 20), clone_loc_5715, 'column', offset_5714)
                    # SSA branch for the else part of an if statement (line 87)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Name to a Name (line 91):
                    
                    # Assigning a Name to a Name (line 91):
                    # Getting the type of 'localization' (line 91)
                    localization_5716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'localization')
                    # Assigning a type to the variable 'clone_loc' (line 91)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'clone_loc', localization_5716)
                    # SSA join for if statement (line 87)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Subscript (line 93):
                
                # Assigning a Call to a Subscript (line 93):
                
                # Call to TypeError(...): (line 93)
                # Processing the call arguments (line 93)
                # Getting the type of 'clone_loc' (line 93)
                clone_loc_5718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 46), 'clone_loc', False)
                
                # Call to format(...): (line 93)
                # Processing the call arguments (line 93)
                # Getting the type of 'call_description' (line 93)
                call_description_5721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 99), 'call_description', False)
                # Getting the type of 'i' (line 94)
                i_5722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 99), 'i', False)
                int_5723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 103), 'int')
                # Applying the binary operator '+' (line 94)
                result_add_5724 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 99), '+', i_5722, int_5723)
                
                # Processing the call keyword arguments (line 93)
                kwargs_5725 = {}
                str_5719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 57), 'str', '{0}: Argument {1} is not defined')
                # Obtaining the member 'format' of a type (line 93)
                format_5720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 57), str_5719, 'format')
                # Calling format(args, kwargs) (line 93)
                format_call_result_5726 = invoke(stypy.reporting.localization.Localization(__file__, 93, 57), format_5720, *[call_description_5721, result_add_5724], **kwargs_5725)
                
                # Processing the call keyword arguments (line 93)
                kwargs_5727 = {}
                # Getting the type of 'TypeError' (line 93)
                TypeError_5717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 36), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 93)
                TypeError_call_result_5728 = invoke(stypy.reporting.localization.Localization(__file__, 93, 36), TypeError_5717, *[clone_loc_5718, format_call_result_5726], **kwargs_5727)
                
                # Getting the type of 'arg_types_list' (line 93)
                arg_types_list_5729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'arg_types_list')
                # Getting the type of 'i' (line 93)
                i_5730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'i')
                # Storing an element on a container (line 93)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 16), arg_types_list_5729, (i_5730, TypeError_call_result_5728))
                # SSA join for if statement (line 83)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 63)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Subscript to a Subscript (line 97):
        
        # Assigning a Subscript to a Subscript (line 97):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 97)
        i_5731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 38), 'i')
        # Getting the type of 'arg_types' (line 97)
        arg_types_5732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'arg_types')
        # Obtaining the member '__getitem__' of a type (line 97)
        getitem___5733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 28), arg_types_5732, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 97)
        subscript_call_result_5734 = invoke(stypy.reporting.localization.Localization(__file__, 97, 28), getitem___5733, i_5731)
        
        # Getting the type of 'arg_types_list' (line 97)
        arg_types_list_5735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'arg_types_list')
        # Getting the type of 'i' (line 97)
        i_5736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'i')
        # Storing an element on a container (line 97)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 8), arg_types_list_5735, (i_5736, subscript_call_result_5734))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Dict to a Name (line 100):
    
    # Assigning a Dict to a Name (line 100):
    
    # Obtaining an instance of the builtin type 'dict' (line 100)
    dict_5737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 19), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 100)
    
    # Assigning a type to the variable 'final_kwargs' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'final_kwargs', dict_5737)
    
    
    # Call to items(...): (line 101)
    # Processing the call keyword arguments (line 101)
    kwargs_5740 = {}
    # Getting the type of 'kwargs_types' (line 101)
    kwargs_types_5738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'kwargs_types', False)
    # Obtaining the member 'items' of a type (line 101)
    items_5739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 22), kwargs_types_5738, 'items')
    # Calling items(args, kwargs) (line 101)
    items_call_result_5741 = invoke(stypy.reporting.localization.Localization(__file__, 101, 22), items_5739, *[], **kwargs_5740)
    
    # Assigning a type to the variable 'items_call_result_5741' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'items_call_result_5741', items_call_result_5741)
    # Testing if the for loop is going to be iterated (line 101)
    # Testing the type of a for loop iterable (line 101)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 101, 4), items_call_result_5741)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 101, 4), items_call_result_5741):
        # Getting the type of the for loop variable (line 101)
        for_loop_var_5742 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 101, 4), items_call_result_5741)
        # Assigning a type to the variable 'key' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 4), for_loop_var_5742, 2, 0))
        # Assigning a type to the variable 'value' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 4), for_loop_var_5742, 2, 1))
        # SSA begins for a for statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to isinstance(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'value' (line 102)
        value_5744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 22), 'value', False)
        # Getting the type of 'union_type_copy' (line 102)
        union_type_copy_5745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 29), 'union_type_copy', False)
        # Obtaining the member 'UnionType' of a type (line 102)
        UnionType_5746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 29), union_type_copy_5745, 'UnionType')
        # Processing the call keyword arguments (line 102)
        kwargs_5747 = {}
        # Getting the type of 'isinstance' (line 102)
        isinstance_5743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 102)
        isinstance_call_result_5748 = invoke(stypy.reporting.localization.Localization(__file__, 102, 11), isinstance_5743, *[value_5744, UnionType_5746], **kwargs_5747)
        
        # Testing if the type of an if condition is none (line 102)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 102, 8), isinstance_call_result_5748):
            
            # Call to isinstance(...): (line 111)
            # Processing the call arguments (line 111)
            # Getting the type of 'value' (line 111)
            value_5782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'value', False)
            # Getting the type of 'UndefinedType' (line 111)
            UndefinedType_5783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 33), 'UndefinedType', False)
            # Processing the call keyword arguments (line 111)
            kwargs_5784 = {}
            # Getting the type of 'isinstance' (line 111)
            isinstance_5781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 111)
            isinstance_call_result_5785 = invoke(stypy.reporting.localization.Localization(__file__, 111, 15), isinstance_5781, *[value_5782, UndefinedType_5783], **kwargs_5784)
            
            # Testing if the type of an if condition is none (line 111)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 111, 12), isinstance_call_result_5785):
                pass
            else:
                
                # Testing the type of an if condition (line 111)
                if_condition_5786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 12), isinstance_call_result_5785)
                # Assigning a type to the variable 'if_condition_5786' (line 111)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'if_condition_5786', if_condition_5786)
                # SSA begins for if statement (line 111)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 112):
                
                # Assigning a Call to a Subscript (line 112):
                
                # Call to TypeError(...): (line 112)
                # Processing the call arguments (line 112)
                # Getting the type of 'localization' (line 112)
                localization_5788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 46), 'localization', False)
                
                # Call to format(...): (line 113)
                # Processing the call arguments (line 113)
                # Getting the type of 'call_description' (line 113)
                call_description_5791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 96), 'call_description', False)
                # Getting the type of 'key' (line 114)
                key_5792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 96), 'key', False)
                # Processing the call keyword arguments (line 113)
                kwargs_5793 = {}
                str_5789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 46), 'str', '{0}: Keyword argument {1} is not defined')
                # Obtaining the member 'format' of a type (line 113)
                format_5790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 46), str_5789, 'format')
                # Calling format(args, kwargs) (line 113)
                format_call_result_5794 = invoke(stypy.reporting.localization.Localization(__file__, 113, 46), format_5790, *[call_description_5791, key_5792], **kwargs_5793)
                
                # Processing the call keyword arguments (line 112)
                kwargs_5795 = {}
                # Getting the type of 'TypeError' (line 112)
                TypeError_5787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 36), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 112)
                TypeError_call_result_5796 = invoke(stypy.reporting.localization.Localization(__file__, 112, 36), TypeError_5787, *[localization_5788, format_call_result_5794], **kwargs_5795)
                
                # Getting the type of 'final_kwargs' (line 112)
                final_kwargs_5797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'final_kwargs')
                # Getting the type of 'key' (line 112)
                key_5798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 29), 'key')
                # Storing an element on a container (line 112)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 16), final_kwargs_5797, (key_5798, TypeError_call_result_5796))
                # SSA join for if statement (line 111)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 102)
            if_condition_5749 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 8), isinstance_call_result_5748)
            # Assigning a type to the variable 'if_condition_5749' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'if_condition_5749', if_condition_5749)
            # SSA begins for if statement (line 102)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 103):
            
            # Assigning a Call to a Name (line 103):
            
            # Call to filter(...): (line 103)
            # Processing the call arguments (line 103)

            @norecursion
            def _stypy_temp_lambda_10(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_10'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_10', 103, 37, True)
                # Passed parameters checking function
                _stypy_temp_lambda_10.stypy_localization = localization
                _stypy_temp_lambda_10.stypy_type_of_self = None
                _stypy_temp_lambda_10.stypy_type_store = module_type_store
                _stypy_temp_lambda_10.stypy_function_name = '_stypy_temp_lambda_10'
                _stypy_temp_lambda_10.stypy_param_names_list = ['elem']
                _stypy_temp_lambda_10.stypy_varargs_param_name = None
                _stypy_temp_lambda_10.stypy_kwargs_param_name = None
                _stypy_temp_lambda_10.stypy_call_defaults = defaults
                _stypy_temp_lambda_10.stypy_call_varargs = varargs
                _stypy_temp_lambda_10.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_10', ['elem'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_10', ['elem'], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to isinstance(...): (line 103)
                # Processing the call arguments (line 103)
                # Getting the type of 'elem' (line 103)
                elem_5752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 61), 'elem', False)
                # Getting the type of 'UndefinedType' (line 103)
                UndefinedType_5753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 67), 'UndefinedType', False)
                # Processing the call keyword arguments (line 103)
                kwargs_5754 = {}
                # Getting the type of 'isinstance' (line 103)
                isinstance_5751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 50), 'isinstance', False)
                # Calling isinstance(args, kwargs) (line 103)
                isinstance_call_result_5755 = invoke(stypy.reporting.localization.Localization(__file__, 103, 50), isinstance_5751, *[elem_5752, UndefinedType_5753], **kwargs_5754)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 103)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 37), 'stypy_return_type', isinstance_call_result_5755)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_10' in the type store
                # Getting the type of 'stypy_return_type' (line 103)
                stypy_return_type_5756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 37), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_5756)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_10'
                return stypy_return_type_5756

            # Assigning a type to the variable '_stypy_temp_lambda_10' (line 103)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 37), '_stypy_temp_lambda_10', _stypy_temp_lambda_10)
            # Getting the type of '_stypy_temp_lambda_10' (line 103)
            _stypy_temp_lambda_10_5757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 37), '_stypy_temp_lambda_10')
            # Getting the type of 'value' (line 103)
            value_5758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 83), 'value', False)
            # Obtaining the member 'types' of a type (line 103)
            types_5759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 83), value_5758, 'types')
            # Processing the call keyword arguments (line 103)
            kwargs_5760 = {}
            # Getting the type of 'filter' (line 103)
            filter_5750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), 'filter', False)
            # Calling filter(args, kwargs) (line 103)
            filter_call_result_5761 = invoke(stypy.reporting.localization.Localization(__file__, 103, 30), filter_5750, *[_stypy_temp_lambda_10_5757, types_5759], **kwargs_5760)
            
            # Assigning a type to the variable 'exist_undefined' (line 103)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'exist_undefined', filter_call_result_5761)
            # Getting the type of 'exist_undefined' (line 104)
            exist_undefined_5762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'exist_undefined')
            # Testing if the type of an if condition is none (line 104)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 104, 12), exist_undefined_5762):
                pass
            else:
                
                # Testing the type of an if condition (line 104)
                if_condition_5763 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 12), exist_undefined_5762)
                # Assigning a type to the variable 'if_condition_5763' (line 104)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'if_condition_5763', if_condition_5763)
                # SSA begins for if statement (line 104)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to instance(...): (line 105)
                # Processing the call arguments (line 105)
                # Getting the type of 'localization' (line 105)
                localization_5766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 37), 'localization', False)
                
                # Call to format(...): (line 106)
                # Processing the call arguments (line 106)
                # Getting the type of 'call_description' (line 106)
                call_description_5769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 91), 'call_description', False)
                # Getting the type of 'key' (line 107)
                key_5770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 91), 'key', False)
                # Processing the call keyword arguments (line 106)
                kwargs_5771 = {}
                str_5767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 37), 'str', '{0}: Keyword argument {1} could be undefined')
                # Obtaining the member 'format' of a type (line 106)
                format_5768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 37), str_5767, 'format')
                # Calling format(args, kwargs) (line 106)
                format_call_result_5772 = invoke(stypy.reporting.localization.Localization(__file__, 106, 37), format_5768, *[call_description_5769, key_5770], **kwargs_5771)
                
                # Processing the call keyword arguments (line 105)
                kwargs_5773 = {}
                # Getting the type of 'TypeWarning' (line 105)
                TypeWarning_5764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'TypeWarning', False)
                # Obtaining the member 'instance' of a type (line 105)
                instance_5765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 16), TypeWarning_5764, 'instance')
                # Calling instance(args, kwargs) (line 105)
                instance_call_result_5774 = invoke(stypy.reporting.localization.Localization(__file__, 105, 16), instance_5765, *[localization_5766, format_call_result_5772], **kwargs_5773)
                
                # SSA join for if statement (line 104)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Subscript (line 108):
            
            # Assigning a Call to a Subscript (line 108):
            
            # Call to strip_undefined_type_from_union_type(...): (line 108)
            # Processing the call arguments (line 108)
            # Getting the type of 'value' (line 108)
            value_5776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 69), 'value', False)
            # Processing the call keyword arguments (line 108)
            kwargs_5777 = {}
            # Getting the type of 'strip_undefined_type_from_union_type' (line 108)
            strip_undefined_type_from_union_type_5775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 32), 'strip_undefined_type_from_union_type', False)
            # Calling strip_undefined_type_from_union_type(args, kwargs) (line 108)
            strip_undefined_type_from_union_type_call_result_5778 = invoke(stypy.reporting.localization.Localization(__file__, 108, 32), strip_undefined_type_from_union_type_5775, *[value_5776], **kwargs_5777)
            
            # Getting the type of 'final_kwargs' (line 108)
            final_kwargs_5779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'final_kwargs')
            # Getting the type of 'key' (line 108)
            key_5780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'key')
            # Storing an element on a container (line 108)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 12), final_kwargs_5779, (key_5780, strip_undefined_type_from_union_type_call_result_5778))
            # SSA branch for the else part of an if statement (line 102)
            module_type_store.open_ssa_branch('else')
            
            # Call to isinstance(...): (line 111)
            # Processing the call arguments (line 111)
            # Getting the type of 'value' (line 111)
            value_5782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'value', False)
            # Getting the type of 'UndefinedType' (line 111)
            UndefinedType_5783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 33), 'UndefinedType', False)
            # Processing the call keyword arguments (line 111)
            kwargs_5784 = {}
            # Getting the type of 'isinstance' (line 111)
            isinstance_5781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 111)
            isinstance_call_result_5785 = invoke(stypy.reporting.localization.Localization(__file__, 111, 15), isinstance_5781, *[value_5782, UndefinedType_5783], **kwargs_5784)
            
            # Testing if the type of an if condition is none (line 111)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 111, 12), isinstance_call_result_5785):
                pass
            else:
                
                # Testing the type of an if condition (line 111)
                if_condition_5786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 12), isinstance_call_result_5785)
                # Assigning a type to the variable 'if_condition_5786' (line 111)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'if_condition_5786', if_condition_5786)
                # SSA begins for if statement (line 111)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 112):
                
                # Assigning a Call to a Subscript (line 112):
                
                # Call to TypeError(...): (line 112)
                # Processing the call arguments (line 112)
                # Getting the type of 'localization' (line 112)
                localization_5788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 46), 'localization', False)
                
                # Call to format(...): (line 113)
                # Processing the call arguments (line 113)
                # Getting the type of 'call_description' (line 113)
                call_description_5791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 96), 'call_description', False)
                # Getting the type of 'key' (line 114)
                key_5792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 96), 'key', False)
                # Processing the call keyword arguments (line 113)
                kwargs_5793 = {}
                str_5789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 46), 'str', '{0}: Keyword argument {1} is not defined')
                # Obtaining the member 'format' of a type (line 113)
                format_5790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 46), str_5789, 'format')
                # Calling format(args, kwargs) (line 113)
                format_call_result_5794 = invoke(stypy.reporting.localization.Localization(__file__, 113, 46), format_5790, *[call_description_5791, key_5792], **kwargs_5793)
                
                # Processing the call keyword arguments (line 112)
                kwargs_5795 = {}
                # Getting the type of 'TypeError' (line 112)
                TypeError_5787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 36), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 112)
                TypeError_call_result_5796 = invoke(stypy.reporting.localization.Localization(__file__, 112, 36), TypeError_5787, *[localization_5788, format_call_result_5794], **kwargs_5795)
                
                # Getting the type of 'final_kwargs' (line 112)
                final_kwargs_5797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'final_kwargs')
                # Getting the type of 'key' (line 112)
                key_5798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 29), 'key')
                # Storing an element on a container (line 112)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 16), final_kwargs_5797, (key_5798, TypeError_call_result_5796))
                # SSA join for if statement (line 111)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 102)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Subscript (line 116):
        
        # Assigning a Name to a Subscript (line 116):
        # Getting the type of 'value' (line 116)
        value_5799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 28), 'value')
        # Getting the type of 'final_kwargs' (line 116)
        final_kwargs_5800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'final_kwargs')
        # Getting the type of 'key' (line 116)
        key_5801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 21), 'key')
        # Storing an element on a container (line 116)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 8), final_kwargs_5800, (key_5801, value_5799))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 118)
    tuple_5802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 118)
    # Adding element type (line 118)
    
    # Call to tuple(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'arg_types_list' (line 118)
    arg_types_list_5804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), 'arg_types_list', False)
    # Processing the call keyword arguments (line 118)
    kwargs_5805 = {}
    # Getting the type of 'tuple' (line 118)
    tuple_5803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 11), 'tuple', False)
    # Calling tuple(args, kwargs) (line 118)
    tuple_call_result_5806 = invoke(stypy.reporting.localization.Localization(__file__, 118, 11), tuple_5803, *[arg_types_list_5804], **kwargs_5805)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 11), tuple_5802, tuple_call_result_5806)
    # Adding element type (line 118)
    # Getting the type of 'final_kwargs' (line 118)
    final_kwargs_5807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 34), 'final_kwargs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 11), tuple_5802, final_kwargs_5807)
    
    # Assigning a type to the variable 'stypy_return_type' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type', tuple_5802)
    
    # ################# End of 'check_undefined_type_within_parameters(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_undefined_type_within_parameters' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_5808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5808)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_undefined_type_within_parameters'
    return stypy_return_type_5808

# Assigning a type to the variable 'check_undefined_type_within_parameters' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'check_undefined_type_within_parameters', check_undefined_type_within_parameters)

@norecursion
def __type_error_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__type_error_str'
    module_type_store = module_type_store.open_function_context('__type_error_str', 124, 0, False)
    
    # Passed parameters checking function
    __type_error_str.stypy_localization = localization
    __type_error_str.stypy_type_of_self = None
    __type_error_str.stypy_type_store = module_type_store
    __type_error_str.stypy_function_name = '__type_error_str'
    __type_error_str.stypy_param_names_list = ['arg']
    __type_error_str.stypy_varargs_param_name = None
    __type_error_str.stypy_kwargs_param_name = None
    __type_error_str.stypy_call_defaults = defaults
    __type_error_str.stypy_call_varargs = varargs
    __type_error_str.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__type_error_str', ['arg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__type_error_str', localization, ['arg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__type_error_str(...)' code ##################

    str_5809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, (-1)), 'str', '\n    Helper function of the following one.\n    If arg is a type error, this avoids printing all the TypeError information and only prints the name. This is\n    convenient when pretty-printing calls and its passed parameters to report errors, because if we print the full\n    error information (the same one that is returned by stypy at the end) the message will be unclear.\n    :param arg:\n    :return:\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 133)
    # Getting the type of 'TypeError' (line 133)
    TypeError_5810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'TypeError')
    # Getting the type of 'arg' (line 133)
    arg_5811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 18), 'arg')
    
    (may_be_5812, more_types_in_union_5813) = may_be_subtype(TypeError_5810, arg_5811)

    if may_be_5812:

        if more_types_in_union_5813:
            # Runtime conditional SSA (line 133)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'arg' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'arg', remove_not_subtype_from_union(arg_5811, TypeError))
        str_5814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 15), 'str', 'TypeError')
        # Assigning a type to the variable 'stypy_return_type' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'stypy_return_type', str_5814)

        if more_types_in_union_5813:
            # Runtime conditional SSA for else branch (line 133)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_5812) or more_types_in_union_5813):
        # Assigning a type to the variable 'arg' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'arg', remove_subtype_from_union(arg_5811, TypeError))
        
        # Call to str(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'arg' (line 136)
        arg_5816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'arg', False)
        # Processing the call keyword arguments (line 136)
        kwargs_5817 = {}
        # Getting the type of 'str' (line 136)
        str_5815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 15), 'str', False)
        # Calling str(args, kwargs) (line 136)
        str_call_result_5818 = invoke(stypy.reporting.localization.Localization(__file__, 136, 15), str_5815, *[arg_5816], **kwargs_5817)
        
        # Assigning a type to the variable 'stypy_return_type' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'stypy_return_type', str_call_result_5818)

        if (may_be_5812 and more_types_in_union_5813):
            # SSA join for if statement (line 133)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '__type_error_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__type_error_str' in the type store
    # Getting the type of 'stypy_return_type' (line 124)
    stypy_return_type_5819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5819)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__type_error_str'
    return stypy_return_type_5819

# Assigning a type to the variable '__type_error_str' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), '__type_error_str', __type_error_str)

@norecursion
def __format_type_list(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__format_type_list'
    module_type_store = module_type_store.open_function_context('__format_type_list', 139, 0, False)
    
    # Passed parameters checking function
    __format_type_list.stypy_localization = localization
    __format_type_list.stypy_type_of_self = None
    __format_type_list.stypy_type_store = module_type_store
    __format_type_list.stypy_function_name = '__format_type_list'
    __format_type_list.stypy_param_names_list = []
    __format_type_list.stypy_varargs_param_name = 'arg_types'
    __format_type_list.stypy_kwargs_param_name = 'kwargs_types'
    __format_type_list.stypy_call_defaults = defaults
    __format_type_list.stypy_call_varargs = varargs
    __format_type_list.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__format_type_list', [], 'arg_types', 'kwargs_types', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__format_type_list', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__format_type_list(...)' code ##################

    str_5820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, (-1)), 'str', '\n    Pretty-print passed parameter list\n    :param arg_types:\n    :param kwargs_types:\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 146):
    
    # Assigning a Call to a Name (line 146):
    
    # Call to map(...): (line 146)
    # Processing the call arguments (line 146)

    @norecursion
    def _stypy_temp_lambda_11(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_11'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_11', 146, 23, True)
        # Passed parameters checking function
        _stypy_temp_lambda_11.stypy_localization = localization
        _stypy_temp_lambda_11.stypy_type_of_self = None
        _stypy_temp_lambda_11.stypy_type_store = module_type_store
        _stypy_temp_lambda_11.stypy_function_name = '_stypy_temp_lambda_11'
        _stypy_temp_lambda_11.stypy_param_names_list = ['elem']
        _stypy_temp_lambda_11.stypy_varargs_param_name = None
        _stypy_temp_lambda_11.stypy_kwargs_param_name = None
        _stypy_temp_lambda_11.stypy_call_defaults = defaults
        _stypy_temp_lambda_11.stypy_call_varargs = varargs
        _stypy_temp_lambda_11.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_11', ['elem'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_11', ['elem'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to __type_error_str(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'elem' (line 146)
        elem_5823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 53), 'elem', False)
        # Processing the call keyword arguments (line 146)
        kwargs_5824 = {}
        # Getting the type of '__type_error_str' (line 146)
        type_error_str_5822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 36), '__type_error_str', False)
        # Calling __type_error_str(args, kwargs) (line 146)
        type_error_str_call_result_5825 = invoke(stypy.reporting.localization.Localization(__file__, 146, 36), type_error_str_5822, *[elem_5823], **kwargs_5824)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 23), 'stypy_return_type', type_error_str_call_result_5825)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_11' in the type store
        # Getting the type of 'stypy_return_type' (line 146)
        stypy_return_type_5826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 23), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5826)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_11'
        return stypy_return_type_5826

    # Assigning a type to the variable '_stypy_temp_lambda_11' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 23), '_stypy_temp_lambda_11', _stypy_temp_lambda_11)
    # Getting the type of '_stypy_temp_lambda_11' (line 146)
    _stypy_temp_lambda_11_5827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 23), '_stypy_temp_lambda_11')
    
    # Obtaining the type of the subscript
    int_5828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 70), 'int')
    # Getting the type of 'arg_types' (line 146)
    arg_types_5829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 60), 'arg_types', False)
    # Obtaining the member '__getitem__' of a type (line 146)
    getitem___5830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 60), arg_types_5829, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 146)
    subscript_call_result_5831 = invoke(stypy.reporting.localization.Localization(__file__, 146, 60), getitem___5830, int_5828)
    
    # Processing the call keyword arguments (line 146)
    kwargs_5832 = {}
    # Getting the type of 'map' (line 146)
    map_5821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 19), 'map', False)
    # Calling map(args, kwargs) (line 146)
    map_call_result_5833 = invoke(stypy.reporting.localization.Localization(__file__, 146, 19), map_5821, *[_stypy_temp_lambda_11_5827, subscript_call_result_5831], **kwargs_5832)
    
    # Assigning a type to the variable 'arg_str_list' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'arg_str_list', map_call_result_5833)
    
    # Assigning a Str to a Name (line 147):
    
    # Assigning a Str to a Name (line 147):
    str_5834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 14), 'str', '')
    # Assigning a type to the variable 'arg_str' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'arg_str', str_5834)
    
    # Getting the type of 'arg_str_list' (line 148)
    arg_str_list_5835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'arg_str_list')
    # Assigning a type to the variable 'arg_str_list_5835' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'arg_str_list_5835', arg_str_list_5835)
    # Testing if the for loop is going to be iterated (line 148)
    # Testing the type of a for loop iterable (line 148)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 148, 4), arg_str_list_5835)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 148, 4), arg_str_list_5835):
        # Getting the type of the for loop variable (line 148)
        for_loop_var_5836 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 148, 4), arg_str_list_5835)
        # Assigning a type to the variable 'arg' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'arg', for_loop_var_5836)
        # SSA begins for a for statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'arg_str' (line 149)
        arg_str_5837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'arg_str')
        # Getting the type of 'arg' (line 149)
        arg_5838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 19), 'arg')
        str_5839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 25), 'str', ', ')
        # Applying the binary operator '+' (line 149)
        result_add_5840 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 19), '+', arg_5838, str_5839)
        
        # Applying the binary operator '+=' (line 149)
        result_iadd_5841 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 8), '+=', arg_str_5837, result_add_5840)
        # Assigning a type to the variable 'arg_str' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'arg_str', result_iadd_5841)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    
    # Call to len(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'arg_str' (line 151)
    arg_str_5843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'arg_str', False)
    # Processing the call keyword arguments (line 151)
    kwargs_5844 = {}
    # Getting the type of 'len' (line 151)
    len_5842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 7), 'len', False)
    # Calling len(args, kwargs) (line 151)
    len_call_result_5845 = invoke(stypy.reporting.localization.Localization(__file__, 151, 7), len_5842, *[arg_str_5843], **kwargs_5844)
    
    int_5846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 22), 'int')
    # Applying the binary operator '>' (line 151)
    result_gt_5847 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 7), '>', len_call_result_5845, int_5846)
    
    # Testing if the type of an if condition is none (line 151)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 151, 4), result_gt_5847):
        pass
    else:
        
        # Testing the type of an if condition (line 151)
        if_condition_5848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 4), result_gt_5847)
        # Assigning a type to the variable 'if_condition_5848' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'if_condition_5848', if_condition_5848)
        # SSA begins for if statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 152):
        
        # Assigning a Subscript to a Name (line 152):
        
        # Obtaining the type of the subscript
        int_5849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 27), 'int')
        slice_5850 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 152, 18), None, int_5849, None)
        # Getting the type of 'arg_str' (line 152)
        arg_str_5851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'arg_str')
        # Obtaining the member '__getitem__' of a type (line 152)
        getitem___5852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 18), arg_str_5851, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 152)
        subscript_call_result_5853 = invoke(stypy.reporting.localization.Localization(__file__, 152, 18), getitem___5852, slice_5850)
        
        # Assigning a type to the variable 'arg_str' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'arg_str', subscript_call_result_5853)
        # SSA join for if statement (line 151)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 154):
    
    # Assigning a Call to a Name (line 154):
    
    # Call to map(...): (line 154)
    # Processing the call arguments (line 154)

    @norecursion
    def _stypy_temp_lambda_12(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_12'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_12', 154, 25, True)
        # Passed parameters checking function
        _stypy_temp_lambda_12.stypy_localization = localization
        _stypy_temp_lambda_12.stypy_type_of_self = None
        _stypy_temp_lambda_12.stypy_type_store = module_type_store
        _stypy_temp_lambda_12.stypy_function_name = '_stypy_temp_lambda_12'
        _stypy_temp_lambda_12.stypy_param_names_list = ['elem']
        _stypy_temp_lambda_12.stypy_varargs_param_name = None
        _stypy_temp_lambda_12.stypy_kwargs_param_name = None
        _stypy_temp_lambda_12.stypy_call_defaults = defaults
        _stypy_temp_lambda_12.stypy_call_varargs = varargs
        _stypy_temp_lambda_12.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_12', ['elem'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_12', ['elem'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to __type_error_str(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 'elem' (line 154)
        elem_5856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 55), 'elem', False)
        # Processing the call keyword arguments (line 154)
        kwargs_5857 = {}
        # Getting the type of '__type_error_str' (line 154)
        type_error_str_5855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 38), '__type_error_str', False)
        # Calling __type_error_str(args, kwargs) (line 154)
        type_error_str_call_result_5858 = invoke(stypy.reporting.localization.Localization(__file__, 154, 38), type_error_str_5855, *[elem_5856], **kwargs_5857)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 25), 'stypy_return_type', type_error_str_call_result_5858)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_12' in the type store
        # Getting the type of 'stypy_return_type' (line 154)
        stypy_return_type_5859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 25), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5859)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_12'
        return stypy_return_type_5859

    # Assigning a type to the variable '_stypy_temp_lambda_12' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 25), '_stypy_temp_lambda_12', _stypy_temp_lambda_12)
    # Getting the type of '_stypy_temp_lambda_12' (line 154)
    _stypy_temp_lambda_12_5860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 25), '_stypy_temp_lambda_12')
    # Getting the type of 'kwargs_types' (line 154)
    kwargs_types_5861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 62), 'kwargs_types', False)
    # Processing the call keyword arguments (line 154)
    kwargs_5862 = {}
    # Getting the type of 'map' (line 154)
    map_5854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 21), 'map', False)
    # Calling map(args, kwargs) (line 154)
    map_call_result_5863 = invoke(stypy.reporting.localization.Localization(__file__, 154, 21), map_5854, *[_stypy_temp_lambda_12_5860, kwargs_types_5861], **kwargs_5862)
    
    # Assigning a type to the variable 'kwarg_str_list' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'kwarg_str_list', map_call_result_5863)
    
    # Assigning a Str to a Name (line 155):
    
    # Assigning a Str to a Name (line 155):
    str_5864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 16), 'str', '')
    # Assigning a type to the variable 'kwarg_str' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'kwarg_str', str_5864)
    
    # Getting the type of 'kwarg_str_list' (line 156)
    kwarg_str_list_5865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'kwarg_str_list')
    # Assigning a type to the variable 'kwarg_str_list_5865' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'kwarg_str_list_5865', kwarg_str_list_5865)
    # Testing if the for loop is going to be iterated (line 156)
    # Testing the type of a for loop iterable (line 156)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 156, 4), kwarg_str_list_5865)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 156, 4), kwarg_str_list_5865):
        # Getting the type of the for loop variable (line 156)
        for_loop_var_5866 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 156, 4), kwarg_str_list_5865)
        # Assigning a type to the variable 'arg' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'arg', for_loop_var_5866)
        # SSA begins for a for statement (line 156)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'kwarg_str' (line 157)
        kwarg_str_5867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'kwarg_str')
        # Getting the type of 'arg' (line 157)
        arg_5868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 21), 'arg')
        str_5869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 27), 'str', ', ')
        # Applying the binary operator '+' (line 157)
        result_add_5870 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 21), '+', arg_5868, str_5869)
        
        # Applying the binary operator '+=' (line 157)
        result_iadd_5871 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 8), '+=', kwarg_str_5867, result_add_5870)
        # Assigning a type to the variable 'kwarg_str' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'kwarg_str', result_iadd_5871)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    
    # Call to len(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'kwarg_str' (line 159)
    kwarg_str_5873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'kwarg_str', False)
    # Processing the call keyword arguments (line 159)
    kwargs_5874 = {}
    # Getting the type of 'len' (line 159)
    len_5872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 7), 'len', False)
    # Calling len(args, kwargs) (line 159)
    len_call_result_5875 = invoke(stypy.reporting.localization.Localization(__file__, 159, 7), len_5872, *[kwarg_str_5873], **kwargs_5874)
    
    int_5876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 24), 'int')
    # Applying the binary operator '>' (line 159)
    result_gt_5877 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 7), '>', len_call_result_5875, int_5876)
    
    # Testing if the type of an if condition is none (line 159)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 159, 4), result_gt_5877):
        pass
    else:
        
        # Testing the type of an if condition (line 159)
        if_condition_5878 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 4), result_gt_5877)
        # Assigning a type to the variable 'if_condition_5878' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'if_condition_5878', if_condition_5878)
        # SSA begins for if statement (line 159)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 160):
        
        # Assigning a Subscript to a Name (line 160):
        
        # Obtaining the type of the subscript
        int_5879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 31), 'int')
        slice_5880 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 160, 20), None, int_5879, None)
        # Getting the type of 'kwarg_str' (line 160)
        kwarg_str_5881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'kwarg_str')
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___5882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 20), kwarg_str_5881, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_5883 = invoke(stypy.reporting.localization.Localization(__file__, 160, 20), getitem___5882, slice_5880)
        
        # Assigning a type to the variable 'kwarg_str' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'kwarg_str', subscript_call_result_5883)
        
        # Assigning a BinOp to a Name (line 161):
        
        # Assigning a BinOp to a Name (line 161):
        str_5884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 20), 'str', '{')
        # Getting the type of 'kwarg_str' (line 161)
        kwarg_str_5885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 26), 'kwarg_str')
        # Applying the binary operator '+' (line 161)
        result_add_5886 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 20), '+', str_5884, kwarg_str_5885)
        
        str_5887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 38), 'str', '}')
        # Applying the binary operator '+' (line 161)
        result_add_5888 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 36), '+', result_add_5886, str_5887)
        
        # Assigning a type to the variable 'kwarg_str' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'kwarg_str', result_add_5888)
        # SSA join for if statement (line 159)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Obtaining an instance of the builtin type 'tuple' (line 163)
    tuple_5889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 163)
    # Adding element type (line 163)
    # Getting the type of 'arg_str' (line 163)
    arg_str_5890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'arg_str')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 11), tuple_5889, arg_str_5890)
    # Adding element type (line 163)
    # Getting the type of 'kwarg_str' (line 163)
    kwarg_str_5891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'kwarg_str')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 11), tuple_5889, kwarg_str_5891)
    
    # Assigning a type to the variable 'stypy_return_type' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'stypy_return_type', tuple_5889)
    
    # ################# End of '__format_type_list(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__format_type_list' in the type store
    # Getting the type of 'stypy_return_type' (line 139)
    stypy_return_type_5892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5892)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__format_type_list'
    return stypy_return_type_5892

# Assigning a type to the variable '__format_type_list' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), '__format_type_list', __format_type_list)

@norecursion
def __format_callable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__format_callable'
    module_type_store = module_type_store.open_function_context('__format_callable', 166, 0, False)
    
    # Passed parameters checking function
    __format_callable.stypy_localization = localization
    __format_callable.stypy_type_of_self = None
    __format_callable.stypy_type_store = module_type_store
    __format_callable.stypy_function_name = '__format_callable'
    __format_callable.stypy_param_names_list = ['callable_']
    __format_callable.stypy_varargs_param_name = None
    __format_callable.stypy_kwargs_param_name = None
    __format_callable.stypy_call_defaults = defaults
    __format_callable.stypy_call_varargs = varargs
    __format_callable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__format_callable', ['callable_'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__format_callable', localization, ['callable_'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__format_callable(...)' code ##################

    str_5893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, (-1)), 'str', '\n    Pretty-print a callable entity\n    :param callable_:\n    :return:\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 172)
    str_5894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 26), 'str', '__name__')
    # Getting the type of 'callable_' (line 172)
    callable__5895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'callable_')
    
    (may_be_5896, more_types_in_union_5897) = may_provide_member(str_5894, callable__5895)

    if may_be_5896:

        if more_types_in_union_5897:
            # Runtime conditional SSA (line 172)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'callable_' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'callable_', remove_not_member_provider_from_union(callable__5895, '__name__'))
        # Getting the type of 'callable_' (line 173)
        callable__5898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'callable_')
        # Obtaining the member '__name__' of a type (line 173)
        name___5899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 15), callable__5898, '__name__')
        # Assigning a type to the variable 'stypy_return_type' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'stypy_return_type', name___5899)

        if more_types_in_union_5897:
            # Runtime conditional SSA for else branch (line 172)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_5896) or more_types_in_union_5897):
        # Assigning a type to the variable 'callable_' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'callable_', remove_member_provider_from_union(callable__5895, '__name__'))
        
        # Call to str(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'callable_' (line 175)
        callable__5901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 19), 'callable_', False)
        # Processing the call keyword arguments (line 175)
        kwargs_5902 = {}
        # Getting the type of 'str' (line 175)
        str_5900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'str', False)
        # Calling str(args, kwargs) (line 175)
        str_call_result_5903 = invoke(stypy.reporting.localization.Localization(__file__, 175, 15), str_5900, *[callable__5901], **kwargs_5902)
        
        # Assigning a type to the variable 'stypy_return_type' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'stypy_return_type', str_call_result_5903)

        if (may_be_5896 and more_types_in_union_5897):
            # SSA join for if statement (line 172)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of '__format_callable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__format_callable' in the type store
    # Getting the type of 'stypy_return_type' (line 166)
    stypy_return_type_5904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5904)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__format_callable'
    return stypy_return_type_5904

# Assigning a type to the variable '__format_callable' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), '__format_callable', __format_callable)

@norecursion
def format_call(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'format_call'
    module_type_store = module_type_store.open_function_context('format_call', 178, 0, False)
    
    # Passed parameters checking function
    format_call.stypy_localization = localization
    format_call.stypy_type_of_self = None
    format_call.stypy_type_store = module_type_store
    format_call.stypy_function_name = 'format_call'
    format_call.stypy_param_names_list = ['callable_', 'arg_types', 'kwarg_types']
    format_call.stypy_varargs_param_name = None
    format_call.stypy_kwargs_param_name = None
    format_call.stypy_call_defaults = defaults
    format_call.stypy_call_varargs = varargs
    format_call.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'format_call', ['callable_', 'arg_types', 'kwarg_types'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'format_call', localization, ['callable_', 'arg_types', 'kwarg_types'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'format_call(...)' code ##################

    str_5905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, (-1)), 'str', '\n    Pretty-print calls and its passed parameters, for error reporting, using the previously defined functions\n    :param callable_:\n    :param arg_types:\n    :param kwarg_types:\n    :return:\n    ')
    
    # Assigning a Call to a Tuple (line 186):
    
    # Assigning a Call to a Name:
    
    # Call to __format_type_list(...): (line 186)
    # Processing the call arguments (line 186)
    # Getting the type of 'arg_types' (line 186)
    arg_types_5907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 44), 'arg_types', False)
    
    # Call to values(...): (line 186)
    # Processing the call keyword arguments (line 186)
    kwargs_5910 = {}
    # Getting the type of 'kwarg_types' (line 186)
    kwarg_types_5908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 55), 'kwarg_types', False)
    # Obtaining the member 'values' of a type (line 186)
    values_5909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 55), kwarg_types_5908, 'values')
    # Calling values(args, kwargs) (line 186)
    values_call_result_5911 = invoke(stypy.reporting.localization.Localization(__file__, 186, 55), values_5909, *[], **kwargs_5910)
    
    # Processing the call keyword arguments (line 186)
    kwargs_5912 = {}
    # Getting the type of '__format_type_list' (line 186)
    format_type_list_5906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), '__format_type_list', False)
    # Calling __format_type_list(args, kwargs) (line 186)
    format_type_list_call_result_5913 = invoke(stypy.reporting.localization.Localization(__file__, 186, 25), format_type_list_5906, *[arg_types_5907, values_call_result_5911], **kwargs_5912)
    
    # Assigning a type to the variable 'call_assignment_5512' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'call_assignment_5512', format_type_list_call_result_5913)
    
    # Assigning a Call to a Name (line 186):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_5512' (line 186)
    call_assignment_5512_5914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'call_assignment_5512', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_5915 = stypy_get_value_from_tuple(call_assignment_5512_5914, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_5513' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'call_assignment_5513', stypy_get_value_from_tuple_call_result_5915)
    
    # Assigning a Name to a Name (line 186):
    # Getting the type of 'call_assignment_5513' (line 186)
    call_assignment_5513_5916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'call_assignment_5513')
    # Assigning a type to the variable 'arg_str' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'arg_str', call_assignment_5513_5916)
    
    # Assigning a Call to a Name (line 186):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_5512' (line 186)
    call_assignment_5512_5917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'call_assignment_5512', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_5918 = stypy_get_value_from_tuple(call_assignment_5512_5917, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_5514' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'call_assignment_5514', stypy_get_value_from_tuple_call_result_5918)
    
    # Assigning a Name to a Name (line 186):
    # Getting the type of 'call_assignment_5514' (line 186)
    call_assignment_5514_5919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'call_assignment_5514')
    # Assigning a type to the variable 'kwarg_str' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 13), 'kwarg_str', call_assignment_5514_5919)
    
    # Assigning a Call to a Name (line 187):
    
    # Assigning a Call to a Name (line 187):
    
    # Call to __format_callable(...): (line 187)
    # Processing the call arguments (line 187)
    # Getting the type of 'callable_' (line 187)
    callable__5921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 37), 'callable_', False)
    # Processing the call keyword arguments (line 187)
    kwargs_5922 = {}
    # Getting the type of '__format_callable' (line 187)
    format_callable_5920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 19), '__format_callable', False)
    # Calling __format_callable(args, kwargs) (line 187)
    format_callable_call_result_5923 = invoke(stypy.reporting.localization.Localization(__file__, 187, 19), format_callable_5920, *[callable__5921], **kwargs_5922)
    
    # Assigning a type to the variable 'callable_str' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'callable_str', format_callable_call_result_5923)
    
    
    # Call to len(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'kwarg_str' (line 188)
    kwarg_str_5925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'kwarg_str', False)
    # Processing the call keyword arguments (line 188)
    kwargs_5926 = {}
    # Getting the type of 'len' (line 188)
    len_5924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 7), 'len', False)
    # Calling len(args, kwargs) (line 188)
    len_call_result_5927 = invoke(stypy.reporting.localization.Localization(__file__, 188, 7), len_5924, *[kwarg_str_5925], **kwargs_5926)
    
    int_5928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 25), 'int')
    # Applying the binary operator '==' (line 188)
    result_eq_5929 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 7), '==', len_call_result_5927, int_5928)
    
    # Testing if the type of an if condition is none (line 188)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 188, 4), result_eq_5929):
        str_5940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 15), 'str', '\t')
        # Getting the type of 'callable_str' (line 191)
        callable_str_5941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 22), 'callable_str')
        # Applying the binary operator '+' (line 191)
        result_add_5942 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 15), '+', str_5940, callable_str_5941)
        
        str_5943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 37), 'str', '(')
        # Applying the binary operator '+' (line 191)
        result_add_5944 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 35), '+', result_add_5942, str_5943)
        
        # Getting the type of 'arg_str' (line 191)
        arg_str_5945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 43), 'arg_str')
        # Applying the binary operator '+' (line 191)
        result_add_5946 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 41), '+', result_add_5944, arg_str_5945)
        
        str_5947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 53), 'str', ', ')
        # Applying the binary operator '+' (line 191)
        result_add_5948 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 51), '+', result_add_5946, str_5947)
        
        # Getting the type of 'kwarg_str' (line 191)
        kwarg_str_5949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 60), 'kwarg_str')
        # Applying the binary operator '+' (line 191)
        result_add_5950 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 58), '+', result_add_5948, kwarg_str_5949)
        
        str_5951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 72), 'str', ')')
        # Applying the binary operator '+' (line 191)
        result_add_5952 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 70), '+', result_add_5950, str_5951)
        
        # Assigning a type to the variable 'stypy_return_type' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'stypy_return_type', result_add_5952)
    else:
        
        # Testing the type of an if condition (line 188)
        if_condition_5930 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 4), result_eq_5929)
        # Assigning a type to the variable 'if_condition_5930' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'if_condition_5930', if_condition_5930)
        # SSA begins for if statement (line 188)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_5931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 15), 'str', '\t')
        # Getting the type of 'callable_str' (line 189)
        callable_str_5932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 22), 'callable_str')
        # Applying the binary operator '+' (line 189)
        result_add_5933 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 15), '+', str_5931, callable_str_5932)
        
        str_5934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 37), 'str', '(')
        # Applying the binary operator '+' (line 189)
        result_add_5935 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 35), '+', result_add_5933, str_5934)
        
        # Getting the type of 'arg_str' (line 189)
        arg_str_5936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 43), 'arg_str')
        # Applying the binary operator '+' (line 189)
        result_add_5937 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 41), '+', result_add_5935, arg_str_5936)
        
        str_5938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 53), 'str', ')')
        # Applying the binary operator '+' (line 189)
        result_add_5939 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 51), '+', result_add_5937, str_5938)
        
        # Assigning a type to the variable 'stypy_return_type' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'stypy_return_type', result_add_5939)
        # SSA branch for the else part of an if statement (line 188)
        module_type_store.open_ssa_branch('else')
        str_5940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 15), 'str', '\t')
        # Getting the type of 'callable_str' (line 191)
        callable_str_5941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 22), 'callable_str')
        # Applying the binary operator '+' (line 191)
        result_add_5942 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 15), '+', str_5940, callable_str_5941)
        
        str_5943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 37), 'str', '(')
        # Applying the binary operator '+' (line 191)
        result_add_5944 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 35), '+', result_add_5942, str_5943)
        
        # Getting the type of 'arg_str' (line 191)
        arg_str_5945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 43), 'arg_str')
        # Applying the binary operator '+' (line 191)
        result_add_5946 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 41), '+', result_add_5944, arg_str_5945)
        
        str_5947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 53), 'str', ', ')
        # Applying the binary operator '+' (line 191)
        result_add_5948 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 51), '+', result_add_5946, str_5947)
        
        # Getting the type of 'kwarg_str' (line 191)
        kwarg_str_5949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 60), 'kwarg_str')
        # Applying the binary operator '+' (line 191)
        result_add_5950 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 58), '+', result_add_5948, kwarg_str_5949)
        
        str_5951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 72), 'str', ')')
        # Applying the binary operator '+' (line 191)
        result_add_5952 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 70), '+', result_add_5950, str_5951)
        
        # Assigning a type to the variable 'stypy_return_type' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'stypy_return_type', result_add_5952)
        # SSA join for if statement (line 188)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'format_call(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'format_call' in the type store
    # Getting the type of 'stypy_return_type' (line 178)
    stypy_return_type_5953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5953)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'format_call'
    return stypy_return_type_5953

# Assigning a type to the variable 'format_call' (line 178)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'format_call', format_call)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
