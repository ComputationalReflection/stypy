
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import os
2: import sys
3: import inspect
4: import types
5: 
6: from ....python_lib_copy.python_types_copy.type_copy import Type
7: from ....python_lib_copy.python_types_copy.type_inference_copy.union_type_copy import UnionType
8: from ....python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType
9: from ....stypy_parameters_copy import type_inference_file_directory_name, type_data_file_postfix
10: from ....errors_copy.type_error_copy import TypeError
11: from ....type_store_copy import typestore_copy
12: 
13: '''
14: File with functions that are used when unit testing the generated type inference code checking the type inference code
15: type store against the type data file of the checked program
16: '''
17: 
18: 
19: def __filter_reserved_vars(types_):
20:     '''
21:     For the types_ list, eliminates the references to the TypeDataFileWriter class, to not to check this private object
22:     not part of the original program.
23:     :param types_: Type list
24:     :return:
25:     '''
26:     return filter(lambda elem: not 'TypeDataFileWriter' == elem, types_)
27: 
28: 
29: def __equal_types(expected_var, inferred_context_var):
30:     '''
31:     Helper function to check if the types of two vars can be considered equal from a unit testing point of view
32:     :param expected_var: Expected type
33:     :param inferred_context_var: Inferred type
34:     :return: bool
35:     '''
36: 
37:     # Identity equality
38:     if expected_var == inferred_context_var:
39:         return True
40: 
41:     # TypeInferenceProxy or TypeErrors
42:     if isinstance(inferred_context_var, Type):
43:         # Modules
44:         if expected_var is types.ModuleType:
45:             return inspect.ismodule(inferred_context_var.get_python_entity()) or isinstance(inferred_context_var, typestore_copy.TypeStore)
46: 
47:         if expected_var is types.ClassType:
48:             return inspect.isclass(inferred_context_var.get_python_type())
49: 
50:         if expected_var is TypeError:
51:             return isinstance(inferred_context_var, TypeError)
52: 
53:         direct_comp = inferred_context_var.get_python_type() == expected_var
54:         if not direct_comp and isinstance(expected_var, UnionType) and isinstance(inferred_context_var, UnionType):
55:             return len(expected_var.types) == len(inferred_context_var.types)
56:         return direct_comp
57: 
58:     # Functions
59:     if expected_var == types.FunctionType:
60:         return inspect.isfunction(inferred_context_var)
61: 
62:     # Builtin functions
63:     if expected_var == types.BuiltinFunctionType:
64:         return inspect.isfunction(inferred_context_var)
65: 
66:     # Undefined
67:     if isinstance(inferred_context_var, UndefinedType):
68:         return isinstance(expected_var, UndefinedType)
69: 
70:     # Classes
71:     if expected_var is types.ClassType:
72:         return inspect.isclass(inferred_context_var)
73: 
74:     # Tuples
75:     if expected_var is types.TupleType:
76:         return isinstance(inferred_context_var, tuple)
77: 
78:     # Object instances
79:     if expected_var is types.InstanceType:
80:         return type(inferred_context_var) is types.InstanceType
81:         #return inferred_context_var.get_python_type() == types.InstanceType
82: 
83:     return expected_var == inferred_context_var
84: 
85: 
86: def check_type_store(type_store, executed_file, verbose):
87:     '''
88:     This functions picks a type store of the source code of a file, calculate its associated type data file, loads
89:     it and compare variable per variable the type store type of all variables against the one declared in the type
90:     data file, printing found errors
91:     :param type_store: Type store of the program
92:     :param executed_file: File to load the attached type data file
93:     :param verbose: Verbose output? (bool)
94:     :return: 0 (No error), 1 (Type mismatch in at least one variable), 2 (no associated type data file found)
95:     '''
96:     dirname = os.path.dirname(executed_file) + "/" + type_inference_file_directory_name + "/"
97:     filename = executed_file.split("/")[-1].split(".")[0]
98:     sys.path.append(dirname)
99: 
100:     data_file = filename + type_data_file_postfix
101:     result = 0
102: 
103:     try:
104:         data = __import__(data_file)
105: 
106:         expected_types = data.test_types
107: 
108:         for context_name in expected_types:
109:             inferred_context = type_store.get_last_function_context_for(context_name)
110:             expected_vars = expected_types[context_name]
111:             for var in __filter_reserved_vars(expected_vars):
112:                 if not __equal_types(expected_vars[var], inferred_context[var]):
113:                     #if verbose:
114:                     print "Type mismatch for name '{0}' in context '{3}': {1} expected, but {2} found".format(var,
115:                                                                                                                 expected_vars[
116:                                                                                                                     var],
117:                                                                                                                 inferred_context[
118:                                                                                                                     var],
119:                                                                                                                 context_name)
120:                     result = 1  # Error: Inferred types are not the ones that we expected
121:     except Exception as exc:
122:         if verbose:
123:             print "Type checking error: " + str(exc)
124:         return 2  # Error: Data file not found or some error happened during variable testing
125:     finally:
126:         sys.path.remove(dirname)
127: 
128:     if verbose and result == 0:
129:         print "All checks OK"
130: 
131:     return result  # No Error
132: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import os' statement (line 1)
import os

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import sys' statement (line 2)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import inspect' statement (line 3)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import types' statement (line 4)
import types

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy import Type' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')
import_3070 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy')

if (type(import_3070) is not StypyTypeError):

    if (import_3070 != 'pyd_module'):
        __import__(import_3070)
        sys_modules_3071 = sys.modules[import_3070]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', sys_modules_3071.module_type_store, module_type_store, ['Type'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_3071, sys_modules_3071.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy import Type

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', None, module_type_store, ['Type'], [Type])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', import_3070)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.union_type_copy import UnionType' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')
import_3072 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.union_type_copy')

if (type(import_3072) is not StypyTypeError):

    if (import_3072 != 'pyd_module'):
        __import__(import_3072)
        sys_modules_3073 = sys.modules[import_3072]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.union_type_copy', sys_modules_3073.module_type_store, module_type_store, ['UnionType'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_3073, sys_modules_3073.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.union_type_copy import UnionType

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.union_type_copy', None, module_type_store, ['UnionType'], [UnionType])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.union_type_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.union_type_copy', import_3072)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')
import_3074 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy')

if (type(import_3074) is not StypyTypeError):

    if (import_3074 != 'pyd_module'):
        __import__(import_3074)
        sys_modules_3075 = sys.modules[import_3074]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', sys_modules_3075.module_type_store, module_type_store, ['UndefinedType'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_3075, sys_modules_3075.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', None, module_type_store, ['UndefinedType'], [UndefinedType])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', import_3074)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.stypy_parameters_copy import type_inference_file_directory_name, type_data_file_postfix' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')
import_3076 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.stypy_parameters_copy')

if (type(import_3076) is not StypyTypeError):

    if (import_3076 != 'pyd_module'):
        __import__(import_3076)
        sys_modules_3077 = sys.modules[import_3076]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.stypy_parameters_copy', sys_modules_3077.module_type_store, module_type_store, ['type_inference_file_directory_name', 'type_data_file_postfix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_3077, sys_modules_3077.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.stypy_parameters_copy import type_inference_file_directory_name, type_data_file_postfix

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.stypy_parameters_copy', None, module_type_store, ['type_inference_file_directory_name', 'type_data_file_postfix'], [type_inference_file_directory_name, type_data_file_postfix])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.stypy_parameters_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.stypy_parameters_copy', import_3076)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')
import_3078 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy')

if (type(import_3078) is not StypyTypeError):

    if (import_3078 != 'pyd_module'):
        __import__(import_3078)
        sys_modules_3079 = sys.modules[import_3078]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', sys_modules_3079.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_3079, sys_modules_3079.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', import_3078)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy import typestore_copy' statement (line 11)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')
import_3080 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy')

if (type(import_3080) is not StypyTypeError):

    if (import_3080 != 'pyd_module'):
        __import__(import_3080)
        sys_modules_3081 = sys.modules[import_3080]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy', sys_modules_3081.module_type_store, module_type_store, ['typestore_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_3081, sys_modules_3081.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy import typestore_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy', None, module_type_store, ['typestore_copy'], [typestore_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy', import_3080)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')

str_3082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, (-1)), 'str', '\nFile with functions that are used when unit testing the generated type inference code checking the type inference code\ntype store against the type data file of the checked program\n')

@norecursion
def __filter_reserved_vars(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__filter_reserved_vars'
    module_type_store = module_type_store.open_function_context('__filter_reserved_vars', 19, 0, False)
    
    # Passed parameters checking function
    __filter_reserved_vars.stypy_localization = localization
    __filter_reserved_vars.stypy_type_of_self = None
    __filter_reserved_vars.stypy_type_store = module_type_store
    __filter_reserved_vars.stypy_function_name = '__filter_reserved_vars'
    __filter_reserved_vars.stypy_param_names_list = ['types_']
    __filter_reserved_vars.stypy_varargs_param_name = None
    __filter_reserved_vars.stypy_kwargs_param_name = None
    __filter_reserved_vars.stypy_call_defaults = defaults
    __filter_reserved_vars.stypy_call_varargs = varargs
    __filter_reserved_vars.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__filter_reserved_vars', ['types_'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__filter_reserved_vars', localization, ['types_'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__filter_reserved_vars(...)' code ##################

    str_3083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, (-1)), 'str', '\n    For the types_ list, eliminates the references to the TypeDataFileWriter class, to not to check this private object\n    not part of the original program.\n    :param types_: Type list\n    :return:\n    ')
    
    # Call to filter(...): (line 26)
    # Processing the call arguments (line 26)

    @norecursion
    def _stypy_temp_lambda_3(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_3'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_3', 26, 18, True)
        # Passed parameters checking function
        _stypy_temp_lambda_3.stypy_localization = localization
        _stypy_temp_lambda_3.stypy_type_of_self = None
        _stypy_temp_lambda_3.stypy_type_store = module_type_store
        _stypy_temp_lambda_3.stypy_function_name = '_stypy_temp_lambda_3'
        _stypy_temp_lambda_3.stypy_param_names_list = ['elem']
        _stypy_temp_lambda_3.stypy_varargs_param_name = None
        _stypy_temp_lambda_3.stypy_kwargs_param_name = None
        _stypy_temp_lambda_3.stypy_call_defaults = defaults
        _stypy_temp_lambda_3.stypy_call_varargs = varargs
        _stypy_temp_lambda_3.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_3', ['elem'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_3', ['elem'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        
        str_3085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 35), 'str', 'TypeDataFileWriter')
        # Getting the type of 'elem' (line 26)
        elem_3086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 59), 'elem', False)
        # Applying the binary operator '==' (line 26)
        result_eq_3087 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 35), '==', str_3085, elem_3086)
        
        # Applying the 'not' unary operator (line 26)
        result_not__3088 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 31), 'not', result_eq_3087)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'stypy_return_type', result_not__3088)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_3' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_3089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3089)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_3'
        return stypy_return_type_3089

    # Assigning a type to the variable '_stypy_temp_lambda_3' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), '_stypy_temp_lambda_3', _stypy_temp_lambda_3)
    # Getting the type of '_stypy_temp_lambda_3' (line 26)
    _stypy_temp_lambda_3_3090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), '_stypy_temp_lambda_3')
    # Getting the type of 'types_' (line 26)
    types__3091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 65), 'types_', False)
    # Processing the call keyword arguments (line 26)
    kwargs_3092 = {}
    # Getting the type of 'filter' (line 26)
    filter_3084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'filter', False)
    # Calling filter(args, kwargs) (line 26)
    filter_call_result_3093 = invoke(stypy.reporting.localization.Localization(__file__, 26, 11), filter_3084, *[_stypy_temp_lambda_3_3090, types__3091], **kwargs_3092)
    
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type', filter_call_result_3093)
    
    # ################# End of '__filter_reserved_vars(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__filter_reserved_vars' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_3094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3094)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__filter_reserved_vars'
    return stypy_return_type_3094

# Assigning a type to the variable '__filter_reserved_vars' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), '__filter_reserved_vars', __filter_reserved_vars)

@norecursion
def __equal_types(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__equal_types'
    module_type_store = module_type_store.open_function_context('__equal_types', 29, 0, False)
    
    # Passed parameters checking function
    __equal_types.stypy_localization = localization
    __equal_types.stypy_type_of_self = None
    __equal_types.stypy_type_store = module_type_store
    __equal_types.stypy_function_name = '__equal_types'
    __equal_types.stypy_param_names_list = ['expected_var', 'inferred_context_var']
    __equal_types.stypy_varargs_param_name = None
    __equal_types.stypy_kwargs_param_name = None
    __equal_types.stypy_call_defaults = defaults
    __equal_types.stypy_call_varargs = varargs
    __equal_types.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__equal_types', ['expected_var', 'inferred_context_var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__equal_types', localization, ['expected_var', 'inferred_context_var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__equal_types(...)' code ##################

    str_3095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'str', '\n    Helper function to check if the types of two vars can be considered equal from a unit testing point of view\n    :param expected_var: Expected type\n    :param inferred_context_var: Inferred type\n    :return: bool\n    ')
    
    # Getting the type of 'expected_var' (line 38)
    expected_var_3096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 7), 'expected_var')
    # Getting the type of 'inferred_context_var' (line 38)
    inferred_context_var_3097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'inferred_context_var')
    # Applying the binary operator '==' (line 38)
    result_eq_3098 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 7), '==', expected_var_3096, inferred_context_var_3097)
    
    # Testing if the type of an if condition is none (line 38)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 38, 4), result_eq_3098):
        pass
    else:
        
        # Testing the type of an if condition (line 38)
        if_condition_3099 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 4), result_eq_3098)
        # Assigning a type to the variable 'if_condition_3099' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'if_condition_3099', if_condition_3099)
        # SSA begins for if statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 39)
        True_3100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', True_3100)
        # SSA join for if statement (line 38)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to isinstance(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'inferred_context_var' (line 42)
    inferred_context_var_3102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 18), 'inferred_context_var', False)
    # Getting the type of 'Type' (line 42)
    Type_3103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 40), 'Type', False)
    # Processing the call keyword arguments (line 42)
    kwargs_3104 = {}
    # Getting the type of 'isinstance' (line 42)
    isinstance_3101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 42)
    isinstance_call_result_3105 = invoke(stypy.reporting.localization.Localization(__file__, 42, 7), isinstance_3101, *[inferred_context_var_3102, Type_3103], **kwargs_3104)
    
    # Testing if the type of an if condition is none (line 42)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 42, 4), isinstance_call_result_3105):
        pass
    else:
        
        # Testing the type of an if condition (line 42)
        if_condition_3106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 4), isinstance_call_result_3105)
        # Assigning a type to the variable 'if_condition_3106' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'if_condition_3106', if_condition_3106)
        # SSA begins for if statement (line 42)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'expected_var' (line 44)
        expected_var_3107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'expected_var')
        # Getting the type of 'types' (line 44)
        types_3108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'types')
        # Obtaining the member 'ModuleType' of a type (line 44)
        ModuleType_3109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 27), types_3108, 'ModuleType')
        # Applying the binary operator 'is' (line 44)
        result_is__3110 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), 'is', expected_var_3107, ModuleType_3109)
        
        # Testing if the type of an if condition is none (line 44)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 44, 8), result_is__3110):
            pass
        else:
            
            # Testing the type of an if condition (line 44)
            if_condition_3111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 8), result_is__3110)
            # Assigning a type to the variable 'if_condition_3111' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'if_condition_3111', if_condition_3111)
            # SSA begins for if statement (line 44)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Evaluating a boolean operation
            
            # Call to ismodule(...): (line 45)
            # Processing the call arguments (line 45)
            
            # Call to get_python_entity(...): (line 45)
            # Processing the call keyword arguments (line 45)
            kwargs_3116 = {}
            # Getting the type of 'inferred_context_var' (line 45)
            inferred_context_var_3114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 36), 'inferred_context_var', False)
            # Obtaining the member 'get_python_entity' of a type (line 45)
            get_python_entity_3115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 36), inferred_context_var_3114, 'get_python_entity')
            # Calling get_python_entity(args, kwargs) (line 45)
            get_python_entity_call_result_3117 = invoke(stypy.reporting.localization.Localization(__file__, 45, 36), get_python_entity_3115, *[], **kwargs_3116)
            
            # Processing the call keyword arguments (line 45)
            kwargs_3118 = {}
            # Getting the type of 'inspect' (line 45)
            inspect_3112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'inspect', False)
            # Obtaining the member 'ismodule' of a type (line 45)
            ismodule_3113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 19), inspect_3112, 'ismodule')
            # Calling ismodule(args, kwargs) (line 45)
            ismodule_call_result_3119 = invoke(stypy.reporting.localization.Localization(__file__, 45, 19), ismodule_3113, *[get_python_entity_call_result_3117], **kwargs_3118)
            
            
            # Call to isinstance(...): (line 45)
            # Processing the call arguments (line 45)
            # Getting the type of 'inferred_context_var' (line 45)
            inferred_context_var_3121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 92), 'inferred_context_var', False)
            # Getting the type of 'typestore_copy' (line 45)
            typestore_copy_3122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 114), 'typestore_copy', False)
            # Obtaining the member 'TypeStore' of a type (line 45)
            TypeStore_3123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 114), typestore_copy_3122, 'TypeStore')
            # Processing the call keyword arguments (line 45)
            kwargs_3124 = {}
            # Getting the type of 'isinstance' (line 45)
            isinstance_3120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 81), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 45)
            isinstance_call_result_3125 = invoke(stypy.reporting.localization.Localization(__file__, 45, 81), isinstance_3120, *[inferred_context_var_3121, TypeStore_3123], **kwargs_3124)
            
            # Applying the binary operator 'or' (line 45)
            result_or_keyword_3126 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 19), 'or', ismodule_call_result_3119, isinstance_call_result_3125)
            
            # Assigning a type to the variable 'stypy_return_type' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'stypy_return_type', result_or_keyword_3126)
            # SSA join for if statement (line 44)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'expected_var' (line 47)
        expected_var_3127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'expected_var')
        # Getting the type of 'types' (line 47)
        types_3128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'types')
        # Obtaining the member 'ClassType' of a type (line 47)
        ClassType_3129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 27), types_3128, 'ClassType')
        # Applying the binary operator 'is' (line 47)
        result_is__3130 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), 'is', expected_var_3127, ClassType_3129)
        
        # Testing if the type of an if condition is none (line 47)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 47, 8), result_is__3130):
            pass
        else:
            
            # Testing the type of an if condition (line 47)
            if_condition_3131 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 8), result_is__3130)
            # Assigning a type to the variable 'if_condition_3131' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'if_condition_3131', if_condition_3131)
            # SSA begins for if statement (line 47)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to isclass(...): (line 48)
            # Processing the call arguments (line 48)
            
            # Call to get_python_type(...): (line 48)
            # Processing the call keyword arguments (line 48)
            kwargs_3136 = {}
            # Getting the type of 'inferred_context_var' (line 48)
            inferred_context_var_3134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 35), 'inferred_context_var', False)
            # Obtaining the member 'get_python_type' of a type (line 48)
            get_python_type_3135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 35), inferred_context_var_3134, 'get_python_type')
            # Calling get_python_type(args, kwargs) (line 48)
            get_python_type_call_result_3137 = invoke(stypy.reporting.localization.Localization(__file__, 48, 35), get_python_type_3135, *[], **kwargs_3136)
            
            # Processing the call keyword arguments (line 48)
            kwargs_3138 = {}
            # Getting the type of 'inspect' (line 48)
            inspect_3132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'inspect', False)
            # Obtaining the member 'isclass' of a type (line 48)
            isclass_3133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 19), inspect_3132, 'isclass')
            # Calling isclass(args, kwargs) (line 48)
            isclass_call_result_3139 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), isclass_3133, *[get_python_type_call_result_3137], **kwargs_3138)
            
            # Assigning a type to the variable 'stypy_return_type' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'stypy_return_type', isclass_call_result_3139)
            # SSA join for if statement (line 47)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'expected_var' (line 50)
        expected_var_3140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'expected_var')
        # Getting the type of 'TypeError' (line 50)
        TypeError_3141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 27), 'TypeError')
        # Applying the binary operator 'is' (line 50)
        result_is__3142 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 11), 'is', expected_var_3140, TypeError_3141)
        
        # Testing if the type of an if condition is none (line 50)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 50, 8), result_is__3142):
            pass
        else:
            
            # Testing the type of an if condition (line 50)
            if_condition_3143 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 8), result_is__3142)
            # Assigning a type to the variable 'if_condition_3143' (line 50)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'if_condition_3143', if_condition_3143)
            # SSA begins for if statement (line 50)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to isinstance(...): (line 51)
            # Processing the call arguments (line 51)
            # Getting the type of 'inferred_context_var' (line 51)
            inferred_context_var_3145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 30), 'inferred_context_var', False)
            # Getting the type of 'TypeError' (line 51)
            TypeError_3146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 52), 'TypeError', False)
            # Processing the call keyword arguments (line 51)
            kwargs_3147 = {}
            # Getting the type of 'isinstance' (line 51)
            isinstance_3144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 51)
            isinstance_call_result_3148 = invoke(stypy.reporting.localization.Localization(__file__, 51, 19), isinstance_3144, *[inferred_context_var_3145, TypeError_3146], **kwargs_3147)
            
            # Assigning a type to the variable 'stypy_return_type' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'stypy_return_type', isinstance_call_result_3148)
            # SSA join for if statement (line 50)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Compare to a Name (line 53):
        
        
        # Call to get_python_type(...): (line 53)
        # Processing the call keyword arguments (line 53)
        kwargs_3151 = {}
        # Getting the type of 'inferred_context_var' (line 53)
        inferred_context_var_3149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 22), 'inferred_context_var', False)
        # Obtaining the member 'get_python_type' of a type (line 53)
        get_python_type_3150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 22), inferred_context_var_3149, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 53)
        get_python_type_call_result_3152 = invoke(stypy.reporting.localization.Localization(__file__, 53, 22), get_python_type_3150, *[], **kwargs_3151)
        
        # Getting the type of 'expected_var' (line 53)
        expected_var_3153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 64), 'expected_var')
        # Applying the binary operator '==' (line 53)
        result_eq_3154 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 22), '==', get_python_type_call_result_3152, expected_var_3153)
        
        # Assigning a type to the variable 'direct_comp' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'direct_comp', result_eq_3154)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'direct_comp' (line 54)
        direct_comp_3155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'direct_comp')
        # Applying the 'not' unary operator (line 54)
        result_not__3156 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 11), 'not', direct_comp_3155)
        
        
        # Call to isinstance(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'expected_var' (line 54)
        expected_var_3158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 42), 'expected_var', False)
        # Getting the type of 'UnionType' (line 54)
        UnionType_3159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 56), 'UnionType', False)
        # Processing the call keyword arguments (line 54)
        kwargs_3160 = {}
        # Getting the type of 'isinstance' (line 54)
        isinstance_3157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 31), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 54)
        isinstance_call_result_3161 = invoke(stypy.reporting.localization.Localization(__file__, 54, 31), isinstance_3157, *[expected_var_3158, UnionType_3159], **kwargs_3160)
        
        # Applying the binary operator 'and' (line 54)
        result_and_keyword_3162 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 11), 'and', result_not__3156, isinstance_call_result_3161)
        
        # Call to isinstance(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'inferred_context_var' (line 54)
        inferred_context_var_3164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 82), 'inferred_context_var', False)
        # Getting the type of 'UnionType' (line 54)
        UnionType_3165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 104), 'UnionType', False)
        # Processing the call keyword arguments (line 54)
        kwargs_3166 = {}
        # Getting the type of 'isinstance' (line 54)
        isinstance_3163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 71), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 54)
        isinstance_call_result_3167 = invoke(stypy.reporting.localization.Localization(__file__, 54, 71), isinstance_3163, *[inferred_context_var_3164, UnionType_3165], **kwargs_3166)
        
        # Applying the binary operator 'and' (line 54)
        result_and_keyword_3168 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 11), 'and', result_and_keyword_3162, isinstance_call_result_3167)
        
        # Testing if the type of an if condition is none (line 54)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 54, 8), result_and_keyword_3168):
            pass
        else:
            
            # Testing the type of an if condition (line 54)
            if_condition_3169 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 8), result_and_keyword_3168)
            # Assigning a type to the variable 'if_condition_3169' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'if_condition_3169', if_condition_3169)
            # SSA begins for if statement (line 54)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to len(...): (line 55)
            # Processing the call arguments (line 55)
            # Getting the type of 'expected_var' (line 55)
            expected_var_3171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'expected_var', False)
            # Obtaining the member 'types' of a type (line 55)
            types_3172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 23), expected_var_3171, 'types')
            # Processing the call keyword arguments (line 55)
            kwargs_3173 = {}
            # Getting the type of 'len' (line 55)
            len_3170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'len', False)
            # Calling len(args, kwargs) (line 55)
            len_call_result_3174 = invoke(stypy.reporting.localization.Localization(__file__, 55, 19), len_3170, *[types_3172], **kwargs_3173)
            
            
            # Call to len(...): (line 55)
            # Processing the call arguments (line 55)
            # Getting the type of 'inferred_context_var' (line 55)
            inferred_context_var_3176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 50), 'inferred_context_var', False)
            # Obtaining the member 'types' of a type (line 55)
            types_3177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 50), inferred_context_var_3176, 'types')
            # Processing the call keyword arguments (line 55)
            kwargs_3178 = {}
            # Getting the type of 'len' (line 55)
            len_3175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 46), 'len', False)
            # Calling len(args, kwargs) (line 55)
            len_call_result_3179 = invoke(stypy.reporting.localization.Localization(__file__, 55, 46), len_3175, *[types_3177], **kwargs_3178)
            
            # Applying the binary operator '==' (line 55)
            result_eq_3180 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 19), '==', len_call_result_3174, len_call_result_3179)
            
            # Assigning a type to the variable 'stypy_return_type' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'stypy_return_type', result_eq_3180)
            # SSA join for if statement (line 54)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'direct_comp' (line 56)
        direct_comp_3181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'direct_comp')
        # Assigning a type to the variable 'stypy_return_type' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'stypy_return_type', direct_comp_3181)
        # SSA join for if statement (line 42)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'expected_var' (line 59)
    expected_var_3182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 7), 'expected_var')
    # Getting the type of 'types' (line 59)
    types_3183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'types')
    # Obtaining the member 'FunctionType' of a type (line 59)
    FunctionType_3184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 23), types_3183, 'FunctionType')
    # Applying the binary operator '==' (line 59)
    result_eq_3185 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 7), '==', expected_var_3182, FunctionType_3184)
    
    # Testing if the type of an if condition is none (line 59)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 59, 4), result_eq_3185):
        pass
    else:
        
        # Testing the type of an if condition (line 59)
        if_condition_3186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 4), result_eq_3185)
        # Assigning a type to the variable 'if_condition_3186' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'if_condition_3186', if_condition_3186)
        # SSA begins for if statement (line 59)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to isfunction(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'inferred_context_var' (line 60)
        inferred_context_var_3189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'inferred_context_var', False)
        # Processing the call keyword arguments (line 60)
        kwargs_3190 = {}
        # Getting the type of 'inspect' (line 60)
        inspect_3187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'inspect', False)
        # Obtaining the member 'isfunction' of a type (line 60)
        isfunction_3188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 15), inspect_3187, 'isfunction')
        # Calling isfunction(args, kwargs) (line 60)
        isfunction_call_result_3191 = invoke(stypy.reporting.localization.Localization(__file__, 60, 15), isfunction_3188, *[inferred_context_var_3189], **kwargs_3190)
        
        # Assigning a type to the variable 'stypy_return_type' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'stypy_return_type', isfunction_call_result_3191)
        # SSA join for if statement (line 59)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'expected_var' (line 63)
    expected_var_3192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 7), 'expected_var')
    # Getting the type of 'types' (line 63)
    types_3193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 23), 'types')
    # Obtaining the member 'BuiltinFunctionType' of a type (line 63)
    BuiltinFunctionType_3194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 23), types_3193, 'BuiltinFunctionType')
    # Applying the binary operator '==' (line 63)
    result_eq_3195 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 7), '==', expected_var_3192, BuiltinFunctionType_3194)
    
    # Testing if the type of an if condition is none (line 63)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 63, 4), result_eq_3195):
        pass
    else:
        
        # Testing the type of an if condition (line 63)
        if_condition_3196 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 4), result_eq_3195)
        # Assigning a type to the variable 'if_condition_3196' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'if_condition_3196', if_condition_3196)
        # SSA begins for if statement (line 63)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to isfunction(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'inferred_context_var' (line 64)
        inferred_context_var_3199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'inferred_context_var', False)
        # Processing the call keyword arguments (line 64)
        kwargs_3200 = {}
        # Getting the type of 'inspect' (line 64)
        inspect_3197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'inspect', False)
        # Obtaining the member 'isfunction' of a type (line 64)
        isfunction_3198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 15), inspect_3197, 'isfunction')
        # Calling isfunction(args, kwargs) (line 64)
        isfunction_call_result_3201 = invoke(stypy.reporting.localization.Localization(__file__, 64, 15), isfunction_3198, *[inferred_context_var_3199], **kwargs_3200)
        
        # Assigning a type to the variable 'stypy_return_type' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'stypy_return_type', isfunction_call_result_3201)
        # SSA join for if statement (line 63)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to isinstance(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'inferred_context_var' (line 67)
    inferred_context_var_3203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'inferred_context_var', False)
    # Getting the type of 'UndefinedType' (line 67)
    UndefinedType_3204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 40), 'UndefinedType', False)
    # Processing the call keyword arguments (line 67)
    kwargs_3205 = {}
    # Getting the type of 'isinstance' (line 67)
    isinstance_3202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 67)
    isinstance_call_result_3206 = invoke(stypy.reporting.localization.Localization(__file__, 67, 7), isinstance_3202, *[inferred_context_var_3203, UndefinedType_3204], **kwargs_3205)
    
    # Testing if the type of an if condition is none (line 67)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 67, 4), isinstance_call_result_3206):
        pass
    else:
        
        # Testing the type of an if condition (line 67)
        if_condition_3207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 4), isinstance_call_result_3206)
        # Assigning a type to the variable 'if_condition_3207' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'if_condition_3207', if_condition_3207)
        # SSA begins for if statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to isinstance(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'expected_var' (line 68)
        expected_var_3209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 26), 'expected_var', False)
        # Getting the type of 'UndefinedType' (line 68)
        UndefinedType_3210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 40), 'UndefinedType', False)
        # Processing the call keyword arguments (line 68)
        kwargs_3211 = {}
        # Getting the type of 'isinstance' (line 68)
        isinstance_3208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 68)
        isinstance_call_result_3212 = invoke(stypy.reporting.localization.Localization(__file__, 68, 15), isinstance_3208, *[expected_var_3209, UndefinedType_3210], **kwargs_3211)
        
        # Assigning a type to the variable 'stypy_return_type' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'stypy_return_type', isinstance_call_result_3212)
        # SSA join for if statement (line 67)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'expected_var' (line 71)
    expected_var_3213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 7), 'expected_var')
    # Getting the type of 'types' (line 71)
    types_3214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'types')
    # Obtaining the member 'ClassType' of a type (line 71)
    ClassType_3215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 23), types_3214, 'ClassType')
    # Applying the binary operator 'is' (line 71)
    result_is__3216 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 7), 'is', expected_var_3213, ClassType_3215)
    
    # Testing if the type of an if condition is none (line 71)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 71, 4), result_is__3216):
        pass
    else:
        
        # Testing the type of an if condition (line 71)
        if_condition_3217 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 4), result_is__3216)
        # Assigning a type to the variable 'if_condition_3217' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'if_condition_3217', if_condition_3217)
        # SSA begins for if statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to isclass(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'inferred_context_var' (line 72)
        inferred_context_var_3220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 31), 'inferred_context_var', False)
        # Processing the call keyword arguments (line 72)
        kwargs_3221 = {}
        # Getting the type of 'inspect' (line 72)
        inspect_3218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 72)
        isclass_3219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 15), inspect_3218, 'isclass')
        # Calling isclass(args, kwargs) (line 72)
        isclass_call_result_3222 = invoke(stypy.reporting.localization.Localization(__file__, 72, 15), isclass_3219, *[inferred_context_var_3220], **kwargs_3221)
        
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'stypy_return_type', isclass_call_result_3222)
        # SSA join for if statement (line 71)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'expected_var' (line 75)
    expected_var_3223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 7), 'expected_var')
    # Getting the type of 'types' (line 75)
    types_3224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), 'types')
    # Obtaining the member 'TupleType' of a type (line 75)
    TupleType_3225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 23), types_3224, 'TupleType')
    # Applying the binary operator 'is' (line 75)
    result_is__3226 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 7), 'is', expected_var_3223, TupleType_3225)
    
    # Testing if the type of an if condition is none (line 75)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 4), result_is__3226):
        pass
    else:
        
        # Testing the type of an if condition (line 75)
        if_condition_3227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 4), result_is__3226)
        # Assigning a type to the variable 'if_condition_3227' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'if_condition_3227', if_condition_3227)
        # SSA begins for if statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to isinstance(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'inferred_context_var' (line 76)
        inferred_context_var_3229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'inferred_context_var', False)
        # Getting the type of 'tuple' (line 76)
        tuple_3230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 48), 'tuple', False)
        # Processing the call keyword arguments (line 76)
        kwargs_3231 = {}
        # Getting the type of 'isinstance' (line 76)
        isinstance_3228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 76)
        isinstance_call_result_3232 = invoke(stypy.reporting.localization.Localization(__file__, 76, 15), isinstance_3228, *[inferred_context_var_3229, tuple_3230], **kwargs_3231)
        
        # Assigning a type to the variable 'stypy_return_type' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type', isinstance_call_result_3232)
        # SSA join for if statement (line 75)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'expected_var' (line 79)
    expected_var_3233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 7), 'expected_var')
    # Getting the type of 'types' (line 79)
    types_3234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'types')
    # Obtaining the member 'InstanceType' of a type (line 79)
    InstanceType_3235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 23), types_3234, 'InstanceType')
    # Applying the binary operator 'is' (line 79)
    result_is__3236 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 7), 'is', expected_var_3233, InstanceType_3235)
    
    # Testing if the type of an if condition is none (line 79)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 79, 4), result_is__3236):
        pass
    else:
        
        # Testing the type of an if condition (line 79)
        if_condition_3237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 4), result_is__3236)
        # Assigning a type to the variable 'if_condition_3237' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'if_condition_3237', if_condition_3237)
        # SSA begins for if statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to type(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'inferred_context_var' (line 80)
        inferred_context_var_3239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'inferred_context_var', False)
        # Processing the call keyword arguments (line 80)
        kwargs_3240 = {}
        # Getting the type of 'type' (line 80)
        type_3238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'type', False)
        # Calling type(args, kwargs) (line 80)
        type_call_result_3241 = invoke(stypy.reporting.localization.Localization(__file__, 80, 15), type_3238, *[inferred_context_var_3239], **kwargs_3240)
        
        # Getting the type of 'types' (line 80)
        types_3242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 45), 'types')
        # Obtaining the member 'InstanceType' of a type (line 80)
        InstanceType_3243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 45), types_3242, 'InstanceType')
        # Applying the binary operator 'is' (line 80)
        result_is__3244 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 15), 'is', type_call_result_3241, InstanceType_3243)
        
        # Assigning a type to the variable 'stypy_return_type' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'stypy_return_type', result_is__3244)
        # SSA join for if statement (line 79)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'expected_var' (line 83)
    expected_var_3245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'expected_var')
    # Getting the type of 'inferred_context_var' (line 83)
    inferred_context_var_3246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 27), 'inferred_context_var')
    # Applying the binary operator '==' (line 83)
    result_eq_3247 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 11), '==', expected_var_3245, inferred_context_var_3246)
    
    # Assigning a type to the variable 'stypy_return_type' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type', result_eq_3247)
    
    # ################# End of '__equal_types(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__equal_types' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_3248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3248)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__equal_types'
    return stypy_return_type_3248

# Assigning a type to the variable '__equal_types' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), '__equal_types', __equal_types)

@norecursion
def check_type_store(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_type_store'
    module_type_store = module_type_store.open_function_context('check_type_store', 86, 0, False)
    
    # Passed parameters checking function
    check_type_store.stypy_localization = localization
    check_type_store.stypy_type_of_self = None
    check_type_store.stypy_type_store = module_type_store
    check_type_store.stypy_function_name = 'check_type_store'
    check_type_store.stypy_param_names_list = ['type_store', 'executed_file', 'verbose']
    check_type_store.stypy_varargs_param_name = None
    check_type_store.stypy_kwargs_param_name = None
    check_type_store.stypy_call_defaults = defaults
    check_type_store.stypy_call_varargs = varargs
    check_type_store.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_type_store', ['type_store', 'executed_file', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_type_store', localization, ['type_store', 'executed_file', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_type_store(...)' code ##################

    str_3249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'str', '\n    This functions picks a type store of the source code of a file, calculate its associated type data file, loads\n    it and compare variable per variable the type store type of all variables against the one declared in the type\n    data file, printing found errors\n    :param type_store: Type store of the program\n    :param executed_file: File to load the attached type data file\n    :param verbose: Verbose output? (bool)\n    :return: 0 (No error), 1 (Type mismatch in at least one variable), 2 (no associated type data file found)\n    ')
    
    # Assigning a BinOp to a Name (line 96):
    
    # Call to dirname(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'executed_file' (line 96)
    executed_file_3253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 30), 'executed_file', False)
    # Processing the call keyword arguments (line 96)
    kwargs_3254 = {}
    # Getting the type of 'os' (line 96)
    os_3250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 14), 'os', False)
    # Obtaining the member 'path' of a type (line 96)
    path_3251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 14), os_3250, 'path')
    # Obtaining the member 'dirname' of a type (line 96)
    dirname_3252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 14), path_3251, 'dirname')
    # Calling dirname(args, kwargs) (line 96)
    dirname_call_result_3255 = invoke(stypy.reporting.localization.Localization(__file__, 96, 14), dirname_3252, *[executed_file_3253], **kwargs_3254)
    
    str_3256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 47), 'str', '/')
    # Applying the binary operator '+' (line 96)
    result_add_3257 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 14), '+', dirname_call_result_3255, str_3256)
    
    # Getting the type of 'type_inference_file_directory_name' (line 96)
    type_inference_file_directory_name_3258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 53), 'type_inference_file_directory_name')
    # Applying the binary operator '+' (line 96)
    result_add_3259 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 51), '+', result_add_3257, type_inference_file_directory_name_3258)
    
    str_3260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 90), 'str', '/')
    # Applying the binary operator '+' (line 96)
    result_add_3261 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 88), '+', result_add_3259, str_3260)
    
    # Assigning a type to the variable 'dirname' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'dirname', result_add_3261)
    
    # Assigning a Subscript to a Name (line 97):
    
    # Obtaining the type of the subscript
    int_3262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 55), 'int')
    
    # Call to split(...): (line 97)
    # Processing the call arguments (line 97)
    str_3272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 50), 'str', '.')
    # Processing the call keyword arguments (line 97)
    kwargs_3273 = {}
    
    # Obtaining the type of the subscript
    int_3263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 40), 'int')
    
    # Call to split(...): (line 97)
    # Processing the call arguments (line 97)
    str_3266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 35), 'str', '/')
    # Processing the call keyword arguments (line 97)
    kwargs_3267 = {}
    # Getting the type of 'executed_file' (line 97)
    executed_file_3264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'executed_file', False)
    # Obtaining the member 'split' of a type (line 97)
    split_3265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), executed_file_3264, 'split')
    # Calling split(args, kwargs) (line 97)
    split_call_result_3268 = invoke(stypy.reporting.localization.Localization(__file__, 97, 15), split_3265, *[str_3266], **kwargs_3267)
    
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___3269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), split_call_result_3268, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_3270 = invoke(stypy.reporting.localization.Localization(__file__, 97, 15), getitem___3269, int_3263)
    
    # Obtaining the member 'split' of a type (line 97)
    split_3271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), subscript_call_result_3270, 'split')
    # Calling split(args, kwargs) (line 97)
    split_call_result_3274 = invoke(stypy.reporting.localization.Localization(__file__, 97, 15), split_3271, *[str_3272], **kwargs_3273)
    
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___3275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), split_call_result_3274, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_3276 = invoke(stypy.reporting.localization.Localization(__file__, 97, 15), getitem___3275, int_3262)
    
    # Assigning a type to the variable 'filename' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'filename', subscript_call_result_3276)
    
    # Call to append(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'dirname' (line 98)
    dirname_3280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'dirname', False)
    # Processing the call keyword arguments (line 98)
    kwargs_3281 = {}
    # Getting the type of 'sys' (line 98)
    sys_3277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'sys', False)
    # Obtaining the member 'path' of a type (line 98)
    path_3278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), sys_3277, 'path')
    # Obtaining the member 'append' of a type (line 98)
    append_3279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), path_3278, 'append')
    # Calling append(args, kwargs) (line 98)
    append_call_result_3282 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), append_3279, *[dirname_3280], **kwargs_3281)
    
    
    # Assigning a BinOp to a Name (line 100):
    # Getting the type of 'filename' (line 100)
    filename_3283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'filename')
    # Getting the type of 'type_data_file_postfix' (line 100)
    type_data_file_postfix_3284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'type_data_file_postfix')
    # Applying the binary operator '+' (line 100)
    result_add_3285 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 16), '+', filename_3283, type_data_file_postfix_3284)
    
    # Assigning a type to the variable 'data_file' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'data_file', result_add_3285)
    
    # Assigning a Num to a Name (line 101):
    int_3286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 13), 'int')
    # Assigning a type to the variable 'result' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'result', int_3286)
    
    # Try-finally block (line 103)
    
    
    # SSA begins for try-except statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 104):
    
    # Call to __import__(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'data_file' (line 104)
    data_file_3288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 26), 'data_file', False)
    # Processing the call keyword arguments (line 104)
    kwargs_3289 = {}
    # Getting the type of '__import__' (line 104)
    import___3287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), '__import__', False)
    # Calling __import__(args, kwargs) (line 104)
    import___call_result_3290 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), import___3287, *[data_file_3288], **kwargs_3289)
    
    # Assigning a type to the variable 'data' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'data', import___call_result_3290)
    
    # Assigning a Attribute to a Name (line 106):
    # Getting the type of 'data' (line 106)
    data_3291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 25), 'data')
    # Obtaining the member 'test_types' of a type (line 106)
    test_types_3292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 25), data_3291, 'test_types')
    # Assigning a type to the variable 'expected_types' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'expected_types', test_types_3292)
    
    # Getting the type of 'expected_types' (line 108)
    expected_types_3293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 28), 'expected_types')
    # Assigning a type to the variable 'expected_types_3293' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'expected_types_3293', expected_types_3293)
    # Testing if the for loop is going to be iterated (line 108)
    # Testing the type of a for loop iterable (line 108)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 8), expected_types_3293)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 108, 8), expected_types_3293):
        # Getting the type of the for loop variable (line 108)
        for_loop_var_3294 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 8), expected_types_3293)
        # Assigning a type to the variable 'context_name' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'context_name', for_loop_var_3294)
        # SSA begins for a for statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 109):
        
        # Call to get_last_function_context_for(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'context_name' (line 109)
        context_name_3297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 72), 'context_name', False)
        # Processing the call keyword arguments (line 109)
        kwargs_3298 = {}
        # Getting the type of 'type_store' (line 109)
        type_store_3295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 31), 'type_store', False)
        # Obtaining the member 'get_last_function_context_for' of a type (line 109)
        get_last_function_context_for_3296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 31), type_store_3295, 'get_last_function_context_for')
        # Calling get_last_function_context_for(args, kwargs) (line 109)
        get_last_function_context_for_call_result_3299 = invoke(stypy.reporting.localization.Localization(__file__, 109, 31), get_last_function_context_for_3296, *[context_name_3297], **kwargs_3298)
        
        # Assigning a type to the variable 'inferred_context' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'inferred_context', get_last_function_context_for_call_result_3299)
        
        # Assigning a Subscript to a Name (line 110):
        
        # Obtaining the type of the subscript
        # Getting the type of 'context_name' (line 110)
        context_name_3300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 43), 'context_name')
        # Getting the type of 'expected_types' (line 110)
        expected_types_3301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 28), 'expected_types')
        # Obtaining the member '__getitem__' of a type (line 110)
        getitem___3302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 28), expected_types_3301, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 110)
        subscript_call_result_3303 = invoke(stypy.reporting.localization.Localization(__file__, 110, 28), getitem___3302, context_name_3300)
        
        # Assigning a type to the variable 'expected_vars' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'expected_vars', subscript_call_result_3303)
        
        
        # Call to __filter_reserved_vars(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'expected_vars' (line 111)
        expected_vars_3305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 46), 'expected_vars', False)
        # Processing the call keyword arguments (line 111)
        kwargs_3306 = {}
        # Getting the type of '__filter_reserved_vars' (line 111)
        filter_reserved_vars_3304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), '__filter_reserved_vars', False)
        # Calling __filter_reserved_vars(args, kwargs) (line 111)
        filter_reserved_vars_call_result_3307 = invoke(stypy.reporting.localization.Localization(__file__, 111, 23), filter_reserved_vars_3304, *[expected_vars_3305], **kwargs_3306)
        
        # Assigning a type to the variable 'filter_reserved_vars_call_result_3307' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'filter_reserved_vars_call_result_3307', filter_reserved_vars_call_result_3307)
        # Testing if the for loop is going to be iterated (line 111)
        # Testing the type of a for loop iterable (line 111)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 12), filter_reserved_vars_call_result_3307)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 111, 12), filter_reserved_vars_call_result_3307):
            # Getting the type of the for loop variable (line 111)
            for_loop_var_3308 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 12), filter_reserved_vars_call_result_3307)
            # Assigning a type to the variable 'var' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'var', for_loop_var_3308)
            # SSA begins for a for statement (line 111)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to __equal_types(...): (line 112)
            # Processing the call arguments (line 112)
            
            # Obtaining the type of the subscript
            # Getting the type of 'var' (line 112)
            var_3310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 51), 'var', False)
            # Getting the type of 'expected_vars' (line 112)
            expected_vars_3311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 37), 'expected_vars', False)
            # Obtaining the member '__getitem__' of a type (line 112)
            getitem___3312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 37), expected_vars_3311, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 112)
            subscript_call_result_3313 = invoke(stypy.reporting.localization.Localization(__file__, 112, 37), getitem___3312, var_3310)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'var' (line 112)
            var_3314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 74), 'var', False)
            # Getting the type of 'inferred_context' (line 112)
            inferred_context_3315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 57), 'inferred_context', False)
            # Obtaining the member '__getitem__' of a type (line 112)
            getitem___3316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 57), inferred_context_3315, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 112)
            subscript_call_result_3317 = invoke(stypy.reporting.localization.Localization(__file__, 112, 57), getitem___3316, var_3314)
            
            # Processing the call keyword arguments (line 112)
            kwargs_3318 = {}
            # Getting the type of '__equal_types' (line 112)
            equal_types_3309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), '__equal_types', False)
            # Calling __equal_types(args, kwargs) (line 112)
            equal_types_call_result_3319 = invoke(stypy.reporting.localization.Localization(__file__, 112, 23), equal_types_3309, *[subscript_call_result_3313, subscript_call_result_3317], **kwargs_3318)
            
            # Applying the 'not' unary operator (line 112)
            result_not__3320 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 19), 'not', equal_types_call_result_3319)
            
            # Testing if the type of an if condition is none (line 112)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 112, 16), result_not__3320):
                pass
            else:
                
                # Testing the type of an if condition (line 112)
                if_condition_3321 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 16), result_not__3320)
                # Assigning a type to the variable 'if_condition_3321' (line 112)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'if_condition_3321', if_condition_3321)
                # SSA begins for if statement (line 112)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to format(...): (line 114)
                # Processing the call arguments (line 114)
                # Getting the type of 'var' (line 114)
                var_3324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 110), 'var', False)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var' (line 116)
                var_3325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 116), 'var', False)
                # Getting the type of 'expected_vars' (line 115)
                expected_vars_3326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 112), 'expected_vars', False)
                # Obtaining the member '__getitem__' of a type (line 115)
                getitem___3327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 112), expected_vars_3326, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 115)
                subscript_call_result_3328 = invoke(stypy.reporting.localization.Localization(__file__, 115, 112), getitem___3327, var_3325)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var' (line 118)
                var_3329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 116), 'var', False)
                # Getting the type of 'inferred_context' (line 117)
                inferred_context_3330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 112), 'inferred_context', False)
                # Obtaining the member '__getitem__' of a type (line 117)
                getitem___3331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 112), inferred_context_3330, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 117)
                subscript_call_result_3332 = invoke(stypy.reporting.localization.Localization(__file__, 117, 112), getitem___3331, var_3329)
                
                # Getting the type of 'context_name' (line 119)
                context_name_3333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 112), 'context_name', False)
                # Processing the call keyword arguments (line 114)
                kwargs_3334 = {}
                str_3322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 26), 'str', "Type mismatch for name '{0}' in context '{3}': {1} expected, but {2} found")
                # Obtaining the member 'format' of a type (line 114)
                format_3323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 26), str_3322, 'format')
                # Calling format(args, kwargs) (line 114)
                format_call_result_3335 = invoke(stypy.reporting.localization.Localization(__file__, 114, 26), format_3323, *[var_3324, subscript_call_result_3328, subscript_call_result_3332, context_name_3333], **kwargs_3334)
                
                
                # Assigning a Num to a Name (line 120):
                int_3336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 29), 'int')
                # Assigning a type to the variable 'result' (line 120)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'result', int_3336)
                # SSA join for if statement (line 112)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # SSA branch for the except part of a try statement (line 103)
    # SSA branch for the except 'Exception' branch of a try statement (line 103)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 121)
    Exception_3337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'Exception')
    # Assigning a type to the variable 'exc' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'exc', Exception_3337)
    # Getting the type of 'verbose' (line 122)
    verbose_3338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'verbose')
    # Testing if the type of an if condition is none (line 122)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 122, 8), verbose_3338):
        pass
    else:
        
        # Testing the type of an if condition (line 122)
        if_condition_3339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 8), verbose_3338)
        # Assigning a type to the variable 'if_condition_3339' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'if_condition_3339', if_condition_3339)
        # SSA begins for if statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_3340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 18), 'str', 'Type checking error: ')
        
        # Call to str(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'exc' (line 123)
        exc_3342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 48), 'exc', False)
        # Processing the call keyword arguments (line 123)
        kwargs_3343 = {}
        # Getting the type of 'str' (line 123)
        str_3341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 44), 'str', False)
        # Calling str(args, kwargs) (line 123)
        str_call_result_3344 = invoke(stypy.reporting.localization.Localization(__file__, 123, 44), str_3341, *[exc_3342], **kwargs_3343)
        
        # Applying the binary operator '+' (line 123)
        result_add_3345 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 18), '+', str_3340, str_call_result_3344)
        
        # SSA join for if statement (line 122)
        module_type_store = module_type_store.join_ssa_context()
        

    int_3346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'stypy_return_type', int_3346)
    # SSA join for try-except statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 103)
    
    # Call to remove(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'dirname' (line 126)
    dirname_3350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'dirname', False)
    # Processing the call keyword arguments (line 126)
    kwargs_3351 = {}
    # Getting the type of 'sys' (line 126)
    sys_3347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'sys', False)
    # Obtaining the member 'path' of a type (line 126)
    path_3348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), sys_3347, 'path')
    # Obtaining the member 'remove' of a type (line 126)
    remove_3349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), path_3348, 'remove')
    # Calling remove(args, kwargs) (line 126)
    remove_call_result_3352 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), remove_3349, *[dirname_3350], **kwargs_3351)
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'verbose' (line 128)
    verbose_3353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 7), 'verbose')
    
    # Getting the type of 'result' (line 128)
    result_3354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'result')
    int_3355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 29), 'int')
    # Applying the binary operator '==' (line 128)
    result_eq_3356 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 19), '==', result_3354, int_3355)
    
    # Applying the binary operator 'and' (line 128)
    result_and_keyword_3357 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 7), 'and', verbose_3353, result_eq_3356)
    
    # Testing if the type of an if condition is none (line 128)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 128, 4), result_and_keyword_3357):
        pass
    else:
        
        # Testing the type of an if condition (line 128)
        if_condition_3358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 4), result_and_keyword_3357)
        # Assigning a type to the variable 'if_condition_3358' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'if_condition_3358', if_condition_3358)
        # SSA begins for if statement (line 128)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_3359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 14), 'str', 'All checks OK')
        # SSA join for if statement (line 128)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'result' (line 131)
    result_3360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type', result_3360)
    
    # ################# End of 'check_type_store(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_type_store' in the type store
    # Getting the type of 'stypy_return_type' (line 86)
    stypy_return_type_3361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3361)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_type_store'
    return stypy_return_type_3361

# Assigning a type to the variable 'check_type_store' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'check_type_store', check_type_store)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
