
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from stypy_copy import stypy_parameters_copy
2: 
3: 
4: class ModuleLineNumbering:
5:     '''
6:     This is an utility class to put line numbers to source code lines. Numbered source code lines are added to the
7:     beginning of generated type inference programs to improve its readability if the generated source code has to be
8:     reviewed. Functions of this class are also used to report better errors.
9:     '''
10:     file_numbered_code_cache = dict()
11: 
12:     def __init__(self):
13:         pass
14: 
15:     @staticmethod
16:     def clear_cache():
17:         '''
18:         Numbered lines of source files are cached to improve performance. This clears this cache.
19:         :return:
20:         '''
21:         ModuleLineNumbering.file_numbered_code_cache = dict()
22: 
23:     @staticmethod
24:     def __normalize_path_name(path_name):
25:         '''
26:         Convert file paths into a normalized from
27:         :param path_name: File path
28:         :return: Normalized file path
29:         '''
30:         path_name = path_name.replace("\\", "/")
31:         return path_name
32: 
33:     @staticmethod
34:     def __calculate_line_numbers(file_name, module_code):
35:         '''
36:         Utility method to put numbers to the lines of a source code file, caching it once done
37:         :param file_name: Name of the file
38:         :param module_code: Code of the file
39:         :return: str with the original source code, attaching line numbers to it
40:         '''
41:         if file_name in ModuleLineNumbering.file_numbered_code_cache.keys():
42:             return ModuleLineNumbering.file_numbered_code_cache[file_name]
43: 
44:         numbered_original_code_lines = module_code.split('\n')
45: 
46:         number_line = dict()
47:         for i in range(len(numbered_original_code_lines)):
48:             number_line[i + 1] = numbered_original_code_lines[i]
49: 
50:         ModuleLineNumbering.file_numbered_code_cache[
51:             ModuleLineNumbering.__normalize_path_name(file_name)] = number_line
52: 
53:         return number_line
54: 
55:     @staticmethod
56:     def put_line_numbers_to_module_code(file_name, module_code):
57:         '''
58:         Put numbers to the lines of a source code file, caching it once done
59:         :param file_name: Name of the file
60:         :param module_code: Code of the file
61:         :return: str with the original source code, attaching line numbers to it
62:         '''
63:         number_line = ModuleLineNumbering.__calculate_line_numbers(file_name, module_code)
64:         numbered_original_code = ""
65:         for number, code in number_line.items():
66:             numbered_original_code += str(number) + ": " + code + "\n"
67: 
68:         return numbered_original_code
69: 
70:     @staticmethod
71:     def __get_original_source_code_file(file_name):
72:         '''
73:         From a type inference code file name, obtain the original source code file name
74:         :param file_name: File name (of a type inference program)
75:         :return: File name (of a Python program)
76:         '''
77:         if stypy_parameters_copy.type_inference_file_postfix in file_name:
78:             file_name = file_name.replace(stypy_parameters_copy.type_inference_file_postfix, "")
79: 
80:         if stypy_parameters_copy.type_inference_file_directory_name in file_name:
81:             file_name = file_name.replace(stypy_parameters_copy.type_inference_file_directory_name + "/", "")
82: 
83:         return file_name
84: 
85:     @staticmethod
86:     # TODO: Revise this, if the code is not cached, is this returning something?
87:     def get_line_numbered_module_code(file_name):
88:         '''
89:         Get the numbered source code of the passed file name
90:         :param file_name: File name
91:         :return: Numbered source code (str)
92:         '''
93:         normalized_file_name = ModuleLineNumbering.__normalize_path_name(file_name)
94:         normalized_file_name = ModuleLineNumbering.__get_original_source_code_file(normalized_file_name)
95: 
96:         try:
97:             for file in ModuleLineNumbering.file_numbered_code_cache.keys():
98:                 if file in normalized_file_name:
99:                     return ModuleLineNumbering.file_numbered_code_cache[file]
100:         except:
101:             return None
102: 
103:     @staticmethod
104:     def get_line_from_module_code(file_name, line_number):
105:         '''
106:         Get the source code line line_number from the source code of file_name. This is used to report type errors,
107:         when we also include the source line.
108: 
109:         :param file_name: Python src file
110:         :param line_number: Line to get
111:         :return: str (line of source code)
112:         '''
113:         normalized_file_name = ModuleLineNumbering.__normalize_path_name(file_name)
114:         normalized_file_name = ModuleLineNumbering.__get_original_source_code_file(normalized_file_name)
115: 
116:         linenumbers = ModuleLineNumbering.get_line_numbered_module_code(normalized_file_name)
117:         if linenumbers is not None:
118:             try:
119:                 return linenumbers[line_number]
120:             except:
121:                 return None
122:         return None
123: 
124:     @staticmethod
125:     def get_column_from_module_code(file_name, line_number, col_offset):
126:         '''
127:         Calculates the position of col_offset inside the line_number of the file file_name, so we can physically locate
128:          the column within the file to report meaningful errors. This is used then reporting type error, when we also
129:          include the error line source code and the position within the line that has the error.
130:         :param file_name:
131:         :param line_number:
132:         :param col_offset:
133:         :return:
134:         '''
135:         normalized_file_name = ModuleLineNumbering.__normalize_path_name(file_name)
136:         normalized_file_name = ModuleLineNumbering.__get_original_source_code_file(normalized_file_name)
137: 
138:         line = ModuleLineNumbering.get_line_from_module_code(normalized_file_name, line_number)
139:         if line is None:
140:             return None
141: 
142:         blank_line = " " * col_offset + "^"
143: 
144:         return blank_line
145: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from stypy_copy import stypy_parameters_copy' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/reporting_copy/')
import_13920 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy')

if (type(import_13920) is not StypyTypeError):

    if (import_13920 != 'pyd_module'):
        __import__(import_13920)
        sys_modules_13921 = sys.modules[import_13920]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy', sys_modules_13921.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_13921, sys_modules_13921.module_type_store, module_type_store)
    else:
        from stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'stypy_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy', import_13920)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/reporting_copy/')

# Declaration of the 'ModuleLineNumbering' class

class ModuleLineNumbering:
    str_13922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, (-1)), 'str', '\n    This is an utility class to put line numbers to source code lines. Numbered source code lines are added to the\n    beginning of generated type inference programs to improve its readability if the generated source code has to be\n    reviewed. Functions of this class are also used to report better errors.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ModuleLineNumbering.__init__', [], None, None, defaults, varargs, kwargs)

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


    @staticmethod
    @norecursion
    def clear_cache(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clear_cache'
        module_type_store = module_type_store.open_function_context('clear_cache', 15, 4, False)
        
        # Passed parameters checking function
        ModuleLineNumbering.clear_cache.__dict__.__setitem__('stypy_localization', localization)
        ModuleLineNumbering.clear_cache.__dict__.__setitem__('stypy_type_of_self', None)
        ModuleLineNumbering.clear_cache.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleLineNumbering.clear_cache.__dict__.__setitem__('stypy_function_name', 'clear_cache')
        ModuleLineNumbering.clear_cache.__dict__.__setitem__('stypy_param_names_list', [])
        ModuleLineNumbering.clear_cache.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleLineNumbering.clear_cache.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleLineNumbering.clear_cache.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleLineNumbering.clear_cache.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleLineNumbering.clear_cache.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleLineNumbering.clear_cache.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'clear_cache', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clear_cache', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clear_cache(...)' code ##################

        str_13923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, (-1)), 'str', '\n        Numbered lines of source files are cached to improve performance. This clears this cache.\n        :return:\n        ')
        
        # Assigning a Call to a Attribute (line 21):
        
        # Call to dict(...): (line 21)
        # Processing the call keyword arguments (line 21)
        kwargs_13925 = {}
        # Getting the type of 'dict' (line 21)
        dict_13924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 55), 'dict', False)
        # Calling dict(args, kwargs) (line 21)
        dict_call_result_13926 = invoke(stypy.reporting.localization.Localization(__file__, 21, 55), dict_13924, *[], **kwargs_13925)
        
        # Getting the type of 'ModuleLineNumbering' (line 21)
        ModuleLineNumbering_13927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'ModuleLineNumbering')
        # Setting the type of the member 'file_numbered_code_cache' of a type (line 21)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), ModuleLineNumbering_13927, 'file_numbered_code_cache', dict_call_result_13926)
        
        # ################# End of 'clear_cache(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clear_cache' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_13928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13928)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clear_cache'
        return stypy_return_type_13928


    @staticmethod
    @norecursion
    def __normalize_path_name(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__normalize_path_name'
        module_type_store = module_type_store.open_function_context('__normalize_path_name', 23, 4, False)
        
        # Passed parameters checking function
        ModuleLineNumbering.__normalize_path_name.__dict__.__setitem__('stypy_localization', localization)
        ModuleLineNumbering.__normalize_path_name.__dict__.__setitem__('stypy_type_of_self', None)
        ModuleLineNumbering.__normalize_path_name.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleLineNumbering.__normalize_path_name.__dict__.__setitem__('stypy_function_name', '__normalize_path_name')
        ModuleLineNumbering.__normalize_path_name.__dict__.__setitem__('stypy_param_names_list', ['path_name'])
        ModuleLineNumbering.__normalize_path_name.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleLineNumbering.__normalize_path_name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleLineNumbering.__normalize_path_name.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleLineNumbering.__normalize_path_name.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleLineNumbering.__normalize_path_name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleLineNumbering.__normalize_path_name.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, '__normalize_path_name', ['path_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__normalize_path_name', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__normalize_path_name(...)' code ##################

        str_13929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, (-1)), 'str', '\n        Convert file paths into a normalized from\n        :param path_name: File path\n        :return: Normalized file path\n        ')
        
        # Assigning a Call to a Name (line 30):
        
        # Call to replace(...): (line 30)
        # Processing the call arguments (line 30)
        str_13932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 38), 'str', '\\')
        str_13933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 44), 'str', '/')
        # Processing the call keyword arguments (line 30)
        kwargs_13934 = {}
        # Getting the type of 'path_name' (line 30)
        path_name_13930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 20), 'path_name', False)
        # Obtaining the member 'replace' of a type (line 30)
        replace_13931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 20), path_name_13930, 'replace')
        # Calling replace(args, kwargs) (line 30)
        replace_call_result_13935 = invoke(stypy.reporting.localization.Localization(__file__, 30, 20), replace_13931, *[str_13932, str_13933], **kwargs_13934)
        
        # Assigning a type to the variable 'path_name' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'path_name', replace_call_result_13935)
        # Getting the type of 'path_name' (line 31)
        path_name_13936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'path_name')
        # Assigning a type to the variable 'stypy_return_type' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'stypy_return_type', path_name_13936)
        
        # ################# End of '__normalize_path_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__normalize_path_name' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_13937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13937)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__normalize_path_name'
        return stypy_return_type_13937


    @staticmethod
    @norecursion
    def __calculate_line_numbers(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__calculate_line_numbers'
        module_type_store = module_type_store.open_function_context('__calculate_line_numbers', 33, 4, False)
        
        # Passed parameters checking function
        ModuleLineNumbering.__calculate_line_numbers.__dict__.__setitem__('stypy_localization', localization)
        ModuleLineNumbering.__calculate_line_numbers.__dict__.__setitem__('stypy_type_of_self', None)
        ModuleLineNumbering.__calculate_line_numbers.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleLineNumbering.__calculate_line_numbers.__dict__.__setitem__('stypy_function_name', '__calculate_line_numbers')
        ModuleLineNumbering.__calculate_line_numbers.__dict__.__setitem__('stypy_param_names_list', ['file_name', 'module_code'])
        ModuleLineNumbering.__calculate_line_numbers.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleLineNumbering.__calculate_line_numbers.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleLineNumbering.__calculate_line_numbers.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleLineNumbering.__calculate_line_numbers.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleLineNumbering.__calculate_line_numbers.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleLineNumbering.__calculate_line_numbers.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, '__calculate_line_numbers', ['file_name', 'module_code'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__calculate_line_numbers', localization, ['module_code'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__calculate_line_numbers(...)' code ##################

        str_13938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'str', '\n        Utility method to put numbers to the lines of a source code file, caching it once done\n        :param file_name: Name of the file\n        :param module_code: Code of the file\n        :return: str with the original source code, attaching line numbers to it\n        ')
        
        # Getting the type of 'file_name' (line 41)
        file_name_13939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'file_name')
        
        # Call to keys(...): (line 41)
        # Processing the call keyword arguments (line 41)
        kwargs_13943 = {}
        # Getting the type of 'ModuleLineNumbering' (line 41)
        ModuleLineNumbering_13940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), 'ModuleLineNumbering', False)
        # Obtaining the member 'file_numbered_code_cache' of a type (line 41)
        file_numbered_code_cache_13941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 24), ModuleLineNumbering_13940, 'file_numbered_code_cache')
        # Obtaining the member 'keys' of a type (line 41)
        keys_13942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 24), file_numbered_code_cache_13941, 'keys')
        # Calling keys(args, kwargs) (line 41)
        keys_call_result_13944 = invoke(stypy.reporting.localization.Localization(__file__, 41, 24), keys_13942, *[], **kwargs_13943)
        
        # Applying the binary operator 'in' (line 41)
        result_contains_13945 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 11), 'in', file_name_13939, keys_call_result_13944)
        
        # Testing if the type of an if condition is none (line 41)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 41, 8), result_contains_13945):
            pass
        else:
            
            # Testing the type of an if condition (line 41)
            if_condition_13946 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 8), result_contains_13945)
            # Assigning a type to the variable 'if_condition_13946' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'if_condition_13946', if_condition_13946)
            # SSA begins for if statement (line 41)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            # Getting the type of 'file_name' (line 42)
            file_name_13947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 64), 'file_name')
            # Getting the type of 'ModuleLineNumbering' (line 42)
            ModuleLineNumbering_13948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 19), 'ModuleLineNumbering')
            # Obtaining the member 'file_numbered_code_cache' of a type (line 42)
            file_numbered_code_cache_13949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 19), ModuleLineNumbering_13948, 'file_numbered_code_cache')
            # Obtaining the member '__getitem__' of a type (line 42)
            getitem___13950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 19), file_numbered_code_cache_13949, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 42)
            subscript_call_result_13951 = invoke(stypy.reporting.localization.Localization(__file__, 42, 19), getitem___13950, file_name_13947)
            
            # Assigning a type to the variable 'stypy_return_type' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'stypy_return_type', subscript_call_result_13951)
            # SSA join for if statement (line 41)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 44):
        
        # Call to split(...): (line 44)
        # Processing the call arguments (line 44)
        str_13954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 57), 'str', '\n')
        # Processing the call keyword arguments (line 44)
        kwargs_13955 = {}
        # Getting the type of 'module_code' (line 44)
        module_code_13952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 39), 'module_code', False)
        # Obtaining the member 'split' of a type (line 44)
        split_13953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 39), module_code_13952, 'split')
        # Calling split(args, kwargs) (line 44)
        split_call_result_13956 = invoke(stypy.reporting.localization.Localization(__file__, 44, 39), split_13953, *[str_13954], **kwargs_13955)
        
        # Assigning a type to the variable 'numbered_original_code_lines' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'numbered_original_code_lines', split_call_result_13956)
        
        # Assigning a Call to a Name (line 46):
        
        # Call to dict(...): (line 46)
        # Processing the call keyword arguments (line 46)
        kwargs_13958 = {}
        # Getting the type of 'dict' (line 46)
        dict_13957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'dict', False)
        # Calling dict(args, kwargs) (line 46)
        dict_call_result_13959 = invoke(stypy.reporting.localization.Localization(__file__, 46, 22), dict_13957, *[], **kwargs_13958)
        
        # Assigning a type to the variable 'number_line' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'number_line', dict_call_result_13959)
        
        
        # Call to range(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Call to len(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'numbered_original_code_lines' (line 47)
        numbered_original_code_lines_13962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'numbered_original_code_lines', False)
        # Processing the call keyword arguments (line 47)
        kwargs_13963 = {}
        # Getting the type of 'len' (line 47)
        len_13961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'len', False)
        # Calling len(args, kwargs) (line 47)
        len_call_result_13964 = invoke(stypy.reporting.localization.Localization(__file__, 47, 23), len_13961, *[numbered_original_code_lines_13962], **kwargs_13963)
        
        # Processing the call keyword arguments (line 47)
        kwargs_13965 = {}
        # Getting the type of 'range' (line 47)
        range_13960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'range', False)
        # Calling range(args, kwargs) (line 47)
        range_call_result_13966 = invoke(stypy.reporting.localization.Localization(__file__, 47, 17), range_13960, *[len_call_result_13964], **kwargs_13965)
        
        # Assigning a type to the variable 'range_call_result_13966' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'range_call_result_13966', range_call_result_13966)
        # Testing if the for loop is going to be iterated (line 47)
        # Testing the type of a for loop iterable (line 47)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 47, 8), range_call_result_13966)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 47, 8), range_call_result_13966):
            # Getting the type of the for loop variable (line 47)
            for_loop_var_13967 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 47, 8), range_call_result_13966)
            # Assigning a type to the variable 'i' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'i', for_loop_var_13967)
            # SSA begins for a for statement (line 47)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Subscript (line 48):
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 48)
            i_13968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 62), 'i')
            # Getting the type of 'numbered_original_code_lines' (line 48)
            numbered_original_code_lines_13969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 33), 'numbered_original_code_lines')
            # Obtaining the member '__getitem__' of a type (line 48)
            getitem___13970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 33), numbered_original_code_lines_13969, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 48)
            subscript_call_result_13971 = invoke(stypy.reporting.localization.Localization(__file__, 48, 33), getitem___13970, i_13968)
            
            # Getting the type of 'number_line' (line 48)
            number_line_13972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'number_line')
            # Getting the type of 'i' (line 48)
            i_13973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'i')
            int_13974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 28), 'int')
            # Applying the binary operator '+' (line 48)
            result_add_13975 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 24), '+', i_13973, int_13974)
            
            # Storing an element on a container (line 48)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 12), number_line_13972, (result_add_13975, subscript_call_result_13971))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Name to a Subscript (line 50):
        # Getting the type of 'number_line' (line 51)
        number_line_13976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 68), 'number_line')
        # Getting the type of 'ModuleLineNumbering' (line 50)
        ModuleLineNumbering_13977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'ModuleLineNumbering')
        # Obtaining the member 'file_numbered_code_cache' of a type (line 50)
        file_numbered_code_cache_13978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), ModuleLineNumbering_13977, 'file_numbered_code_cache')
        
        # Call to __normalize_path_name(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'file_name' (line 51)
        file_name_13981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 54), 'file_name', False)
        # Processing the call keyword arguments (line 51)
        kwargs_13982 = {}
        # Getting the type of 'ModuleLineNumbering' (line 51)
        ModuleLineNumbering_13979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'ModuleLineNumbering', False)
        # Obtaining the member '__normalize_path_name' of a type (line 51)
        normalize_path_name_13980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), ModuleLineNumbering_13979, '__normalize_path_name')
        # Calling __normalize_path_name(args, kwargs) (line 51)
        normalize_path_name_call_result_13983 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), normalize_path_name_13980, *[file_name_13981], **kwargs_13982)
        
        # Storing an element on a container (line 50)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 8), file_numbered_code_cache_13978, (normalize_path_name_call_result_13983, number_line_13976))
        # Getting the type of 'number_line' (line 53)
        number_line_13984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'number_line')
        # Assigning a type to the variable 'stypy_return_type' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'stypy_return_type', number_line_13984)
        
        # ################# End of '__calculate_line_numbers(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__calculate_line_numbers' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_13985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13985)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__calculate_line_numbers'
        return stypy_return_type_13985


    @staticmethod
    @norecursion
    def put_line_numbers_to_module_code(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'put_line_numbers_to_module_code'
        module_type_store = module_type_store.open_function_context('put_line_numbers_to_module_code', 55, 4, False)
        
        # Passed parameters checking function
        ModuleLineNumbering.put_line_numbers_to_module_code.__dict__.__setitem__('stypy_localization', localization)
        ModuleLineNumbering.put_line_numbers_to_module_code.__dict__.__setitem__('stypy_type_of_self', None)
        ModuleLineNumbering.put_line_numbers_to_module_code.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleLineNumbering.put_line_numbers_to_module_code.__dict__.__setitem__('stypy_function_name', 'put_line_numbers_to_module_code')
        ModuleLineNumbering.put_line_numbers_to_module_code.__dict__.__setitem__('stypy_param_names_list', ['file_name', 'module_code'])
        ModuleLineNumbering.put_line_numbers_to_module_code.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleLineNumbering.put_line_numbers_to_module_code.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleLineNumbering.put_line_numbers_to_module_code.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleLineNumbering.put_line_numbers_to_module_code.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleLineNumbering.put_line_numbers_to_module_code.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleLineNumbering.put_line_numbers_to_module_code.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'put_line_numbers_to_module_code', ['file_name', 'module_code'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'put_line_numbers_to_module_code', localization, ['module_code'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'put_line_numbers_to_module_code(...)' code ##################

        str_13986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'str', '\n        Put numbers to the lines of a source code file, caching it once done\n        :param file_name: Name of the file\n        :param module_code: Code of the file\n        :return: str with the original source code, attaching line numbers to it\n        ')
        
        # Assigning a Call to a Name (line 63):
        
        # Call to __calculate_line_numbers(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'file_name' (line 63)
        file_name_13989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 67), 'file_name', False)
        # Getting the type of 'module_code' (line 63)
        module_code_13990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 78), 'module_code', False)
        # Processing the call keyword arguments (line 63)
        kwargs_13991 = {}
        # Getting the type of 'ModuleLineNumbering' (line 63)
        ModuleLineNumbering_13987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'ModuleLineNumbering', False)
        # Obtaining the member '__calculate_line_numbers' of a type (line 63)
        calculate_line_numbers_13988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 22), ModuleLineNumbering_13987, '__calculate_line_numbers')
        # Calling __calculate_line_numbers(args, kwargs) (line 63)
        calculate_line_numbers_call_result_13992 = invoke(stypy.reporting.localization.Localization(__file__, 63, 22), calculate_line_numbers_13988, *[file_name_13989, module_code_13990], **kwargs_13991)
        
        # Assigning a type to the variable 'number_line' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'number_line', calculate_line_numbers_call_result_13992)
        
        # Assigning a Str to a Name (line 64):
        str_13993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 33), 'str', '')
        # Assigning a type to the variable 'numbered_original_code' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'numbered_original_code', str_13993)
        
        
        # Call to items(...): (line 65)
        # Processing the call keyword arguments (line 65)
        kwargs_13996 = {}
        # Getting the type of 'number_line' (line 65)
        number_line_13994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'number_line', False)
        # Obtaining the member 'items' of a type (line 65)
        items_13995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 28), number_line_13994, 'items')
        # Calling items(args, kwargs) (line 65)
        items_call_result_13997 = invoke(stypy.reporting.localization.Localization(__file__, 65, 28), items_13995, *[], **kwargs_13996)
        
        # Assigning a type to the variable 'items_call_result_13997' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'items_call_result_13997', items_call_result_13997)
        # Testing if the for loop is going to be iterated (line 65)
        # Testing the type of a for loop iterable (line 65)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 65, 8), items_call_result_13997)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 65, 8), items_call_result_13997):
            # Getting the type of the for loop variable (line 65)
            for_loop_var_13998 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 65, 8), items_call_result_13997)
            # Assigning a type to the variable 'number' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'number', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 8), for_loop_var_13998, 2, 0))
            # Assigning a type to the variable 'code' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'code', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 8), for_loop_var_13998, 2, 1))
            # SSA begins for a for statement (line 65)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'numbered_original_code' (line 66)
            numbered_original_code_13999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'numbered_original_code')
            
            # Call to str(...): (line 66)
            # Processing the call arguments (line 66)
            # Getting the type of 'number' (line 66)
            number_14001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 42), 'number', False)
            # Processing the call keyword arguments (line 66)
            kwargs_14002 = {}
            # Getting the type of 'str' (line 66)
            str_14000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'str', False)
            # Calling str(args, kwargs) (line 66)
            str_call_result_14003 = invoke(stypy.reporting.localization.Localization(__file__, 66, 38), str_14000, *[number_14001], **kwargs_14002)
            
            str_14004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 52), 'str', ': ')
            # Applying the binary operator '+' (line 66)
            result_add_14005 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 38), '+', str_call_result_14003, str_14004)
            
            # Getting the type of 'code' (line 66)
            code_14006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 59), 'code')
            # Applying the binary operator '+' (line 66)
            result_add_14007 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 57), '+', result_add_14005, code_14006)
            
            str_14008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 66), 'str', '\n')
            # Applying the binary operator '+' (line 66)
            result_add_14009 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 64), '+', result_add_14007, str_14008)
            
            # Applying the binary operator '+=' (line 66)
            result_iadd_14010 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 12), '+=', numbered_original_code_13999, result_add_14009)
            # Assigning a type to the variable 'numbered_original_code' (line 66)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'numbered_original_code', result_iadd_14010)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'numbered_original_code' (line 68)
        numbered_original_code_14011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'numbered_original_code')
        # Assigning a type to the variable 'stypy_return_type' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'stypy_return_type', numbered_original_code_14011)
        
        # ################# End of 'put_line_numbers_to_module_code(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'put_line_numbers_to_module_code' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_14012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14012)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'put_line_numbers_to_module_code'
        return stypy_return_type_14012


    @staticmethod
    @norecursion
    def __get_original_source_code_file(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__get_original_source_code_file'
        module_type_store = module_type_store.open_function_context('__get_original_source_code_file', 70, 4, False)
        
        # Passed parameters checking function
        ModuleLineNumbering.__get_original_source_code_file.__dict__.__setitem__('stypy_localization', localization)
        ModuleLineNumbering.__get_original_source_code_file.__dict__.__setitem__('stypy_type_of_self', None)
        ModuleLineNumbering.__get_original_source_code_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleLineNumbering.__get_original_source_code_file.__dict__.__setitem__('stypy_function_name', '__get_original_source_code_file')
        ModuleLineNumbering.__get_original_source_code_file.__dict__.__setitem__('stypy_param_names_list', ['file_name'])
        ModuleLineNumbering.__get_original_source_code_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleLineNumbering.__get_original_source_code_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleLineNumbering.__get_original_source_code_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleLineNumbering.__get_original_source_code_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleLineNumbering.__get_original_source_code_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleLineNumbering.__get_original_source_code_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, '__get_original_source_code_file', ['file_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__get_original_source_code_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__get_original_source_code_file(...)' code ##################

        str_14013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, (-1)), 'str', '\n        From a type inference code file name, obtain the original source code file name\n        :param file_name: File name (of a type inference program)\n        :return: File name (of a Python program)\n        ')
        
        # Getting the type of 'stypy_parameters_copy' (line 77)
        stypy_parameters_copy_14014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'stypy_parameters_copy')
        # Obtaining the member 'type_inference_file_postfix' of a type (line 77)
        type_inference_file_postfix_14015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 11), stypy_parameters_copy_14014, 'type_inference_file_postfix')
        # Getting the type of 'file_name' (line 77)
        file_name_14016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 64), 'file_name')
        # Applying the binary operator 'in' (line 77)
        result_contains_14017 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 11), 'in', type_inference_file_postfix_14015, file_name_14016)
        
        # Testing if the type of an if condition is none (line 77)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 77, 8), result_contains_14017):
            pass
        else:
            
            # Testing the type of an if condition (line 77)
            if_condition_14018 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 8), result_contains_14017)
            # Assigning a type to the variable 'if_condition_14018' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'if_condition_14018', if_condition_14018)
            # SSA begins for if statement (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 78):
            
            # Call to replace(...): (line 78)
            # Processing the call arguments (line 78)
            # Getting the type of 'stypy_parameters_copy' (line 78)
            stypy_parameters_copy_14021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 42), 'stypy_parameters_copy', False)
            # Obtaining the member 'type_inference_file_postfix' of a type (line 78)
            type_inference_file_postfix_14022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 42), stypy_parameters_copy_14021, 'type_inference_file_postfix')
            str_14023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 93), 'str', '')
            # Processing the call keyword arguments (line 78)
            kwargs_14024 = {}
            # Getting the type of 'file_name' (line 78)
            file_name_14019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'file_name', False)
            # Obtaining the member 'replace' of a type (line 78)
            replace_14020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 24), file_name_14019, 'replace')
            # Calling replace(args, kwargs) (line 78)
            replace_call_result_14025 = invoke(stypy.reporting.localization.Localization(__file__, 78, 24), replace_14020, *[type_inference_file_postfix_14022, str_14023], **kwargs_14024)
            
            # Assigning a type to the variable 'file_name' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'file_name', replace_call_result_14025)
            # SSA join for if statement (line 77)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'stypy_parameters_copy' (line 80)
        stypy_parameters_copy_14026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'stypy_parameters_copy')
        # Obtaining the member 'type_inference_file_directory_name' of a type (line 80)
        type_inference_file_directory_name_14027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 11), stypy_parameters_copy_14026, 'type_inference_file_directory_name')
        # Getting the type of 'file_name' (line 80)
        file_name_14028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 71), 'file_name')
        # Applying the binary operator 'in' (line 80)
        result_contains_14029 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 11), 'in', type_inference_file_directory_name_14027, file_name_14028)
        
        # Testing if the type of an if condition is none (line 80)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 8), result_contains_14029):
            pass
        else:
            
            # Testing the type of an if condition (line 80)
            if_condition_14030 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 8), result_contains_14029)
            # Assigning a type to the variable 'if_condition_14030' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'if_condition_14030', if_condition_14030)
            # SSA begins for if statement (line 80)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 81):
            
            # Call to replace(...): (line 81)
            # Processing the call arguments (line 81)
            # Getting the type of 'stypy_parameters_copy' (line 81)
            stypy_parameters_copy_14033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 42), 'stypy_parameters_copy', False)
            # Obtaining the member 'type_inference_file_directory_name' of a type (line 81)
            type_inference_file_directory_name_14034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 42), stypy_parameters_copy_14033, 'type_inference_file_directory_name')
            str_14035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 101), 'str', '/')
            # Applying the binary operator '+' (line 81)
            result_add_14036 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 42), '+', type_inference_file_directory_name_14034, str_14035)
            
            str_14037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 106), 'str', '')
            # Processing the call keyword arguments (line 81)
            kwargs_14038 = {}
            # Getting the type of 'file_name' (line 81)
            file_name_14031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 24), 'file_name', False)
            # Obtaining the member 'replace' of a type (line 81)
            replace_14032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 24), file_name_14031, 'replace')
            # Calling replace(args, kwargs) (line 81)
            replace_call_result_14039 = invoke(stypy.reporting.localization.Localization(__file__, 81, 24), replace_14032, *[result_add_14036, str_14037], **kwargs_14038)
            
            # Assigning a type to the variable 'file_name' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'file_name', replace_call_result_14039)
            # SSA join for if statement (line 80)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'file_name' (line 83)
        file_name_14040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'file_name')
        # Assigning a type to the variable 'stypy_return_type' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'stypy_return_type', file_name_14040)
        
        # ################# End of '__get_original_source_code_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_original_source_code_file' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_14041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14041)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_original_source_code_file'
        return stypy_return_type_14041


    @staticmethod
    @norecursion
    def get_line_numbered_module_code(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_line_numbered_module_code'
        module_type_store = module_type_store.open_function_context('get_line_numbered_module_code', 85, 4, False)
        
        # Passed parameters checking function
        ModuleLineNumbering.get_line_numbered_module_code.__dict__.__setitem__('stypy_localization', localization)
        ModuleLineNumbering.get_line_numbered_module_code.__dict__.__setitem__('stypy_type_of_self', None)
        ModuleLineNumbering.get_line_numbered_module_code.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleLineNumbering.get_line_numbered_module_code.__dict__.__setitem__('stypy_function_name', 'get_line_numbered_module_code')
        ModuleLineNumbering.get_line_numbered_module_code.__dict__.__setitem__('stypy_param_names_list', ['file_name'])
        ModuleLineNumbering.get_line_numbered_module_code.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleLineNumbering.get_line_numbered_module_code.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleLineNumbering.get_line_numbered_module_code.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleLineNumbering.get_line_numbered_module_code.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleLineNumbering.get_line_numbered_module_code.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleLineNumbering.get_line_numbered_module_code.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, 'get_line_numbered_module_code', ['file_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_line_numbered_module_code', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_line_numbered_module_code(...)' code ##################

        str_14042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, (-1)), 'str', '\n        Get the numbered source code of the passed file name\n        :param file_name: File name\n        :return: Numbered source code (str)\n        ')
        
        # Assigning a Call to a Name (line 93):
        
        # Call to __normalize_path_name(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'file_name' (line 93)
        file_name_14045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 73), 'file_name', False)
        # Processing the call keyword arguments (line 93)
        kwargs_14046 = {}
        # Getting the type of 'ModuleLineNumbering' (line 93)
        ModuleLineNumbering_14043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'ModuleLineNumbering', False)
        # Obtaining the member '__normalize_path_name' of a type (line 93)
        normalize_path_name_14044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 31), ModuleLineNumbering_14043, '__normalize_path_name')
        # Calling __normalize_path_name(args, kwargs) (line 93)
        normalize_path_name_call_result_14047 = invoke(stypy.reporting.localization.Localization(__file__, 93, 31), normalize_path_name_14044, *[file_name_14045], **kwargs_14046)
        
        # Assigning a type to the variable 'normalized_file_name' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'normalized_file_name', normalize_path_name_call_result_14047)
        
        # Assigning a Call to a Name (line 94):
        
        # Call to __get_original_source_code_file(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'normalized_file_name' (line 94)
        normalized_file_name_14050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 83), 'normalized_file_name', False)
        # Processing the call keyword arguments (line 94)
        kwargs_14051 = {}
        # Getting the type of 'ModuleLineNumbering' (line 94)
        ModuleLineNumbering_14048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 31), 'ModuleLineNumbering', False)
        # Obtaining the member '__get_original_source_code_file' of a type (line 94)
        get_original_source_code_file_14049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 31), ModuleLineNumbering_14048, '__get_original_source_code_file')
        # Calling __get_original_source_code_file(args, kwargs) (line 94)
        get_original_source_code_file_call_result_14052 = invoke(stypy.reporting.localization.Localization(__file__, 94, 31), get_original_source_code_file_14049, *[normalized_file_name_14050], **kwargs_14051)
        
        # Assigning a type to the variable 'normalized_file_name' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'normalized_file_name', get_original_source_code_file_call_result_14052)
        
        
        # SSA begins for try-except statement (line 96)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        
        # Call to keys(...): (line 97)
        # Processing the call keyword arguments (line 97)
        kwargs_14056 = {}
        # Getting the type of 'ModuleLineNumbering' (line 97)
        ModuleLineNumbering_14053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'ModuleLineNumbering', False)
        # Obtaining the member 'file_numbered_code_cache' of a type (line 97)
        file_numbered_code_cache_14054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 24), ModuleLineNumbering_14053, 'file_numbered_code_cache')
        # Obtaining the member 'keys' of a type (line 97)
        keys_14055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 24), file_numbered_code_cache_14054, 'keys')
        # Calling keys(args, kwargs) (line 97)
        keys_call_result_14057 = invoke(stypy.reporting.localization.Localization(__file__, 97, 24), keys_14055, *[], **kwargs_14056)
        
        # Assigning a type to the variable 'keys_call_result_14057' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'keys_call_result_14057', keys_call_result_14057)
        # Testing if the for loop is going to be iterated (line 97)
        # Testing the type of a for loop iterable (line 97)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 97, 12), keys_call_result_14057)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 97, 12), keys_call_result_14057):
            # Getting the type of the for loop variable (line 97)
            for_loop_var_14058 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 97, 12), keys_call_result_14057)
            # Assigning a type to the variable 'file' (line 97)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'file', for_loop_var_14058)
            # SSA begins for a for statement (line 97)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'file' (line 98)
            file_14059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 19), 'file')
            # Getting the type of 'normalized_file_name' (line 98)
            normalized_file_name_14060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'normalized_file_name')
            # Applying the binary operator 'in' (line 98)
            result_contains_14061 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 19), 'in', file_14059, normalized_file_name_14060)
            
            # Testing if the type of an if condition is none (line 98)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 98, 16), result_contains_14061):
                pass
            else:
                
                # Testing the type of an if condition (line 98)
                if_condition_14062 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 16), result_contains_14061)
                # Assigning a type to the variable 'if_condition_14062' (line 98)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'if_condition_14062', if_condition_14062)
                # SSA begins for if statement (line 98)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Obtaining the type of the subscript
                # Getting the type of 'file' (line 99)
                file_14063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 72), 'file')
                # Getting the type of 'ModuleLineNumbering' (line 99)
                ModuleLineNumbering_14064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 27), 'ModuleLineNumbering')
                # Obtaining the member 'file_numbered_code_cache' of a type (line 99)
                file_numbered_code_cache_14065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 27), ModuleLineNumbering_14064, 'file_numbered_code_cache')
                # Obtaining the member '__getitem__' of a type (line 99)
                getitem___14066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 27), file_numbered_code_cache_14065, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 99)
                subscript_call_result_14067 = invoke(stypy.reporting.localization.Localization(__file__, 99, 27), getitem___14066, file_14063)
                
                # Assigning a type to the variable 'stypy_return_type' (line 99)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'stypy_return_type', subscript_call_result_14067)
                # SSA join for if statement (line 98)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA branch for the except part of a try statement (line 96)
        # SSA branch for the except '<any exception>' branch of a try statement (line 96)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'None' (line 101)
        None_14068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'stypy_return_type', None_14068)
        # SSA join for try-except statement (line 96)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_line_numbered_module_code(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_line_numbered_module_code' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_14069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14069)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_line_numbered_module_code'
        return stypy_return_type_14069


    @staticmethod
    @norecursion
    def get_line_from_module_code(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_line_from_module_code'
        module_type_store = module_type_store.open_function_context('get_line_from_module_code', 103, 4, False)
        
        # Passed parameters checking function
        ModuleLineNumbering.get_line_from_module_code.__dict__.__setitem__('stypy_localization', localization)
        ModuleLineNumbering.get_line_from_module_code.__dict__.__setitem__('stypy_type_of_self', None)
        ModuleLineNumbering.get_line_from_module_code.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleLineNumbering.get_line_from_module_code.__dict__.__setitem__('stypy_function_name', 'get_line_from_module_code')
        ModuleLineNumbering.get_line_from_module_code.__dict__.__setitem__('stypy_param_names_list', ['file_name', 'line_number'])
        ModuleLineNumbering.get_line_from_module_code.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleLineNumbering.get_line_from_module_code.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleLineNumbering.get_line_from_module_code.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleLineNumbering.get_line_from_module_code.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleLineNumbering.get_line_from_module_code.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleLineNumbering.get_line_from_module_code.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, 'get_line_from_module_code', ['file_name', 'line_number'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_line_from_module_code', localization, ['line_number'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_line_from_module_code(...)' code ##################

        str_14070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, (-1)), 'str', '\n        Get the source code line line_number from the source code of file_name. This is used to report type errors,\n        when we also include the source line.\n\n        :param file_name: Python src file\n        :param line_number: Line to get\n        :return: str (line of source code)\n        ')
        
        # Assigning a Call to a Name (line 113):
        
        # Call to __normalize_path_name(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'file_name' (line 113)
        file_name_14073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 73), 'file_name', False)
        # Processing the call keyword arguments (line 113)
        kwargs_14074 = {}
        # Getting the type of 'ModuleLineNumbering' (line 113)
        ModuleLineNumbering_14071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'ModuleLineNumbering', False)
        # Obtaining the member '__normalize_path_name' of a type (line 113)
        normalize_path_name_14072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 31), ModuleLineNumbering_14071, '__normalize_path_name')
        # Calling __normalize_path_name(args, kwargs) (line 113)
        normalize_path_name_call_result_14075 = invoke(stypy.reporting.localization.Localization(__file__, 113, 31), normalize_path_name_14072, *[file_name_14073], **kwargs_14074)
        
        # Assigning a type to the variable 'normalized_file_name' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'normalized_file_name', normalize_path_name_call_result_14075)
        
        # Assigning a Call to a Name (line 114):
        
        # Call to __get_original_source_code_file(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'normalized_file_name' (line 114)
        normalized_file_name_14078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 83), 'normalized_file_name', False)
        # Processing the call keyword arguments (line 114)
        kwargs_14079 = {}
        # Getting the type of 'ModuleLineNumbering' (line 114)
        ModuleLineNumbering_14076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 31), 'ModuleLineNumbering', False)
        # Obtaining the member '__get_original_source_code_file' of a type (line 114)
        get_original_source_code_file_14077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 31), ModuleLineNumbering_14076, '__get_original_source_code_file')
        # Calling __get_original_source_code_file(args, kwargs) (line 114)
        get_original_source_code_file_call_result_14080 = invoke(stypy.reporting.localization.Localization(__file__, 114, 31), get_original_source_code_file_14077, *[normalized_file_name_14078], **kwargs_14079)
        
        # Assigning a type to the variable 'normalized_file_name' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'normalized_file_name', get_original_source_code_file_call_result_14080)
        
        # Assigning a Call to a Name (line 116):
        
        # Call to get_line_numbered_module_code(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'normalized_file_name' (line 116)
        normalized_file_name_14083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 72), 'normalized_file_name', False)
        # Processing the call keyword arguments (line 116)
        kwargs_14084 = {}
        # Getting the type of 'ModuleLineNumbering' (line 116)
        ModuleLineNumbering_14081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 22), 'ModuleLineNumbering', False)
        # Obtaining the member 'get_line_numbered_module_code' of a type (line 116)
        get_line_numbered_module_code_14082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 22), ModuleLineNumbering_14081, 'get_line_numbered_module_code')
        # Calling get_line_numbered_module_code(args, kwargs) (line 116)
        get_line_numbered_module_code_call_result_14085 = invoke(stypy.reporting.localization.Localization(__file__, 116, 22), get_line_numbered_module_code_14082, *[normalized_file_name_14083], **kwargs_14084)
        
        # Assigning a type to the variable 'linenumbers' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'linenumbers', get_line_numbered_module_code_call_result_14085)
        
        # Type idiom detected: calculating its left and rigth part (line 117)
        # Getting the type of 'linenumbers' (line 117)
        linenumbers_14086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'linenumbers')
        # Getting the type of 'None' (line 117)
        None_14087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 30), 'None')
        
        (may_be_14088, more_types_in_union_14089) = may_not_be_none(linenumbers_14086, None_14087)

        if may_be_14088:

            if more_types_in_union_14089:
                # Runtime conditional SSA (line 117)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # SSA begins for try-except statement (line 118)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Obtaining the type of the subscript
            # Getting the type of 'line_number' (line 119)
            line_number_14090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 35), 'line_number')
            # Getting the type of 'linenumbers' (line 119)
            linenumbers_14091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'linenumbers')
            # Obtaining the member '__getitem__' of a type (line 119)
            getitem___14092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 23), linenumbers_14091, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 119)
            subscript_call_result_14093 = invoke(stypy.reporting.localization.Localization(__file__, 119, 23), getitem___14092, line_number_14090)
            
            # Assigning a type to the variable 'stypy_return_type' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'stypy_return_type', subscript_call_result_14093)
            # SSA branch for the except part of a try statement (line 118)
            # SSA branch for the except '<any exception>' branch of a try statement (line 118)
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'None' (line 121)
            None_14094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'stypy_return_type', None_14094)
            # SSA join for try-except statement (line 118)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_14089:
                # SSA join for if statement (line 117)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'None' (line 122)
        None_14095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'stypy_return_type', None_14095)
        
        # ################# End of 'get_line_from_module_code(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_line_from_module_code' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_14096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14096)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_line_from_module_code'
        return stypy_return_type_14096


    @staticmethod
    @norecursion
    def get_column_from_module_code(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_column_from_module_code'
        module_type_store = module_type_store.open_function_context('get_column_from_module_code', 124, 4, False)
        
        # Passed parameters checking function
        ModuleLineNumbering.get_column_from_module_code.__dict__.__setitem__('stypy_localization', localization)
        ModuleLineNumbering.get_column_from_module_code.__dict__.__setitem__('stypy_type_of_self', None)
        ModuleLineNumbering.get_column_from_module_code.__dict__.__setitem__('stypy_type_store', module_type_store)
        ModuleLineNumbering.get_column_from_module_code.__dict__.__setitem__('stypy_function_name', 'get_column_from_module_code')
        ModuleLineNumbering.get_column_from_module_code.__dict__.__setitem__('stypy_param_names_list', ['file_name', 'line_number', 'col_offset'])
        ModuleLineNumbering.get_column_from_module_code.__dict__.__setitem__('stypy_varargs_param_name', None)
        ModuleLineNumbering.get_column_from_module_code.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ModuleLineNumbering.get_column_from_module_code.__dict__.__setitem__('stypy_call_defaults', defaults)
        ModuleLineNumbering.get_column_from_module_code.__dict__.__setitem__('stypy_call_varargs', varargs)
        ModuleLineNumbering.get_column_from_module_code.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ModuleLineNumbering.get_column_from_module_code.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, None, module_type_store, 'get_column_from_module_code', ['file_name', 'line_number', 'col_offset'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_column_from_module_code', localization, ['line_number', 'col_offset'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_column_from_module_code(...)' code ##################

        str_14097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, (-1)), 'str', '\n        Calculates the position of col_offset inside the line_number of the file file_name, so we can physically locate\n         the column within the file to report meaningful errors. This is used then reporting type error, when we also\n         include the error line source code and the position within the line that has the error.\n        :param file_name:\n        :param line_number:\n        :param col_offset:\n        :return:\n        ')
        
        # Assigning a Call to a Name (line 135):
        
        # Call to __normalize_path_name(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'file_name' (line 135)
        file_name_14100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 73), 'file_name', False)
        # Processing the call keyword arguments (line 135)
        kwargs_14101 = {}
        # Getting the type of 'ModuleLineNumbering' (line 135)
        ModuleLineNumbering_14098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 31), 'ModuleLineNumbering', False)
        # Obtaining the member '__normalize_path_name' of a type (line 135)
        normalize_path_name_14099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 31), ModuleLineNumbering_14098, '__normalize_path_name')
        # Calling __normalize_path_name(args, kwargs) (line 135)
        normalize_path_name_call_result_14102 = invoke(stypy.reporting.localization.Localization(__file__, 135, 31), normalize_path_name_14099, *[file_name_14100], **kwargs_14101)
        
        # Assigning a type to the variable 'normalized_file_name' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'normalized_file_name', normalize_path_name_call_result_14102)
        
        # Assigning a Call to a Name (line 136):
        
        # Call to __get_original_source_code_file(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'normalized_file_name' (line 136)
        normalized_file_name_14105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 83), 'normalized_file_name', False)
        # Processing the call keyword arguments (line 136)
        kwargs_14106 = {}
        # Getting the type of 'ModuleLineNumbering' (line 136)
        ModuleLineNumbering_14103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 31), 'ModuleLineNumbering', False)
        # Obtaining the member '__get_original_source_code_file' of a type (line 136)
        get_original_source_code_file_14104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 31), ModuleLineNumbering_14103, '__get_original_source_code_file')
        # Calling __get_original_source_code_file(args, kwargs) (line 136)
        get_original_source_code_file_call_result_14107 = invoke(stypy.reporting.localization.Localization(__file__, 136, 31), get_original_source_code_file_14104, *[normalized_file_name_14105], **kwargs_14106)
        
        # Assigning a type to the variable 'normalized_file_name' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'normalized_file_name', get_original_source_code_file_call_result_14107)
        
        # Assigning a Call to a Name (line 138):
        
        # Call to get_line_from_module_code(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'normalized_file_name' (line 138)
        normalized_file_name_14110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 61), 'normalized_file_name', False)
        # Getting the type of 'line_number' (line 138)
        line_number_14111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 83), 'line_number', False)
        # Processing the call keyword arguments (line 138)
        kwargs_14112 = {}
        # Getting the type of 'ModuleLineNumbering' (line 138)
        ModuleLineNumbering_14108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'ModuleLineNumbering', False)
        # Obtaining the member 'get_line_from_module_code' of a type (line 138)
        get_line_from_module_code_14109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), ModuleLineNumbering_14108, 'get_line_from_module_code')
        # Calling get_line_from_module_code(args, kwargs) (line 138)
        get_line_from_module_code_call_result_14113 = invoke(stypy.reporting.localization.Localization(__file__, 138, 15), get_line_from_module_code_14109, *[normalized_file_name_14110, line_number_14111], **kwargs_14112)
        
        # Assigning a type to the variable 'line' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'line', get_line_from_module_code_call_result_14113)
        
        # Type idiom detected: calculating its left and rigth part (line 139)
        # Getting the type of 'line' (line 139)
        line_14114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 'line')
        # Getting the type of 'None' (line 139)
        None_14115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 19), 'None')
        
        (may_be_14116, more_types_in_union_14117) = may_be_none(line_14114, None_14115)

        if may_be_14116:

            if more_types_in_union_14117:
                # Runtime conditional SSA (line 139)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'None' (line 140)
            None_14118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 140)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'stypy_return_type', None_14118)

            if more_types_in_union_14117:
                # SSA join for if statement (line 139)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'line' (line 139)
        line_14119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'line')
        # Assigning a type to the variable 'line' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'line', remove_type_from_union(line_14119, types.NoneType))
        
        # Assigning a BinOp to a Name (line 142):
        str_14120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 21), 'str', ' ')
        # Getting the type of 'col_offset' (line 142)
        col_offset_14121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 27), 'col_offset')
        # Applying the binary operator '*' (line 142)
        result_mul_14122 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 21), '*', str_14120, col_offset_14121)
        
        str_14123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 40), 'str', '^')
        # Applying the binary operator '+' (line 142)
        result_add_14124 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 21), '+', result_mul_14122, str_14123)
        
        # Assigning a type to the variable 'blank_line' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'blank_line', result_add_14124)
        # Getting the type of 'blank_line' (line 144)
        blank_line_14125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'blank_line')
        # Assigning a type to the variable 'stypy_return_type' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'stypy_return_type', blank_line_14125)
        
        # ################# End of 'get_column_from_module_code(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_column_from_module_code' in the type store
        # Getting the type of 'stypy_return_type' (line 124)
        stypy_return_type_14126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_14126)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_column_from_module_code'
        return stypy_return_type_14126


# Assigning a type to the variable 'ModuleLineNumbering' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'ModuleLineNumbering', ModuleLineNumbering)

# Assigning a Call to a Name (line 10):

# Call to dict(...): (line 10)
# Processing the call keyword arguments (line 10)
kwargs_14128 = {}
# Getting the type of 'dict' (line 10)
dict_14127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 31), 'dict', False)
# Calling dict(args, kwargs) (line 10)
dict_call_result_14129 = invoke(stypy.reporting.localization.Localization(__file__, 10, 31), dict_14127, *[], **kwargs_14128)

# Getting the type of 'ModuleLineNumbering'
ModuleLineNumbering_14130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ModuleLineNumbering')
# Setting the type of the member 'file_numbered_code_cache' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ModuleLineNumbering_14130, 'file_numbered_code_cache', dict_call_result_14129)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
