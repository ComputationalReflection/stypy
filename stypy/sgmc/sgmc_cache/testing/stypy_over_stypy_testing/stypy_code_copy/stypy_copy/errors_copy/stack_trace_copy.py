
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from singleton_copy import Singleton
2: from stypy_copy import stypy_parameters_copy
3: import stypy_copy
4: 
5: 
6: @Singleton
7: class StackTrace:
8:     '''
9:     This class allow TypeErrors to enhance the information they provide including the stack trace that lead to the
10:     line that produced the time error. This way we can precisely trace inside the program where is the type error in
11:     order to fix it. StackTrace information is built in the type inference program generated code and are accessed
12:     through TypeErrors, so no direct usage of this class is expected. There is a single stack trace object per execution
13:     flow.
14:     '''
15:     def __init__(self):
16:         self.stack = []
17: 
18:     def set(self, file_name, line, column, function_name, declared_arguments, arguments):
19:         '''
20:         Sets the stack trace information corresponding to a function call
21:         :param file_name: .py file where the function is placed
22:         :param line: Line when the function is declared
23:         :param column: Column when the function is declared
24:         :param function_name: Function name that is called
25:         :param declared_arguments: Arguments declared in the function code
26:         :param arguments: Passed arguments in the call
27:         :return:
28:         '''
29:         self.stack.append((file_name, line, column, function_name, declared_arguments, arguments))
30: 
31:     def unset(self):
32:         '''
33:         Pops the last added stack trace (at function exit)
34:         :return:
35:         '''
36:         self.stack.pop()
37: 
38:     def __format_file_name(self, file_name):
39:         '''
40:         Pretty-print the .py file name
41:         :param file_name:
42:         :return:
43:         '''
44:         file_name = file_name.split('/')[-1]
45:         file_name = file_name.replace(stypy_parameters_copy.type_inference_file_postfix, '')
46:         file_name = file_name.replace(stypy_parameters_copy.type_inference_file_directory_name, '')
47: 
48:         return file_name
49: 
50:     def __format_type(self, type_):
51:         '''
52:         Pretty-prints types
53:         :param type_:
54:         :return:
55:         '''
56:         if isinstance(type_, stypy_copy.errors.type_error.TypeError):
57:             return "TypeError"
58:         return str(type_)
59: 
60:     def __pretty_string_params(self, declared_arguments, arguments):
61:         '''
62:         Pretty-prints function parameters
63:         :param declared_arguments:
64:         :param arguments:
65:         :return:
66:         '''
67:         zipped = zip(declared_arguments, arguments)
68:         ret_str = ""
69:         for tuple_ in zipped:
70:             ret_str += tuple_[0] + ": " + self.__format_type(tuple_[1]) + ", "
71: 
72:         return ret_str[:-2]
73: 
74:     def __pretty_string_vargargs(self, arguments):
75:         '''
76:         Pretty-prints the variable list of arguments of a function
77:         :param arguments:
78:         :return:
79:         '''
80:         if len(arguments) == 0:
81:             return ""
82: 
83:         ret_str = ", *starargs=["
84:         for arg in arguments:
85:             ret_str += self.__format_type(arg) + ", "
86: 
87:         return ret_str[:-2] + "]"
88: 
89:     def __pretty_string_kwargs(self, arguments):
90:         '''
91:         Pretty-prints the keyword arguments of a function
92:         :param arguments:
93:         :return:
94:         '''
95:         if len(arguments) == 0:
96:             return ""
97: 
98:         ret_str = ", **kwargs={"
99:         for key, arg in arguments.items():
100:             ret_str += str(key) + ": " + self.__format_type(arg) + ", "
101: 
102:         return ret_str[:-2] + "}"
103: 
104:     def to_pretty_string(self):
105:         '''
106:         Prints each called function header and its parameters in a human-readable way, comprising the full stack
107:         trace information stored in this object.
108:         :return:
109:         '''
110:         if len(self.stack) == 0:
111:             return ""
112:         s = "Call stack: [\n"
113: 
114:         for i in xrange(len(self.stack) - 1, -1, -1):
115:             file_name, line, column, function_name, declared_arguments, arguments = self.stack[i]
116: 
117:             file_name = self.__format_file_name(file_name)
118: 
119:             s += " - File '%s' (line %s, column %s)\n   Invocation to '%s(%s%s%s)'\n" % \
120:                  (file_name, line, column, function_name, self.__pretty_string_params(declared_arguments, arguments[0]),
121:                   self.__pretty_string_vargargs(arguments[1]), self.__pretty_string_kwargs(arguments[2]))
122:         s += "]"
123:         return s
124: 
125:     def __str__(self):
126:         return self.to_pretty_string()
127: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from singleton_copy import Singleton' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/errors_copy/')
import_3110 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'singleton_copy')

if (type(import_3110) is not StypyTypeError):

    if (import_3110 != 'pyd_module'):
        __import__(import_3110)
        sys_modules_3111 = sys.modules[import_3110]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'singleton_copy', sys_modules_3111.module_type_store, module_type_store, ['Singleton'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_3111, sys_modules_3111.module_type_store, module_type_store)
    else:
        from singleton_copy import Singleton

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'singleton_copy', None, module_type_store, ['Singleton'], [Singleton])

else:
    # Assigning a type to the variable 'singleton_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'singleton_copy', import_3110)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/errors_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from stypy_copy import stypy_parameters_copy' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/errors_copy/')
import_3112 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy')

if (type(import_3112) is not StypyTypeError):

    if (import_3112 != 'pyd_module'):
        __import__(import_3112)
        sys_modules_3113 = sys.modules[import_3112]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy', sys_modules_3113.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_3113, sys_modules_3113.module_type_store, module_type_store)
    else:
        from stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'stypy_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy', import_3112)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/errors_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import stypy_copy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/errors_copy/')
import_3114 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy')

if (type(import_3114) is not StypyTypeError):

    if (import_3114 != 'pyd_module'):
        __import__(import_3114)
        sys_modules_3115 = sys.modules[import_3114]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy', sys_modules_3115.module_type_store, module_type_store)
    else:
        import stypy_copy

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy', stypy_copy, module_type_store)

else:
    # Assigning a type to the variable 'stypy_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy', import_3114)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/errors_copy/')

# Declaration of the 'StackTrace' class
# Getting the type of 'Singleton' (line 6)
Singleton_3116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'Singleton')

class StackTrace:
    str_3117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, (-1)), 'str', '\n    This class allow TypeErrors to enhance the information they provide including the stack trace that lead to the\n    line that produced the time error. This way we can precisely trace inside the program where is the type error in\n    order to fix it. StackTrace information is built in the type inference program generated code and are accessed\n    through TypeErrors, so no direct usage of this class is expected. There is a single stack trace object per execution\n    flow.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StackTrace.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 16):
        
        # Assigning a List to a Attribute (line 16):
        
        # Obtaining an instance of the builtin type 'list' (line 16)
        list_3118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 16)
        
        # Getting the type of 'self' (line 16)
        self_3119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self')
        # Setting the type of the member 'stack' of a type (line 16)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), self_3119, 'stack', list_3118)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set'
        module_type_store = module_type_store.open_function_context('set', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StackTrace.set.__dict__.__setitem__('stypy_localization', localization)
        StackTrace.set.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StackTrace.set.__dict__.__setitem__('stypy_type_store', module_type_store)
        StackTrace.set.__dict__.__setitem__('stypy_function_name', 'StackTrace.set')
        StackTrace.set.__dict__.__setitem__('stypy_param_names_list', ['file_name', 'line', 'column', 'function_name', 'declared_arguments', 'arguments'])
        StackTrace.set.__dict__.__setitem__('stypy_varargs_param_name', None)
        StackTrace.set.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StackTrace.set.__dict__.__setitem__('stypy_call_defaults', defaults)
        StackTrace.set.__dict__.__setitem__('stypy_call_varargs', varargs)
        StackTrace.set.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StackTrace.set.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StackTrace.set', ['file_name', 'line', 'column', 'function_name', 'declared_arguments', 'arguments'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set', localization, ['file_name', 'line', 'column', 'function_name', 'declared_arguments', 'arguments'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set(...)' code ##################

        str_3120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, (-1)), 'str', '\n        Sets the stack trace information corresponding to a function call\n        :param file_name: .py file where the function is placed\n        :param line: Line when the function is declared\n        :param column: Column when the function is declared\n        :param function_name: Function name that is called\n        :param declared_arguments: Arguments declared in the function code\n        :param arguments: Passed arguments in the call\n        :return:\n        ')
        
        # Call to append(...): (line 29)
        # Processing the call arguments (line 29)
        
        # Obtaining an instance of the builtin type 'tuple' (line 29)
        tuple_3124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 29)
        # Adding element type (line 29)
        # Getting the type of 'file_name' (line 29)
        file_name_3125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 27), 'file_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 27), tuple_3124, file_name_3125)
        # Adding element type (line 29)
        # Getting the type of 'line' (line 29)
        line_3126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 38), 'line', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 27), tuple_3124, line_3126)
        # Adding element type (line 29)
        # Getting the type of 'column' (line 29)
        column_3127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 44), 'column', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 27), tuple_3124, column_3127)
        # Adding element type (line 29)
        # Getting the type of 'function_name' (line 29)
        function_name_3128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 52), 'function_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 27), tuple_3124, function_name_3128)
        # Adding element type (line 29)
        # Getting the type of 'declared_arguments' (line 29)
        declared_arguments_3129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 67), 'declared_arguments', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 27), tuple_3124, declared_arguments_3129)
        # Adding element type (line 29)
        # Getting the type of 'arguments' (line 29)
        arguments_3130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 87), 'arguments', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 27), tuple_3124, arguments_3130)
        
        # Processing the call keyword arguments (line 29)
        kwargs_3131 = {}
        # Getting the type of 'self' (line 29)
        self_3121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self', False)
        # Obtaining the member 'stack' of a type (line 29)
        stack_3122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_3121, 'stack')
        # Obtaining the member 'append' of a type (line 29)
        append_3123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), stack_3122, 'append')
        # Calling append(args, kwargs) (line 29)
        append_call_result_3132 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), append_3123, *[tuple_3124], **kwargs_3131)
        
        
        # ################# End of 'set(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_3133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3133)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set'
        return stypy_return_type_3133


    @norecursion
    def unset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'unset'
        module_type_store = module_type_store.open_function_context('unset', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StackTrace.unset.__dict__.__setitem__('stypy_localization', localization)
        StackTrace.unset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StackTrace.unset.__dict__.__setitem__('stypy_type_store', module_type_store)
        StackTrace.unset.__dict__.__setitem__('stypy_function_name', 'StackTrace.unset')
        StackTrace.unset.__dict__.__setitem__('stypy_param_names_list', [])
        StackTrace.unset.__dict__.__setitem__('stypy_varargs_param_name', None)
        StackTrace.unset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StackTrace.unset.__dict__.__setitem__('stypy_call_defaults', defaults)
        StackTrace.unset.__dict__.__setitem__('stypy_call_varargs', varargs)
        StackTrace.unset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StackTrace.unset.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StackTrace.unset', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'unset', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'unset(...)' code ##################

        str_3134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'str', '\n        Pops the last added stack trace (at function exit)\n        :return:\n        ')
        
        # Call to pop(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_3138 = {}
        # Getting the type of 'self' (line 36)
        self_3135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self', False)
        # Obtaining the member 'stack' of a type (line 36)
        stack_3136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_3135, 'stack')
        # Obtaining the member 'pop' of a type (line 36)
        pop_3137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), stack_3136, 'pop')
        # Calling pop(args, kwargs) (line 36)
        pop_call_result_3139 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), pop_3137, *[], **kwargs_3138)
        
        
        # ################# End of 'unset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'unset' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_3140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3140)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'unset'
        return stypy_return_type_3140


    @norecursion
    def __format_file_name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__format_file_name'
        module_type_store = module_type_store.open_function_context('__format_file_name', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StackTrace.__format_file_name.__dict__.__setitem__('stypy_localization', localization)
        StackTrace.__format_file_name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StackTrace.__format_file_name.__dict__.__setitem__('stypy_type_store', module_type_store)
        StackTrace.__format_file_name.__dict__.__setitem__('stypy_function_name', 'StackTrace.__format_file_name')
        StackTrace.__format_file_name.__dict__.__setitem__('stypy_param_names_list', ['file_name'])
        StackTrace.__format_file_name.__dict__.__setitem__('stypy_varargs_param_name', None)
        StackTrace.__format_file_name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StackTrace.__format_file_name.__dict__.__setitem__('stypy_call_defaults', defaults)
        StackTrace.__format_file_name.__dict__.__setitem__('stypy_call_varargs', varargs)
        StackTrace.__format_file_name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StackTrace.__format_file_name.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StackTrace.__format_file_name', ['file_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__format_file_name', localization, ['file_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__format_file_name(...)' code ##################

        str_3141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', '\n        Pretty-print the .py file name\n        :param file_name:\n        :return:\n        ')
        
        # Assigning a Subscript to a Name (line 44):
        
        # Assigning a Subscript to a Name (line 44):
        
        # Obtaining the type of the subscript
        int_3142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 41), 'int')
        
        # Call to split(...): (line 44)
        # Processing the call arguments (line 44)
        str_3145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 36), 'str', '/')
        # Processing the call keyword arguments (line 44)
        kwargs_3146 = {}
        # Getting the type of 'file_name' (line 44)
        file_name_3143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'file_name', False)
        # Obtaining the member 'split' of a type (line 44)
        split_3144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 20), file_name_3143, 'split')
        # Calling split(args, kwargs) (line 44)
        split_call_result_3147 = invoke(stypy.reporting.localization.Localization(__file__, 44, 20), split_3144, *[str_3145], **kwargs_3146)
        
        # Obtaining the member '__getitem__' of a type (line 44)
        getitem___3148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 20), split_call_result_3147, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 44)
        subscript_call_result_3149 = invoke(stypy.reporting.localization.Localization(__file__, 44, 20), getitem___3148, int_3142)
        
        # Assigning a type to the variable 'file_name' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'file_name', subscript_call_result_3149)
        
        # Assigning a Call to a Name (line 45):
        
        # Assigning a Call to a Name (line 45):
        
        # Call to replace(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'stypy_parameters_copy' (line 45)
        stypy_parameters_copy_3152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 38), 'stypy_parameters_copy', False)
        # Obtaining the member 'type_inference_file_postfix' of a type (line 45)
        type_inference_file_postfix_3153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 38), stypy_parameters_copy_3152, 'type_inference_file_postfix')
        str_3154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 89), 'str', '')
        # Processing the call keyword arguments (line 45)
        kwargs_3155 = {}
        # Getting the type of 'file_name' (line 45)
        file_name_3150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'file_name', False)
        # Obtaining the member 'replace' of a type (line 45)
        replace_3151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 20), file_name_3150, 'replace')
        # Calling replace(args, kwargs) (line 45)
        replace_call_result_3156 = invoke(stypy.reporting.localization.Localization(__file__, 45, 20), replace_3151, *[type_inference_file_postfix_3153, str_3154], **kwargs_3155)
        
        # Assigning a type to the variable 'file_name' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'file_name', replace_call_result_3156)
        
        # Assigning a Call to a Name (line 46):
        
        # Assigning a Call to a Name (line 46):
        
        # Call to replace(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'stypy_parameters_copy' (line 46)
        stypy_parameters_copy_3159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 38), 'stypy_parameters_copy', False)
        # Obtaining the member 'type_inference_file_directory_name' of a type (line 46)
        type_inference_file_directory_name_3160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 38), stypy_parameters_copy_3159, 'type_inference_file_directory_name')
        str_3161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 96), 'str', '')
        # Processing the call keyword arguments (line 46)
        kwargs_3162 = {}
        # Getting the type of 'file_name' (line 46)
        file_name_3157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'file_name', False)
        # Obtaining the member 'replace' of a type (line 46)
        replace_3158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 20), file_name_3157, 'replace')
        # Calling replace(args, kwargs) (line 46)
        replace_call_result_3163 = invoke(stypy.reporting.localization.Localization(__file__, 46, 20), replace_3158, *[type_inference_file_directory_name_3160, str_3161], **kwargs_3162)
        
        # Assigning a type to the variable 'file_name' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'file_name', replace_call_result_3163)
        # Getting the type of 'file_name' (line 48)
        file_name_3164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'file_name')
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', file_name_3164)
        
        # ################# End of '__format_file_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__format_file_name' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_3165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3165)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__format_file_name'
        return stypy_return_type_3165


    @norecursion
    def __format_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__format_type'
        module_type_store = module_type_store.open_function_context('__format_type', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StackTrace.__format_type.__dict__.__setitem__('stypy_localization', localization)
        StackTrace.__format_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StackTrace.__format_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        StackTrace.__format_type.__dict__.__setitem__('stypy_function_name', 'StackTrace.__format_type')
        StackTrace.__format_type.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        StackTrace.__format_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        StackTrace.__format_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StackTrace.__format_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        StackTrace.__format_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        StackTrace.__format_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StackTrace.__format_type.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StackTrace.__format_type', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__format_type', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__format_type(...)' code ##################

        str_3166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'str', '\n        Pretty-prints types\n        :param type_:\n        :return:\n        ')
        
        # Call to isinstance(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'type_' (line 56)
        type__3168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 22), 'type_', False)
        # Getting the type of 'stypy_copy' (line 56)
        stypy_copy_3169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 29), 'stypy_copy', False)
        # Obtaining the member 'errors' of a type (line 56)
        errors_3170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 29), stypy_copy_3169, 'errors')
        # Obtaining the member 'type_error' of a type (line 56)
        type_error_3171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 29), errors_3170, 'type_error')
        # Obtaining the member 'TypeError' of a type (line 56)
        TypeError_3172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 29), type_error_3171, 'TypeError')
        # Processing the call keyword arguments (line 56)
        kwargs_3173 = {}
        # Getting the type of 'isinstance' (line 56)
        isinstance_3167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 56)
        isinstance_call_result_3174 = invoke(stypy.reporting.localization.Localization(__file__, 56, 11), isinstance_3167, *[type__3168, TypeError_3172], **kwargs_3173)
        
        # Testing if the type of an if condition is none (line 56)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 56, 8), isinstance_call_result_3174):
            pass
        else:
            
            # Testing the type of an if condition (line 56)
            if_condition_3175 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 8), isinstance_call_result_3174)
            # Assigning a type to the variable 'if_condition_3175' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'if_condition_3175', if_condition_3175)
            # SSA begins for if statement (line 56)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_3176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 19), 'str', 'TypeError')
            # Assigning a type to the variable 'stypy_return_type' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'stypy_return_type', str_3176)
            # SSA join for if statement (line 56)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to str(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'type_' (line 58)
        type__3178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'type_', False)
        # Processing the call keyword arguments (line 58)
        kwargs_3179 = {}
        # Getting the type of 'str' (line 58)
        str_3177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'str', False)
        # Calling str(args, kwargs) (line 58)
        str_call_result_3180 = invoke(stypy.reporting.localization.Localization(__file__, 58, 15), str_3177, *[type__3178], **kwargs_3179)
        
        # Assigning a type to the variable 'stypy_return_type' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'stypy_return_type', str_call_result_3180)
        
        # ################# End of '__format_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__format_type' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_3181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3181)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__format_type'
        return stypy_return_type_3181


    @norecursion
    def __pretty_string_params(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__pretty_string_params'
        module_type_store = module_type_store.open_function_context('__pretty_string_params', 60, 4, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StackTrace.__pretty_string_params.__dict__.__setitem__('stypy_localization', localization)
        StackTrace.__pretty_string_params.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StackTrace.__pretty_string_params.__dict__.__setitem__('stypy_type_store', module_type_store)
        StackTrace.__pretty_string_params.__dict__.__setitem__('stypy_function_name', 'StackTrace.__pretty_string_params')
        StackTrace.__pretty_string_params.__dict__.__setitem__('stypy_param_names_list', ['declared_arguments', 'arguments'])
        StackTrace.__pretty_string_params.__dict__.__setitem__('stypy_varargs_param_name', None)
        StackTrace.__pretty_string_params.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StackTrace.__pretty_string_params.__dict__.__setitem__('stypy_call_defaults', defaults)
        StackTrace.__pretty_string_params.__dict__.__setitem__('stypy_call_varargs', varargs)
        StackTrace.__pretty_string_params.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StackTrace.__pretty_string_params.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StackTrace.__pretty_string_params', ['declared_arguments', 'arguments'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__pretty_string_params', localization, ['declared_arguments', 'arguments'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__pretty_string_params(...)' code ##################

        str_3182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, (-1)), 'str', '\n        Pretty-prints function parameters\n        :param declared_arguments:\n        :param arguments:\n        :return:\n        ')
        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Call to zip(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'declared_arguments' (line 67)
        declared_arguments_3184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 21), 'declared_arguments', False)
        # Getting the type of 'arguments' (line 67)
        arguments_3185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 41), 'arguments', False)
        # Processing the call keyword arguments (line 67)
        kwargs_3186 = {}
        # Getting the type of 'zip' (line 67)
        zip_3183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'zip', False)
        # Calling zip(args, kwargs) (line 67)
        zip_call_result_3187 = invoke(stypy.reporting.localization.Localization(__file__, 67, 17), zip_3183, *[declared_arguments_3184, arguments_3185], **kwargs_3186)
        
        # Assigning a type to the variable 'zipped' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'zipped', zip_call_result_3187)
        
        # Assigning a Str to a Name (line 68):
        
        # Assigning a Str to a Name (line 68):
        str_3188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 18), 'str', '')
        # Assigning a type to the variable 'ret_str' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'ret_str', str_3188)
        
        # Getting the type of 'zipped' (line 69)
        zipped_3189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 22), 'zipped')
        # Assigning a type to the variable 'zipped_3189' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'zipped_3189', zipped_3189)
        # Testing if the for loop is going to be iterated (line 69)
        # Testing the type of a for loop iterable (line 69)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 69, 8), zipped_3189)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 69, 8), zipped_3189):
            # Getting the type of the for loop variable (line 69)
            for_loop_var_3190 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 69, 8), zipped_3189)
            # Assigning a type to the variable 'tuple_' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'tuple_', for_loop_var_3190)
            # SSA begins for a for statement (line 69)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'ret_str' (line 70)
            ret_str_3191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'ret_str')
            
            # Obtaining the type of the subscript
            int_3192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 30), 'int')
            # Getting the type of 'tuple_' (line 70)
            tuple__3193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'tuple_')
            # Obtaining the member '__getitem__' of a type (line 70)
            getitem___3194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 23), tuple__3193, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 70)
            subscript_call_result_3195 = invoke(stypy.reporting.localization.Localization(__file__, 70, 23), getitem___3194, int_3192)
            
            str_3196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 35), 'str', ': ')
            # Applying the binary operator '+' (line 70)
            result_add_3197 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 23), '+', subscript_call_result_3195, str_3196)
            
            
            # Call to __format_type(...): (line 70)
            # Processing the call arguments (line 70)
            
            # Obtaining the type of the subscript
            int_3200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 68), 'int')
            # Getting the type of 'tuple_' (line 70)
            tuple__3201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 61), 'tuple_', False)
            # Obtaining the member '__getitem__' of a type (line 70)
            getitem___3202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 61), tuple__3201, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 70)
            subscript_call_result_3203 = invoke(stypy.reporting.localization.Localization(__file__, 70, 61), getitem___3202, int_3200)
            
            # Processing the call keyword arguments (line 70)
            kwargs_3204 = {}
            # Getting the type of 'self' (line 70)
            self_3198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 42), 'self', False)
            # Obtaining the member '__format_type' of a type (line 70)
            format_type_3199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 42), self_3198, '__format_type')
            # Calling __format_type(args, kwargs) (line 70)
            format_type_call_result_3205 = invoke(stypy.reporting.localization.Localization(__file__, 70, 42), format_type_3199, *[subscript_call_result_3203], **kwargs_3204)
            
            # Applying the binary operator '+' (line 70)
            result_add_3206 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 40), '+', result_add_3197, format_type_call_result_3205)
            
            str_3207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 74), 'str', ', ')
            # Applying the binary operator '+' (line 70)
            result_add_3208 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 72), '+', result_add_3206, str_3207)
            
            # Applying the binary operator '+=' (line 70)
            result_iadd_3209 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 12), '+=', ret_str_3191, result_add_3208)
            # Assigning a type to the variable 'ret_str' (line 70)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'ret_str', result_iadd_3209)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining the type of the subscript
        int_3210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 24), 'int')
        slice_3211 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 72, 15), None, int_3210, None)
        # Getting the type of 'ret_str' (line 72)
        ret_str_3212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'ret_str')
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___3213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 15), ret_str_3212, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_3214 = invoke(stypy.reporting.localization.Localization(__file__, 72, 15), getitem___3213, slice_3211)
        
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'stypy_return_type', subscript_call_result_3214)
        
        # ################# End of '__pretty_string_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__pretty_string_params' in the type store
        # Getting the type of 'stypy_return_type' (line 60)
        stypy_return_type_3215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3215)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__pretty_string_params'
        return stypy_return_type_3215


    @norecursion
    def __pretty_string_vargargs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__pretty_string_vargargs'
        module_type_store = module_type_store.open_function_context('__pretty_string_vargargs', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StackTrace.__pretty_string_vargargs.__dict__.__setitem__('stypy_localization', localization)
        StackTrace.__pretty_string_vargargs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StackTrace.__pretty_string_vargargs.__dict__.__setitem__('stypy_type_store', module_type_store)
        StackTrace.__pretty_string_vargargs.__dict__.__setitem__('stypy_function_name', 'StackTrace.__pretty_string_vargargs')
        StackTrace.__pretty_string_vargargs.__dict__.__setitem__('stypy_param_names_list', ['arguments'])
        StackTrace.__pretty_string_vargargs.__dict__.__setitem__('stypy_varargs_param_name', None)
        StackTrace.__pretty_string_vargargs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StackTrace.__pretty_string_vargargs.__dict__.__setitem__('stypy_call_defaults', defaults)
        StackTrace.__pretty_string_vargargs.__dict__.__setitem__('stypy_call_varargs', varargs)
        StackTrace.__pretty_string_vargargs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StackTrace.__pretty_string_vargargs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StackTrace.__pretty_string_vargargs', ['arguments'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__pretty_string_vargargs', localization, ['arguments'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__pretty_string_vargargs(...)' code ##################

        str_3216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, (-1)), 'str', '\n        Pretty-prints the variable list of arguments of a function\n        :param arguments:\n        :return:\n        ')
        
        
        # Call to len(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'arguments' (line 80)
        arguments_3218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'arguments', False)
        # Processing the call keyword arguments (line 80)
        kwargs_3219 = {}
        # Getting the type of 'len' (line 80)
        len_3217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'len', False)
        # Calling len(args, kwargs) (line 80)
        len_call_result_3220 = invoke(stypy.reporting.localization.Localization(__file__, 80, 11), len_3217, *[arguments_3218], **kwargs_3219)
        
        int_3221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 29), 'int')
        # Applying the binary operator '==' (line 80)
        result_eq_3222 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 11), '==', len_call_result_3220, int_3221)
        
        # Testing if the type of an if condition is none (line 80)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 8), result_eq_3222):
            pass
        else:
            
            # Testing the type of an if condition (line 80)
            if_condition_3223 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 8), result_eq_3222)
            # Assigning a type to the variable 'if_condition_3223' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'if_condition_3223', if_condition_3223)
            # SSA begins for if statement (line 80)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_3224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 19), 'str', '')
            # Assigning a type to the variable 'stypy_return_type' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'stypy_return_type', str_3224)
            # SSA join for if statement (line 80)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Str to a Name (line 83):
        
        # Assigning a Str to a Name (line 83):
        str_3225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 18), 'str', ', *starargs=[')
        # Assigning a type to the variable 'ret_str' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'ret_str', str_3225)
        
        # Getting the type of 'arguments' (line 84)
        arguments_3226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'arguments')
        # Assigning a type to the variable 'arguments_3226' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'arguments_3226', arguments_3226)
        # Testing if the for loop is going to be iterated (line 84)
        # Testing the type of a for loop iterable (line 84)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 84, 8), arguments_3226)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 84, 8), arguments_3226):
            # Getting the type of the for loop variable (line 84)
            for_loop_var_3227 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 84, 8), arguments_3226)
            # Assigning a type to the variable 'arg' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'arg', for_loop_var_3227)
            # SSA begins for a for statement (line 84)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'ret_str' (line 85)
            ret_str_3228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'ret_str')
            
            # Call to __format_type(...): (line 85)
            # Processing the call arguments (line 85)
            # Getting the type of 'arg' (line 85)
            arg_3231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 42), 'arg', False)
            # Processing the call keyword arguments (line 85)
            kwargs_3232 = {}
            # Getting the type of 'self' (line 85)
            self_3229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'self', False)
            # Obtaining the member '__format_type' of a type (line 85)
            format_type_3230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 23), self_3229, '__format_type')
            # Calling __format_type(args, kwargs) (line 85)
            format_type_call_result_3233 = invoke(stypy.reporting.localization.Localization(__file__, 85, 23), format_type_3230, *[arg_3231], **kwargs_3232)
            
            str_3234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 49), 'str', ', ')
            # Applying the binary operator '+' (line 85)
            result_add_3235 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 23), '+', format_type_call_result_3233, str_3234)
            
            # Applying the binary operator '+=' (line 85)
            result_iadd_3236 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 12), '+=', ret_str_3228, result_add_3235)
            # Assigning a type to the variable 'ret_str' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'ret_str', result_iadd_3236)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining the type of the subscript
        int_3237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 24), 'int')
        slice_3238 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 87, 15), None, int_3237, None)
        # Getting the type of 'ret_str' (line 87)
        ret_str_3239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'ret_str')
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___3240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 15), ret_str_3239, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_3241 = invoke(stypy.reporting.localization.Localization(__file__, 87, 15), getitem___3240, slice_3238)
        
        str_3242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 30), 'str', ']')
        # Applying the binary operator '+' (line 87)
        result_add_3243 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 15), '+', subscript_call_result_3241, str_3242)
        
        # Assigning a type to the variable 'stypy_return_type' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type', result_add_3243)
        
        # ################# End of '__pretty_string_vargargs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__pretty_string_vargargs' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_3244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3244)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__pretty_string_vargargs'
        return stypy_return_type_3244


    @norecursion
    def __pretty_string_kwargs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__pretty_string_kwargs'
        module_type_store = module_type_store.open_function_context('__pretty_string_kwargs', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StackTrace.__pretty_string_kwargs.__dict__.__setitem__('stypy_localization', localization)
        StackTrace.__pretty_string_kwargs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StackTrace.__pretty_string_kwargs.__dict__.__setitem__('stypy_type_store', module_type_store)
        StackTrace.__pretty_string_kwargs.__dict__.__setitem__('stypy_function_name', 'StackTrace.__pretty_string_kwargs')
        StackTrace.__pretty_string_kwargs.__dict__.__setitem__('stypy_param_names_list', ['arguments'])
        StackTrace.__pretty_string_kwargs.__dict__.__setitem__('stypy_varargs_param_name', None)
        StackTrace.__pretty_string_kwargs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StackTrace.__pretty_string_kwargs.__dict__.__setitem__('stypy_call_defaults', defaults)
        StackTrace.__pretty_string_kwargs.__dict__.__setitem__('stypy_call_varargs', varargs)
        StackTrace.__pretty_string_kwargs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StackTrace.__pretty_string_kwargs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StackTrace.__pretty_string_kwargs', ['arguments'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__pretty_string_kwargs', localization, ['arguments'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__pretty_string_kwargs(...)' code ##################

        str_3245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, (-1)), 'str', '\n        Pretty-prints the keyword arguments of a function\n        :param arguments:\n        :return:\n        ')
        
        
        # Call to len(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'arguments' (line 95)
        arguments_3247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'arguments', False)
        # Processing the call keyword arguments (line 95)
        kwargs_3248 = {}
        # Getting the type of 'len' (line 95)
        len_3246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'len', False)
        # Calling len(args, kwargs) (line 95)
        len_call_result_3249 = invoke(stypy.reporting.localization.Localization(__file__, 95, 11), len_3246, *[arguments_3247], **kwargs_3248)
        
        int_3250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 29), 'int')
        # Applying the binary operator '==' (line 95)
        result_eq_3251 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 11), '==', len_call_result_3249, int_3250)
        
        # Testing if the type of an if condition is none (line 95)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 95, 8), result_eq_3251):
            pass
        else:
            
            # Testing the type of an if condition (line 95)
            if_condition_3252 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 8), result_eq_3251)
            # Assigning a type to the variable 'if_condition_3252' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'if_condition_3252', if_condition_3252)
            # SSA begins for if statement (line 95)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_3253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 19), 'str', '')
            # Assigning a type to the variable 'stypy_return_type' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'stypy_return_type', str_3253)
            # SSA join for if statement (line 95)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Str to a Name (line 98):
        
        # Assigning a Str to a Name (line 98):
        str_3254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 18), 'str', ', **kwargs={')
        # Assigning a type to the variable 'ret_str' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'ret_str', str_3254)
        
        
        # Call to items(...): (line 99)
        # Processing the call keyword arguments (line 99)
        kwargs_3257 = {}
        # Getting the type of 'arguments' (line 99)
        arguments_3255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'arguments', False)
        # Obtaining the member 'items' of a type (line 99)
        items_3256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 24), arguments_3255, 'items')
        # Calling items(args, kwargs) (line 99)
        items_call_result_3258 = invoke(stypy.reporting.localization.Localization(__file__, 99, 24), items_3256, *[], **kwargs_3257)
        
        # Assigning a type to the variable 'items_call_result_3258' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'items_call_result_3258', items_call_result_3258)
        # Testing if the for loop is going to be iterated (line 99)
        # Testing the type of a for loop iterable (line 99)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 99, 8), items_call_result_3258)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 99, 8), items_call_result_3258):
            # Getting the type of the for loop variable (line 99)
            for_loop_var_3259 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 99, 8), items_call_result_3258)
            # Assigning a type to the variable 'key' (line 99)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 8), for_loop_var_3259, 2, 0))
            # Assigning a type to the variable 'arg' (line 99)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'arg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 8), for_loop_var_3259, 2, 1))
            # SSA begins for a for statement (line 99)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'ret_str' (line 100)
            ret_str_3260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'ret_str')
            
            # Call to str(...): (line 100)
            # Processing the call arguments (line 100)
            # Getting the type of 'key' (line 100)
            key_3262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'key', False)
            # Processing the call keyword arguments (line 100)
            kwargs_3263 = {}
            # Getting the type of 'str' (line 100)
            str_3261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'str', False)
            # Calling str(args, kwargs) (line 100)
            str_call_result_3264 = invoke(stypy.reporting.localization.Localization(__file__, 100, 23), str_3261, *[key_3262], **kwargs_3263)
            
            str_3265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 34), 'str', ': ')
            # Applying the binary operator '+' (line 100)
            result_add_3266 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 23), '+', str_call_result_3264, str_3265)
            
            
            # Call to __format_type(...): (line 100)
            # Processing the call arguments (line 100)
            # Getting the type of 'arg' (line 100)
            arg_3269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 60), 'arg', False)
            # Processing the call keyword arguments (line 100)
            kwargs_3270 = {}
            # Getting the type of 'self' (line 100)
            self_3267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 41), 'self', False)
            # Obtaining the member '__format_type' of a type (line 100)
            format_type_3268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 41), self_3267, '__format_type')
            # Calling __format_type(args, kwargs) (line 100)
            format_type_call_result_3271 = invoke(stypy.reporting.localization.Localization(__file__, 100, 41), format_type_3268, *[arg_3269], **kwargs_3270)
            
            # Applying the binary operator '+' (line 100)
            result_add_3272 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 39), '+', result_add_3266, format_type_call_result_3271)
            
            str_3273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 67), 'str', ', ')
            # Applying the binary operator '+' (line 100)
            result_add_3274 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 65), '+', result_add_3272, str_3273)
            
            # Applying the binary operator '+=' (line 100)
            result_iadd_3275 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 12), '+=', ret_str_3260, result_add_3274)
            # Assigning a type to the variable 'ret_str' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'ret_str', result_iadd_3275)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining the type of the subscript
        int_3276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 24), 'int')
        slice_3277 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 102, 15), None, int_3276, None)
        # Getting the type of 'ret_str' (line 102)
        ret_str_3278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'ret_str')
        # Obtaining the member '__getitem__' of a type (line 102)
        getitem___3279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 15), ret_str_3278, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 102)
        subscript_call_result_3280 = invoke(stypy.reporting.localization.Localization(__file__, 102, 15), getitem___3279, slice_3277)
        
        str_3281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 30), 'str', '}')
        # Applying the binary operator '+' (line 102)
        result_add_3282 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 15), '+', subscript_call_result_3280, str_3281)
        
        # Assigning a type to the variable 'stypy_return_type' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'stypy_return_type', result_add_3282)
        
        # ################# End of '__pretty_string_kwargs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__pretty_string_kwargs' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_3283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3283)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__pretty_string_kwargs'
        return stypy_return_type_3283


    @norecursion
    def to_pretty_string(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'to_pretty_string'
        module_type_store = module_type_store.open_function_context('to_pretty_string', 104, 4, False)
        # Assigning a type to the variable 'self' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StackTrace.to_pretty_string.__dict__.__setitem__('stypy_localization', localization)
        StackTrace.to_pretty_string.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StackTrace.to_pretty_string.__dict__.__setitem__('stypy_type_store', module_type_store)
        StackTrace.to_pretty_string.__dict__.__setitem__('stypy_function_name', 'StackTrace.to_pretty_string')
        StackTrace.to_pretty_string.__dict__.__setitem__('stypy_param_names_list', [])
        StackTrace.to_pretty_string.__dict__.__setitem__('stypy_varargs_param_name', None)
        StackTrace.to_pretty_string.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StackTrace.to_pretty_string.__dict__.__setitem__('stypy_call_defaults', defaults)
        StackTrace.to_pretty_string.__dict__.__setitem__('stypy_call_varargs', varargs)
        StackTrace.to_pretty_string.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StackTrace.to_pretty_string.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StackTrace.to_pretty_string', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'to_pretty_string', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'to_pretty_string(...)' code ##################

        str_3284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'str', '\n        Prints each called function header and its parameters in a human-readable way, comprising the full stack\n        trace information stored in this object.\n        :return:\n        ')
        
        
        # Call to len(...): (line 110)
        # Processing the call arguments (line 110)
        # Getting the type of 'self' (line 110)
        self_3286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 15), 'self', False)
        # Obtaining the member 'stack' of a type (line 110)
        stack_3287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 15), self_3286, 'stack')
        # Processing the call keyword arguments (line 110)
        kwargs_3288 = {}
        # Getting the type of 'len' (line 110)
        len_3285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'len', False)
        # Calling len(args, kwargs) (line 110)
        len_call_result_3289 = invoke(stypy.reporting.localization.Localization(__file__, 110, 11), len_3285, *[stack_3287], **kwargs_3288)
        
        int_3290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 30), 'int')
        # Applying the binary operator '==' (line 110)
        result_eq_3291 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 11), '==', len_call_result_3289, int_3290)
        
        # Testing if the type of an if condition is none (line 110)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 110, 8), result_eq_3291):
            pass
        else:
            
            # Testing the type of an if condition (line 110)
            if_condition_3292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 8), result_eq_3291)
            # Assigning a type to the variable 'if_condition_3292' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'if_condition_3292', if_condition_3292)
            # SSA begins for if statement (line 110)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_3293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 19), 'str', '')
            # Assigning a type to the variable 'stypy_return_type' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'stypy_return_type', str_3293)
            # SSA join for if statement (line 110)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Str to a Name (line 112):
        
        # Assigning a Str to a Name (line 112):
        str_3294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 12), 'str', 'Call stack: [\n')
        # Assigning a type to the variable 's' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 's', str_3294)
        
        
        # Call to xrange(...): (line 114)
        # Processing the call arguments (line 114)
        
        # Call to len(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'self' (line 114)
        self_3297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 28), 'self', False)
        # Obtaining the member 'stack' of a type (line 114)
        stack_3298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 28), self_3297, 'stack')
        # Processing the call keyword arguments (line 114)
        kwargs_3299 = {}
        # Getting the type of 'len' (line 114)
        len_3296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'len', False)
        # Calling len(args, kwargs) (line 114)
        len_call_result_3300 = invoke(stypy.reporting.localization.Localization(__file__, 114, 24), len_3296, *[stack_3298], **kwargs_3299)
        
        int_3301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 42), 'int')
        # Applying the binary operator '-' (line 114)
        result_sub_3302 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 24), '-', len_call_result_3300, int_3301)
        
        int_3303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 45), 'int')
        int_3304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 49), 'int')
        # Processing the call keyword arguments (line 114)
        kwargs_3305 = {}
        # Getting the type of 'xrange' (line 114)
        xrange_3295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 114)
        xrange_call_result_3306 = invoke(stypy.reporting.localization.Localization(__file__, 114, 17), xrange_3295, *[result_sub_3302, int_3303, int_3304], **kwargs_3305)
        
        # Assigning a type to the variable 'xrange_call_result_3306' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'xrange_call_result_3306', xrange_call_result_3306)
        # Testing if the for loop is going to be iterated (line 114)
        # Testing the type of a for loop iterable (line 114)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 114, 8), xrange_call_result_3306)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 114, 8), xrange_call_result_3306):
            # Getting the type of the for loop variable (line 114)
            for_loop_var_3307 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 114, 8), xrange_call_result_3306)
            # Assigning a type to the variable 'i' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'i', for_loop_var_3307)
            # SSA begins for a for statement (line 114)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Tuple (line 115):
            
            # Assigning a Subscript to a Name (line 115):
            
            # Obtaining the type of the subscript
            int_3308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 12), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 115)
            i_3309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 95), 'i')
            # Getting the type of 'self' (line 115)
            self_3310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 84), 'self')
            # Obtaining the member 'stack' of a type (line 115)
            stack_3311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 84), self_3310, 'stack')
            # Obtaining the member '__getitem__' of a type (line 115)
            getitem___3312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 84), stack_3311, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 115)
            subscript_call_result_3313 = invoke(stypy.reporting.localization.Localization(__file__, 115, 84), getitem___3312, i_3309)
            
            # Obtaining the member '__getitem__' of a type (line 115)
            getitem___3314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), subscript_call_result_3313, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 115)
            subscript_call_result_3315 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), getitem___3314, int_3308)
            
            # Assigning a type to the variable 'tuple_var_assignment_3104' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_3104', subscript_call_result_3315)
            
            # Assigning a Subscript to a Name (line 115):
            
            # Obtaining the type of the subscript
            int_3316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 12), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 115)
            i_3317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 95), 'i')
            # Getting the type of 'self' (line 115)
            self_3318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 84), 'self')
            # Obtaining the member 'stack' of a type (line 115)
            stack_3319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 84), self_3318, 'stack')
            # Obtaining the member '__getitem__' of a type (line 115)
            getitem___3320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 84), stack_3319, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 115)
            subscript_call_result_3321 = invoke(stypy.reporting.localization.Localization(__file__, 115, 84), getitem___3320, i_3317)
            
            # Obtaining the member '__getitem__' of a type (line 115)
            getitem___3322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), subscript_call_result_3321, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 115)
            subscript_call_result_3323 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), getitem___3322, int_3316)
            
            # Assigning a type to the variable 'tuple_var_assignment_3105' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_3105', subscript_call_result_3323)
            
            # Assigning a Subscript to a Name (line 115):
            
            # Obtaining the type of the subscript
            int_3324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 12), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 115)
            i_3325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 95), 'i')
            # Getting the type of 'self' (line 115)
            self_3326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 84), 'self')
            # Obtaining the member 'stack' of a type (line 115)
            stack_3327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 84), self_3326, 'stack')
            # Obtaining the member '__getitem__' of a type (line 115)
            getitem___3328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 84), stack_3327, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 115)
            subscript_call_result_3329 = invoke(stypy.reporting.localization.Localization(__file__, 115, 84), getitem___3328, i_3325)
            
            # Obtaining the member '__getitem__' of a type (line 115)
            getitem___3330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), subscript_call_result_3329, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 115)
            subscript_call_result_3331 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), getitem___3330, int_3324)
            
            # Assigning a type to the variable 'tuple_var_assignment_3106' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_3106', subscript_call_result_3331)
            
            # Assigning a Subscript to a Name (line 115):
            
            # Obtaining the type of the subscript
            int_3332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 12), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 115)
            i_3333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 95), 'i')
            # Getting the type of 'self' (line 115)
            self_3334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 84), 'self')
            # Obtaining the member 'stack' of a type (line 115)
            stack_3335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 84), self_3334, 'stack')
            # Obtaining the member '__getitem__' of a type (line 115)
            getitem___3336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 84), stack_3335, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 115)
            subscript_call_result_3337 = invoke(stypy.reporting.localization.Localization(__file__, 115, 84), getitem___3336, i_3333)
            
            # Obtaining the member '__getitem__' of a type (line 115)
            getitem___3338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), subscript_call_result_3337, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 115)
            subscript_call_result_3339 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), getitem___3338, int_3332)
            
            # Assigning a type to the variable 'tuple_var_assignment_3107' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_3107', subscript_call_result_3339)
            
            # Assigning a Subscript to a Name (line 115):
            
            # Obtaining the type of the subscript
            int_3340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 12), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 115)
            i_3341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 95), 'i')
            # Getting the type of 'self' (line 115)
            self_3342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 84), 'self')
            # Obtaining the member 'stack' of a type (line 115)
            stack_3343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 84), self_3342, 'stack')
            # Obtaining the member '__getitem__' of a type (line 115)
            getitem___3344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 84), stack_3343, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 115)
            subscript_call_result_3345 = invoke(stypy.reporting.localization.Localization(__file__, 115, 84), getitem___3344, i_3341)
            
            # Obtaining the member '__getitem__' of a type (line 115)
            getitem___3346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), subscript_call_result_3345, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 115)
            subscript_call_result_3347 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), getitem___3346, int_3340)
            
            # Assigning a type to the variable 'tuple_var_assignment_3108' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_3108', subscript_call_result_3347)
            
            # Assigning a Subscript to a Name (line 115):
            
            # Obtaining the type of the subscript
            int_3348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 12), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 115)
            i_3349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 95), 'i')
            # Getting the type of 'self' (line 115)
            self_3350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 84), 'self')
            # Obtaining the member 'stack' of a type (line 115)
            stack_3351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 84), self_3350, 'stack')
            # Obtaining the member '__getitem__' of a type (line 115)
            getitem___3352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 84), stack_3351, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 115)
            subscript_call_result_3353 = invoke(stypy.reporting.localization.Localization(__file__, 115, 84), getitem___3352, i_3349)
            
            # Obtaining the member '__getitem__' of a type (line 115)
            getitem___3354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), subscript_call_result_3353, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 115)
            subscript_call_result_3355 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), getitem___3354, int_3348)
            
            # Assigning a type to the variable 'tuple_var_assignment_3109' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_3109', subscript_call_result_3355)
            
            # Assigning a Name to a Name (line 115):
            # Getting the type of 'tuple_var_assignment_3104' (line 115)
            tuple_var_assignment_3104_3356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_3104')
            # Assigning a type to the variable 'file_name' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'file_name', tuple_var_assignment_3104_3356)
            
            # Assigning a Name to a Name (line 115):
            # Getting the type of 'tuple_var_assignment_3105' (line 115)
            tuple_var_assignment_3105_3357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_3105')
            # Assigning a type to the variable 'line' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 23), 'line', tuple_var_assignment_3105_3357)
            
            # Assigning a Name to a Name (line 115):
            # Getting the type of 'tuple_var_assignment_3106' (line 115)
            tuple_var_assignment_3106_3358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_3106')
            # Assigning a type to the variable 'column' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 29), 'column', tuple_var_assignment_3106_3358)
            
            # Assigning a Name to a Name (line 115):
            # Getting the type of 'tuple_var_assignment_3107' (line 115)
            tuple_var_assignment_3107_3359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_3107')
            # Assigning a type to the variable 'function_name' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 37), 'function_name', tuple_var_assignment_3107_3359)
            
            # Assigning a Name to a Name (line 115):
            # Getting the type of 'tuple_var_assignment_3108' (line 115)
            tuple_var_assignment_3108_3360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_3108')
            # Assigning a type to the variable 'declared_arguments' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 52), 'declared_arguments', tuple_var_assignment_3108_3360)
            
            # Assigning a Name to a Name (line 115):
            # Getting the type of 'tuple_var_assignment_3109' (line 115)
            tuple_var_assignment_3109_3361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'tuple_var_assignment_3109')
            # Assigning a type to the variable 'arguments' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 72), 'arguments', tuple_var_assignment_3109_3361)
            
            # Assigning a Call to a Name (line 117):
            
            # Assigning a Call to a Name (line 117):
            
            # Call to __format_file_name(...): (line 117)
            # Processing the call arguments (line 117)
            # Getting the type of 'file_name' (line 117)
            file_name_3364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 48), 'file_name', False)
            # Processing the call keyword arguments (line 117)
            kwargs_3365 = {}
            # Getting the type of 'self' (line 117)
            self_3362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 24), 'self', False)
            # Obtaining the member '__format_file_name' of a type (line 117)
            format_file_name_3363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 24), self_3362, '__format_file_name')
            # Calling __format_file_name(args, kwargs) (line 117)
            format_file_name_call_result_3366 = invoke(stypy.reporting.localization.Localization(__file__, 117, 24), format_file_name_3363, *[file_name_3364], **kwargs_3365)
            
            # Assigning a type to the variable 'file_name' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'file_name', format_file_name_call_result_3366)
            
            # Getting the type of 's' (line 119)
            s_3367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 's')
            str_3368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 17), 'str', " - File '%s' (line %s, column %s)\n   Invocation to '%s(%s%s%s)'\n")
            
            # Obtaining an instance of the builtin type 'tuple' (line 120)
            tuple_3369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 18), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 120)
            # Adding element type (line 120)
            # Getting the type of 'file_name' (line 120)
            file_name_3370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 18), 'file_name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 18), tuple_3369, file_name_3370)
            # Adding element type (line 120)
            # Getting the type of 'line' (line 120)
            line_3371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 29), 'line')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 18), tuple_3369, line_3371)
            # Adding element type (line 120)
            # Getting the type of 'column' (line 120)
            column_3372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 35), 'column')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 18), tuple_3369, column_3372)
            # Adding element type (line 120)
            # Getting the type of 'function_name' (line 120)
            function_name_3373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 43), 'function_name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 18), tuple_3369, function_name_3373)
            # Adding element type (line 120)
            
            # Call to __pretty_string_params(...): (line 120)
            # Processing the call arguments (line 120)
            # Getting the type of 'declared_arguments' (line 120)
            declared_arguments_3376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 86), 'declared_arguments', False)
            
            # Obtaining the type of the subscript
            int_3377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 116), 'int')
            # Getting the type of 'arguments' (line 120)
            arguments_3378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 106), 'arguments', False)
            # Obtaining the member '__getitem__' of a type (line 120)
            getitem___3379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 106), arguments_3378, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 120)
            subscript_call_result_3380 = invoke(stypy.reporting.localization.Localization(__file__, 120, 106), getitem___3379, int_3377)
            
            # Processing the call keyword arguments (line 120)
            kwargs_3381 = {}
            # Getting the type of 'self' (line 120)
            self_3374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 58), 'self', False)
            # Obtaining the member '__pretty_string_params' of a type (line 120)
            pretty_string_params_3375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 58), self_3374, '__pretty_string_params')
            # Calling __pretty_string_params(args, kwargs) (line 120)
            pretty_string_params_call_result_3382 = invoke(stypy.reporting.localization.Localization(__file__, 120, 58), pretty_string_params_3375, *[declared_arguments_3376, subscript_call_result_3380], **kwargs_3381)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 18), tuple_3369, pretty_string_params_call_result_3382)
            # Adding element type (line 120)
            
            # Call to __pretty_string_vargargs(...): (line 121)
            # Processing the call arguments (line 121)
            
            # Obtaining the type of the subscript
            int_3385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 58), 'int')
            # Getting the type of 'arguments' (line 121)
            arguments_3386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 48), 'arguments', False)
            # Obtaining the member '__getitem__' of a type (line 121)
            getitem___3387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 48), arguments_3386, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 121)
            subscript_call_result_3388 = invoke(stypy.reporting.localization.Localization(__file__, 121, 48), getitem___3387, int_3385)
            
            # Processing the call keyword arguments (line 121)
            kwargs_3389 = {}
            # Getting the type of 'self' (line 121)
            self_3383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 18), 'self', False)
            # Obtaining the member '__pretty_string_vargargs' of a type (line 121)
            pretty_string_vargargs_3384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 18), self_3383, '__pretty_string_vargargs')
            # Calling __pretty_string_vargargs(args, kwargs) (line 121)
            pretty_string_vargargs_call_result_3390 = invoke(stypy.reporting.localization.Localization(__file__, 121, 18), pretty_string_vargargs_3384, *[subscript_call_result_3388], **kwargs_3389)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 18), tuple_3369, pretty_string_vargargs_call_result_3390)
            # Adding element type (line 120)
            
            # Call to __pretty_string_kwargs(...): (line 121)
            # Processing the call arguments (line 121)
            
            # Obtaining the type of the subscript
            int_3393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 101), 'int')
            # Getting the type of 'arguments' (line 121)
            arguments_3394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 91), 'arguments', False)
            # Obtaining the member '__getitem__' of a type (line 121)
            getitem___3395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 91), arguments_3394, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 121)
            subscript_call_result_3396 = invoke(stypy.reporting.localization.Localization(__file__, 121, 91), getitem___3395, int_3393)
            
            # Processing the call keyword arguments (line 121)
            kwargs_3397 = {}
            # Getting the type of 'self' (line 121)
            self_3391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 63), 'self', False)
            # Obtaining the member '__pretty_string_kwargs' of a type (line 121)
            pretty_string_kwargs_3392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 63), self_3391, '__pretty_string_kwargs')
            # Calling __pretty_string_kwargs(args, kwargs) (line 121)
            pretty_string_kwargs_call_result_3398 = invoke(stypy.reporting.localization.Localization(__file__, 121, 63), pretty_string_kwargs_3392, *[subscript_call_result_3396], **kwargs_3397)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 18), tuple_3369, pretty_string_kwargs_call_result_3398)
            
            # Applying the binary operator '%' (line 119)
            result_mod_3399 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 17), '%', str_3368, tuple_3369)
            
            # Applying the binary operator '+=' (line 119)
            result_iadd_3400 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 12), '+=', s_3367, result_mod_3399)
            # Assigning a type to the variable 's' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 's', result_iadd_3400)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 's' (line 122)
        s_3401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 's')
        str_3402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 13), 'str', ']')
        # Applying the binary operator '+=' (line 122)
        result_iadd_3403 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 8), '+=', s_3401, str_3402)
        # Assigning a type to the variable 's' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 's', result_iadd_3403)
        
        # Getting the type of 's' (line 123)
        s_3404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type', s_3404)
        
        # ################# End of 'to_pretty_string(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'to_pretty_string' in the type store
        # Getting the type of 'stypy_return_type' (line 104)
        stypy_return_type_3405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3405)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'to_pretty_string'
        return stypy_return_type_3405


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        StackTrace.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        StackTrace.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        StackTrace.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        StackTrace.stypy__str__.__dict__.__setitem__('stypy_function_name', 'StackTrace.stypy__str__')
        StackTrace.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        StackTrace.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        StackTrace.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        StackTrace.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        StackTrace.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        StackTrace.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        StackTrace.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'StackTrace.stypy__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        
        # Call to to_pretty_string(...): (line 126)
        # Processing the call keyword arguments (line 126)
        kwargs_3408 = {}
        # Getting the type of 'self' (line 126)
        self_3406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'self', False)
        # Obtaining the member 'to_pretty_string' of a type (line 126)
        to_pretty_string_3407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 15), self_3406, 'to_pretty_string')
        # Calling to_pretty_string(args, kwargs) (line 126)
        to_pretty_string_call_result_3409 = invoke(stypy.reporting.localization.Localization(__file__, 126, 15), to_pretty_string_3407, *[], **kwargs_3408)
        
        # Assigning a type to the variable 'stypy_return_type' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type', to_pretty_string_call_result_3409)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_3410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3410)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_3410


# Assigning a type to the variable 'StackTrace' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'StackTrace', StackTrace)
# Getting the type of 'StackTrace' (line 6)
StackTrace_3411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'StackTrace')
class_3412 = invoke(stypy.reporting.localization.Localization(__file__, 6, 0), Singleton_3116, StackTrace_3411)
# Assigning a type to the variable 'StackTrace' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'StackTrace', class_3412)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
