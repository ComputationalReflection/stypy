
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from singleton_copy import Singleton
2: from ...stypy_copy import stypy_parameters_copy
3: from ... import stypy_copy
4: 
5: @Singleton
6: class StackTrace:
7:     '''
8:     This class allow TypeErrors to enhance the information they provide including the stack trace that lead to the
9:     line that produced the time error. This way we can precisely trace inside the program where is the type error in
10:     order to fix it. StackTrace information is built in the type inference program generated code and are accessed
11:     through TypeErrors, so no direct usage of this class is expected. There is a single stack trace object per execution
12:     flow.
13:     '''
14:     def __init__(self):
15:         self.stack = []
16: 
17:     def set(self, file_name, line, column, function_name, declared_arguments, arguments):
18:         '''
19:         Sets the stack trace information corresponding to a function call
20:         :param file_name: .py file where the function is placed
21:         :param line: Line when the function is declared
22:         :param column: Column when the function is declared
23:         :param function_name: Function name that is called
24:         :param declared_arguments: Arguments declared in the function code
25:         :param arguments: Passed arguments in the call
26:         :return:
27:         '''
28:         self.stack.append((file_name, line, column, function_name, declared_arguments, arguments))
29: 
30:     def unset(self):
31:         '''
32:         Pops the last added stack trace (at function exit)
33:         :return:
34:         '''
35:         self.stack.pop()
36: 
37:     def __format_file_name(self, file_name):
38:         '''
39:         Pretty-print the .py file name
40:         :param file_name:
41:         :return:
42:         '''
43:         file_name = file_name.split('/')[-1]
44:         file_name = file_name.replace(stypy_parameters_copy.type_inference_file_postfix, '')
45:         file_name = file_name.replace(stypy_parameters_copy.type_inference_file_directory_name, '')
46: 
47:         return file_name
48: 
49:     def __format_type(self, type_):
50:         '''
51:         Pretty-prints types
52:         :param type_:
53:         :return:
54:         '''
55:         if isinstance(type_, stypy_copy.errors.type_error.TypeError):
56:             return "TypeError"
57:         return str(type_)
58: 
59:     def __pretty_string_params(self, declared_arguments, arguments):
60:         '''
61:         Pretty-prints function parameters
62:         :param declared_arguments:
63:         :param arguments:
64:         :return:
65:         '''
66:         zipped = zip(declared_arguments, arguments)
67:         ret_str = ""
68:         for tuple_ in zipped:
69:             ret_str += tuple_[0] + ": " + self.__format_type(tuple_[1]) + ", "
70: 
71:         return ret_str[:-2]
72: 
73:     def __pretty_string_vargargs(self, arguments):
74:         '''
75:         Pretty-prints the variable list of arguments of a function
76:         :param arguments:
77:         :return:
78:         '''
79:         if len(arguments) == 0:
80:             return ""
81: 
82:         ret_str = ", *starargs=["
83:         for arg in arguments:
84:             ret_str += self.__format_type(arg) + ", "
85: 
86:         return ret_str[:-2] + "]"
87: 
88:     def __pretty_string_kwargs(self, arguments):
89:         '''
90:         Pretty-prints the keyword arguments of a function
91:         :param arguments:
92:         :return:
93:         '''
94:         if len(arguments) == 0:
95:             return ""
96: 
97:         ret_str = ", **kwargs={"
98:         for key, arg in arguments.items():
99:             ret_str += str(key) + ": " + self.__format_type(arg) + ", "
100: 
101:         return ret_str[:-2] + "}"
102: 
103:     def to_pretty_string(self):
104:         '''
105:         Prints each called function header and its parameters in a human-readable way, comprising the full stack
106:         trace information stored in this object.
107:         :return:
108:         '''
109:         if len(self.stack) == 0:
110:             return ""
111:         s = "Call stack: [\n"
112: 
113:         for i in xrange(len(self.stack) - 1, -1, -1):
114:             file_name, line, column, function_name, declared_arguments, arguments = self.stack[i]
115: 
116:             file_name = self.__format_file_name(file_name)
117: 
118:             s += " - File '%s' (line %s, column %s)\n   Invocation to '%s(%s%s%s)'\n" % \
119:                  (file_name, line, column, function_name, self.__pretty_string_params(declared_arguments, arguments[0]),
120:                   self.__pretty_string_vargargs(arguments[1]), self.__pretty_string_kwargs(arguments[2]))
121:         s += "]"
122:         return s
123: 
124:     def __str__(self):
125:         return self.to_pretty_string()
126: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from singleton_copy import Singleton' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')
import_3396 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'singleton_copy')

if (type(import_3396) is not StypyTypeError):

    if (import_3396 != 'pyd_module'):
        __import__(import_3396)
        sys_modules_3397 = sys.modules[import_3396]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'singleton_copy', sys_modules_3397.module_type_store, module_type_store, ['Singleton'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_3397, sys_modules_3397.module_type_store, module_type_store)
    else:
        from singleton_copy import Singleton

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'singleton_copy', None, module_type_store, ['Singleton'], [Singleton])

else:
    # Assigning a type to the variable 'singleton_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'singleton_copy', import_3396)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')
import_3398 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy')

if (type(import_3398) is not StypyTypeError):

    if (import_3398 != 'pyd_module'):
        __import__(import_3398)
        sys_modules_3399 = sys.modules[import_3398]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', sys_modules_3399.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_3399, sys_modules_3399.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', import_3398)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from testing.test_programs.stypy_code_copy import stypy_copy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')
import_3400 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy')

if (type(import_3400) is not StypyTypeError):

    if (import_3400 != 'pyd_module'):
        __import__(import_3400)
        sys_modules_3401 = sys.modules[import_3400]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy', sys_modules_3401.module_type_store, module_type_store, ['stypy_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_3401, sys_modules_3401.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy import stypy_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy', None, module_type_store, ['stypy_copy'], [stypy_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy', import_3400)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')

# Declaration of the 'StackTrace' class
# Getting the type of 'Singleton' (line 5)
Singleton_3402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Singleton')

class StackTrace:
    str_3403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'str', '\n    This class allow TypeErrors to enhance the information they provide including the stack trace that lead to the\n    line that produced the time error. This way we can precisely trace inside the program where is the type error in\n    order to fix it. StackTrace information is built in the type inference program generated code and are accessed\n    through TypeErrors, so no direct usage of this class is expected. There is a single stack trace object per execution\n    flow.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
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

        
        # Assigning a List to a Attribute (line 15):
        
        # Assigning a List to a Attribute (line 15):
        
        # Obtaining an instance of the builtin type 'list' (line 15)
        list_3404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 21), 'list')
        # Adding type elements to the builtin type 'list' instance (line 15)
        
        # Getting the type of 'self' (line 15)
        self_3405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self')
        # Setting the type of the member 'stack' of a type (line 15)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_3405, 'stack', list_3404)
        
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
        module_type_store = module_type_store.open_function_context('set', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
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

        str_3406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, (-1)), 'str', '\n        Sets the stack trace information corresponding to a function call\n        :param file_name: .py file where the function is placed\n        :param line: Line when the function is declared\n        :param column: Column when the function is declared\n        :param function_name: Function name that is called\n        :param declared_arguments: Arguments declared in the function code\n        :param arguments: Passed arguments in the call\n        :return:\n        ')
        
        # Call to append(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Obtaining an instance of the builtin type 'tuple' (line 28)
        tuple_3410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 28)
        # Adding element type (line 28)
        # Getting the type of 'file_name' (line 28)
        file_name_3411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 27), 'file_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 27), tuple_3410, file_name_3411)
        # Adding element type (line 28)
        # Getting the type of 'line' (line 28)
        line_3412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 38), 'line', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 27), tuple_3410, line_3412)
        # Adding element type (line 28)
        # Getting the type of 'column' (line 28)
        column_3413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 44), 'column', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 27), tuple_3410, column_3413)
        # Adding element type (line 28)
        # Getting the type of 'function_name' (line 28)
        function_name_3414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 52), 'function_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 27), tuple_3410, function_name_3414)
        # Adding element type (line 28)
        # Getting the type of 'declared_arguments' (line 28)
        declared_arguments_3415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 67), 'declared_arguments', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 27), tuple_3410, declared_arguments_3415)
        # Adding element type (line 28)
        # Getting the type of 'arguments' (line 28)
        arguments_3416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 87), 'arguments', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 27), tuple_3410, arguments_3416)
        
        # Processing the call keyword arguments (line 28)
        kwargs_3417 = {}
        # Getting the type of 'self' (line 28)
        self_3407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self', False)
        # Obtaining the member 'stack' of a type (line 28)
        stack_3408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_3407, 'stack')
        # Obtaining the member 'append' of a type (line 28)
        append_3409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), stack_3408, 'append')
        # Calling append(args, kwargs) (line 28)
        append_call_result_3418 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), append_3409, *[tuple_3410], **kwargs_3417)
        
        
        # ################# End of 'set(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_3419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3419)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set'
        return stypy_return_type_3419


    @norecursion
    def unset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'unset'
        module_type_store = module_type_store.open_function_context('unset', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
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

        str_3420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, (-1)), 'str', '\n        Pops the last added stack trace (at function exit)\n        :return:\n        ')
        
        # Call to pop(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_3424 = {}
        # Getting the type of 'self' (line 35)
        self_3421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self', False)
        # Obtaining the member 'stack' of a type (line 35)
        stack_3422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_3421, 'stack')
        # Obtaining the member 'pop' of a type (line 35)
        pop_3423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), stack_3422, 'pop')
        # Calling pop(args, kwargs) (line 35)
        pop_call_result_3425 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), pop_3423, *[], **kwargs_3424)
        
        
        # ################# End of 'unset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'unset' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_3426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3426)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'unset'
        return stypy_return_type_3426


    @norecursion
    def __format_file_name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__format_file_name'
        module_type_store = module_type_store.open_function_context('__format_file_name', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
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

        str_3427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, (-1)), 'str', '\n        Pretty-print the .py file name\n        :param file_name:\n        :return:\n        ')
        
        # Assigning a Subscript to a Name (line 43):
        
        # Assigning a Subscript to a Name (line 43):
        
        # Obtaining the type of the subscript
        int_3428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 41), 'int')
        
        # Call to split(...): (line 43)
        # Processing the call arguments (line 43)
        str_3431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 36), 'str', '/')
        # Processing the call keyword arguments (line 43)
        kwargs_3432 = {}
        # Getting the type of 'file_name' (line 43)
        file_name_3429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'file_name', False)
        # Obtaining the member 'split' of a type (line 43)
        split_3430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 20), file_name_3429, 'split')
        # Calling split(args, kwargs) (line 43)
        split_call_result_3433 = invoke(stypy.reporting.localization.Localization(__file__, 43, 20), split_3430, *[str_3431], **kwargs_3432)
        
        # Obtaining the member '__getitem__' of a type (line 43)
        getitem___3434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 20), split_call_result_3433, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 43)
        subscript_call_result_3435 = invoke(stypy.reporting.localization.Localization(__file__, 43, 20), getitem___3434, int_3428)
        
        # Assigning a type to the variable 'file_name' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'file_name', subscript_call_result_3435)
        
        # Assigning a Call to a Name (line 44):
        
        # Assigning a Call to a Name (line 44):
        
        # Call to replace(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'stypy_parameters_copy' (line 44)
        stypy_parameters_copy_3438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 38), 'stypy_parameters_copy', False)
        # Obtaining the member 'type_inference_file_postfix' of a type (line 44)
        type_inference_file_postfix_3439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 38), stypy_parameters_copy_3438, 'type_inference_file_postfix')
        str_3440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 89), 'str', '')
        # Processing the call keyword arguments (line 44)
        kwargs_3441 = {}
        # Getting the type of 'file_name' (line 44)
        file_name_3436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'file_name', False)
        # Obtaining the member 'replace' of a type (line 44)
        replace_3437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 20), file_name_3436, 'replace')
        # Calling replace(args, kwargs) (line 44)
        replace_call_result_3442 = invoke(stypy.reporting.localization.Localization(__file__, 44, 20), replace_3437, *[type_inference_file_postfix_3439, str_3440], **kwargs_3441)
        
        # Assigning a type to the variable 'file_name' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'file_name', replace_call_result_3442)
        
        # Assigning a Call to a Name (line 45):
        
        # Assigning a Call to a Name (line 45):
        
        # Call to replace(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'stypy_parameters_copy' (line 45)
        stypy_parameters_copy_3445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 38), 'stypy_parameters_copy', False)
        # Obtaining the member 'type_inference_file_directory_name' of a type (line 45)
        type_inference_file_directory_name_3446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 38), stypy_parameters_copy_3445, 'type_inference_file_directory_name')
        str_3447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 96), 'str', '')
        # Processing the call keyword arguments (line 45)
        kwargs_3448 = {}
        # Getting the type of 'file_name' (line 45)
        file_name_3443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'file_name', False)
        # Obtaining the member 'replace' of a type (line 45)
        replace_3444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 20), file_name_3443, 'replace')
        # Calling replace(args, kwargs) (line 45)
        replace_call_result_3449 = invoke(stypy.reporting.localization.Localization(__file__, 45, 20), replace_3444, *[type_inference_file_directory_name_3446, str_3447], **kwargs_3448)
        
        # Assigning a type to the variable 'file_name' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'file_name', replace_call_result_3449)
        # Getting the type of 'file_name' (line 47)
        file_name_3450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'file_name')
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', file_name_3450)
        
        # ################# End of '__format_file_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__format_file_name' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_3451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3451)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__format_file_name'
        return stypy_return_type_3451


    @norecursion
    def __format_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__format_type'
        module_type_store = module_type_store.open_function_context('__format_type', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
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

        str_3452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', '\n        Pretty-prints types\n        :param type_:\n        :return:\n        ')
        
        # Call to isinstance(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'type_' (line 55)
        type__3454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 22), 'type_', False)
        # Getting the type of 'stypy_copy' (line 55)
        stypy_copy_3455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 29), 'stypy_copy', False)
        # Obtaining the member 'errors' of a type (line 55)
        errors_3456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 29), stypy_copy_3455, 'errors')
        # Obtaining the member 'type_error' of a type (line 55)
        type_error_3457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 29), errors_3456, 'type_error')
        # Obtaining the member 'TypeError' of a type (line 55)
        TypeError_3458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 29), type_error_3457, 'TypeError')
        # Processing the call keyword arguments (line 55)
        kwargs_3459 = {}
        # Getting the type of 'isinstance' (line 55)
        isinstance_3453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 55)
        isinstance_call_result_3460 = invoke(stypy.reporting.localization.Localization(__file__, 55, 11), isinstance_3453, *[type__3454, TypeError_3458], **kwargs_3459)
        
        # Testing if the type of an if condition is none (line 55)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 55, 8), isinstance_call_result_3460):
            pass
        else:
            
            # Testing the type of an if condition (line 55)
            if_condition_3461 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 8), isinstance_call_result_3460)
            # Assigning a type to the variable 'if_condition_3461' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'if_condition_3461', if_condition_3461)
            # SSA begins for if statement (line 55)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_3462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 19), 'str', 'TypeError')
            # Assigning a type to the variable 'stypy_return_type' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'stypy_return_type', str_3462)
            # SSA join for if statement (line 55)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to str(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'type_' (line 57)
        type__3464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'type_', False)
        # Processing the call keyword arguments (line 57)
        kwargs_3465 = {}
        # Getting the type of 'str' (line 57)
        str_3463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'str', False)
        # Calling str(args, kwargs) (line 57)
        str_call_result_3466 = invoke(stypy.reporting.localization.Localization(__file__, 57, 15), str_3463, *[type__3464], **kwargs_3465)
        
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', str_call_result_3466)
        
        # ################# End of '__format_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__format_type' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_3467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3467)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__format_type'
        return stypy_return_type_3467


    @norecursion
    def __pretty_string_params(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__pretty_string_params'
        module_type_store = module_type_store.open_function_context('__pretty_string_params', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
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

        str_3468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, (-1)), 'str', '\n        Pretty-prints function parameters\n        :param declared_arguments:\n        :param arguments:\n        :return:\n        ')
        
        # Assigning a Call to a Name (line 66):
        
        # Assigning a Call to a Name (line 66):
        
        # Call to zip(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'declared_arguments' (line 66)
        declared_arguments_3470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'declared_arguments', False)
        # Getting the type of 'arguments' (line 66)
        arguments_3471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 41), 'arguments', False)
        # Processing the call keyword arguments (line 66)
        kwargs_3472 = {}
        # Getting the type of 'zip' (line 66)
        zip_3469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'zip', False)
        # Calling zip(args, kwargs) (line 66)
        zip_call_result_3473 = invoke(stypy.reporting.localization.Localization(__file__, 66, 17), zip_3469, *[declared_arguments_3470, arguments_3471], **kwargs_3472)
        
        # Assigning a type to the variable 'zipped' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'zipped', zip_call_result_3473)
        
        # Assigning a Str to a Name (line 67):
        
        # Assigning a Str to a Name (line 67):
        str_3474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 18), 'str', '')
        # Assigning a type to the variable 'ret_str' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'ret_str', str_3474)
        
        # Getting the type of 'zipped' (line 68)
        zipped_3475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 22), 'zipped')
        # Assigning a type to the variable 'zipped_3475' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'zipped_3475', zipped_3475)
        # Testing if the for loop is going to be iterated (line 68)
        # Testing the type of a for loop iterable (line 68)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 68, 8), zipped_3475)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 68, 8), zipped_3475):
            # Getting the type of the for loop variable (line 68)
            for_loop_var_3476 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 68, 8), zipped_3475)
            # Assigning a type to the variable 'tuple_' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tuple_', for_loop_var_3476)
            # SSA begins for a for statement (line 68)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'ret_str' (line 69)
            ret_str_3477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'ret_str')
            
            # Obtaining the type of the subscript
            int_3478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 30), 'int')
            # Getting the type of 'tuple_' (line 69)
            tuple__3479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), 'tuple_')
            # Obtaining the member '__getitem__' of a type (line 69)
            getitem___3480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 23), tuple__3479, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 69)
            subscript_call_result_3481 = invoke(stypy.reporting.localization.Localization(__file__, 69, 23), getitem___3480, int_3478)
            
            str_3482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 35), 'str', ': ')
            # Applying the binary operator '+' (line 69)
            result_add_3483 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 23), '+', subscript_call_result_3481, str_3482)
            
            
            # Call to __format_type(...): (line 69)
            # Processing the call arguments (line 69)
            
            # Obtaining the type of the subscript
            int_3486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 68), 'int')
            # Getting the type of 'tuple_' (line 69)
            tuple__3487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 61), 'tuple_', False)
            # Obtaining the member '__getitem__' of a type (line 69)
            getitem___3488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 61), tuple__3487, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 69)
            subscript_call_result_3489 = invoke(stypy.reporting.localization.Localization(__file__, 69, 61), getitem___3488, int_3486)
            
            # Processing the call keyword arguments (line 69)
            kwargs_3490 = {}
            # Getting the type of 'self' (line 69)
            self_3484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 42), 'self', False)
            # Obtaining the member '__format_type' of a type (line 69)
            format_type_3485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 42), self_3484, '__format_type')
            # Calling __format_type(args, kwargs) (line 69)
            format_type_call_result_3491 = invoke(stypy.reporting.localization.Localization(__file__, 69, 42), format_type_3485, *[subscript_call_result_3489], **kwargs_3490)
            
            # Applying the binary operator '+' (line 69)
            result_add_3492 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 40), '+', result_add_3483, format_type_call_result_3491)
            
            str_3493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 74), 'str', ', ')
            # Applying the binary operator '+' (line 69)
            result_add_3494 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 72), '+', result_add_3492, str_3493)
            
            # Applying the binary operator '+=' (line 69)
            result_iadd_3495 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 12), '+=', ret_str_3477, result_add_3494)
            # Assigning a type to the variable 'ret_str' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'ret_str', result_iadd_3495)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining the type of the subscript
        int_3496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 24), 'int')
        slice_3497 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 71, 15), None, int_3496, None)
        # Getting the type of 'ret_str' (line 71)
        ret_str_3498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 'ret_str')
        # Obtaining the member '__getitem__' of a type (line 71)
        getitem___3499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 15), ret_str_3498, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 71)
        subscript_call_result_3500 = invoke(stypy.reporting.localization.Localization(__file__, 71, 15), getitem___3499, slice_3497)
        
        # Assigning a type to the variable 'stypy_return_type' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'stypy_return_type', subscript_call_result_3500)
        
        # ################# End of '__pretty_string_params(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__pretty_string_params' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_3501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3501)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__pretty_string_params'
        return stypy_return_type_3501


    @norecursion
    def __pretty_string_vargargs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__pretty_string_vargargs'
        module_type_store = module_type_store.open_function_context('__pretty_string_vargargs', 73, 4, False)
        # Assigning a type to the variable 'self' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'self', type_of_self)
        
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

        str_3502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, (-1)), 'str', '\n        Pretty-prints the variable list of arguments of a function\n        :param arguments:\n        :return:\n        ')
        
        
        # Call to len(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'arguments' (line 79)
        arguments_3504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'arguments', False)
        # Processing the call keyword arguments (line 79)
        kwargs_3505 = {}
        # Getting the type of 'len' (line 79)
        len_3503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'len', False)
        # Calling len(args, kwargs) (line 79)
        len_call_result_3506 = invoke(stypy.reporting.localization.Localization(__file__, 79, 11), len_3503, *[arguments_3504], **kwargs_3505)
        
        int_3507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 29), 'int')
        # Applying the binary operator '==' (line 79)
        result_eq_3508 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 11), '==', len_call_result_3506, int_3507)
        
        # Testing if the type of an if condition is none (line 79)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 79, 8), result_eq_3508):
            pass
        else:
            
            # Testing the type of an if condition (line 79)
            if_condition_3509 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 8), result_eq_3508)
            # Assigning a type to the variable 'if_condition_3509' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'if_condition_3509', if_condition_3509)
            # SSA begins for if statement (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_3510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 19), 'str', '')
            # Assigning a type to the variable 'stypy_return_type' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'stypy_return_type', str_3510)
            # SSA join for if statement (line 79)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Str to a Name (line 82):
        
        # Assigning a Str to a Name (line 82):
        str_3511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'str', ', *starargs=[')
        # Assigning a type to the variable 'ret_str' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'ret_str', str_3511)
        
        # Getting the type of 'arguments' (line 83)
        arguments_3512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'arguments')
        # Assigning a type to the variable 'arguments_3512' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'arguments_3512', arguments_3512)
        # Testing if the for loop is going to be iterated (line 83)
        # Testing the type of a for loop iterable (line 83)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 83, 8), arguments_3512)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 83, 8), arguments_3512):
            # Getting the type of the for loop variable (line 83)
            for_loop_var_3513 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 83, 8), arguments_3512)
            # Assigning a type to the variable 'arg' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'arg', for_loop_var_3513)
            # SSA begins for a for statement (line 83)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'ret_str' (line 84)
            ret_str_3514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'ret_str')
            
            # Call to __format_type(...): (line 84)
            # Processing the call arguments (line 84)
            # Getting the type of 'arg' (line 84)
            arg_3517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 42), 'arg', False)
            # Processing the call keyword arguments (line 84)
            kwargs_3518 = {}
            # Getting the type of 'self' (line 84)
            self_3515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'self', False)
            # Obtaining the member '__format_type' of a type (line 84)
            format_type_3516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 23), self_3515, '__format_type')
            # Calling __format_type(args, kwargs) (line 84)
            format_type_call_result_3519 = invoke(stypy.reporting.localization.Localization(__file__, 84, 23), format_type_3516, *[arg_3517], **kwargs_3518)
            
            str_3520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 49), 'str', ', ')
            # Applying the binary operator '+' (line 84)
            result_add_3521 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 23), '+', format_type_call_result_3519, str_3520)
            
            # Applying the binary operator '+=' (line 84)
            result_iadd_3522 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 12), '+=', ret_str_3514, result_add_3521)
            # Assigning a type to the variable 'ret_str' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'ret_str', result_iadd_3522)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining the type of the subscript
        int_3523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 24), 'int')
        slice_3524 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 86, 15), None, int_3523, None)
        # Getting the type of 'ret_str' (line 86)
        ret_str_3525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'ret_str')
        # Obtaining the member '__getitem__' of a type (line 86)
        getitem___3526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 15), ret_str_3525, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 86)
        subscript_call_result_3527 = invoke(stypy.reporting.localization.Localization(__file__, 86, 15), getitem___3526, slice_3524)
        
        str_3528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 30), 'str', ']')
        # Applying the binary operator '+' (line 86)
        result_add_3529 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 15), '+', subscript_call_result_3527, str_3528)
        
        # Assigning a type to the variable 'stypy_return_type' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'stypy_return_type', result_add_3529)
        
        # ################# End of '__pretty_string_vargargs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__pretty_string_vargargs' in the type store
        # Getting the type of 'stypy_return_type' (line 73)
        stypy_return_type_3530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3530)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__pretty_string_vargargs'
        return stypy_return_type_3530


    @norecursion
    def __pretty_string_kwargs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__pretty_string_kwargs'
        module_type_store = module_type_store.open_function_context('__pretty_string_kwargs', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
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

        str_3531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, (-1)), 'str', '\n        Pretty-prints the keyword arguments of a function\n        :param arguments:\n        :return:\n        ')
        
        
        # Call to len(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'arguments' (line 94)
        arguments_3533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'arguments', False)
        # Processing the call keyword arguments (line 94)
        kwargs_3534 = {}
        # Getting the type of 'len' (line 94)
        len_3532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'len', False)
        # Calling len(args, kwargs) (line 94)
        len_call_result_3535 = invoke(stypy.reporting.localization.Localization(__file__, 94, 11), len_3532, *[arguments_3533], **kwargs_3534)
        
        int_3536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 29), 'int')
        # Applying the binary operator '==' (line 94)
        result_eq_3537 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 11), '==', len_call_result_3535, int_3536)
        
        # Testing if the type of an if condition is none (line 94)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 94, 8), result_eq_3537):
            pass
        else:
            
            # Testing the type of an if condition (line 94)
            if_condition_3538 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 8), result_eq_3537)
            # Assigning a type to the variable 'if_condition_3538' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'if_condition_3538', if_condition_3538)
            # SSA begins for if statement (line 94)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_3539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 19), 'str', '')
            # Assigning a type to the variable 'stypy_return_type' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'stypy_return_type', str_3539)
            # SSA join for if statement (line 94)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Str to a Name (line 97):
        
        # Assigning a Str to a Name (line 97):
        str_3540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 18), 'str', ', **kwargs={')
        # Assigning a type to the variable 'ret_str' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'ret_str', str_3540)
        
        
        # Call to items(...): (line 98)
        # Processing the call keyword arguments (line 98)
        kwargs_3543 = {}
        # Getting the type of 'arguments' (line 98)
        arguments_3541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'arguments', False)
        # Obtaining the member 'items' of a type (line 98)
        items_3542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 24), arguments_3541, 'items')
        # Calling items(args, kwargs) (line 98)
        items_call_result_3544 = invoke(stypy.reporting.localization.Localization(__file__, 98, 24), items_3542, *[], **kwargs_3543)
        
        # Assigning a type to the variable 'items_call_result_3544' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'items_call_result_3544', items_call_result_3544)
        # Testing if the for loop is going to be iterated (line 98)
        # Testing the type of a for loop iterable (line 98)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 98, 8), items_call_result_3544)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 98, 8), items_call_result_3544):
            # Getting the type of the for loop variable (line 98)
            for_loop_var_3545 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 98, 8), items_call_result_3544)
            # Assigning a type to the variable 'key' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 8), for_loop_var_3545, 2, 0))
            # Assigning a type to the variable 'arg' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'arg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 8), for_loop_var_3545, 2, 1))
            # SSA begins for a for statement (line 98)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'ret_str' (line 99)
            ret_str_3546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'ret_str')
            
            # Call to str(...): (line 99)
            # Processing the call arguments (line 99)
            # Getting the type of 'key' (line 99)
            key_3548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 27), 'key', False)
            # Processing the call keyword arguments (line 99)
            kwargs_3549 = {}
            # Getting the type of 'str' (line 99)
            str_3547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 23), 'str', False)
            # Calling str(args, kwargs) (line 99)
            str_call_result_3550 = invoke(stypy.reporting.localization.Localization(__file__, 99, 23), str_3547, *[key_3548], **kwargs_3549)
            
            str_3551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 34), 'str', ': ')
            # Applying the binary operator '+' (line 99)
            result_add_3552 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 23), '+', str_call_result_3550, str_3551)
            
            
            # Call to __format_type(...): (line 99)
            # Processing the call arguments (line 99)
            # Getting the type of 'arg' (line 99)
            arg_3555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 60), 'arg', False)
            # Processing the call keyword arguments (line 99)
            kwargs_3556 = {}
            # Getting the type of 'self' (line 99)
            self_3553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 41), 'self', False)
            # Obtaining the member '__format_type' of a type (line 99)
            format_type_3554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 41), self_3553, '__format_type')
            # Calling __format_type(args, kwargs) (line 99)
            format_type_call_result_3557 = invoke(stypy.reporting.localization.Localization(__file__, 99, 41), format_type_3554, *[arg_3555], **kwargs_3556)
            
            # Applying the binary operator '+' (line 99)
            result_add_3558 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 39), '+', result_add_3552, format_type_call_result_3557)
            
            str_3559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 67), 'str', ', ')
            # Applying the binary operator '+' (line 99)
            result_add_3560 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 65), '+', result_add_3558, str_3559)
            
            # Applying the binary operator '+=' (line 99)
            result_iadd_3561 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 12), '+=', ret_str_3546, result_add_3560)
            # Assigning a type to the variable 'ret_str' (line 99)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'ret_str', result_iadd_3561)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining the type of the subscript
        int_3562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 24), 'int')
        slice_3563 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 101, 15), None, int_3562, None)
        # Getting the type of 'ret_str' (line 101)
        ret_str_3564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'ret_str')
        # Obtaining the member '__getitem__' of a type (line 101)
        getitem___3565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 15), ret_str_3564, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 101)
        subscript_call_result_3566 = invoke(stypy.reporting.localization.Localization(__file__, 101, 15), getitem___3565, slice_3563)
        
        str_3567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 30), 'str', '}')
        # Applying the binary operator '+' (line 101)
        result_add_3568 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 15), '+', subscript_call_result_3566, str_3567)
        
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'stypy_return_type', result_add_3568)
        
        # ################# End of '__pretty_string_kwargs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__pretty_string_kwargs' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_3569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3569)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__pretty_string_kwargs'
        return stypy_return_type_3569


    @norecursion
    def to_pretty_string(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'to_pretty_string'
        module_type_store = module_type_store.open_function_context('to_pretty_string', 103, 4, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'self', type_of_self)
        
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

        str_3570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, (-1)), 'str', '\n        Prints each called function header and its parameters in a human-readable way, comprising the full stack\n        trace information stored in this object.\n        :return:\n        ')
        
        
        # Call to len(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'self' (line 109)
        self_3572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'self', False)
        # Obtaining the member 'stack' of a type (line 109)
        stack_3573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 15), self_3572, 'stack')
        # Processing the call keyword arguments (line 109)
        kwargs_3574 = {}
        # Getting the type of 'len' (line 109)
        len_3571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'len', False)
        # Calling len(args, kwargs) (line 109)
        len_call_result_3575 = invoke(stypy.reporting.localization.Localization(__file__, 109, 11), len_3571, *[stack_3573], **kwargs_3574)
        
        int_3576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 30), 'int')
        # Applying the binary operator '==' (line 109)
        result_eq_3577 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 11), '==', len_call_result_3575, int_3576)
        
        # Testing if the type of an if condition is none (line 109)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 109, 8), result_eq_3577):
            pass
        else:
            
            # Testing the type of an if condition (line 109)
            if_condition_3578 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 8), result_eq_3577)
            # Assigning a type to the variable 'if_condition_3578' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'if_condition_3578', if_condition_3578)
            # SSA begins for if statement (line 109)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_3579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 19), 'str', '')
            # Assigning a type to the variable 'stypy_return_type' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'stypy_return_type', str_3579)
            # SSA join for if statement (line 109)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Str to a Name (line 111):
        
        # Assigning a Str to a Name (line 111):
        str_3580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 12), 'str', 'Call stack: [\n')
        # Assigning a type to the variable 's' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 's', str_3580)
        
        
        # Call to xrange(...): (line 113)
        # Processing the call arguments (line 113)
        
        # Call to len(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'self' (line 113)
        self_3583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 28), 'self', False)
        # Obtaining the member 'stack' of a type (line 113)
        stack_3584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 28), self_3583, 'stack')
        # Processing the call keyword arguments (line 113)
        kwargs_3585 = {}
        # Getting the type of 'len' (line 113)
        len_3582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 24), 'len', False)
        # Calling len(args, kwargs) (line 113)
        len_call_result_3586 = invoke(stypy.reporting.localization.Localization(__file__, 113, 24), len_3582, *[stack_3584], **kwargs_3585)
        
        int_3587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 42), 'int')
        # Applying the binary operator '-' (line 113)
        result_sub_3588 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 24), '-', len_call_result_3586, int_3587)
        
        int_3589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 45), 'int')
        int_3590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 49), 'int')
        # Processing the call keyword arguments (line 113)
        kwargs_3591 = {}
        # Getting the type of 'xrange' (line 113)
        xrange_3581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 113)
        xrange_call_result_3592 = invoke(stypy.reporting.localization.Localization(__file__, 113, 17), xrange_3581, *[result_sub_3588, int_3589, int_3590], **kwargs_3591)
        
        # Assigning a type to the variable 'xrange_call_result_3592' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'xrange_call_result_3592', xrange_call_result_3592)
        # Testing if the for loop is going to be iterated (line 113)
        # Testing the type of a for loop iterable (line 113)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 113, 8), xrange_call_result_3592)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 113, 8), xrange_call_result_3592):
            # Getting the type of the for loop variable (line 113)
            for_loop_var_3593 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 113, 8), xrange_call_result_3592)
            # Assigning a type to the variable 'i' (line 113)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'i', for_loop_var_3593)
            # SSA begins for a for statement (line 113)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Tuple (line 114):
            
            # Assigning a Subscript to a Name (line 114):
            
            # Obtaining the type of the subscript
            int_3594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 114)
            i_3595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 95), 'i')
            # Getting the type of 'self' (line 114)
            self_3596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 84), 'self')
            # Obtaining the member 'stack' of a type (line 114)
            stack_3597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 84), self_3596, 'stack')
            # Obtaining the member '__getitem__' of a type (line 114)
            getitem___3598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 84), stack_3597, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 114)
            subscript_call_result_3599 = invoke(stypy.reporting.localization.Localization(__file__, 114, 84), getitem___3598, i_3595)
            
            # Obtaining the member '__getitem__' of a type (line 114)
            getitem___3600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), subscript_call_result_3599, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 114)
            subscript_call_result_3601 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), getitem___3600, int_3594)
            
            # Assigning a type to the variable 'tuple_var_assignment_3390' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple_var_assignment_3390', subscript_call_result_3601)
            
            # Assigning a Subscript to a Name (line 114):
            
            # Obtaining the type of the subscript
            int_3602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 114)
            i_3603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 95), 'i')
            # Getting the type of 'self' (line 114)
            self_3604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 84), 'self')
            # Obtaining the member 'stack' of a type (line 114)
            stack_3605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 84), self_3604, 'stack')
            # Obtaining the member '__getitem__' of a type (line 114)
            getitem___3606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 84), stack_3605, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 114)
            subscript_call_result_3607 = invoke(stypy.reporting.localization.Localization(__file__, 114, 84), getitem___3606, i_3603)
            
            # Obtaining the member '__getitem__' of a type (line 114)
            getitem___3608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), subscript_call_result_3607, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 114)
            subscript_call_result_3609 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), getitem___3608, int_3602)
            
            # Assigning a type to the variable 'tuple_var_assignment_3391' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple_var_assignment_3391', subscript_call_result_3609)
            
            # Assigning a Subscript to a Name (line 114):
            
            # Obtaining the type of the subscript
            int_3610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 114)
            i_3611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 95), 'i')
            # Getting the type of 'self' (line 114)
            self_3612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 84), 'self')
            # Obtaining the member 'stack' of a type (line 114)
            stack_3613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 84), self_3612, 'stack')
            # Obtaining the member '__getitem__' of a type (line 114)
            getitem___3614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 84), stack_3613, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 114)
            subscript_call_result_3615 = invoke(stypy.reporting.localization.Localization(__file__, 114, 84), getitem___3614, i_3611)
            
            # Obtaining the member '__getitem__' of a type (line 114)
            getitem___3616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), subscript_call_result_3615, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 114)
            subscript_call_result_3617 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), getitem___3616, int_3610)
            
            # Assigning a type to the variable 'tuple_var_assignment_3392' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple_var_assignment_3392', subscript_call_result_3617)
            
            # Assigning a Subscript to a Name (line 114):
            
            # Obtaining the type of the subscript
            int_3618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 114)
            i_3619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 95), 'i')
            # Getting the type of 'self' (line 114)
            self_3620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 84), 'self')
            # Obtaining the member 'stack' of a type (line 114)
            stack_3621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 84), self_3620, 'stack')
            # Obtaining the member '__getitem__' of a type (line 114)
            getitem___3622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 84), stack_3621, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 114)
            subscript_call_result_3623 = invoke(stypy.reporting.localization.Localization(__file__, 114, 84), getitem___3622, i_3619)
            
            # Obtaining the member '__getitem__' of a type (line 114)
            getitem___3624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), subscript_call_result_3623, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 114)
            subscript_call_result_3625 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), getitem___3624, int_3618)
            
            # Assigning a type to the variable 'tuple_var_assignment_3393' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple_var_assignment_3393', subscript_call_result_3625)
            
            # Assigning a Subscript to a Name (line 114):
            
            # Obtaining the type of the subscript
            int_3626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 114)
            i_3627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 95), 'i')
            # Getting the type of 'self' (line 114)
            self_3628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 84), 'self')
            # Obtaining the member 'stack' of a type (line 114)
            stack_3629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 84), self_3628, 'stack')
            # Obtaining the member '__getitem__' of a type (line 114)
            getitem___3630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 84), stack_3629, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 114)
            subscript_call_result_3631 = invoke(stypy.reporting.localization.Localization(__file__, 114, 84), getitem___3630, i_3627)
            
            # Obtaining the member '__getitem__' of a type (line 114)
            getitem___3632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), subscript_call_result_3631, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 114)
            subscript_call_result_3633 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), getitem___3632, int_3626)
            
            # Assigning a type to the variable 'tuple_var_assignment_3394' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple_var_assignment_3394', subscript_call_result_3633)
            
            # Assigning a Subscript to a Name (line 114):
            
            # Obtaining the type of the subscript
            int_3634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 114)
            i_3635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 95), 'i')
            # Getting the type of 'self' (line 114)
            self_3636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 84), 'self')
            # Obtaining the member 'stack' of a type (line 114)
            stack_3637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 84), self_3636, 'stack')
            # Obtaining the member '__getitem__' of a type (line 114)
            getitem___3638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 84), stack_3637, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 114)
            subscript_call_result_3639 = invoke(stypy.reporting.localization.Localization(__file__, 114, 84), getitem___3638, i_3635)
            
            # Obtaining the member '__getitem__' of a type (line 114)
            getitem___3640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), subscript_call_result_3639, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 114)
            subscript_call_result_3641 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), getitem___3640, int_3634)
            
            # Assigning a type to the variable 'tuple_var_assignment_3395' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple_var_assignment_3395', subscript_call_result_3641)
            
            # Assigning a Name to a Name (line 114):
            # Getting the type of 'tuple_var_assignment_3390' (line 114)
            tuple_var_assignment_3390_3642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple_var_assignment_3390')
            # Assigning a type to the variable 'file_name' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'file_name', tuple_var_assignment_3390_3642)
            
            # Assigning a Name to a Name (line 114):
            # Getting the type of 'tuple_var_assignment_3391' (line 114)
            tuple_var_assignment_3391_3643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple_var_assignment_3391')
            # Assigning a type to the variable 'line' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'line', tuple_var_assignment_3391_3643)
            
            # Assigning a Name to a Name (line 114):
            # Getting the type of 'tuple_var_assignment_3392' (line 114)
            tuple_var_assignment_3392_3644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple_var_assignment_3392')
            # Assigning a type to the variable 'column' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 29), 'column', tuple_var_assignment_3392_3644)
            
            # Assigning a Name to a Name (line 114):
            # Getting the type of 'tuple_var_assignment_3393' (line 114)
            tuple_var_assignment_3393_3645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple_var_assignment_3393')
            # Assigning a type to the variable 'function_name' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 37), 'function_name', tuple_var_assignment_3393_3645)
            
            # Assigning a Name to a Name (line 114):
            # Getting the type of 'tuple_var_assignment_3394' (line 114)
            tuple_var_assignment_3394_3646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple_var_assignment_3394')
            # Assigning a type to the variable 'declared_arguments' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 52), 'declared_arguments', tuple_var_assignment_3394_3646)
            
            # Assigning a Name to a Name (line 114):
            # Getting the type of 'tuple_var_assignment_3395' (line 114)
            tuple_var_assignment_3395_3647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'tuple_var_assignment_3395')
            # Assigning a type to the variable 'arguments' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 72), 'arguments', tuple_var_assignment_3395_3647)
            
            # Assigning a Call to a Name (line 116):
            
            # Assigning a Call to a Name (line 116):
            
            # Call to __format_file_name(...): (line 116)
            # Processing the call arguments (line 116)
            # Getting the type of 'file_name' (line 116)
            file_name_3650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 48), 'file_name', False)
            # Processing the call keyword arguments (line 116)
            kwargs_3651 = {}
            # Getting the type of 'self' (line 116)
            self_3648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'self', False)
            # Obtaining the member '__format_file_name' of a type (line 116)
            format_file_name_3649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 24), self_3648, '__format_file_name')
            # Calling __format_file_name(args, kwargs) (line 116)
            format_file_name_call_result_3652 = invoke(stypy.reporting.localization.Localization(__file__, 116, 24), format_file_name_3649, *[file_name_3650], **kwargs_3651)
            
            # Assigning a type to the variable 'file_name' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'file_name', format_file_name_call_result_3652)
            
            # Getting the type of 's' (line 118)
            s_3653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 's')
            str_3654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 17), 'str', " - File '%s' (line %s, column %s)\n   Invocation to '%s(%s%s%s)'\n")
            
            # Obtaining an instance of the builtin type 'tuple' (line 119)
            tuple_3655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 18), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 119)
            # Adding element type (line 119)
            # Getting the type of 'file_name' (line 119)
            file_name_3656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 18), 'file_name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 18), tuple_3655, file_name_3656)
            # Adding element type (line 119)
            # Getting the type of 'line' (line 119)
            line_3657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 29), 'line')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 18), tuple_3655, line_3657)
            # Adding element type (line 119)
            # Getting the type of 'column' (line 119)
            column_3658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 35), 'column')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 18), tuple_3655, column_3658)
            # Adding element type (line 119)
            # Getting the type of 'function_name' (line 119)
            function_name_3659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 43), 'function_name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 18), tuple_3655, function_name_3659)
            # Adding element type (line 119)
            
            # Call to __pretty_string_params(...): (line 119)
            # Processing the call arguments (line 119)
            # Getting the type of 'declared_arguments' (line 119)
            declared_arguments_3662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 86), 'declared_arguments', False)
            
            # Obtaining the type of the subscript
            int_3663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 116), 'int')
            # Getting the type of 'arguments' (line 119)
            arguments_3664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 106), 'arguments', False)
            # Obtaining the member '__getitem__' of a type (line 119)
            getitem___3665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 106), arguments_3664, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 119)
            subscript_call_result_3666 = invoke(stypy.reporting.localization.Localization(__file__, 119, 106), getitem___3665, int_3663)
            
            # Processing the call keyword arguments (line 119)
            kwargs_3667 = {}
            # Getting the type of 'self' (line 119)
            self_3660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 58), 'self', False)
            # Obtaining the member '__pretty_string_params' of a type (line 119)
            pretty_string_params_3661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 58), self_3660, '__pretty_string_params')
            # Calling __pretty_string_params(args, kwargs) (line 119)
            pretty_string_params_call_result_3668 = invoke(stypy.reporting.localization.Localization(__file__, 119, 58), pretty_string_params_3661, *[declared_arguments_3662, subscript_call_result_3666], **kwargs_3667)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 18), tuple_3655, pretty_string_params_call_result_3668)
            # Adding element type (line 119)
            
            # Call to __pretty_string_vargargs(...): (line 120)
            # Processing the call arguments (line 120)
            
            # Obtaining the type of the subscript
            int_3671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 58), 'int')
            # Getting the type of 'arguments' (line 120)
            arguments_3672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 48), 'arguments', False)
            # Obtaining the member '__getitem__' of a type (line 120)
            getitem___3673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 48), arguments_3672, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 120)
            subscript_call_result_3674 = invoke(stypy.reporting.localization.Localization(__file__, 120, 48), getitem___3673, int_3671)
            
            # Processing the call keyword arguments (line 120)
            kwargs_3675 = {}
            # Getting the type of 'self' (line 120)
            self_3669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 18), 'self', False)
            # Obtaining the member '__pretty_string_vargargs' of a type (line 120)
            pretty_string_vargargs_3670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 18), self_3669, '__pretty_string_vargargs')
            # Calling __pretty_string_vargargs(args, kwargs) (line 120)
            pretty_string_vargargs_call_result_3676 = invoke(stypy.reporting.localization.Localization(__file__, 120, 18), pretty_string_vargargs_3670, *[subscript_call_result_3674], **kwargs_3675)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 18), tuple_3655, pretty_string_vargargs_call_result_3676)
            # Adding element type (line 119)
            
            # Call to __pretty_string_kwargs(...): (line 120)
            # Processing the call arguments (line 120)
            
            # Obtaining the type of the subscript
            int_3679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 101), 'int')
            # Getting the type of 'arguments' (line 120)
            arguments_3680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 91), 'arguments', False)
            # Obtaining the member '__getitem__' of a type (line 120)
            getitem___3681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 91), arguments_3680, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 120)
            subscript_call_result_3682 = invoke(stypy.reporting.localization.Localization(__file__, 120, 91), getitem___3681, int_3679)
            
            # Processing the call keyword arguments (line 120)
            kwargs_3683 = {}
            # Getting the type of 'self' (line 120)
            self_3677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 63), 'self', False)
            # Obtaining the member '__pretty_string_kwargs' of a type (line 120)
            pretty_string_kwargs_3678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 63), self_3677, '__pretty_string_kwargs')
            # Calling __pretty_string_kwargs(args, kwargs) (line 120)
            pretty_string_kwargs_call_result_3684 = invoke(stypy.reporting.localization.Localization(__file__, 120, 63), pretty_string_kwargs_3678, *[subscript_call_result_3682], **kwargs_3683)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 18), tuple_3655, pretty_string_kwargs_call_result_3684)
            
            # Applying the binary operator '%' (line 118)
            result_mod_3685 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 17), '%', str_3654, tuple_3655)
            
            # Applying the binary operator '+=' (line 118)
            result_iadd_3686 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 12), '+=', s_3653, result_mod_3685)
            # Assigning a type to the variable 's' (line 118)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 's', result_iadd_3686)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 's' (line 121)
        s_3687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 's')
        str_3688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 13), 'str', ']')
        # Applying the binary operator '+=' (line 121)
        result_iadd_3689 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 8), '+=', s_3687, str_3688)
        # Assigning a type to the variable 's' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 's', result_iadd_3689)
        
        # Getting the type of 's' (line 122)
        s_3690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'stypy_return_type', s_3690)
        
        # ################# End of 'to_pretty_string(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'to_pretty_string' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_3691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3691)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'to_pretty_string'
        return stypy_return_type_3691


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 124, 4, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'self', type_of_self)
        
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

        
        # Call to to_pretty_string(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_3694 = {}
        # Getting the type of 'self' (line 125)
        self_3692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'self', False)
        # Obtaining the member 'to_pretty_string' of a type (line 125)
        to_pretty_string_3693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 15), self_3692, 'to_pretty_string')
        # Calling to_pretty_string(args, kwargs) (line 125)
        to_pretty_string_call_result_3695 = invoke(stypy.reporting.localization.Localization(__file__, 125, 15), to_pretty_string_3693, *[], **kwargs_3694)
        
        # Assigning a type to the variable 'stypy_return_type' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'stypy_return_type', to_pretty_string_call_result_3695)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 124)
        stypy_return_type_3696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3696)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_3696


# Assigning a type to the variable 'StackTrace' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'StackTrace', StackTrace)
# Getting the type of 'StackTrace' (line 5)
StackTrace_3697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'StackTrace')
class_3698 = invoke(stypy.reporting.localization.Localization(__file__, 5, 0), Singleton_3402, StackTrace_3697)
# Assigning a type to the variable 'StackTrace' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'StackTrace', class_3698)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
