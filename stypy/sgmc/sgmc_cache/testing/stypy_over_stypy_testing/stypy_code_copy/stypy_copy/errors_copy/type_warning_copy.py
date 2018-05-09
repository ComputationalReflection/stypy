
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from stypy_copy.reporting_copy.module_line_numbering_copy import ModuleLineNumbering
2: from stypy_copy import errors_copy
3: from stypy_copy import stypy_parameters_copy
4: 
5: 
6: class TypeWarning:
7:     '''
8:     Class to model type warnings. This means that the type of a variable has been detected as invalid, but other
9:     options for types of these variables exist within the execution flow that result in a correct behavior. Therefore
10:     all the incorrect behaviors among an operation are warnings because there is a combination that can be valid.
11:     '''
12: 
13:     # In strict mode all warnings are errors and warnings itself do not exist. If we try to create one, a TypeError
14:     # is returned instead.
15:     warnings_as_errors = False
16: 
17:     # All the warnings produced during the execution are stored here
18:     warnings = []
19: 
20:     def __init__(self, localization, msg, prints_msg=True):
21:         '''
22:         Creates a warning with the provided message.
23:         :param localization: Caller information
24:         :param msg: Warning message
25:         :param prints_msg: As TypeErrors, TypeWarnings can also be silent if reporting them is not activated
26:         :return:
27:         '''
28:         self.msg = msg
29:         self.localization = localization
30: 
31:         if prints_msg:
32:             TypeWarning.warnings.append(self)
33:             # Create the message here to capture the execution point, as stack traces are dynamic.
34:             self.warn_msg = self.__msg()
35: 
36:     @staticmethod
37:     def instance(localization, msg, prints_msg=True):
38:         '''
39:         Creates a TypeWarning instance (or a TypeError if strict mode is used).
40:         :param localization: Caller information
41:         :param msg: Warning message
42:         :param prints_msg: Silent warning?
43:         :return:
44:         '''
45:         if TypeWarning.warnings_as_errors:
46:             return errors_copy.type_error_copy.TypeError(localization, msg, prints_msg)
47:         else:
48:             return TypeWarning(localization, msg, prints_msg)
49: 
50:     def __str__(self):
51:         return self.warn_msg
52: 
53:     def __format_file_name(self):
54:         '''
55:         Pretty-prints file name
56:         :return:
57:         '''
58:         file_name = self.localization.file_name.split('/')[-1]
59:         file_name = file_name.split('\\')[-1]
60:         file_name = file_name.replace(stypy_parameters_copy.type_inference_file_postfix, '')
61:         file_name = file_name.replace(stypy_parameters_copy.type_inference_file_directory_name, '')
62: 
63:         return file_name
64: 
65:     def __msg(self):
66:         '''
67:         Composes the full warning message, using the message, the localization, current file name and
68:         the stack trace. If available, it also displays the source code line when the warning is produced and a
69:         ^ marker indicating the position within the warning line.
70:         :return:
71:         '''
72:         file_name = self.__format_file_name()
73: 
74:         source_code = ModuleLineNumbering.get_line_from_module_code(self.localization.file_name, self.localization.line)
75:         col_offset = ModuleLineNumbering.get_column_from_module_code(self.localization.file_name,
76:                                                                      self.localization.line, self.localization.column)
77:         if source_code is not None:
78:             return "Warning in file '%s' (line %s, column %s):\n%s\n%s\n\t%s.\n\n%s" % \
79:                    (file_name, self.localization.line, self.localization.column,
80:                     source_code, "" + col_offset,
81:                     self.msg.strip(), self.localization.stack_trace)
82: 
83:         return "Warning in file '%s' (line %s, column %s):\n%s.\n\n%s" % \
84:                (file_name, self.localization.line, self.localization.column,
85:                 self.msg, self.localization.stack_trace)
86: 
87:     @staticmethod
88:     def print_warning_msgs():
89:         '''
90:         Prints all the warning messages that were produced during a program analysis. Just for debugging
91:         :return:
92:         '''
93:         for err in TypeWarning.warnings:
94:             print err
95: 
96:     @staticmethod
97:     def reset_warning_msgs():
98:         TypeWarning.warnings = []
99: 
100:     @staticmethod
101:     def get_warning_msgs():
102:         '''
103:         Gets all the warning messages that were produced during a program analysis.
104:         :return: All the errors, sorted by line number
105:         '''
106:         return sorted(TypeWarning.warnings, key=lambda warning: warning.localization.line)
107: 
108:     # TODO: Remove?
109:     # @classmethod
110:     # def set_warning_msgs(cls, warn_list):
111:     #     '''
112:     #     Substitute the warning messages list by the provided one
113:     #     :param warn_list: New warning list
114:     #     :return:
115:     #     '''
116:     #     TypeWarning.warnings = warn_list
117: 
118:     # @classmethod
119:     # def clone_existing_warnings(cls):
120:     #     '''
121:     #     Clones the warning list
122:     #     :return:
123:     #     '''
124:     #     result = list()
125:     #
126:     #     for warning in TypeWarning.warnings:
127:     #         result.append(warning)
128:     #
129:     #     return result
130: 
131: 
132: class UnreferencedLocalVariableTypeWarning(TypeWarning):
133:     pass
134:     '''
135:     This special type of warning is only used if coding advices are activated. It models those cases when a global
136:     variable is read and later on is written to without using the global keyword. Python decides to report an error
137:     in this case, but in the source line that reads the value instead of the source line that write a value to the
138:     variable. A coding advice is generated if this kind of programming pattern is detected within the program.
139:     '''
140:     def __init__(self, localization, name, context):
141:         TypeWarning.__init__(self, localization,
142:                              "Read access detected over a global name '{0}'. Any attempt of writing to this "
143:                              "name without using the 'global' keyword first will result into an 'UnboundLocalError: "
144:                              "local variable '{0}' referenced before assignment' runtime error".format(name))
145:         self.name = name
146:         self.context = context
147: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from stypy_copy.reporting_copy.module_line_numbering_copy import ModuleLineNumbering' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/errors_copy/')
import_3682 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.reporting_copy.module_line_numbering_copy')

if (type(import_3682) is not StypyTypeError):

    if (import_3682 != 'pyd_module'):
        __import__(import_3682)
        sys_modules_3683 = sys.modules[import_3682]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.reporting_copy.module_line_numbering_copy', sys_modules_3683.module_type_store, module_type_store, ['ModuleLineNumbering'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_3683, sys_modules_3683.module_type_store, module_type_store)
    else:
        from stypy_copy.reporting_copy.module_line_numbering_copy import ModuleLineNumbering

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.reporting_copy.module_line_numbering_copy', None, module_type_store, ['ModuleLineNumbering'], [ModuleLineNumbering])

else:
    # Assigning a type to the variable 'stypy_copy.reporting_copy.module_line_numbering_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.reporting_copy.module_line_numbering_copy', import_3682)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/errors_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from stypy_copy import errors_copy' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/errors_copy/')
import_3684 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy')

if (type(import_3684) is not StypyTypeError):

    if (import_3684 != 'pyd_module'):
        __import__(import_3684)
        sys_modules_3685 = sys.modules[import_3684]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy', sys_modules_3685.module_type_store, module_type_store, ['errors_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_3685, sys_modules_3685.module_type_store, module_type_store)
    else:
        from stypy_copy import errors_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy', None, module_type_store, ['errors_copy'], [errors_copy])

else:
    # Assigning a type to the variable 'stypy_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy', import_3684)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/errors_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from stypy_copy import stypy_parameters_copy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/errors_copy/')
import_3686 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy')

if (type(import_3686) is not StypyTypeError):

    if (import_3686 != 'pyd_module'):
        __import__(import_3686)
        sys_modules_3687 = sys.modules[import_3686]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy', sys_modules_3687.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_3687, sys_modules_3687.module_type_store, module_type_store)
    else:
        from stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'stypy_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy', import_3686)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/errors_copy/')

# Declaration of the 'TypeWarning' class

class TypeWarning:
    str_3688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, (-1)), 'str', '\n    Class to model type warnings. This means that the type of a variable has been detected as invalid, but other\n    options for types of these variables exist within the execution flow that result in a correct behavior. Therefore\n    all the incorrect behaviors among an operation are warnings because there is a combination that can be valid.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 20)
        True_3689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 53), 'True')
        defaults = [True_3689]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeWarning.__init__', ['localization', 'msg', 'prints_msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['localization', 'msg', 'prints_msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_3690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, (-1)), 'str', '\n        Creates a warning with the provided message.\n        :param localization: Caller information\n        :param msg: Warning message\n        :param prints_msg: As TypeErrors, TypeWarnings can also be silent if reporting them is not activated\n        :return:\n        ')
        
        # Assigning a Name to a Attribute (line 28):
        # Getting the type of 'msg' (line 28)
        msg_3691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'msg')
        # Getting the type of 'self' (line 28)
        self_3692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'msg' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_3692, 'msg', msg_3691)
        
        # Assigning a Name to a Attribute (line 29):
        # Getting the type of 'localization' (line 29)
        localization_3693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 28), 'localization')
        # Getting the type of 'self' (line 29)
        self_3694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'localization' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_3694, 'localization', localization_3693)
        # Getting the type of 'prints_msg' (line 31)
        prints_msg_3695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'prints_msg')
        # Testing if the type of an if condition is none (line 31)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 31, 8), prints_msg_3695):
            pass
        else:
            
            # Testing the type of an if condition (line 31)
            if_condition_3696 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 8), prints_msg_3695)
            # Assigning a type to the variable 'if_condition_3696' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'if_condition_3696', if_condition_3696)
            # SSA begins for if statement (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 32)
            # Processing the call arguments (line 32)
            # Getting the type of 'self' (line 32)
            self_3700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 40), 'self', False)
            # Processing the call keyword arguments (line 32)
            kwargs_3701 = {}
            # Getting the type of 'TypeWarning' (line 32)
            TypeWarning_3697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'TypeWarning', False)
            # Obtaining the member 'warnings' of a type (line 32)
            warnings_3698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), TypeWarning_3697, 'warnings')
            # Obtaining the member 'append' of a type (line 32)
            append_3699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), warnings_3698, 'append')
            # Calling append(args, kwargs) (line 32)
            append_call_result_3702 = invoke(stypy.reporting.localization.Localization(__file__, 32, 12), append_3699, *[self_3700], **kwargs_3701)
            
            
            # Assigning a Call to a Attribute (line 34):
            
            # Call to __msg(...): (line 34)
            # Processing the call keyword arguments (line 34)
            kwargs_3705 = {}
            # Getting the type of 'self' (line 34)
            self_3703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 28), 'self', False)
            # Obtaining the member '__msg' of a type (line 34)
            msg_3704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 28), self_3703, '__msg')
            # Calling __msg(args, kwargs) (line 34)
            msg_call_result_3706 = invoke(stypy.reporting.localization.Localization(__file__, 34, 28), msg_3704, *[], **kwargs_3705)
            
            # Getting the type of 'self' (line 34)
            self_3707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'self')
            # Setting the type of the member 'warn_msg' of a type (line 34)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), self_3707, 'warn_msg', msg_call_result_3706)
            # SSA join for if statement (line 31)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @staticmethod
    @norecursion
    def instance(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 37)
        True_3708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 47), 'True')
        defaults = [True_3708]
        # Create a new context for function 'instance'
        module_type_store = module_type_store.open_function_context('instance', 36, 4, False)
        
        # Passed parameters checking function
        TypeWarning.instance.__dict__.__setitem__('stypy_localization', localization)
        TypeWarning.instance.__dict__.__setitem__('stypy_type_of_self', None)
        TypeWarning.instance.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeWarning.instance.__dict__.__setitem__('stypy_function_name', 'instance')
        TypeWarning.instance.__dict__.__setitem__('stypy_param_names_list', ['localization', 'msg', 'prints_msg'])
        TypeWarning.instance.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeWarning.instance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeWarning.instance.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeWarning.instance.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeWarning.instance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeWarning.instance.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, None, module_type_store, 'instance', ['localization', 'msg', 'prints_msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'instance', localization, ['msg', 'prints_msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'instance(...)' code ##################

        str_3709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, (-1)), 'str', '\n        Creates a TypeWarning instance (or a TypeError if strict mode is used).\n        :param localization: Caller information\n        :param msg: Warning message\n        :param prints_msg: Silent warning?\n        :return:\n        ')
        # Getting the type of 'TypeWarning' (line 45)
        TypeWarning_3710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'TypeWarning')
        # Obtaining the member 'warnings_as_errors' of a type (line 45)
        warnings_as_errors_3711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 11), TypeWarning_3710, 'warnings_as_errors')
        # Testing if the type of an if condition is none (line 45)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 45, 8), warnings_as_errors_3711):
            
            # Call to TypeWarning(...): (line 48)
            # Processing the call arguments (line 48)
            # Getting the type of 'localization' (line 48)
            localization_3722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 31), 'localization', False)
            # Getting the type of 'msg' (line 48)
            msg_3723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 45), 'msg', False)
            # Getting the type of 'prints_msg' (line 48)
            prints_msg_3724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 50), 'prints_msg', False)
            # Processing the call keyword arguments (line 48)
            kwargs_3725 = {}
            # Getting the type of 'TypeWarning' (line 48)
            TypeWarning_3721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'TypeWarning', False)
            # Calling TypeWarning(args, kwargs) (line 48)
            TypeWarning_call_result_3726 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), TypeWarning_3721, *[localization_3722, msg_3723, prints_msg_3724], **kwargs_3725)
            
            # Assigning a type to the variable 'stypy_return_type' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'stypy_return_type', TypeWarning_call_result_3726)
        else:
            
            # Testing the type of an if condition (line 45)
            if_condition_3712 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 8), warnings_as_errors_3711)
            # Assigning a type to the variable 'if_condition_3712' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'if_condition_3712', if_condition_3712)
            # SSA begins for if statement (line 45)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 46)
            # Processing the call arguments (line 46)
            # Getting the type of 'localization' (line 46)
            localization_3716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 57), 'localization', False)
            # Getting the type of 'msg' (line 46)
            msg_3717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 71), 'msg', False)
            # Getting the type of 'prints_msg' (line 46)
            prints_msg_3718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 76), 'prints_msg', False)
            # Processing the call keyword arguments (line 46)
            kwargs_3719 = {}
            # Getting the type of 'errors_copy' (line 46)
            errors_copy_3713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'errors_copy', False)
            # Obtaining the member 'type_error_copy' of a type (line 46)
            type_error_copy_3714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 19), errors_copy_3713, 'type_error_copy')
            # Obtaining the member 'TypeError' of a type (line 46)
            TypeError_3715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 19), type_error_copy_3714, 'TypeError')
            # Calling TypeError(args, kwargs) (line 46)
            TypeError_call_result_3720 = invoke(stypy.reporting.localization.Localization(__file__, 46, 19), TypeError_3715, *[localization_3716, msg_3717, prints_msg_3718], **kwargs_3719)
            
            # Assigning a type to the variable 'stypy_return_type' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'stypy_return_type', TypeError_call_result_3720)
            # SSA branch for the else part of an if statement (line 45)
            module_type_store.open_ssa_branch('else')
            
            # Call to TypeWarning(...): (line 48)
            # Processing the call arguments (line 48)
            # Getting the type of 'localization' (line 48)
            localization_3722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 31), 'localization', False)
            # Getting the type of 'msg' (line 48)
            msg_3723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 45), 'msg', False)
            # Getting the type of 'prints_msg' (line 48)
            prints_msg_3724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 50), 'prints_msg', False)
            # Processing the call keyword arguments (line 48)
            kwargs_3725 = {}
            # Getting the type of 'TypeWarning' (line 48)
            TypeWarning_3721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'TypeWarning', False)
            # Calling TypeWarning(args, kwargs) (line 48)
            TypeWarning_call_result_3726 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), TypeWarning_3721, *[localization_3722, msg_3723, prints_msg_3724], **kwargs_3725)
            
            # Assigning a type to the variable 'stypy_return_type' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'stypy_return_type', TypeWarning_call_result_3726)
            # SSA join for if statement (line 45)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'instance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'instance' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_3727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3727)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'instance'
        return stypy_return_type_3727


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeWarning.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        TypeWarning.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeWarning.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeWarning.stypy__str__.__dict__.__setitem__('stypy_function_name', 'TypeWarning.stypy__str__')
        TypeWarning.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        TypeWarning.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeWarning.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeWarning.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeWarning.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeWarning.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeWarning.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeWarning.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        # Getting the type of 'self' (line 51)
        self_3728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'self')
        # Obtaining the member 'warn_msg' of a type (line 51)
        warn_msg_3729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 15), self_3728, 'warn_msg')
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', warn_msg_3729)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_3730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3730)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_3730


    @norecursion
    def __format_file_name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__format_file_name'
        module_type_store = module_type_store.open_function_context('__format_file_name', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeWarning.__format_file_name.__dict__.__setitem__('stypy_localization', localization)
        TypeWarning.__format_file_name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeWarning.__format_file_name.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeWarning.__format_file_name.__dict__.__setitem__('stypy_function_name', 'TypeWarning.__format_file_name')
        TypeWarning.__format_file_name.__dict__.__setitem__('stypy_param_names_list', [])
        TypeWarning.__format_file_name.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeWarning.__format_file_name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeWarning.__format_file_name.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeWarning.__format_file_name.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeWarning.__format_file_name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeWarning.__format_file_name.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeWarning.__format_file_name', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__format_file_name', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__format_file_name(...)' code ##################

        str_3731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, (-1)), 'str', '\n        Pretty-prints file name\n        :return:\n        ')
        
        # Assigning a Subscript to a Name (line 58):
        
        # Obtaining the type of the subscript
        int_3732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 59), 'int')
        
        # Call to split(...): (line 58)
        # Processing the call arguments (line 58)
        str_3737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 54), 'str', '/')
        # Processing the call keyword arguments (line 58)
        kwargs_3738 = {}
        # Getting the type of 'self' (line 58)
        self_3733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'self', False)
        # Obtaining the member 'localization' of a type (line 58)
        localization_3734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 20), self_3733, 'localization')
        # Obtaining the member 'file_name' of a type (line 58)
        file_name_3735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 20), localization_3734, 'file_name')
        # Obtaining the member 'split' of a type (line 58)
        split_3736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 20), file_name_3735, 'split')
        # Calling split(args, kwargs) (line 58)
        split_call_result_3739 = invoke(stypy.reporting.localization.Localization(__file__, 58, 20), split_3736, *[str_3737], **kwargs_3738)
        
        # Obtaining the member '__getitem__' of a type (line 58)
        getitem___3740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 20), split_call_result_3739, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 58)
        subscript_call_result_3741 = invoke(stypy.reporting.localization.Localization(__file__, 58, 20), getitem___3740, int_3732)
        
        # Assigning a type to the variable 'file_name' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'file_name', subscript_call_result_3741)
        
        # Assigning a Subscript to a Name (line 59):
        
        # Obtaining the type of the subscript
        int_3742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 42), 'int')
        
        # Call to split(...): (line 59)
        # Processing the call arguments (line 59)
        str_3745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 36), 'str', '\\')
        # Processing the call keyword arguments (line 59)
        kwargs_3746 = {}
        # Getting the type of 'file_name' (line 59)
        file_name_3743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'file_name', False)
        # Obtaining the member 'split' of a type (line 59)
        split_3744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 20), file_name_3743, 'split')
        # Calling split(args, kwargs) (line 59)
        split_call_result_3747 = invoke(stypy.reporting.localization.Localization(__file__, 59, 20), split_3744, *[str_3745], **kwargs_3746)
        
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___3748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 20), split_call_result_3747, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_3749 = invoke(stypy.reporting.localization.Localization(__file__, 59, 20), getitem___3748, int_3742)
        
        # Assigning a type to the variable 'file_name' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'file_name', subscript_call_result_3749)
        
        # Assigning a Call to a Name (line 60):
        
        # Call to replace(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'stypy_parameters_copy' (line 60)
        stypy_parameters_copy_3752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 38), 'stypy_parameters_copy', False)
        # Obtaining the member 'type_inference_file_postfix' of a type (line 60)
        type_inference_file_postfix_3753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 38), stypy_parameters_copy_3752, 'type_inference_file_postfix')
        str_3754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 89), 'str', '')
        # Processing the call keyword arguments (line 60)
        kwargs_3755 = {}
        # Getting the type of 'file_name' (line 60)
        file_name_3750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'file_name', False)
        # Obtaining the member 'replace' of a type (line 60)
        replace_3751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 20), file_name_3750, 'replace')
        # Calling replace(args, kwargs) (line 60)
        replace_call_result_3756 = invoke(stypy.reporting.localization.Localization(__file__, 60, 20), replace_3751, *[type_inference_file_postfix_3753, str_3754], **kwargs_3755)
        
        # Assigning a type to the variable 'file_name' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'file_name', replace_call_result_3756)
        
        # Assigning a Call to a Name (line 61):
        
        # Call to replace(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'stypy_parameters_copy' (line 61)
        stypy_parameters_copy_3759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 38), 'stypy_parameters_copy', False)
        # Obtaining the member 'type_inference_file_directory_name' of a type (line 61)
        type_inference_file_directory_name_3760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 38), stypy_parameters_copy_3759, 'type_inference_file_directory_name')
        str_3761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 96), 'str', '')
        # Processing the call keyword arguments (line 61)
        kwargs_3762 = {}
        # Getting the type of 'file_name' (line 61)
        file_name_3757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'file_name', False)
        # Obtaining the member 'replace' of a type (line 61)
        replace_3758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 20), file_name_3757, 'replace')
        # Calling replace(args, kwargs) (line 61)
        replace_call_result_3763 = invoke(stypy.reporting.localization.Localization(__file__, 61, 20), replace_3758, *[type_inference_file_directory_name_3760, str_3761], **kwargs_3762)
        
        # Assigning a type to the variable 'file_name' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'file_name', replace_call_result_3763)
        # Getting the type of 'file_name' (line 63)
        file_name_3764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'file_name')
        # Assigning a type to the variable 'stypy_return_type' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'stypy_return_type', file_name_3764)
        
        # ################# End of '__format_file_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__format_file_name' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_3765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3765)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__format_file_name'
        return stypy_return_type_3765


    @norecursion
    def __msg(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__msg'
        module_type_store = module_type_store.open_function_context('__msg', 65, 4, False)
        # Assigning a type to the variable 'self' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeWarning.__msg.__dict__.__setitem__('stypy_localization', localization)
        TypeWarning.__msg.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeWarning.__msg.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeWarning.__msg.__dict__.__setitem__('stypy_function_name', 'TypeWarning.__msg')
        TypeWarning.__msg.__dict__.__setitem__('stypy_param_names_list', [])
        TypeWarning.__msg.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeWarning.__msg.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeWarning.__msg.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeWarning.__msg.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeWarning.__msg.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeWarning.__msg.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeWarning.__msg', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__msg', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__msg(...)' code ##################

        str_3766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, (-1)), 'str', '\n        Composes the full warning message, using the message, the localization, current file name and\n        the stack trace. If available, it also displays the source code line when the warning is produced and a\n        ^ marker indicating the position within the warning line.\n        :return:\n        ')
        
        # Assigning a Call to a Name (line 72):
        
        # Call to __format_file_name(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_3769 = {}
        # Getting the type of 'self' (line 72)
        self_3767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'self', False)
        # Obtaining the member '__format_file_name' of a type (line 72)
        format_file_name_3768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 20), self_3767, '__format_file_name')
        # Calling __format_file_name(args, kwargs) (line 72)
        format_file_name_call_result_3770 = invoke(stypy.reporting.localization.Localization(__file__, 72, 20), format_file_name_3768, *[], **kwargs_3769)
        
        # Assigning a type to the variable 'file_name' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'file_name', format_file_name_call_result_3770)
        
        # Assigning a Call to a Name (line 74):
        
        # Call to get_line_from_module_code(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'self' (line 74)
        self_3773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 68), 'self', False)
        # Obtaining the member 'localization' of a type (line 74)
        localization_3774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 68), self_3773, 'localization')
        # Obtaining the member 'file_name' of a type (line 74)
        file_name_3775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 68), localization_3774, 'file_name')
        # Getting the type of 'self' (line 74)
        self_3776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 97), 'self', False)
        # Obtaining the member 'localization' of a type (line 74)
        localization_3777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 97), self_3776, 'localization')
        # Obtaining the member 'line' of a type (line 74)
        line_3778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 97), localization_3777, 'line')
        # Processing the call keyword arguments (line 74)
        kwargs_3779 = {}
        # Getting the type of 'ModuleLineNumbering' (line 74)
        ModuleLineNumbering_3771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 22), 'ModuleLineNumbering', False)
        # Obtaining the member 'get_line_from_module_code' of a type (line 74)
        get_line_from_module_code_3772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 22), ModuleLineNumbering_3771, 'get_line_from_module_code')
        # Calling get_line_from_module_code(args, kwargs) (line 74)
        get_line_from_module_code_call_result_3780 = invoke(stypy.reporting.localization.Localization(__file__, 74, 22), get_line_from_module_code_3772, *[file_name_3775, line_3778], **kwargs_3779)
        
        # Assigning a type to the variable 'source_code' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'source_code', get_line_from_module_code_call_result_3780)
        
        # Assigning a Call to a Name (line 75):
        
        # Call to get_column_from_module_code(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'self' (line 75)
        self_3783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 69), 'self', False)
        # Obtaining the member 'localization' of a type (line 75)
        localization_3784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 69), self_3783, 'localization')
        # Obtaining the member 'file_name' of a type (line 75)
        file_name_3785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 69), localization_3784, 'file_name')
        # Getting the type of 'self' (line 76)
        self_3786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 69), 'self', False)
        # Obtaining the member 'localization' of a type (line 76)
        localization_3787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 69), self_3786, 'localization')
        # Obtaining the member 'line' of a type (line 76)
        line_3788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 69), localization_3787, 'line')
        # Getting the type of 'self' (line 76)
        self_3789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 93), 'self', False)
        # Obtaining the member 'localization' of a type (line 76)
        localization_3790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 93), self_3789, 'localization')
        # Obtaining the member 'column' of a type (line 76)
        column_3791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 93), localization_3790, 'column')
        # Processing the call keyword arguments (line 75)
        kwargs_3792 = {}
        # Getting the type of 'ModuleLineNumbering' (line 75)
        ModuleLineNumbering_3781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 21), 'ModuleLineNumbering', False)
        # Obtaining the member 'get_column_from_module_code' of a type (line 75)
        get_column_from_module_code_3782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 21), ModuleLineNumbering_3781, 'get_column_from_module_code')
        # Calling get_column_from_module_code(args, kwargs) (line 75)
        get_column_from_module_code_call_result_3793 = invoke(stypy.reporting.localization.Localization(__file__, 75, 21), get_column_from_module_code_3782, *[file_name_3785, line_3788, column_3791], **kwargs_3792)
        
        # Assigning a type to the variable 'col_offset' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'col_offset', get_column_from_module_code_call_result_3793)
        
        # Type idiom detected: calculating its left and rigth part (line 77)
        # Getting the type of 'source_code' (line 77)
        source_code_3794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'source_code')
        # Getting the type of 'None' (line 77)
        None_3795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 30), 'None')
        
        (may_be_3796, more_types_in_union_3797) = may_not_be_none(source_code_3794, None_3795)

        if may_be_3796:

            if more_types_in_union_3797:
                # Runtime conditional SSA (line 77)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            str_3798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 19), 'str', "Warning in file '%s' (line %s, column %s):\n%s\n%s\n\t%s.\n\n%s")
            
            # Obtaining an instance of the builtin type 'tuple' (line 79)
            tuple_3799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 20), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 79)
            # Adding element type (line 79)
            # Getting the type of 'file_name' (line 79)
            file_name_3800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'file_name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 20), tuple_3799, file_name_3800)
            # Adding element type (line 79)
            # Getting the type of 'self' (line 79)
            self_3801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 31), 'self')
            # Obtaining the member 'localization' of a type (line 79)
            localization_3802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 31), self_3801, 'localization')
            # Obtaining the member 'line' of a type (line 79)
            line_3803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 31), localization_3802, 'line')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 20), tuple_3799, line_3803)
            # Adding element type (line 79)
            # Getting the type of 'self' (line 79)
            self_3804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 55), 'self')
            # Obtaining the member 'localization' of a type (line 79)
            localization_3805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 55), self_3804, 'localization')
            # Obtaining the member 'column' of a type (line 79)
            column_3806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 55), localization_3805, 'column')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 20), tuple_3799, column_3806)
            # Adding element type (line 79)
            # Getting the type of 'source_code' (line 80)
            source_code_3807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'source_code')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 20), tuple_3799, source_code_3807)
            # Adding element type (line 79)
            str_3808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 33), 'str', '')
            # Getting the type of 'col_offset' (line 80)
            col_offset_3809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 38), 'col_offset')
            # Applying the binary operator '+' (line 80)
            result_add_3810 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 33), '+', str_3808, col_offset_3809)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 20), tuple_3799, result_add_3810)
            # Adding element type (line 79)
            
            # Call to strip(...): (line 81)
            # Processing the call keyword arguments (line 81)
            kwargs_3814 = {}
            # Getting the type of 'self' (line 81)
            self_3811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'self', False)
            # Obtaining the member 'msg' of a type (line 81)
            msg_3812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 20), self_3811, 'msg')
            # Obtaining the member 'strip' of a type (line 81)
            strip_3813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 20), msg_3812, 'strip')
            # Calling strip(args, kwargs) (line 81)
            strip_call_result_3815 = invoke(stypy.reporting.localization.Localization(__file__, 81, 20), strip_3813, *[], **kwargs_3814)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 20), tuple_3799, strip_call_result_3815)
            # Adding element type (line 79)
            # Getting the type of 'self' (line 81)
            self_3816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 38), 'self')
            # Obtaining the member 'localization' of a type (line 81)
            localization_3817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 38), self_3816, 'localization')
            # Obtaining the member 'stack_trace' of a type (line 81)
            stack_trace_3818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 38), localization_3817, 'stack_trace')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 20), tuple_3799, stack_trace_3818)
            
            # Applying the binary operator '%' (line 78)
            result_mod_3819 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 19), '%', str_3798, tuple_3799)
            
            # Assigning a type to the variable 'stypy_return_type' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'stypy_return_type', result_mod_3819)

            if more_types_in_union_3797:
                # SSA join for if statement (line 77)
                module_type_store = module_type_store.join_ssa_context()


        
        str_3820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 15), 'str', "Warning in file '%s' (line %s, column %s):\n%s.\n\n%s")
        
        # Obtaining an instance of the builtin type 'tuple' (line 84)
        tuple_3821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 84)
        # Adding element type (line 84)
        # Getting the type of 'file_name' (line 84)
        file_name_3822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'file_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 16), tuple_3821, file_name_3822)
        # Adding element type (line 84)
        # Getting the type of 'self' (line 84)
        self_3823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'self')
        # Obtaining the member 'localization' of a type (line 84)
        localization_3824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 27), self_3823, 'localization')
        # Obtaining the member 'line' of a type (line 84)
        line_3825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 27), localization_3824, 'line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 16), tuple_3821, line_3825)
        # Adding element type (line 84)
        # Getting the type of 'self' (line 84)
        self_3826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 51), 'self')
        # Obtaining the member 'localization' of a type (line 84)
        localization_3827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 51), self_3826, 'localization')
        # Obtaining the member 'column' of a type (line 84)
        column_3828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 51), localization_3827, 'column')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 16), tuple_3821, column_3828)
        # Adding element type (line 84)
        # Getting the type of 'self' (line 85)
        self_3829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'self')
        # Obtaining the member 'msg' of a type (line 85)
        msg_3830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 16), self_3829, 'msg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 16), tuple_3821, msg_3830)
        # Adding element type (line 84)
        # Getting the type of 'self' (line 85)
        self_3831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 26), 'self')
        # Obtaining the member 'localization' of a type (line 85)
        localization_3832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 26), self_3831, 'localization')
        # Obtaining the member 'stack_trace' of a type (line 85)
        stack_trace_3833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 26), localization_3832, 'stack_trace')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 16), tuple_3821, stack_trace_3833)
        
        # Applying the binary operator '%' (line 83)
        result_mod_3834 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 15), '%', str_3820, tuple_3821)
        
        # Assigning a type to the variable 'stypy_return_type' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'stypy_return_type', result_mod_3834)
        
        # ################# End of '__msg(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__msg' in the type store
        # Getting the type of 'stypy_return_type' (line 65)
        stypy_return_type_3835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3835)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__msg'
        return stypy_return_type_3835


    @staticmethod
    @norecursion
    def print_warning_msgs(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_warning_msgs'
        module_type_store = module_type_store.open_function_context('print_warning_msgs', 87, 4, False)
        
        # Passed parameters checking function
        TypeWarning.print_warning_msgs.__dict__.__setitem__('stypy_localization', localization)
        TypeWarning.print_warning_msgs.__dict__.__setitem__('stypy_type_of_self', None)
        TypeWarning.print_warning_msgs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeWarning.print_warning_msgs.__dict__.__setitem__('stypy_function_name', 'print_warning_msgs')
        TypeWarning.print_warning_msgs.__dict__.__setitem__('stypy_param_names_list', [])
        TypeWarning.print_warning_msgs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeWarning.print_warning_msgs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeWarning.print_warning_msgs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeWarning.print_warning_msgs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeWarning.print_warning_msgs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeWarning.print_warning_msgs.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'print_warning_msgs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_warning_msgs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_warning_msgs(...)' code ##################

        str_3836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, (-1)), 'str', '\n        Prints all the warning messages that were produced during a program analysis. Just for debugging\n        :return:\n        ')
        
        # Getting the type of 'TypeWarning' (line 93)
        TypeWarning_3837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'TypeWarning')
        # Obtaining the member 'warnings' of a type (line 93)
        warnings_3838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 19), TypeWarning_3837, 'warnings')
        # Assigning a type to the variable 'warnings_3838' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'warnings_3838', warnings_3838)
        # Testing if the for loop is going to be iterated (line 93)
        # Testing the type of a for loop iterable (line 93)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 93, 8), warnings_3838)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 93, 8), warnings_3838):
            # Getting the type of the for loop variable (line 93)
            for_loop_var_3839 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 93, 8), warnings_3838)
            # Assigning a type to the variable 'err' (line 93)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'err', for_loop_var_3839)
            # SSA begins for a for statement (line 93)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'err' (line 94)
            err_3840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 18), 'err')
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'print_warning_msgs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_warning_msgs' in the type store
        # Getting the type of 'stypy_return_type' (line 87)
        stypy_return_type_3841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3841)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_warning_msgs'
        return stypy_return_type_3841


    @staticmethod
    @norecursion
    def reset_warning_msgs(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'reset_warning_msgs'
        module_type_store = module_type_store.open_function_context('reset_warning_msgs', 96, 4, False)
        
        # Passed parameters checking function
        TypeWarning.reset_warning_msgs.__dict__.__setitem__('stypy_localization', localization)
        TypeWarning.reset_warning_msgs.__dict__.__setitem__('stypy_type_of_self', None)
        TypeWarning.reset_warning_msgs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeWarning.reset_warning_msgs.__dict__.__setitem__('stypy_function_name', 'reset_warning_msgs')
        TypeWarning.reset_warning_msgs.__dict__.__setitem__('stypy_param_names_list', [])
        TypeWarning.reset_warning_msgs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeWarning.reset_warning_msgs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeWarning.reset_warning_msgs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeWarning.reset_warning_msgs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeWarning.reset_warning_msgs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeWarning.reset_warning_msgs.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'reset_warning_msgs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reset_warning_msgs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reset_warning_msgs(...)' code ##################

        
        # Assigning a List to a Attribute (line 98):
        
        # Obtaining an instance of the builtin type 'list' (line 98)
        list_3842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 98)
        
        # Getting the type of 'TypeWarning' (line 98)
        TypeWarning_3843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'TypeWarning')
        # Setting the type of the member 'warnings' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), TypeWarning_3843, 'warnings', list_3842)
        
        # ################# End of 'reset_warning_msgs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reset_warning_msgs' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_3844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3844)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reset_warning_msgs'
        return stypy_return_type_3844


    @staticmethod
    @norecursion
    def get_warning_msgs(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_warning_msgs'
        module_type_store = module_type_store.open_function_context('get_warning_msgs', 100, 4, False)
        
        # Passed parameters checking function
        TypeWarning.get_warning_msgs.__dict__.__setitem__('stypy_localization', localization)
        TypeWarning.get_warning_msgs.__dict__.__setitem__('stypy_type_of_self', None)
        TypeWarning.get_warning_msgs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeWarning.get_warning_msgs.__dict__.__setitem__('stypy_function_name', 'get_warning_msgs')
        TypeWarning.get_warning_msgs.__dict__.__setitem__('stypy_param_names_list', [])
        TypeWarning.get_warning_msgs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeWarning.get_warning_msgs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeWarning.get_warning_msgs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeWarning.get_warning_msgs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeWarning.get_warning_msgs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeWarning.get_warning_msgs.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'get_warning_msgs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_warning_msgs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_warning_msgs(...)' code ##################

        str_3845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, (-1)), 'str', '\n        Gets all the warning messages that were produced during a program analysis.\n        :return: All the errors, sorted by line number\n        ')
        
        # Call to sorted(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'TypeWarning' (line 106)
        TypeWarning_3847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 22), 'TypeWarning', False)
        # Obtaining the member 'warnings' of a type (line 106)
        warnings_3848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 22), TypeWarning_3847, 'warnings')
        # Processing the call keyword arguments (line 106)

        @norecursion
        def _stypy_temp_lambda_4(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_4'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_4', 106, 48, True)
            # Passed parameters checking function
            _stypy_temp_lambda_4.stypy_localization = localization
            _stypy_temp_lambda_4.stypy_type_of_self = None
            _stypy_temp_lambda_4.stypy_type_store = module_type_store
            _stypy_temp_lambda_4.stypy_function_name = '_stypy_temp_lambda_4'
            _stypy_temp_lambda_4.stypy_param_names_list = ['warning']
            _stypy_temp_lambda_4.stypy_varargs_param_name = None
            _stypy_temp_lambda_4.stypy_kwargs_param_name = None
            _stypy_temp_lambda_4.stypy_call_defaults = defaults
            _stypy_temp_lambda_4.stypy_call_varargs = varargs
            _stypy_temp_lambda_4.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_4', ['warning'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_4', ['warning'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'warning' (line 106)
            warning_3849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 64), 'warning', False)
            # Obtaining the member 'localization' of a type (line 106)
            localization_3850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 64), warning_3849, 'localization')
            # Obtaining the member 'line' of a type (line 106)
            line_3851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 64), localization_3850, 'line')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 48), 'stypy_return_type', line_3851)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_4' in the type store
            # Getting the type of 'stypy_return_type' (line 106)
            stypy_return_type_3852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 48), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3852)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_4'
            return stypy_return_type_3852

        # Assigning a type to the variable '_stypy_temp_lambda_4' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 48), '_stypy_temp_lambda_4', _stypy_temp_lambda_4)
        # Getting the type of '_stypy_temp_lambda_4' (line 106)
        _stypy_temp_lambda_4_3853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 48), '_stypy_temp_lambda_4')
        keyword_3854 = _stypy_temp_lambda_4_3853
        kwargs_3855 = {'key': keyword_3854}
        # Getting the type of 'sorted' (line 106)
        sorted_3846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'sorted', False)
        # Calling sorted(args, kwargs) (line 106)
        sorted_call_result_3856 = invoke(stypy.reporting.localization.Localization(__file__, 106, 15), sorted_3846, *[warnings_3848], **kwargs_3855)
        
        # Assigning a type to the variable 'stypy_return_type' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'stypy_return_type', sorted_call_result_3856)
        
        # ################# End of 'get_warning_msgs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_warning_msgs' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_3857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3857)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_warning_msgs'
        return stypy_return_type_3857


# Assigning a type to the variable 'TypeWarning' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'TypeWarning', TypeWarning)

# Assigning a Name to a Name (line 15):
# Getting the type of 'False' (line 15)
False_3858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 25), 'False')
# Getting the type of 'TypeWarning'
TypeWarning_3859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeWarning')
# Setting the type of the member 'warnings_as_errors' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeWarning_3859, 'warnings_as_errors', False_3858)

# Assigning a List to a Name (line 18):

# Obtaining an instance of the builtin type 'list' (line 18)
list_3860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)

# Getting the type of 'TypeWarning'
TypeWarning_3861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeWarning')
# Setting the type of the member 'warnings' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeWarning_3861, 'warnings', list_3860)
# Declaration of the 'UnreferencedLocalVariableTypeWarning' class
# Getting the type of 'TypeWarning' (line 132)
TypeWarning_3862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 43), 'TypeWarning')

class UnreferencedLocalVariableTypeWarning(TypeWarning_3862, ):
    pass
    str_3863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, (-1)), 'str', '\n    This special type of warning is only used if coding advices are activated. It models those cases when a global\n    variable is read and later on is written to without using the global keyword. Python decides to report an error\n    in this case, but in the source line that reads the value instead of the source line that write a value to the\n    variable. A coding advice is generated if this kind of programming pattern is detected within the program.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnreferencedLocalVariableTypeWarning.__init__', ['localization', 'name', 'context'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['localization', 'name', 'context'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'self' (line 141)
        self_3866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 29), 'self', False)
        # Getting the type of 'localization' (line 141)
        localization_3867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 35), 'localization', False)
        
        # Call to format(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'name' (line 144)
        name_3870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 103), 'name', False)
        # Processing the call keyword arguments (line 142)
        kwargs_3871 = {}
        str_3868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 29), 'str', "Read access detected over a global name '{0}'. Any attempt of writing to this name without using the 'global' keyword first will result into an 'UnboundLocalError: local variable '{0}' referenced before assignment' runtime error")
        # Obtaining the member 'format' of a type (line 142)
        format_3869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 29), str_3868, 'format')
        # Calling format(args, kwargs) (line 142)
        format_call_result_3872 = invoke(stypy.reporting.localization.Localization(__file__, 142, 29), format_3869, *[name_3870], **kwargs_3871)
        
        # Processing the call keyword arguments (line 141)
        kwargs_3873 = {}
        # Getting the type of 'TypeWarning' (line 141)
        TypeWarning_3864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'TypeWarning', False)
        # Obtaining the member '__init__' of a type (line 141)
        init___3865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), TypeWarning_3864, '__init__')
        # Calling __init__(args, kwargs) (line 141)
        init___call_result_3874 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), init___3865, *[self_3866, localization_3867, format_call_result_3872], **kwargs_3873)
        
        
        # Assigning a Name to a Attribute (line 145):
        # Getting the type of 'name' (line 145)
        name_3875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 20), 'name')
        # Getting the type of 'self' (line 145)
        self_3876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'self')
        # Setting the type of the member 'name' of a type (line 145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), self_3876, 'name', name_3875)
        
        # Assigning a Name to a Attribute (line 146):
        # Getting the type of 'context' (line 146)
        context_3877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 23), 'context')
        # Getting the type of 'self' (line 146)
        self_3878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self')
        # Setting the type of the member 'context' of a type (line 146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_3878, 'context', context_3877)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'UnreferencedLocalVariableTypeWarning' (line 132)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 0), 'UnreferencedLocalVariableTypeWarning', UnreferencedLocalVariableTypeWarning)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
