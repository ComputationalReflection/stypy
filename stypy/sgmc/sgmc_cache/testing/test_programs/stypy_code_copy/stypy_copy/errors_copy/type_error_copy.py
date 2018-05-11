
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from ...stypy_copy.python_lib_copy.python_types_copy.type_copy import Type
2: import type_warning_copy
3: from ...stypy_copy.reporting_copy.module_line_numbering_copy import ModuleLineNumbering
4: from ..python_lib_copy.python_types_copy.type_inference_copy.localization_copy import Localization
5: from ...stypy_copy import stypy_parameters_copy
6: 
7: 
8: class TypeError(Type):
9:     '''
10:     A TypeError represent some kind of error found when handling types in a type inference program. It can be whatever
11:     type error we found: misuse of type members, incorrect operator application, wrong types for a certain operation...
12:      This class is very important in stypy because it models all the program type errors that later on will be reported
13:       to the user.
14:     '''
15: 
16:     # The class stores a list with all the produced type errors so far
17:     errors = []
18:     # This flag is used to indicate that a certain sentence of the program has used an unsupported feature. Therefore
19:     # types cannot be accurately determined on the subsequent execution of the type inference program, and further
20:     # TypeErrors will only report this fact to avoid reporting false errors.
21:     usage_of_unsupported_feature = False
22: 
23:     def __init__(self, localization=None, msg="", prints_msg=True):
24:         '''
25:         Creates a particular instance of a type error amd adds it to the error list
26:         :param localization: Caller information
27:         :param msg: Error to report to the user
28:         :param prints_msg: Determines if this error is silent (report its message) or not. Some error are silent
29:         because they are generated to generate a more accurate TypeError later once the program determines that
30:         TypeErrors exist on certain places of the analyzed program. This is used in certain situations to avoid
31:         reporting the same error multiple times
32:         :return:
33:         '''
34:         if TypeError.usage_of_unsupported_feature:
35:             self.msg = "The type of this member could not be obtained due to the previous usage of an unsupported " \
36:                        "stypy feature"
37:         else:
38:             self.msg = msg
39: 
40:         if localization is None:
41:             localization = Localization(__file__, 1, 0)
42: 
43:         self.localization = localization
44: 
45:         if prints_msg and not TypeError.usage_of_unsupported_feature:
46:             TypeError.errors.append(self)
47: 
48:         # The error_msg is the full error to report to the user, composed by the passed msg and the stack trace.
49:         # We calculate it here to "capture" the precise execution point when the error is produced as stack trace is
50:         # dynamic and changes during the execution
51:         self.error_msg = self.__msg()
52: 
53:     def turn_to_warning(self):
54:         '''
55:         Sometimes type errors have to be converted to warnings as some correct paths in the code exist although errors
56:         are detected. This is used, for example, when performing calls with union types. If some combinations are
57:         erroneus but at least one is possible, the errors for the wrong parameter type combinations are turned to
58:         warnings to report them precisely.
59:         :return:
60:         '''
61:         type_warning_copy.TypeWarning.instance(self.localization, self.msg)
62:         TypeError.remove_error_msg(self)
63: 
64:     def __str__(self):
65:         '''
66:         Visual representation of the error (full message: error + stack trace)
67:         :return:
68:         '''
69:         return self.error_msg
70: 
71:     def __format_file_name(self):
72:         '''
73:         Pretty-prints file name
74:         :return:
75:         '''
76:         file_name = self.localization.file_name.split('/')[-1]
77:         file_name = file_name.split('\\')[-1]
78:         file_name = file_name.replace(stypy_parameters_copy.type_inference_file_postfix, '')
79:         file_name = file_name.replace(stypy_parameters_copy.type_inference_file_directory_name, '')
80: 
81:         return file_name
82: 
83:     def __msg(self):
84:         '''
85:         Composes the full error message, using the error message, the error localization, current file name and
86:         the stack trace. If available, it also displays the source code line when the error is produced and a
87:         ^ marker indicating the position within the error line.
88:         :return:
89:         '''
90:         file_name = self.__format_file_name()
91: 
92:         source_code = ModuleLineNumbering.get_line_from_module_code(self.localization.file_name, self.localization.line)
93:         col_offset = ModuleLineNumbering.get_column_from_module_code(self.localization.file_name,
94:                                                                      self.localization.line, self.localization.column)
95:         if source_code is not None:
96:             return "Compiler error in file '%s' (line %s, column %s):\n%s\n%s\n\t%s.\n\n%s" % \
97:                    (file_name, self.localization.line, self.localization.column,
98:                     source_code, "" + col_offset,
99:                     self.msg.strip(), self.localization.stack_trace)
100: 
101:         return "Compiler error in file '%s' (line %s, column %s):\n%s.\n\n%s" % \
102:                (file_name, self.localization.line, self.localization.column,
103:                 self.msg, self.localization.stack_trace)
104: 
105:     @staticmethod
106:     def print_error_msgs():
107:         '''
108:         Prints all the error messages that were produced during a program analysis. Just for debugging
109:         :return:
110:         '''
111:         for err in TypeError.errors:
112:             print err
113: 
114:     @staticmethod
115:     def get_error_msgs():
116:         '''
117:         Gets all the error messages that were produced during a program analysis.
118:         :return: All the errors, sorted by line number
119:         '''
120:         return sorted(TypeError.errors, key=lambda error: error.localization.line)
121: 
122:     @staticmethod
123:     def remove_error_msg(error_obj):
124:         '''
125:         Deletes an error message from the global error list. As we said, error messages might be turn to warnings, so
126:         we must delete them afterwards
127:         :param error_obj:
128:         :return:
129:         '''
130:         if isinstance(error_obj, list):
131:             for error in error_obj:
132:                 TypeError.errors.remove(error)
133:         else:
134:             try:
135:                 TypeError.errors.remove(error_obj)
136:             except:
137:                 pass
138: 
139:     @staticmethod
140:     def reset_error_msgs():
141:         '''
142:         Clears the global error message list
143:         :return:
144:         '''
145:         TypeError.errors = []
146: 
147:     # ############################## OTHER TYPE METHODS ###############################
148:     '''
149:     As errors are also stypy Type objects, they must provide the rest of its interface methods in order to allow
150:     the analysis of the program in an orthogonal fashion. These method do nothing, as they don't make sense within
151:     a TypeError. If methods of this object report errors upon called, the error reporting will display repeated
152:     errors at the end.
153:     '''
154:     def get_python_entity(self):
155:         return self
156: 
157:     def get_python_type(self):
158:         return self
159: 
160:     def get_instance(self):
161:         return None
162: 
163:     # ############################## STORED TYPE METHODS ###############################
164: 
165:     def can_store_elements(self):
166:         return False
167: 
168:     def can_store_keypairs(self):
169:         return False
170: 
171:     def get_elements_type(self):
172:         return self
173: 
174:     def is_empty(self):
175:         return self
176: 
177:     def set_elements_type(self, localization, elements_type, record_annotation=True):
178:         return self
179: 
180:     def add_type(self, localization, type_, record_annotation=True):
181:         return self
182: 
183:     def add_types_from_list(self, localization, type_list, record_annotation=True):
184:         return self
185: 
186:     def add_key_and_value_type(self, localization, type_tuple, record_annotation=True):
187:         return self
188: 
189:     # ############################## MEMBER TYPE GET / SET ###############################
190: 
191:     def get_type_of_member(self, localization, member_name):
192:         return self
193: 
194:     def set_type_of_member(self, localization, member_name, member_value):
195:         return self
196: 
197:     # ############################## MEMBER INVOKATION ###############################
198: 
199:     def invoke(self, localization, *args, **kwargs):
200:         return self
201: 
202:     # ############################## STRUCTURAL REFLECTION ###############################
203: 
204:     def delete_member(self, localization, member):
205:         return self
206: 
207:     def supports_structural_reflection(self):
208:         return False
209: 
210:     def change_type(self, localization, new_type):
211:         return self
212: 
213:     def change_base_types(self, localization, new_types):
214:         return self
215: 
216:     def add_base_types(self, localization, new_types):
217:         return self
218: 
219:     # ############################## TYPE CLONING ###############################
220: 
221:     def clone(self):
222:         return self
223: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy import Type' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')
import_3699 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy')

if (type(import_3699) is not StypyTypeError):

    if (import_3699 != 'pyd_module'):
        __import__(import_3699)
        sys_modules_3700 = sys.modules[import_3699]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', sys_modules_3700.module_type_store, module_type_store, ['Type'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_3700, sys_modules_3700.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy import Type

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', None, module_type_store, ['Type'], [Type])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', import_3699)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import type_warning_copy' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')
import_3701 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'type_warning_copy')

if (type(import_3701) is not StypyTypeError):

    if (import_3701 != 'pyd_module'):
        __import__(import_3701)
        sys_modules_3702 = sys.modules[import_3701]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'type_warning_copy', sys_modules_3702.module_type_store, module_type_store)
    else:
        import type_warning_copy

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'type_warning_copy', type_warning_copy, module_type_store)

else:
    # Assigning a type to the variable 'type_warning_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'type_warning_copy', import_3701)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.module_line_numbering_copy import ModuleLineNumbering' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')
import_3703 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.module_line_numbering_copy')

if (type(import_3703) is not StypyTypeError):

    if (import_3703 != 'pyd_module'):
        __import__(import_3703)
        sys_modules_3704 = sys.modules[import_3703]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.module_line_numbering_copy', sys_modules_3704.module_type_store, module_type_store, ['ModuleLineNumbering'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_3704, sys_modules_3704.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.module_line_numbering_copy import ModuleLineNumbering

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.module_line_numbering_copy', None, module_type_store, ['ModuleLineNumbering'], [ModuleLineNumbering])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.module_line_numbering_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.module_line_numbering_copy', import_3703)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy import Localization' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')
import_3705 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy')

if (type(import_3705) is not StypyTypeError):

    if (import_3705 != 'pyd_module'):
        __import__(import_3705)
        sys_modules_3706 = sys.modules[import_3705]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy', sys_modules_3706.module_type_store, module_type_store, ['Localization'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_3706, sys_modules_3706.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy import Localization

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy', None, module_type_store, ['Localization'], [Localization])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.localization_copy', import_3705)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')
import_3707 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy')

if (type(import_3707) is not StypyTypeError):

    if (import_3707 != 'pyd_module'):
        __import__(import_3707)
        sys_modules_3708 = sys.modules[import_3707]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', sys_modules_3708.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_3708, sys_modules_3708.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', import_3707)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')

# Declaration of the 'TypeError' class
# Getting the type of 'Type' (line 8)
Type_3709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 16), 'Type')

class TypeError(Type_3709, ):
    str_3710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, (-1)), 'str', '\n    A TypeError represent some kind of error found when handling types in a type inference program. It can be whatever\n    type error we found: misuse of type members, incorrect operator application, wrong types for a certain operation...\n     This class is very important in stypy because it models all the program type errors that later on will be reported\n      to the user.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 23)
        None_3711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 36), 'None')
        str_3712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 46), 'str', '')
        # Getting the type of 'True' (line 23)
        True_3713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 61), 'True')
        defaults = [None_3711, str_3712, True_3713]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.__init__', ['localization', 'msg', 'prints_msg'], None, None, defaults, varargs, kwargs)

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

        str_3714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, (-1)), 'str', '\n        Creates a particular instance of a type error amd adds it to the error list\n        :param localization: Caller information\n        :param msg: Error to report to the user\n        :param prints_msg: Determines if this error is silent (report its message) or not. Some error are silent\n        because they are generated to generate a more accurate TypeError later once the program determines that\n        TypeErrors exist on certain places of the analyzed program. This is used in certain situations to avoid\n        reporting the same error multiple times\n        :return:\n        ')
        # Getting the type of 'TypeError' (line 34)
        TypeError_3715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'TypeError')
        # Obtaining the member 'usage_of_unsupported_feature' of a type (line 34)
        usage_of_unsupported_feature_3716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 11), TypeError_3715, 'usage_of_unsupported_feature')
        # Testing if the type of an if condition is none (line 34)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 34, 8), usage_of_unsupported_feature_3716):
            
            # Assigning a Name to a Attribute (line 38):
            # Getting the type of 'msg' (line 38)
            msg_3720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'msg')
            # Getting the type of 'self' (line 38)
            self_3721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'self')
            # Setting the type of the member 'msg' of a type (line 38)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), self_3721, 'msg', msg_3720)
        else:
            
            # Testing the type of an if condition (line 34)
            if_condition_3717 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 8), usage_of_unsupported_feature_3716)
            # Assigning a type to the variable 'if_condition_3717' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'if_condition_3717', if_condition_3717)
            # SSA begins for if statement (line 34)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Attribute (line 35):
            str_3718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'str', 'The type of this member could not be obtained due to the previous usage of an unsupported stypy feature')
            # Getting the type of 'self' (line 35)
            self_3719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'self')
            # Setting the type of the member 'msg' of a type (line 35)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), self_3719, 'msg', str_3718)
            # SSA branch for the else part of an if statement (line 34)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Attribute (line 38):
            # Getting the type of 'msg' (line 38)
            msg_3720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'msg')
            # Getting the type of 'self' (line 38)
            self_3721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'self')
            # Setting the type of the member 'msg' of a type (line 38)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), self_3721, 'msg', msg_3720)
            # SSA join for if statement (line 34)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Type idiom detected: calculating its left and rigth part (line 40)
        # Getting the type of 'localization' (line 40)
        localization_3722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'localization')
        # Getting the type of 'None' (line 40)
        None_3723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 27), 'None')
        
        (may_be_3724, more_types_in_union_3725) = may_be_none(localization_3722, None_3723)

        if may_be_3724:

            if more_types_in_union_3725:
                # Runtime conditional SSA (line 40)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 41):
            
            # Call to Localization(...): (line 41)
            # Processing the call arguments (line 41)
            # Getting the type of '__file__' (line 41)
            file___3727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 40), '__file__', False)
            int_3728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 50), 'int')
            int_3729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 53), 'int')
            # Processing the call keyword arguments (line 41)
            kwargs_3730 = {}
            # Getting the type of 'Localization' (line 41)
            Localization_3726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 27), 'Localization', False)
            # Calling Localization(args, kwargs) (line 41)
            Localization_call_result_3731 = invoke(stypy.reporting.localization.Localization(__file__, 41, 27), Localization_3726, *[file___3727, int_3728, int_3729], **kwargs_3730)
            
            # Assigning a type to the variable 'localization' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'localization', Localization_call_result_3731)

            if more_types_in_union_3725:
                # SSA join for if statement (line 40)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 43):
        # Getting the type of 'localization' (line 43)
        localization_3732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 28), 'localization')
        # Getting the type of 'self' (line 43)
        self_3733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member 'localization' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_3733, 'localization', localization_3732)
        
        # Evaluating a boolean operation
        # Getting the type of 'prints_msg' (line 45)
        prints_msg_3734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'prints_msg')
        
        # Getting the type of 'TypeError' (line 45)
        TypeError_3735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 30), 'TypeError')
        # Obtaining the member 'usage_of_unsupported_feature' of a type (line 45)
        usage_of_unsupported_feature_3736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 30), TypeError_3735, 'usage_of_unsupported_feature')
        # Applying the 'not' unary operator (line 45)
        result_not__3737 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 26), 'not', usage_of_unsupported_feature_3736)
        
        # Applying the binary operator 'and' (line 45)
        result_and_keyword_3738 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 11), 'and', prints_msg_3734, result_not__3737)
        
        # Testing if the type of an if condition is none (line 45)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 45, 8), result_and_keyword_3738):
            pass
        else:
            
            # Testing the type of an if condition (line 45)
            if_condition_3739 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 8), result_and_keyword_3738)
            # Assigning a type to the variable 'if_condition_3739' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'if_condition_3739', if_condition_3739)
            # SSA begins for if statement (line 45)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 46)
            # Processing the call arguments (line 46)
            # Getting the type of 'self' (line 46)
            self_3743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 36), 'self', False)
            # Processing the call keyword arguments (line 46)
            kwargs_3744 = {}
            # Getting the type of 'TypeError' (line 46)
            TypeError_3740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'TypeError', False)
            # Obtaining the member 'errors' of a type (line 46)
            errors_3741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), TypeError_3740, 'errors')
            # Obtaining the member 'append' of a type (line 46)
            append_3742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), errors_3741, 'append')
            # Calling append(args, kwargs) (line 46)
            append_call_result_3745 = invoke(stypy.reporting.localization.Localization(__file__, 46, 12), append_3742, *[self_3743], **kwargs_3744)
            
            # SSA join for if statement (line 45)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Attribute (line 51):
        
        # Call to __msg(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_3748 = {}
        # Getting the type of 'self' (line 51)
        self_3746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'self', False)
        # Obtaining the member '__msg' of a type (line 51)
        msg_3747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 25), self_3746, '__msg')
        # Calling __msg(args, kwargs) (line 51)
        msg_call_result_3749 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), msg_3747, *[], **kwargs_3748)
        
        # Getting the type of 'self' (line 51)
        self_3750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self')
        # Setting the type of the member 'error_msg' of a type (line 51)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_3750, 'error_msg', msg_call_result_3749)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def turn_to_warning(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'turn_to_warning'
        module_type_store = module_type_store.open_function_context('turn_to_warning', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.turn_to_warning.__dict__.__setitem__('stypy_localization', localization)
        TypeError.turn_to_warning.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.turn_to_warning.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.turn_to_warning.__dict__.__setitem__('stypy_function_name', 'TypeError.turn_to_warning')
        TypeError.turn_to_warning.__dict__.__setitem__('stypy_param_names_list', [])
        TypeError.turn_to_warning.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.turn_to_warning.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.turn_to_warning.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.turn_to_warning.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.turn_to_warning.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.turn_to_warning.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.turn_to_warning', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'turn_to_warning', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'turn_to_warning(...)' code ##################

        str_3751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, (-1)), 'str', '\n        Sometimes type errors have to be converted to warnings as some correct paths in the code exist although errors\n        are detected. This is used, for example, when performing calls with union types. If some combinations are\n        erroneus but at least one is possible, the errors for the wrong parameter type combinations are turned to\n        warnings to report them precisely.\n        :return:\n        ')
        
        # Call to instance(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'self' (line 61)
        self_3755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 47), 'self', False)
        # Obtaining the member 'localization' of a type (line 61)
        localization_3756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 47), self_3755, 'localization')
        # Getting the type of 'self' (line 61)
        self_3757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 66), 'self', False)
        # Obtaining the member 'msg' of a type (line 61)
        msg_3758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 66), self_3757, 'msg')
        # Processing the call keyword arguments (line 61)
        kwargs_3759 = {}
        # Getting the type of 'type_warning_copy' (line 61)
        type_warning_copy_3752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'type_warning_copy', False)
        # Obtaining the member 'TypeWarning' of a type (line 61)
        TypeWarning_3753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), type_warning_copy_3752, 'TypeWarning')
        # Obtaining the member 'instance' of a type (line 61)
        instance_3754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), TypeWarning_3753, 'instance')
        # Calling instance(args, kwargs) (line 61)
        instance_call_result_3760 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), instance_3754, *[localization_3756, msg_3758], **kwargs_3759)
        
        
        # Call to remove_error_msg(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_3763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 35), 'self', False)
        # Processing the call keyword arguments (line 62)
        kwargs_3764 = {}
        # Getting the type of 'TypeError' (line 62)
        TypeError_3761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'TypeError', False)
        # Obtaining the member 'remove_error_msg' of a type (line 62)
        remove_error_msg_3762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), TypeError_3761, 'remove_error_msg')
        # Calling remove_error_msg(args, kwargs) (line 62)
        remove_error_msg_call_result_3765 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), remove_error_msg_3762, *[self_3763], **kwargs_3764)
        
        
        # ################# End of 'turn_to_warning(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'turn_to_warning' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_3766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3766)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'turn_to_warning'
        return stypy_return_type_3766


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        TypeError.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.stypy__str__.__dict__.__setitem__('stypy_function_name', 'TypeError.stypy__str__')
        TypeError.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        TypeError.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        str_3767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'str', '\n        Visual representation of the error (full message: error + stack trace)\n        :return:\n        ')
        # Getting the type of 'self' (line 69)
        self_3768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'self')
        # Obtaining the member 'error_msg' of a type (line 69)
        error_msg_3769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 15), self_3768, 'error_msg')
        # Assigning a type to the variable 'stypy_return_type' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'stypy_return_type', error_msg_3769)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_3770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3770)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_3770


    @norecursion
    def __format_file_name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__format_file_name'
        module_type_store = module_type_store.open_function_context('__format_file_name', 71, 4, False)
        # Assigning a type to the variable 'self' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.__format_file_name.__dict__.__setitem__('stypy_localization', localization)
        TypeError.__format_file_name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.__format_file_name.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.__format_file_name.__dict__.__setitem__('stypy_function_name', 'TypeError.__format_file_name')
        TypeError.__format_file_name.__dict__.__setitem__('stypy_param_names_list', [])
        TypeError.__format_file_name.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.__format_file_name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.__format_file_name.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.__format_file_name.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.__format_file_name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.__format_file_name.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.__format_file_name', [], None, None, defaults, varargs, kwargs)

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

        str_3771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, (-1)), 'str', '\n        Pretty-prints file name\n        :return:\n        ')
        
        # Assigning a Subscript to a Name (line 76):
        
        # Obtaining the type of the subscript
        int_3772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 59), 'int')
        
        # Call to split(...): (line 76)
        # Processing the call arguments (line 76)
        str_3777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 54), 'str', '/')
        # Processing the call keyword arguments (line 76)
        kwargs_3778 = {}
        # Getting the type of 'self' (line 76)
        self_3773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'self', False)
        # Obtaining the member 'localization' of a type (line 76)
        localization_3774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 20), self_3773, 'localization')
        # Obtaining the member 'file_name' of a type (line 76)
        file_name_3775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 20), localization_3774, 'file_name')
        # Obtaining the member 'split' of a type (line 76)
        split_3776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 20), file_name_3775, 'split')
        # Calling split(args, kwargs) (line 76)
        split_call_result_3779 = invoke(stypy.reporting.localization.Localization(__file__, 76, 20), split_3776, *[str_3777], **kwargs_3778)
        
        # Obtaining the member '__getitem__' of a type (line 76)
        getitem___3780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 20), split_call_result_3779, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 76)
        subscript_call_result_3781 = invoke(stypy.reporting.localization.Localization(__file__, 76, 20), getitem___3780, int_3772)
        
        # Assigning a type to the variable 'file_name' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'file_name', subscript_call_result_3781)
        
        # Assigning a Subscript to a Name (line 77):
        
        # Obtaining the type of the subscript
        int_3782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 42), 'int')
        
        # Call to split(...): (line 77)
        # Processing the call arguments (line 77)
        str_3785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 36), 'str', '\\')
        # Processing the call keyword arguments (line 77)
        kwargs_3786 = {}
        # Getting the type of 'file_name' (line 77)
        file_name_3783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'file_name', False)
        # Obtaining the member 'split' of a type (line 77)
        split_3784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 20), file_name_3783, 'split')
        # Calling split(args, kwargs) (line 77)
        split_call_result_3787 = invoke(stypy.reporting.localization.Localization(__file__, 77, 20), split_3784, *[str_3785], **kwargs_3786)
        
        # Obtaining the member '__getitem__' of a type (line 77)
        getitem___3788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 20), split_call_result_3787, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 77)
        subscript_call_result_3789 = invoke(stypy.reporting.localization.Localization(__file__, 77, 20), getitem___3788, int_3782)
        
        # Assigning a type to the variable 'file_name' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'file_name', subscript_call_result_3789)
        
        # Assigning a Call to a Name (line 78):
        
        # Call to replace(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'stypy_parameters_copy' (line 78)
        stypy_parameters_copy_3792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 38), 'stypy_parameters_copy', False)
        # Obtaining the member 'type_inference_file_postfix' of a type (line 78)
        type_inference_file_postfix_3793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 38), stypy_parameters_copy_3792, 'type_inference_file_postfix')
        str_3794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 89), 'str', '')
        # Processing the call keyword arguments (line 78)
        kwargs_3795 = {}
        # Getting the type of 'file_name' (line 78)
        file_name_3790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'file_name', False)
        # Obtaining the member 'replace' of a type (line 78)
        replace_3791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 20), file_name_3790, 'replace')
        # Calling replace(args, kwargs) (line 78)
        replace_call_result_3796 = invoke(stypy.reporting.localization.Localization(__file__, 78, 20), replace_3791, *[type_inference_file_postfix_3793, str_3794], **kwargs_3795)
        
        # Assigning a type to the variable 'file_name' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'file_name', replace_call_result_3796)
        
        # Assigning a Call to a Name (line 79):
        
        # Call to replace(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'stypy_parameters_copy' (line 79)
        stypy_parameters_copy_3799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 38), 'stypy_parameters_copy', False)
        # Obtaining the member 'type_inference_file_directory_name' of a type (line 79)
        type_inference_file_directory_name_3800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 38), stypy_parameters_copy_3799, 'type_inference_file_directory_name')
        str_3801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 96), 'str', '')
        # Processing the call keyword arguments (line 79)
        kwargs_3802 = {}
        # Getting the type of 'file_name' (line 79)
        file_name_3797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'file_name', False)
        # Obtaining the member 'replace' of a type (line 79)
        replace_3798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 20), file_name_3797, 'replace')
        # Calling replace(args, kwargs) (line 79)
        replace_call_result_3803 = invoke(stypy.reporting.localization.Localization(__file__, 79, 20), replace_3798, *[type_inference_file_directory_name_3800, str_3801], **kwargs_3802)
        
        # Assigning a type to the variable 'file_name' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'file_name', replace_call_result_3803)
        # Getting the type of 'file_name' (line 81)
        file_name_3804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'file_name')
        # Assigning a type to the variable 'stypy_return_type' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'stypy_return_type', file_name_3804)
        
        # ################# End of '__format_file_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__format_file_name' in the type store
        # Getting the type of 'stypy_return_type' (line 71)
        stypy_return_type_3805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3805)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__format_file_name'
        return stypy_return_type_3805


    @norecursion
    def __msg(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__msg'
        module_type_store = module_type_store.open_function_context('__msg', 83, 4, False)
        # Assigning a type to the variable 'self' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.__msg.__dict__.__setitem__('stypy_localization', localization)
        TypeError.__msg.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.__msg.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.__msg.__dict__.__setitem__('stypy_function_name', 'TypeError.__msg')
        TypeError.__msg.__dict__.__setitem__('stypy_param_names_list', [])
        TypeError.__msg.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.__msg.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.__msg.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.__msg.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.__msg.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.__msg.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.__msg', [], None, None, defaults, varargs, kwargs)

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

        str_3806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', '\n        Composes the full error message, using the error message, the error localization, current file name and\n        the stack trace. If available, it also displays the source code line when the error is produced and a\n        ^ marker indicating the position within the error line.\n        :return:\n        ')
        
        # Assigning a Call to a Name (line 90):
        
        # Call to __format_file_name(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_3809 = {}
        # Getting the type of 'self' (line 90)
        self_3807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'self', False)
        # Obtaining the member '__format_file_name' of a type (line 90)
        format_file_name_3808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 20), self_3807, '__format_file_name')
        # Calling __format_file_name(args, kwargs) (line 90)
        format_file_name_call_result_3810 = invoke(stypy.reporting.localization.Localization(__file__, 90, 20), format_file_name_3808, *[], **kwargs_3809)
        
        # Assigning a type to the variable 'file_name' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'file_name', format_file_name_call_result_3810)
        
        # Assigning a Call to a Name (line 92):
        
        # Call to get_line_from_module_code(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'self' (line 92)
        self_3813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 68), 'self', False)
        # Obtaining the member 'localization' of a type (line 92)
        localization_3814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 68), self_3813, 'localization')
        # Obtaining the member 'file_name' of a type (line 92)
        file_name_3815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 68), localization_3814, 'file_name')
        # Getting the type of 'self' (line 92)
        self_3816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 97), 'self', False)
        # Obtaining the member 'localization' of a type (line 92)
        localization_3817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 97), self_3816, 'localization')
        # Obtaining the member 'line' of a type (line 92)
        line_3818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 97), localization_3817, 'line')
        # Processing the call keyword arguments (line 92)
        kwargs_3819 = {}
        # Getting the type of 'ModuleLineNumbering' (line 92)
        ModuleLineNumbering_3811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 22), 'ModuleLineNumbering', False)
        # Obtaining the member 'get_line_from_module_code' of a type (line 92)
        get_line_from_module_code_3812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 22), ModuleLineNumbering_3811, 'get_line_from_module_code')
        # Calling get_line_from_module_code(args, kwargs) (line 92)
        get_line_from_module_code_call_result_3820 = invoke(stypy.reporting.localization.Localization(__file__, 92, 22), get_line_from_module_code_3812, *[file_name_3815, line_3818], **kwargs_3819)
        
        # Assigning a type to the variable 'source_code' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'source_code', get_line_from_module_code_call_result_3820)
        
        # Assigning a Call to a Name (line 93):
        
        # Call to get_column_from_module_code(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'self' (line 93)
        self_3823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 69), 'self', False)
        # Obtaining the member 'localization' of a type (line 93)
        localization_3824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 69), self_3823, 'localization')
        # Obtaining the member 'file_name' of a type (line 93)
        file_name_3825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 69), localization_3824, 'file_name')
        # Getting the type of 'self' (line 94)
        self_3826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 69), 'self', False)
        # Obtaining the member 'localization' of a type (line 94)
        localization_3827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 69), self_3826, 'localization')
        # Obtaining the member 'line' of a type (line 94)
        line_3828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 69), localization_3827, 'line')
        # Getting the type of 'self' (line 94)
        self_3829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 93), 'self', False)
        # Obtaining the member 'localization' of a type (line 94)
        localization_3830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 93), self_3829, 'localization')
        # Obtaining the member 'column' of a type (line 94)
        column_3831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 93), localization_3830, 'column')
        # Processing the call keyword arguments (line 93)
        kwargs_3832 = {}
        # Getting the type of 'ModuleLineNumbering' (line 93)
        ModuleLineNumbering_3821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 21), 'ModuleLineNumbering', False)
        # Obtaining the member 'get_column_from_module_code' of a type (line 93)
        get_column_from_module_code_3822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 21), ModuleLineNumbering_3821, 'get_column_from_module_code')
        # Calling get_column_from_module_code(args, kwargs) (line 93)
        get_column_from_module_code_call_result_3833 = invoke(stypy.reporting.localization.Localization(__file__, 93, 21), get_column_from_module_code_3822, *[file_name_3825, line_3828, column_3831], **kwargs_3832)
        
        # Assigning a type to the variable 'col_offset' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'col_offset', get_column_from_module_code_call_result_3833)
        
        # Type idiom detected: calculating its left and rigth part (line 95)
        # Getting the type of 'source_code' (line 95)
        source_code_3834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'source_code')
        # Getting the type of 'None' (line 95)
        None_3835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 30), 'None')
        
        (may_be_3836, more_types_in_union_3837) = may_not_be_none(source_code_3834, None_3835)

        if may_be_3836:

            if more_types_in_union_3837:
                # Runtime conditional SSA (line 95)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            str_3838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 19), 'str', "Compiler error in file '%s' (line %s, column %s):\n%s\n%s\n\t%s.\n\n%s")
            
            # Obtaining an instance of the builtin type 'tuple' (line 97)
            tuple_3839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 20), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 97)
            # Adding element type (line 97)
            # Getting the type of 'file_name' (line 97)
            file_name_3840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'file_name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 20), tuple_3839, file_name_3840)
            # Adding element type (line 97)
            # Getting the type of 'self' (line 97)
            self_3841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 31), 'self')
            # Obtaining the member 'localization' of a type (line 97)
            localization_3842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 31), self_3841, 'localization')
            # Obtaining the member 'line' of a type (line 97)
            line_3843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 31), localization_3842, 'line')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 20), tuple_3839, line_3843)
            # Adding element type (line 97)
            # Getting the type of 'self' (line 97)
            self_3844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 55), 'self')
            # Obtaining the member 'localization' of a type (line 97)
            localization_3845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 55), self_3844, 'localization')
            # Obtaining the member 'column' of a type (line 97)
            column_3846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 55), localization_3845, 'column')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 20), tuple_3839, column_3846)
            # Adding element type (line 97)
            # Getting the type of 'source_code' (line 98)
            source_code_3847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'source_code')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 20), tuple_3839, source_code_3847)
            # Adding element type (line 97)
            str_3848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 33), 'str', '')
            # Getting the type of 'col_offset' (line 98)
            col_offset_3849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 38), 'col_offset')
            # Applying the binary operator '+' (line 98)
            result_add_3850 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 33), '+', str_3848, col_offset_3849)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 20), tuple_3839, result_add_3850)
            # Adding element type (line 97)
            
            # Call to strip(...): (line 99)
            # Processing the call keyword arguments (line 99)
            kwargs_3854 = {}
            # Getting the type of 'self' (line 99)
            self_3851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'self', False)
            # Obtaining the member 'msg' of a type (line 99)
            msg_3852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 20), self_3851, 'msg')
            # Obtaining the member 'strip' of a type (line 99)
            strip_3853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 20), msg_3852, 'strip')
            # Calling strip(args, kwargs) (line 99)
            strip_call_result_3855 = invoke(stypy.reporting.localization.Localization(__file__, 99, 20), strip_3853, *[], **kwargs_3854)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 20), tuple_3839, strip_call_result_3855)
            # Adding element type (line 97)
            # Getting the type of 'self' (line 99)
            self_3856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 38), 'self')
            # Obtaining the member 'localization' of a type (line 99)
            localization_3857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 38), self_3856, 'localization')
            # Obtaining the member 'stack_trace' of a type (line 99)
            stack_trace_3858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 38), localization_3857, 'stack_trace')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 20), tuple_3839, stack_trace_3858)
            
            # Applying the binary operator '%' (line 96)
            result_mod_3859 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 19), '%', str_3838, tuple_3839)
            
            # Assigning a type to the variable 'stypy_return_type' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'stypy_return_type', result_mod_3859)

            if more_types_in_union_3837:
                # SSA join for if statement (line 95)
                module_type_store = module_type_store.join_ssa_context()


        
        str_3860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 15), 'str', "Compiler error in file '%s' (line %s, column %s):\n%s.\n\n%s")
        
        # Obtaining an instance of the builtin type 'tuple' (line 102)
        tuple_3861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 102)
        # Adding element type (line 102)
        # Getting the type of 'file_name' (line 102)
        file_name_3862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'file_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 16), tuple_3861, file_name_3862)
        # Adding element type (line 102)
        # Getting the type of 'self' (line 102)
        self_3863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 'self')
        # Obtaining the member 'localization' of a type (line 102)
        localization_3864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 27), self_3863, 'localization')
        # Obtaining the member 'line' of a type (line 102)
        line_3865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 27), localization_3864, 'line')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 16), tuple_3861, line_3865)
        # Adding element type (line 102)
        # Getting the type of 'self' (line 102)
        self_3866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 51), 'self')
        # Obtaining the member 'localization' of a type (line 102)
        localization_3867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 51), self_3866, 'localization')
        # Obtaining the member 'column' of a type (line 102)
        column_3868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 51), localization_3867, 'column')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 16), tuple_3861, column_3868)
        # Adding element type (line 102)
        # Getting the type of 'self' (line 103)
        self_3869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'self')
        # Obtaining the member 'msg' of a type (line 103)
        msg_3870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 16), self_3869, 'msg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 16), tuple_3861, msg_3870)
        # Adding element type (line 102)
        # Getting the type of 'self' (line 103)
        self_3871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'self')
        # Obtaining the member 'localization' of a type (line 103)
        localization_3872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 26), self_3871, 'localization')
        # Obtaining the member 'stack_trace' of a type (line 103)
        stack_trace_3873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 26), localization_3872, 'stack_trace')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 16), tuple_3861, stack_trace_3873)
        
        # Applying the binary operator '%' (line 101)
        result_mod_3874 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 15), '%', str_3860, tuple_3861)
        
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'stypy_return_type', result_mod_3874)
        
        # ################# End of '__msg(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__msg' in the type store
        # Getting the type of 'stypy_return_type' (line 83)
        stypy_return_type_3875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3875)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__msg'
        return stypy_return_type_3875


    @staticmethod
    @norecursion
    def print_error_msgs(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'print_error_msgs'
        module_type_store = module_type_store.open_function_context('print_error_msgs', 105, 4, False)
        
        # Passed parameters checking function
        TypeError.print_error_msgs.__dict__.__setitem__('stypy_localization', localization)
        TypeError.print_error_msgs.__dict__.__setitem__('stypy_type_of_self', None)
        TypeError.print_error_msgs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.print_error_msgs.__dict__.__setitem__('stypy_function_name', 'print_error_msgs')
        TypeError.print_error_msgs.__dict__.__setitem__('stypy_param_names_list', [])
        TypeError.print_error_msgs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.print_error_msgs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.print_error_msgs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.print_error_msgs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.print_error_msgs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.print_error_msgs.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'print_error_msgs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'print_error_msgs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'print_error_msgs(...)' code ##################

        str_3876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, (-1)), 'str', '\n        Prints all the error messages that were produced during a program analysis. Just for debugging\n        :return:\n        ')
        
        # Getting the type of 'TypeError' (line 111)
        TypeError_3877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'TypeError')
        # Obtaining the member 'errors' of a type (line 111)
        errors_3878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 19), TypeError_3877, 'errors')
        # Assigning a type to the variable 'errors_3878' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'errors_3878', errors_3878)
        # Testing if the for loop is going to be iterated (line 111)
        # Testing the type of a for loop iterable (line 111)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 8), errors_3878)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 111, 8), errors_3878):
            # Getting the type of the for loop variable (line 111)
            for_loop_var_3879 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 8), errors_3878)
            # Assigning a type to the variable 'err' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'err', for_loop_var_3879)
            # SSA begins for a for statement (line 111)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'err' (line 112)
            err_3880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 18), 'err')
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'print_error_msgs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'print_error_msgs' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_3881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3881)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'print_error_msgs'
        return stypy_return_type_3881


    @staticmethod
    @norecursion
    def get_error_msgs(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_error_msgs'
        module_type_store = module_type_store.open_function_context('get_error_msgs', 114, 4, False)
        
        # Passed parameters checking function
        TypeError.get_error_msgs.__dict__.__setitem__('stypy_localization', localization)
        TypeError.get_error_msgs.__dict__.__setitem__('stypy_type_of_self', None)
        TypeError.get_error_msgs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.get_error_msgs.__dict__.__setitem__('stypy_function_name', 'get_error_msgs')
        TypeError.get_error_msgs.__dict__.__setitem__('stypy_param_names_list', [])
        TypeError.get_error_msgs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.get_error_msgs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.get_error_msgs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.get_error_msgs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.get_error_msgs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.get_error_msgs.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'get_error_msgs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_error_msgs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_error_msgs(...)' code ##################

        str_3882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, (-1)), 'str', '\n        Gets all the error messages that were produced during a program analysis.\n        :return: All the errors, sorted by line number\n        ')
        
        # Call to sorted(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'TypeError' (line 120)
        TypeError_3884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'TypeError', False)
        # Obtaining the member 'errors' of a type (line 120)
        errors_3885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 22), TypeError_3884, 'errors')
        # Processing the call keyword arguments (line 120)

        @norecursion
        def _stypy_temp_lambda_4(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_4'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_4', 120, 44, True)
            # Passed parameters checking function
            _stypy_temp_lambda_4.stypy_localization = localization
            _stypy_temp_lambda_4.stypy_type_of_self = None
            _stypy_temp_lambda_4.stypy_type_store = module_type_store
            _stypy_temp_lambda_4.stypy_function_name = '_stypy_temp_lambda_4'
            _stypy_temp_lambda_4.stypy_param_names_list = ['error']
            _stypy_temp_lambda_4.stypy_varargs_param_name = None
            _stypy_temp_lambda_4.stypy_kwargs_param_name = None
            _stypy_temp_lambda_4.stypy_call_defaults = defaults
            _stypy_temp_lambda_4.stypy_call_varargs = varargs
            _stypy_temp_lambda_4.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_4', ['error'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_4', ['error'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'error' (line 120)
            error_3886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 58), 'error', False)
            # Obtaining the member 'localization' of a type (line 120)
            localization_3887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 58), error_3886, 'localization')
            # Obtaining the member 'line' of a type (line 120)
            line_3888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 58), localization_3887, 'line')
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 120)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 44), 'stypy_return_type', line_3888)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_4' in the type store
            # Getting the type of 'stypy_return_type' (line 120)
            stypy_return_type_3889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 44), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3889)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_4'
            return stypy_return_type_3889

        # Assigning a type to the variable '_stypy_temp_lambda_4' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 44), '_stypy_temp_lambda_4', _stypy_temp_lambda_4)
        # Getting the type of '_stypy_temp_lambda_4' (line 120)
        _stypy_temp_lambda_4_3890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 44), '_stypy_temp_lambda_4')
        keyword_3891 = _stypy_temp_lambda_4_3890
        kwargs_3892 = {'key': keyword_3891}
        # Getting the type of 'sorted' (line 120)
        sorted_3883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'sorted', False)
        # Calling sorted(args, kwargs) (line 120)
        sorted_call_result_3893 = invoke(stypy.reporting.localization.Localization(__file__, 120, 15), sorted_3883, *[errors_3885], **kwargs_3892)
        
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', sorted_call_result_3893)
        
        # ################# End of 'get_error_msgs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_error_msgs' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_3894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3894)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_error_msgs'
        return stypy_return_type_3894


    @staticmethod
    @norecursion
    def remove_error_msg(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'remove_error_msg'
        module_type_store = module_type_store.open_function_context('remove_error_msg', 122, 4, False)
        
        # Passed parameters checking function
        TypeError.remove_error_msg.__dict__.__setitem__('stypy_localization', localization)
        TypeError.remove_error_msg.__dict__.__setitem__('stypy_type_of_self', None)
        TypeError.remove_error_msg.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.remove_error_msg.__dict__.__setitem__('stypy_function_name', 'remove_error_msg')
        TypeError.remove_error_msg.__dict__.__setitem__('stypy_param_names_list', ['error_obj'])
        TypeError.remove_error_msg.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.remove_error_msg.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.remove_error_msg.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.remove_error_msg.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.remove_error_msg.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.remove_error_msg.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, 'remove_error_msg', ['error_obj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'remove_error_msg', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'remove_error_msg(...)' code ##################

        str_3895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, (-1)), 'str', '\n        Deletes an error message from the global error list. As we said, error messages might be turn to warnings, so\n        we must delete them afterwards\n        :param error_obj:\n        :return:\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 130)
        # Getting the type of 'list' (line 130)
        list_3896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 33), 'list')
        # Getting the type of 'error_obj' (line 130)
        error_obj_3897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 22), 'error_obj')
        
        (may_be_3898, more_types_in_union_3899) = may_be_subtype(list_3896, error_obj_3897)

        if may_be_3898:

            if more_types_in_union_3899:
                # Runtime conditional SSA (line 130)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'error_obj' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'error_obj', remove_not_subtype_from_union(error_obj_3897, list))
            
            # Getting the type of 'error_obj' (line 131)
            error_obj_3900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 25), 'error_obj')
            # Assigning a type to the variable 'error_obj_3900' (line 131)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'error_obj_3900', error_obj_3900)
            # Testing if the for loop is going to be iterated (line 131)
            # Testing the type of a for loop iterable (line 131)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 131, 12), error_obj_3900)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 131, 12), error_obj_3900):
                # Getting the type of the for loop variable (line 131)
                for_loop_var_3901 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 131, 12), error_obj_3900)
                # Assigning a type to the variable 'error' (line 131)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'error', for_loop_var_3901)
                # SSA begins for a for statement (line 131)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to remove(...): (line 132)
                # Processing the call arguments (line 132)
                # Getting the type of 'error' (line 132)
                error_3905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 40), 'error', False)
                # Processing the call keyword arguments (line 132)
                kwargs_3906 = {}
                # Getting the type of 'TypeError' (line 132)
                TypeError_3902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'TypeError', False)
                # Obtaining the member 'errors' of a type (line 132)
                errors_3903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 16), TypeError_3902, 'errors')
                # Obtaining the member 'remove' of a type (line 132)
                remove_3904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 16), errors_3903, 'remove')
                # Calling remove(args, kwargs) (line 132)
                remove_call_result_3907 = invoke(stypy.reporting.localization.Localization(__file__, 132, 16), remove_3904, *[error_3905], **kwargs_3906)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            

            if more_types_in_union_3899:
                # Runtime conditional SSA for else branch (line 130)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_3898) or more_types_in_union_3899):
            # Assigning a type to the variable 'error_obj' (line 130)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'error_obj', remove_subtype_from_union(error_obj_3897, list))
            
            
            # SSA begins for try-except statement (line 134)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Call to remove(...): (line 135)
            # Processing the call arguments (line 135)
            # Getting the type of 'error_obj' (line 135)
            error_obj_3911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 40), 'error_obj', False)
            # Processing the call keyword arguments (line 135)
            kwargs_3912 = {}
            # Getting the type of 'TypeError' (line 135)
            TypeError_3908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'TypeError', False)
            # Obtaining the member 'errors' of a type (line 135)
            errors_3909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 16), TypeError_3908, 'errors')
            # Obtaining the member 'remove' of a type (line 135)
            remove_3910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 16), errors_3909, 'remove')
            # Calling remove(args, kwargs) (line 135)
            remove_call_result_3913 = invoke(stypy.reporting.localization.Localization(__file__, 135, 16), remove_3910, *[error_obj_3911], **kwargs_3912)
            
            # SSA branch for the except part of a try statement (line 134)
            # SSA branch for the except '<any exception>' branch of a try statement (line 134)
            module_type_store.open_ssa_branch('except')
            pass
            # SSA join for try-except statement (line 134)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_3898 and more_types_in_union_3899):
                # SSA join for if statement (line 130)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'remove_error_msg(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'remove_error_msg' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_3914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3914)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'remove_error_msg'
        return stypy_return_type_3914


    @staticmethod
    @norecursion
    def reset_error_msgs(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'reset_error_msgs'
        module_type_store = module_type_store.open_function_context('reset_error_msgs', 139, 4, False)
        
        # Passed parameters checking function
        TypeError.reset_error_msgs.__dict__.__setitem__('stypy_localization', localization)
        TypeError.reset_error_msgs.__dict__.__setitem__('stypy_type_of_self', None)
        TypeError.reset_error_msgs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.reset_error_msgs.__dict__.__setitem__('stypy_function_name', 'reset_error_msgs')
        TypeError.reset_error_msgs.__dict__.__setitem__('stypy_param_names_list', [])
        TypeError.reset_error_msgs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.reset_error_msgs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.reset_error_msgs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.reset_error_msgs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.reset_error_msgs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.reset_error_msgs.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'reset_error_msgs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reset_error_msgs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reset_error_msgs(...)' code ##################

        str_3915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, (-1)), 'str', '\n        Clears the global error message list\n        :return:\n        ')
        
        # Assigning a List to a Attribute (line 145):
        
        # Obtaining an instance of the builtin type 'list' (line 145)
        list_3916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 145)
        
        # Getting the type of 'TypeError' (line 145)
        TypeError_3917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'TypeError')
        # Setting the type of the member 'errors' of a type (line 145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), TypeError_3917, 'errors', list_3916)
        
        # ################# End of 'reset_error_msgs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reset_error_msgs' in the type store
        # Getting the type of 'stypy_return_type' (line 139)
        stypy_return_type_3918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3918)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reset_error_msgs'
        return stypy_return_type_3918

    str_3919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, (-1)), 'str', "\n    As errors are also stypy Type objects, they must provide the rest of its interface methods in order to allow\n    the analysis of the program in an orthogonal fashion. These method do nothing, as they don't make sense within\n    a TypeError. If methods of this object report errors upon called, the error reporting will display repeated\n    errors at the end.\n    ")

    @norecursion
    def get_python_entity(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_python_entity'
        module_type_store = module_type_store.open_function_context('get_python_entity', 154, 4, False)
        # Assigning a type to the variable 'self' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.get_python_entity.__dict__.__setitem__('stypy_localization', localization)
        TypeError.get_python_entity.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.get_python_entity.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.get_python_entity.__dict__.__setitem__('stypy_function_name', 'TypeError.get_python_entity')
        TypeError.get_python_entity.__dict__.__setitem__('stypy_param_names_list', [])
        TypeError.get_python_entity.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.get_python_entity.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.get_python_entity.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.get_python_entity.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.get_python_entity.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.get_python_entity.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.get_python_entity', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_python_entity', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_python_entity(...)' code ##################

        # Getting the type of 'self' (line 155)
        self_3920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'stypy_return_type', self_3920)
        
        # ################# End of 'get_python_entity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_python_entity' in the type store
        # Getting the type of 'stypy_return_type' (line 154)
        stypy_return_type_3921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3921)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_python_entity'
        return stypy_return_type_3921


    @norecursion
    def get_python_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_python_type'
        module_type_store = module_type_store.open_function_context('get_python_type', 157, 4, False)
        # Assigning a type to the variable 'self' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.get_python_type.__dict__.__setitem__('stypy_localization', localization)
        TypeError.get_python_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.get_python_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.get_python_type.__dict__.__setitem__('stypy_function_name', 'TypeError.get_python_type')
        TypeError.get_python_type.__dict__.__setitem__('stypy_param_names_list', [])
        TypeError.get_python_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.get_python_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.get_python_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.get_python_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.get_python_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.get_python_type.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.get_python_type', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_python_type', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_python_type(...)' code ##################

        # Getting the type of 'self' (line 158)
        self_3922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'stypy_return_type', self_3922)
        
        # ################# End of 'get_python_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_python_type' in the type store
        # Getting the type of 'stypy_return_type' (line 157)
        stypy_return_type_3923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3923)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_python_type'
        return stypy_return_type_3923


    @norecursion
    def get_instance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_instance'
        module_type_store = module_type_store.open_function_context('get_instance', 160, 4, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.get_instance.__dict__.__setitem__('stypy_localization', localization)
        TypeError.get_instance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.get_instance.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.get_instance.__dict__.__setitem__('stypy_function_name', 'TypeError.get_instance')
        TypeError.get_instance.__dict__.__setitem__('stypy_param_names_list', [])
        TypeError.get_instance.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.get_instance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.get_instance.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.get_instance.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.get_instance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.get_instance.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.get_instance', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_instance', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_instance(...)' code ##################

        # Getting the type of 'None' (line 161)
        None_3924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'stypy_return_type', None_3924)
        
        # ################# End of 'get_instance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_instance' in the type store
        # Getting the type of 'stypy_return_type' (line 160)
        stypy_return_type_3925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3925)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_instance'
        return stypy_return_type_3925


    @norecursion
    def can_store_elements(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'can_store_elements'
        module_type_store = module_type_store.open_function_context('can_store_elements', 165, 4, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.can_store_elements.__dict__.__setitem__('stypy_localization', localization)
        TypeError.can_store_elements.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.can_store_elements.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.can_store_elements.__dict__.__setitem__('stypy_function_name', 'TypeError.can_store_elements')
        TypeError.can_store_elements.__dict__.__setitem__('stypy_param_names_list', [])
        TypeError.can_store_elements.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.can_store_elements.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.can_store_elements.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.can_store_elements.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.can_store_elements.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.can_store_elements.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.can_store_elements', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'can_store_elements', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'can_store_elements(...)' code ##################

        # Getting the type of 'False' (line 166)
        False_3926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'stypy_return_type', False_3926)
        
        # ################# End of 'can_store_elements(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'can_store_elements' in the type store
        # Getting the type of 'stypy_return_type' (line 165)
        stypy_return_type_3927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3927)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'can_store_elements'
        return stypy_return_type_3927


    @norecursion
    def can_store_keypairs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'can_store_keypairs'
        module_type_store = module_type_store.open_function_context('can_store_keypairs', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.can_store_keypairs.__dict__.__setitem__('stypy_localization', localization)
        TypeError.can_store_keypairs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.can_store_keypairs.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.can_store_keypairs.__dict__.__setitem__('stypy_function_name', 'TypeError.can_store_keypairs')
        TypeError.can_store_keypairs.__dict__.__setitem__('stypy_param_names_list', [])
        TypeError.can_store_keypairs.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.can_store_keypairs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.can_store_keypairs.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.can_store_keypairs.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.can_store_keypairs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.can_store_keypairs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.can_store_keypairs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'can_store_keypairs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'can_store_keypairs(...)' code ##################

        # Getting the type of 'False' (line 169)
        False_3928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'stypy_return_type', False_3928)
        
        # ################# End of 'can_store_keypairs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'can_store_keypairs' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_3929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3929)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'can_store_keypairs'
        return stypy_return_type_3929


    @norecursion
    def get_elements_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_elements_type'
        module_type_store = module_type_store.open_function_context('get_elements_type', 171, 4, False)
        # Assigning a type to the variable 'self' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.get_elements_type.__dict__.__setitem__('stypy_localization', localization)
        TypeError.get_elements_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.get_elements_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.get_elements_type.__dict__.__setitem__('stypy_function_name', 'TypeError.get_elements_type')
        TypeError.get_elements_type.__dict__.__setitem__('stypy_param_names_list', [])
        TypeError.get_elements_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.get_elements_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.get_elements_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.get_elements_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.get_elements_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.get_elements_type.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.get_elements_type', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_elements_type', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_elements_type(...)' code ##################

        # Getting the type of 'self' (line 172)
        self_3930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'stypy_return_type', self_3930)
        
        # ################# End of 'get_elements_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_elements_type' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_3931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3931)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_elements_type'
        return stypy_return_type_3931


    @norecursion
    def is_empty(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_empty'
        module_type_store = module_type_store.open_function_context('is_empty', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.is_empty.__dict__.__setitem__('stypy_localization', localization)
        TypeError.is_empty.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.is_empty.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.is_empty.__dict__.__setitem__('stypy_function_name', 'TypeError.is_empty')
        TypeError.is_empty.__dict__.__setitem__('stypy_param_names_list', [])
        TypeError.is_empty.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.is_empty.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.is_empty.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.is_empty.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.is_empty.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.is_empty.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.is_empty', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_empty', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_empty(...)' code ##################

        # Getting the type of 'self' (line 175)
        self_3932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'stypy_return_type', self_3932)
        
        # ################# End of 'is_empty(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_empty' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_3933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3933)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_empty'
        return stypy_return_type_3933


    @norecursion
    def set_elements_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 177)
        True_3934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 79), 'True')
        defaults = [True_3934]
        # Create a new context for function 'set_elements_type'
        module_type_store = module_type_store.open_function_context('set_elements_type', 177, 4, False)
        # Assigning a type to the variable 'self' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.set_elements_type.__dict__.__setitem__('stypy_localization', localization)
        TypeError.set_elements_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.set_elements_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.set_elements_type.__dict__.__setitem__('stypy_function_name', 'TypeError.set_elements_type')
        TypeError.set_elements_type.__dict__.__setitem__('stypy_param_names_list', ['localization', 'elements_type', 'record_annotation'])
        TypeError.set_elements_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.set_elements_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.set_elements_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.set_elements_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.set_elements_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.set_elements_type.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.set_elements_type', ['localization', 'elements_type', 'record_annotation'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_elements_type', localization, ['localization', 'elements_type', 'record_annotation'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_elements_type(...)' code ##################

        # Getting the type of 'self' (line 178)
        self_3935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'stypy_return_type', self_3935)
        
        # ################# End of 'set_elements_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_elements_type' in the type store
        # Getting the type of 'stypy_return_type' (line 177)
        stypy_return_type_3936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3936)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_elements_type'
        return stypy_return_type_3936


    @norecursion
    def add_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 180)
        True_3937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 62), 'True')
        defaults = [True_3937]
        # Create a new context for function 'add_type'
        module_type_store = module_type_store.open_function_context('add_type', 180, 4, False)
        # Assigning a type to the variable 'self' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.add_type.__dict__.__setitem__('stypy_localization', localization)
        TypeError.add_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.add_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.add_type.__dict__.__setitem__('stypy_function_name', 'TypeError.add_type')
        TypeError.add_type.__dict__.__setitem__('stypy_param_names_list', ['localization', 'type_', 'record_annotation'])
        TypeError.add_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.add_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.add_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.add_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.add_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.add_type.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.add_type', ['localization', 'type_', 'record_annotation'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_type', localization, ['localization', 'type_', 'record_annotation'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_type(...)' code ##################

        # Getting the type of 'self' (line 181)
        self_3938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'stypy_return_type', self_3938)
        
        # ################# End of 'add_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_type' in the type store
        # Getting the type of 'stypy_return_type' (line 180)
        stypy_return_type_3939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3939)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_type'
        return stypy_return_type_3939


    @norecursion
    def add_types_from_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 183)
        True_3940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 77), 'True')
        defaults = [True_3940]
        # Create a new context for function 'add_types_from_list'
        module_type_store = module_type_store.open_function_context('add_types_from_list', 183, 4, False)
        # Assigning a type to the variable 'self' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.add_types_from_list.__dict__.__setitem__('stypy_localization', localization)
        TypeError.add_types_from_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.add_types_from_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.add_types_from_list.__dict__.__setitem__('stypy_function_name', 'TypeError.add_types_from_list')
        TypeError.add_types_from_list.__dict__.__setitem__('stypy_param_names_list', ['localization', 'type_list', 'record_annotation'])
        TypeError.add_types_from_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.add_types_from_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.add_types_from_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.add_types_from_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.add_types_from_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.add_types_from_list.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.add_types_from_list', ['localization', 'type_list', 'record_annotation'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_types_from_list', localization, ['localization', 'type_list', 'record_annotation'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_types_from_list(...)' code ##################

        # Getting the type of 'self' (line 184)
        self_3941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'stypy_return_type', self_3941)
        
        # ################# End of 'add_types_from_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_types_from_list' in the type store
        # Getting the type of 'stypy_return_type' (line 183)
        stypy_return_type_3942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3942)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_types_from_list'
        return stypy_return_type_3942


    @norecursion
    def add_key_and_value_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'True' (line 186)
        True_3943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 81), 'True')
        defaults = [True_3943]
        # Create a new context for function 'add_key_and_value_type'
        module_type_store = module_type_store.open_function_context('add_key_and_value_type', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.add_key_and_value_type.__dict__.__setitem__('stypy_localization', localization)
        TypeError.add_key_and_value_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.add_key_and_value_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.add_key_and_value_type.__dict__.__setitem__('stypy_function_name', 'TypeError.add_key_and_value_type')
        TypeError.add_key_and_value_type.__dict__.__setitem__('stypy_param_names_list', ['localization', 'type_tuple', 'record_annotation'])
        TypeError.add_key_and_value_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.add_key_and_value_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.add_key_and_value_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.add_key_and_value_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.add_key_and_value_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.add_key_and_value_type.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.add_key_and_value_type', ['localization', 'type_tuple', 'record_annotation'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_key_and_value_type', localization, ['localization', 'type_tuple', 'record_annotation'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_key_and_value_type(...)' code ##################

        # Getting the type of 'self' (line 187)
        self_3944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'stypy_return_type', self_3944)
        
        # ################# End of 'add_key_and_value_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_key_and_value_type' in the type store
        # Getting the type of 'stypy_return_type' (line 186)
        stypy_return_type_3945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3945)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_key_and_value_type'
        return stypy_return_type_3945


    @norecursion
    def get_type_of_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_type_of_member'
        module_type_store = module_type_store.open_function_context('get_type_of_member', 191, 4, False)
        # Assigning a type to the variable 'self' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.get_type_of_member.__dict__.__setitem__('stypy_localization', localization)
        TypeError.get_type_of_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.get_type_of_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.get_type_of_member.__dict__.__setitem__('stypy_function_name', 'TypeError.get_type_of_member')
        TypeError.get_type_of_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member_name'])
        TypeError.get_type_of_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.get_type_of_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.get_type_of_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.get_type_of_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.get_type_of_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.get_type_of_member.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.get_type_of_member', ['localization', 'member_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_type_of_member', localization, ['localization', 'member_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_type_of_member(...)' code ##################

        # Getting the type of 'self' (line 192)
        self_3946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'stypy_return_type', self_3946)
        
        # ################# End of 'get_type_of_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_type_of_member' in the type store
        # Getting the type of 'stypy_return_type' (line 191)
        stypy_return_type_3947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3947)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_type_of_member'
        return stypy_return_type_3947


    @norecursion
    def set_type_of_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_type_of_member'
        module_type_store = module_type_store.open_function_context('set_type_of_member', 194, 4, False)
        # Assigning a type to the variable 'self' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.set_type_of_member.__dict__.__setitem__('stypy_localization', localization)
        TypeError.set_type_of_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.set_type_of_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.set_type_of_member.__dict__.__setitem__('stypy_function_name', 'TypeError.set_type_of_member')
        TypeError.set_type_of_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member_name', 'member_value'])
        TypeError.set_type_of_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.set_type_of_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.set_type_of_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.set_type_of_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.set_type_of_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.set_type_of_member.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.set_type_of_member', ['localization', 'member_name', 'member_value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_type_of_member', localization, ['localization', 'member_name', 'member_value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_type_of_member(...)' code ##################

        # Getting the type of 'self' (line 195)
        self_3948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'stypy_return_type', self_3948)
        
        # ################# End of 'set_type_of_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_type_of_member' in the type store
        # Getting the type of 'stypy_return_type' (line 194)
        stypy_return_type_3949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3949)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_type_of_member'
        return stypy_return_type_3949


    @norecursion
    def invoke(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'invoke'
        module_type_store = module_type_store.open_function_context('invoke', 199, 4, False)
        # Assigning a type to the variable 'self' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.invoke.__dict__.__setitem__('stypy_localization', localization)
        TypeError.invoke.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.invoke.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.invoke.__dict__.__setitem__('stypy_function_name', 'TypeError.invoke')
        TypeError.invoke.__dict__.__setitem__('stypy_param_names_list', ['localization'])
        TypeError.invoke.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        TypeError.invoke.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        TypeError.invoke.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.invoke.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.invoke.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.invoke.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.invoke', ['localization'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'invoke', localization, ['localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'invoke(...)' code ##################

        # Getting the type of 'self' (line 200)
        self_3950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'stypy_return_type', self_3950)
        
        # ################# End of 'invoke(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'invoke' in the type store
        # Getting the type of 'stypy_return_type' (line 199)
        stypy_return_type_3951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3951)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'invoke'
        return stypy_return_type_3951


    @norecursion
    def delete_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'delete_member'
        module_type_store = module_type_store.open_function_context('delete_member', 204, 4, False)
        # Assigning a type to the variable 'self' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.delete_member.__dict__.__setitem__('stypy_localization', localization)
        TypeError.delete_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.delete_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.delete_member.__dict__.__setitem__('stypy_function_name', 'TypeError.delete_member')
        TypeError.delete_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member'])
        TypeError.delete_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.delete_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.delete_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.delete_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.delete_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.delete_member.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.delete_member', ['localization', 'member'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'delete_member', localization, ['localization', 'member'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'delete_member(...)' code ##################

        # Getting the type of 'self' (line 205)
        self_3952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'stypy_return_type', self_3952)
        
        # ################# End of 'delete_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'delete_member' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_3953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3953)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'delete_member'
        return stypy_return_type_3953


    @norecursion
    def supports_structural_reflection(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'supports_structural_reflection'
        module_type_store = module_type_store.open_function_context('supports_structural_reflection', 207, 4, False)
        # Assigning a type to the variable 'self' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.supports_structural_reflection.__dict__.__setitem__('stypy_localization', localization)
        TypeError.supports_structural_reflection.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.supports_structural_reflection.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.supports_structural_reflection.__dict__.__setitem__('stypy_function_name', 'TypeError.supports_structural_reflection')
        TypeError.supports_structural_reflection.__dict__.__setitem__('stypy_param_names_list', [])
        TypeError.supports_structural_reflection.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.supports_structural_reflection.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.supports_structural_reflection.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.supports_structural_reflection.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.supports_structural_reflection.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.supports_structural_reflection.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.supports_structural_reflection', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'supports_structural_reflection', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'supports_structural_reflection(...)' code ##################

        # Getting the type of 'False' (line 208)
        False_3954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type', False_3954)
        
        # ################# End of 'supports_structural_reflection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'supports_structural_reflection' in the type store
        # Getting the type of 'stypy_return_type' (line 207)
        stypy_return_type_3955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3955)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'supports_structural_reflection'
        return stypy_return_type_3955


    @norecursion
    def change_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'change_type'
        module_type_store = module_type_store.open_function_context('change_type', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.change_type.__dict__.__setitem__('stypy_localization', localization)
        TypeError.change_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.change_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.change_type.__dict__.__setitem__('stypy_function_name', 'TypeError.change_type')
        TypeError.change_type.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_type'])
        TypeError.change_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.change_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.change_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.change_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.change_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.change_type.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.change_type', ['localization', 'new_type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'change_type', localization, ['localization', 'new_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'change_type(...)' code ##################

        # Getting the type of 'self' (line 211)
        self_3956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'stypy_return_type', self_3956)
        
        # ################# End of 'change_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_type' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_3957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3957)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_type'
        return stypy_return_type_3957


    @norecursion
    def change_base_types(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'change_base_types'
        module_type_store = module_type_store.open_function_context('change_base_types', 213, 4, False)
        # Assigning a type to the variable 'self' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.change_base_types.__dict__.__setitem__('stypy_localization', localization)
        TypeError.change_base_types.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.change_base_types.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.change_base_types.__dict__.__setitem__('stypy_function_name', 'TypeError.change_base_types')
        TypeError.change_base_types.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_types'])
        TypeError.change_base_types.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.change_base_types.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.change_base_types.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.change_base_types.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.change_base_types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.change_base_types.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.change_base_types', ['localization', 'new_types'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'change_base_types', localization, ['localization', 'new_types'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'change_base_types(...)' code ##################

        # Getting the type of 'self' (line 214)
        self_3958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'stypy_return_type', self_3958)
        
        # ################# End of 'change_base_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_base_types' in the type store
        # Getting the type of 'stypy_return_type' (line 213)
        stypy_return_type_3959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3959)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_base_types'
        return stypy_return_type_3959


    @norecursion
    def add_base_types(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_base_types'
        module_type_store = module_type_store.open_function_context('add_base_types', 216, 4, False)
        # Assigning a type to the variable 'self' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.add_base_types.__dict__.__setitem__('stypy_localization', localization)
        TypeError.add_base_types.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.add_base_types.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.add_base_types.__dict__.__setitem__('stypy_function_name', 'TypeError.add_base_types')
        TypeError.add_base_types.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_types'])
        TypeError.add_base_types.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.add_base_types.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.add_base_types.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.add_base_types.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.add_base_types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.add_base_types.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.add_base_types', ['localization', 'new_types'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_base_types', localization, ['localization', 'new_types'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_base_types(...)' code ##################

        # Getting the type of 'self' (line 217)
        self_3960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'stypy_return_type', self_3960)
        
        # ################# End of 'add_base_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_base_types' in the type store
        # Getting the type of 'stypy_return_type' (line 216)
        stypy_return_type_3961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3961)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_base_types'
        return stypy_return_type_3961


    @norecursion
    def clone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clone'
        module_type_store = module_type_store.open_function_context('clone', 221, 4, False)
        # Assigning a type to the variable 'self' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeError.clone.__dict__.__setitem__('stypy_localization', localization)
        TypeError.clone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeError.clone.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeError.clone.__dict__.__setitem__('stypy_function_name', 'TypeError.clone')
        TypeError.clone.__dict__.__setitem__('stypy_param_names_list', [])
        TypeError.clone.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeError.clone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeError.clone.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeError.clone.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeError.clone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeError.clone.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeError.clone', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clone', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clone(...)' code ##################

        # Getting the type of 'self' (line 222)
        self_3962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 222)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'stypy_return_type', self_3962)
        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 221)
        stypy_return_type_3963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3963)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_3963


# Assigning a type to the variable 'TypeError' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'TypeError', TypeError)

# Assigning a List to a Name (line 17):

# Obtaining an instance of the builtin type 'list' (line 17)
list_3964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)

# Getting the type of 'TypeError'
TypeError_3965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeError')
# Setting the type of the member 'errors' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeError_3965, 'errors', list_3964)

# Assigning a Name to a Name (line 21):
# Getting the type of 'False' (line 21)
False_3966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 35), 'False')
# Getting the type of 'TypeError'
TypeError_3967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeError')
# Setting the type of the member 'usage_of_unsupported_feature' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeError_3967, 'usage_of_unsupported_feature', False_3966)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
