
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import copy
2: from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type
3: from stypy_copy.reporting_copy import print_utils_copy
4: from stypy_copy.errors_copy.type_error_copy import TypeError
5: 
6: class FunctionContext:
7:     '''
8:     Models a function/method local context, containing all its variables and types in a dictionary. A type store holds
9:     a stack of function contexts, one per called function, tracking all its local context. This class also have the
10:     optional feature of annotating types to create type-annotated programs, allowing the type annotation inside
11:     functions code
12:     '''
13:     annotate_types = True
14: 
15:     def __init__(self, function_name, is_main_context=False):
16:         '''
17:         Initializes the function context for function function_name
18:         :param function_name: Name of the function
19:         :param is_main_context: Whether it is the main context or not. There can be only a function context in the
20:         program.
21:         :return:
22:         '''
23: 
24:         # Types of local variables/parameters (name: type)
25:         self.types_of = {}
26: 
27:         # Function name
28:         self.function_name = function_name
29: 
30:         # Global variables applicable to the function
31:         self.global_vars = []
32: 
33:         # Aliases of variables aplicable to the function
34:         self.aliases = dict()
35: 
36:         self.is_main_context = is_main_context
37: 
38:         # Context information
39:         # Declared named argument list
40:         self.declared_argument_name_list = None
41: 
42:         # Declared varargs variable name (if any)
43:         self.declared_varargs_var = None
44: 
45:         # Declared keyword arguments variable name (if any)
46:         self.declared_kwargs_var = None
47: 
48:         # Declared defaults for parameters (if any)
49:         self.declared_defaults = None
50: 
51:         # Position of the function inside the source code
52:         self.declaration_line = -1
53:         self.declaration_column = -1
54: 
55:         # Return type of the function
56:         self.return_type = None
57: 
58:     def get_header_str(self):
59:         '''
60:         Obtains an appropriate str to pretty-print the function context, formatting the header of the represented
61:         function.
62:         :return: str
63:         '''
64:         txt = ""
65:         arg_str = ""
66:         if self.declared_argument_name_list is not None:
67:             for arg in self.declared_argument_name_list:
68:                 arg_str += str(arg) + ": " + str(self.get_type_of(arg)) + ", "
69: 
70:             if arg_str is not "":
71:                 arg_str = arg_str[:-2]
72: 
73:         if self.declared_varargs_var is not None:
74:             if arg_str is not "":
75:                 arg_str += ", "
76:             str_varargs = "*" + str(self.declared_varargs_var) + ": " + str(self.get_type_of(self.declared_varargs_var))
77: 
78:             arg_str += str_varargs
79: 
80:         if self.declared_kwargs_var is not None:
81:             if arg_str is not "":
82:                 arg_str += ", "
83:             str_kwargs = "**"+str(self.declared_kwargs_var) + ": " + str(self.get_type_of(self.declared_kwargs_var))
84: 
85:             arg_str += str_kwargs
86: 
87:         txt += str(self.function_name) + "(" + arg_str + ") -> " + print_utils.get_type_str(self.return_type)
88: 
89:         return txt
90: 
91:     def __repr__(self):
92:         '''
93:         String representation of the function context
94:         :return: str
95:         '''
96:         txt = ""
97:         if self.is_main_context:
98:             txt += "Program '" + str(self.function_name) + "'\n"
99:         else:
100:             if self.declaration_line is not -1:
101:                 txt = self.get_header_str()
102:                 txt += " (Line: " + str(self.declaration_line) + ", Column: " + str(self.declaration_column) + ")\n"
103: 
104:         for name in self.types_of:
105:             type_ = self.types_of[name]
106:             if isinstance(type_, TypeError):
107:                 txt += "\t" + name + ": TypeError\n"
108:             else:
109:                 txt += "\t" + name + ": " + str(type_) + "\n"
110: 
111:         return txt
112: 
113:     def __str__(self):
114:         '''
115:         String representation of the function context
116:         :return: str
117:         '''
118:         return self.__repr__()
119: 
120:     def __contains__(self, item):
121:         '''
122:         in operator, to determine if the function context contains a local variable
123:         :param item:
124:         :return:
125:         '''
126:         return item in self.types_of.keys()
127: 
128:     def add_alias(self, alias_name, variable_name):
129:         '''
130:         Adds an alias to the alias storage of this function context
131:         :param alias_name: Name of the alias
132:         :param variable_name: Name of the aliased variable
133:         :return:
134:         '''
135:         self.aliases[alias_name] = variable_name
136: 
137:     def get_type_of(self, variable_name):
138:         '''
139:         Returns the type of a variable or parameter in the local context
140:         :param variable_name: Name of the variable in the context
141:         :return: The variable type or None if the variable do not belong to this context locally
142:         '''
143:         if variable_name in self.aliases.keys():
144:             variable_name = self.aliases[variable_name]
145: 
146:         if variable_name in self.types_of:
147:             return self.types_of[variable_name]
148: 
149:         return None
150: 
151:     def set_type_of(self, name, type_, localization):
152:         '''
153:         Sets the type of name to type in this local context
154:         :param name: Name to search
155:         :param type: Type to assign to name
156:         '''
157:         if self.annotate_types:
158:             self.annotation_record.annotate_type(localization.line, localization.column, name, type_)
159: 
160:         if name in self.aliases.keys():
161:             name = self.aliases[name]
162:         self.types_of[name] = type_
163: 
164:     def del_type_of(self, variable_name):
165:         '''
166:         Deletes the type of a variable or parameter in the local context
167:         :param variable_name: Name of the variable in the context
168:         '''
169:         if variable_name in self.types_of:
170:             del self.types_of[variable_name]
171: 
172:         return None
173: 
174:     def __iter__(self):
175:         '''
176:         Allows iteration through all the variable names stored in the context.
177:         :return: Each variable name stored in the context
178:         '''
179:         for variable_name in self.types_of:
180:             yield variable_name
181: 
182:     def __getitem__(self, item):
183:         '''
184:         Allows the usage of the [] operator to access variable types by variable name
185:         :param item: Variable name
186:         :return: Same as get_type_of
187:         '''
188:         return self.get_type_of(item)
189: 
190:     def __len__(self):
191:         '''
192:         len operator, returning the amount of stored local variables
193:         :return:
194:         '''
195:         return len(self.types_of)
196: 
197:     def clone(self):
198:         '''
199:         Clones the whole function context. The returned function context is a deep copy of the current one
200:         :return: Cloned function context
201:         '''
202:         cloned_obj = FunctionContext(self.function_name)
203: 
204:         cloned_obj.global_vars = copy.deepcopy(self.global_vars)
205: 
206:         for key, value in self.types_of.iteritems():
207:             if isinstance(value, Type):
208:                 new_obj = value.clone()
209:             else:
210:                 new_obj = copy.deepcopy(value)
211: 
212:             cloned_obj.types_of[key] = new_obj
213: 
214:         cloned_obj.aliases = copy.deepcopy(self.aliases)
215:         cloned_obj.annotation_record = self.annotation_record
216:         cloned_obj.is_main_context = self.is_main_context
217: 
218:         # Context information
219:         cloned_obj.declared_argument_name_list = self.declared_argument_name_list
220:         cloned_obj.declared_varargs_var = self.declared_varargs_var
221:         cloned_obj.declared_kwargs_var = self.declared_kwargs_var
222:         cloned_obj.declared_defaults = self.declared_defaults
223: 
224:         cloned_obj.declaration_line = self.declaration_line
225:         cloned_obj.declaration_column = self.declaration_column
226: 
227:         cloned_obj.return_type = self.return_type
228: 
229:         return cloned_obj
230: 
231:     def copy(self):
232:         '''
233:         Copies this function context into a newly created one and return it. The copied function context is a shallow
234:         copy.
235:         :return: Copy of this function context
236:         '''
237:         copied_obj = FunctionContext(self.function_name)
238: 
239:         copied_obj.global_vars = self.global_vars
240:         copied_obj.types_of = self.types_of
241: 
242:         copied_obj.aliases = self.aliases
243:         copied_obj.annotation_record = self.annotation_record
244:         copied_obj.is_main_context = self.is_main_context
245: 
246:         # Context information
247:         copied_obj.declared_argument_name_list = self.declared_argument_name_list
248:         copied_obj.declared_varargs_var = self.declared_varargs_var
249:         copied_obj.declared_kwargs_var = self.declared_kwargs_var
250:         copied_obj.declared_defaults = self.declared_defaults
251: 
252:         copied_obj.declaration_line = self.declaration_line
253:         copied_obj.declaration_column = self.declaration_column
254: 
255:         copied_obj.return_type = self.return_type
256: 
257:         return copied_obj
258: 
259: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import copy' statement (line 1)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/type_store_copy/')
import_12021 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy')

if (type(import_12021) is not StypyTypeError):

    if (import_12021 != 'pyd_module'):
        __import__(import_12021)
        sys_modules_12022 = sys.modules[import_12021]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy', sys_modules_12022.module_type_store, module_type_store, ['Type'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_12022, sys_modules_12022.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy', None, module_type_store, ['Type'], [Type])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy', import_12021)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/type_store_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from stypy_copy.reporting_copy import print_utils_copy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/type_store_copy/')
import_12023 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.reporting_copy')

if (type(import_12023) is not StypyTypeError):

    if (import_12023 != 'pyd_module'):
        __import__(import_12023)
        sys_modules_12024 = sys.modules[import_12023]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.reporting_copy', sys_modules_12024.module_type_store, module_type_store, ['print_utils_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_12024, sys_modules_12024.module_type_store, module_type_store)
    else:
        from stypy_copy.reporting_copy import print_utils_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.reporting_copy', None, module_type_store, ['print_utils_copy'], [print_utils_copy])

else:
    # Assigning a type to the variable 'stypy_copy.reporting_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.reporting_copy', import_12023)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/type_store_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/type_store_copy/')
import_12025 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.errors_copy.type_error_copy')

if (type(import_12025) is not StypyTypeError):

    if (import_12025 != 'pyd_module'):
        __import__(import_12025)
        sys_modules_12026 = sys.modules[import_12025]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.errors_copy.type_error_copy', sys_modules_12026.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_12026, sys_modules_12026.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.type_error_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.errors_copy.type_error_copy', import_12025)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/type_store_copy/')

# Declaration of the 'FunctionContext' class

class FunctionContext:
    str_12027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, (-1)), 'str', '\n    Models a function/method local context, containing all its variables and types in a dictionary. A type store holds\n    a stack of function contexts, one per called function, tracking all its local context. This class also have the\n    optional feature of annotating types to create type-annotated programs, allowing the type annotation inside\n    functions code\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 15)
        False_12028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 54), 'False')
        defaults = [False_12028]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionContext.__init__', ['function_name', 'is_main_context'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['function_name', 'is_main_context'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_12029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', '\n        Initializes the function context for function function_name\n        :param function_name: Name of the function\n        :param is_main_context: Whether it is the main context or not. There can be only a function context in the\n        program.\n        :return:\n        ')
        
        # Assigning a Dict to a Attribute (line 25):
        
        # Obtaining an instance of the builtin type 'dict' (line 25)
        dict_12030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 25)
        
        # Getting the type of 'self' (line 25)
        self_12031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member 'types_of' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_12031, 'types_of', dict_12030)
        
        # Assigning a Name to a Attribute (line 28):
        # Getting the type of 'function_name' (line 28)
        function_name_12032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'function_name')
        # Getting the type of 'self' (line 28)
        self_12033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'function_name' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_12033, 'function_name', function_name_12032)
        
        # Assigning a List to a Attribute (line 31):
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_12034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        
        # Getting the type of 'self' (line 31)
        self_12035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'global_vars' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_12035, 'global_vars', list_12034)
        
        # Assigning a Call to a Attribute (line 34):
        
        # Call to dict(...): (line 34)
        # Processing the call keyword arguments (line 34)
        kwargs_12037 = {}
        # Getting the type of 'dict' (line 34)
        dict_12036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'dict', False)
        # Calling dict(args, kwargs) (line 34)
        dict_call_result_12038 = invoke(stypy.reporting.localization.Localization(__file__, 34, 23), dict_12036, *[], **kwargs_12037)
        
        # Getting the type of 'self' (line 34)
        self_12039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'aliases' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_12039, 'aliases', dict_call_result_12038)
        
        # Assigning a Name to a Attribute (line 36):
        # Getting the type of 'is_main_context' (line 36)
        is_main_context_12040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'is_main_context')
        # Getting the type of 'self' (line 36)
        self_12041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self')
        # Setting the type of the member 'is_main_context' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_12041, 'is_main_context', is_main_context_12040)
        
        # Assigning a Name to a Attribute (line 40):
        # Getting the type of 'None' (line 40)
        None_12042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 43), 'None')
        # Getting the type of 'self' (line 40)
        self_12043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self')
        # Setting the type of the member 'declared_argument_name_list' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_12043, 'declared_argument_name_list', None_12042)
        
        # Assigning a Name to a Attribute (line 43):
        # Getting the type of 'None' (line 43)
        None_12044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 36), 'None')
        # Getting the type of 'self' (line 43)
        self_12045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member 'declared_varargs_var' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_12045, 'declared_varargs_var', None_12044)
        
        # Assigning a Name to a Attribute (line 46):
        # Getting the type of 'None' (line 46)
        None_12046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 35), 'None')
        # Getting the type of 'self' (line 46)
        self_12047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self')
        # Setting the type of the member 'declared_kwargs_var' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_12047, 'declared_kwargs_var', None_12046)
        
        # Assigning a Name to a Attribute (line 49):
        # Getting the type of 'None' (line 49)
        None_12048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 33), 'None')
        # Getting the type of 'self' (line 49)
        self_12049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'self')
        # Setting the type of the member 'declared_defaults' of a type (line 49)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), self_12049, 'declared_defaults', None_12048)
        
        # Assigning a Num to a Attribute (line 52):
        int_12050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 32), 'int')
        # Getting the type of 'self' (line 52)
        self_12051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self')
        # Setting the type of the member 'declaration_line' of a type (line 52)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_12051, 'declaration_line', int_12050)
        
        # Assigning a Num to a Attribute (line 53):
        int_12052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 34), 'int')
        # Getting the type of 'self' (line 53)
        self_12053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self')
        # Setting the type of the member 'declaration_column' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_12053, 'declaration_column', int_12052)
        
        # Assigning a Name to a Attribute (line 56):
        # Getting the type of 'None' (line 56)
        None_12054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 27), 'None')
        # Getting the type of 'self' (line 56)
        self_12055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member 'return_type' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_12055, 'return_type', None_12054)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_header_str(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_header_str'
        module_type_store = module_type_store.open_function_context('get_header_str', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionContext.get_header_str.__dict__.__setitem__('stypy_localization', localization)
        FunctionContext.get_header_str.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionContext.get_header_str.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionContext.get_header_str.__dict__.__setitem__('stypy_function_name', 'FunctionContext.get_header_str')
        FunctionContext.get_header_str.__dict__.__setitem__('stypy_param_names_list', [])
        FunctionContext.get_header_str.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionContext.get_header_str.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionContext.get_header_str.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionContext.get_header_str.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionContext.get_header_str.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionContext.get_header_str.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionContext.get_header_str', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_header_str', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_header_str(...)' code ##################

        str_12056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, (-1)), 'str', '\n        Obtains an appropriate str to pretty-print the function context, formatting the header of the represented\n        function.\n        :return: str\n        ')
        
        # Assigning a Str to a Name (line 64):
        str_12057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 14), 'str', '')
        # Assigning a type to the variable 'txt' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'txt', str_12057)
        
        # Assigning a Str to a Name (line 65):
        str_12058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 18), 'str', '')
        # Assigning a type to the variable 'arg_str' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'arg_str', str_12058)
        
        # Getting the type of 'self' (line 66)
        self_12059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'self')
        # Obtaining the member 'declared_argument_name_list' of a type (line 66)
        declared_argument_name_list_12060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 11), self_12059, 'declared_argument_name_list')
        # Getting the type of 'None' (line 66)
        None_12061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 51), 'None')
        # Applying the binary operator 'isnot' (line 66)
        result_is_not_12062 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 11), 'isnot', declared_argument_name_list_12060, None_12061)
        
        # Testing if the type of an if condition is none (line 66)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 66, 8), result_is_not_12062):
            pass
        else:
            
            # Testing the type of an if condition (line 66)
            if_condition_12063 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 8), result_is_not_12062)
            # Assigning a type to the variable 'if_condition_12063' (line 66)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'if_condition_12063', if_condition_12063)
            # SSA begins for if statement (line 66)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'self' (line 67)
            self_12064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'self')
            # Obtaining the member 'declared_argument_name_list' of a type (line 67)
            declared_argument_name_list_12065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 23), self_12064, 'declared_argument_name_list')
            # Assigning a type to the variable 'declared_argument_name_list_12065' (line 67)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'declared_argument_name_list_12065', declared_argument_name_list_12065)
            # Testing if the for loop is going to be iterated (line 67)
            # Testing the type of a for loop iterable (line 67)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 67, 12), declared_argument_name_list_12065)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 67, 12), declared_argument_name_list_12065):
                # Getting the type of the for loop variable (line 67)
                for_loop_var_12066 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 67, 12), declared_argument_name_list_12065)
                # Assigning a type to the variable 'arg' (line 67)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'arg', for_loop_var_12066)
                # SSA begins for a for statement (line 67)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'arg_str' (line 68)
                arg_str_12067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'arg_str')
                
                # Call to str(...): (line 68)
                # Processing the call arguments (line 68)
                # Getting the type of 'arg' (line 68)
                arg_12069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 31), 'arg', False)
                # Processing the call keyword arguments (line 68)
                kwargs_12070 = {}
                # Getting the type of 'str' (line 68)
                str_12068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 27), 'str', False)
                # Calling str(args, kwargs) (line 68)
                str_call_result_12071 = invoke(stypy.reporting.localization.Localization(__file__, 68, 27), str_12068, *[arg_12069], **kwargs_12070)
                
                str_12072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 38), 'str', ': ')
                # Applying the binary operator '+' (line 68)
                result_add_12073 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 27), '+', str_call_result_12071, str_12072)
                
                
                # Call to str(...): (line 68)
                # Processing the call arguments (line 68)
                
                # Call to get_type_of(...): (line 68)
                # Processing the call arguments (line 68)
                # Getting the type of 'arg' (line 68)
                arg_12077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 66), 'arg', False)
                # Processing the call keyword arguments (line 68)
                kwargs_12078 = {}
                # Getting the type of 'self' (line 68)
                self_12075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 49), 'self', False)
                # Obtaining the member 'get_type_of' of a type (line 68)
                get_type_of_12076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 49), self_12075, 'get_type_of')
                # Calling get_type_of(args, kwargs) (line 68)
                get_type_of_call_result_12079 = invoke(stypy.reporting.localization.Localization(__file__, 68, 49), get_type_of_12076, *[arg_12077], **kwargs_12078)
                
                # Processing the call keyword arguments (line 68)
                kwargs_12080 = {}
                # Getting the type of 'str' (line 68)
                str_12074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 45), 'str', False)
                # Calling str(args, kwargs) (line 68)
                str_call_result_12081 = invoke(stypy.reporting.localization.Localization(__file__, 68, 45), str_12074, *[get_type_of_call_result_12079], **kwargs_12080)
                
                # Applying the binary operator '+' (line 68)
                result_add_12082 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 43), '+', result_add_12073, str_call_result_12081)
                
                str_12083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 74), 'str', ', ')
                # Applying the binary operator '+' (line 68)
                result_add_12084 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 72), '+', result_add_12082, str_12083)
                
                # Applying the binary operator '+=' (line 68)
                result_iadd_12085 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 16), '+=', arg_str_12067, result_add_12084)
                # Assigning a type to the variable 'arg_str' (line 68)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'arg_str', result_iadd_12085)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Getting the type of 'arg_str' (line 70)
            arg_str_12086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'arg_str')
            str_12087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 30), 'str', '')
            # Applying the binary operator 'isnot' (line 70)
            result_is_not_12088 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 15), 'isnot', arg_str_12086, str_12087)
            
            # Testing if the type of an if condition is none (line 70)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 12), result_is_not_12088):
                pass
            else:
                
                # Testing the type of an if condition (line 70)
                if_condition_12089 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 12), result_is_not_12088)
                # Assigning a type to the variable 'if_condition_12089' (line 70)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'if_condition_12089', if_condition_12089)
                # SSA begins for if statement (line 70)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Name (line 71):
                
                # Obtaining the type of the subscript
                int_12090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 35), 'int')
                slice_12091 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 71, 26), None, int_12090, None)
                # Getting the type of 'arg_str' (line 71)
                arg_str_12092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 26), 'arg_str')
                # Obtaining the member '__getitem__' of a type (line 71)
                getitem___12093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 26), arg_str_12092, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 71)
                subscript_call_result_12094 = invoke(stypy.reporting.localization.Localization(__file__, 71, 26), getitem___12093, slice_12091)
                
                # Assigning a type to the variable 'arg_str' (line 71)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'arg_str', subscript_call_result_12094)
                # SSA join for if statement (line 70)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 66)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 73)
        self_12095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'self')
        # Obtaining the member 'declared_varargs_var' of a type (line 73)
        declared_varargs_var_12096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 11), self_12095, 'declared_varargs_var')
        # Getting the type of 'None' (line 73)
        None_12097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'None')
        # Applying the binary operator 'isnot' (line 73)
        result_is_not_12098 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 11), 'isnot', declared_varargs_var_12096, None_12097)
        
        # Testing if the type of an if condition is none (line 73)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 73, 8), result_is_not_12098):
            pass
        else:
            
            # Testing the type of an if condition (line 73)
            if_condition_12099 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 8), result_is_not_12098)
            # Assigning a type to the variable 'if_condition_12099' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'if_condition_12099', if_condition_12099)
            # SSA begins for if statement (line 73)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'arg_str' (line 74)
            arg_str_12100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'arg_str')
            str_12101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 30), 'str', '')
            # Applying the binary operator 'isnot' (line 74)
            result_is_not_12102 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 15), 'isnot', arg_str_12100, str_12101)
            
            # Testing if the type of an if condition is none (line 74)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 12), result_is_not_12102):
                pass
            else:
                
                # Testing the type of an if condition (line 74)
                if_condition_12103 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 12), result_is_not_12102)
                # Assigning a type to the variable 'if_condition_12103' (line 74)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'if_condition_12103', if_condition_12103)
                # SSA begins for if statement (line 74)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'arg_str' (line 75)
                arg_str_12104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'arg_str')
                str_12105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 27), 'str', ', ')
                # Applying the binary operator '+=' (line 75)
                result_iadd_12106 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 16), '+=', arg_str_12104, str_12105)
                # Assigning a type to the variable 'arg_str' (line 75)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'arg_str', result_iadd_12106)
                
                # SSA join for if statement (line 74)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a BinOp to a Name (line 76):
            str_12107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 26), 'str', '*')
            
            # Call to str(...): (line 76)
            # Processing the call arguments (line 76)
            # Getting the type of 'self' (line 76)
            self_12109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 36), 'self', False)
            # Obtaining the member 'declared_varargs_var' of a type (line 76)
            declared_varargs_var_12110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 36), self_12109, 'declared_varargs_var')
            # Processing the call keyword arguments (line 76)
            kwargs_12111 = {}
            # Getting the type of 'str' (line 76)
            str_12108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 32), 'str', False)
            # Calling str(args, kwargs) (line 76)
            str_call_result_12112 = invoke(stypy.reporting.localization.Localization(__file__, 76, 32), str_12108, *[declared_varargs_var_12110], **kwargs_12111)
            
            # Applying the binary operator '+' (line 76)
            result_add_12113 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 26), '+', str_12107, str_call_result_12112)
            
            str_12114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 65), 'str', ': ')
            # Applying the binary operator '+' (line 76)
            result_add_12115 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 63), '+', result_add_12113, str_12114)
            
            
            # Call to str(...): (line 76)
            # Processing the call arguments (line 76)
            
            # Call to get_type_of(...): (line 76)
            # Processing the call arguments (line 76)
            # Getting the type of 'self' (line 76)
            self_12119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 93), 'self', False)
            # Obtaining the member 'declared_varargs_var' of a type (line 76)
            declared_varargs_var_12120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 93), self_12119, 'declared_varargs_var')
            # Processing the call keyword arguments (line 76)
            kwargs_12121 = {}
            # Getting the type of 'self' (line 76)
            self_12117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 76), 'self', False)
            # Obtaining the member 'get_type_of' of a type (line 76)
            get_type_of_12118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 76), self_12117, 'get_type_of')
            # Calling get_type_of(args, kwargs) (line 76)
            get_type_of_call_result_12122 = invoke(stypy.reporting.localization.Localization(__file__, 76, 76), get_type_of_12118, *[declared_varargs_var_12120], **kwargs_12121)
            
            # Processing the call keyword arguments (line 76)
            kwargs_12123 = {}
            # Getting the type of 'str' (line 76)
            str_12116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 72), 'str', False)
            # Calling str(args, kwargs) (line 76)
            str_call_result_12124 = invoke(stypy.reporting.localization.Localization(__file__, 76, 72), str_12116, *[get_type_of_call_result_12122], **kwargs_12123)
            
            # Applying the binary operator '+' (line 76)
            result_add_12125 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 70), '+', result_add_12115, str_call_result_12124)
            
            # Assigning a type to the variable 'str_varargs' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'str_varargs', result_add_12125)
            
            # Getting the type of 'arg_str' (line 78)
            arg_str_12126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'arg_str')
            # Getting the type of 'str_varargs' (line 78)
            str_varargs_12127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'str_varargs')
            # Applying the binary operator '+=' (line 78)
            result_iadd_12128 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 12), '+=', arg_str_12126, str_varargs_12127)
            # Assigning a type to the variable 'arg_str' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'arg_str', result_iadd_12128)
            
            # SSA join for if statement (line 73)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 80)
        self_12129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'self')
        # Obtaining the member 'declared_kwargs_var' of a type (line 80)
        declared_kwargs_var_12130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 11), self_12129, 'declared_kwargs_var')
        # Getting the type of 'None' (line 80)
        None_12131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 43), 'None')
        # Applying the binary operator 'isnot' (line 80)
        result_is_not_12132 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 11), 'isnot', declared_kwargs_var_12130, None_12131)
        
        # Testing if the type of an if condition is none (line 80)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 8), result_is_not_12132):
            pass
        else:
            
            # Testing the type of an if condition (line 80)
            if_condition_12133 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 8), result_is_not_12132)
            # Assigning a type to the variable 'if_condition_12133' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'if_condition_12133', if_condition_12133)
            # SSA begins for if statement (line 80)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'arg_str' (line 81)
            arg_str_12134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'arg_str')
            str_12135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 30), 'str', '')
            # Applying the binary operator 'isnot' (line 81)
            result_is_not_12136 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 15), 'isnot', arg_str_12134, str_12135)
            
            # Testing if the type of an if condition is none (line 81)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 81, 12), result_is_not_12136):
                pass
            else:
                
                # Testing the type of an if condition (line 81)
                if_condition_12137 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 12), result_is_not_12136)
                # Assigning a type to the variable 'if_condition_12137' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'if_condition_12137', if_condition_12137)
                # SSA begins for if statement (line 81)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'arg_str' (line 82)
                arg_str_12138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'arg_str')
                str_12139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 27), 'str', ', ')
                # Applying the binary operator '+=' (line 82)
                result_iadd_12140 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 16), '+=', arg_str_12138, str_12139)
                # Assigning a type to the variable 'arg_str' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'arg_str', result_iadd_12140)
                
                # SSA join for if statement (line 81)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a BinOp to a Name (line 83):
            str_12141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 25), 'str', '**')
            
            # Call to str(...): (line 83)
            # Processing the call arguments (line 83)
            # Getting the type of 'self' (line 83)
            self_12143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 34), 'self', False)
            # Obtaining the member 'declared_kwargs_var' of a type (line 83)
            declared_kwargs_var_12144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 34), self_12143, 'declared_kwargs_var')
            # Processing the call keyword arguments (line 83)
            kwargs_12145 = {}
            # Getting the type of 'str' (line 83)
            str_12142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'str', False)
            # Calling str(args, kwargs) (line 83)
            str_call_result_12146 = invoke(stypy.reporting.localization.Localization(__file__, 83, 30), str_12142, *[declared_kwargs_var_12144], **kwargs_12145)
            
            # Applying the binary operator '+' (line 83)
            result_add_12147 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 25), '+', str_12141, str_call_result_12146)
            
            str_12148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 62), 'str', ': ')
            # Applying the binary operator '+' (line 83)
            result_add_12149 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 60), '+', result_add_12147, str_12148)
            
            
            # Call to str(...): (line 83)
            # Processing the call arguments (line 83)
            
            # Call to get_type_of(...): (line 83)
            # Processing the call arguments (line 83)
            # Getting the type of 'self' (line 83)
            self_12153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 90), 'self', False)
            # Obtaining the member 'declared_kwargs_var' of a type (line 83)
            declared_kwargs_var_12154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 90), self_12153, 'declared_kwargs_var')
            # Processing the call keyword arguments (line 83)
            kwargs_12155 = {}
            # Getting the type of 'self' (line 83)
            self_12151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 73), 'self', False)
            # Obtaining the member 'get_type_of' of a type (line 83)
            get_type_of_12152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 73), self_12151, 'get_type_of')
            # Calling get_type_of(args, kwargs) (line 83)
            get_type_of_call_result_12156 = invoke(stypy.reporting.localization.Localization(__file__, 83, 73), get_type_of_12152, *[declared_kwargs_var_12154], **kwargs_12155)
            
            # Processing the call keyword arguments (line 83)
            kwargs_12157 = {}
            # Getting the type of 'str' (line 83)
            str_12150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 69), 'str', False)
            # Calling str(args, kwargs) (line 83)
            str_call_result_12158 = invoke(stypy.reporting.localization.Localization(__file__, 83, 69), str_12150, *[get_type_of_call_result_12156], **kwargs_12157)
            
            # Applying the binary operator '+' (line 83)
            result_add_12159 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 67), '+', result_add_12149, str_call_result_12158)
            
            # Assigning a type to the variable 'str_kwargs' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'str_kwargs', result_add_12159)
            
            # Getting the type of 'arg_str' (line 85)
            arg_str_12160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'arg_str')
            # Getting the type of 'str_kwargs' (line 85)
            str_kwargs_12161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'str_kwargs')
            # Applying the binary operator '+=' (line 85)
            result_iadd_12162 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 12), '+=', arg_str_12160, str_kwargs_12161)
            # Assigning a type to the variable 'arg_str' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'arg_str', result_iadd_12162)
            
            # SSA join for if statement (line 80)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'txt' (line 87)
        txt_12163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'txt')
        
        # Call to str(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'self' (line 87)
        self_12165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'self', False)
        # Obtaining the member 'function_name' of a type (line 87)
        function_name_12166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 19), self_12165, 'function_name')
        # Processing the call keyword arguments (line 87)
        kwargs_12167 = {}
        # Getting the type of 'str' (line 87)
        str_12164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'str', False)
        # Calling str(args, kwargs) (line 87)
        str_call_result_12168 = invoke(stypy.reporting.localization.Localization(__file__, 87, 15), str_12164, *[function_name_12166], **kwargs_12167)
        
        str_12169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 41), 'str', '(')
        # Applying the binary operator '+' (line 87)
        result_add_12170 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 15), '+', str_call_result_12168, str_12169)
        
        # Getting the type of 'arg_str' (line 87)
        arg_str_12171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 47), 'arg_str')
        # Applying the binary operator '+' (line 87)
        result_add_12172 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 45), '+', result_add_12170, arg_str_12171)
        
        str_12173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 57), 'str', ') -> ')
        # Applying the binary operator '+' (line 87)
        result_add_12174 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 55), '+', result_add_12172, str_12173)
        
        
        # Call to get_type_str(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'self' (line 87)
        self_12177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 92), 'self', False)
        # Obtaining the member 'return_type' of a type (line 87)
        return_type_12178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 92), self_12177, 'return_type')
        # Processing the call keyword arguments (line 87)
        kwargs_12179 = {}
        # Getting the type of 'print_utils' (line 87)
        print_utils_12175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 67), 'print_utils', False)
        # Obtaining the member 'get_type_str' of a type (line 87)
        get_type_str_12176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 67), print_utils_12175, 'get_type_str')
        # Calling get_type_str(args, kwargs) (line 87)
        get_type_str_call_result_12180 = invoke(stypy.reporting.localization.Localization(__file__, 87, 67), get_type_str_12176, *[return_type_12178], **kwargs_12179)
        
        # Applying the binary operator '+' (line 87)
        result_add_12181 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 65), '+', result_add_12174, get_type_str_call_result_12180)
        
        # Applying the binary operator '+=' (line 87)
        result_iadd_12182 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 8), '+=', txt_12163, result_add_12181)
        # Assigning a type to the variable 'txt' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'txt', result_iadd_12182)
        
        # Getting the type of 'txt' (line 89)
        txt_12183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'txt')
        # Assigning a type to the variable 'stypy_return_type' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'stypy_return_type', txt_12183)
        
        # ################# End of 'get_header_str(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_header_str' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_12184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12184)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_header_str'
        return stypy_return_type_12184


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 91, 4, False)
        # Assigning a type to the variable 'self' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionContext.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        FunctionContext.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionContext.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionContext.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'FunctionContext.stypy__repr__')
        FunctionContext.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        FunctionContext.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionContext.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionContext.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionContext.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionContext.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionContext.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionContext.stypy__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        str_12185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'str', '\n        String representation of the function context\n        :return: str\n        ')
        
        # Assigning a Str to a Name (line 96):
        str_12186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 14), 'str', '')
        # Assigning a type to the variable 'txt' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'txt', str_12186)
        # Getting the type of 'self' (line 97)
        self_12187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'self')
        # Obtaining the member 'is_main_context' of a type (line 97)
        is_main_context_12188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 11), self_12187, 'is_main_context')
        # Testing if the type of an if condition is none (line 97)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 97, 8), is_main_context_12188):
            
            # Getting the type of 'self' (line 100)
            self_12201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'self')
            # Obtaining the member 'declaration_line' of a type (line 100)
            declaration_line_12202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), self_12201, 'declaration_line')
            int_12203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 44), 'int')
            # Applying the binary operator 'isnot' (line 100)
            result_is_not_12204 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), 'isnot', declaration_line_12202, int_12203)
            
            # Testing if the type of an if condition is none (line 100)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 100, 12), result_is_not_12204):
                pass
            else:
                
                # Testing the type of an if condition (line 100)
                if_condition_12205 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 12), result_is_not_12204)
                # Assigning a type to the variable 'if_condition_12205' (line 100)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'if_condition_12205', if_condition_12205)
                # SSA begins for if statement (line 100)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 101):
                
                # Call to get_header_str(...): (line 101)
                # Processing the call keyword arguments (line 101)
                kwargs_12208 = {}
                # Getting the type of 'self' (line 101)
                self_12206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'self', False)
                # Obtaining the member 'get_header_str' of a type (line 101)
                get_header_str_12207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 22), self_12206, 'get_header_str')
                # Calling get_header_str(args, kwargs) (line 101)
                get_header_str_call_result_12209 = invoke(stypy.reporting.localization.Localization(__file__, 101, 22), get_header_str_12207, *[], **kwargs_12208)
                
                # Assigning a type to the variable 'txt' (line 101)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'txt', get_header_str_call_result_12209)
                
                # Getting the type of 'txt' (line 102)
                txt_12210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'txt')
                str_12211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 23), 'str', ' (Line: ')
                
                # Call to str(...): (line 102)
                # Processing the call arguments (line 102)
                # Getting the type of 'self' (line 102)
                self_12213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 40), 'self', False)
                # Obtaining the member 'declaration_line' of a type (line 102)
                declaration_line_12214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 40), self_12213, 'declaration_line')
                # Processing the call keyword arguments (line 102)
                kwargs_12215 = {}
                # Getting the type of 'str' (line 102)
                str_12212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 36), 'str', False)
                # Calling str(args, kwargs) (line 102)
                str_call_result_12216 = invoke(stypy.reporting.localization.Localization(__file__, 102, 36), str_12212, *[declaration_line_12214], **kwargs_12215)
                
                # Applying the binary operator '+' (line 102)
                result_add_12217 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 23), '+', str_12211, str_call_result_12216)
                
                str_12218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 65), 'str', ', Column: ')
                # Applying the binary operator '+' (line 102)
                result_add_12219 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 63), '+', result_add_12217, str_12218)
                
                
                # Call to str(...): (line 102)
                # Processing the call arguments (line 102)
                # Getting the type of 'self' (line 102)
                self_12221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 84), 'self', False)
                # Obtaining the member 'declaration_column' of a type (line 102)
                declaration_column_12222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 84), self_12221, 'declaration_column')
                # Processing the call keyword arguments (line 102)
                kwargs_12223 = {}
                # Getting the type of 'str' (line 102)
                str_12220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 80), 'str', False)
                # Calling str(args, kwargs) (line 102)
                str_call_result_12224 = invoke(stypy.reporting.localization.Localization(__file__, 102, 80), str_12220, *[declaration_column_12222], **kwargs_12223)
                
                # Applying the binary operator '+' (line 102)
                result_add_12225 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 78), '+', result_add_12219, str_call_result_12224)
                
                str_12226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 111), 'str', ')\n')
                # Applying the binary operator '+' (line 102)
                result_add_12227 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 109), '+', result_add_12225, str_12226)
                
                # Applying the binary operator '+=' (line 102)
                result_iadd_12228 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 16), '+=', txt_12210, result_add_12227)
                # Assigning a type to the variable 'txt' (line 102)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'txt', result_iadd_12228)
                
                # SSA join for if statement (line 100)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 97)
            if_condition_12189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), is_main_context_12188)
            # Assigning a type to the variable 'if_condition_12189' (line 97)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_12189', if_condition_12189)
            # SSA begins for if statement (line 97)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'txt' (line 98)
            txt_12190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'txt')
            str_12191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 19), 'str', "Program '")
            
            # Call to str(...): (line 98)
            # Processing the call arguments (line 98)
            # Getting the type of 'self' (line 98)
            self_12193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 37), 'self', False)
            # Obtaining the member 'function_name' of a type (line 98)
            function_name_12194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 37), self_12193, 'function_name')
            # Processing the call keyword arguments (line 98)
            kwargs_12195 = {}
            # Getting the type of 'str' (line 98)
            str_12192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'str', False)
            # Calling str(args, kwargs) (line 98)
            str_call_result_12196 = invoke(stypy.reporting.localization.Localization(__file__, 98, 33), str_12192, *[function_name_12194], **kwargs_12195)
            
            # Applying the binary operator '+' (line 98)
            result_add_12197 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 19), '+', str_12191, str_call_result_12196)
            
            str_12198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 59), 'str', "'\n")
            # Applying the binary operator '+' (line 98)
            result_add_12199 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 57), '+', result_add_12197, str_12198)
            
            # Applying the binary operator '+=' (line 98)
            result_iadd_12200 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 12), '+=', txt_12190, result_add_12199)
            # Assigning a type to the variable 'txt' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'txt', result_iadd_12200)
            
            # SSA branch for the else part of an if statement (line 97)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'self' (line 100)
            self_12201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'self')
            # Obtaining the member 'declaration_line' of a type (line 100)
            declaration_line_12202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), self_12201, 'declaration_line')
            int_12203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 44), 'int')
            # Applying the binary operator 'isnot' (line 100)
            result_is_not_12204 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), 'isnot', declaration_line_12202, int_12203)
            
            # Testing if the type of an if condition is none (line 100)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 100, 12), result_is_not_12204):
                pass
            else:
                
                # Testing the type of an if condition (line 100)
                if_condition_12205 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 12), result_is_not_12204)
                # Assigning a type to the variable 'if_condition_12205' (line 100)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'if_condition_12205', if_condition_12205)
                # SSA begins for if statement (line 100)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 101):
                
                # Call to get_header_str(...): (line 101)
                # Processing the call keyword arguments (line 101)
                kwargs_12208 = {}
                # Getting the type of 'self' (line 101)
                self_12206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'self', False)
                # Obtaining the member 'get_header_str' of a type (line 101)
                get_header_str_12207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 22), self_12206, 'get_header_str')
                # Calling get_header_str(args, kwargs) (line 101)
                get_header_str_call_result_12209 = invoke(stypy.reporting.localization.Localization(__file__, 101, 22), get_header_str_12207, *[], **kwargs_12208)
                
                # Assigning a type to the variable 'txt' (line 101)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'txt', get_header_str_call_result_12209)
                
                # Getting the type of 'txt' (line 102)
                txt_12210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'txt')
                str_12211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 23), 'str', ' (Line: ')
                
                # Call to str(...): (line 102)
                # Processing the call arguments (line 102)
                # Getting the type of 'self' (line 102)
                self_12213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 40), 'self', False)
                # Obtaining the member 'declaration_line' of a type (line 102)
                declaration_line_12214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 40), self_12213, 'declaration_line')
                # Processing the call keyword arguments (line 102)
                kwargs_12215 = {}
                # Getting the type of 'str' (line 102)
                str_12212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 36), 'str', False)
                # Calling str(args, kwargs) (line 102)
                str_call_result_12216 = invoke(stypy.reporting.localization.Localization(__file__, 102, 36), str_12212, *[declaration_line_12214], **kwargs_12215)
                
                # Applying the binary operator '+' (line 102)
                result_add_12217 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 23), '+', str_12211, str_call_result_12216)
                
                str_12218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 65), 'str', ', Column: ')
                # Applying the binary operator '+' (line 102)
                result_add_12219 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 63), '+', result_add_12217, str_12218)
                
                
                # Call to str(...): (line 102)
                # Processing the call arguments (line 102)
                # Getting the type of 'self' (line 102)
                self_12221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 84), 'self', False)
                # Obtaining the member 'declaration_column' of a type (line 102)
                declaration_column_12222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 84), self_12221, 'declaration_column')
                # Processing the call keyword arguments (line 102)
                kwargs_12223 = {}
                # Getting the type of 'str' (line 102)
                str_12220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 80), 'str', False)
                # Calling str(args, kwargs) (line 102)
                str_call_result_12224 = invoke(stypy.reporting.localization.Localization(__file__, 102, 80), str_12220, *[declaration_column_12222], **kwargs_12223)
                
                # Applying the binary operator '+' (line 102)
                result_add_12225 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 78), '+', result_add_12219, str_call_result_12224)
                
                str_12226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 111), 'str', ')\n')
                # Applying the binary operator '+' (line 102)
                result_add_12227 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 109), '+', result_add_12225, str_12226)
                
                # Applying the binary operator '+=' (line 102)
                result_iadd_12228 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 16), '+=', txt_12210, result_add_12227)
                # Assigning a type to the variable 'txt' (line 102)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'txt', result_iadd_12228)
                
                # SSA join for if statement (line 100)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 97)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 104)
        self_12229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'self')
        # Obtaining the member 'types_of' of a type (line 104)
        types_of_12230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 20), self_12229, 'types_of')
        # Assigning a type to the variable 'types_of_12230' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'types_of_12230', types_of_12230)
        # Testing if the for loop is going to be iterated (line 104)
        # Testing the type of a for loop iterable (line 104)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 104, 8), types_of_12230)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 104, 8), types_of_12230):
            # Getting the type of the for loop variable (line 104)
            for_loop_var_12231 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 104, 8), types_of_12230)
            # Assigning a type to the variable 'name' (line 104)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'name', for_loop_var_12231)
            # SSA begins for a for statement (line 104)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 105):
            
            # Obtaining the type of the subscript
            # Getting the type of 'name' (line 105)
            name_12232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 34), 'name')
            # Getting the type of 'self' (line 105)
            self_12233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 20), 'self')
            # Obtaining the member 'types_of' of a type (line 105)
            types_of_12234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 20), self_12233, 'types_of')
            # Obtaining the member '__getitem__' of a type (line 105)
            getitem___12235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 20), types_of_12234, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 105)
            subscript_call_result_12236 = invoke(stypy.reporting.localization.Localization(__file__, 105, 20), getitem___12235, name_12232)
            
            # Assigning a type to the variable 'type_' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'type_', subscript_call_result_12236)
            
            # Type idiom detected: calculating its left and rigth part (line 106)
            # Getting the type of 'TypeError' (line 106)
            TypeError_12237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'TypeError')
            # Getting the type of 'type_' (line 106)
            type__12238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'type_')
            
            (may_be_12239, more_types_in_union_12240) = may_be_subtype(TypeError_12237, type__12238)

            if may_be_12239:

                if more_types_in_union_12240:
                    # Runtime conditional SSA (line 106)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'type_' (line 106)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'type_', remove_not_subtype_from_union(type__12238, TypeError))
                
                # Getting the type of 'txt' (line 107)
                txt_12241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'txt')
                str_12242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 23), 'str', '\t')
                # Getting the type of 'name' (line 107)
                name_12243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 30), 'name')
                # Applying the binary operator '+' (line 107)
                result_add_12244 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 23), '+', str_12242, name_12243)
                
                str_12245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 37), 'str', ': TypeError\n')
                # Applying the binary operator '+' (line 107)
                result_add_12246 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 35), '+', result_add_12244, str_12245)
                
                # Applying the binary operator '+=' (line 107)
                result_iadd_12247 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 16), '+=', txt_12241, result_add_12246)
                # Assigning a type to the variable 'txt' (line 107)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'txt', result_iadd_12247)
                

                if more_types_in_union_12240:
                    # Runtime conditional SSA for else branch (line 106)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_12239) or more_types_in_union_12240):
                # Assigning a type to the variable 'type_' (line 106)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'type_', remove_subtype_from_union(type__12238, TypeError))
                
                # Getting the type of 'txt' (line 109)
                txt_12248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'txt')
                str_12249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'str', '\t')
                # Getting the type of 'name' (line 109)
                name_12250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 30), 'name')
                # Applying the binary operator '+' (line 109)
                result_add_12251 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 23), '+', str_12249, name_12250)
                
                str_12252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 37), 'str', ': ')
                # Applying the binary operator '+' (line 109)
                result_add_12253 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 35), '+', result_add_12251, str_12252)
                
                
                # Call to str(...): (line 109)
                # Processing the call arguments (line 109)
                # Getting the type of 'type_' (line 109)
                type__12255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 48), 'type_', False)
                # Processing the call keyword arguments (line 109)
                kwargs_12256 = {}
                # Getting the type of 'str' (line 109)
                str_12254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 44), 'str', False)
                # Calling str(args, kwargs) (line 109)
                str_call_result_12257 = invoke(stypy.reporting.localization.Localization(__file__, 109, 44), str_12254, *[type__12255], **kwargs_12256)
                
                # Applying the binary operator '+' (line 109)
                result_add_12258 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 42), '+', result_add_12253, str_call_result_12257)
                
                str_12259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 57), 'str', '\n')
                # Applying the binary operator '+' (line 109)
                result_add_12260 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 55), '+', result_add_12258, str_12259)
                
                # Applying the binary operator '+=' (line 109)
                result_iadd_12261 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 16), '+=', txt_12248, result_add_12260)
                # Assigning a type to the variable 'txt' (line 109)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'txt', result_iadd_12261)
                

                if (may_be_12239 and more_types_in_union_12240):
                    # SSA join for if statement (line 106)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'txt' (line 111)
        txt_12262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'txt')
        # Assigning a type to the variable 'stypy_return_type' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'stypy_return_type', txt_12262)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_12263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12263)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_12263


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 113, 4, False)
        # Assigning a type to the variable 'self' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionContext.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        FunctionContext.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionContext.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionContext.stypy__str__.__dict__.__setitem__('stypy_function_name', 'FunctionContext.stypy__str__')
        FunctionContext.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        FunctionContext.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionContext.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionContext.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionContext.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionContext.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionContext.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionContext.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        str_12264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', '\n        String representation of the function context\n        :return: str\n        ')
        
        # Call to __repr__(...): (line 118)
        # Processing the call keyword arguments (line 118)
        kwargs_12267 = {}
        # Getting the type of 'self' (line 118)
        self_12265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'self', False)
        # Obtaining the member '__repr__' of a type (line 118)
        repr___12266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 15), self_12265, '__repr__')
        # Calling __repr__(args, kwargs) (line 118)
        repr___call_result_12268 = invoke(stypy.reporting.localization.Localization(__file__, 118, 15), repr___12266, *[], **kwargs_12267)
        
        # Assigning a type to the variable 'stypy_return_type' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'stypy_return_type', repr___call_result_12268)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 113)
        stypy_return_type_12269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12269)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_12269


    @norecursion
    def __contains__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__contains__'
        module_type_store = module_type_store.open_function_context('__contains__', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionContext.__contains__.__dict__.__setitem__('stypy_localization', localization)
        FunctionContext.__contains__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionContext.__contains__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionContext.__contains__.__dict__.__setitem__('stypy_function_name', 'FunctionContext.__contains__')
        FunctionContext.__contains__.__dict__.__setitem__('stypy_param_names_list', ['item'])
        FunctionContext.__contains__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionContext.__contains__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionContext.__contains__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionContext.__contains__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionContext.__contains__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionContext.__contains__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionContext.__contains__', ['item'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__contains__', localization, ['item'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__contains__(...)' code ##################

        str_12270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, (-1)), 'str', '\n        in operator, to determine if the function context contains a local variable\n        :param item:\n        :return:\n        ')
        
        # Getting the type of 'item' (line 126)
        item_12271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'item')
        
        # Call to keys(...): (line 126)
        # Processing the call keyword arguments (line 126)
        kwargs_12275 = {}
        # Getting the type of 'self' (line 126)
        self_12272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'self', False)
        # Obtaining the member 'types_of' of a type (line 126)
        types_of_12273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), self_12272, 'types_of')
        # Obtaining the member 'keys' of a type (line 126)
        keys_12274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), types_of_12273, 'keys')
        # Calling keys(args, kwargs) (line 126)
        keys_call_result_12276 = invoke(stypy.reporting.localization.Localization(__file__, 126, 23), keys_12274, *[], **kwargs_12275)
        
        # Applying the binary operator 'in' (line 126)
        result_contains_12277 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 15), 'in', item_12271, keys_call_result_12276)
        
        # Assigning a type to the variable 'stypy_return_type' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type', result_contains_12277)
        
        # ################# End of '__contains__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__contains__' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_12278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12278)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__contains__'
        return stypy_return_type_12278


    @norecursion
    def add_alias(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_alias'
        module_type_store = module_type_store.open_function_context('add_alias', 128, 4, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionContext.add_alias.__dict__.__setitem__('stypy_localization', localization)
        FunctionContext.add_alias.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionContext.add_alias.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionContext.add_alias.__dict__.__setitem__('stypy_function_name', 'FunctionContext.add_alias')
        FunctionContext.add_alias.__dict__.__setitem__('stypy_param_names_list', ['alias_name', 'variable_name'])
        FunctionContext.add_alias.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionContext.add_alias.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionContext.add_alias.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionContext.add_alias.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionContext.add_alias.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionContext.add_alias.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionContext.add_alias', ['alias_name', 'variable_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_alias', localization, ['alias_name', 'variable_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_alias(...)' code ##################

        str_12279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, (-1)), 'str', '\n        Adds an alias to the alias storage of this function context\n        :param alias_name: Name of the alias\n        :param variable_name: Name of the aliased variable\n        :return:\n        ')
        
        # Assigning a Name to a Subscript (line 135):
        # Getting the type of 'variable_name' (line 135)
        variable_name_12280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 35), 'variable_name')
        # Getting the type of 'self' (line 135)
        self_12281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'self')
        # Obtaining the member 'aliases' of a type (line 135)
        aliases_12282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), self_12281, 'aliases')
        # Getting the type of 'alias_name' (line 135)
        alias_name_12283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 21), 'alias_name')
        # Storing an element on a container (line 135)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), aliases_12282, (alias_name_12283, variable_name_12280))
        
        # ################# End of 'add_alias(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_alias' in the type store
        # Getting the type of 'stypy_return_type' (line 128)
        stypy_return_type_12284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12284)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_alias'
        return stypy_return_type_12284


    @norecursion
    def get_type_of(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_type_of'
        module_type_store = module_type_store.open_function_context('get_type_of', 137, 4, False)
        # Assigning a type to the variable 'self' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionContext.get_type_of.__dict__.__setitem__('stypy_localization', localization)
        FunctionContext.get_type_of.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionContext.get_type_of.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionContext.get_type_of.__dict__.__setitem__('stypy_function_name', 'FunctionContext.get_type_of')
        FunctionContext.get_type_of.__dict__.__setitem__('stypy_param_names_list', ['variable_name'])
        FunctionContext.get_type_of.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionContext.get_type_of.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionContext.get_type_of.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionContext.get_type_of.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionContext.get_type_of.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionContext.get_type_of.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionContext.get_type_of', ['variable_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_type_of', localization, ['variable_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_type_of(...)' code ##################

        str_12285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, (-1)), 'str', '\n        Returns the type of a variable or parameter in the local context\n        :param variable_name: Name of the variable in the context\n        :return: The variable type or None if the variable do not belong to this context locally\n        ')
        
        # Getting the type of 'variable_name' (line 143)
        variable_name_12286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 11), 'variable_name')
        
        # Call to keys(...): (line 143)
        # Processing the call keyword arguments (line 143)
        kwargs_12290 = {}
        # Getting the type of 'self' (line 143)
        self_12287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 28), 'self', False)
        # Obtaining the member 'aliases' of a type (line 143)
        aliases_12288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 28), self_12287, 'aliases')
        # Obtaining the member 'keys' of a type (line 143)
        keys_12289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 28), aliases_12288, 'keys')
        # Calling keys(args, kwargs) (line 143)
        keys_call_result_12291 = invoke(stypy.reporting.localization.Localization(__file__, 143, 28), keys_12289, *[], **kwargs_12290)
        
        # Applying the binary operator 'in' (line 143)
        result_contains_12292 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 11), 'in', variable_name_12286, keys_call_result_12291)
        
        # Testing if the type of an if condition is none (line 143)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 143, 8), result_contains_12292):
            pass
        else:
            
            # Testing the type of an if condition (line 143)
            if_condition_12293 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 8), result_contains_12292)
            # Assigning a type to the variable 'if_condition_12293' (line 143)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'if_condition_12293', if_condition_12293)
            # SSA begins for if statement (line 143)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 144):
            
            # Obtaining the type of the subscript
            # Getting the type of 'variable_name' (line 144)
            variable_name_12294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 41), 'variable_name')
            # Getting the type of 'self' (line 144)
            self_12295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 28), 'self')
            # Obtaining the member 'aliases' of a type (line 144)
            aliases_12296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 28), self_12295, 'aliases')
            # Obtaining the member '__getitem__' of a type (line 144)
            getitem___12297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 28), aliases_12296, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 144)
            subscript_call_result_12298 = invoke(stypy.reporting.localization.Localization(__file__, 144, 28), getitem___12297, variable_name_12294)
            
            # Assigning a type to the variable 'variable_name' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'variable_name', subscript_call_result_12298)
            # SSA join for if statement (line 143)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'variable_name' (line 146)
        variable_name_12299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'variable_name')
        # Getting the type of 'self' (line 146)
        self_12300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 28), 'self')
        # Obtaining the member 'types_of' of a type (line 146)
        types_of_12301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 28), self_12300, 'types_of')
        # Applying the binary operator 'in' (line 146)
        result_contains_12302 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 11), 'in', variable_name_12299, types_of_12301)
        
        # Testing if the type of an if condition is none (line 146)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 146, 8), result_contains_12302):
            pass
        else:
            
            # Testing the type of an if condition (line 146)
            if_condition_12303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 8), result_contains_12302)
            # Assigning a type to the variable 'if_condition_12303' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'if_condition_12303', if_condition_12303)
            # SSA begins for if statement (line 146)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            # Getting the type of 'variable_name' (line 147)
            variable_name_12304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 33), 'variable_name')
            # Getting the type of 'self' (line 147)
            self_12305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 19), 'self')
            # Obtaining the member 'types_of' of a type (line 147)
            types_of_12306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 19), self_12305, 'types_of')
            # Obtaining the member '__getitem__' of a type (line 147)
            getitem___12307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 19), types_of_12306, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 147)
            subscript_call_result_12308 = invoke(stypy.reporting.localization.Localization(__file__, 147, 19), getitem___12307, variable_name_12304)
            
            # Assigning a type to the variable 'stypy_return_type' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'stypy_return_type', subscript_call_result_12308)
            # SSA join for if statement (line 146)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 149)
        None_12309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type', None_12309)
        
        # ################# End of 'get_type_of(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_type_of' in the type store
        # Getting the type of 'stypy_return_type' (line 137)
        stypy_return_type_12310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12310)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_type_of'
        return stypy_return_type_12310


    @norecursion
    def set_type_of(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_type_of'
        module_type_store = module_type_store.open_function_context('set_type_of', 151, 4, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionContext.set_type_of.__dict__.__setitem__('stypy_localization', localization)
        FunctionContext.set_type_of.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionContext.set_type_of.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionContext.set_type_of.__dict__.__setitem__('stypy_function_name', 'FunctionContext.set_type_of')
        FunctionContext.set_type_of.__dict__.__setitem__('stypy_param_names_list', ['name', 'type_', 'localization'])
        FunctionContext.set_type_of.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionContext.set_type_of.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionContext.set_type_of.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionContext.set_type_of.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionContext.set_type_of.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionContext.set_type_of.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionContext.set_type_of', ['name', 'type_', 'localization'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_type_of', localization, ['name', 'type_', 'localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_type_of(...)' code ##################

        str_12311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, (-1)), 'str', '\n        Sets the type of name to type in this local context\n        :param name: Name to search\n        :param type: Type to assign to name\n        ')
        # Getting the type of 'self' (line 157)
        self_12312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 11), 'self')
        # Obtaining the member 'annotate_types' of a type (line 157)
        annotate_types_12313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 11), self_12312, 'annotate_types')
        # Testing if the type of an if condition is none (line 157)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 157, 8), annotate_types_12313):
            pass
        else:
            
            # Testing the type of an if condition (line 157)
            if_condition_12314 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 8), annotate_types_12313)
            # Assigning a type to the variable 'if_condition_12314' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'if_condition_12314', if_condition_12314)
            # SSA begins for if statement (line 157)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to annotate_type(...): (line 158)
            # Processing the call arguments (line 158)
            # Getting the type of 'localization' (line 158)
            localization_12318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 49), 'localization', False)
            # Obtaining the member 'line' of a type (line 158)
            line_12319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 49), localization_12318, 'line')
            # Getting the type of 'localization' (line 158)
            localization_12320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 68), 'localization', False)
            # Obtaining the member 'column' of a type (line 158)
            column_12321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 68), localization_12320, 'column')
            # Getting the type of 'name' (line 158)
            name_12322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 89), 'name', False)
            # Getting the type of 'type_' (line 158)
            type__12323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 95), 'type_', False)
            # Processing the call keyword arguments (line 158)
            kwargs_12324 = {}
            # Getting the type of 'self' (line 158)
            self_12315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'self', False)
            # Obtaining the member 'annotation_record' of a type (line 158)
            annotation_record_12316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 12), self_12315, 'annotation_record')
            # Obtaining the member 'annotate_type' of a type (line 158)
            annotate_type_12317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 12), annotation_record_12316, 'annotate_type')
            # Calling annotate_type(args, kwargs) (line 158)
            annotate_type_call_result_12325 = invoke(stypy.reporting.localization.Localization(__file__, 158, 12), annotate_type_12317, *[line_12319, column_12321, name_12322, type__12323], **kwargs_12324)
            
            # SSA join for if statement (line 157)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'name' (line 160)
        name_12326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'name')
        
        # Call to keys(...): (line 160)
        # Processing the call keyword arguments (line 160)
        kwargs_12330 = {}
        # Getting the type of 'self' (line 160)
        self_12327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 19), 'self', False)
        # Obtaining the member 'aliases' of a type (line 160)
        aliases_12328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 19), self_12327, 'aliases')
        # Obtaining the member 'keys' of a type (line 160)
        keys_12329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 19), aliases_12328, 'keys')
        # Calling keys(args, kwargs) (line 160)
        keys_call_result_12331 = invoke(stypy.reporting.localization.Localization(__file__, 160, 19), keys_12329, *[], **kwargs_12330)
        
        # Applying the binary operator 'in' (line 160)
        result_contains_12332 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 11), 'in', name_12326, keys_call_result_12331)
        
        # Testing if the type of an if condition is none (line 160)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 160, 8), result_contains_12332):
            pass
        else:
            
            # Testing the type of an if condition (line 160)
            if_condition_12333 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 8), result_contains_12332)
            # Assigning a type to the variable 'if_condition_12333' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'if_condition_12333', if_condition_12333)
            # SSA begins for if statement (line 160)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 161):
            
            # Obtaining the type of the subscript
            # Getting the type of 'name' (line 161)
            name_12334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 32), 'name')
            # Getting the type of 'self' (line 161)
            self_12335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 19), 'self')
            # Obtaining the member 'aliases' of a type (line 161)
            aliases_12336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 19), self_12335, 'aliases')
            # Obtaining the member '__getitem__' of a type (line 161)
            getitem___12337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 19), aliases_12336, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 161)
            subscript_call_result_12338 = invoke(stypy.reporting.localization.Localization(__file__, 161, 19), getitem___12337, name_12334)
            
            # Assigning a type to the variable 'name' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'name', subscript_call_result_12338)
            # SSA join for if statement (line 160)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Subscript (line 162):
        # Getting the type of 'type_' (line 162)
        type__12339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 30), 'type_')
        # Getting the type of 'self' (line 162)
        self_12340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'self')
        # Obtaining the member 'types_of' of a type (line 162)
        types_of_12341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), self_12340, 'types_of')
        # Getting the type of 'name' (line 162)
        name_12342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 22), 'name')
        # Storing an element on a container (line 162)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 8), types_of_12341, (name_12342, type__12339))
        
        # ################# End of 'set_type_of(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_type_of' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_12343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12343)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_type_of'
        return stypy_return_type_12343


    @norecursion
    def del_type_of(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'del_type_of'
        module_type_store = module_type_store.open_function_context('del_type_of', 164, 4, False)
        # Assigning a type to the variable 'self' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionContext.del_type_of.__dict__.__setitem__('stypy_localization', localization)
        FunctionContext.del_type_of.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionContext.del_type_of.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionContext.del_type_of.__dict__.__setitem__('stypy_function_name', 'FunctionContext.del_type_of')
        FunctionContext.del_type_of.__dict__.__setitem__('stypy_param_names_list', ['variable_name'])
        FunctionContext.del_type_of.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionContext.del_type_of.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionContext.del_type_of.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionContext.del_type_of.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionContext.del_type_of.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionContext.del_type_of.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionContext.del_type_of', ['variable_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'del_type_of', localization, ['variable_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'del_type_of(...)' code ##################

        str_12344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, (-1)), 'str', '\n        Deletes the type of a variable or parameter in the local context\n        :param variable_name: Name of the variable in the context\n        ')
        
        # Getting the type of 'variable_name' (line 169)
        variable_name_12345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 11), 'variable_name')
        # Getting the type of 'self' (line 169)
        self_12346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'self')
        # Obtaining the member 'types_of' of a type (line 169)
        types_of_12347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 28), self_12346, 'types_of')
        # Applying the binary operator 'in' (line 169)
        result_contains_12348 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 11), 'in', variable_name_12345, types_of_12347)
        
        # Testing if the type of an if condition is none (line 169)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 169, 8), result_contains_12348):
            pass
        else:
            
            # Testing the type of an if condition (line 169)
            if_condition_12349 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 8), result_contains_12348)
            # Assigning a type to the variable 'if_condition_12349' (line 169)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'if_condition_12349', if_condition_12349)
            # SSA begins for if statement (line 169)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Deleting a member
            # Getting the type of 'self' (line 170)
            self_12350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'self')
            # Obtaining the member 'types_of' of a type (line 170)
            types_of_12351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 16), self_12350, 'types_of')
            
            # Obtaining the type of the subscript
            # Getting the type of 'variable_name' (line 170)
            variable_name_12352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 30), 'variable_name')
            # Getting the type of 'self' (line 170)
            self_12353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'self')
            # Obtaining the member 'types_of' of a type (line 170)
            types_of_12354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 16), self_12353, 'types_of')
            # Obtaining the member '__getitem__' of a type (line 170)
            getitem___12355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 16), types_of_12354, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 170)
            subscript_call_result_12356 = invoke(stypy.reporting.localization.Localization(__file__, 170, 16), getitem___12355, variable_name_12352)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 12), types_of_12351, subscript_call_result_12356)
            # SSA join for if statement (line 169)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 172)
        None_12357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'stypy_return_type', None_12357)
        
        # ################# End of 'del_type_of(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'del_type_of' in the type store
        # Getting the type of 'stypy_return_type' (line 164)
        stypy_return_type_12358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12358)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'del_type_of'
        return stypy_return_type_12358


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionContext.__iter__.__dict__.__setitem__('stypy_localization', localization)
        FunctionContext.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionContext.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionContext.__iter__.__dict__.__setitem__('stypy_function_name', 'FunctionContext.__iter__')
        FunctionContext.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        FunctionContext.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionContext.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionContext.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionContext.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionContext.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionContext.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionContext.__iter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iter__(...)' code ##################

        str_12359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, (-1)), 'str', '\n        Allows iteration through all the variable names stored in the context.\n        :return: Each variable name stored in the context\n        ')
        
        # Getting the type of 'self' (line 179)
        self_12360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 29), 'self')
        # Obtaining the member 'types_of' of a type (line 179)
        types_of_12361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 29), self_12360, 'types_of')
        # Assigning a type to the variable 'types_of_12361' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'types_of_12361', types_of_12361)
        # Testing if the for loop is going to be iterated (line 179)
        # Testing the type of a for loop iterable (line 179)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 179, 8), types_of_12361)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 179, 8), types_of_12361):
            # Getting the type of the for loop variable (line 179)
            for_loop_var_12362 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 179, 8), types_of_12361)
            # Assigning a type to the variable 'variable_name' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'variable_name', for_loop_var_12362)
            # SSA begins for a for statement (line 179)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Creating a generator
            # Getting the type of 'variable_name' (line 180)
            variable_name_12363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 18), 'variable_name')
            GeneratorType_12364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 12), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 12), GeneratorType_12364, variable_name_12363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'stypy_return_type', GeneratorType_12364)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_12365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12365)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_12365


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 182, 4, False)
        # Assigning a type to the variable 'self' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionContext.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        FunctionContext.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionContext.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionContext.__getitem__.__dict__.__setitem__('stypy_function_name', 'FunctionContext.__getitem__')
        FunctionContext.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['item'])
        FunctionContext.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionContext.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionContext.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionContext.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionContext.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionContext.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionContext.__getitem__', ['item'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['item'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        str_12366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, (-1)), 'str', '\n        Allows the usage of the [] operator to access variable types by variable name\n        :param item: Variable name\n        :return: Same as get_type_of\n        ')
        
        # Call to get_type_of(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'item' (line 188)
        item_12369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 32), 'item', False)
        # Processing the call keyword arguments (line 188)
        kwargs_12370 = {}
        # Getting the type of 'self' (line 188)
        self_12367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'self', False)
        # Obtaining the member 'get_type_of' of a type (line 188)
        get_type_of_12368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 15), self_12367, 'get_type_of')
        # Calling get_type_of(args, kwargs) (line 188)
        get_type_of_call_result_12371 = invoke(stypy.reporting.localization.Localization(__file__, 188, 15), get_type_of_12368, *[item_12369], **kwargs_12370)
        
        # Assigning a type to the variable 'stypy_return_type' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'stypy_return_type', get_type_of_call_result_12371)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 182)
        stypy_return_type_12372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12372)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_12372


    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 190, 4, False)
        # Assigning a type to the variable 'self' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionContext.__len__.__dict__.__setitem__('stypy_localization', localization)
        FunctionContext.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionContext.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionContext.__len__.__dict__.__setitem__('stypy_function_name', 'FunctionContext.__len__')
        FunctionContext.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        FunctionContext.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionContext.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionContext.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionContext.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionContext.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionContext.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionContext.__len__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__len__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__len__(...)' code ##################

        str_12373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, (-1)), 'str', '\n        len operator, returning the amount of stored local variables\n        :return:\n        ')
        
        # Call to len(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'self' (line 195)
        self_12375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 19), 'self', False)
        # Obtaining the member 'types_of' of a type (line 195)
        types_of_12376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 19), self_12375, 'types_of')
        # Processing the call keyword arguments (line 195)
        kwargs_12377 = {}
        # Getting the type of 'len' (line 195)
        len_12374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'len', False)
        # Calling len(args, kwargs) (line 195)
        len_call_result_12378 = invoke(stypy.reporting.localization.Localization(__file__, 195, 15), len_12374, *[types_of_12376], **kwargs_12377)
        
        # Assigning a type to the variable 'stypy_return_type' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'stypy_return_type', len_call_result_12378)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 190)
        stypy_return_type_12379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12379)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_12379


    @norecursion
    def clone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clone'
        module_type_store = module_type_store.open_function_context('clone', 197, 4, False)
        # Assigning a type to the variable 'self' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionContext.clone.__dict__.__setitem__('stypy_localization', localization)
        FunctionContext.clone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionContext.clone.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionContext.clone.__dict__.__setitem__('stypy_function_name', 'FunctionContext.clone')
        FunctionContext.clone.__dict__.__setitem__('stypy_param_names_list', [])
        FunctionContext.clone.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionContext.clone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionContext.clone.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionContext.clone.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionContext.clone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionContext.clone.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionContext.clone', [], None, None, defaults, varargs, kwargs)

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

        str_12380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, (-1)), 'str', '\n        Clones the whole function context. The returned function context is a deep copy of the current one\n        :return: Cloned function context\n        ')
        
        # Assigning a Call to a Name (line 202):
        
        # Call to FunctionContext(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'self' (line 202)
        self_12382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 37), 'self', False)
        # Obtaining the member 'function_name' of a type (line 202)
        function_name_12383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 37), self_12382, 'function_name')
        # Processing the call keyword arguments (line 202)
        kwargs_12384 = {}
        # Getting the type of 'FunctionContext' (line 202)
        FunctionContext_12381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 21), 'FunctionContext', False)
        # Calling FunctionContext(args, kwargs) (line 202)
        FunctionContext_call_result_12385 = invoke(stypy.reporting.localization.Localization(__file__, 202, 21), FunctionContext_12381, *[function_name_12383], **kwargs_12384)
        
        # Assigning a type to the variable 'cloned_obj' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'cloned_obj', FunctionContext_call_result_12385)
        
        # Assigning a Call to a Attribute (line 204):
        
        # Call to deepcopy(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'self' (line 204)
        self_12388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 47), 'self', False)
        # Obtaining the member 'global_vars' of a type (line 204)
        global_vars_12389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 47), self_12388, 'global_vars')
        # Processing the call keyword arguments (line 204)
        kwargs_12390 = {}
        # Getting the type of 'copy' (line 204)
        copy_12386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 33), 'copy', False)
        # Obtaining the member 'deepcopy' of a type (line 204)
        deepcopy_12387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 33), copy_12386, 'deepcopy')
        # Calling deepcopy(args, kwargs) (line 204)
        deepcopy_call_result_12391 = invoke(stypy.reporting.localization.Localization(__file__, 204, 33), deepcopy_12387, *[global_vars_12389], **kwargs_12390)
        
        # Getting the type of 'cloned_obj' (line 204)
        cloned_obj_12392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'cloned_obj')
        # Setting the type of the member 'global_vars' of a type (line 204)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), cloned_obj_12392, 'global_vars', deepcopy_call_result_12391)
        
        
        # Call to iteritems(...): (line 206)
        # Processing the call keyword arguments (line 206)
        kwargs_12396 = {}
        # Getting the type of 'self' (line 206)
        self_12393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'self', False)
        # Obtaining the member 'types_of' of a type (line 206)
        types_of_12394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 26), self_12393, 'types_of')
        # Obtaining the member 'iteritems' of a type (line 206)
        iteritems_12395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 26), types_of_12394, 'iteritems')
        # Calling iteritems(args, kwargs) (line 206)
        iteritems_call_result_12397 = invoke(stypy.reporting.localization.Localization(__file__, 206, 26), iteritems_12395, *[], **kwargs_12396)
        
        # Assigning a type to the variable 'iteritems_call_result_12397' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'iteritems_call_result_12397', iteritems_call_result_12397)
        # Testing if the for loop is going to be iterated (line 206)
        # Testing the type of a for loop iterable (line 206)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 206, 8), iteritems_call_result_12397)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 206, 8), iteritems_call_result_12397):
            # Getting the type of the for loop variable (line 206)
            for_loop_var_12398 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 206, 8), iteritems_call_result_12397)
            # Assigning a type to the variable 'key' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 8), for_loop_var_12398, 2, 0))
            # Assigning a type to the variable 'value' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 8), for_loop_var_12398, 2, 1))
            # SSA begins for a for statement (line 206)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to isinstance(...): (line 207)
            # Processing the call arguments (line 207)
            # Getting the type of 'value' (line 207)
            value_12400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 26), 'value', False)
            # Getting the type of 'Type' (line 207)
            Type_12401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 33), 'Type', False)
            # Processing the call keyword arguments (line 207)
            kwargs_12402 = {}
            # Getting the type of 'isinstance' (line 207)
            isinstance_12399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 207)
            isinstance_call_result_12403 = invoke(stypy.reporting.localization.Localization(__file__, 207, 15), isinstance_12399, *[value_12400, Type_12401], **kwargs_12402)
            
            # Testing if the type of an if condition is none (line 207)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 207, 12), isinstance_call_result_12403):
                
                # Assigning a Call to a Name (line 210):
                
                # Call to deepcopy(...): (line 210)
                # Processing the call arguments (line 210)
                # Getting the type of 'value' (line 210)
                value_12411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 40), 'value', False)
                # Processing the call keyword arguments (line 210)
                kwargs_12412 = {}
                # Getting the type of 'copy' (line 210)
                copy_12409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 26), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 210)
                deepcopy_12410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 26), copy_12409, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 210)
                deepcopy_call_result_12413 = invoke(stypy.reporting.localization.Localization(__file__, 210, 26), deepcopy_12410, *[value_12411], **kwargs_12412)
                
                # Assigning a type to the variable 'new_obj' (line 210)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'new_obj', deepcopy_call_result_12413)
            else:
                
                # Testing the type of an if condition (line 207)
                if_condition_12404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 12), isinstance_call_result_12403)
                # Assigning a type to the variable 'if_condition_12404' (line 207)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'if_condition_12404', if_condition_12404)
                # SSA begins for if statement (line 207)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 208):
                
                # Call to clone(...): (line 208)
                # Processing the call keyword arguments (line 208)
                kwargs_12407 = {}
                # Getting the type of 'value' (line 208)
                value_12405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 26), 'value', False)
                # Obtaining the member 'clone' of a type (line 208)
                clone_12406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 26), value_12405, 'clone')
                # Calling clone(args, kwargs) (line 208)
                clone_call_result_12408 = invoke(stypy.reporting.localization.Localization(__file__, 208, 26), clone_12406, *[], **kwargs_12407)
                
                # Assigning a type to the variable 'new_obj' (line 208)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'new_obj', clone_call_result_12408)
                # SSA branch for the else part of an if statement (line 207)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 210):
                
                # Call to deepcopy(...): (line 210)
                # Processing the call arguments (line 210)
                # Getting the type of 'value' (line 210)
                value_12411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 40), 'value', False)
                # Processing the call keyword arguments (line 210)
                kwargs_12412 = {}
                # Getting the type of 'copy' (line 210)
                copy_12409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 26), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 210)
                deepcopy_12410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 26), copy_12409, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 210)
                deepcopy_call_result_12413 = invoke(stypy.reporting.localization.Localization(__file__, 210, 26), deepcopy_12410, *[value_12411], **kwargs_12412)
                
                # Assigning a type to the variable 'new_obj' (line 210)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'new_obj', deepcopy_call_result_12413)
                # SSA join for if statement (line 207)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Name to a Subscript (line 212):
            # Getting the type of 'new_obj' (line 212)
            new_obj_12414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 39), 'new_obj')
            # Getting the type of 'cloned_obj' (line 212)
            cloned_obj_12415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'cloned_obj')
            # Obtaining the member 'types_of' of a type (line 212)
            types_of_12416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), cloned_obj_12415, 'types_of')
            # Getting the type of 'key' (line 212)
            key_12417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 32), 'key')
            # Storing an element on a container (line 212)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 12), types_of_12416, (key_12417, new_obj_12414))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Attribute (line 214):
        
        # Call to deepcopy(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'self' (line 214)
        self_12420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 43), 'self', False)
        # Obtaining the member 'aliases' of a type (line 214)
        aliases_12421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 43), self_12420, 'aliases')
        # Processing the call keyword arguments (line 214)
        kwargs_12422 = {}
        # Getting the type of 'copy' (line 214)
        copy_12418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 29), 'copy', False)
        # Obtaining the member 'deepcopy' of a type (line 214)
        deepcopy_12419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 29), copy_12418, 'deepcopy')
        # Calling deepcopy(args, kwargs) (line 214)
        deepcopy_call_result_12423 = invoke(stypy.reporting.localization.Localization(__file__, 214, 29), deepcopy_12419, *[aliases_12421], **kwargs_12422)
        
        # Getting the type of 'cloned_obj' (line 214)
        cloned_obj_12424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'cloned_obj')
        # Setting the type of the member 'aliases' of a type (line 214)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), cloned_obj_12424, 'aliases', deepcopy_call_result_12423)
        
        # Assigning a Attribute to a Attribute (line 215):
        # Getting the type of 'self' (line 215)
        self_12425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 39), 'self')
        # Obtaining the member 'annotation_record' of a type (line 215)
        annotation_record_12426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 39), self_12425, 'annotation_record')
        # Getting the type of 'cloned_obj' (line 215)
        cloned_obj_12427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'cloned_obj')
        # Setting the type of the member 'annotation_record' of a type (line 215)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), cloned_obj_12427, 'annotation_record', annotation_record_12426)
        
        # Assigning a Attribute to a Attribute (line 216):
        # Getting the type of 'self' (line 216)
        self_12428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 37), 'self')
        # Obtaining the member 'is_main_context' of a type (line 216)
        is_main_context_12429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 37), self_12428, 'is_main_context')
        # Getting the type of 'cloned_obj' (line 216)
        cloned_obj_12430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'cloned_obj')
        # Setting the type of the member 'is_main_context' of a type (line 216)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), cloned_obj_12430, 'is_main_context', is_main_context_12429)
        
        # Assigning a Attribute to a Attribute (line 219):
        # Getting the type of 'self' (line 219)
        self_12431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 49), 'self')
        # Obtaining the member 'declared_argument_name_list' of a type (line 219)
        declared_argument_name_list_12432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 49), self_12431, 'declared_argument_name_list')
        # Getting the type of 'cloned_obj' (line 219)
        cloned_obj_12433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'cloned_obj')
        # Setting the type of the member 'declared_argument_name_list' of a type (line 219)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), cloned_obj_12433, 'declared_argument_name_list', declared_argument_name_list_12432)
        
        # Assigning a Attribute to a Attribute (line 220):
        # Getting the type of 'self' (line 220)
        self_12434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 42), 'self')
        # Obtaining the member 'declared_varargs_var' of a type (line 220)
        declared_varargs_var_12435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 42), self_12434, 'declared_varargs_var')
        # Getting the type of 'cloned_obj' (line 220)
        cloned_obj_12436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'cloned_obj')
        # Setting the type of the member 'declared_varargs_var' of a type (line 220)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), cloned_obj_12436, 'declared_varargs_var', declared_varargs_var_12435)
        
        # Assigning a Attribute to a Attribute (line 221):
        # Getting the type of 'self' (line 221)
        self_12437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 41), 'self')
        # Obtaining the member 'declared_kwargs_var' of a type (line 221)
        declared_kwargs_var_12438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 41), self_12437, 'declared_kwargs_var')
        # Getting the type of 'cloned_obj' (line 221)
        cloned_obj_12439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'cloned_obj')
        # Setting the type of the member 'declared_kwargs_var' of a type (line 221)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), cloned_obj_12439, 'declared_kwargs_var', declared_kwargs_var_12438)
        
        # Assigning a Attribute to a Attribute (line 222):
        # Getting the type of 'self' (line 222)
        self_12440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 39), 'self')
        # Obtaining the member 'declared_defaults' of a type (line 222)
        declared_defaults_12441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 39), self_12440, 'declared_defaults')
        # Getting the type of 'cloned_obj' (line 222)
        cloned_obj_12442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'cloned_obj')
        # Setting the type of the member 'declared_defaults' of a type (line 222)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), cloned_obj_12442, 'declared_defaults', declared_defaults_12441)
        
        # Assigning a Attribute to a Attribute (line 224):
        # Getting the type of 'self' (line 224)
        self_12443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 38), 'self')
        # Obtaining the member 'declaration_line' of a type (line 224)
        declaration_line_12444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 38), self_12443, 'declaration_line')
        # Getting the type of 'cloned_obj' (line 224)
        cloned_obj_12445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'cloned_obj')
        # Setting the type of the member 'declaration_line' of a type (line 224)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), cloned_obj_12445, 'declaration_line', declaration_line_12444)
        
        # Assigning a Attribute to a Attribute (line 225):
        # Getting the type of 'self' (line 225)
        self_12446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 40), 'self')
        # Obtaining the member 'declaration_column' of a type (line 225)
        declaration_column_12447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 40), self_12446, 'declaration_column')
        # Getting the type of 'cloned_obj' (line 225)
        cloned_obj_12448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'cloned_obj')
        # Setting the type of the member 'declaration_column' of a type (line 225)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), cloned_obj_12448, 'declaration_column', declaration_column_12447)
        
        # Assigning a Attribute to a Attribute (line 227):
        # Getting the type of 'self' (line 227)
        self_12449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 33), 'self')
        # Obtaining the member 'return_type' of a type (line 227)
        return_type_12450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 33), self_12449, 'return_type')
        # Getting the type of 'cloned_obj' (line 227)
        cloned_obj_12451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'cloned_obj')
        # Setting the type of the member 'return_type' of a type (line 227)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), cloned_obj_12451, 'return_type', return_type_12450)
        # Getting the type of 'cloned_obj' (line 229)
        cloned_obj_12452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'cloned_obj')
        # Assigning a type to the variable 'stypy_return_type' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'stypy_return_type', cloned_obj_12452)
        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 197)
        stypy_return_type_12453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12453)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_12453


    @norecursion
    def copy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy'
        module_type_store = module_type_store.open_function_context('copy', 231, 4, False)
        # Assigning a type to the variable 'self' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FunctionContext.copy.__dict__.__setitem__('stypy_localization', localization)
        FunctionContext.copy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FunctionContext.copy.__dict__.__setitem__('stypy_type_store', module_type_store)
        FunctionContext.copy.__dict__.__setitem__('stypy_function_name', 'FunctionContext.copy')
        FunctionContext.copy.__dict__.__setitem__('stypy_param_names_list', [])
        FunctionContext.copy.__dict__.__setitem__('stypy_varargs_param_name', None)
        FunctionContext.copy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FunctionContext.copy.__dict__.__setitem__('stypy_call_defaults', defaults)
        FunctionContext.copy.__dict__.__setitem__('stypy_call_varargs', varargs)
        FunctionContext.copy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FunctionContext.copy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FunctionContext.copy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy(...)' code ##################

        str_12454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, (-1)), 'str', '\n        Copies this function context into a newly created one and return it. The copied function context is a shallow\n        copy.\n        :return: Copy of this function context\n        ')
        
        # Assigning a Call to a Name (line 237):
        
        # Call to FunctionContext(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'self' (line 237)
        self_12456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 37), 'self', False)
        # Obtaining the member 'function_name' of a type (line 237)
        function_name_12457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 37), self_12456, 'function_name')
        # Processing the call keyword arguments (line 237)
        kwargs_12458 = {}
        # Getting the type of 'FunctionContext' (line 237)
        FunctionContext_12455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 21), 'FunctionContext', False)
        # Calling FunctionContext(args, kwargs) (line 237)
        FunctionContext_call_result_12459 = invoke(stypy.reporting.localization.Localization(__file__, 237, 21), FunctionContext_12455, *[function_name_12457], **kwargs_12458)
        
        # Assigning a type to the variable 'copied_obj' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'copied_obj', FunctionContext_call_result_12459)
        
        # Assigning a Attribute to a Attribute (line 239):
        # Getting the type of 'self' (line 239)
        self_12460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 33), 'self')
        # Obtaining the member 'global_vars' of a type (line 239)
        global_vars_12461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 33), self_12460, 'global_vars')
        # Getting the type of 'copied_obj' (line 239)
        copied_obj_12462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'copied_obj')
        # Setting the type of the member 'global_vars' of a type (line 239)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), copied_obj_12462, 'global_vars', global_vars_12461)
        
        # Assigning a Attribute to a Attribute (line 240):
        # Getting the type of 'self' (line 240)
        self_12463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 30), 'self')
        # Obtaining the member 'types_of' of a type (line 240)
        types_of_12464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 30), self_12463, 'types_of')
        # Getting the type of 'copied_obj' (line 240)
        copied_obj_12465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'copied_obj')
        # Setting the type of the member 'types_of' of a type (line 240)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), copied_obj_12465, 'types_of', types_of_12464)
        
        # Assigning a Attribute to a Attribute (line 242):
        # Getting the type of 'self' (line 242)
        self_12466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 29), 'self')
        # Obtaining the member 'aliases' of a type (line 242)
        aliases_12467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 29), self_12466, 'aliases')
        # Getting the type of 'copied_obj' (line 242)
        copied_obj_12468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'copied_obj')
        # Setting the type of the member 'aliases' of a type (line 242)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), copied_obj_12468, 'aliases', aliases_12467)
        
        # Assigning a Attribute to a Attribute (line 243):
        # Getting the type of 'self' (line 243)
        self_12469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 39), 'self')
        # Obtaining the member 'annotation_record' of a type (line 243)
        annotation_record_12470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 39), self_12469, 'annotation_record')
        # Getting the type of 'copied_obj' (line 243)
        copied_obj_12471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'copied_obj')
        # Setting the type of the member 'annotation_record' of a type (line 243)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), copied_obj_12471, 'annotation_record', annotation_record_12470)
        
        # Assigning a Attribute to a Attribute (line 244):
        # Getting the type of 'self' (line 244)
        self_12472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 37), 'self')
        # Obtaining the member 'is_main_context' of a type (line 244)
        is_main_context_12473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 37), self_12472, 'is_main_context')
        # Getting the type of 'copied_obj' (line 244)
        copied_obj_12474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'copied_obj')
        # Setting the type of the member 'is_main_context' of a type (line 244)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), copied_obj_12474, 'is_main_context', is_main_context_12473)
        
        # Assigning a Attribute to a Attribute (line 247):
        # Getting the type of 'self' (line 247)
        self_12475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 49), 'self')
        # Obtaining the member 'declared_argument_name_list' of a type (line 247)
        declared_argument_name_list_12476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 49), self_12475, 'declared_argument_name_list')
        # Getting the type of 'copied_obj' (line 247)
        copied_obj_12477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'copied_obj')
        # Setting the type of the member 'declared_argument_name_list' of a type (line 247)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), copied_obj_12477, 'declared_argument_name_list', declared_argument_name_list_12476)
        
        # Assigning a Attribute to a Attribute (line 248):
        # Getting the type of 'self' (line 248)
        self_12478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 42), 'self')
        # Obtaining the member 'declared_varargs_var' of a type (line 248)
        declared_varargs_var_12479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 42), self_12478, 'declared_varargs_var')
        # Getting the type of 'copied_obj' (line 248)
        copied_obj_12480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'copied_obj')
        # Setting the type of the member 'declared_varargs_var' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), copied_obj_12480, 'declared_varargs_var', declared_varargs_var_12479)
        
        # Assigning a Attribute to a Attribute (line 249):
        # Getting the type of 'self' (line 249)
        self_12481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 41), 'self')
        # Obtaining the member 'declared_kwargs_var' of a type (line 249)
        declared_kwargs_var_12482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 41), self_12481, 'declared_kwargs_var')
        # Getting the type of 'copied_obj' (line 249)
        copied_obj_12483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'copied_obj')
        # Setting the type of the member 'declared_kwargs_var' of a type (line 249)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), copied_obj_12483, 'declared_kwargs_var', declared_kwargs_var_12482)
        
        # Assigning a Attribute to a Attribute (line 250):
        # Getting the type of 'self' (line 250)
        self_12484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 39), 'self')
        # Obtaining the member 'declared_defaults' of a type (line 250)
        declared_defaults_12485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 39), self_12484, 'declared_defaults')
        # Getting the type of 'copied_obj' (line 250)
        copied_obj_12486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'copied_obj')
        # Setting the type of the member 'declared_defaults' of a type (line 250)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), copied_obj_12486, 'declared_defaults', declared_defaults_12485)
        
        # Assigning a Attribute to a Attribute (line 252):
        # Getting the type of 'self' (line 252)
        self_12487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 38), 'self')
        # Obtaining the member 'declaration_line' of a type (line 252)
        declaration_line_12488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 38), self_12487, 'declaration_line')
        # Getting the type of 'copied_obj' (line 252)
        copied_obj_12489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'copied_obj')
        # Setting the type of the member 'declaration_line' of a type (line 252)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), copied_obj_12489, 'declaration_line', declaration_line_12488)
        
        # Assigning a Attribute to a Attribute (line 253):
        # Getting the type of 'self' (line 253)
        self_12490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 40), 'self')
        # Obtaining the member 'declaration_column' of a type (line 253)
        declaration_column_12491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 40), self_12490, 'declaration_column')
        # Getting the type of 'copied_obj' (line 253)
        copied_obj_12492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'copied_obj')
        # Setting the type of the member 'declaration_column' of a type (line 253)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), copied_obj_12492, 'declaration_column', declaration_column_12491)
        
        # Assigning a Attribute to a Attribute (line 255):
        # Getting the type of 'self' (line 255)
        self_12493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 33), 'self')
        # Obtaining the member 'return_type' of a type (line 255)
        return_type_12494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 33), self_12493, 'return_type')
        # Getting the type of 'copied_obj' (line 255)
        copied_obj_12495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'copied_obj')
        # Setting the type of the member 'return_type' of a type (line 255)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), copied_obj_12495, 'return_type', return_type_12494)
        # Getting the type of 'copied_obj' (line 257)
        copied_obj_12496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'copied_obj')
        # Assigning a type to the variable 'stypy_return_type' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'stypy_return_type', copied_obj_12496)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_12497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_12497)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_12497


# Assigning a type to the variable 'FunctionContext' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'FunctionContext', FunctionContext)

# Assigning a Name to a Name (line 13):
# Getting the type of 'True' (line 13)
True_12498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 21), 'True')
# Getting the type of 'FunctionContext'
FunctionContext_12499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FunctionContext')
# Setting the type of the member 'annotate_types' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FunctionContext_12499, 'annotate_types', True_12498)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
