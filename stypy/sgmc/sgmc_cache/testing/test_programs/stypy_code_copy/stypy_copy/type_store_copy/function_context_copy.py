
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import copy
2: from ...stypy_copy.python_lib_copy.python_types_copy.type_copy import Type
3: from ...stypy_copy.reporting_copy import print_utils_copy
4: from ...stypy_copy.errors_copy.type_error_copy import TypeError
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
87:         txt += str(self.function_name) + "(" + arg_str + ") -> " + print_utils_copy.get_type_str(self.return_type)
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

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy import Type' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')
import_17172 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy')

if (type(import_17172) is not StypyTypeError):

    if (import_17172 != 'pyd_module'):
        __import__(import_17172)
        sys_modules_17173 = sys.modules[import_17172]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', sys_modules_17173.module_type_store, module_type_store, ['Type'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_17173, sys_modules_17173.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy import Type

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', None, module_type_store, ['Type'], [Type])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', import_17172)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy import print_utils_copy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')
import_17174 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy')

if (type(import_17174) is not StypyTypeError):

    if (import_17174 != 'pyd_module'):
        __import__(import_17174)
        sys_modules_17175 = sys.modules[import_17174]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy', sys_modules_17175.module_type_store, module_type_store, ['print_utils_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_17175, sys_modules_17175.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy import print_utils_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy', None, module_type_store, ['print_utils_copy'], [print_utils_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy', import_17174)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')
import_17176 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy')

if (type(import_17176) is not StypyTypeError):

    if (import_17176 != 'pyd_module'):
        __import__(import_17176)
        sys_modules_17177 = sys.modules[import_17176]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', sys_modules_17177.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_17177, sys_modules_17177.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', import_17176)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/type_store_copy/')

# Declaration of the 'FunctionContext' class

class FunctionContext:
    str_17178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, (-1)), 'str', '\n    Models a function/method local context, containing all its variables and types in a dictionary. A type store holds\n    a stack of function contexts, one per called function, tracking all its local context. This class also have the\n    optional feature of annotating types to create type-annotated programs, allowing the type annotation inside\n    functions code\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 15)
        False_17179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 54), 'False')
        defaults = [False_17179]
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

        str_17180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', '\n        Initializes the function context for function function_name\n        :param function_name: Name of the function\n        :param is_main_context: Whether it is the main context or not. There can be only a function context in the\n        program.\n        :return:\n        ')
        
        # Assigning a Dict to a Attribute (line 25):
        
        # Obtaining an instance of the builtin type 'dict' (line 25)
        dict_17181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 25)
        
        # Getting the type of 'self' (line 25)
        self_17182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member 'types_of' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_17182, 'types_of', dict_17181)
        
        # Assigning a Name to a Attribute (line 28):
        # Getting the type of 'function_name' (line 28)
        function_name_17183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'function_name')
        # Getting the type of 'self' (line 28)
        self_17184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'function_name' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_17184, 'function_name', function_name_17183)
        
        # Assigning a List to a Attribute (line 31):
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_17185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        
        # Getting the type of 'self' (line 31)
        self_17186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'global_vars' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_17186, 'global_vars', list_17185)
        
        # Assigning a Call to a Attribute (line 34):
        
        # Call to dict(...): (line 34)
        # Processing the call keyword arguments (line 34)
        kwargs_17188 = {}
        # Getting the type of 'dict' (line 34)
        dict_17187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'dict', False)
        # Calling dict(args, kwargs) (line 34)
        dict_call_result_17189 = invoke(stypy.reporting.localization.Localization(__file__, 34, 23), dict_17187, *[], **kwargs_17188)
        
        # Getting the type of 'self' (line 34)
        self_17190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'aliases' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_17190, 'aliases', dict_call_result_17189)
        
        # Assigning a Name to a Attribute (line 36):
        # Getting the type of 'is_main_context' (line 36)
        is_main_context_17191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'is_main_context')
        # Getting the type of 'self' (line 36)
        self_17192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self')
        # Setting the type of the member 'is_main_context' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_17192, 'is_main_context', is_main_context_17191)
        
        # Assigning a Name to a Attribute (line 40):
        # Getting the type of 'None' (line 40)
        None_17193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 43), 'None')
        # Getting the type of 'self' (line 40)
        self_17194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self')
        # Setting the type of the member 'declared_argument_name_list' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_17194, 'declared_argument_name_list', None_17193)
        
        # Assigning a Name to a Attribute (line 43):
        # Getting the type of 'None' (line 43)
        None_17195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 36), 'None')
        # Getting the type of 'self' (line 43)
        self_17196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member 'declared_varargs_var' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_17196, 'declared_varargs_var', None_17195)
        
        # Assigning a Name to a Attribute (line 46):
        # Getting the type of 'None' (line 46)
        None_17197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 35), 'None')
        # Getting the type of 'self' (line 46)
        self_17198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self')
        # Setting the type of the member 'declared_kwargs_var' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_17198, 'declared_kwargs_var', None_17197)
        
        # Assigning a Name to a Attribute (line 49):
        # Getting the type of 'None' (line 49)
        None_17199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 33), 'None')
        # Getting the type of 'self' (line 49)
        self_17200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'self')
        # Setting the type of the member 'declared_defaults' of a type (line 49)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), self_17200, 'declared_defaults', None_17199)
        
        # Assigning a Num to a Attribute (line 52):
        int_17201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 32), 'int')
        # Getting the type of 'self' (line 52)
        self_17202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self')
        # Setting the type of the member 'declaration_line' of a type (line 52)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_17202, 'declaration_line', int_17201)
        
        # Assigning a Num to a Attribute (line 53):
        int_17203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 34), 'int')
        # Getting the type of 'self' (line 53)
        self_17204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self')
        # Setting the type of the member 'declaration_column' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_17204, 'declaration_column', int_17203)
        
        # Assigning a Name to a Attribute (line 56):
        # Getting the type of 'None' (line 56)
        None_17205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 27), 'None')
        # Getting the type of 'self' (line 56)
        self_17206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member 'return_type' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_17206, 'return_type', None_17205)
        
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

        str_17207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, (-1)), 'str', '\n        Obtains an appropriate str to pretty-print the function context, formatting the header of the represented\n        function.\n        :return: str\n        ')
        
        # Assigning a Str to a Name (line 64):
        str_17208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 14), 'str', '')
        # Assigning a type to the variable 'txt' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'txt', str_17208)
        
        # Assigning a Str to a Name (line 65):
        str_17209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 18), 'str', '')
        # Assigning a type to the variable 'arg_str' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'arg_str', str_17209)
        
        # Getting the type of 'self' (line 66)
        self_17210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'self')
        # Obtaining the member 'declared_argument_name_list' of a type (line 66)
        declared_argument_name_list_17211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 11), self_17210, 'declared_argument_name_list')
        # Getting the type of 'None' (line 66)
        None_17212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 51), 'None')
        # Applying the binary operator 'isnot' (line 66)
        result_is_not_17213 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 11), 'isnot', declared_argument_name_list_17211, None_17212)
        
        # Testing if the type of an if condition is none (line 66)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 66, 8), result_is_not_17213):
            pass
        else:
            
            # Testing the type of an if condition (line 66)
            if_condition_17214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 8), result_is_not_17213)
            # Assigning a type to the variable 'if_condition_17214' (line 66)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'if_condition_17214', if_condition_17214)
            # SSA begins for if statement (line 66)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'self' (line 67)
            self_17215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'self')
            # Obtaining the member 'declared_argument_name_list' of a type (line 67)
            declared_argument_name_list_17216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 23), self_17215, 'declared_argument_name_list')
            # Assigning a type to the variable 'declared_argument_name_list_17216' (line 67)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'declared_argument_name_list_17216', declared_argument_name_list_17216)
            # Testing if the for loop is going to be iterated (line 67)
            # Testing the type of a for loop iterable (line 67)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 67, 12), declared_argument_name_list_17216)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 67, 12), declared_argument_name_list_17216):
                # Getting the type of the for loop variable (line 67)
                for_loop_var_17217 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 67, 12), declared_argument_name_list_17216)
                # Assigning a type to the variable 'arg' (line 67)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'arg', for_loop_var_17217)
                # SSA begins for a for statement (line 67)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'arg_str' (line 68)
                arg_str_17218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'arg_str')
                
                # Call to str(...): (line 68)
                # Processing the call arguments (line 68)
                # Getting the type of 'arg' (line 68)
                arg_17220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 31), 'arg', False)
                # Processing the call keyword arguments (line 68)
                kwargs_17221 = {}
                # Getting the type of 'str' (line 68)
                str_17219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 27), 'str', False)
                # Calling str(args, kwargs) (line 68)
                str_call_result_17222 = invoke(stypy.reporting.localization.Localization(__file__, 68, 27), str_17219, *[arg_17220], **kwargs_17221)
                
                str_17223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 38), 'str', ': ')
                # Applying the binary operator '+' (line 68)
                result_add_17224 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 27), '+', str_call_result_17222, str_17223)
                
                
                # Call to str(...): (line 68)
                # Processing the call arguments (line 68)
                
                # Call to get_type_of(...): (line 68)
                # Processing the call arguments (line 68)
                # Getting the type of 'arg' (line 68)
                arg_17228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 66), 'arg', False)
                # Processing the call keyword arguments (line 68)
                kwargs_17229 = {}
                # Getting the type of 'self' (line 68)
                self_17226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 49), 'self', False)
                # Obtaining the member 'get_type_of' of a type (line 68)
                get_type_of_17227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 49), self_17226, 'get_type_of')
                # Calling get_type_of(args, kwargs) (line 68)
                get_type_of_call_result_17230 = invoke(stypy.reporting.localization.Localization(__file__, 68, 49), get_type_of_17227, *[arg_17228], **kwargs_17229)
                
                # Processing the call keyword arguments (line 68)
                kwargs_17231 = {}
                # Getting the type of 'str' (line 68)
                str_17225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 45), 'str', False)
                # Calling str(args, kwargs) (line 68)
                str_call_result_17232 = invoke(stypy.reporting.localization.Localization(__file__, 68, 45), str_17225, *[get_type_of_call_result_17230], **kwargs_17231)
                
                # Applying the binary operator '+' (line 68)
                result_add_17233 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 43), '+', result_add_17224, str_call_result_17232)
                
                str_17234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 74), 'str', ', ')
                # Applying the binary operator '+' (line 68)
                result_add_17235 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 72), '+', result_add_17233, str_17234)
                
                # Applying the binary operator '+=' (line 68)
                result_iadd_17236 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 16), '+=', arg_str_17218, result_add_17235)
                # Assigning a type to the variable 'arg_str' (line 68)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'arg_str', result_iadd_17236)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Getting the type of 'arg_str' (line 70)
            arg_str_17237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'arg_str')
            str_17238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 30), 'str', '')
            # Applying the binary operator 'isnot' (line 70)
            result_is_not_17239 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 15), 'isnot', arg_str_17237, str_17238)
            
            # Testing if the type of an if condition is none (line 70)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 12), result_is_not_17239):
                pass
            else:
                
                # Testing the type of an if condition (line 70)
                if_condition_17240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 12), result_is_not_17239)
                # Assigning a type to the variable 'if_condition_17240' (line 70)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'if_condition_17240', if_condition_17240)
                # SSA begins for if statement (line 70)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Name (line 71):
                
                # Obtaining the type of the subscript
                int_17241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 35), 'int')
                slice_17242 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 71, 26), None, int_17241, None)
                # Getting the type of 'arg_str' (line 71)
                arg_str_17243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 26), 'arg_str')
                # Obtaining the member '__getitem__' of a type (line 71)
                getitem___17244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 26), arg_str_17243, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 71)
                subscript_call_result_17245 = invoke(stypy.reporting.localization.Localization(__file__, 71, 26), getitem___17244, slice_17242)
                
                # Assigning a type to the variable 'arg_str' (line 71)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'arg_str', subscript_call_result_17245)
                # SSA join for if statement (line 70)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 66)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 73)
        self_17246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'self')
        # Obtaining the member 'declared_varargs_var' of a type (line 73)
        declared_varargs_var_17247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 11), self_17246, 'declared_varargs_var')
        # Getting the type of 'None' (line 73)
        None_17248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 44), 'None')
        # Applying the binary operator 'isnot' (line 73)
        result_is_not_17249 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 11), 'isnot', declared_varargs_var_17247, None_17248)
        
        # Testing if the type of an if condition is none (line 73)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 73, 8), result_is_not_17249):
            pass
        else:
            
            # Testing the type of an if condition (line 73)
            if_condition_17250 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 8), result_is_not_17249)
            # Assigning a type to the variable 'if_condition_17250' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'if_condition_17250', if_condition_17250)
            # SSA begins for if statement (line 73)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'arg_str' (line 74)
            arg_str_17251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'arg_str')
            str_17252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 30), 'str', '')
            # Applying the binary operator 'isnot' (line 74)
            result_is_not_17253 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 15), 'isnot', arg_str_17251, str_17252)
            
            # Testing if the type of an if condition is none (line 74)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 12), result_is_not_17253):
                pass
            else:
                
                # Testing the type of an if condition (line 74)
                if_condition_17254 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 12), result_is_not_17253)
                # Assigning a type to the variable 'if_condition_17254' (line 74)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'if_condition_17254', if_condition_17254)
                # SSA begins for if statement (line 74)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'arg_str' (line 75)
                arg_str_17255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'arg_str')
                str_17256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 27), 'str', ', ')
                # Applying the binary operator '+=' (line 75)
                result_iadd_17257 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 16), '+=', arg_str_17255, str_17256)
                # Assigning a type to the variable 'arg_str' (line 75)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'arg_str', result_iadd_17257)
                
                # SSA join for if statement (line 74)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a BinOp to a Name (line 76):
            str_17258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 26), 'str', '*')
            
            # Call to str(...): (line 76)
            # Processing the call arguments (line 76)
            # Getting the type of 'self' (line 76)
            self_17260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 36), 'self', False)
            # Obtaining the member 'declared_varargs_var' of a type (line 76)
            declared_varargs_var_17261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 36), self_17260, 'declared_varargs_var')
            # Processing the call keyword arguments (line 76)
            kwargs_17262 = {}
            # Getting the type of 'str' (line 76)
            str_17259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 32), 'str', False)
            # Calling str(args, kwargs) (line 76)
            str_call_result_17263 = invoke(stypy.reporting.localization.Localization(__file__, 76, 32), str_17259, *[declared_varargs_var_17261], **kwargs_17262)
            
            # Applying the binary operator '+' (line 76)
            result_add_17264 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 26), '+', str_17258, str_call_result_17263)
            
            str_17265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 65), 'str', ': ')
            # Applying the binary operator '+' (line 76)
            result_add_17266 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 63), '+', result_add_17264, str_17265)
            
            
            # Call to str(...): (line 76)
            # Processing the call arguments (line 76)
            
            # Call to get_type_of(...): (line 76)
            # Processing the call arguments (line 76)
            # Getting the type of 'self' (line 76)
            self_17270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 93), 'self', False)
            # Obtaining the member 'declared_varargs_var' of a type (line 76)
            declared_varargs_var_17271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 93), self_17270, 'declared_varargs_var')
            # Processing the call keyword arguments (line 76)
            kwargs_17272 = {}
            # Getting the type of 'self' (line 76)
            self_17268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 76), 'self', False)
            # Obtaining the member 'get_type_of' of a type (line 76)
            get_type_of_17269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 76), self_17268, 'get_type_of')
            # Calling get_type_of(args, kwargs) (line 76)
            get_type_of_call_result_17273 = invoke(stypy.reporting.localization.Localization(__file__, 76, 76), get_type_of_17269, *[declared_varargs_var_17271], **kwargs_17272)
            
            # Processing the call keyword arguments (line 76)
            kwargs_17274 = {}
            # Getting the type of 'str' (line 76)
            str_17267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 72), 'str', False)
            # Calling str(args, kwargs) (line 76)
            str_call_result_17275 = invoke(stypy.reporting.localization.Localization(__file__, 76, 72), str_17267, *[get_type_of_call_result_17273], **kwargs_17274)
            
            # Applying the binary operator '+' (line 76)
            result_add_17276 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 70), '+', result_add_17266, str_call_result_17275)
            
            # Assigning a type to the variable 'str_varargs' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'str_varargs', result_add_17276)
            
            # Getting the type of 'arg_str' (line 78)
            arg_str_17277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'arg_str')
            # Getting the type of 'str_varargs' (line 78)
            str_varargs_17278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'str_varargs')
            # Applying the binary operator '+=' (line 78)
            result_iadd_17279 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 12), '+=', arg_str_17277, str_varargs_17278)
            # Assigning a type to the variable 'arg_str' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'arg_str', result_iadd_17279)
            
            # SSA join for if statement (line 73)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 80)
        self_17280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'self')
        # Obtaining the member 'declared_kwargs_var' of a type (line 80)
        declared_kwargs_var_17281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 11), self_17280, 'declared_kwargs_var')
        # Getting the type of 'None' (line 80)
        None_17282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 43), 'None')
        # Applying the binary operator 'isnot' (line 80)
        result_is_not_17283 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 11), 'isnot', declared_kwargs_var_17281, None_17282)
        
        # Testing if the type of an if condition is none (line 80)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 8), result_is_not_17283):
            pass
        else:
            
            # Testing the type of an if condition (line 80)
            if_condition_17284 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 8), result_is_not_17283)
            # Assigning a type to the variable 'if_condition_17284' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'if_condition_17284', if_condition_17284)
            # SSA begins for if statement (line 80)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'arg_str' (line 81)
            arg_str_17285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'arg_str')
            str_17286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 30), 'str', '')
            # Applying the binary operator 'isnot' (line 81)
            result_is_not_17287 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 15), 'isnot', arg_str_17285, str_17286)
            
            # Testing if the type of an if condition is none (line 81)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 81, 12), result_is_not_17287):
                pass
            else:
                
                # Testing the type of an if condition (line 81)
                if_condition_17288 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 12), result_is_not_17287)
                # Assigning a type to the variable 'if_condition_17288' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'if_condition_17288', if_condition_17288)
                # SSA begins for if statement (line 81)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'arg_str' (line 82)
                arg_str_17289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'arg_str')
                str_17290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 27), 'str', ', ')
                # Applying the binary operator '+=' (line 82)
                result_iadd_17291 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 16), '+=', arg_str_17289, str_17290)
                # Assigning a type to the variable 'arg_str' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'arg_str', result_iadd_17291)
                
                # SSA join for if statement (line 81)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a BinOp to a Name (line 83):
            str_17292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 25), 'str', '**')
            
            # Call to str(...): (line 83)
            # Processing the call arguments (line 83)
            # Getting the type of 'self' (line 83)
            self_17294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 34), 'self', False)
            # Obtaining the member 'declared_kwargs_var' of a type (line 83)
            declared_kwargs_var_17295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 34), self_17294, 'declared_kwargs_var')
            # Processing the call keyword arguments (line 83)
            kwargs_17296 = {}
            # Getting the type of 'str' (line 83)
            str_17293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 30), 'str', False)
            # Calling str(args, kwargs) (line 83)
            str_call_result_17297 = invoke(stypy.reporting.localization.Localization(__file__, 83, 30), str_17293, *[declared_kwargs_var_17295], **kwargs_17296)
            
            # Applying the binary operator '+' (line 83)
            result_add_17298 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 25), '+', str_17292, str_call_result_17297)
            
            str_17299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 62), 'str', ': ')
            # Applying the binary operator '+' (line 83)
            result_add_17300 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 60), '+', result_add_17298, str_17299)
            
            
            # Call to str(...): (line 83)
            # Processing the call arguments (line 83)
            
            # Call to get_type_of(...): (line 83)
            # Processing the call arguments (line 83)
            # Getting the type of 'self' (line 83)
            self_17304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 90), 'self', False)
            # Obtaining the member 'declared_kwargs_var' of a type (line 83)
            declared_kwargs_var_17305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 90), self_17304, 'declared_kwargs_var')
            # Processing the call keyword arguments (line 83)
            kwargs_17306 = {}
            # Getting the type of 'self' (line 83)
            self_17302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 73), 'self', False)
            # Obtaining the member 'get_type_of' of a type (line 83)
            get_type_of_17303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 73), self_17302, 'get_type_of')
            # Calling get_type_of(args, kwargs) (line 83)
            get_type_of_call_result_17307 = invoke(stypy.reporting.localization.Localization(__file__, 83, 73), get_type_of_17303, *[declared_kwargs_var_17305], **kwargs_17306)
            
            # Processing the call keyword arguments (line 83)
            kwargs_17308 = {}
            # Getting the type of 'str' (line 83)
            str_17301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 69), 'str', False)
            # Calling str(args, kwargs) (line 83)
            str_call_result_17309 = invoke(stypy.reporting.localization.Localization(__file__, 83, 69), str_17301, *[get_type_of_call_result_17307], **kwargs_17308)
            
            # Applying the binary operator '+' (line 83)
            result_add_17310 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 67), '+', result_add_17300, str_call_result_17309)
            
            # Assigning a type to the variable 'str_kwargs' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'str_kwargs', result_add_17310)
            
            # Getting the type of 'arg_str' (line 85)
            arg_str_17311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'arg_str')
            # Getting the type of 'str_kwargs' (line 85)
            str_kwargs_17312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'str_kwargs')
            # Applying the binary operator '+=' (line 85)
            result_iadd_17313 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 12), '+=', arg_str_17311, str_kwargs_17312)
            # Assigning a type to the variable 'arg_str' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'arg_str', result_iadd_17313)
            
            # SSA join for if statement (line 80)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'txt' (line 87)
        txt_17314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'txt')
        
        # Call to str(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'self' (line 87)
        self_17316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'self', False)
        # Obtaining the member 'function_name' of a type (line 87)
        function_name_17317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 19), self_17316, 'function_name')
        # Processing the call keyword arguments (line 87)
        kwargs_17318 = {}
        # Getting the type of 'str' (line 87)
        str_17315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'str', False)
        # Calling str(args, kwargs) (line 87)
        str_call_result_17319 = invoke(stypy.reporting.localization.Localization(__file__, 87, 15), str_17315, *[function_name_17317], **kwargs_17318)
        
        str_17320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 41), 'str', '(')
        # Applying the binary operator '+' (line 87)
        result_add_17321 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 15), '+', str_call_result_17319, str_17320)
        
        # Getting the type of 'arg_str' (line 87)
        arg_str_17322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 47), 'arg_str')
        # Applying the binary operator '+' (line 87)
        result_add_17323 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 45), '+', result_add_17321, arg_str_17322)
        
        str_17324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 57), 'str', ') -> ')
        # Applying the binary operator '+' (line 87)
        result_add_17325 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 55), '+', result_add_17323, str_17324)
        
        
        # Call to get_type_str(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'self' (line 87)
        self_17328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 97), 'self', False)
        # Obtaining the member 'return_type' of a type (line 87)
        return_type_17329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 97), self_17328, 'return_type')
        # Processing the call keyword arguments (line 87)
        kwargs_17330 = {}
        # Getting the type of 'print_utils_copy' (line 87)
        print_utils_copy_17326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 67), 'print_utils_copy', False)
        # Obtaining the member 'get_type_str' of a type (line 87)
        get_type_str_17327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 67), print_utils_copy_17326, 'get_type_str')
        # Calling get_type_str(args, kwargs) (line 87)
        get_type_str_call_result_17331 = invoke(stypy.reporting.localization.Localization(__file__, 87, 67), get_type_str_17327, *[return_type_17329], **kwargs_17330)
        
        # Applying the binary operator '+' (line 87)
        result_add_17332 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 65), '+', result_add_17325, get_type_str_call_result_17331)
        
        # Applying the binary operator '+=' (line 87)
        result_iadd_17333 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 8), '+=', txt_17314, result_add_17332)
        # Assigning a type to the variable 'txt' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'txt', result_iadd_17333)
        
        # Getting the type of 'txt' (line 89)
        txt_17334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'txt')
        # Assigning a type to the variable 'stypy_return_type' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'stypy_return_type', txt_17334)
        
        # ################# End of 'get_header_str(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_header_str' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_17335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17335)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_header_str'
        return stypy_return_type_17335


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

        str_17336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'str', '\n        String representation of the function context\n        :return: str\n        ')
        
        # Assigning a Str to a Name (line 96):
        str_17337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 14), 'str', '')
        # Assigning a type to the variable 'txt' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'txt', str_17337)
        # Getting the type of 'self' (line 97)
        self_17338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'self')
        # Obtaining the member 'is_main_context' of a type (line 97)
        is_main_context_17339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 11), self_17338, 'is_main_context')
        # Testing if the type of an if condition is none (line 97)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 97, 8), is_main_context_17339):
            
            # Getting the type of 'self' (line 100)
            self_17352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'self')
            # Obtaining the member 'declaration_line' of a type (line 100)
            declaration_line_17353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), self_17352, 'declaration_line')
            int_17354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 44), 'int')
            # Applying the binary operator 'isnot' (line 100)
            result_is_not_17355 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), 'isnot', declaration_line_17353, int_17354)
            
            # Testing if the type of an if condition is none (line 100)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 100, 12), result_is_not_17355):
                pass
            else:
                
                # Testing the type of an if condition (line 100)
                if_condition_17356 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 12), result_is_not_17355)
                # Assigning a type to the variable 'if_condition_17356' (line 100)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'if_condition_17356', if_condition_17356)
                # SSA begins for if statement (line 100)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 101):
                
                # Call to get_header_str(...): (line 101)
                # Processing the call keyword arguments (line 101)
                kwargs_17359 = {}
                # Getting the type of 'self' (line 101)
                self_17357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'self', False)
                # Obtaining the member 'get_header_str' of a type (line 101)
                get_header_str_17358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 22), self_17357, 'get_header_str')
                # Calling get_header_str(args, kwargs) (line 101)
                get_header_str_call_result_17360 = invoke(stypy.reporting.localization.Localization(__file__, 101, 22), get_header_str_17358, *[], **kwargs_17359)
                
                # Assigning a type to the variable 'txt' (line 101)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'txt', get_header_str_call_result_17360)
                
                # Getting the type of 'txt' (line 102)
                txt_17361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'txt')
                str_17362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 23), 'str', ' (Line: ')
                
                # Call to str(...): (line 102)
                # Processing the call arguments (line 102)
                # Getting the type of 'self' (line 102)
                self_17364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 40), 'self', False)
                # Obtaining the member 'declaration_line' of a type (line 102)
                declaration_line_17365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 40), self_17364, 'declaration_line')
                # Processing the call keyword arguments (line 102)
                kwargs_17366 = {}
                # Getting the type of 'str' (line 102)
                str_17363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 36), 'str', False)
                # Calling str(args, kwargs) (line 102)
                str_call_result_17367 = invoke(stypy.reporting.localization.Localization(__file__, 102, 36), str_17363, *[declaration_line_17365], **kwargs_17366)
                
                # Applying the binary operator '+' (line 102)
                result_add_17368 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 23), '+', str_17362, str_call_result_17367)
                
                str_17369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 65), 'str', ', Column: ')
                # Applying the binary operator '+' (line 102)
                result_add_17370 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 63), '+', result_add_17368, str_17369)
                
                
                # Call to str(...): (line 102)
                # Processing the call arguments (line 102)
                # Getting the type of 'self' (line 102)
                self_17372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 84), 'self', False)
                # Obtaining the member 'declaration_column' of a type (line 102)
                declaration_column_17373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 84), self_17372, 'declaration_column')
                # Processing the call keyword arguments (line 102)
                kwargs_17374 = {}
                # Getting the type of 'str' (line 102)
                str_17371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 80), 'str', False)
                # Calling str(args, kwargs) (line 102)
                str_call_result_17375 = invoke(stypy.reporting.localization.Localization(__file__, 102, 80), str_17371, *[declaration_column_17373], **kwargs_17374)
                
                # Applying the binary operator '+' (line 102)
                result_add_17376 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 78), '+', result_add_17370, str_call_result_17375)
                
                str_17377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 111), 'str', ')\n')
                # Applying the binary operator '+' (line 102)
                result_add_17378 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 109), '+', result_add_17376, str_17377)
                
                # Applying the binary operator '+=' (line 102)
                result_iadd_17379 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 16), '+=', txt_17361, result_add_17378)
                # Assigning a type to the variable 'txt' (line 102)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'txt', result_iadd_17379)
                
                # SSA join for if statement (line 100)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 97)
            if_condition_17340 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), is_main_context_17339)
            # Assigning a type to the variable 'if_condition_17340' (line 97)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_17340', if_condition_17340)
            # SSA begins for if statement (line 97)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'txt' (line 98)
            txt_17341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'txt')
            str_17342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 19), 'str', "Program '")
            
            # Call to str(...): (line 98)
            # Processing the call arguments (line 98)
            # Getting the type of 'self' (line 98)
            self_17344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 37), 'self', False)
            # Obtaining the member 'function_name' of a type (line 98)
            function_name_17345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 37), self_17344, 'function_name')
            # Processing the call keyword arguments (line 98)
            kwargs_17346 = {}
            # Getting the type of 'str' (line 98)
            str_17343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'str', False)
            # Calling str(args, kwargs) (line 98)
            str_call_result_17347 = invoke(stypy.reporting.localization.Localization(__file__, 98, 33), str_17343, *[function_name_17345], **kwargs_17346)
            
            # Applying the binary operator '+' (line 98)
            result_add_17348 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 19), '+', str_17342, str_call_result_17347)
            
            str_17349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 59), 'str', "'\n")
            # Applying the binary operator '+' (line 98)
            result_add_17350 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 57), '+', result_add_17348, str_17349)
            
            # Applying the binary operator '+=' (line 98)
            result_iadd_17351 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 12), '+=', txt_17341, result_add_17350)
            # Assigning a type to the variable 'txt' (line 98)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'txt', result_iadd_17351)
            
            # SSA branch for the else part of an if statement (line 97)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'self' (line 100)
            self_17352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'self')
            # Obtaining the member 'declaration_line' of a type (line 100)
            declaration_line_17353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 15), self_17352, 'declaration_line')
            int_17354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 44), 'int')
            # Applying the binary operator 'isnot' (line 100)
            result_is_not_17355 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), 'isnot', declaration_line_17353, int_17354)
            
            # Testing if the type of an if condition is none (line 100)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 100, 12), result_is_not_17355):
                pass
            else:
                
                # Testing the type of an if condition (line 100)
                if_condition_17356 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 12), result_is_not_17355)
                # Assigning a type to the variable 'if_condition_17356' (line 100)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'if_condition_17356', if_condition_17356)
                # SSA begins for if statement (line 100)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 101):
                
                # Call to get_header_str(...): (line 101)
                # Processing the call keyword arguments (line 101)
                kwargs_17359 = {}
                # Getting the type of 'self' (line 101)
                self_17357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'self', False)
                # Obtaining the member 'get_header_str' of a type (line 101)
                get_header_str_17358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 22), self_17357, 'get_header_str')
                # Calling get_header_str(args, kwargs) (line 101)
                get_header_str_call_result_17360 = invoke(stypy.reporting.localization.Localization(__file__, 101, 22), get_header_str_17358, *[], **kwargs_17359)
                
                # Assigning a type to the variable 'txt' (line 101)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'txt', get_header_str_call_result_17360)
                
                # Getting the type of 'txt' (line 102)
                txt_17361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'txt')
                str_17362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 23), 'str', ' (Line: ')
                
                # Call to str(...): (line 102)
                # Processing the call arguments (line 102)
                # Getting the type of 'self' (line 102)
                self_17364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 40), 'self', False)
                # Obtaining the member 'declaration_line' of a type (line 102)
                declaration_line_17365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 40), self_17364, 'declaration_line')
                # Processing the call keyword arguments (line 102)
                kwargs_17366 = {}
                # Getting the type of 'str' (line 102)
                str_17363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 36), 'str', False)
                # Calling str(args, kwargs) (line 102)
                str_call_result_17367 = invoke(stypy.reporting.localization.Localization(__file__, 102, 36), str_17363, *[declaration_line_17365], **kwargs_17366)
                
                # Applying the binary operator '+' (line 102)
                result_add_17368 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 23), '+', str_17362, str_call_result_17367)
                
                str_17369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 65), 'str', ', Column: ')
                # Applying the binary operator '+' (line 102)
                result_add_17370 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 63), '+', result_add_17368, str_17369)
                
                
                # Call to str(...): (line 102)
                # Processing the call arguments (line 102)
                # Getting the type of 'self' (line 102)
                self_17372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 84), 'self', False)
                # Obtaining the member 'declaration_column' of a type (line 102)
                declaration_column_17373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 84), self_17372, 'declaration_column')
                # Processing the call keyword arguments (line 102)
                kwargs_17374 = {}
                # Getting the type of 'str' (line 102)
                str_17371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 80), 'str', False)
                # Calling str(args, kwargs) (line 102)
                str_call_result_17375 = invoke(stypy.reporting.localization.Localization(__file__, 102, 80), str_17371, *[declaration_column_17373], **kwargs_17374)
                
                # Applying the binary operator '+' (line 102)
                result_add_17376 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 78), '+', result_add_17370, str_call_result_17375)
                
                str_17377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 111), 'str', ')\n')
                # Applying the binary operator '+' (line 102)
                result_add_17378 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 109), '+', result_add_17376, str_17377)
                
                # Applying the binary operator '+=' (line 102)
                result_iadd_17379 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 16), '+=', txt_17361, result_add_17378)
                # Assigning a type to the variable 'txt' (line 102)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'txt', result_iadd_17379)
                
                # SSA join for if statement (line 100)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 97)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 104)
        self_17380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'self')
        # Obtaining the member 'types_of' of a type (line 104)
        types_of_17381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 20), self_17380, 'types_of')
        # Assigning a type to the variable 'types_of_17381' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'types_of_17381', types_of_17381)
        # Testing if the for loop is going to be iterated (line 104)
        # Testing the type of a for loop iterable (line 104)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 104, 8), types_of_17381)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 104, 8), types_of_17381):
            # Getting the type of the for loop variable (line 104)
            for_loop_var_17382 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 104, 8), types_of_17381)
            # Assigning a type to the variable 'name' (line 104)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'name', for_loop_var_17382)
            # SSA begins for a for statement (line 104)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 105):
            
            # Obtaining the type of the subscript
            # Getting the type of 'name' (line 105)
            name_17383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 34), 'name')
            # Getting the type of 'self' (line 105)
            self_17384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 20), 'self')
            # Obtaining the member 'types_of' of a type (line 105)
            types_of_17385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 20), self_17384, 'types_of')
            # Obtaining the member '__getitem__' of a type (line 105)
            getitem___17386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 20), types_of_17385, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 105)
            subscript_call_result_17387 = invoke(stypy.reporting.localization.Localization(__file__, 105, 20), getitem___17386, name_17383)
            
            # Assigning a type to the variable 'type_' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'type_', subscript_call_result_17387)
            
            # Type idiom detected: calculating its left and rigth part (line 106)
            # Getting the type of 'TypeError' (line 106)
            TypeError_17388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'TypeError')
            # Getting the type of 'type_' (line 106)
            type__17389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'type_')
            
            (may_be_17390, more_types_in_union_17391) = may_be_subtype(TypeError_17388, type__17389)

            if may_be_17390:

                if more_types_in_union_17391:
                    # Runtime conditional SSA (line 106)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'type_' (line 106)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'type_', remove_not_subtype_from_union(type__17389, TypeError))
                
                # Getting the type of 'txt' (line 107)
                txt_17392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'txt')
                str_17393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 23), 'str', '\t')
                # Getting the type of 'name' (line 107)
                name_17394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 30), 'name')
                # Applying the binary operator '+' (line 107)
                result_add_17395 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 23), '+', str_17393, name_17394)
                
                str_17396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 37), 'str', ': TypeError\n')
                # Applying the binary operator '+' (line 107)
                result_add_17397 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 35), '+', result_add_17395, str_17396)
                
                # Applying the binary operator '+=' (line 107)
                result_iadd_17398 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 16), '+=', txt_17392, result_add_17397)
                # Assigning a type to the variable 'txt' (line 107)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'txt', result_iadd_17398)
                

                if more_types_in_union_17391:
                    # Runtime conditional SSA for else branch (line 106)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_17390) or more_types_in_union_17391):
                # Assigning a type to the variable 'type_' (line 106)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'type_', remove_subtype_from_union(type__17389, TypeError))
                
                # Getting the type of 'txt' (line 109)
                txt_17399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'txt')
                str_17400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 23), 'str', '\t')
                # Getting the type of 'name' (line 109)
                name_17401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 30), 'name')
                # Applying the binary operator '+' (line 109)
                result_add_17402 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 23), '+', str_17400, name_17401)
                
                str_17403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 37), 'str', ': ')
                # Applying the binary operator '+' (line 109)
                result_add_17404 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 35), '+', result_add_17402, str_17403)
                
                
                # Call to str(...): (line 109)
                # Processing the call arguments (line 109)
                # Getting the type of 'type_' (line 109)
                type__17406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 48), 'type_', False)
                # Processing the call keyword arguments (line 109)
                kwargs_17407 = {}
                # Getting the type of 'str' (line 109)
                str_17405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 44), 'str', False)
                # Calling str(args, kwargs) (line 109)
                str_call_result_17408 = invoke(stypy.reporting.localization.Localization(__file__, 109, 44), str_17405, *[type__17406], **kwargs_17407)
                
                # Applying the binary operator '+' (line 109)
                result_add_17409 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 42), '+', result_add_17404, str_call_result_17408)
                
                str_17410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 57), 'str', '\n')
                # Applying the binary operator '+' (line 109)
                result_add_17411 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 55), '+', result_add_17409, str_17410)
                
                # Applying the binary operator '+=' (line 109)
                result_iadd_17412 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 16), '+=', txt_17399, result_add_17411)
                # Assigning a type to the variable 'txt' (line 109)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'txt', result_iadd_17412)
                

                if (may_be_17390 and more_types_in_union_17391):
                    # SSA join for if statement (line 106)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'txt' (line 111)
        txt_17413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'txt')
        # Assigning a type to the variable 'stypy_return_type' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'stypy_return_type', txt_17413)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_17414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17414)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_17414


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

        str_17415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, (-1)), 'str', '\n        String representation of the function context\n        :return: str\n        ')
        
        # Call to __repr__(...): (line 118)
        # Processing the call keyword arguments (line 118)
        kwargs_17418 = {}
        # Getting the type of 'self' (line 118)
        self_17416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'self', False)
        # Obtaining the member '__repr__' of a type (line 118)
        repr___17417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 15), self_17416, '__repr__')
        # Calling __repr__(args, kwargs) (line 118)
        repr___call_result_17419 = invoke(stypy.reporting.localization.Localization(__file__, 118, 15), repr___17417, *[], **kwargs_17418)
        
        # Assigning a type to the variable 'stypy_return_type' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'stypy_return_type', repr___call_result_17419)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 113)
        stypy_return_type_17420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17420)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_17420


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

        str_17421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, (-1)), 'str', '\n        in operator, to determine if the function context contains a local variable\n        :param item:\n        :return:\n        ')
        
        # Getting the type of 'item' (line 126)
        item_17422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'item')
        
        # Call to keys(...): (line 126)
        # Processing the call keyword arguments (line 126)
        kwargs_17426 = {}
        # Getting the type of 'self' (line 126)
        self_17423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'self', False)
        # Obtaining the member 'types_of' of a type (line 126)
        types_of_17424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), self_17423, 'types_of')
        # Obtaining the member 'keys' of a type (line 126)
        keys_17425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), types_of_17424, 'keys')
        # Calling keys(args, kwargs) (line 126)
        keys_call_result_17427 = invoke(stypy.reporting.localization.Localization(__file__, 126, 23), keys_17425, *[], **kwargs_17426)
        
        # Applying the binary operator 'in' (line 126)
        result_contains_17428 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 15), 'in', item_17422, keys_call_result_17427)
        
        # Assigning a type to the variable 'stypy_return_type' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type', result_contains_17428)
        
        # ################# End of '__contains__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__contains__' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_17429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17429)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__contains__'
        return stypy_return_type_17429


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

        str_17430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, (-1)), 'str', '\n        Adds an alias to the alias storage of this function context\n        :param alias_name: Name of the alias\n        :param variable_name: Name of the aliased variable\n        :return:\n        ')
        
        # Assigning a Name to a Subscript (line 135):
        # Getting the type of 'variable_name' (line 135)
        variable_name_17431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 35), 'variable_name')
        # Getting the type of 'self' (line 135)
        self_17432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'self')
        # Obtaining the member 'aliases' of a type (line 135)
        aliases_17433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), self_17432, 'aliases')
        # Getting the type of 'alias_name' (line 135)
        alias_name_17434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 21), 'alias_name')
        # Storing an element on a container (line 135)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 8), aliases_17433, (alias_name_17434, variable_name_17431))
        
        # ################# End of 'add_alias(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_alias' in the type store
        # Getting the type of 'stypy_return_type' (line 128)
        stypy_return_type_17435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17435)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_alias'
        return stypy_return_type_17435


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

        str_17436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, (-1)), 'str', '\n        Returns the type of a variable or parameter in the local context\n        :param variable_name: Name of the variable in the context\n        :return: The variable type or None if the variable do not belong to this context locally\n        ')
        
        # Getting the type of 'variable_name' (line 143)
        variable_name_17437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 11), 'variable_name')
        
        # Call to keys(...): (line 143)
        # Processing the call keyword arguments (line 143)
        kwargs_17441 = {}
        # Getting the type of 'self' (line 143)
        self_17438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 28), 'self', False)
        # Obtaining the member 'aliases' of a type (line 143)
        aliases_17439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 28), self_17438, 'aliases')
        # Obtaining the member 'keys' of a type (line 143)
        keys_17440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 28), aliases_17439, 'keys')
        # Calling keys(args, kwargs) (line 143)
        keys_call_result_17442 = invoke(stypy.reporting.localization.Localization(__file__, 143, 28), keys_17440, *[], **kwargs_17441)
        
        # Applying the binary operator 'in' (line 143)
        result_contains_17443 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 11), 'in', variable_name_17437, keys_call_result_17442)
        
        # Testing if the type of an if condition is none (line 143)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 143, 8), result_contains_17443):
            pass
        else:
            
            # Testing the type of an if condition (line 143)
            if_condition_17444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 8), result_contains_17443)
            # Assigning a type to the variable 'if_condition_17444' (line 143)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'if_condition_17444', if_condition_17444)
            # SSA begins for if statement (line 143)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 144):
            
            # Obtaining the type of the subscript
            # Getting the type of 'variable_name' (line 144)
            variable_name_17445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 41), 'variable_name')
            # Getting the type of 'self' (line 144)
            self_17446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 28), 'self')
            # Obtaining the member 'aliases' of a type (line 144)
            aliases_17447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 28), self_17446, 'aliases')
            # Obtaining the member '__getitem__' of a type (line 144)
            getitem___17448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 28), aliases_17447, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 144)
            subscript_call_result_17449 = invoke(stypy.reporting.localization.Localization(__file__, 144, 28), getitem___17448, variable_name_17445)
            
            # Assigning a type to the variable 'variable_name' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'variable_name', subscript_call_result_17449)
            # SSA join for if statement (line 143)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'variable_name' (line 146)
        variable_name_17450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'variable_name')
        # Getting the type of 'self' (line 146)
        self_17451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 28), 'self')
        # Obtaining the member 'types_of' of a type (line 146)
        types_of_17452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 28), self_17451, 'types_of')
        # Applying the binary operator 'in' (line 146)
        result_contains_17453 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 11), 'in', variable_name_17450, types_of_17452)
        
        # Testing if the type of an if condition is none (line 146)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 146, 8), result_contains_17453):
            pass
        else:
            
            # Testing the type of an if condition (line 146)
            if_condition_17454 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 8), result_contains_17453)
            # Assigning a type to the variable 'if_condition_17454' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'if_condition_17454', if_condition_17454)
            # SSA begins for if statement (line 146)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            # Getting the type of 'variable_name' (line 147)
            variable_name_17455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 33), 'variable_name')
            # Getting the type of 'self' (line 147)
            self_17456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 19), 'self')
            # Obtaining the member 'types_of' of a type (line 147)
            types_of_17457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 19), self_17456, 'types_of')
            # Obtaining the member '__getitem__' of a type (line 147)
            getitem___17458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 19), types_of_17457, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 147)
            subscript_call_result_17459 = invoke(stypy.reporting.localization.Localization(__file__, 147, 19), getitem___17458, variable_name_17455)
            
            # Assigning a type to the variable 'stypy_return_type' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'stypy_return_type', subscript_call_result_17459)
            # SSA join for if statement (line 146)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 149)
        None_17460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type', None_17460)
        
        # ################# End of 'get_type_of(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_type_of' in the type store
        # Getting the type of 'stypy_return_type' (line 137)
        stypy_return_type_17461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17461)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_type_of'
        return stypy_return_type_17461


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

        str_17462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, (-1)), 'str', '\n        Sets the type of name to type in this local context\n        :param name: Name to search\n        :param type: Type to assign to name\n        ')
        # Getting the type of 'self' (line 157)
        self_17463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 11), 'self')
        # Obtaining the member 'annotate_types' of a type (line 157)
        annotate_types_17464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 11), self_17463, 'annotate_types')
        # Testing if the type of an if condition is none (line 157)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 157, 8), annotate_types_17464):
            pass
        else:
            
            # Testing the type of an if condition (line 157)
            if_condition_17465 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 8), annotate_types_17464)
            # Assigning a type to the variable 'if_condition_17465' (line 157)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'if_condition_17465', if_condition_17465)
            # SSA begins for if statement (line 157)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to annotate_type(...): (line 158)
            # Processing the call arguments (line 158)
            # Getting the type of 'localization' (line 158)
            localization_17469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 49), 'localization', False)
            # Obtaining the member 'line' of a type (line 158)
            line_17470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 49), localization_17469, 'line')
            # Getting the type of 'localization' (line 158)
            localization_17471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 68), 'localization', False)
            # Obtaining the member 'column' of a type (line 158)
            column_17472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 68), localization_17471, 'column')
            # Getting the type of 'name' (line 158)
            name_17473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 89), 'name', False)
            # Getting the type of 'type_' (line 158)
            type__17474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 95), 'type_', False)
            # Processing the call keyword arguments (line 158)
            kwargs_17475 = {}
            # Getting the type of 'self' (line 158)
            self_17466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 12), 'self', False)
            # Obtaining the member 'annotation_record' of a type (line 158)
            annotation_record_17467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 12), self_17466, 'annotation_record')
            # Obtaining the member 'annotate_type' of a type (line 158)
            annotate_type_17468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 12), annotation_record_17467, 'annotate_type')
            # Calling annotate_type(args, kwargs) (line 158)
            annotate_type_call_result_17476 = invoke(stypy.reporting.localization.Localization(__file__, 158, 12), annotate_type_17468, *[line_17470, column_17472, name_17473, type__17474], **kwargs_17475)
            
            # SSA join for if statement (line 157)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'name' (line 160)
        name_17477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'name')
        
        # Call to keys(...): (line 160)
        # Processing the call keyword arguments (line 160)
        kwargs_17481 = {}
        # Getting the type of 'self' (line 160)
        self_17478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 19), 'self', False)
        # Obtaining the member 'aliases' of a type (line 160)
        aliases_17479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 19), self_17478, 'aliases')
        # Obtaining the member 'keys' of a type (line 160)
        keys_17480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 19), aliases_17479, 'keys')
        # Calling keys(args, kwargs) (line 160)
        keys_call_result_17482 = invoke(stypy.reporting.localization.Localization(__file__, 160, 19), keys_17480, *[], **kwargs_17481)
        
        # Applying the binary operator 'in' (line 160)
        result_contains_17483 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 11), 'in', name_17477, keys_call_result_17482)
        
        # Testing if the type of an if condition is none (line 160)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 160, 8), result_contains_17483):
            pass
        else:
            
            # Testing the type of an if condition (line 160)
            if_condition_17484 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 8), result_contains_17483)
            # Assigning a type to the variable 'if_condition_17484' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'if_condition_17484', if_condition_17484)
            # SSA begins for if statement (line 160)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 161):
            
            # Obtaining the type of the subscript
            # Getting the type of 'name' (line 161)
            name_17485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 32), 'name')
            # Getting the type of 'self' (line 161)
            self_17486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 19), 'self')
            # Obtaining the member 'aliases' of a type (line 161)
            aliases_17487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 19), self_17486, 'aliases')
            # Obtaining the member '__getitem__' of a type (line 161)
            getitem___17488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 19), aliases_17487, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 161)
            subscript_call_result_17489 = invoke(stypy.reporting.localization.Localization(__file__, 161, 19), getitem___17488, name_17485)
            
            # Assigning a type to the variable 'name' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'name', subscript_call_result_17489)
            # SSA join for if statement (line 160)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Subscript (line 162):
        # Getting the type of 'type_' (line 162)
        type__17490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 30), 'type_')
        # Getting the type of 'self' (line 162)
        self_17491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'self')
        # Obtaining the member 'types_of' of a type (line 162)
        types_of_17492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), self_17491, 'types_of')
        # Getting the type of 'name' (line 162)
        name_17493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 22), 'name')
        # Storing an element on a container (line 162)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 8), types_of_17492, (name_17493, type__17490))
        
        # ################# End of 'set_type_of(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_type_of' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_17494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17494)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_type_of'
        return stypy_return_type_17494


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

        str_17495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, (-1)), 'str', '\n        Deletes the type of a variable or parameter in the local context\n        :param variable_name: Name of the variable in the context\n        ')
        
        # Getting the type of 'variable_name' (line 169)
        variable_name_17496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 11), 'variable_name')
        # Getting the type of 'self' (line 169)
        self_17497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 28), 'self')
        # Obtaining the member 'types_of' of a type (line 169)
        types_of_17498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 28), self_17497, 'types_of')
        # Applying the binary operator 'in' (line 169)
        result_contains_17499 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 11), 'in', variable_name_17496, types_of_17498)
        
        # Testing if the type of an if condition is none (line 169)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 169, 8), result_contains_17499):
            pass
        else:
            
            # Testing the type of an if condition (line 169)
            if_condition_17500 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 8), result_contains_17499)
            # Assigning a type to the variable 'if_condition_17500' (line 169)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'if_condition_17500', if_condition_17500)
            # SSA begins for if statement (line 169)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Deleting a member
            # Getting the type of 'self' (line 170)
            self_17501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'self')
            # Obtaining the member 'types_of' of a type (line 170)
            types_of_17502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 16), self_17501, 'types_of')
            
            # Obtaining the type of the subscript
            # Getting the type of 'variable_name' (line 170)
            variable_name_17503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 30), 'variable_name')
            # Getting the type of 'self' (line 170)
            self_17504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'self')
            # Obtaining the member 'types_of' of a type (line 170)
            types_of_17505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 16), self_17504, 'types_of')
            # Obtaining the member '__getitem__' of a type (line 170)
            getitem___17506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 16), types_of_17505, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 170)
            subscript_call_result_17507 = invoke(stypy.reporting.localization.Localization(__file__, 170, 16), getitem___17506, variable_name_17503)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 12), types_of_17502, subscript_call_result_17507)
            # SSA join for if statement (line 169)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'None' (line 172)
        None_17508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'stypy_return_type', None_17508)
        
        # ################# End of 'del_type_of(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'del_type_of' in the type store
        # Getting the type of 'stypy_return_type' (line 164)
        stypy_return_type_17509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17509)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'del_type_of'
        return stypy_return_type_17509


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

        str_17510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, (-1)), 'str', '\n        Allows iteration through all the variable names stored in the context.\n        :return: Each variable name stored in the context\n        ')
        
        # Getting the type of 'self' (line 179)
        self_17511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 29), 'self')
        # Obtaining the member 'types_of' of a type (line 179)
        types_of_17512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 29), self_17511, 'types_of')
        # Assigning a type to the variable 'types_of_17512' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'types_of_17512', types_of_17512)
        # Testing if the for loop is going to be iterated (line 179)
        # Testing the type of a for loop iterable (line 179)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 179, 8), types_of_17512)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 179, 8), types_of_17512):
            # Getting the type of the for loop variable (line 179)
            for_loop_var_17513 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 179, 8), types_of_17512)
            # Assigning a type to the variable 'variable_name' (line 179)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'variable_name', for_loop_var_17513)
            # SSA begins for a for statement (line 179)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Creating a generator
            # Getting the type of 'variable_name' (line 180)
            variable_name_17514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 18), 'variable_name')
            GeneratorType_17515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 12), 'GeneratorType')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 12), GeneratorType_17515, variable_name_17514)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'stypy_return_type', GeneratorType_17515)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_17516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17516)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_17516


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

        str_17517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, (-1)), 'str', '\n        Allows the usage of the [] operator to access variable types by variable name\n        :param item: Variable name\n        :return: Same as get_type_of\n        ')
        
        # Call to get_type_of(...): (line 188)
        # Processing the call arguments (line 188)
        # Getting the type of 'item' (line 188)
        item_17520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 32), 'item', False)
        # Processing the call keyword arguments (line 188)
        kwargs_17521 = {}
        # Getting the type of 'self' (line 188)
        self_17518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'self', False)
        # Obtaining the member 'get_type_of' of a type (line 188)
        get_type_of_17519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 15), self_17518, 'get_type_of')
        # Calling get_type_of(args, kwargs) (line 188)
        get_type_of_call_result_17522 = invoke(stypy.reporting.localization.Localization(__file__, 188, 15), get_type_of_17519, *[item_17520], **kwargs_17521)
        
        # Assigning a type to the variable 'stypy_return_type' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'stypy_return_type', get_type_of_call_result_17522)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 182)
        stypy_return_type_17523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17523)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_17523


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

        str_17524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, (-1)), 'str', '\n        len operator, returning the amount of stored local variables\n        :return:\n        ')
        
        # Call to len(...): (line 195)
        # Processing the call arguments (line 195)
        # Getting the type of 'self' (line 195)
        self_17526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 19), 'self', False)
        # Obtaining the member 'types_of' of a type (line 195)
        types_of_17527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 19), self_17526, 'types_of')
        # Processing the call keyword arguments (line 195)
        kwargs_17528 = {}
        # Getting the type of 'len' (line 195)
        len_17525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'len', False)
        # Calling len(args, kwargs) (line 195)
        len_call_result_17529 = invoke(stypy.reporting.localization.Localization(__file__, 195, 15), len_17525, *[types_of_17527], **kwargs_17528)
        
        # Assigning a type to the variable 'stypy_return_type' (line 195)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'stypy_return_type', len_call_result_17529)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 190)
        stypy_return_type_17530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17530)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_17530


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

        str_17531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, (-1)), 'str', '\n        Clones the whole function context. The returned function context is a deep copy of the current one\n        :return: Cloned function context\n        ')
        
        # Assigning a Call to a Name (line 202):
        
        # Call to FunctionContext(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'self' (line 202)
        self_17533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 37), 'self', False)
        # Obtaining the member 'function_name' of a type (line 202)
        function_name_17534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 37), self_17533, 'function_name')
        # Processing the call keyword arguments (line 202)
        kwargs_17535 = {}
        # Getting the type of 'FunctionContext' (line 202)
        FunctionContext_17532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 21), 'FunctionContext', False)
        # Calling FunctionContext(args, kwargs) (line 202)
        FunctionContext_call_result_17536 = invoke(stypy.reporting.localization.Localization(__file__, 202, 21), FunctionContext_17532, *[function_name_17534], **kwargs_17535)
        
        # Assigning a type to the variable 'cloned_obj' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'cloned_obj', FunctionContext_call_result_17536)
        
        # Assigning a Call to a Attribute (line 204):
        
        # Call to deepcopy(...): (line 204)
        # Processing the call arguments (line 204)
        # Getting the type of 'self' (line 204)
        self_17539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 47), 'self', False)
        # Obtaining the member 'global_vars' of a type (line 204)
        global_vars_17540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 47), self_17539, 'global_vars')
        # Processing the call keyword arguments (line 204)
        kwargs_17541 = {}
        # Getting the type of 'copy' (line 204)
        copy_17537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 33), 'copy', False)
        # Obtaining the member 'deepcopy' of a type (line 204)
        deepcopy_17538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 33), copy_17537, 'deepcopy')
        # Calling deepcopy(args, kwargs) (line 204)
        deepcopy_call_result_17542 = invoke(stypy.reporting.localization.Localization(__file__, 204, 33), deepcopy_17538, *[global_vars_17540], **kwargs_17541)
        
        # Getting the type of 'cloned_obj' (line 204)
        cloned_obj_17543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'cloned_obj')
        # Setting the type of the member 'global_vars' of a type (line 204)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), cloned_obj_17543, 'global_vars', deepcopy_call_result_17542)
        
        
        # Call to iteritems(...): (line 206)
        # Processing the call keyword arguments (line 206)
        kwargs_17547 = {}
        # Getting the type of 'self' (line 206)
        self_17544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'self', False)
        # Obtaining the member 'types_of' of a type (line 206)
        types_of_17545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 26), self_17544, 'types_of')
        # Obtaining the member 'iteritems' of a type (line 206)
        iteritems_17546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 26), types_of_17545, 'iteritems')
        # Calling iteritems(args, kwargs) (line 206)
        iteritems_call_result_17548 = invoke(stypy.reporting.localization.Localization(__file__, 206, 26), iteritems_17546, *[], **kwargs_17547)
        
        # Assigning a type to the variable 'iteritems_call_result_17548' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'iteritems_call_result_17548', iteritems_call_result_17548)
        # Testing if the for loop is going to be iterated (line 206)
        # Testing the type of a for loop iterable (line 206)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 206, 8), iteritems_call_result_17548)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 206, 8), iteritems_call_result_17548):
            # Getting the type of the for loop variable (line 206)
            for_loop_var_17549 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 206, 8), iteritems_call_result_17548)
            # Assigning a type to the variable 'key' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 8), for_loop_var_17549, 2, 0))
            # Assigning a type to the variable 'value' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 206, 8), for_loop_var_17549, 2, 1))
            # SSA begins for a for statement (line 206)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to isinstance(...): (line 207)
            # Processing the call arguments (line 207)
            # Getting the type of 'value' (line 207)
            value_17551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 26), 'value', False)
            # Getting the type of 'Type' (line 207)
            Type_17552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 33), 'Type', False)
            # Processing the call keyword arguments (line 207)
            kwargs_17553 = {}
            # Getting the type of 'isinstance' (line 207)
            isinstance_17550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 207)
            isinstance_call_result_17554 = invoke(stypy.reporting.localization.Localization(__file__, 207, 15), isinstance_17550, *[value_17551, Type_17552], **kwargs_17553)
            
            # Testing if the type of an if condition is none (line 207)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 207, 12), isinstance_call_result_17554):
                
                # Assigning a Call to a Name (line 210):
                
                # Call to deepcopy(...): (line 210)
                # Processing the call arguments (line 210)
                # Getting the type of 'value' (line 210)
                value_17562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 40), 'value', False)
                # Processing the call keyword arguments (line 210)
                kwargs_17563 = {}
                # Getting the type of 'copy' (line 210)
                copy_17560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 26), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 210)
                deepcopy_17561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 26), copy_17560, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 210)
                deepcopy_call_result_17564 = invoke(stypy.reporting.localization.Localization(__file__, 210, 26), deepcopy_17561, *[value_17562], **kwargs_17563)
                
                # Assigning a type to the variable 'new_obj' (line 210)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'new_obj', deepcopy_call_result_17564)
            else:
                
                # Testing the type of an if condition (line 207)
                if_condition_17555 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 12), isinstance_call_result_17554)
                # Assigning a type to the variable 'if_condition_17555' (line 207)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'if_condition_17555', if_condition_17555)
                # SSA begins for if statement (line 207)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 208):
                
                # Call to clone(...): (line 208)
                # Processing the call keyword arguments (line 208)
                kwargs_17558 = {}
                # Getting the type of 'value' (line 208)
                value_17556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 26), 'value', False)
                # Obtaining the member 'clone' of a type (line 208)
                clone_17557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 26), value_17556, 'clone')
                # Calling clone(args, kwargs) (line 208)
                clone_call_result_17559 = invoke(stypy.reporting.localization.Localization(__file__, 208, 26), clone_17557, *[], **kwargs_17558)
                
                # Assigning a type to the variable 'new_obj' (line 208)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'new_obj', clone_call_result_17559)
                # SSA branch for the else part of an if statement (line 207)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 210):
                
                # Call to deepcopy(...): (line 210)
                # Processing the call arguments (line 210)
                # Getting the type of 'value' (line 210)
                value_17562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 40), 'value', False)
                # Processing the call keyword arguments (line 210)
                kwargs_17563 = {}
                # Getting the type of 'copy' (line 210)
                copy_17560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 26), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 210)
                deepcopy_17561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 26), copy_17560, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 210)
                deepcopy_call_result_17564 = invoke(stypy.reporting.localization.Localization(__file__, 210, 26), deepcopy_17561, *[value_17562], **kwargs_17563)
                
                # Assigning a type to the variable 'new_obj' (line 210)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'new_obj', deepcopy_call_result_17564)
                # SSA join for if statement (line 207)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Name to a Subscript (line 212):
            # Getting the type of 'new_obj' (line 212)
            new_obj_17565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 39), 'new_obj')
            # Getting the type of 'cloned_obj' (line 212)
            cloned_obj_17566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'cloned_obj')
            # Obtaining the member 'types_of' of a type (line 212)
            types_of_17567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), cloned_obj_17566, 'types_of')
            # Getting the type of 'key' (line 212)
            key_17568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 32), 'key')
            # Storing an element on a container (line 212)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 12), types_of_17567, (key_17568, new_obj_17565))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Attribute (line 214):
        
        # Call to deepcopy(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'self' (line 214)
        self_17571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 43), 'self', False)
        # Obtaining the member 'aliases' of a type (line 214)
        aliases_17572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 43), self_17571, 'aliases')
        # Processing the call keyword arguments (line 214)
        kwargs_17573 = {}
        # Getting the type of 'copy' (line 214)
        copy_17569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 29), 'copy', False)
        # Obtaining the member 'deepcopy' of a type (line 214)
        deepcopy_17570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 29), copy_17569, 'deepcopy')
        # Calling deepcopy(args, kwargs) (line 214)
        deepcopy_call_result_17574 = invoke(stypy.reporting.localization.Localization(__file__, 214, 29), deepcopy_17570, *[aliases_17572], **kwargs_17573)
        
        # Getting the type of 'cloned_obj' (line 214)
        cloned_obj_17575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'cloned_obj')
        # Setting the type of the member 'aliases' of a type (line 214)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 8), cloned_obj_17575, 'aliases', deepcopy_call_result_17574)
        
        # Assigning a Attribute to a Attribute (line 215):
        # Getting the type of 'self' (line 215)
        self_17576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 39), 'self')
        # Obtaining the member 'annotation_record' of a type (line 215)
        annotation_record_17577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 39), self_17576, 'annotation_record')
        # Getting the type of 'cloned_obj' (line 215)
        cloned_obj_17578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'cloned_obj')
        # Setting the type of the member 'annotation_record' of a type (line 215)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), cloned_obj_17578, 'annotation_record', annotation_record_17577)
        
        # Assigning a Attribute to a Attribute (line 216):
        # Getting the type of 'self' (line 216)
        self_17579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 37), 'self')
        # Obtaining the member 'is_main_context' of a type (line 216)
        is_main_context_17580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 37), self_17579, 'is_main_context')
        # Getting the type of 'cloned_obj' (line 216)
        cloned_obj_17581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'cloned_obj')
        # Setting the type of the member 'is_main_context' of a type (line 216)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), cloned_obj_17581, 'is_main_context', is_main_context_17580)
        
        # Assigning a Attribute to a Attribute (line 219):
        # Getting the type of 'self' (line 219)
        self_17582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 49), 'self')
        # Obtaining the member 'declared_argument_name_list' of a type (line 219)
        declared_argument_name_list_17583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 49), self_17582, 'declared_argument_name_list')
        # Getting the type of 'cloned_obj' (line 219)
        cloned_obj_17584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'cloned_obj')
        # Setting the type of the member 'declared_argument_name_list' of a type (line 219)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 8), cloned_obj_17584, 'declared_argument_name_list', declared_argument_name_list_17583)
        
        # Assigning a Attribute to a Attribute (line 220):
        # Getting the type of 'self' (line 220)
        self_17585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 42), 'self')
        # Obtaining the member 'declared_varargs_var' of a type (line 220)
        declared_varargs_var_17586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 42), self_17585, 'declared_varargs_var')
        # Getting the type of 'cloned_obj' (line 220)
        cloned_obj_17587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'cloned_obj')
        # Setting the type of the member 'declared_varargs_var' of a type (line 220)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), cloned_obj_17587, 'declared_varargs_var', declared_varargs_var_17586)
        
        # Assigning a Attribute to a Attribute (line 221):
        # Getting the type of 'self' (line 221)
        self_17588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 41), 'self')
        # Obtaining the member 'declared_kwargs_var' of a type (line 221)
        declared_kwargs_var_17589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 41), self_17588, 'declared_kwargs_var')
        # Getting the type of 'cloned_obj' (line 221)
        cloned_obj_17590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'cloned_obj')
        # Setting the type of the member 'declared_kwargs_var' of a type (line 221)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), cloned_obj_17590, 'declared_kwargs_var', declared_kwargs_var_17589)
        
        # Assigning a Attribute to a Attribute (line 222):
        # Getting the type of 'self' (line 222)
        self_17591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 39), 'self')
        # Obtaining the member 'declared_defaults' of a type (line 222)
        declared_defaults_17592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 39), self_17591, 'declared_defaults')
        # Getting the type of 'cloned_obj' (line 222)
        cloned_obj_17593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'cloned_obj')
        # Setting the type of the member 'declared_defaults' of a type (line 222)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), cloned_obj_17593, 'declared_defaults', declared_defaults_17592)
        
        # Assigning a Attribute to a Attribute (line 224):
        # Getting the type of 'self' (line 224)
        self_17594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 38), 'self')
        # Obtaining the member 'declaration_line' of a type (line 224)
        declaration_line_17595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 38), self_17594, 'declaration_line')
        # Getting the type of 'cloned_obj' (line 224)
        cloned_obj_17596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'cloned_obj')
        # Setting the type of the member 'declaration_line' of a type (line 224)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), cloned_obj_17596, 'declaration_line', declaration_line_17595)
        
        # Assigning a Attribute to a Attribute (line 225):
        # Getting the type of 'self' (line 225)
        self_17597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 40), 'self')
        # Obtaining the member 'declaration_column' of a type (line 225)
        declaration_column_17598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 40), self_17597, 'declaration_column')
        # Getting the type of 'cloned_obj' (line 225)
        cloned_obj_17599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'cloned_obj')
        # Setting the type of the member 'declaration_column' of a type (line 225)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 8), cloned_obj_17599, 'declaration_column', declaration_column_17598)
        
        # Assigning a Attribute to a Attribute (line 227):
        # Getting the type of 'self' (line 227)
        self_17600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 33), 'self')
        # Obtaining the member 'return_type' of a type (line 227)
        return_type_17601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 33), self_17600, 'return_type')
        # Getting the type of 'cloned_obj' (line 227)
        cloned_obj_17602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'cloned_obj')
        # Setting the type of the member 'return_type' of a type (line 227)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 8), cloned_obj_17602, 'return_type', return_type_17601)
        # Getting the type of 'cloned_obj' (line 229)
        cloned_obj_17603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'cloned_obj')
        # Assigning a type to the variable 'stypy_return_type' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'stypy_return_type', cloned_obj_17603)
        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 197)
        stypy_return_type_17604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17604)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_17604


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

        str_17605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, (-1)), 'str', '\n        Copies this function context into a newly created one and return it. The copied function context is a shallow\n        copy.\n        :return: Copy of this function context\n        ')
        
        # Assigning a Call to a Name (line 237):
        
        # Call to FunctionContext(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'self' (line 237)
        self_17607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 37), 'self', False)
        # Obtaining the member 'function_name' of a type (line 237)
        function_name_17608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 37), self_17607, 'function_name')
        # Processing the call keyword arguments (line 237)
        kwargs_17609 = {}
        # Getting the type of 'FunctionContext' (line 237)
        FunctionContext_17606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 21), 'FunctionContext', False)
        # Calling FunctionContext(args, kwargs) (line 237)
        FunctionContext_call_result_17610 = invoke(stypy.reporting.localization.Localization(__file__, 237, 21), FunctionContext_17606, *[function_name_17608], **kwargs_17609)
        
        # Assigning a type to the variable 'copied_obj' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'copied_obj', FunctionContext_call_result_17610)
        
        # Assigning a Attribute to a Attribute (line 239):
        # Getting the type of 'self' (line 239)
        self_17611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 33), 'self')
        # Obtaining the member 'global_vars' of a type (line 239)
        global_vars_17612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 33), self_17611, 'global_vars')
        # Getting the type of 'copied_obj' (line 239)
        copied_obj_17613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'copied_obj')
        # Setting the type of the member 'global_vars' of a type (line 239)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), copied_obj_17613, 'global_vars', global_vars_17612)
        
        # Assigning a Attribute to a Attribute (line 240):
        # Getting the type of 'self' (line 240)
        self_17614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 30), 'self')
        # Obtaining the member 'types_of' of a type (line 240)
        types_of_17615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 30), self_17614, 'types_of')
        # Getting the type of 'copied_obj' (line 240)
        copied_obj_17616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'copied_obj')
        # Setting the type of the member 'types_of' of a type (line 240)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), copied_obj_17616, 'types_of', types_of_17615)
        
        # Assigning a Attribute to a Attribute (line 242):
        # Getting the type of 'self' (line 242)
        self_17617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 29), 'self')
        # Obtaining the member 'aliases' of a type (line 242)
        aliases_17618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 29), self_17617, 'aliases')
        # Getting the type of 'copied_obj' (line 242)
        copied_obj_17619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'copied_obj')
        # Setting the type of the member 'aliases' of a type (line 242)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), copied_obj_17619, 'aliases', aliases_17618)
        
        # Assigning a Attribute to a Attribute (line 243):
        # Getting the type of 'self' (line 243)
        self_17620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 39), 'self')
        # Obtaining the member 'annotation_record' of a type (line 243)
        annotation_record_17621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 39), self_17620, 'annotation_record')
        # Getting the type of 'copied_obj' (line 243)
        copied_obj_17622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'copied_obj')
        # Setting the type of the member 'annotation_record' of a type (line 243)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 8), copied_obj_17622, 'annotation_record', annotation_record_17621)
        
        # Assigning a Attribute to a Attribute (line 244):
        # Getting the type of 'self' (line 244)
        self_17623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 37), 'self')
        # Obtaining the member 'is_main_context' of a type (line 244)
        is_main_context_17624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 37), self_17623, 'is_main_context')
        # Getting the type of 'copied_obj' (line 244)
        copied_obj_17625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'copied_obj')
        # Setting the type of the member 'is_main_context' of a type (line 244)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), copied_obj_17625, 'is_main_context', is_main_context_17624)
        
        # Assigning a Attribute to a Attribute (line 247):
        # Getting the type of 'self' (line 247)
        self_17626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 49), 'self')
        # Obtaining the member 'declared_argument_name_list' of a type (line 247)
        declared_argument_name_list_17627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 49), self_17626, 'declared_argument_name_list')
        # Getting the type of 'copied_obj' (line 247)
        copied_obj_17628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'copied_obj')
        # Setting the type of the member 'declared_argument_name_list' of a type (line 247)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), copied_obj_17628, 'declared_argument_name_list', declared_argument_name_list_17627)
        
        # Assigning a Attribute to a Attribute (line 248):
        # Getting the type of 'self' (line 248)
        self_17629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 42), 'self')
        # Obtaining the member 'declared_varargs_var' of a type (line 248)
        declared_varargs_var_17630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 42), self_17629, 'declared_varargs_var')
        # Getting the type of 'copied_obj' (line 248)
        copied_obj_17631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'copied_obj')
        # Setting the type of the member 'declared_varargs_var' of a type (line 248)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 8), copied_obj_17631, 'declared_varargs_var', declared_varargs_var_17630)
        
        # Assigning a Attribute to a Attribute (line 249):
        # Getting the type of 'self' (line 249)
        self_17632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 41), 'self')
        # Obtaining the member 'declared_kwargs_var' of a type (line 249)
        declared_kwargs_var_17633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 41), self_17632, 'declared_kwargs_var')
        # Getting the type of 'copied_obj' (line 249)
        copied_obj_17634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'copied_obj')
        # Setting the type of the member 'declared_kwargs_var' of a type (line 249)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 8), copied_obj_17634, 'declared_kwargs_var', declared_kwargs_var_17633)
        
        # Assigning a Attribute to a Attribute (line 250):
        # Getting the type of 'self' (line 250)
        self_17635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 39), 'self')
        # Obtaining the member 'declared_defaults' of a type (line 250)
        declared_defaults_17636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 39), self_17635, 'declared_defaults')
        # Getting the type of 'copied_obj' (line 250)
        copied_obj_17637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'copied_obj')
        # Setting the type of the member 'declared_defaults' of a type (line 250)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), copied_obj_17637, 'declared_defaults', declared_defaults_17636)
        
        # Assigning a Attribute to a Attribute (line 252):
        # Getting the type of 'self' (line 252)
        self_17638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 38), 'self')
        # Obtaining the member 'declaration_line' of a type (line 252)
        declaration_line_17639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 38), self_17638, 'declaration_line')
        # Getting the type of 'copied_obj' (line 252)
        copied_obj_17640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'copied_obj')
        # Setting the type of the member 'declaration_line' of a type (line 252)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), copied_obj_17640, 'declaration_line', declaration_line_17639)
        
        # Assigning a Attribute to a Attribute (line 253):
        # Getting the type of 'self' (line 253)
        self_17641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 40), 'self')
        # Obtaining the member 'declaration_column' of a type (line 253)
        declaration_column_17642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 40), self_17641, 'declaration_column')
        # Getting the type of 'copied_obj' (line 253)
        copied_obj_17643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'copied_obj')
        # Setting the type of the member 'declaration_column' of a type (line 253)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), copied_obj_17643, 'declaration_column', declaration_column_17642)
        
        # Assigning a Attribute to a Attribute (line 255):
        # Getting the type of 'self' (line 255)
        self_17644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 33), 'self')
        # Obtaining the member 'return_type' of a type (line 255)
        return_type_17645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 33), self_17644, 'return_type')
        # Getting the type of 'copied_obj' (line 255)
        copied_obj_17646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'copied_obj')
        # Setting the type of the member 'return_type' of a type (line 255)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 8), copied_obj_17646, 'return_type', return_type_17645)
        # Getting the type of 'copied_obj' (line 257)
        copied_obj_17647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'copied_obj')
        # Assigning a type to the variable 'stypy_return_type' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'stypy_return_type', copied_obj_17647)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_17648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17648)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_17648


# Assigning a type to the variable 'FunctionContext' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'FunctionContext', FunctionContext)

# Assigning a Name to a Name (line 13):
# Getting the type of 'True' (line 13)
True_17649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 21), 'True')
# Getting the type of 'FunctionContext'
FunctionContext_17650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FunctionContext')
# Setting the type of the member 'annotate_types' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FunctionContext_17650, 'annotate_types', True_17649)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
