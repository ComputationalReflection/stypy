
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import os
2: 
3: from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_management_copy import get_type_name
4: from stypy_copy import stypy_parameters_copy
5: 
6: 
7: class TypeDataFileWriter:
8:     '''
9:     A simple writer to write type data files, that are used to unit test the code generation of stypy when applied
10:     to the programs included in the test battery. A type data file has a format like this, written by this code:
11: 
12:     import types
13:     from stypy import union_type
14:     from stypy.python_lib.python_types.type_inference.undefined_type import UndefinedType
15: 
16:     test_types = {
17:         '__init__': {
18:             'StringComp': int,
19:             'self': types.InstanceType,
20:             'Discr': int,
21:             'PtrComp': types.NoneType,
22:             'IntComp': int,
23:             'EnumComp': int,
24:         },
25:         'Proc5': {
26:         },
27:         'Proc4': {
28:             'BoolLoc': int #bool,
29:         },
30:         'Func1': {
31:             'CharLoc2': str,
32:             'CharLoc1': str,
33:             'CharPar2': str,
34:             'CharPar1': str,
35:         },
36:         'Func2': {
37:             'StrParI1': str,
38:             'CharLoc': union_type.UnionType.create_union_type_from_types(str, UndefinedType()),
39:             'StrParI2': str,
40:             'IntLoc': int,
41:         },
42:         'Proc7': {
43:             'IntParOut': int,
44:             'IntLoc': int,
45:             'IntParI1': int,
46:             'IntParI2': int,
47:         },
48:         'Proc8': {
49:             'Array1Par': list,
50:             'IntParI2': int,
51:             'IntParI1': int,
52:             'Array2Par': list,
53:             'IntLoc': int,
54:             'IntIndex': int,
55:         },
56:         'copy': {
57:             'self': types.InstanceType,
58:         },
59:         'Proc3': {
60:             'PtrParOut': types.InstanceType,
61:         },
62:         'Func3': {
63:             'EnumLoc': int,
64:             'EnumParIn': int,
65:         },
66:         'Proc6': {
67:             'EnumParIn': int,
68:             'EnumParOut': int,
69:         },
70:         'Proc1': {
71:             'NextRecord': types.InstanceType,
72:             'PtrParIn': types.InstanceType,
73:         },
74:         'Proc2': {
75:             'EnumLoc': int,
76:             'IntParIO': int,
77:             'IntLoc': int,
78:         },
79:         'Proc0': {
80:             'EnumLoc': int,
81:             'String2Loc': str,
82:             'IntLoc2': int,
83:             'IntLoc3': int,
84:             'String1Loc': str,
85:             'IntLoc1': int,
86:             'i': int,
87:             'CharIndex': str,
88:             'benchtime': float,
89:             'loopsPerBenchtime': float,
90:             'loops': int,
91:             'nulltime': float,
92:             'starttime': float,
93:         },
94:         'pystones': {
95:             'loops': int,
96:         },
97:         'main': {
98:             'stones': int, #should be float
99:             'loops': int,
100:             'benchtime': int, #should be float
101:         },
102:         '__main__': {
103:             'Array1Glob': list,
104:             'loops': int,
105:             'TRUE': int,
106:             'Record': types.ClassType,
107:             'Func3': types.LambdaType,
108:             'Func2': types.LambdaType,
109:             'Func1': types.LambdaType,
110:             'Array2Glob': list,
111:             'clock': types.BuiltinFunctionType,
112:             'BoolGlob': union_type.UnionType.create_union_type_from_types(int, bool),
113:             'LOOPS': int,
114:             'main': types.LambdaType,
115:             'Proc8': types.LambdaType,
116:             'Char2Glob': str,
117:             'pystones': types.LambdaType,
118:             'PtrGlbNext': union_type.UnionType.create_union_type_from_types(types.InstanceType, types.NoneType),
119:             'nargs': int,
120:             'sys': types.ModuleType,
121:             'TypeDataFileWriter': types.ClassType,
122:             'IntGlob': int,
123:             'Ident4': int,
124:             'Ident5': int,
125:             'FALSE': int,
126:             'Ident1': int,
127:             'Ident2': int,
128:             'Ident3': int,
129:             'Char1Glob': str,
130:             'PtrGlb': types.NoneType, #types.InstanceType,
131:             'error': types.LambdaType,
132:             'Proc5': types.LambdaType,
133:             'Proc4': types.LambdaType,
134:             'Proc7': types.LambdaType,
135:             'Proc6': types.LambdaType,
136:             'Proc1': types.LambdaType,
137:             'Proc0': types.LambdaType,
138:             'Proc3': types.LambdaType,
139:             'Proc2': types.LambdaType,
140:         },
141:     }
142:     As we see, there are a fixed number of imports and a dictionary called test_types with str keys and dict values.
143:     Each key correspond to the name of a function/method and the value is the variable table (name: type) expected in
144:     this context.
145:     '''
146: 
147:     def __init__(self, file_path):
148:         '''
149:         Creates a writer for type data files
150:         :param file_path: File to write to
151:         :return:
152:         '''
153:         self.already_processed_contexts = []
154:         self.type_file_txt = "import types\n\ntest_types = {\n"
155:         file_path = file_path.replace('\\', '/')
156:         self.file_path = file_path
157:         self.dest_folder = os.path.dirname(file_path)
158:         self.type_file = (file_path.split('/')[-1])[0:-3].split('__')[
159:                              0] + stypy_parameters_copy.type_data_file_postfix + ".py"
160: 
161:     def add_type_dict_for_main_context(self, var_dict):
162:         '''
163:         Add the dictionary of variables for the main context
164:         :param var_dict: dictionary of name: type
165:         :return:
166:         '''
167:         self.__add_type_dict_for_context(var_dict)
168: 
169:     def add_type_dict_for_context(self, var_dict):
170:         '''
171:         Add the dictionary of variables for a function context. Function name is automatically obtained by traversin
172:         the call stack. Please note that this function is used in type data autogenerator programs, therefore we can
173:         obtain this data using this technique
174:         :param var_dict: dictionary of name: type
175:         :return:
176:         '''
177:         import traceback
178: 
179:         func_name = traceback.extract_stack(None, 2)[0][2]
180: 
181:         self.__add_type_dict_for_context(var_dict, func_name)
182: 
183:     def __add_type_dict_for_context(self, var_dict, context="__main__"):
184:         '''
185:         Helper method for the previous one
186:         :param var_dict:
187:         :param context:
188:         :return:
189:         '''
190:         if context in self.already_processed_contexts:
191:             return
192: 
193:         vars_ = filter(lambda var_: "__" not in var_ and not var_ == 'stypy' and not var_ == 'type_test',
194:                        var_dict.keys())
195: 
196:         self.type_file_txt += "    '" + context + "': {\n"
197:         for var in vars_:
198:             self.type_file_txt += "        '" + var + "': " + get_type_name(type(var_dict[var])) + ", \n"
199: 
200:         self.type_file_txt += "    " + "}, \n"
201: 
202:         self.already_processed_contexts.append(context)
203: 
204:     def generate_type_data_file(self):
205:         '''
206:         Generates the type data file
207:         :return:
208:         '''
209:         # print self.dest_folder
210:         # print self.type_file
211:         self.type_file_txt += "}\n"
212:         with open(self.dest_folder + "/" + self.type_file, 'w') as outfile:
213:             outfile.write(self.type_file_txt)
214: 

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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_management_copy import get_type_name' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')
import_2601 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_management_copy')

if (type(import_2601) is not StypyTypeError):

    if (import_2601 != 'pyd_module'):
        __import__(import_2601)
        sys_modules_2602 = sys.modules[import_2601]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_management_copy', sys_modules_2602.module_type_store, module_type_store, ['get_type_name'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_2602, sys_modules_2602.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_management_copy import get_type_name

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_management_copy', None, module_type_store, ['get_type_name'], [get_type_name])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_management_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_management_copy', import_2601)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from stypy_copy import stypy_parameters_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')
import_2603 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy')

if (type(import_2603) is not StypyTypeError):

    if (import_2603 != 'pyd_module'):
        __import__(import_2603)
        sys_modules_2604 = sys.modules[import_2603]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy', sys_modules_2604.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_2604, sys_modules_2604.module_type_store, module_type_store)
    else:
        from stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'stypy_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy', import_2603)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')

# Declaration of the 'TypeDataFileWriter' class

class TypeDataFileWriter:
    str_2605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, (-1)), 'str', "\n    A simple writer to write type data files, that are used to unit test the code generation of stypy when applied\n    to the programs included in the test battery. A type data file has a format like this, written by this code:\n\n    import types\n    from stypy import union_type\n    from stypy.python_lib.python_types.type_inference.undefined_type import UndefinedType\n\n    test_types = {\n        '__init__': {\n            'StringComp': int,\n            'self': types.InstanceType,\n            'Discr': int,\n            'PtrComp': types.NoneType,\n            'IntComp': int,\n            'EnumComp': int,\n        },\n        'Proc5': {\n        },\n        'Proc4': {\n            'BoolLoc': int #bool,\n        },\n        'Func1': {\n            'CharLoc2': str,\n            'CharLoc1': str,\n            'CharPar2': str,\n            'CharPar1': str,\n        },\n        'Func2': {\n            'StrParI1': str,\n            'CharLoc': union_type.UnionType.create_union_type_from_types(str, UndefinedType()),\n            'StrParI2': str,\n            'IntLoc': int,\n        },\n        'Proc7': {\n            'IntParOut': int,\n            'IntLoc': int,\n            'IntParI1': int,\n            'IntParI2': int,\n        },\n        'Proc8': {\n            'Array1Par': list,\n            'IntParI2': int,\n            'IntParI1': int,\n            'Array2Par': list,\n            'IntLoc': int,\n            'IntIndex': int,\n        },\n        'copy': {\n            'self': types.InstanceType,\n        },\n        'Proc3': {\n            'PtrParOut': types.InstanceType,\n        },\n        'Func3': {\n            'EnumLoc': int,\n            'EnumParIn': int,\n        },\n        'Proc6': {\n            'EnumParIn': int,\n            'EnumParOut': int,\n        },\n        'Proc1': {\n            'NextRecord': types.InstanceType,\n            'PtrParIn': types.InstanceType,\n        },\n        'Proc2': {\n            'EnumLoc': int,\n            'IntParIO': int,\n            'IntLoc': int,\n        },\n        'Proc0': {\n            'EnumLoc': int,\n            'String2Loc': str,\n            'IntLoc2': int,\n            'IntLoc3': int,\n            'String1Loc': str,\n            'IntLoc1': int,\n            'i': int,\n            'CharIndex': str,\n            'benchtime': float,\n            'loopsPerBenchtime': float,\n            'loops': int,\n            'nulltime': float,\n            'starttime': float,\n        },\n        'pystones': {\n            'loops': int,\n        },\n        'main': {\n            'stones': int, #should be float\n            'loops': int,\n            'benchtime': int, #should be float\n        },\n        '__main__': {\n            'Array1Glob': list,\n            'loops': int,\n            'TRUE': int,\n            'Record': types.ClassType,\n            'Func3': types.LambdaType,\n            'Func2': types.LambdaType,\n            'Func1': types.LambdaType,\n            'Array2Glob': list,\n            'clock': types.BuiltinFunctionType,\n            'BoolGlob': union_type.UnionType.create_union_type_from_types(int, bool),\n            'LOOPS': int,\n            'main': types.LambdaType,\n            'Proc8': types.LambdaType,\n            'Char2Glob': str,\n            'pystones': types.LambdaType,\n            'PtrGlbNext': union_type.UnionType.create_union_type_from_types(types.InstanceType, types.NoneType),\n            'nargs': int,\n            'sys': types.ModuleType,\n            'TypeDataFileWriter': types.ClassType,\n            'IntGlob': int,\n            'Ident4': int,\n            'Ident5': int,\n            'FALSE': int,\n            'Ident1': int,\n            'Ident2': int,\n            'Ident3': int,\n            'Char1Glob': str,\n            'PtrGlb': types.NoneType, #types.InstanceType,\n            'error': types.LambdaType,\n            'Proc5': types.LambdaType,\n            'Proc4': types.LambdaType,\n            'Proc7': types.LambdaType,\n            'Proc6': types.LambdaType,\n            'Proc1': types.LambdaType,\n            'Proc0': types.LambdaType,\n            'Proc3': types.LambdaType,\n            'Proc2': types.LambdaType,\n        },\n    }\n    As we see, there are a fixed number of imports and a dictionary called test_types with str keys and dict values.\n    Each key correspond to the name of a function/method and the value is the variable table (name: type) expected in\n    this context.\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeDataFileWriter.__init__', ['file_path'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['file_path'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_2606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, (-1)), 'str', '\n        Creates a writer for type data files\n        :param file_path: File to write to\n        :return:\n        ')
        
        # Assigning a List to a Attribute (line 153):
        
        # Obtaining an instance of the builtin type 'list' (line 153)
        list_2607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 153)
        
        # Getting the type of 'self' (line 153)
        self_2608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'self')
        # Setting the type of the member 'already_processed_contexts' of a type (line 153)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), self_2608, 'already_processed_contexts', list_2607)
        
        # Assigning a Str to a Attribute (line 154):
        str_2609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 29), 'str', 'import types\n\ntest_types = {\n')
        # Getting the type of 'self' (line 154)
        self_2610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'self')
        # Setting the type of the member 'type_file_txt' of a type (line 154)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), self_2610, 'type_file_txt', str_2609)
        
        # Assigning a Call to a Name (line 155):
        
        # Call to replace(...): (line 155)
        # Processing the call arguments (line 155)
        str_2613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 38), 'str', '\\')
        str_2614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 44), 'str', '/')
        # Processing the call keyword arguments (line 155)
        kwargs_2615 = {}
        # Getting the type of 'file_path' (line 155)
        file_path_2611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'file_path', False)
        # Obtaining the member 'replace' of a type (line 155)
        replace_2612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 20), file_path_2611, 'replace')
        # Calling replace(args, kwargs) (line 155)
        replace_call_result_2616 = invoke(stypy.reporting.localization.Localization(__file__, 155, 20), replace_2612, *[str_2613, str_2614], **kwargs_2615)
        
        # Assigning a type to the variable 'file_path' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'file_path', replace_call_result_2616)
        
        # Assigning a Name to a Attribute (line 156):
        # Getting the type of 'file_path' (line 156)
        file_path_2617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 25), 'file_path')
        # Getting the type of 'self' (line 156)
        self_2618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'self')
        # Setting the type of the member 'file_path' of a type (line 156)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 8), self_2618, 'file_path', file_path_2617)
        
        # Assigning a Call to a Attribute (line 157):
        
        # Call to dirname(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'file_path' (line 157)
        file_path_2622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 43), 'file_path', False)
        # Processing the call keyword arguments (line 157)
        kwargs_2623 = {}
        # Getting the type of 'os' (line 157)
        os_2619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 157)
        path_2620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 27), os_2619, 'path')
        # Obtaining the member 'dirname' of a type (line 157)
        dirname_2621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 27), path_2620, 'dirname')
        # Calling dirname(args, kwargs) (line 157)
        dirname_call_result_2624 = invoke(stypy.reporting.localization.Localization(__file__, 157, 27), dirname_2621, *[file_path_2622], **kwargs_2623)
        
        # Getting the type of 'self' (line 157)
        self_2625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self')
        # Setting the type of the member 'dest_folder' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), self_2625, 'dest_folder', dirname_call_result_2624)
        
        # Assigning a BinOp to a Attribute (line 158):
        
        # Obtaining the type of the subscript
        int_2626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 29), 'int')
        
        # Call to split(...): (line 158)
        # Processing the call arguments (line 158)
        str_2641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 64), 'str', '__')
        # Processing the call keyword arguments (line 158)
        kwargs_2642 = {}
        
        # Obtaining the type of the subscript
        int_2627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 52), 'int')
        int_2628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 54), 'int')
        slice_2629 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 158, 26), int_2627, int_2628, None)
        
        # Obtaining the type of the subscript
        int_2630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 47), 'int')
        
        # Call to split(...): (line 158)
        # Processing the call arguments (line 158)
        str_2633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 42), 'str', '/')
        # Processing the call keyword arguments (line 158)
        kwargs_2634 = {}
        # Getting the type of 'file_path' (line 158)
        file_path_2631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 26), 'file_path', False)
        # Obtaining the member 'split' of a type (line 158)
        split_2632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 26), file_path_2631, 'split')
        # Calling split(args, kwargs) (line 158)
        split_call_result_2635 = invoke(stypy.reporting.localization.Localization(__file__, 158, 26), split_2632, *[str_2633], **kwargs_2634)
        
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___2636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 26), split_call_result_2635, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_2637 = invoke(stypy.reporting.localization.Localization(__file__, 158, 26), getitem___2636, int_2630)
        
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___2638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 26), subscript_call_result_2637, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_2639 = invoke(stypy.reporting.localization.Localization(__file__, 158, 26), getitem___2638, slice_2629)
        
        # Obtaining the member 'split' of a type (line 158)
        split_2640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 26), subscript_call_result_2639, 'split')
        # Calling split(args, kwargs) (line 158)
        split_call_result_2643 = invoke(stypy.reporting.localization.Localization(__file__, 158, 26), split_2640, *[str_2641], **kwargs_2642)
        
        # Obtaining the member '__getitem__' of a type (line 158)
        getitem___2644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 26), split_call_result_2643, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 158)
        subscript_call_result_2645 = invoke(stypy.reporting.localization.Localization(__file__, 158, 26), getitem___2644, int_2626)
        
        # Getting the type of 'stypy_parameters_copy' (line 159)
        stypy_parameters_copy_2646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 34), 'stypy_parameters_copy')
        # Obtaining the member 'type_data_file_postfix' of a type (line 159)
        type_data_file_postfix_2647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 34), stypy_parameters_copy_2646, 'type_data_file_postfix')
        # Applying the binary operator '+' (line 158)
        result_add_2648 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 25), '+', subscript_call_result_2645, type_data_file_postfix_2647)
        
        str_2649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 81), 'str', '.py')
        # Applying the binary operator '+' (line 159)
        result_add_2650 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 79), '+', result_add_2648, str_2649)
        
        # Getting the type of 'self' (line 158)
        self_2651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'self')
        # Setting the type of the member 'type_file' of a type (line 158)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), self_2651, 'type_file', result_add_2650)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def add_type_dict_for_main_context(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_type_dict_for_main_context'
        module_type_store = module_type_store.open_function_context('add_type_dict_for_main_context', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeDataFileWriter.add_type_dict_for_main_context.__dict__.__setitem__('stypy_localization', localization)
        TypeDataFileWriter.add_type_dict_for_main_context.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeDataFileWriter.add_type_dict_for_main_context.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeDataFileWriter.add_type_dict_for_main_context.__dict__.__setitem__('stypy_function_name', 'TypeDataFileWriter.add_type_dict_for_main_context')
        TypeDataFileWriter.add_type_dict_for_main_context.__dict__.__setitem__('stypy_param_names_list', ['var_dict'])
        TypeDataFileWriter.add_type_dict_for_main_context.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeDataFileWriter.add_type_dict_for_main_context.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeDataFileWriter.add_type_dict_for_main_context.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeDataFileWriter.add_type_dict_for_main_context.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeDataFileWriter.add_type_dict_for_main_context.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeDataFileWriter.add_type_dict_for_main_context.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeDataFileWriter.add_type_dict_for_main_context', ['var_dict'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_type_dict_for_main_context', localization, ['var_dict'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_type_dict_for_main_context(...)' code ##################

        str_2652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, (-1)), 'str', '\n        Add the dictionary of variables for the main context\n        :param var_dict: dictionary of name: type\n        :return:\n        ')
        
        # Call to __add_type_dict_for_context(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'var_dict' (line 167)
        var_dict_2655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 41), 'var_dict', False)
        # Processing the call keyword arguments (line 167)
        kwargs_2656 = {}
        # Getting the type of 'self' (line 167)
        self_2653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'self', False)
        # Obtaining the member '__add_type_dict_for_context' of a type (line 167)
        add_type_dict_for_context_2654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), self_2653, '__add_type_dict_for_context')
        # Calling __add_type_dict_for_context(args, kwargs) (line 167)
        add_type_dict_for_context_call_result_2657 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), add_type_dict_for_context_2654, *[var_dict_2655], **kwargs_2656)
        
        
        # ################# End of 'add_type_dict_for_main_context(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_type_dict_for_main_context' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_2658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2658)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_type_dict_for_main_context'
        return stypy_return_type_2658


    @norecursion
    def add_type_dict_for_context(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_type_dict_for_context'
        module_type_store = module_type_store.open_function_context('add_type_dict_for_context', 169, 4, False)
        # Assigning a type to the variable 'self' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeDataFileWriter.add_type_dict_for_context.__dict__.__setitem__('stypy_localization', localization)
        TypeDataFileWriter.add_type_dict_for_context.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeDataFileWriter.add_type_dict_for_context.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeDataFileWriter.add_type_dict_for_context.__dict__.__setitem__('stypy_function_name', 'TypeDataFileWriter.add_type_dict_for_context')
        TypeDataFileWriter.add_type_dict_for_context.__dict__.__setitem__('stypy_param_names_list', ['var_dict'])
        TypeDataFileWriter.add_type_dict_for_context.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeDataFileWriter.add_type_dict_for_context.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeDataFileWriter.add_type_dict_for_context.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeDataFileWriter.add_type_dict_for_context.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeDataFileWriter.add_type_dict_for_context.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeDataFileWriter.add_type_dict_for_context.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeDataFileWriter.add_type_dict_for_context', ['var_dict'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_type_dict_for_context', localization, ['var_dict'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_type_dict_for_context(...)' code ##################

        str_2659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, (-1)), 'str', '\n        Add the dictionary of variables for a function context. Function name is automatically obtained by traversin\n        the call stack. Please note that this function is used in type data autogenerator programs, therefore we can\n        obtain this data using this technique\n        :param var_dict: dictionary of name: type\n        :return:\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 177, 8))
        
        # 'import traceback' statement (line 177)
        import traceback

        import_module(stypy.reporting.localization.Localization(__file__, 177, 8), 'traceback', traceback, module_type_store)
        
        
        # Assigning a Subscript to a Name (line 179):
        
        # Obtaining the type of the subscript
        int_2660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 56), 'int')
        
        # Obtaining the type of the subscript
        int_2661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 53), 'int')
        
        # Call to extract_stack(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'None' (line 179)
        None_2664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 44), 'None', False)
        int_2665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 50), 'int')
        # Processing the call keyword arguments (line 179)
        kwargs_2666 = {}
        # Getting the type of 'traceback' (line 179)
        traceback_2662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'traceback', False)
        # Obtaining the member 'extract_stack' of a type (line 179)
        extract_stack_2663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 20), traceback_2662, 'extract_stack')
        # Calling extract_stack(args, kwargs) (line 179)
        extract_stack_call_result_2667 = invoke(stypy.reporting.localization.Localization(__file__, 179, 20), extract_stack_2663, *[None_2664, int_2665], **kwargs_2666)
        
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___2668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 20), extract_stack_call_result_2667, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_2669 = invoke(stypy.reporting.localization.Localization(__file__, 179, 20), getitem___2668, int_2661)
        
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___2670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 20), subscript_call_result_2669, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_2671 = invoke(stypy.reporting.localization.Localization(__file__, 179, 20), getitem___2670, int_2660)
        
        # Assigning a type to the variable 'func_name' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'func_name', subscript_call_result_2671)
        
        # Call to __add_type_dict_for_context(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'var_dict' (line 181)
        var_dict_2674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 41), 'var_dict', False)
        # Getting the type of 'func_name' (line 181)
        func_name_2675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 51), 'func_name', False)
        # Processing the call keyword arguments (line 181)
        kwargs_2676 = {}
        # Getting the type of 'self' (line 181)
        self_2672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'self', False)
        # Obtaining the member '__add_type_dict_for_context' of a type (line 181)
        add_type_dict_for_context_2673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), self_2672, '__add_type_dict_for_context')
        # Calling __add_type_dict_for_context(args, kwargs) (line 181)
        add_type_dict_for_context_call_result_2677 = invoke(stypy.reporting.localization.Localization(__file__, 181, 8), add_type_dict_for_context_2673, *[var_dict_2674, func_name_2675], **kwargs_2676)
        
        
        # ################# End of 'add_type_dict_for_context(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_type_dict_for_context' in the type store
        # Getting the type of 'stypy_return_type' (line 169)
        stypy_return_type_2678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2678)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_type_dict_for_context'
        return stypy_return_type_2678


    @norecursion
    def __add_type_dict_for_context(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_2679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 60), 'str', '__main__')
        defaults = [str_2679]
        # Create a new context for function '__add_type_dict_for_context'
        module_type_store = module_type_store.open_function_context('__add_type_dict_for_context', 183, 4, False)
        # Assigning a type to the variable 'self' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeDataFileWriter.__add_type_dict_for_context.__dict__.__setitem__('stypy_localization', localization)
        TypeDataFileWriter.__add_type_dict_for_context.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeDataFileWriter.__add_type_dict_for_context.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeDataFileWriter.__add_type_dict_for_context.__dict__.__setitem__('stypy_function_name', 'TypeDataFileWriter.__add_type_dict_for_context')
        TypeDataFileWriter.__add_type_dict_for_context.__dict__.__setitem__('stypy_param_names_list', ['var_dict', 'context'])
        TypeDataFileWriter.__add_type_dict_for_context.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeDataFileWriter.__add_type_dict_for_context.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeDataFileWriter.__add_type_dict_for_context.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeDataFileWriter.__add_type_dict_for_context.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeDataFileWriter.__add_type_dict_for_context.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeDataFileWriter.__add_type_dict_for_context.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeDataFileWriter.__add_type_dict_for_context', ['var_dict', 'context'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__add_type_dict_for_context', localization, ['var_dict', 'context'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__add_type_dict_for_context(...)' code ##################

        str_2680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, (-1)), 'str', '\n        Helper method for the previous one\n        :param var_dict:\n        :param context:\n        :return:\n        ')
        
        # Getting the type of 'context' (line 190)
        context_2681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 11), 'context')
        # Getting the type of 'self' (line 190)
        self_2682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 22), 'self')
        # Obtaining the member 'already_processed_contexts' of a type (line 190)
        already_processed_contexts_2683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 22), self_2682, 'already_processed_contexts')
        # Applying the binary operator 'in' (line 190)
        result_contains_2684 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 11), 'in', context_2681, already_processed_contexts_2683)
        
        # Testing if the type of an if condition is none (line 190)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 190, 8), result_contains_2684):
            pass
        else:
            
            # Testing the type of an if condition (line 190)
            if_condition_2685 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 8), result_contains_2684)
            # Assigning a type to the variable 'if_condition_2685' (line 190)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'if_condition_2685', if_condition_2685)
            # SSA begins for if statement (line 190)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 191)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 190)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 193):
        
        # Call to filter(...): (line 193)
        # Processing the call arguments (line 193)

        @norecursion
        def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_1'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 193, 23, True)
            # Passed parameters checking function
            _stypy_temp_lambda_1.stypy_localization = localization
            _stypy_temp_lambda_1.stypy_type_of_self = None
            _stypy_temp_lambda_1.stypy_type_store = module_type_store
            _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
            _stypy_temp_lambda_1.stypy_param_names_list = ['var_']
            _stypy_temp_lambda_1.stypy_varargs_param_name = None
            _stypy_temp_lambda_1.stypy_kwargs_param_name = None
            _stypy_temp_lambda_1.stypy_call_defaults = defaults
            _stypy_temp_lambda_1.stypy_call_varargs = varargs
            _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['var_'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_1', ['var_'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Evaluating a boolean operation
            
            str_2687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 36), 'str', '__')
            # Getting the type of 'var_' (line 193)
            var__2688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 48), 'var_', False)
            # Applying the binary operator 'notin' (line 193)
            result_contains_2689 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 36), 'notin', str_2687, var__2688)
            
            
            
            # Getting the type of 'var_' (line 193)
            var__2690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 61), 'var_', False)
            str_2691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 69), 'str', 'stypy')
            # Applying the binary operator '==' (line 193)
            result_eq_2692 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 61), '==', var__2690, str_2691)
            
            # Applying the 'not' unary operator (line 193)
            result_not__2693 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 57), 'not', result_eq_2692)
            
            # Applying the binary operator 'and' (line 193)
            result_and_keyword_2694 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 36), 'and', result_contains_2689, result_not__2693)
            
            
            # Getting the type of 'var_' (line 193)
            var__2695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 85), 'var_', False)
            str_2696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 93), 'str', 'type_test')
            # Applying the binary operator '==' (line 193)
            result_eq_2697 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 85), '==', var__2695, str_2696)
            
            # Applying the 'not' unary operator (line 193)
            result_not__2698 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 81), 'not', result_eq_2697)
            
            # Applying the binary operator 'and' (line 193)
            result_and_keyword_2699 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 36), 'and', result_and_keyword_2694, result_not__2698)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 193)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'stypy_return_type', result_and_keyword_2699)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_1' in the type store
            # Getting the type of 'stypy_return_type' (line 193)
            stypy_return_type_2700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_2700)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_1'
            return stypy_return_type_2700

        # Assigning a type to the variable '_stypy_temp_lambda_1' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
        # Getting the type of '_stypy_temp_lambda_1' (line 193)
        _stypy_temp_lambda_1_2701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), '_stypy_temp_lambda_1')
        
        # Call to keys(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_2704 = {}
        # Getting the type of 'var_dict' (line 194)
        var_dict_2702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 23), 'var_dict', False)
        # Obtaining the member 'keys' of a type (line 194)
        keys_2703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 23), var_dict_2702, 'keys')
        # Calling keys(args, kwargs) (line 194)
        keys_call_result_2705 = invoke(stypy.reporting.localization.Localization(__file__, 194, 23), keys_2703, *[], **kwargs_2704)
        
        # Processing the call keyword arguments (line 193)
        kwargs_2706 = {}
        # Getting the type of 'filter' (line 193)
        filter_2686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'filter', False)
        # Calling filter(args, kwargs) (line 193)
        filter_call_result_2707 = invoke(stypy.reporting.localization.Localization(__file__, 193, 16), filter_2686, *[_stypy_temp_lambda_1_2701, keys_call_result_2705], **kwargs_2706)
        
        # Assigning a type to the variable 'vars_' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'vars_', filter_call_result_2707)
        
        # Getting the type of 'self' (line 196)
        self_2708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'self')
        # Obtaining the member 'type_file_txt' of a type (line 196)
        type_file_txt_2709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), self_2708, 'type_file_txt')
        str_2710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 30), 'str', "    '")
        # Getting the type of 'context' (line 196)
        context_2711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 40), 'context')
        # Applying the binary operator '+' (line 196)
        result_add_2712 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 30), '+', str_2710, context_2711)
        
        str_2713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 50), 'str', "': {\n")
        # Applying the binary operator '+' (line 196)
        result_add_2714 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 48), '+', result_add_2712, str_2713)
        
        # Applying the binary operator '+=' (line 196)
        result_iadd_2715 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 8), '+=', type_file_txt_2709, result_add_2714)
        # Getting the type of 'self' (line 196)
        self_2716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'self')
        # Setting the type of the member 'type_file_txt' of a type (line 196)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), self_2716, 'type_file_txt', result_iadd_2715)
        
        
        # Getting the type of 'vars_' (line 197)
        vars__2717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'vars_')
        # Assigning a type to the variable 'vars__2717' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'vars__2717', vars__2717)
        # Testing if the for loop is going to be iterated (line 197)
        # Testing the type of a for loop iterable (line 197)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 197, 8), vars__2717)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 197, 8), vars__2717):
            # Getting the type of the for loop variable (line 197)
            for_loop_var_2718 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 197, 8), vars__2717)
            # Assigning a type to the variable 'var' (line 197)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'var', for_loop_var_2718)
            # SSA begins for a for statement (line 197)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'self' (line 198)
            self_2719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'self')
            # Obtaining the member 'type_file_txt' of a type (line 198)
            type_file_txt_2720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 12), self_2719, 'type_file_txt')
            str_2721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 34), 'str', "        '")
            # Getting the type of 'var' (line 198)
            var_2722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 48), 'var')
            # Applying the binary operator '+' (line 198)
            result_add_2723 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 34), '+', str_2721, var_2722)
            
            str_2724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 54), 'str', "': ")
            # Applying the binary operator '+' (line 198)
            result_add_2725 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 52), '+', result_add_2723, str_2724)
            
            
            # Call to get_type_name(...): (line 198)
            # Processing the call arguments (line 198)
            
            # Call to type(...): (line 198)
            # Processing the call arguments (line 198)
            
            # Obtaining the type of the subscript
            # Getting the type of 'var' (line 198)
            var_2728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 90), 'var', False)
            # Getting the type of 'var_dict' (line 198)
            var_dict_2729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 81), 'var_dict', False)
            # Obtaining the member '__getitem__' of a type (line 198)
            getitem___2730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 81), var_dict_2729, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 198)
            subscript_call_result_2731 = invoke(stypy.reporting.localization.Localization(__file__, 198, 81), getitem___2730, var_2728)
            
            # Processing the call keyword arguments (line 198)
            kwargs_2732 = {}
            # Getting the type of 'type' (line 198)
            type_2727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 76), 'type', False)
            # Calling type(args, kwargs) (line 198)
            type_call_result_2733 = invoke(stypy.reporting.localization.Localization(__file__, 198, 76), type_2727, *[subscript_call_result_2731], **kwargs_2732)
            
            # Processing the call keyword arguments (line 198)
            kwargs_2734 = {}
            # Getting the type of 'get_type_name' (line 198)
            get_type_name_2726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 62), 'get_type_name', False)
            # Calling get_type_name(args, kwargs) (line 198)
            get_type_name_call_result_2735 = invoke(stypy.reporting.localization.Localization(__file__, 198, 62), get_type_name_2726, *[type_call_result_2733], **kwargs_2734)
            
            # Applying the binary operator '+' (line 198)
            result_add_2736 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 60), '+', result_add_2725, get_type_name_call_result_2735)
            
            str_2737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 99), 'str', ', \n')
            # Applying the binary operator '+' (line 198)
            result_add_2738 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 97), '+', result_add_2736, str_2737)
            
            # Applying the binary operator '+=' (line 198)
            result_iadd_2739 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 12), '+=', type_file_txt_2720, result_add_2738)
            # Getting the type of 'self' (line 198)
            self_2740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'self')
            # Setting the type of the member 'type_file_txt' of a type (line 198)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 12), self_2740, 'type_file_txt', result_iadd_2739)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'self' (line 200)
        self_2741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'self')
        # Obtaining the member 'type_file_txt' of a type (line 200)
        type_file_txt_2742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), self_2741, 'type_file_txt')
        str_2743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 30), 'str', '    ')
        str_2744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 39), 'str', '}, \n')
        # Applying the binary operator '+' (line 200)
        result_add_2745 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 30), '+', str_2743, str_2744)
        
        # Applying the binary operator '+=' (line 200)
        result_iadd_2746 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 8), '+=', type_file_txt_2742, result_add_2745)
        # Getting the type of 'self' (line 200)
        self_2747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'self')
        # Setting the type of the member 'type_file_txt' of a type (line 200)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 8), self_2747, 'type_file_txt', result_iadd_2746)
        
        
        # Call to append(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'context' (line 202)
        context_2751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 47), 'context', False)
        # Processing the call keyword arguments (line 202)
        kwargs_2752 = {}
        # Getting the type of 'self' (line 202)
        self_2748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self', False)
        # Obtaining the member 'already_processed_contexts' of a type (line 202)
        already_processed_contexts_2749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_2748, 'already_processed_contexts')
        # Obtaining the member 'append' of a type (line 202)
        append_2750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), already_processed_contexts_2749, 'append')
        # Calling append(args, kwargs) (line 202)
        append_call_result_2753 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), append_2750, *[context_2751], **kwargs_2752)
        
        
        # ################# End of '__add_type_dict_for_context(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add_type_dict_for_context' in the type store
        # Getting the type of 'stypy_return_type' (line 183)
        stypy_return_type_2754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2754)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add_type_dict_for_context'
        return stypy_return_type_2754


    @norecursion
    def generate_type_data_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'generate_type_data_file'
        module_type_store = module_type_store.open_function_context('generate_type_data_file', 204, 4, False)
        # Assigning a type to the variable 'self' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeDataFileWriter.generate_type_data_file.__dict__.__setitem__('stypy_localization', localization)
        TypeDataFileWriter.generate_type_data_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeDataFileWriter.generate_type_data_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeDataFileWriter.generate_type_data_file.__dict__.__setitem__('stypy_function_name', 'TypeDataFileWriter.generate_type_data_file')
        TypeDataFileWriter.generate_type_data_file.__dict__.__setitem__('stypy_param_names_list', [])
        TypeDataFileWriter.generate_type_data_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeDataFileWriter.generate_type_data_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeDataFileWriter.generate_type_data_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeDataFileWriter.generate_type_data_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeDataFileWriter.generate_type_data_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeDataFileWriter.generate_type_data_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeDataFileWriter.generate_type_data_file', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generate_type_data_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generate_type_data_file(...)' code ##################

        str_2755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, (-1)), 'str', '\n        Generates the type data file\n        :return:\n        ')
        
        # Getting the type of 'self' (line 211)
        self_2756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'self')
        # Obtaining the member 'type_file_txt' of a type (line 211)
        type_file_txt_2757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), self_2756, 'type_file_txt')
        str_2758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 30), 'str', '}\n')
        # Applying the binary operator '+=' (line 211)
        result_iadd_2759 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 8), '+=', type_file_txt_2757, str_2758)
        # Getting the type of 'self' (line 211)
        self_2760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'self')
        # Setting the type of the member 'type_file_txt' of a type (line 211)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 8), self_2760, 'type_file_txt', result_iadd_2759)
        
        
        # Call to open(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'self' (line 212)
        self_2762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 18), 'self', False)
        # Obtaining the member 'dest_folder' of a type (line 212)
        dest_folder_2763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 18), self_2762, 'dest_folder')
        str_2764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 37), 'str', '/')
        # Applying the binary operator '+' (line 212)
        result_add_2765 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 18), '+', dest_folder_2763, str_2764)
        
        # Getting the type of 'self' (line 212)
        self_2766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 43), 'self', False)
        # Obtaining the member 'type_file' of a type (line 212)
        type_file_2767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 43), self_2766, 'type_file')
        # Applying the binary operator '+' (line 212)
        result_add_2768 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 41), '+', result_add_2765, type_file_2767)
        
        str_2769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 59), 'str', 'w')
        # Processing the call keyword arguments (line 212)
        kwargs_2770 = {}
        # Getting the type of 'open' (line 212)
        open_2761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 13), 'open', False)
        # Calling open(args, kwargs) (line 212)
        open_call_result_2771 = invoke(stypy.reporting.localization.Localization(__file__, 212, 13), open_2761, *[result_add_2768, str_2769], **kwargs_2770)
        
        with_2772 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 212, 13), open_call_result_2771, 'with parameter', '__enter__', '__exit__')

        if with_2772:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 212)
            enter___2773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 13), open_call_result_2771, '__enter__')
            with_enter_2774 = invoke(stypy.reporting.localization.Localization(__file__, 212, 13), enter___2773)
            # Assigning a type to the variable 'outfile' (line 212)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 13), 'outfile', with_enter_2774)
            
            # Call to write(...): (line 213)
            # Processing the call arguments (line 213)
            # Getting the type of 'self' (line 213)
            self_2777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 26), 'self', False)
            # Obtaining the member 'type_file_txt' of a type (line 213)
            type_file_txt_2778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 26), self_2777, 'type_file_txt')
            # Processing the call keyword arguments (line 213)
            kwargs_2779 = {}
            # Getting the type of 'outfile' (line 213)
            outfile_2775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'outfile', False)
            # Obtaining the member 'write' of a type (line 213)
            write_2776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 12), outfile_2775, 'write')
            # Calling write(args, kwargs) (line 213)
            write_call_result_2780 = invoke(stypy.reporting.localization.Localization(__file__, 213, 12), write_2776, *[type_file_txt_2778], **kwargs_2779)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 212)
            exit___2781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 13), open_call_result_2771, '__exit__')
            with_exit_2782 = invoke(stypy.reporting.localization.Localization(__file__, 212, 13), exit___2781, None, None, None)

        
        # ################# End of 'generate_type_data_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generate_type_data_file' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_2783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2783)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generate_type_data_file'
        return stypy_return_type_2783


# Assigning a type to the variable 'TypeDataFileWriter' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'TypeDataFileWriter', TypeDataFileWriter)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
