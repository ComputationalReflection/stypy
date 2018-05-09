
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import ast
2: import collections
3: 
4: from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import stypy_functions_copy
5: from stypy_copy.reporting_copy import print_utils_copy
6: from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy
7: from stypy_copy.type_store_copy.type_annotation_record_copy import TypeAnnotationRecord
8: 
9: 
10: class TypeAnnotationVisitor(ast.NodeVisitor):
11:     '''
12:     This visitor is used to generate a version of the original Python source code with type annotations attached
13:     to each of its lines, for those lines that have a type change for any of its variables recorded. It uses the
14:     information stored in a TypeAnnotationRecord instance attached to the processed file to extract annotations for
15:     a certain line, merge them and print them in comments just before the source code line that has the variables
16:     of the annotations.
17:     '''
18: 
19:     def __init__(self, file_name, type_store):
20:         file_name = file_name.replace("\\", "/")
21:         self.file_name = file_name
22:         self.type_store = type_store
23:         self.fcontexts = type_store.get_all_processed_function_contexts()
24: 
25:     @staticmethod
26:     def __mergue_annotations(annotations):
27:         '''
28:         Picks annotations stored in a list of tuples and merge those belonging to the same variable, creating
29:         union types if necessary (same variable with more than one type)
30:         :param annotations:
31:         :return:
32:         '''
33:         str_annotation = ""
34:         vars_dict = dict()
35:         for tuple_ in annotations:
36:             if not print_utils_copy.is_private_variable_name(tuple_[0]):
37:                 if tuple_[0] not in vars_dict:
38:                     vars_dict[tuple_[0]] = tuple_[1]
39:                 else:
40:                     vars_dict[tuple_[0]] = union_type_copy.UnionType.add(vars_dict[tuple_[0]], tuple_[1])
41: 
42:         for (name, type) in vars_dict.items():
43:             str_annotation += str(name) + ": " + print_utils_copy.get_type_str(type) + "; "
44: 
45:         if len(str_annotation) > 2:
46:             str_annotation = str_annotation[:-2]
47: 
48:         return str_annotation
49: 
50:     def __get_type_annotations(self, line):
51:         '''
52:         Get the type annotations associated with a source code line of the original Python program
53:         :param line: Line number
54:         :return: str with the formatted annotations, ready to be written
55:         '''
56:         str_annotation = ""
57:         all_annotations = []
58:         for fcontext in self.fcontexts:
59:             annotations = TypeAnnotationRecord.get_instance_for_file(self.file_name).get_annotations_for_line(line)
60:             if annotations is not None:
61:                 all_annotations.extend(annotations)
62:                 # str_annotation += self.__mergue_annotations(annotations)
63:                 # for tuple_ in annotations:
64:                 #     if not print_utils.is_private_variable_name(tuple_[0]):
65:                 #         str_annotation += str(tuple_[0]) + ": " + print_utils.get_type_str(tuple_[1]) + "; "
66: 
67:         # if len(str_annotation) > 2:
68:         #     str_annotation = str_annotation[:-2]
69:         str_annotation = self.__mergue_annotations(all_annotations)
70:         return str_annotation
71: 
72:     def __get_type_annotations_for_function(self, fname, line):
73:         '''
74:         Gets the annotations belonging to a certain function whose name is fname and is declared in the passed source
75:         code line, to avoid obtaining the wrong function in case there are multiple functions with the same name.
76:         This is used to annotate the possible types of the parameters of a function, checking all the calls that this
77:         function has during the program execution
78:         :param fname: Function name
79:         :param line: Source code line
80:         :return: str with the parameters of the functions and its annotated types
81:         '''
82:         str_annotation = ""
83:         for fcontext in self.fcontexts:
84:             if fcontext.function_name == fname and fcontext.declaration_line == line:
85:                 header_str = fcontext.get_header_str()
86:                 if header_str not in str_annotation:
87:                     str_annotation += header_str + " /\ "
88: 
89:         if len(str_annotation) > 2:
90:             str_annotation = str_annotation[:-3]
91: 
92:         return str_annotation
93: 
94:     def __visit_instruction_body(self, body):
95:         '''
96:         Visits all the instructions of a body, calculating its possible type annotations, turning it AST comment nodes
97:         and returning a list with the comment node and the original node. This way each source code line with
98:         annotations will appear in the generated file just below a comment with its annotations.
99:         :param body: Body of instructions
100:         :return: list
101:         '''
102:         new_stmts = []
103: 
104:         annotations = []
105:         # Visit all body instructions
106:         for stmt in body:
107:             stmts = self.visit(stmt)
108:             if hasattr(stmt, "lineno"):
109:                 annotations = self.__get_type_annotations(stmt.lineno)
110:                 if not annotations == "":
111:                     annotations = stypy_functions_copy.create_src_comment(annotations)
112:                     stmts = stypy_functions_copy.flatten_lists(annotations, stmts)
113: 
114:             if isinstance(stmts, list):
115:                 new_stmts.extend(stmts)
116:             else:
117:                 new_stmts.append(stmts)
118: 
119:         return new_stmts
120: 
121:     '''
122:     The rest of visit_ methods belong to those nodes that may have instruction bodies. These bodies are processed by
123:     the previous function so any instruction can have its possible type annotations generated. All follow the same
124:     coding pattern.
125:     '''
126: 
127:     def generic_visit(self, node):
128:         if hasattr(node, 'body'):
129:             if isinstance(node.body, collections.Iterable):
130:                 stmts = self.__visit_instruction_body(node.body)
131:             else:
132:                 stmts = self.__visit_instruction_body([node.body])
133: 
134:             node.body = stmts
135: 
136:         if hasattr(node, 'orelse'):
137:             if isinstance(node.orelse, collections.Iterable):
138:                 stmts = self.__visit_instruction_body(node.orelse)
139:             else:
140:                 stmts = self.__visit_instruction_body([node.orelse])
141: 
142:             node.orelse = stmts
143: 
144:         return node
145: 
146:     # ######################################### MAIN MODULE #############################################
147: 
148:     def visit_Module(self, node):
149:         stmts = self.__visit_instruction_body(node.body)
150: 
151:         node.body = stmts
152:         return node
153: 
154:     # ######################################### FUNCTIONS #############################################
155: 
156:     def visit_FunctionDef(self, node):
157:         annotations = self.__get_type_annotations_for_function(node.name, node.lineno)
158:         stmts = self.__visit_instruction_body(node.body)
159:         node.body = stmts
160: 
161:         if not annotations == "":
162:             annotations = stypy_functions_copy.create_src_comment(annotations)
163:         else:
164:             annotations = stypy_functions_copy.create_src_comment("<Dead code detected>")
165: 
166:         return stypy_functions_copy.flatten_lists(annotations, node)
167: 
168:     def visit_If(self, node):
169:         stmts = self.__visit_instruction_body(node.body)
170:         node.body = stmts
171: 
172:         stmts = self.__visit_instruction_body(node.orelse)
173:         node.orelse = stmts
174: 
175:         return node
176: 
177:     def visit_While(self, node):
178:         stmts = self.__visit_instruction_body(node.body)
179:         node.body = stmts
180: 
181:         stmts = self.__visit_instruction_body(node.orelse)
182:         node.orelse = stmts
183: 
184:         return node
185: 
186:     def visit_For(self, node):
187:         stmts = self.__visit_instruction_body(node.body)
188:         node.body = stmts
189: 
190:         stmts = self.__visit_instruction_body(node.orelse)
191:         node.orelse = stmts
192: 
193:         return node
194: 
195:     def visit_ClassDef(self, node):
196:         stmts = self.__visit_instruction_body(node.body)
197:         node.body = stmts
198: 
199:         return node
200: 
201:     def visit_With(self, node):
202:         stmts = self.__visit_instruction_body(node.body)
203:         node.body = stmts
204: 
205:         return node
206: 
207:     def visit_TryExcept(self, node):
208:         stmts = self.__visit_instruction_body(node.body)
209:         node.body = stmts
210: 
211:         for handler in node.handlers:
212:             stmts = self.__visit_instruction_body(handler.body)
213:             handler.body = stmts
214: 
215:         stmts = self.__visit_instruction_body(node.orelse)
216:         node.orelse = stmts
217:         return node
218: 
219:     def visit_TryFinally(self, node):
220:         stmts = self.__visit_instruction_body(node.body)
221:         node.body = stmts
222: 
223:         stmts = self.__visit_instruction_body(node.finalbody)
224:         node.finalbody = stmts
225: 
226:         return node
227: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import ast' statement (line 1)
import ast

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'ast', ast, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import collections' statement (line 2)
import collections

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'collections', collections, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import stypy_functions_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_annotation_copy/')
import_5205 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy')

if (type(import_5205) is not StypyTypeError):

    if (import_5205 != 'pyd_module'):
        __import__(import_5205)
        sys_modules_5206 = sys.modules[import_5205]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', sys_modules_5206.module_type_store, module_type_store, ['stypy_functions_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_5206, sys_modules_5206.module_type_store, module_type_store)
    else:
        from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import stypy_functions_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', None, module_type_store, ['stypy_functions_copy'], [stypy_functions_copy])

else:
    # Assigning a type to the variable 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', import_5205)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_annotation_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from stypy_copy.reporting_copy import print_utils_copy' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_annotation_copy/')
import_5207 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.reporting_copy')

if (type(import_5207) is not StypyTypeError):

    if (import_5207 != 'pyd_module'):
        __import__(import_5207)
        sys_modules_5208 = sys.modules[import_5207]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.reporting_copy', sys_modules_5208.module_type_store, module_type_store, ['print_utils_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_5208, sys_modules_5208.module_type_store, module_type_store)
    else:
        from stypy_copy.reporting_copy import print_utils_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.reporting_copy', None, module_type_store, ['print_utils_copy'], [print_utils_copy])

else:
    # Assigning a type to the variable 'stypy_copy.reporting_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.reporting_copy', import_5207)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_annotation_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_annotation_copy/')
import_5209 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_5209) is not StypyTypeError):

    if (import_5209 != 'pyd_module'):
        __import__(import_5209)
        sys_modules_5210 = sys.modules[import_5209]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_5210.module_type_store, module_type_store, ['union_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_5210, sys_modules_5210.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['union_type_copy'], [union_type_copy])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_5209)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_annotation_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from stypy_copy.type_store_copy.type_annotation_record_copy import TypeAnnotationRecord' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_annotation_copy/')
import_5211 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.type_store_copy.type_annotation_record_copy')

if (type(import_5211) is not StypyTypeError):

    if (import_5211 != 'pyd_module'):
        __import__(import_5211)
        sys_modules_5212 = sys.modules[import_5211]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.type_store_copy.type_annotation_record_copy', sys_modules_5212.module_type_store, module_type_store, ['TypeAnnotationRecord'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_5212, sys_modules_5212.module_type_store, module_type_store)
    else:
        from stypy_copy.type_store_copy.type_annotation_record_copy import TypeAnnotationRecord

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.type_store_copy.type_annotation_record_copy', None, module_type_store, ['TypeAnnotationRecord'], [TypeAnnotationRecord])

else:
    # Assigning a type to the variable 'stypy_copy.type_store_copy.type_annotation_record_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.type_store_copy.type_annotation_record_copy', import_5211)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_annotation_copy/')

# Declaration of the 'TypeAnnotationVisitor' class
# Getting the type of 'ast' (line 10)
ast_5213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 28), 'ast')
# Obtaining the member 'NodeVisitor' of a type (line 10)
NodeVisitor_5214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 28), ast_5213, 'NodeVisitor')

class TypeAnnotationVisitor(NodeVisitor_5214, ):
    str_5215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\n    This visitor is used to generate a version of the original Python source code with type annotations attached\n    to each of its lines, for those lines that have a type change for any of its variables recorded. It uses the\n    information stored in a TypeAnnotationRecord instance attached to the processed file to extract annotations for\n    a certain line, merge them and print them in comments just before the source code line that has the variables\n    of the annotations.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationVisitor.__init__', ['file_name', 'type_store'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['file_name', 'type_store'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Name (line 20):
        
        # Call to replace(...): (line 20)
        # Processing the call arguments (line 20)
        str_5218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 38), 'str', '\\')
        str_5219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 44), 'str', '/')
        # Processing the call keyword arguments (line 20)
        kwargs_5220 = {}
        # Getting the type of 'file_name' (line 20)
        file_name_5216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 20), 'file_name', False)
        # Obtaining the member 'replace' of a type (line 20)
        replace_5217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 20), file_name_5216, 'replace')
        # Calling replace(args, kwargs) (line 20)
        replace_call_result_5221 = invoke(stypy.reporting.localization.Localization(__file__, 20, 20), replace_5217, *[str_5218, str_5219], **kwargs_5220)
        
        # Assigning a type to the variable 'file_name' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'file_name', replace_call_result_5221)
        
        # Assigning a Name to a Attribute (line 21):
        # Getting the type of 'file_name' (line 21)
        file_name_5222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 25), 'file_name')
        # Getting the type of 'self' (line 21)
        self_5223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self')
        # Setting the type of the member 'file_name' of a type (line 21)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_5223, 'file_name', file_name_5222)
        
        # Assigning a Name to a Attribute (line 22):
        # Getting the type of 'type_store' (line 22)
        type_store_5224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 26), 'type_store')
        # Getting the type of 'self' (line 22)
        self_5225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Setting the type of the member 'type_store' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_5225, 'type_store', type_store_5224)
        
        # Assigning a Call to a Attribute (line 23):
        
        # Call to get_all_processed_function_contexts(...): (line 23)
        # Processing the call keyword arguments (line 23)
        kwargs_5228 = {}
        # Getting the type of 'type_store' (line 23)
        type_store_5226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 25), 'type_store', False)
        # Obtaining the member 'get_all_processed_function_contexts' of a type (line 23)
        get_all_processed_function_contexts_5227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 25), type_store_5226, 'get_all_processed_function_contexts')
        # Calling get_all_processed_function_contexts(args, kwargs) (line 23)
        get_all_processed_function_contexts_call_result_5229 = invoke(stypy.reporting.localization.Localization(__file__, 23, 25), get_all_processed_function_contexts_5227, *[], **kwargs_5228)
        
        # Getting the type of 'self' (line 23)
        self_5230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Setting the type of the member 'fcontexts' of a type (line 23)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_5230, 'fcontexts', get_all_processed_function_contexts_call_result_5229)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @staticmethod
    @norecursion
    def __mergue_annotations(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mergue_annotations'
        module_type_store = module_type_store.open_function_context('__mergue_annotations', 25, 4, False)
        
        # Passed parameters checking function
        TypeAnnotationVisitor.__mergue_annotations.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationVisitor.__mergue_annotations.__dict__.__setitem__('stypy_type_of_self', None)
        TypeAnnotationVisitor.__mergue_annotations.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationVisitor.__mergue_annotations.__dict__.__setitem__('stypy_function_name', '__mergue_annotations')
        TypeAnnotationVisitor.__mergue_annotations.__dict__.__setitem__('stypy_param_names_list', ['annotations'])
        TypeAnnotationVisitor.__mergue_annotations.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationVisitor.__mergue_annotations.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationVisitor.__mergue_annotations.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationVisitor.__mergue_annotations.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationVisitor.__mergue_annotations.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationVisitor.__mergue_annotations.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, '__mergue_annotations', ['annotations'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__mergue_annotations', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__mergue_annotations(...)' code ##################

        str_5231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'str', '\n        Picks annotations stored in a list of tuples and merge those belonging to the same variable, creating\n        union types if necessary (same variable with more than one type)\n        :param annotations:\n        :return:\n        ')
        
        # Assigning a Str to a Name (line 33):
        str_5232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 25), 'str', '')
        # Assigning a type to the variable 'str_annotation' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'str_annotation', str_5232)
        
        # Assigning a Call to a Name (line 34):
        
        # Call to dict(...): (line 34)
        # Processing the call keyword arguments (line 34)
        kwargs_5234 = {}
        # Getting the type of 'dict' (line 34)
        dict_5233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'dict', False)
        # Calling dict(args, kwargs) (line 34)
        dict_call_result_5235 = invoke(stypy.reporting.localization.Localization(__file__, 34, 20), dict_5233, *[], **kwargs_5234)
        
        # Assigning a type to the variable 'vars_dict' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'vars_dict', dict_call_result_5235)
        
        # Getting the type of 'annotations' (line 35)
        annotations_5236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 22), 'annotations')
        # Assigning a type to the variable 'annotations_5236' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'annotations_5236', annotations_5236)
        # Testing if the for loop is going to be iterated (line 35)
        # Testing the type of a for loop iterable (line 35)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 35, 8), annotations_5236)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 35, 8), annotations_5236):
            # Getting the type of the for loop variable (line 35)
            for_loop_var_5237 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 35, 8), annotations_5236)
            # Assigning a type to the variable 'tuple_' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'tuple_', for_loop_var_5237)
            # SSA begins for a for statement (line 35)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to is_private_variable_name(...): (line 36)
            # Processing the call arguments (line 36)
            
            # Obtaining the type of the subscript
            int_5240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 68), 'int')
            # Getting the type of 'tuple_' (line 36)
            tuple__5241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 61), 'tuple_', False)
            # Obtaining the member '__getitem__' of a type (line 36)
            getitem___5242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 61), tuple__5241, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 36)
            subscript_call_result_5243 = invoke(stypy.reporting.localization.Localization(__file__, 36, 61), getitem___5242, int_5240)
            
            # Processing the call keyword arguments (line 36)
            kwargs_5244 = {}
            # Getting the type of 'print_utils_copy' (line 36)
            print_utils_copy_5238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'print_utils_copy', False)
            # Obtaining the member 'is_private_variable_name' of a type (line 36)
            is_private_variable_name_5239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 19), print_utils_copy_5238, 'is_private_variable_name')
            # Calling is_private_variable_name(args, kwargs) (line 36)
            is_private_variable_name_call_result_5245 = invoke(stypy.reporting.localization.Localization(__file__, 36, 19), is_private_variable_name_5239, *[subscript_call_result_5243], **kwargs_5244)
            
            # Applying the 'not' unary operator (line 36)
            result_not__5246 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 15), 'not', is_private_variable_name_call_result_5245)
            
            # Testing if the type of an if condition is none (line 36)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 36, 12), result_not__5246):
                pass
            else:
                
                # Testing the type of an if condition (line 36)
                if_condition_5247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 12), result_not__5246)
                # Assigning a type to the variable 'if_condition_5247' (line 36)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'if_condition_5247', if_condition_5247)
                # SSA begins for if statement (line 36)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Obtaining the type of the subscript
                int_5248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 26), 'int')
                # Getting the type of 'tuple_' (line 37)
                tuple__5249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 19), 'tuple_')
                # Obtaining the member '__getitem__' of a type (line 37)
                getitem___5250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 19), tuple__5249, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 37)
                subscript_call_result_5251 = invoke(stypy.reporting.localization.Localization(__file__, 37, 19), getitem___5250, int_5248)
                
                # Getting the type of 'vars_dict' (line 37)
                vars_dict_5252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 36), 'vars_dict')
                # Applying the binary operator 'notin' (line 37)
                result_contains_5253 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 19), 'notin', subscript_call_result_5251, vars_dict_5252)
                
                # Testing if the type of an if condition is none (line 37)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 37, 16), result_contains_5253):
                    
                    # Assigning a Call to a Subscript (line 40):
                    
                    # Call to add(...): (line 40)
                    # Processing the call arguments (line 40)
                    
                    # Obtaining the type of the subscript
                    
                    # Obtaining the type of the subscript
                    int_5267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 90), 'int')
                    # Getting the type of 'tuple_' (line 40)
                    tuple__5268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 83), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 40)
                    getitem___5269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 83), tuple__5268, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
                    subscript_call_result_5270 = invoke(stypy.reporting.localization.Localization(__file__, 40, 83), getitem___5269, int_5267)
                    
                    # Getting the type of 'vars_dict' (line 40)
                    vars_dict_5271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 73), 'vars_dict', False)
                    # Obtaining the member '__getitem__' of a type (line 40)
                    getitem___5272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 73), vars_dict_5271, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
                    subscript_call_result_5273 = invoke(stypy.reporting.localization.Localization(__file__, 40, 73), getitem___5272, subscript_call_result_5270)
                    
                    
                    # Obtaining the type of the subscript
                    int_5274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 102), 'int')
                    # Getting the type of 'tuple_' (line 40)
                    tuple__5275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 95), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 40)
                    getitem___5276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 95), tuple__5275, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
                    subscript_call_result_5277 = invoke(stypy.reporting.localization.Localization(__file__, 40, 95), getitem___5276, int_5274)
                    
                    # Processing the call keyword arguments (line 40)
                    kwargs_5278 = {}
                    # Getting the type of 'union_type_copy' (line 40)
                    union_type_copy_5264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 43), 'union_type_copy', False)
                    # Obtaining the member 'UnionType' of a type (line 40)
                    UnionType_5265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 43), union_type_copy_5264, 'UnionType')
                    # Obtaining the member 'add' of a type (line 40)
                    add_5266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 43), UnionType_5265, 'add')
                    # Calling add(args, kwargs) (line 40)
                    add_call_result_5279 = invoke(stypy.reporting.localization.Localization(__file__, 40, 43), add_5266, *[subscript_call_result_5273, subscript_call_result_5277], **kwargs_5278)
                    
                    # Getting the type of 'vars_dict' (line 40)
                    vars_dict_5280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'vars_dict')
                    
                    # Obtaining the type of the subscript
                    int_5281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 37), 'int')
                    # Getting the type of 'tuple_' (line 40)
                    tuple__5282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 30), 'tuple_')
                    # Obtaining the member '__getitem__' of a type (line 40)
                    getitem___5283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 30), tuple__5282, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
                    subscript_call_result_5284 = invoke(stypy.reporting.localization.Localization(__file__, 40, 30), getitem___5283, int_5281)
                    
                    # Storing an element on a container (line 40)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 20), vars_dict_5280, (subscript_call_result_5284, add_call_result_5279))
                else:
                    
                    # Testing the type of an if condition (line 37)
                    if_condition_5254 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 16), result_contains_5253)
                    # Assigning a type to the variable 'if_condition_5254' (line 37)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'if_condition_5254', if_condition_5254)
                    # SSA begins for if statement (line 37)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Subscript to a Subscript (line 38):
                    
                    # Obtaining the type of the subscript
                    int_5255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 50), 'int')
                    # Getting the type of 'tuple_' (line 38)
                    tuple__5256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 43), 'tuple_')
                    # Obtaining the member '__getitem__' of a type (line 38)
                    getitem___5257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 43), tuple__5256, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
                    subscript_call_result_5258 = invoke(stypy.reporting.localization.Localization(__file__, 38, 43), getitem___5257, int_5255)
                    
                    # Getting the type of 'vars_dict' (line 38)
                    vars_dict_5259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 20), 'vars_dict')
                    
                    # Obtaining the type of the subscript
                    int_5260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 37), 'int')
                    # Getting the type of 'tuple_' (line 38)
                    tuple__5261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 30), 'tuple_')
                    # Obtaining the member '__getitem__' of a type (line 38)
                    getitem___5262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 30), tuple__5261, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
                    subscript_call_result_5263 = invoke(stypy.reporting.localization.Localization(__file__, 38, 30), getitem___5262, int_5260)
                    
                    # Storing an element on a container (line 38)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 20), vars_dict_5259, (subscript_call_result_5263, subscript_call_result_5258))
                    # SSA branch for the else part of an if statement (line 37)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Subscript (line 40):
                    
                    # Call to add(...): (line 40)
                    # Processing the call arguments (line 40)
                    
                    # Obtaining the type of the subscript
                    
                    # Obtaining the type of the subscript
                    int_5267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 90), 'int')
                    # Getting the type of 'tuple_' (line 40)
                    tuple__5268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 83), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 40)
                    getitem___5269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 83), tuple__5268, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
                    subscript_call_result_5270 = invoke(stypy.reporting.localization.Localization(__file__, 40, 83), getitem___5269, int_5267)
                    
                    # Getting the type of 'vars_dict' (line 40)
                    vars_dict_5271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 73), 'vars_dict', False)
                    # Obtaining the member '__getitem__' of a type (line 40)
                    getitem___5272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 73), vars_dict_5271, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
                    subscript_call_result_5273 = invoke(stypy.reporting.localization.Localization(__file__, 40, 73), getitem___5272, subscript_call_result_5270)
                    
                    
                    # Obtaining the type of the subscript
                    int_5274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 102), 'int')
                    # Getting the type of 'tuple_' (line 40)
                    tuple__5275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 95), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 40)
                    getitem___5276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 95), tuple__5275, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
                    subscript_call_result_5277 = invoke(stypy.reporting.localization.Localization(__file__, 40, 95), getitem___5276, int_5274)
                    
                    # Processing the call keyword arguments (line 40)
                    kwargs_5278 = {}
                    # Getting the type of 'union_type_copy' (line 40)
                    union_type_copy_5264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 43), 'union_type_copy', False)
                    # Obtaining the member 'UnionType' of a type (line 40)
                    UnionType_5265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 43), union_type_copy_5264, 'UnionType')
                    # Obtaining the member 'add' of a type (line 40)
                    add_5266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 43), UnionType_5265, 'add')
                    # Calling add(args, kwargs) (line 40)
                    add_call_result_5279 = invoke(stypy.reporting.localization.Localization(__file__, 40, 43), add_5266, *[subscript_call_result_5273, subscript_call_result_5277], **kwargs_5278)
                    
                    # Getting the type of 'vars_dict' (line 40)
                    vars_dict_5280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'vars_dict')
                    
                    # Obtaining the type of the subscript
                    int_5281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 37), 'int')
                    # Getting the type of 'tuple_' (line 40)
                    tuple__5282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 30), 'tuple_')
                    # Obtaining the member '__getitem__' of a type (line 40)
                    getitem___5283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 30), tuple__5282, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 40)
                    subscript_call_result_5284 = invoke(stypy.reporting.localization.Localization(__file__, 40, 30), getitem___5283, int_5281)
                    
                    # Storing an element on a container (line 40)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 20), vars_dict_5280, (subscript_call_result_5284, add_call_result_5279))
                    # SSA join for if statement (line 37)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 36)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to items(...): (line 42)
        # Processing the call keyword arguments (line 42)
        kwargs_5287 = {}
        # Getting the type of 'vars_dict' (line 42)
        vars_dict_5285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'vars_dict', False)
        # Obtaining the member 'items' of a type (line 42)
        items_5286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 28), vars_dict_5285, 'items')
        # Calling items(args, kwargs) (line 42)
        items_call_result_5288 = invoke(stypy.reporting.localization.Localization(__file__, 42, 28), items_5286, *[], **kwargs_5287)
        
        # Assigning a type to the variable 'items_call_result_5288' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'items_call_result_5288', items_call_result_5288)
        # Testing if the for loop is going to be iterated (line 42)
        # Testing the type of a for loop iterable (line 42)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 42, 8), items_call_result_5288)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 42, 8), items_call_result_5288):
            # Getting the type of the for loop variable (line 42)
            for_loop_var_5289 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 42, 8), items_call_result_5288)
            # Assigning a type to the variable 'name' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 8), for_loop_var_5289, 2, 0))
            # Assigning a type to the variable 'type' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'type', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 8), for_loop_var_5289, 2, 1))
            # SSA begins for a for statement (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'str_annotation' (line 43)
            str_annotation_5290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'str_annotation')
            
            # Call to str(...): (line 43)
            # Processing the call arguments (line 43)
            # Getting the type of 'name' (line 43)
            name_5292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 34), 'name', False)
            # Processing the call keyword arguments (line 43)
            kwargs_5293 = {}
            # Getting the type of 'str' (line 43)
            str_5291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 30), 'str', False)
            # Calling str(args, kwargs) (line 43)
            str_call_result_5294 = invoke(stypy.reporting.localization.Localization(__file__, 43, 30), str_5291, *[name_5292], **kwargs_5293)
            
            str_5295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 42), 'str', ': ')
            # Applying the binary operator '+' (line 43)
            result_add_5296 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 30), '+', str_call_result_5294, str_5295)
            
            
            # Call to get_type_str(...): (line 43)
            # Processing the call arguments (line 43)
            # Getting the type of 'type' (line 43)
            type_5299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 79), 'type', False)
            # Processing the call keyword arguments (line 43)
            kwargs_5300 = {}
            # Getting the type of 'print_utils_copy' (line 43)
            print_utils_copy_5297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 49), 'print_utils_copy', False)
            # Obtaining the member 'get_type_str' of a type (line 43)
            get_type_str_5298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 49), print_utils_copy_5297, 'get_type_str')
            # Calling get_type_str(args, kwargs) (line 43)
            get_type_str_call_result_5301 = invoke(stypy.reporting.localization.Localization(__file__, 43, 49), get_type_str_5298, *[type_5299], **kwargs_5300)
            
            # Applying the binary operator '+' (line 43)
            result_add_5302 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 47), '+', result_add_5296, get_type_str_call_result_5301)
            
            str_5303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 87), 'str', '; ')
            # Applying the binary operator '+' (line 43)
            result_add_5304 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 85), '+', result_add_5302, str_5303)
            
            # Applying the binary operator '+=' (line 43)
            result_iadd_5305 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 12), '+=', str_annotation_5290, result_add_5304)
            # Assigning a type to the variable 'str_annotation' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'str_annotation', result_iadd_5305)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'str_annotation' (line 45)
        str_annotation_5307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'str_annotation', False)
        # Processing the call keyword arguments (line 45)
        kwargs_5308 = {}
        # Getting the type of 'len' (line 45)
        len_5306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'len', False)
        # Calling len(args, kwargs) (line 45)
        len_call_result_5309 = invoke(stypy.reporting.localization.Localization(__file__, 45, 11), len_5306, *[str_annotation_5307], **kwargs_5308)
        
        int_5310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 33), 'int')
        # Applying the binary operator '>' (line 45)
        result_gt_5311 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 11), '>', len_call_result_5309, int_5310)
        
        # Testing if the type of an if condition is none (line 45)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 45, 8), result_gt_5311):
            pass
        else:
            
            # Testing the type of an if condition (line 45)
            if_condition_5312 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 8), result_gt_5311)
            # Assigning a type to the variable 'if_condition_5312' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'if_condition_5312', if_condition_5312)
            # SSA begins for if statement (line 45)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 46):
            
            # Obtaining the type of the subscript
            int_5313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 45), 'int')
            slice_5314 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 46, 29), None, int_5313, None)
            # Getting the type of 'str_annotation' (line 46)
            str_annotation_5315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 29), 'str_annotation')
            # Obtaining the member '__getitem__' of a type (line 46)
            getitem___5316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 29), str_annotation_5315, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 46)
            subscript_call_result_5317 = invoke(stypy.reporting.localization.Localization(__file__, 46, 29), getitem___5316, slice_5314)
            
            # Assigning a type to the variable 'str_annotation' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'str_annotation', subscript_call_result_5317)
            # SSA join for if statement (line 45)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'str_annotation' (line 48)
        str_annotation_5318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'str_annotation')
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', str_annotation_5318)
        
        # ################# End of '__mergue_annotations(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mergue_annotations' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_5319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5319)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mergue_annotations'
        return stypy_return_type_5319


    @norecursion
    def __get_type_annotations(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__get_type_annotations'
        module_type_store = module_type_store.open_function_context('__get_type_annotations', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeAnnotationVisitor.__get_type_annotations.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationVisitor.__get_type_annotations.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeAnnotationVisitor.__get_type_annotations.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationVisitor.__get_type_annotations.__dict__.__setitem__('stypy_function_name', 'TypeAnnotationVisitor.__get_type_annotations')
        TypeAnnotationVisitor.__get_type_annotations.__dict__.__setitem__('stypy_param_names_list', ['line'])
        TypeAnnotationVisitor.__get_type_annotations.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationVisitor.__get_type_annotations.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationVisitor.__get_type_annotations.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationVisitor.__get_type_annotations.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationVisitor.__get_type_annotations.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationVisitor.__get_type_annotations.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationVisitor.__get_type_annotations', ['line'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__get_type_annotations', localization, ['line'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__get_type_annotations(...)' code ##################

        str_5320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'str', '\n        Get the type annotations associated with a source code line of the original Python program\n        :param line: Line number\n        :return: str with the formatted annotations, ready to be written\n        ')
        
        # Assigning a Str to a Name (line 56):
        str_5321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'str', '')
        # Assigning a type to the variable 'str_annotation' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'str_annotation', str_5321)
        
        # Assigning a List to a Name (line 57):
        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_5322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        
        # Assigning a type to the variable 'all_annotations' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'all_annotations', list_5322)
        
        # Getting the type of 'self' (line 58)
        self_5323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 24), 'self')
        # Obtaining the member 'fcontexts' of a type (line 58)
        fcontexts_5324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 24), self_5323, 'fcontexts')
        # Assigning a type to the variable 'fcontexts_5324' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'fcontexts_5324', fcontexts_5324)
        # Testing if the for loop is going to be iterated (line 58)
        # Testing the type of a for loop iterable (line 58)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 58, 8), fcontexts_5324)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 58, 8), fcontexts_5324):
            # Getting the type of the for loop variable (line 58)
            for_loop_var_5325 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 58, 8), fcontexts_5324)
            # Assigning a type to the variable 'fcontext' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'fcontext', for_loop_var_5325)
            # SSA begins for a for statement (line 58)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 59):
            
            # Call to get_annotations_for_line(...): (line 59)
            # Processing the call arguments (line 59)
            # Getting the type of 'line' (line 59)
            line_5333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 110), 'line', False)
            # Processing the call keyword arguments (line 59)
            kwargs_5334 = {}
            
            # Call to get_instance_for_file(...): (line 59)
            # Processing the call arguments (line 59)
            # Getting the type of 'self' (line 59)
            self_5328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 69), 'self', False)
            # Obtaining the member 'file_name' of a type (line 59)
            file_name_5329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 69), self_5328, 'file_name')
            # Processing the call keyword arguments (line 59)
            kwargs_5330 = {}
            # Getting the type of 'TypeAnnotationRecord' (line 59)
            TypeAnnotationRecord_5326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 26), 'TypeAnnotationRecord', False)
            # Obtaining the member 'get_instance_for_file' of a type (line 59)
            get_instance_for_file_5327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 26), TypeAnnotationRecord_5326, 'get_instance_for_file')
            # Calling get_instance_for_file(args, kwargs) (line 59)
            get_instance_for_file_call_result_5331 = invoke(stypy.reporting.localization.Localization(__file__, 59, 26), get_instance_for_file_5327, *[file_name_5329], **kwargs_5330)
            
            # Obtaining the member 'get_annotations_for_line' of a type (line 59)
            get_annotations_for_line_5332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 26), get_instance_for_file_call_result_5331, 'get_annotations_for_line')
            # Calling get_annotations_for_line(args, kwargs) (line 59)
            get_annotations_for_line_call_result_5335 = invoke(stypy.reporting.localization.Localization(__file__, 59, 26), get_annotations_for_line_5332, *[line_5333], **kwargs_5334)
            
            # Assigning a type to the variable 'annotations' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'annotations', get_annotations_for_line_call_result_5335)
            
            # Type idiom detected: calculating its left and rigth part (line 60)
            # Getting the type of 'annotations' (line 60)
            annotations_5336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'annotations')
            # Getting the type of 'None' (line 60)
            None_5337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'None')
            
            (may_be_5338, more_types_in_union_5339) = may_not_be_none(annotations_5336, None_5337)

            if may_be_5338:

                if more_types_in_union_5339:
                    # Runtime conditional SSA (line 60)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Call to extend(...): (line 61)
                # Processing the call arguments (line 61)
                # Getting the type of 'annotations' (line 61)
                annotations_5342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 39), 'annotations', False)
                # Processing the call keyword arguments (line 61)
                kwargs_5343 = {}
                # Getting the type of 'all_annotations' (line 61)
                all_annotations_5340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'all_annotations', False)
                # Obtaining the member 'extend' of a type (line 61)
                extend_5341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 16), all_annotations_5340, 'extend')
                # Calling extend(args, kwargs) (line 61)
                extend_call_result_5344 = invoke(stypy.reporting.localization.Localization(__file__, 61, 16), extend_5341, *[annotations_5342], **kwargs_5343)
                

                if more_types_in_union_5339:
                    # SSA join for if statement (line 60)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 69):
        
        # Call to __mergue_annotations(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'all_annotations' (line 69)
        all_annotations_5347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 51), 'all_annotations', False)
        # Processing the call keyword arguments (line 69)
        kwargs_5348 = {}
        # Getting the type of 'self' (line 69)
        self_5345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'self', False)
        # Obtaining the member '__mergue_annotations' of a type (line 69)
        mergue_annotations_5346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 25), self_5345, '__mergue_annotations')
        # Calling __mergue_annotations(args, kwargs) (line 69)
        mergue_annotations_call_result_5349 = invoke(stypy.reporting.localization.Localization(__file__, 69, 25), mergue_annotations_5346, *[all_annotations_5347], **kwargs_5348)
        
        # Assigning a type to the variable 'str_annotation' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'str_annotation', mergue_annotations_call_result_5349)
        # Getting the type of 'str_annotation' (line 70)
        str_annotation_5350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'str_annotation')
        # Assigning a type to the variable 'stypy_return_type' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'stypy_return_type', str_annotation_5350)
        
        # ################# End of '__get_type_annotations(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_type_annotations' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_5351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5351)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_type_annotations'
        return stypy_return_type_5351


    @norecursion
    def __get_type_annotations_for_function(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__get_type_annotations_for_function'
        module_type_store = module_type_store.open_function_context('__get_type_annotations_for_function', 72, 4, False)
        # Assigning a type to the variable 'self' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeAnnotationVisitor.__get_type_annotations_for_function.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationVisitor.__get_type_annotations_for_function.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeAnnotationVisitor.__get_type_annotations_for_function.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationVisitor.__get_type_annotations_for_function.__dict__.__setitem__('stypy_function_name', 'TypeAnnotationVisitor.__get_type_annotations_for_function')
        TypeAnnotationVisitor.__get_type_annotations_for_function.__dict__.__setitem__('stypy_param_names_list', ['fname', 'line'])
        TypeAnnotationVisitor.__get_type_annotations_for_function.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationVisitor.__get_type_annotations_for_function.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationVisitor.__get_type_annotations_for_function.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationVisitor.__get_type_annotations_for_function.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationVisitor.__get_type_annotations_for_function.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationVisitor.__get_type_annotations_for_function.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationVisitor.__get_type_annotations_for_function', ['fname', 'line'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__get_type_annotations_for_function', localization, ['fname', 'line'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__get_type_annotations_for_function(...)' code ##################

        str_5352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, (-1)), 'str', '\n        Gets the annotations belonging to a certain function whose name is fname and is declared in the passed source\n        code line, to avoid obtaining the wrong function in case there are multiple functions with the same name.\n        This is used to annotate the possible types of the parameters of a function, checking all the calls that this\n        function has during the program execution\n        :param fname: Function name\n        :param line: Source code line\n        :return: str with the parameters of the functions and its annotated types\n        ')
        
        # Assigning a Str to a Name (line 82):
        str_5353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 25), 'str', '')
        # Assigning a type to the variable 'str_annotation' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'str_annotation', str_5353)
        
        # Getting the type of 'self' (line 83)
        self_5354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 24), 'self')
        # Obtaining the member 'fcontexts' of a type (line 83)
        fcontexts_5355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 24), self_5354, 'fcontexts')
        # Assigning a type to the variable 'fcontexts_5355' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'fcontexts_5355', fcontexts_5355)
        # Testing if the for loop is going to be iterated (line 83)
        # Testing the type of a for loop iterable (line 83)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 83, 8), fcontexts_5355)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 83, 8), fcontexts_5355):
            # Getting the type of the for loop variable (line 83)
            for_loop_var_5356 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 83, 8), fcontexts_5355)
            # Assigning a type to the variable 'fcontext' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'fcontext', for_loop_var_5356)
            # SSA begins for a for statement (line 83)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Evaluating a boolean operation
            
            # Getting the type of 'fcontext' (line 84)
            fcontext_5357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'fcontext')
            # Obtaining the member 'function_name' of a type (line 84)
            function_name_5358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 15), fcontext_5357, 'function_name')
            # Getting the type of 'fname' (line 84)
            fname_5359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 41), 'fname')
            # Applying the binary operator '==' (line 84)
            result_eq_5360 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 15), '==', function_name_5358, fname_5359)
            
            
            # Getting the type of 'fcontext' (line 84)
            fcontext_5361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 51), 'fcontext')
            # Obtaining the member 'declaration_line' of a type (line 84)
            declaration_line_5362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 51), fcontext_5361, 'declaration_line')
            # Getting the type of 'line' (line 84)
            line_5363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 80), 'line')
            # Applying the binary operator '==' (line 84)
            result_eq_5364 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 51), '==', declaration_line_5362, line_5363)
            
            # Applying the binary operator 'and' (line 84)
            result_and_keyword_5365 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 15), 'and', result_eq_5360, result_eq_5364)
            
            # Testing if the type of an if condition is none (line 84)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 84, 12), result_and_keyword_5365):
                pass
            else:
                
                # Testing the type of an if condition (line 84)
                if_condition_5366 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 12), result_and_keyword_5365)
                # Assigning a type to the variable 'if_condition_5366' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'if_condition_5366', if_condition_5366)
                # SSA begins for if statement (line 84)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 85):
                
                # Call to get_header_str(...): (line 85)
                # Processing the call keyword arguments (line 85)
                kwargs_5369 = {}
                # Getting the type of 'fcontext' (line 85)
                fcontext_5367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'fcontext', False)
                # Obtaining the member 'get_header_str' of a type (line 85)
                get_header_str_5368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 29), fcontext_5367, 'get_header_str')
                # Calling get_header_str(args, kwargs) (line 85)
                get_header_str_call_result_5370 = invoke(stypy.reporting.localization.Localization(__file__, 85, 29), get_header_str_5368, *[], **kwargs_5369)
                
                # Assigning a type to the variable 'header_str' (line 85)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'header_str', get_header_str_call_result_5370)
                
                # Getting the type of 'header_str' (line 86)
                header_str_5371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'header_str')
                # Getting the type of 'str_annotation' (line 86)
                str_annotation_5372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 37), 'str_annotation')
                # Applying the binary operator 'notin' (line 86)
                result_contains_5373 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 19), 'notin', header_str_5371, str_annotation_5372)
                
                # Testing if the type of an if condition is none (line 86)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 86, 16), result_contains_5373):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 86)
                    if_condition_5374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 16), result_contains_5373)
                    # Assigning a type to the variable 'if_condition_5374' (line 86)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'if_condition_5374', if_condition_5374)
                    # SSA begins for if statement (line 86)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'str_annotation' (line 87)
                    str_annotation_5375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'str_annotation')
                    # Getting the type of 'header_str' (line 87)
                    header_str_5376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 38), 'header_str')
                    str_5377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 51), 'str', ' /\\ ')
                    # Applying the binary operator '+' (line 87)
                    result_add_5378 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 38), '+', header_str_5376, str_5377)
                    
                    # Applying the binary operator '+=' (line 87)
                    result_iadd_5379 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 20), '+=', str_annotation_5375, result_add_5378)
                    # Assigning a type to the variable 'str_annotation' (line 87)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'str_annotation', result_iadd_5379)
                    
                    # SSA join for if statement (line 86)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 84)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        
        # Call to len(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'str_annotation' (line 89)
        str_annotation_5381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'str_annotation', False)
        # Processing the call keyword arguments (line 89)
        kwargs_5382 = {}
        # Getting the type of 'len' (line 89)
        len_5380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'len', False)
        # Calling len(args, kwargs) (line 89)
        len_call_result_5383 = invoke(stypy.reporting.localization.Localization(__file__, 89, 11), len_5380, *[str_annotation_5381], **kwargs_5382)
        
        int_5384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 33), 'int')
        # Applying the binary operator '>' (line 89)
        result_gt_5385 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 11), '>', len_call_result_5383, int_5384)
        
        # Testing if the type of an if condition is none (line 89)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 8), result_gt_5385):
            pass
        else:
            
            # Testing the type of an if condition (line 89)
            if_condition_5386 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 8), result_gt_5385)
            # Assigning a type to the variable 'if_condition_5386' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'if_condition_5386', if_condition_5386)
            # SSA begins for if statement (line 89)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Name (line 90):
            
            # Obtaining the type of the subscript
            int_5387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 45), 'int')
            slice_5388 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 90, 29), None, int_5387, None)
            # Getting the type of 'str_annotation' (line 90)
            str_annotation_5389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 29), 'str_annotation')
            # Obtaining the member '__getitem__' of a type (line 90)
            getitem___5390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 29), str_annotation_5389, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 90)
            subscript_call_result_5391 = invoke(stypy.reporting.localization.Localization(__file__, 90, 29), getitem___5390, slice_5388)
            
            # Assigning a type to the variable 'str_annotation' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'str_annotation', subscript_call_result_5391)
            # SSA join for if statement (line 89)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'str_annotation' (line 92)
        str_annotation_5392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'str_annotation')
        # Assigning a type to the variable 'stypy_return_type' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'stypy_return_type', str_annotation_5392)
        
        # ################# End of '__get_type_annotations_for_function(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_type_annotations_for_function' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_5393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5393)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_type_annotations_for_function'
        return stypy_return_type_5393


    @norecursion
    def __visit_instruction_body(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__visit_instruction_body'
        module_type_store = module_type_store.open_function_context('__visit_instruction_body', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeAnnotationVisitor.__visit_instruction_body.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationVisitor.__visit_instruction_body.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeAnnotationVisitor.__visit_instruction_body.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationVisitor.__visit_instruction_body.__dict__.__setitem__('stypy_function_name', 'TypeAnnotationVisitor.__visit_instruction_body')
        TypeAnnotationVisitor.__visit_instruction_body.__dict__.__setitem__('stypy_param_names_list', ['body'])
        TypeAnnotationVisitor.__visit_instruction_body.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationVisitor.__visit_instruction_body.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationVisitor.__visit_instruction_body.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationVisitor.__visit_instruction_body.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationVisitor.__visit_instruction_body.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationVisitor.__visit_instruction_body.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationVisitor.__visit_instruction_body', ['body'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__visit_instruction_body', localization, ['body'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__visit_instruction_body(...)' code ##################

        str_5394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, (-1)), 'str', '\n        Visits all the instructions of a body, calculating its possible type annotations, turning it AST comment nodes\n        and returning a list with the comment node and the original node. This way each source code line with\n        annotations will appear in the generated file just below a comment with its annotations.\n        :param body: Body of instructions\n        :return: list\n        ')
        
        # Assigning a List to a Name (line 102):
        
        # Obtaining an instance of the builtin type 'list' (line 102)
        list_5395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 102)
        
        # Assigning a type to the variable 'new_stmts' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'new_stmts', list_5395)
        
        # Assigning a List to a Name (line 104):
        
        # Obtaining an instance of the builtin type 'list' (line 104)
        list_5396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 104)
        
        # Assigning a type to the variable 'annotations' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'annotations', list_5396)
        
        # Getting the type of 'body' (line 106)
        body_5397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'body')
        # Assigning a type to the variable 'body_5397' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'body_5397', body_5397)
        # Testing if the for loop is going to be iterated (line 106)
        # Testing the type of a for loop iterable (line 106)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 106, 8), body_5397)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 106, 8), body_5397):
            # Getting the type of the for loop variable (line 106)
            for_loop_var_5398 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 106, 8), body_5397)
            # Assigning a type to the variable 'stmt' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'stmt', for_loop_var_5398)
            # SSA begins for a for statement (line 106)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 107):
            
            # Call to visit(...): (line 107)
            # Processing the call arguments (line 107)
            # Getting the type of 'stmt' (line 107)
            stmt_5401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 31), 'stmt', False)
            # Processing the call keyword arguments (line 107)
            kwargs_5402 = {}
            # Getting the type of 'self' (line 107)
            self_5399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 20), 'self', False)
            # Obtaining the member 'visit' of a type (line 107)
            visit_5400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 20), self_5399, 'visit')
            # Calling visit(args, kwargs) (line 107)
            visit_call_result_5403 = invoke(stypy.reporting.localization.Localization(__file__, 107, 20), visit_5400, *[stmt_5401], **kwargs_5402)
            
            # Assigning a type to the variable 'stmts' (line 107)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'stmts', visit_call_result_5403)
            
            # Type idiom detected: calculating its left and rigth part (line 108)
            str_5404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 29), 'str', 'lineno')
            # Getting the type of 'stmt' (line 108)
            stmt_5405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), 'stmt')
            
            (may_be_5406, more_types_in_union_5407) = may_provide_member(str_5404, stmt_5405)

            if may_be_5406:

                if more_types_in_union_5407:
                    # Runtime conditional SSA (line 108)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'stmt' (line 108)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'stmt', remove_not_member_provider_from_union(stmt_5405, 'lineno'))
                
                # Assigning a Call to a Name (line 109):
                
                # Call to __get_type_annotations(...): (line 109)
                # Processing the call arguments (line 109)
                # Getting the type of 'stmt' (line 109)
                stmt_5410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 58), 'stmt', False)
                # Obtaining the member 'lineno' of a type (line 109)
                lineno_5411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 58), stmt_5410, 'lineno')
                # Processing the call keyword arguments (line 109)
                kwargs_5412 = {}
                # Getting the type of 'self' (line 109)
                self_5408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 30), 'self', False)
                # Obtaining the member '__get_type_annotations' of a type (line 109)
                get_type_annotations_5409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 30), self_5408, '__get_type_annotations')
                # Calling __get_type_annotations(args, kwargs) (line 109)
                get_type_annotations_call_result_5413 = invoke(stypy.reporting.localization.Localization(__file__, 109, 30), get_type_annotations_5409, *[lineno_5411], **kwargs_5412)
                
                # Assigning a type to the variable 'annotations' (line 109)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'annotations', get_type_annotations_call_result_5413)
                
                
                # Getting the type of 'annotations' (line 110)
                annotations_5414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'annotations')
                str_5415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 38), 'str', '')
                # Applying the binary operator '==' (line 110)
                result_eq_5416 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 23), '==', annotations_5414, str_5415)
                
                # Applying the 'not' unary operator (line 110)
                result_not__5417 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 19), 'not', result_eq_5416)
                
                # Testing if the type of an if condition is none (line 110)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 110, 16), result_not__5417):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 110)
                    if_condition_5418 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 16), result_not__5417)
                    # Assigning a type to the variable 'if_condition_5418' (line 110)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'if_condition_5418', if_condition_5418)
                    # SSA begins for if statement (line 110)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 111):
                    
                    # Call to create_src_comment(...): (line 111)
                    # Processing the call arguments (line 111)
                    # Getting the type of 'annotations' (line 111)
                    annotations_5421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 74), 'annotations', False)
                    # Processing the call keyword arguments (line 111)
                    kwargs_5422 = {}
                    # Getting the type of 'stypy_functions_copy' (line 111)
                    stypy_functions_copy_5419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 34), 'stypy_functions_copy', False)
                    # Obtaining the member 'create_src_comment' of a type (line 111)
                    create_src_comment_5420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 34), stypy_functions_copy_5419, 'create_src_comment')
                    # Calling create_src_comment(args, kwargs) (line 111)
                    create_src_comment_call_result_5423 = invoke(stypy.reporting.localization.Localization(__file__, 111, 34), create_src_comment_5420, *[annotations_5421], **kwargs_5422)
                    
                    # Assigning a type to the variable 'annotations' (line 111)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'annotations', create_src_comment_call_result_5423)
                    
                    # Assigning a Call to a Name (line 112):
                    
                    # Call to flatten_lists(...): (line 112)
                    # Processing the call arguments (line 112)
                    # Getting the type of 'annotations' (line 112)
                    annotations_5426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 63), 'annotations', False)
                    # Getting the type of 'stmts' (line 112)
                    stmts_5427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 76), 'stmts', False)
                    # Processing the call keyword arguments (line 112)
                    kwargs_5428 = {}
                    # Getting the type of 'stypy_functions_copy' (line 112)
                    stypy_functions_copy_5424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'stypy_functions_copy', False)
                    # Obtaining the member 'flatten_lists' of a type (line 112)
                    flatten_lists_5425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 28), stypy_functions_copy_5424, 'flatten_lists')
                    # Calling flatten_lists(args, kwargs) (line 112)
                    flatten_lists_call_result_5429 = invoke(stypy.reporting.localization.Localization(__file__, 112, 28), flatten_lists_5425, *[annotations_5426, stmts_5427], **kwargs_5428)
                    
                    # Assigning a type to the variable 'stmts' (line 112)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'stmts', flatten_lists_call_result_5429)
                    # SSA join for if statement (line 110)
                    module_type_store = module_type_store.join_ssa_context()
                    


                if more_types_in_union_5407:
                    # SSA join for if statement (line 108)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Type idiom detected: calculating its left and rigth part (line 114)
            # Getting the type of 'list' (line 114)
            list_5430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 33), 'list')
            # Getting the type of 'stmts' (line 114)
            stmts_5431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 26), 'stmts')
            
            (may_be_5432, more_types_in_union_5433) = may_be_subtype(list_5430, stmts_5431)

            if may_be_5432:

                if more_types_in_union_5433:
                    # Runtime conditional SSA (line 114)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'stmts' (line 114)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'stmts', remove_not_subtype_from_union(stmts_5431, list))
                
                # Call to extend(...): (line 115)
                # Processing the call arguments (line 115)
                # Getting the type of 'stmts' (line 115)
                stmts_5436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 33), 'stmts', False)
                # Processing the call keyword arguments (line 115)
                kwargs_5437 = {}
                # Getting the type of 'new_stmts' (line 115)
                new_stmts_5434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'new_stmts', False)
                # Obtaining the member 'extend' of a type (line 115)
                extend_5435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 16), new_stmts_5434, 'extend')
                # Calling extend(args, kwargs) (line 115)
                extend_call_result_5438 = invoke(stypy.reporting.localization.Localization(__file__, 115, 16), extend_5435, *[stmts_5436], **kwargs_5437)
                

                if more_types_in_union_5433:
                    # Runtime conditional SSA for else branch (line 114)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_5432) or more_types_in_union_5433):
                # Assigning a type to the variable 'stmts' (line 114)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'stmts', remove_subtype_from_union(stmts_5431, list))
                
                # Call to append(...): (line 117)
                # Processing the call arguments (line 117)
                # Getting the type of 'stmts' (line 117)
                stmts_5441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 33), 'stmts', False)
                # Processing the call keyword arguments (line 117)
                kwargs_5442 = {}
                # Getting the type of 'new_stmts' (line 117)
                new_stmts_5439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'new_stmts', False)
                # Obtaining the member 'append' of a type (line 117)
                append_5440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 16), new_stmts_5439, 'append')
                # Calling append(args, kwargs) (line 117)
                append_call_result_5443 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), append_5440, *[stmts_5441], **kwargs_5442)
                

                if (may_be_5432 and more_types_in_union_5433):
                    # SSA join for if statement (line 114)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'new_stmts' (line 119)
        new_stmts_5444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'new_stmts')
        # Assigning a type to the variable 'stypy_return_type' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'stypy_return_type', new_stmts_5444)
        
        # ################# End of '__visit_instruction_body(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__visit_instruction_body' in the type store
        # Getting the type of 'stypy_return_type' (line 94)
        stypy_return_type_5445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5445)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__visit_instruction_body'
        return stypy_return_type_5445

    str_5446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, (-1)), 'str', '\n    The rest of visit_ methods belong to those nodes that may have instruction bodies. These bodies are processed by\n    the previous function so any instruction can have its possible type annotations generated. All follow the same\n    coding pattern.\n    ')

    @norecursion
    def generic_visit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'generic_visit'
        module_type_store = module_type_store.open_function_context('generic_visit', 127, 4, False)
        # Assigning a type to the variable 'self' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeAnnotationVisitor.generic_visit.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationVisitor.generic_visit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeAnnotationVisitor.generic_visit.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationVisitor.generic_visit.__dict__.__setitem__('stypy_function_name', 'TypeAnnotationVisitor.generic_visit')
        TypeAnnotationVisitor.generic_visit.__dict__.__setitem__('stypy_param_names_list', ['node'])
        TypeAnnotationVisitor.generic_visit.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationVisitor.generic_visit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationVisitor.generic_visit.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationVisitor.generic_visit.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationVisitor.generic_visit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationVisitor.generic_visit.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationVisitor.generic_visit', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generic_visit', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generic_visit(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 128)
        str_5447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 25), 'str', 'body')
        # Getting the type of 'node' (line 128)
        node_5448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'node')
        
        (may_be_5449, more_types_in_union_5450) = may_provide_member(str_5447, node_5448)

        if may_be_5449:

            if more_types_in_union_5450:
                # Runtime conditional SSA (line 128)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'node' (line 128)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'node', remove_not_member_provider_from_union(node_5448, 'body'))
            
            # Call to isinstance(...): (line 129)
            # Processing the call arguments (line 129)
            # Getting the type of 'node' (line 129)
            node_5452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 26), 'node', False)
            # Obtaining the member 'body' of a type (line 129)
            body_5453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 26), node_5452, 'body')
            # Getting the type of 'collections' (line 129)
            collections_5454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 37), 'collections', False)
            # Obtaining the member 'Iterable' of a type (line 129)
            Iterable_5455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 37), collections_5454, 'Iterable')
            # Processing the call keyword arguments (line 129)
            kwargs_5456 = {}
            # Getting the type of 'isinstance' (line 129)
            isinstance_5451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 129)
            isinstance_call_result_5457 = invoke(stypy.reporting.localization.Localization(__file__, 129, 15), isinstance_5451, *[body_5453, Iterable_5455], **kwargs_5456)
            
            # Testing if the type of an if condition is none (line 129)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 129, 12), isinstance_call_result_5457):
                
                # Assigning a Call to a Name (line 132):
                
                # Call to __visit_instruction_body(...): (line 132)
                # Processing the call arguments (line 132)
                
                # Obtaining an instance of the builtin type 'list' (line 132)
                list_5467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 54), 'list')
                # Adding type elements to the builtin type 'list' instance (line 132)
                # Adding element type (line 132)
                # Getting the type of 'node' (line 132)
                node_5468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 55), 'node', False)
                # Obtaining the member 'body' of a type (line 132)
                body_5469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 55), node_5468, 'body')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 54), list_5467, body_5469)
                
                # Processing the call keyword arguments (line 132)
                kwargs_5470 = {}
                # Getting the type of 'self' (line 132)
                self_5465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 24), 'self', False)
                # Obtaining the member '__visit_instruction_body' of a type (line 132)
                visit_instruction_body_5466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 24), self_5465, '__visit_instruction_body')
                # Calling __visit_instruction_body(args, kwargs) (line 132)
                visit_instruction_body_call_result_5471 = invoke(stypy.reporting.localization.Localization(__file__, 132, 24), visit_instruction_body_5466, *[list_5467], **kwargs_5470)
                
                # Assigning a type to the variable 'stmts' (line 132)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'stmts', visit_instruction_body_call_result_5471)
            else:
                
                # Testing the type of an if condition (line 129)
                if_condition_5458 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 12), isinstance_call_result_5457)
                # Assigning a type to the variable 'if_condition_5458' (line 129)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'if_condition_5458', if_condition_5458)
                # SSA begins for if statement (line 129)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 130):
                
                # Call to __visit_instruction_body(...): (line 130)
                # Processing the call arguments (line 130)
                # Getting the type of 'node' (line 130)
                node_5461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 54), 'node', False)
                # Obtaining the member 'body' of a type (line 130)
                body_5462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 54), node_5461, 'body')
                # Processing the call keyword arguments (line 130)
                kwargs_5463 = {}
                # Getting the type of 'self' (line 130)
                self_5459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'self', False)
                # Obtaining the member '__visit_instruction_body' of a type (line 130)
                visit_instruction_body_5460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 24), self_5459, '__visit_instruction_body')
                # Calling __visit_instruction_body(args, kwargs) (line 130)
                visit_instruction_body_call_result_5464 = invoke(stypy.reporting.localization.Localization(__file__, 130, 24), visit_instruction_body_5460, *[body_5462], **kwargs_5463)
                
                # Assigning a type to the variable 'stmts' (line 130)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'stmts', visit_instruction_body_call_result_5464)
                # SSA branch for the else part of an if statement (line 129)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 132):
                
                # Call to __visit_instruction_body(...): (line 132)
                # Processing the call arguments (line 132)
                
                # Obtaining an instance of the builtin type 'list' (line 132)
                list_5467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 54), 'list')
                # Adding type elements to the builtin type 'list' instance (line 132)
                # Adding element type (line 132)
                # Getting the type of 'node' (line 132)
                node_5468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 55), 'node', False)
                # Obtaining the member 'body' of a type (line 132)
                body_5469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 55), node_5468, 'body')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 54), list_5467, body_5469)
                
                # Processing the call keyword arguments (line 132)
                kwargs_5470 = {}
                # Getting the type of 'self' (line 132)
                self_5465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 24), 'self', False)
                # Obtaining the member '__visit_instruction_body' of a type (line 132)
                visit_instruction_body_5466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 24), self_5465, '__visit_instruction_body')
                # Calling __visit_instruction_body(args, kwargs) (line 132)
                visit_instruction_body_call_result_5471 = invoke(stypy.reporting.localization.Localization(__file__, 132, 24), visit_instruction_body_5466, *[list_5467], **kwargs_5470)
                
                # Assigning a type to the variable 'stmts' (line 132)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'stmts', visit_instruction_body_call_result_5471)
                # SSA join for if statement (line 129)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Name to a Attribute (line 134):
            # Getting the type of 'stmts' (line 134)
            stmts_5472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 24), 'stmts')
            # Getting the type of 'node' (line 134)
            node_5473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'node')
            # Setting the type of the member 'body' of a type (line 134)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 12), node_5473, 'body', stmts_5472)

            if more_types_in_union_5450:
                # SSA join for if statement (line 128)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 136)
        str_5474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 25), 'str', 'orelse')
        # Getting the type of 'node' (line 136)
        node_5475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'node')
        
        (may_be_5476, more_types_in_union_5477) = may_provide_member(str_5474, node_5475)

        if may_be_5476:

            if more_types_in_union_5477:
                # Runtime conditional SSA (line 136)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'node' (line 136)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'node', remove_not_member_provider_from_union(node_5475, 'orelse'))
            
            # Call to isinstance(...): (line 137)
            # Processing the call arguments (line 137)
            # Getting the type of 'node' (line 137)
            node_5479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 26), 'node', False)
            # Obtaining the member 'orelse' of a type (line 137)
            orelse_5480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 26), node_5479, 'orelse')
            # Getting the type of 'collections' (line 137)
            collections_5481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 39), 'collections', False)
            # Obtaining the member 'Iterable' of a type (line 137)
            Iterable_5482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 39), collections_5481, 'Iterable')
            # Processing the call keyword arguments (line 137)
            kwargs_5483 = {}
            # Getting the type of 'isinstance' (line 137)
            isinstance_5478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 137)
            isinstance_call_result_5484 = invoke(stypy.reporting.localization.Localization(__file__, 137, 15), isinstance_5478, *[orelse_5480, Iterable_5482], **kwargs_5483)
            
            # Testing if the type of an if condition is none (line 137)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 137, 12), isinstance_call_result_5484):
                
                # Assigning a Call to a Name (line 140):
                
                # Call to __visit_instruction_body(...): (line 140)
                # Processing the call arguments (line 140)
                
                # Obtaining an instance of the builtin type 'list' (line 140)
                list_5494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 54), 'list')
                # Adding type elements to the builtin type 'list' instance (line 140)
                # Adding element type (line 140)
                # Getting the type of 'node' (line 140)
                node_5495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 55), 'node', False)
                # Obtaining the member 'orelse' of a type (line 140)
                orelse_5496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 55), node_5495, 'orelse')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 54), list_5494, orelse_5496)
                
                # Processing the call keyword arguments (line 140)
                kwargs_5497 = {}
                # Getting the type of 'self' (line 140)
                self_5492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 24), 'self', False)
                # Obtaining the member '__visit_instruction_body' of a type (line 140)
                visit_instruction_body_5493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 24), self_5492, '__visit_instruction_body')
                # Calling __visit_instruction_body(args, kwargs) (line 140)
                visit_instruction_body_call_result_5498 = invoke(stypy.reporting.localization.Localization(__file__, 140, 24), visit_instruction_body_5493, *[list_5494], **kwargs_5497)
                
                # Assigning a type to the variable 'stmts' (line 140)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'stmts', visit_instruction_body_call_result_5498)
            else:
                
                # Testing the type of an if condition (line 137)
                if_condition_5485 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 12), isinstance_call_result_5484)
                # Assigning a type to the variable 'if_condition_5485' (line 137)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'if_condition_5485', if_condition_5485)
                # SSA begins for if statement (line 137)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 138):
                
                # Call to __visit_instruction_body(...): (line 138)
                # Processing the call arguments (line 138)
                # Getting the type of 'node' (line 138)
                node_5488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 54), 'node', False)
                # Obtaining the member 'orelse' of a type (line 138)
                orelse_5489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 54), node_5488, 'orelse')
                # Processing the call keyword arguments (line 138)
                kwargs_5490 = {}
                # Getting the type of 'self' (line 138)
                self_5486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 24), 'self', False)
                # Obtaining the member '__visit_instruction_body' of a type (line 138)
                visit_instruction_body_5487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 24), self_5486, '__visit_instruction_body')
                # Calling __visit_instruction_body(args, kwargs) (line 138)
                visit_instruction_body_call_result_5491 = invoke(stypy.reporting.localization.Localization(__file__, 138, 24), visit_instruction_body_5487, *[orelse_5489], **kwargs_5490)
                
                # Assigning a type to the variable 'stmts' (line 138)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'stmts', visit_instruction_body_call_result_5491)
                # SSA branch for the else part of an if statement (line 137)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Name (line 140):
                
                # Call to __visit_instruction_body(...): (line 140)
                # Processing the call arguments (line 140)
                
                # Obtaining an instance of the builtin type 'list' (line 140)
                list_5494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 54), 'list')
                # Adding type elements to the builtin type 'list' instance (line 140)
                # Adding element type (line 140)
                # Getting the type of 'node' (line 140)
                node_5495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 55), 'node', False)
                # Obtaining the member 'orelse' of a type (line 140)
                orelse_5496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 55), node_5495, 'orelse')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 140, 54), list_5494, orelse_5496)
                
                # Processing the call keyword arguments (line 140)
                kwargs_5497 = {}
                # Getting the type of 'self' (line 140)
                self_5492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 24), 'self', False)
                # Obtaining the member '__visit_instruction_body' of a type (line 140)
                visit_instruction_body_5493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 24), self_5492, '__visit_instruction_body')
                # Calling __visit_instruction_body(args, kwargs) (line 140)
                visit_instruction_body_call_result_5498 = invoke(stypy.reporting.localization.Localization(__file__, 140, 24), visit_instruction_body_5493, *[list_5494], **kwargs_5497)
                
                # Assigning a type to the variable 'stmts' (line 140)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'stmts', visit_instruction_body_call_result_5498)
                # SSA join for if statement (line 137)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Name to a Attribute (line 142):
            # Getting the type of 'stmts' (line 142)
            stmts_5499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 26), 'stmts')
            # Getting the type of 'node' (line 142)
            node_5500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'node')
            # Setting the type of the member 'orelse' of a type (line 142)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 12), node_5500, 'orelse', stmts_5499)

            if more_types_in_union_5477:
                # SSA join for if statement (line 136)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'node' (line 144)
        node_5501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'stypy_return_type', node_5501)
        
        # ################# End of 'generic_visit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generic_visit' in the type store
        # Getting the type of 'stypy_return_type' (line 127)
        stypy_return_type_5502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5502)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generic_visit'
        return stypy_return_type_5502


    @norecursion
    def visit_Module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Module'
        module_type_store = module_type_store.open_function_context('visit_Module', 148, 4, False)
        # Assigning a type to the variable 'self' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeAnnotationVisitor.visit_Module.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationVisitor.visit_Module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeAnnotationVisitor.visit_Module.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationVisitor.visit_Module.__dict__.__setitem__('stypy_function_name', 'TypeAnnotationVisitor.visit_Module')
        TypeAnnotationVisitor.visit_Module.__dict__.__setitem__('stypy_param_names_list', ['node'])
        TypeAnnotationVisitor.visit_Module.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationVisitor.visit_Module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationVisitor.visit_Module.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationVisitor.visit_Module.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationVisitor.visit_Module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationVisitor.visit_Module.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationVisitor.visit_Module', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Module', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Module(...)' code ##################

        
        # Assigning a Call to a Name (line 149):
        
        # Call to __visit_instruction_body(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'node' (line 149)
        node_5505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 46), 'node', False)
        # Obtaining the member 'body' of a type (line 149)
        body_5506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 46), node_5505, 'body')
        # Processing the call keyword arguments (line 149)
        kwargs_5507 = {}
        # Getting the type of 'self' (line 149)
        self_5503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 16), 'self', False)
        # Obtaining the member '__visit_instruction_body' of a type (line 149)
        visit_instruction_body_5504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 16), self_5503, '__visit_instruction_body')
        # Calling __visit_instruction_body(args, kwargs) (line 149)
        visit_instruction_body_call_result_5508 = invoke(stypy.reporting.localization.Localization(__file__, 149, 16), visit_instruction_body_5504, *[body_5506], **kwargs_5507)
        
        # Assigning a type to the variable 'stmts' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stmts', visit_instruction_body_call_result_5508)
        
        # Assigning a Name to a Attribute (line 151):
        # Getting the type of 'stmts' (line 151)
        stmts_5509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 20), 'stmts')
        # Getting the type of 'node' (line 151)
        node_5510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'node')
        # Setting the type of the member 'body' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), node_5510, 'body', stmts_5509)
        # Getting the type of 'node' (line 152)
        node_5511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'stypy_return_type', node_5511)
        
        # ################# End of 'visit_Module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Module' in the type store
        # Getting the type of 'stypy_return_type' (line 148)
        stypy_return_type_5512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5512)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Module'
        return stypy_return_type_5512


    @norecursion
    def visit_FunctionDef(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_FunctionDef'
        module_type_store = module_type_store.open_function_context('visit_FunctionDef', 156, 4, False)
        # Assigning a type to the variable 'self' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeAnnotationVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeAnnotationVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_function_name', 'TypeAnnotationVisitor.visit_FunctionDef')
        TypeAnnotationVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_param_names_list', ['node'])
        TypeAnnotationVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationVisitor.visit_FunctionDef', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_FunctionDef', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_FunctionDef(...)' code ##################

        
        # Assigning a Call to a Name (line 157):
        
        # Call to __get_type_annotations_for_function(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'node' (line 157)
        node_5515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 63), 'node', False)
        # Obtaining the member 'name' of a type (line 157)
        name_5516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 63), node_5515, 'name')
        # Getting the type of 'node' (line 157)
        node_5517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 74), 'node', False)
        # Obtaining the member 'lineno' of a type (line 157)
        lineno_5518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 74), node_5517, 'lineno')
        # Processing the call keyword arguments (line 157)
        kwargs_5519 = {}
        # Getting the type of 'self' (line 157)
        self_5513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 22), 'self', False)
        # Obtaining the member '__get_type_annotations_for_function' of a type (line 157)
        get_type_annotations_for_function_5514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 22), self_5513, '__get_type_annotations_for_function')
        # Calling __get_type_annotations_for_function(args, kwargs) (line 157)
        get_type_annotations_for_function_call_result_5520 = invoke(stypy.reporting.localization.Localization(__file__, 157, 22), get_type_annotations_for_function_5514, *[name_5516, lineno_5518], **kwargs_5519)
        
        # Assigning a type to the variable 'annotations' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'annotations', get_type_annotations_for_function_call_result_5520)
        
        # Assigning a Call to a Name (line 158):
        
        # Call to __visit_instruction_body(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'node' (line 158)
        node_5523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 46), 'node', False)
        # Obtaining the member 'body' of a type (line 158)
        body_5524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 46), node_5523, 'body')
        # Processing the call keyword arguments (line 158)
        kwargs_5525 = {}
        # Getting the type of 'self' (line 158)
        self_5521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'self', False)
        # Obtaining the member '__visit_instruction_body' of a type (line 158)
        visit_instruction_body_5522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 16), self_5521, '__visit_instruction_body')
        # Calling __visit_instruction_body(args, kwargs) (line 158)
        visit_instruction_body_call_result_5526 = invoke(stypy.reporting.localization.Localization(__file__, 158, 16), visit_instruction_body_5522, *[body_5524], **kwargs_5525)
        
        # Assigning a type to the variable 'stmts' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'stmts', visit_instruction_body_call_result_5526)
        
        # Assigning a Name to a Attribute (line 159):
        # Getting the type of 'stmts' (line 159)
        stmts_5527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'stmts')
        # Getting the type of 'node' (line 159)
        node_5528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'node')
        # Setting the type of the member 'body' of a type (line 159)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), node_5528, 'body', stmts_5527)
        
        
        # Getting the type of 'annotations' (line 161)
        annotations_5529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'annotations')
        str_5530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 30), 'str', '')
        # Applying the binary operator '==' (line 161)
        result_eq_5531 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 15), '==', annotations_5529, str_5530)
        
        # Applying the 'not' unary operator (line 161)
        result_not__5532 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 11), 'not', result_eq_5531)
        
        # Testing if the type of an if condition is none (line 161)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 161, 8), result_not__5532):
            
            # Assigning a Call to a Name (line 164):
            
            # Call to create_src_comment(...): (line 164)
            # Processing the call arguments (line 164)
            str_5541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 66), 'str', '<Dead code detected>')
            # Processing the call keyword arguments (line 164)
            kwargs_5542 = {}
            # Getting the type of 'stypy_functions_copy' (line 164)
            stypy_functions_copy_5539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 26), 'stypy_functions_copy', False)
            # Obtaining the member 'create_src_comment' of a type (line 164)
            create_src_comment_5540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 26), stypy_functions_copy_5539, 'create_src_comment')
            # Calling create_src_comment(args, kwargs) (line 164)
            create_src_comment_call_result_5543 = invoke(stypy.reporting.localization.Localization(__file__, 164, 26), create_src_comment_5540, *[str_5541], **kwargs_5542)
            
            # Assigning a type to the variable 'annotations' (line 164)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'annotations', create_src_comment_call_result_5543)
        else:
            
            # Testing the type of an if condition (line 161)
            if_condition_5533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 8), result_not__5532)
            # Assigning a type to the variable 'if_condition_5533' (line 161)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'if_condition_5533', if_condition_5533)
            # SSA begins for if statement (line 161)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 162):
            
            # Call to create_src_comment(...): (line 162)
            # Processing the call arguments (line 162)
            # Getting the type of 'annotations' (line 162)
            annotations_5536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 66), 'annotations', False)
            # Processing the call keyword arguments (line 162)
            kwargs_5537 = {}
            # Getting the type of 'stypy_functions_copy' (line 162)
            stypy_functions_copy_5534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 26), 'stypy_functions_copy', False)
            # Obtaining the member 'create_src_comment' of a type (line 162)
            create_src_comment_5535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 26), stypy_functions_copy_5534, 'create_src_comment')
            # Calling create_src_comment(args, kwargs) (line 162)
            create_src_comment_call_result_5538 = invoke(stypy.reporting.localization.Localization(__file__, 162, 26), create_src_comment_5535, *[annotations_5536], **kwargs_5537)
            
            # Assigning a type to the variable 'annotations' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'annotations', create_src_comment_call_result_5538)
            # SSA branch for the else part of an if statement (line 161)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 164):
            
            # Call to create_src_comment(...): (line 164)
            # Processing the call arguments (line 164)
            str_5541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 66), 'str', '<Dead code detected>')
            # Processing the call keyword arguments (line 164)
            kwargs_5542 = {}
            # Getting the type of 'stypy_functions_copy' (line 164)
            stypy_functions_copy_5539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 26), 'stypy_functions_copy', False)
            # Obtaining the member 'create_src_comment' of a type (line 164)
            create_src_comment_5540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 26), stypy_functions_copy_5539, 'create_src_comment')
            # Calling create_src_comment(args, kwargs) (line 164)
            create_src_comment_call_result_5543 = invoke(stypy.reporting.localization.Localization(__file__, 164, 26), create_src_comment_5540, *[str_5541], **kwargs_5542)
            
            # Assigning a type to the variable 'annotations' (line 164)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'annotations', create_src_comment_call_result_5543)
            # SSA join for if statement (line 161)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to flatten_lists(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'annotations' (line 166)
        annotations_5546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 50), 'annotations', False)
        # Getting the type of 'node' (line 166)
        node_5547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 63), 'node', False)
        # Processing the call keyword arguments (line 166)
        kwargs_5548 = {}
        # Getting the type of 'stypy_functions_copy' (line 166)
        stypy_functions_copy_5544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'stypy_functions_copy', False)
        # Obtaining the member 'flatten_lists' of a type (line 166)
        flatten_lists_5545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 15), stypy_functions_copy_5544, 'flatten_lists')
        # Calling flatten_lists(args, kwargs) (line 166)
        flatten_lists_call_result_5549 = invoke(stypy.reporting.localization.Localization(__file__, 166, 15), flatten_lists_5545, *[annotations_5546, node_5547], **kwargs_5548)
        
        # Assigning a type to the variable 'stypy_return_type' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'stypy_return_type', flatten_lists_call_result_5549)
        
        # ################# End of 'visit_FunctionDef(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_FunctionDef' in the type store
        # Getting the type of 'stypy_return_type' (line 156)
        stypy_return_type_5550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5550)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_FunctionDef'
        return stypy_return_type_5550


    @norecursion
    def visit_If(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_If'
        module_type_store = module_type_store.open_function_context('visit_If', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeAnnotationVisitor.visit_If.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationVisitor.visit_If.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeAnnotationVisitor.visit_If.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationVisitor.visit_If.__dict__.__setitem__('stypy_function_name', 'TypeAnnotationVisitor.visit_If')
        TypeAnnotationVisitor.visit_If.__dict__.__setitem__('stypy_param_names_list', ['node'])
        TypeAnnotationVisitor.visit_If.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationVisitor.visit_If.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationVisitor.visit_If.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationVisitor.visit_If.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationVisitor.visit_If.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationVisitor.visit_If.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationVisitor.visit_If', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_If', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_If(...)' code ##################

        
        # Assigning a Call to a Name (line 169):
        
        # Call to __visit_instruction_body(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'node' (line 169)
        node_5553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 46), 'node', False)
        # Obtaining the member 'body' of a type (line 169)
        body_5554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 46), node_5553, 'body')
        # Processing the call keyword arguments (line 169)
        kwargs_5555 = {}
        # Getting the type of 'self' (line 169)
        self_5551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'self', False)
        # Obtaining the member '__visit_instruction_body' of a type (line 169)
        visit_instruction_body_5552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 16), self_5551, '__visit_instruction_body')
        # Calling __visit_instruction_body(args, kwargs) (line 169)
        visit_instruction_body_call_result_5556 = invoke(stypy.reporting.localization.Localization(__file__, 169, 16), visit_instruction_body_5552, *[body_5554], **kwargs_5555)
        
        # Assigning a type to the variable 'stmts' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'stmts', visit_instruction_body_call_result_5556)
        
        # Assigning a Name to a Attribute (line 170):
        # Getting the type of 'stmts' (line 170)
        stmts_5557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 20), 'stmts')
        # Getting the type of 'node' (line 170)
        node_5558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'node')
        # Setting the type of the member 'body' of a type (line 170)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), node_5558, 'body', stmts_5557)
        
        # Assigning a Call to a Name (line 172):
        
        # Call to __visit_instruction_body(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'node' (line 172)
        node_5561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 46), 'node', False)
        # Obtaining the member 'orelse' of a type (line 172)
        orelse_5562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 46), node_5561, 'orelse')
        # Processing the call keyword arguments (line 172)
        kwargs_5563 = {}
        # Getting the type of 'self' (line 172)
        self_5559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'self', False)
        # Obtaining the member '__visit_instruction_body' of a type (line 172)
        visit_instruction_body_5560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 16), self_5559, '__visit_instruction_body')
        # Calling __visit_instruction_body(args, kwargs) (line 172)
        visit_instruction_body_call_result_5564 = invoke(stypy.reporting.localization.Localization(__file__, 172, 16), visit_instruction_body_5560, *[orelse_5562], **kwargs_5563)
        
        # Assigning a type to the variable 'stmts' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'stmts', visit_instruction_body_call_result_5564)
        
        # Assigning a Name to a Attribute (line 173):
        # Getting the type of 'stmts' (line 173)
        stmts_5565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), 'stmts')
        # Getting the type of 'node' (line 173)
        node_5566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'node')
        # Setting the type of the member 'orelse' of a type (line 173)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), node_5566, 'orelse', stmts_5565)
        # Getting the type of 'node' (line 175)
        node_5567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'stypy_return_type', node_5567)
        
        # ################# End of 'visit_If(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_If' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_5568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5568)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_If'
        return stypy_return_type_5568


    @norecursion
    def visit_While(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_While'
        module_type_store = module_type_store.open_function_context('visit_While', 177, 4, False)
        # Assigning a type to the variable 'self' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeAnnotationVisitor.visit_While.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationVisitor.visit_While.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeAnnotationVisitor.visit_While.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationVisitor.visit_While.__dict__.__setitem__('stypy_function_name', 'TypeAnnotationVisitor.visit_While')
        TypeAnnotationVisitor.visit_While.__dict__.__setitem__('stypy_param_names_list', ['node'])
        TypeAnnotationVisitor.visit_While.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationVisitor.visit_While.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationVisitor.visit_While.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationVisitor.visit_While.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationVisitor.visit_While.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationVisitor.visit_While.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationVisitor.visit_While', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_While', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_While(...)' code ##################

        
        # Assigning a Call to a Name (line 178):
        
        # Call to __visit_instruction_body(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'node' (line 178)
        node_5571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 46), 'node', False)
        # Obtaining the member 'body' of a type (line 178)
        body_5572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 46), node_5571, 'body')
        # Processing the call keyword arguments (line 178)
        kwargs_5573 = {}
        # Getting the type of 'self' (line 178)
        self_5569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'self', False)
        # Obtaining the member '__visit_instruction_body' of a type (line 178)
        visit_instruction_body_5570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 16), self_5569, '__visit_instruction_body')
        # Calling __visit_instruction_body(args, kwargs) (line 178)
        visit_instruction_body_call_result_5574 = invoke(stypy.reporting.localization.Localization(__file__, 178, 16), visit_instruction_body_5570, *[body_5572], **kwargs_5573)
        
        # Assigning a type to the variable 'stmts' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'stmts', visit_instruction_body_call_result_5574)
        
        # Assigning a Name to a Attribute (line 179):
        # Getting the type of 'stmts' (line 179)
        stmts_5575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'stmts')
        # Getting the type of 'node' (line 179)
        node_5576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'node')
        # Setting the type of the member 'body' of a type (line 179)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), node_5576, 'body', stmts_5575)
        
        # Assigning a Call to a Name (line 181):
        
        # Call to __visit_instruction_body(...): (line 181)
        # Processing the call arguments (line 181)
        # Getting the type of 'node' (line 181)
        node_5579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 46), 'node', False)
        # Obtaining the member 'orelse' of a type (line 181)
        orelse_5580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 46), node_5579, 'orelse')
        # Processing the call keyword arguments (line 181)
        kwargs_5581 = {}
        # Getting the type of 'self' (line 181)
        self_5577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'self', False)
        # Obtaining the member '__visit_instruction_body' of a type (line 181)
        visit_instruction_body_5578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 16), self_5577, '__visit_instruction_body')
        # Calling __visit_instruction_body(args, kwargs) (line 181)
        visit_instruction_body_call_result_5582 = invoke(stypy.reporting.localization.Localization(__file__, 181, 16), visit_instruction_body_5578, *[orelse_5580], **kwargs_5581)
        
        # Assigning a type to the variable 'stmts' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'stmts', visit_instruction_body_call_result_5582)
        
        # Assigning a Name to a Attribute (line 182):
        # Getting the type of 'stmts' (line 182)
        stmts_5583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'stmts')
        # Getting the type of 'node' (line 182)
        node_5584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'node')
        # Setting the type of the member 'orelse' of a type (line 182)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), node_5584, 'orelse', stmts_5583)
        # Getting the type of 'node' (line 184)
        node_5585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'stypy_return_type', node_5585)
        
        # ################# End of 'visit_While(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_While' in the type store
        # Getting the type of 'stypy_return_type' (line 177)
        stypy_return_type_5586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5586)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_While'
        return stypy_return_type_5586


    @norecursion
    def visit_For(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_For'
        module_type_store = module_type_store.open_function_context('visit_For', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeAnnotationVisitor.visit_For.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationVisitor.visit_For.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeAnnotationVisitor.visit_For.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationVisitor.visit_For.__dict__.__setitem__('stypy_function_name', 'TypeAnnotationVisitor.visit_For')
        TypeAnnotationVisitor.visit_For.__dict__.__setitem__('stypy_param_names_list', ['node'])
        TypeAnnotationVisitor.visit_For.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationVisitor.visit_For.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationVisitor.visit_For.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationVisitor.visit_For.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationVisitor.visit_For.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationVisitor.visit_For.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationVisitor.visit_For', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_For', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_For(...)' code ##################

        
        # Assigning a Call to a Name (line 187):
        
        # Call to __visit_instruction_body(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'node' (line 187)
        node_5589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 46), 'node', False)
        # Obtaining the member 'body' of a type (line 187)
        body_5590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 46), node_5589, 'body')
        # Processing the call keyword arguments (line 187)
        kwargs_5591 = {}
        # Getting the type of 'self' (line 187)
        self_5587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 16), 'self', False)
        # Obtaining the member '__visit_instruction_body' of a type (line 187)
        visit_instruction_body_5588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 16), self_5587, '__visit_instruction_body')
        # Calling __visit_instruction_body(args, kwargs) (line 187)
        visit_instruction_body_call_result_5592 = invoke(stypy.reporting.localization.Localization(__file__, 187, 16), visit_instruction_body_5588, *[body_5590], **kwargs_5591)
        
        # Assigning a type to the variable 'stmts' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'stmts', visit_instruction_body_call_result_5592)
        
        # Assigning a Name to a Attribute (line 188):
        # Getting the type of 'stmts' (line 188)
        stmts_5593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 20), 'stmts')
        # Getting the type of 'node' (line 188)
        node_5594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'node')
        # Setting the type of the member 'body' of a type (line 188)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), node_5594, 'body', stmts_5593)
        
        # Assigning a Call to a Name (line 190):
        
        # Call to __visit_instruction_body(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'node' (line 190)
        node_5597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 46), 'node', False)
        # Obtaining the member 'orelse' of a type (line 190)
        orelse_5598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 46), node_5597, 'orelse')
        # Processing the call keyword arguments (line 190)
        kwargs_5599 = {}
        # Getting the type of 'self' (line 190)
        self_5595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'self', False)
        # Obtaining the member '__visit_instruction_body' of a type (line 190)
        visit_instruction_body_5596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 16), self_5595, '__visit_instruction_body')
        # Calling __visit_instruction_body(args, kwargs) (line 190)
        visit_instruction_body_call_result_5600 = invoke(stypy.reporting.localization.Localization(__file__, 190, 16), visit_instruction_body_5596, *[orelse_5598], **kwargs_5599)
        
        # Assigning a type to the variable 'stmts' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'stmts', visit_instruction_body_call_result_5600)
        
        # Assigning a Name to a Attribute (line 191):
        # Getting the type of 'stmts' (line 191)
        stmts_5601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 22), 'stmts')
        # Getting the type of 'node' (line 191)
        node_5602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'node')
        # Setting the type of the member 'orelse' of a type (line 191)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), node_5602, 'orelse', stmts_5601)
        # Getting the type of 'node' (line 193)
        node_5603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'stypy_return_type', node_5603)
        
        # ################# End of 'visit_For(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_For' in the type store
        # Getting the type of 'stypy_return_type' (line 186)
        stypy_return_type_5604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5604)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_For'
        return stypy_return_type_5604


    @norecursion
    def visit_ClassDef(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_ClassDef'
        module_type_store = module_type_store.open_function_context('visit_ClassDef', 195, 4, False)
        # Assigning a type to the variable 'self' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeAnnotationVisitor.visit_ClassDef.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationVisitor.visit_ClassDef.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeAnnotationVisitor.visit_ClassDef.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationVisitor.visit_ClassDef.__dict__.__setitem__('stypy_function_name', 'TypeAnnotationVisitor.visit_ClassDef')
        TypeAnnotationVisitor.visit_ClassDef.__dict__.__setitem__('stypy_param_names_list', ['node'])
        TypeAnnotationVisitor.visit_ClassDef.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationVisitor.visit_ClassDef.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationVisitor.visit_ClassDef.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationVisitor.visit_ClassDef.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationVisitor.visit_ClassDef.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationVisitor.visit_ClassDef.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationVisitor.visit_ClassDef', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_ClassDef', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_ClassDef(...)' code ##################

        
        # Assigning a Call to a Name (line 196):
        
        # Call to __visit_instruction_body(...): (line 196)
        # Processing the call arguments (line 196)
        # Getting the type of 'node' (line 196)
        node_5607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 46), 'node', False)
        # Obtaining the member 'body' of a type (line 196)
        body_5608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 46), node_5607, 'body')
        # Processing the call keyword arguments (line 196)
        kwargs_5609 = {}
        # Getting the type of 'self' (line 196)
        self_5605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 16), 'self', False)
        # Obtaining the member '__visit_instruction_body' of a type (line 196)
        visit_instruction_body_5606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 16), self_5605, '__visit_instruction_body')
        # Calling __visit_instruction_body(args, kwargs) (line 196)
        visit_instruction_body_call_result_5610 = invoke(stypy.reporting.localization.Localization(__file__, 196, 16), visit_instruction_body_5606, *[body_5608], **kwargs_5609)
        
        # Assigning a type to the variable 'stmts' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'stmts', visit_instruction_body_call_result_5610)
        
        # Assigning a Name to a Attribute (line 197):
        # Getting the type of 'stmts' (line 197)
        stmts_5611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 20), 'stmts')
        # Getting the type of 'node' (line 197)
        node_5612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'node')
        # Setting the type of the member 'body' of a type (line 197)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), node_5612, 'body', stmts_5611)
        # Getting the type of 'node' (line 199)
        node_5613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'stypy_return_type', node_5613)
        
        # ################# End of 'visit_ClassDef(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ClassDef' in the type store
        # Getting the type of 'stypy_return_type' (line 195)
        stypy_return_type_5614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5614)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ClassDef'
        return stypy_return_type_5614


    @norecursion
    def visit_With(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_With'
        module_type_store = module_type_store.open_function_context('visit_With', 201, 4, False)
        # Assigning a type to the variable 'self' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeAnnotationVisitor.visit_With.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationVisitor.visit_With.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeAnnotationVisitor.visit_With.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationVisitor.visit_With.__dict__.__setitem__('stypy_function_name', 'TypeAnnotationVisitor.visit_With')
        TypeAnnotationVisitor.visit_With.__dict__.__setitem__('stypy_param_names_list', ['node'])
        TypeAnnotationVisitor.visit_With.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationVisitor.visit_With.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationVisitor.visit_With.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationVisitor.visit_With.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationVisitor.visit_With.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationVisitor.visit_With.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationVisitor.visit_With', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_With', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_With(...)' code ##################

        
        # Assigning a Call to a Name (line 202):
        
        # Call to __visit_instruction_body(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'node' (line 202)
        node_5617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 46), 'node', False)
        # Obtaining the member 'body' of a type (line 202)
        body_5618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 46), node_5617, 'body')
        # Processing the call keyword arguments (line 202)
        kwargs_5619 = {}
        # Getting the type of 'self' (line 202)
        self_5615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'self', False)
        # Obtaining the member '__visit_instruction_body' of a type (line 202)
        visit_instruction_body_5616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 16), self_5615, '__visit_instruction_body')
        # Calling __visit_instruction_body(args, kwargs) (line 202)
        visit_instruction_body_call_result_5620 = invoke(stypy.reporting.localization.Localization(__file__, 202, 16), visit_instruction_body_5616, *[body_5618], **kwargs_5619)
        
        # Assigning a type to the variable 'stmts' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'stmts', visit_instruction_body_call_result_5620)
        
        # Assigning a Name to a Attribute (line 203):
        # Getting the type of 'stmts' (line 203)
        stmts_5621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 20), 'stmts')
        # Getting the type of 'node' (line 203)
        node_5622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'node')
        # Setting the type of the member 'body' of a type (line 203)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), node_5622, 'body', stmts_5621)
        # Getting the type of 'node' (line 205)
        node_5623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'stypy_return_type', node_5623)
        
        # ################# End of 'visit_With(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_With' in the type store
        # Getting the type of 'stypy_return_type' (line 201)
        stypy_return_type_5624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5624)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_With'
        return stypy_return_type_5624


    @norecursion
    def visit_TryExcept(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_TryExcept'
        module_type_store = module_type_store.open_function_context('visit_TryExcept', 207, 4, False)
        # Assigning a type to the variable 'self' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeAnnotationVisitor.visit_TryExcept.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationVisitor.visit_TryExcept.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeAnnotationVisitor.visit_TryExcept.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationVisitor.visit_TryExcept.__dict__.__setitem__('stypy_function_name', 'TypeAnnotationVisitor.visit_TryExcept')
        TypeAnnotationVisitor.visit_TryExcept.__dict__.__setitem__('stypy_param_names_list', ['node'])
        TypeAnnotationVisitor.visit_TryExcept.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationVisitor.visit_TryExcept.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationVisitor.visit_TryExcept.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationVisitor.visit_TryExcept.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationVisitor.visit_TryExcept.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationVisitor.visit_TryExcept.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationVisitor.visit_TryExcept', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_TryExcept', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_TryExcept(...)' code ##################

        
        # Assigning a Call to a Name (line 208):
        
        # Call to __visit_instruction_body(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'node' (line 208)
        node_5627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 46), 'node', False)
        # Obtaining the member 'body' of a type (line 208)
        body_5628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 46), node_5627, 'body')
        # Processing the call keyword arguments (line 208)
        kwargs_5629 = {}
        # Getting the type of 'self' (line 208)
        self_5625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'self', False)
        # Obtaining the member '__visit_instruction_body' of a type (line 208)
        visit_instruction_body_5626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 16), self_5625, '__visit_instruction_body')
        # Calling __visit_instruction_body(args, kwargs) (line 208)
        visit_instruction_body_call_result_5630 = invoke(stypy.reporting.localization.Localization(__file__, 208, 16), visit_instruction_body_5626, *[body_5628], **kwargs_5629)
        
        # Assigning a type to the variable 'stmts' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stmts', visit_instruction_body_call_result_5630)
        
        # Assigning a Name to a Attribute (line 209):
        # Getting the type of 'stmts' (line 209)
        stmts_5631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'stmts')
        # Getting the type of 'node' (line 209)
        node_5632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 8), 'node')
        # Setting the type of the member 'body' of a type (line 209)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 8), node_5632, 'body', stmts_5631)
        
        # Getting the type of 'node' (line 211)
        node_5633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 23), 'node')
        # Obtaining the member 'handlers' of a type (line 211)
        handlers_5634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 23), node_5633, 'handlers')
        # Assigning a type to the variable 'handlers_5634' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'handlers_5634', handlers_5634)
        # Testing if the for loop is going to be iterated (line 211)
        # Testing the type of a for loop iterable (line 211)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 211, 8), handlers_5634)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 211, 8), handlers_5634):
            # Getting the type of the for loop variable (line 211)
            for_loop_var_5635 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 211, 8), handlers_5634)
            # Assigning a type to the variable 'handler' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'handler', for_loop_var_5635)
            # SSA begins for a for statement (line 211)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 212):
            
            # Call to __visit_instruction_body(...): (line 212)
            # Processing the call arguments (line 212)
            # Getting the type of 'handler' (line 212)
            handler_5638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 50), 'handler', False)
            # Obtaining the member 'body' of a type (line 212)
            body_5639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 50), handler_5638, 'body')
            # Processing the call keyword arguments (line 212)
            kwargs_5640 = {}
            # Getting the type of 'self' (line 212)
            self_5636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'self', False)
            # Obtaining the member '__visit_instruction_body' of a type (line 212)
            visit_instruction_body_5637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 20), self_5636, '__visit_instruction_body')
            # Calling __visit_instruction_body(args, kwargs) (line 212)
            visit_instruction_body_call_result_5641 = invoke(stypy.reporting.localization.Localization(__file__, 212, 20), visit_instruction_body_5637, *[body_5639], **kwargs_5640)
            
            # Assigning a type to the variable 'stmts' (line 212)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'stmts', visit_instruction_body_call_result_5641)
            
            # Assigning a Name to a Attribute (line 213):
            # Getting the type of 'stmts' (line 213)
            stmts_5642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 27), 'stmts')
            # Getting the type of 'handler' (line 213)
            handler_5643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'handler')
            # Setting the type of the member 'body' of a type (line 213)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 12), handler_5643, 'body', stmts_5642)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Name (line 215):
        
        # Call to __visit_instruction_body(...): (line 215)
        # Processing the call arguments (line 215)
        # Getting the type of 'node' (line 215)
        node_5646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 46), 'node', False)
        # Obtaining the member 'orelse' of a type (line 215)
        orelse_5647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 46), node_5646, 'orelse')
        # Processing the call keyword arguments (line 215)
        kwargs_5648 = {}
        # Getting the type of 'self' (line 215)
        self_5644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'self', False)
        # Obtaining the member '__visit_instruction_body' of a type (line 215)
        visit_instruction_body_5645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 16), self_5644, '__visit_instruction_body')
        # Calling __visit_instruction_body(args, kwargs) (line 215)
        visit_instruction_body_call_result_5649 = invoke(stypy.reporting.localization.Localization(__file__, 215, 16), visit_instruction_body_5645, *[orelse_5647], **kwargs_5648)
        
        # Assigning a type to the variable 'stmts' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'stmts', visit_instruction_body_call_result_5649)
        
        # Assigning a Name to a Attribute (line 216):
        # Getting the type of 'stmts' (line 216)
        stmts_5650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 22), 'stmts')
        # Getting the type of 'node' (line 216)
        node_5651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'node')
        # Setting the type of the member 'orelse' of a type (line 216)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), node_5651, 'orelse', stmts_5650)
        # Getting the type of 'node' (line 217)
        node_5652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'stypy_return_type', node_5652)
        
        # ################# End of 'visit_TryExcept(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_TryExcept' in the type store
        # Getting the type of 'stypy_return_type' (line 207)
        stypy_return_type_5653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5653)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_TryExcept'
        return stypy_return_type_5653


    @norecursion
    def visit_TryFinally(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_TryFinally'
        module_type_store = module_type_store.open_function_context('visit_TryFinally', 219, 4, False)
        # Assigning a type to the variable 'self' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeAnnotationVisitor.visit_TryFinally.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationVisitor.visit_TryFinally.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeAnnotationVisitor.visit_TryFinally.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationVisitor.visit_TryFinally.__dict__.__setitem__('stypy_function_name', 'TypeAnnotationVisitor.visit_TryFinally')
        TypeAnnotationVisitor.visit_TryFinally.__dict__.__setitem__('stypy_param_names_list', ['node'])
        TypeAnnotationVisitor.visit_TryFinally.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationVisitor.visit_TryFinally.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationVisitor.visit_TryFinally.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationVisitor.visit_TryFinally.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationVisitor.visit_TryFinally.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationVisitor.visit_TryFinally.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationVisitor.visit_TryFinally', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_TryFinally', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_TryFinally(...)' code ##################

        
        # Assigning a Call to a Name (line 220):
        
        # Call to __visit_instruction_body(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'node' (line 220)
        node_5656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 46), 'node', False)
        # Obtaining the member 'body' of a type (line 220)
        body_5657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 46), node_5656, 'body')
        # Processing the call keyword arguments (line 220)
        kwargs_5658 = {}
        # Getting the type of 'self' (line 220)
        self_5654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'self', False)
        # Obtaining the member '__visit_instruction_body' of a type (line 220)
        visit_instruction_body_5655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 16), self_5654, '__visit_instruction_body')
        # Calling __visit_instruction_body(args, kwargs) (line 220)
        visit_instruction_body_call_result_5659 = invoke(stypy.reporting.localization.Localization(__file__, 220, 16), visit_instruction_body_5655, *[body_5657], **kwargs_5658)
        
        # Assigning a type to the variable 'stmts' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'stmts', visit_instruction_body_call_result_5659)
        
        # Assigning a Name to a Attribute (line 221):
        # Getting the type of 'stmts' (line 221)
        stmts_5660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'stmts')
        # Getting the type of 'node' (line 221)
        node_5661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'node')
        # Setting the type of the member 'body' of a type (line 221)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), node_5661, 'body', stmts_5660)
        
        # Assigning a Call to a Name (line 223):
        
        # Call to __visit_instruction_body(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'node' (line 223)
        node_5664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 46), 'node', False)
        # Obtaining the member 'finalbody' of a type (line 223)
        finalbody_5665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 46), node_5664, 'finalbody')
        # Processing the call keyword arguments (line 223)
        kwargs_5666 = {}
        # Getting the type of 'self' (line 223)
        self_5662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), 'self', False)
        # Obtaining the member '__visit_instruction_body' of a type (line 223)
        visit_instruction_body_5663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 16), self_5662, '__visit_instruction_body')
        # Calling __visit_instruction_body(args, kwargs) (line 223)
        visit_instruction_body_call_result_5667 = invoke(stypy.reporting.localization.Localization(__file__, 223, 16), visit_instruction_body_5663, *[finalbody_5665], **kwargs_5666)
        
        # Assigning a type to the variable 'stmts' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'stmts', visit_instruction_body_call_result_5667)
        
        # Assigning a Name to a Attribute (line 224):
        # Getting the type of 'stmts' (line 224)
        stmts_5668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 25), 'stmts')
        # Getting the type of 'node' (line 224)
        node_5669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'node')
        # Setting the type of the member 'finalbody' of a type (line 224)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), node_5669, 'finalbody', stmts_5668)
        # Getting the type of 'node' (line 226)
        node_5670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'node')
        # Assigning a type to the variable 'stypy_return_type' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'stypy_return_type', node_5670)
        
        # ################# End of 'visit_TryFinally(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_TryFinally' in the type store
        # Getting the type of 'stypy_return_type' (line 219)
        stypy_return_type_5671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5671)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_TryFinally'
        return stypy_return_type_5671


# Assigning a type to the variable 'TypeAnnotationVisitor' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'TypeAnnotationVisitor', TypeAnnotationVisitor)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
