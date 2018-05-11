
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import inspect
2: import sys
3: import imp
4: import copy
5: import types
6: 
7: from .....stypy_copy import stypy_parameters_copy
8: 
9: '''
10: File that contains helper functions to implement the type_inference_proxy.py functionality, grouped here to improve
11: readability of the code.
12: '''
13: user_defined_modules = None
14: last_module_len = 0
15: 
16: def __init_user_defined_modules(default_python_installation_path=stypy_parameters_copy.PYTHON_EXE_PATH):
17:     '''
18:     Initializes the user_defined_modules variable
19:     '''
20:     global user_defined_modules
21:     global last_module_len
22: 
23:     # Empty user_defined_modules? Create values for it by traversing sys.modules and discarding Python library ones.
24:     # This way we locate all the loaded modules that are not part of the Python distribution
25:     normalized_path = default_python_installation_path.replace('/', '\\')
26:     modules = sys.modules.items()
27: 
28:     # No modules loaded or len of modules changed
29:     if user_defined_modules is None or len(modules) != last_module_len:
30:         user_defined_modules = dict((module_name, module_desc) for (module_name, module_desc) in modules
31:                                 if (normalized_path not in str(module_desc) and "built-in" not in
32:                                     str(module_desc)
33:                                     and module_desc is not None))
34:         last_module_len = len(modules)
35: 
36: 
37: def is_user_defined_module(module_name, default_python_installation_path=stypy_parameters_copy.PYTHON_EXE_PATH):
38:     '''
39:     Determines if the passed module_name is a user created module or a Python library one.
40:     :param module_name: Name of the module
41:     :param default_python_installation_path: Python executable to use. Can be left blank as it is initialized by default
42:      with the PYTHON_EXE_PATH parameter
43:     :return: bool
44:     '''
45:     global user_defined_modules
46: 
47:     __init_user_defined_modules(default_python_installation_path)
48: 
49:     return module_name in user_defined_modules
50: 
51: 
52: def is_user_defined_class(cls, default_python_installation_path=stypy_parameters_copy.PYTHON_EXE_PATH):
53:     '''
54:     Determines if the passed class is a user created class or a Python library one.
55:     :param cls: Class
56:     :param default_python_installation_path: Python executable to use. Can be left blank as it is initialized by default
57:      with the PYTHON_EXE_PATH parameter
58:     :return:
59:     '''
60:     global user_defined_modules
61: 
62:     if not inspect.isclass(cls):
63:         return False
64: 
65:     __init_user_defined_modules(default_python_installation_path)
66: 
67:     # A class is user defined if its module is user defined
68:     return is_user_defined_module(cls.__module__, default_python_installation_path)
69: 
70: 
71: def supports_structural_reflection(obj):
72:     '''
73:     Determines if an object supports structural reflection. An object supports it if it has a __dict__ property and its
74:     type is dict (instead of the read-only dictproxy)
75: 
76:     :param obj: Any Python object
77:     :return: bool
78:     '''
79:     if not hasattr(obj, '__dict__'):
80:         return False
81: 
82:     if type(obj.__dict__) is dict:
83:         return True
84:     else:
85:         try:
86:             obj.__dict__["__stypy_probe"] = None
87:             del obj.__dict__["__stypy_probe"]
88:             return True
89:         except:
90:             return False
91: 
92: 
93: def is_class(cls):
94:     '''
95:     Shortcut to inspect.isclass
96:     :param cls: Any Python object
97:     :return:
98:     '''
99:     return inspect.isclass(cls)
100: 
101: 
102: def is_old_style_class(cls):
103:     '''
104:     Python supports two type of classes: old-style classes (those that do not inherit from object) and new-style classes
105:     (those that do inherit from object). The best way to distinguish between them is to check if the class has an
106:      __mro__ (method resolution order) property (only available to new-style classes). Distinguishing between both types
107:      is important specially when dealing with type change or supertype change operations, as new-style classes are
108:      more limited in that sense and both types cannot be mixed in one of these operations.
109:     :param cls: Class to test
110:     :return: bool
111:     '''
112:     if not is_class(cls):
113:         return False
114:     return not hasattr(cls, "__mro__")
115: 
116: 
117: def is_new_style_class(cls):
118:     '''
119:     This method is a shortcut to the opposite of the previous one
120:     :param cls: Class to test
121:     :return: bool
122:     '''
123:     return not is_old_style_class(cls)
124: 
125: 
126: # TODO: Remove?
127: # def supports_type_change(cls):
128: #     '''
129: #     This method check if objects of a class support type changing operations. Only user-defined classes support
130: #     this kind of operation.
131: #     :param cls: Class to test
132: #     :return: bool
133: #     '''
134: #     if not is_class(cls):
135: #         return False
136: #
137: #     return is_user_defined_class(cls)
138: 
139: 
140: # def supports_base_types_change(cls):
141: #     pass
142: 
143: # ############################ PYTHON TYPE CLONING ############################
144: 
145: '''
146: Cloning Python types is a key part of the implementation of the SSA algorithm. However, this is a very difficult task
147: because some types are not meant to be easily cloned. We managed to develop ways to clone any type that can be
148: present in a stypy type store with the following functions, ensuring a proper SSA implementation.
149: '''
150: 
151: 
152: def __duplicate_function(f):
153:     '''
154:     Clone an existing function
155:     :param f: Function to clone
156:     :return: An independent copy of the function
157:     '''
158:     return types.FunctionType(f.func_code, f.func_globals, name=f.func_name,
159:                               argdefs=f.func_defaults,
160:                               closure=f.func_closure)
161: 
162: 
163: def __duplicate_class(clazz):
164:     '''
165:     Clone a class object, creating a duplicate of all its members
166:     :param clazz: Original class
167:     :return: A clone of the class (same name, same members, same inheritance relationship, different identity
168:     '''
169:     # New-style classes duplication
170:     if is_new_style_class(clazz):
171:         return type(clazz.__name__, clazz.__bases__, dict(clazz.__dict__))
172:     else:
173:         # Old-style class duplication
174:         # "Canvas" blank class to write to
175:         class DummyClass:
176:             pass
177: 
178:         DummyClass.__name__ = clazz.__name__
179:         DummyClass.__bases__ = clazz.__bases__
180: 
181:         DummyClass.__dict__ = dict()
182:         for member in clazz.__dict__:
183:             DummyClass.__dict__[member] = clazz.__dict__[member]
184: 
185:         return DummyClass
186: 
187: 
188: def __deepest_possible_copy(type_inference_proxy_obj):
189:     '''
190:     Create a deep copy of the passed type inference proxy, cloning all its members as best as possible to ensure that
191:     deep copies are used whenever possible
192:     :param type_inference_proxy_obj: Original type inference proxy
193:     :return: Clone of the passed object
194:     '''
195: 
196:     # Clone attributes.
197:     try:
198:         # Try the use the Python way of making deep copies first
199:         result = copy.deepcopy(type_inference_proxy_obj)
200:     except:
201:         # If it fails, shallow copy the object attributes
202:         result = copy.copy(type_inference_proxy_obj)
203: 
204:     # Clone represented Python entity
205:     try:
206:         # Is the entity structurally modifiable? If not, just copy it by means of Python API
207:         if not supports_structural_reflection(type_inference_proxy_obj.python_entity):
208:             result.python_entity = copy.deepcopy(type_inference_proxy_obj.python_entity)
209:         else:
210:             # If the structure of the entity is modifiable, we need an independent clone if the entity.
211:             # Classes have an special way of generating clones.
212:             if inspect.isclass(type_inference_proxy_obj.python_entity):
213:                 if type_inference_proxy_obj.instance is None:
214:                     result.python_entity = __duplicate_class(type_inference_proxy_obj.python_entity)
215:                 else:
216:                     # Class instances do not copy its class
217:                     result.python_entity = type_inference_proxy_obj.python_entity
218:             else:
219:                 # Functions also have an special way of cloning them
220:                 if inspect.isfunction(type_inference_proxy_obj.python_entity):
221:                     result.python_entity = __duplicate_function(type_inference_proxy_obj.python_entity)
222:                 else:
223:                     # Deep copy is the default method for the rest of elements
224:                     result.python_entity = copy.deepcopy(type_inference_proxy_obj.python_entity)
225:     except Exception as ex:
226:         # If deep copy fails, we use the shallow copy approach, except from modules, who has an alternate deep copy
227:         # procedure
228:         if inspect.ismodule(type_inference_proxy_obj.python_entity):
229:             result.python_entity = __clone_module(type_inference_proxy_obj.python_entity)
230:         else:
231:             result.python_entity = copy.copy(type_inference_proxy_obj.python_entity)
232: 
233:     # Clone instance (if any)
234:     try:
235:         result.instance = copy.deepcopy(type_inference_proxy_obj.instance)
236:     except:
237:         result.instance = copy.copy(type_inference_proxy_obj.instance)
238: 
239:     # Clone contained types (if any)
240:     if hasattr(type_inference_proxy_obj, type_inference_proxy_obj.contained_elements_property_name):
241:         # try:
242:         # setattr(result, type_inference_proxy_obj.contained_elements_property_name,
243:         #         copy.deepcopy(
244:         #             getattr(type_inference_proxy_obj, type_inference_proxy_obj.contained_elements_property_name)))
245: 
246:         contained_elements = getattr(type_inference_proxy_obj,
247:                                      type_inference_proxy_obj.contained_elements_property_name)
248:         if contained_elements is None:
249:             setattr(result, type_inference_proxy_obj.contained_elements_property_name,
250:                     None)
251:         else:
252:             try:
253:                 # Using the TypeInferenceProxy own clone method for the contained elements
254:                 setattr(result, type_inference_proxy_obj.contained_elements_property_name,
255:                         contained_elements.clone())
256:             except:
257:                 # If cloning fails, manually copy the contents of the contained elements structure
258:                 # Storing a dictionary?
259:                 if isinstance(contained_elements, dict):
260:                     # Reset the stored dictionary of type maps (shallow copy of the original) and set each value
261:                     result.set_elements_type(None, dict(), False)
262:                     for key in contained_elements.keys():
263:                         value = type_inference_proxy_obj.get_values_from_key(None, key)
264:                         result.add_key_and_value_type(None, (key, value), False)
265:                 else:
266:                     # Storing a list?
267:                     setattr(result, type_inference_proxy_obj.contained_elements_property_name,
268:                             copy.deepcopy(contained_elements))
269: 
270:     return result
271: 
272: 
273: def __clone_module(module):
274:     '''
275:     Clone a module. This is done by deleting the loaded module and reloading it again with a different name. Later on,
276:     we restore the unloaded copy.
277:     :param module: Module to clone.
278:     :return: Clone of the module.
279:     '''
280:     original_members = module.__dict__
281:     try:
282:         del sys.modules[module.__name__]
283:     except:
284:         pass
285: 
286:     try:
287:         if "_clone" in module.__name__:
288:             real_module_name = module.__name__.replace("_clone", "")
289:         else:
290:             real_module_name = module.__name__
291:         clone = imp.load_module(module.__name__ + "_clone", *imp.find_module(real_module_name))
292: 
293:         #clone_members = clone.__dict__
294:         for member in original_members:
295:             setattr(clone, member, original_members[member])
296: 
297:     except Exception as e:
298:         clone = module # shallow copy if all else fails
299: 
300:     sys.modules[module.__name__] = module
301:     return clone
302: 
303: 
304: def create_duplicate(entity):
305:     '''
306:     Launch the cloning procedure of a TypeInferenceProxy
307:     :param entity: TypeInferenceProxy to clone
308:     :return: Clone of the passed entity
309:     '''
310:     try:
311:         return __deepest_possible_copy(entity)
312:     except:
313:         return copy.deepcopy(entity)
314: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import inspect' statement (line 1)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import sys' statement (line 2)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import imp' statement (line 3)
import imp

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'imp', imp, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import copy' statement (line 4)
import copy

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'copy', copy, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import types' statement (line 5)
import types

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_12107 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy')

if (type(import_12107) is not StypyTypeError):

    if (import_12107 != 'pyd_module'):
        __import__(import_12107)
        sys_modules_12108 = sys.modules[import_12107]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', sys_modules_12108.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_12108, sys_modules_12108.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', import_12107)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

str_12109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, (-1)), 'str', '\nFile that contains helper functions to implement the type_inference_proxy.py functionality, grouped here to improve\nreadability of the code.\n')

# Assigning a Name to a Name (line 13):
# Getting the type of 'None' (line 13)
None_12110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 23), 'None')
# Assigning a type to the variable 'user_defined_modules' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'user_defined_modules', None_12110)

# Assigning a Num to a Name (line 14):
int_12111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'int')
# Assigning a type to the variable 'last_module_len' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'last_module_len', int_12111)

@norecursion
def __init_user_defined_modules(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'stypy_parameters_copy' (line 16)
    stypy_parameters_copy_12112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 65), 'stypy_parameters_copy')
    # Obtaining the member 'PYTHON_EXE_PATH' of a type (line 16)
    PYTHON_EXE_PATH_12113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 65), stypy_parameters_copy_12112, 'PYTHON_EXE_PATH')
    defaults = [PYTHON_EXE_PATH_12113]
    # Create a new context for function '__init_user_defined_modules'
    module_type_store = module_type_store.open_function_context('__init_user_defined_modules', 16, 0, False)
    
    # Passed parameters checking function
    __init_user_defined_modules.stypy_localization = localization
    __init_user_defined_modules.stypy_type_of_self = None
    __init_user_defined_modules.stypy_type_store = module_type_store
    __init_user_defined_modules.stypy_function_name = '__init_user_defined_modules'
    __init_user_defined_modules.stypy_param_names_list = ['default_python_installation_path']
    __init_user_defined_modules.stypy_varargs_param_name = None
    __init_user_defined_modules.stypy_kwargs_param_name = None
    __init_user_defined_modules.stypy_call_defaults = defaults
    __init_user_defined_modules.stypy_call_varargs = varargs
    __init_user_defined_modules.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__init_user_defined_modules', ['default_python_installation_path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__init_user_defined_modules', localization, ['default_python_installation_path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__init_user_defined_modules(...)' code ##################

    str_12114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', '\n    Initializes the user_defined_modules variable\n    ')
    # Marking variables as global (line 20)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 20, 4), 'user_defined_modules')
    # Marking variables as global (line 21)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 21, 4), 'last_module_len')
    
    # Assigning a Call to a Name (line 25):
    
    # Call to replace(...): (line 25)
    # Processing the call arguments (line 25)
    str_12117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 63), 'str', '/')
    str_12118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 68), 'str', '\\')
    # Processing the call keyword arguments (line 25)
    kwargs_12119 = {}
    # Getting the type of 'default_python_installation_path' (line 25)
    default_python_installation_path_12115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), 'default_python_installation_path', False)
    # Obtaining the member 'replace' of a type (line 25)
    replace_12116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 22), default_python_installation_path_12115, 'replace')
    # Calling replace(args, kwargs) (line 25)
    replace_call_result_12120 = invoke(stypy.reporting.localization.Localization(__file__, 25, 22), replace_12116, *[str_12117, str_12118], **kwargs_12119)
    
    # Assigning a type to the variable 'normalized_path' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'normalized_path', replace_call_result_12120)
    
    # Assigning a Call to a Name (line 26):
    
    # Call to items(...): (line 26)
    # Processing the call keyword arguments (line 26)
    kwargs_12124 = {}
    # Getting the type of 'sys' (line 26)
    sys_12121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'sys', False)
    # Obtaining the member 'modules' of a type (line 26)
    modules_12122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 14), sys_12121, 'modules')
    # Obtaining the member 'items' of a type (line 26)
    items_12123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 14), modules_12122, 'items')
    # Calling items(args, kwargs) (line 26)
    items_call_result_12125 = invoke(stypy.reporting.localization.Localization(__file__, 26, 14), items_12123, *[], **kwargs_12124)
    
    # Assigning a type to the variable 'modules' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'modules', items_call_result_12125)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'user_defined_modules' (line 29)
    user_defined_modules_12126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 7), 'user_defined_modules')
    # Getting the type of 'None' (line 29)
    None_12127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 31), 'None')
    # Applying the binary operator 'is' (line 29)
    result_is__12128 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 7), 'is', user_defined_modules_12126, None_12127)
    
    
    
    # Call to len(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'modules' (line 29)
    modules_12130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 43), 'modules', False)
    # Processing the call keyword arguments (line 29)
    kwargs_12131 = {}
    # Getting the type of 'len' (line 29)
    len_12129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 39), 'len', False)
    # Calling len(args, kwargs) (line 29)
    len_call_result_12132 = invoke(stypy.reporting.localization.Localization(__file__, 29, 39), len_12129, *[modules_12130], **kwargs_12131)
    
    # Getting the type of 'last_module_len' (line 29)
    last_module_len_12133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 55), 'last_module_len')
    # Applying the binary operator '!=' (line 29)
    result_ne_12134 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 39), '!=', len_call_result_12132, last_module_len_12133)
    
    # Applying the binary operator 'or' (line 29)
    result_or_keyword_12135 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 7), 'or', result_is__12128, result_ne_12134)
    
    # Testing if the type of an if condition is none (line 29)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 29, 4), result_or_keyword_12135):
        pass
    else:
        
        # Testing the type of an if condition (line 29)
        if_condition_12136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 4), result_or_keyword_12135)
        # Assigning a type to the variable 'if_condition_12136' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'if_condition_12136', if_condition_12136)
        # SSA begins for if statement (line 29)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 30):
        
        # Call to dict(...): (line 30)
        # Processing the call arguments (line 30)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 30, 36, True)
        # Calculating comprehension expression
        # Getting the type of 'modules' (line 30)
        modules_12158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 97), 'modules', False)
        comprehension_12159 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 36), modules_12158)
        # Assigning a type to the variable 'module_name' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 36), 'module_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 36), comprehension_12159))
        # Assigning a type to the variable 'module_desc' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 36), 'module_desc', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 36), comprehension_12159))
        
        # Evaluating a boolean operation
        
        # Getting the type of 'normalized_path' (line 31)
        normalized_path_12141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 36), 'normalized_path', False)
        
        # Call to str(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'module_desc' (line 31)
        module_desc_12143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 63), 'module_desc', False)
        # Processing the call keyword arguments (line 31)
        kwargs_12144 = {}
        # Getting the type of 'str' (line 31)
        str_12142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 59), 'str', False)
        # Calling str(args, kwargs) (line 31)
        str_call_result_12145 = invoke(stypy.reporting.localization.Localization(__file__, 31, 59), str_12142, *[module_desc_12143], **kwargs_12144)
        
        # Applying the binary operator 'notin' (line 31)
        result_contains_12146 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 36), 'notin', normalized_path_12141, str_call_result_12145)
        
        
        str_12147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 80), 'str', 'built-in')
        
        # Call to str(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'module_desc' (line 32)
        module_desc_12149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 40), 'module_desc', False)
        # Processing the call keyword arguments (line 32)
        kwargs_12150 = {}
        # Getting the type of 'str' (line 32)
        str_12148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 36), 'str', False)
        # Calling str(args, kwargs) (line 32)
        str_call_result_12151 = invoke(stypy.reporting.localization.Localization(__file__, 32, 36), str_12148, *[module_desc_12149], **kwargs_12150)
        
        # Applying the binary operator 'notin' (line 31)
        result_contains_12152 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 80), 'notin', str_12147, str_call_result_12151)
        
        # Applying the binary operator 'and' (line 31)
        result_and_keyword_12153 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 36), 'and', result_contains_12146, result_contains_12152)
        
        # Getting the type of 'module_desc' (line 33)
        module_desc_12154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 40), 'module_desc', False)
        # Getting the type of 'None' (line 33)
        None_12155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 59), 'None', False)
        # Applying the binary operator 'isnot' (line 33)
        result_is_not_12156 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 40), 'isnot', module_desc_12154, None_12155)
        
        # Applying the binary operator 'and' (line 31)
        result_and_keyword_12157 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 36), 'and', result_and_keyword_12153, result_is_not_12156)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 30)
        tuple_12138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 30)
        # Adding element type (line 30)
        # Getting the type of 'module_name' (line 30)
        module_name_12139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 37), 'module_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 37), tuple_12138, module_name_12139)
        # Adding element type (line 30)
        # Getting the type of 'module_desc' (line 30)
        module_desc_12140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 50), 'module_desc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 37), tuple_12138, module_desc_12140)
        
        list_12160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 36), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 36), list_12160, tuple_12138)
        # Processing the call keyword arguments (line 30)
        kwargs_12161 = {}
        # Getting the type of 'dict' (line 30)
        dict_12137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 31), 'dict', False)
        # Calling dict(args, kwargs) (line 30)
        dict_call_result_12162 = invoke(stypy.reporting.localization.Localization(__file__, 30, 31), dict_12137, *[list_12160], **kwargs_12161)
        
        # Assigning a type to the variable 'user_defined_modules' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'user_defined_modules', dict_call_result_12162)
        
        # Assigning a Call to a Name (line 34):
        
        # Call to len(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'modules' (line 34)
        modules_12164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'modules', False)
        # Processing the call keyword arguments (line 34)
        kwargs_12165 = {}
        # Getting the type of 'len' (line 34)
        len_12163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 26), 'len', False)
        # Calling len(args, kwargs) (line 34)
        len_call_result_12166 = invoke(stypy.reporting.localization.Localization(__file__, 34, 26), len_12163, *[modules_12164], **kwargs_12165)
        
        # Assigning a type to the variable 'last_module_len' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'last_module_len', len_call_result_12166)
        # SSA join for if statement (line 29)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of '__init_user_defined_modules(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__init_user_defined_modules' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_12167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12167)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__init_user_defined_modules'
    return stypy_return_type_12167

# Assigning a type to the variable '__init_user_defined_modules' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '__init_user_defined_modules', __init_user_defined_modules)

@norecursion
def is_user_defined_module(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'stypy_parameters_copy' (line 37)
    stypy_parameters_copy_12168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 73), 'stypy_parameters_copy')
    # Obtaining the member 'PYTHON_EXE_PATH' of a type (line 37)
    PYTHON_EXE_PATH_12169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 73), stypy_parameters_copy_12168, 'PYTHON_EXE_PATH')
    defaults = [PYTHON_EXE_PATH_12169]
    # Create a new context for function 'is_user_defined_module'
    module_type_store = module_type_store.open_function_context('is_user_defined_module', 37, 0, False)
    
    # Passed parameters checking function
    is_user_defined_module.stypy_localization = localization
    is_user_defined_module.stypy_type_of_self = None
    is_user_defined_module.stypy_type_store = module_type_store
    is_user_defined_module.stypy_function_name = 'is_user_defined_module'
    is_user_defined_module.stypy_param_names_list = ['module_name', 'default_python_installation_path']
    is_user_defined_module.stypy_varargs_param_name = None
    is_user_defined_module.stypy_kwargs_param_name = None
    is_user_defined_module.stypy_call_defaults = defaults
    is_user_defined_module.stypy_call_varargs = varargs
    is_user_defined_module.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_user_defined_module', ['module_name', 'default_python_installation_path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_user_defined_module', localization, ['module_name', 'default_python_installation_path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_user_defined_module(...)' code ##################

    str_12170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, (-1)), 'str', '\n    Determines if the passed module_name is a user created module or a Python library one.\n    :param module_name: Name of the module\n    :param default_python_installation_path: Python executable to use. Can be left blank as it is initialized by default\n     with the PYTHON_EXE_PATH parameter\n    :return: bool\n    ')
    # Marking variables as global (line 45)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 45, 4), 'user_defined_modules')
    
    # Call to __init_user_defined_modules(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'default_python_installation_path' (line 47)
    default_python_installation_path_12172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 32), 'default_python_installation_path', False)
    # Processing the call keyword arguments (line 47)
    kwargs_12173 = {}
    # Getting the type of '__init_user_defined_modules' (line 47)
    init_user_defined_modules_12171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), '__init_user_defined_modules', False)
    # Calling __init_user_defined_modules(args, kwargs) (line 47)
    init_user_defined_modules_call_result_12174 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), init_user_defined_modules_12171, *[default_python_installation_path_12172], **kwargs_12173)
    
    
    # Getting the type of 'module_name' (line 49)
    module_name_12175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'module_name')
    # Getting the type of 'user_defined_modules' (line 49)
    user_defined_modules_12176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'user_defined_modules')
    # Applying the binary operator 'in' (line 49)
    result_contains_12177 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 11), 'in', module_name_12175, user_defined_modules_12176)
    
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type', result_contains_12177)
    
    # ################# End of 'is_user_defined_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_user_defined_module' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_12178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12178)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_user_defined_module'
    return stypy_return_type_12178

# Assigning a type to the variable 'is_user_defined_module' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'is_user_defined_module', is_user_defined_module)

@norecursion
def is_user_defined_class(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'stypy_parameters_copy' (line 52)
    stypy_parameters_copy_12179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 64), 'stypy_parameters_copy')
    # Obtaining the member 'PYTHON_EXE_PATH' of a type (line 52)
    PYTHON_EXE_PATH_12180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 64), stypy_parameters_copy_12179, 'PYTHON_EXE_PATH')
    defaults = [PYTHON_EXE_PATH_12180]
    # Create a new context for function 'is_user_defined_class'
    module_type_store = module_type_store.open_function_context('is_user_defined_class', 52, 0, False)
    
    # Passed parameters checking function
    is_user_defined_class.stypy_localization = localization
    is_user_defined_class.stypy_type_of_self = None
    is_user_defined_class.stypy_type_store = module_type_store
    is_user_defined_class.stypy_function_name = 'is_user_defined_class'
    is_user_defined_class.stypy_param_names_list = ['cls', 'default_python_installation_path']
    is_user_defined_class.stypy_varargs_param_name = None
    is_user_defined_class.stypy_kwargs_param_name = None
    is_user_defined_class.stypy_call_defaults = defaults
    is_user_defined_class.stypy_call_varargs = varargs
    is_user_defined_class.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_user_defined_class', ['cls', 'default_python_installation_path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_user_defined_class', localization, ['cls', 'default_python_installation_path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_user_defined_class(...)' code ##################

    str_12181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', '\n    Determines if the passed class is a user created class or a Python library one.\n    :param cls: Class\n    :param default_python_installation_path: Python executable to use. Can be left blank as it is initialized by default\n     with the PYTHON_EXE_PATH parameter\n    :return:\n    ')
    # Marking variables as global (line 60)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 60, 4), 'user_defined_modules')
    
    
    # Call to isclass(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'cls' (line 62)
    cls_12184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 27), 'cls', False)
    # Processing the call keyword arguments (line 62)
    kwargs_12185 = {}
    # Getting the type of 'inspect' (line 62)
    inspect_12182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'inspect', False)
    # Obtaining the member 'isclass' of a type (line 62)
    isclass_12183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 11), inspect_12182, 'isclass')
    # Calling isclass(args, kwargs) (line 62)
    isclass_call_result_12186 = invoke(stypy.reporting.localization.Localization(__file__, 62, 11), isclass_12183, *[cls_12184], **kwargs_12185)
    
    # Applying the 'not' unary operator (line 62)
    result_not__12187 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 7), 'not', isclass_call_result_12186)
    
    # Testing if the type of an if condition is none (line 62)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 62, 4), result_not__12187):
        pass
    else:
        
        # Testing the type of an if condition (line 62)
        if_condition_12188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 4), result_not__12187)
        # Assigning a type to the variable 'if_condition_12188' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'if_condition_12188', if_condition_12188)
        # SSA begins for if statement (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 63)
        False_12189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'stypy_return_type', False_12189)
        # SSA join for if statement (line 62)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to __init_user_defined_modules(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'default_python_installation_path' (line 65)
    default_python_installation_path_12191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 32), 'default_python_installation_path', False)
    # Processing the call keyword arguments (line 65)
    kwargs_12192 = {}
    # Getting the type of '__init_user_defined_modules' (line 65)
    init_user_defined_modules_12190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), '__init_user_defined_modules', False)
    # Calling __init_user_defined_modules(args, kwargs) (line 65)
    init_user_defined_modules_call_result_12193 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), init_user_defined_modules_12190, *[default_python_installation_path_12191], **kwargs_12192)
    
    
    # Call to is_user_defined_module(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'cls' (line 68)
    cls_12195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 34), 'cls', False)
    # Obtaining the member '__module__' of a type (line 68)
    module___12196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 34), cls_12195, '__module__')
    # Getting the type of 'default_python_installation_path' (line 68)
    default_python_installation_path_12197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 50), 'default_python_installation_path', False)
    # Processing the call keyword arguments (line 68)
    kwargs_12198 = {}
    # Getting the type of 'is_user_defined_module' (line 68)
    is_user_defined_module_12194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'is_user_defined_module', False)
    # Calling is_user_defined_module(args, kwargs) (line 68)
    is_user_defined_module_call_result_12199 = invoke(stypy.reporting.localization.Localization(__file__, 68, 11), is_user_defined_module_12194, *[module___12196, default_python_installation_path_12197], **kwargs_12198)
    
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type', is_user_defined_module_call_result_12199)
    
    # ################# End of 'is_user_defined_class(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_user_defined_class' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_12200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12200)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_user_defined_class'
    return stypy_return_type_12200

# Assigning a type to the variable 'is_user_defined_class' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'is_user_defined_class', is_user_defined_class)

@norecursion
def supports_structural_reflection(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'supports_structural_reflection'
    module_type_store = module_type_store.open_function_context('supports_structural_reflection', 71, 0, False)
    
    # Passed parameters checking function
    supports_structural_reflection.stypy_localization = localization
    supports_structural_reflection.stypy_type_of_self = None
    supports_structural_reflection.stypy_type_store = module_type_store
    supports_structural_reflection.stypy_function_name = 'supports_structural_reflection'
    supports_structural_reflection.stypy_param_names_list = ['obj']
    supports_structural_reflection.stypy_varargs_param_name = None
    supports_structural_reflection.stypy_kwargs_param_name = None
    supports_structural_reflection.stypy_call_defaults = defaults
    supports_structural_reflection.stypy_call_varargs = varargs
    supports_structural_reflection.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'supports_structural_reflection', ['obj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'supports_structural_reflection', localization, ['obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'supports_structural_reflection(...)' code ##################

    str_12201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, (-1)), 'str', '\n    Determines if an object supports structural reflection. An object supports it if it has a __dict__ property and its\n    type is dict (instead of the read-only dictproxy)\n\n    :param obj: Any Python object\n    :return: bool\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 79)
    str_12202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 24), 'str', '__dict__')
    # Getting the type of 'obj' (line 79)
    obj_12203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'obj')
    
    (may_be_12204, more_types_in_union_12205) = may_not_provide_member(str_12202, obj_12203)

    if may_be_12204:

        if more_types_in_union_12205:
            # Runtime conditional SSA (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'obj' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'obj', remove_member_provider_from_union(obj_12203, '__dict__'))
        # Getting the type of 'False' (line 80)
        False_12206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'stypy_return_type', False_12206)

        if more_types_in_union_12205:
            # SSA join for if statement (line 79)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to type(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'obj' (line 82)
    obj_12208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'obj', False)
    # Obtaining the member '__dict__' of a type (line 82)
    dict___12209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), obj_12208, '__dict__')
    # Processing the call keyword arguments (line 82)
    kwargs_12210 = {}
    # Getting the type of 'type' (line 82)
    type_12207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'type', False)
    # Calling type(args, kwargs) (line 82)
    type_call_result_12211 = invoke(stypy.reporting.localization.Localization(__file__, 82, 4), type_12207, *[dict___12209], **kwargs_12210)
    
    # Getting the type of 'dict' (line 82)
    dict_12212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'dict')
    # Applying the binary operator 'is' (line 82)
    result_is__12213 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 4), 'is', type_call_result_12211, dict_12212)
    
    # Testing if the type of an if condition is none (line 82)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 82, 4), result_is__12213):
        
        
        # SSA begins for try-except statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Name to a Subscript (line 86):
        # Getting the type of 'None' (line 86)
        None_12216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 44), 'None')
        # Getting the type of 'obj' (line 86)
        obj_12217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'obj')
        # Obtaining the member '__dict__' of a type (line 86)
        dict___12218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), obj_12217, '__dict__')
        str_12219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'str', '__stypy_probe')
        # Storing an element on a container (line 86)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 12), dict___12218, (str_12219, None_12216))
        # Deleting a member
        # Getting the type of 'obj' (line 87)
        obj_12220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'obj')
        # Obtaining the member '__dict__' of a type (line 87)
        dict___12221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), obj_12220, '__dict__')
        
        # Obtaining the type of the subscript
        str_12222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 29), 'str', '__stypy_probe')
        # Getting the type of 'obj' (line 87)
        obj_12223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'obj')
        # Obtaining the member '__dict__' of a type (line 87)
        dict___12224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), obj_12223, '__dict__')
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___12225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), dict___12224, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_12226 = invoke(stypy.reporting.localization.Localization(__file__, 87, 16), getitem___12225, str_12222)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 12), dict___12221, subscript_call_result_12226)
        # Getting the type of 'True' (line 88)
        True_12227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'stypy_return_type', True_12227)
        # SSA branch for the except part of a try statement (line 85)
        # SSA branch for the except '<any exception>' branch of a try statement (line 85)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'False' (line 90)
        False_12228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'stypy_return_type', False_12228)
        # SSA join for try-except statement (line 85)
        module_type_store = module_type_store.join_ssa_context()
        
    else:
        
        # Testing the type of an if condition (line 82)
        if_condition_12214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 4), result_is__12213)
        # Assigning a type to the variable 'if_condition_12214' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'if_condition_12214', if_condition_12214)
        # SSA begins for if statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 83)
        True_12215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'stypy_return_type', True_12215)
        # SSA branch for the else part of an if statement (line 82)
        module_type_store.open_ssa_branch('else')
        
        
        # SSA begins for try-except statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Name to a Subscript (line 86):
        # Getting the type of 'None' (line 86)
        None_12216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 44), 'None')
        # Getting the type of 'obj' (line 86)
        obj_12217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'obj')
        # Obtaining the member '__dict__' of a type (line 86)
        dict___12218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), obj_12217, '__dict__')
        str_12219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'str', '__stypy_probe')
        # Storing an element on a container (line 86)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 12), dict___12218, (str_12219, None_12216))
        # Deleting a member
        # Getting the type of 'obj' (line 87)
        obj_12220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'obj')
        # Obtaining the member '__dict__' of a type (line 87)
        dict___12221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), obj_12220, '__dict__')
        
        # Obtaining the type of the subscript
        str_12222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 29), 'str', '__stypy_probe')
        # Getting the type of 'obj' (line 87)
        obj_12223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'obj')
        # Obtaining the member '__dict__' of a type (line 87)
        dict___12224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), obj_12223, '__dict__')
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___12225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), dict___12224, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_12226 = invoke(stypy.reporting.localization.Localization(__file__, 87, 16), getitem___12225, str_12222)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 12), dict___12221, subscript_call_result_12226)
        # Getting the type of 'True' (line 88)
        True_12227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'stypy_return_type', True_12227)
        # SSA branch for the except part of a try statement (line 85)
        # SSA branch for the except '<any exception>' branch of a try statement (line 85)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'False' (line 90)
        False_12228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'stypy_return_type', False_12228)
        # SSA join for try-except statement (line 85)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'supports_structural_reflection(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'supports_structural_reflection' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_12229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12229)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'supports_structural_reflection'
    return stypy_return_type_12229

# Assigning a type to the variable 'supports_structural_reflection' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'supports_structural_reflection', supports_structural_reflection)

@norecursion
def is_class(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_class'
    module_type_store = module_type_store.open_function_context('is_class', 93, 0, False)
    
    # Passed parameters checking function
    is_class.stypy_localization = localization
    is_class.stypy_type_of_self = None
    is_class.stypy_type_store = module_type_store
    is_class.stypy_function_name = 'is_class'
    is_class.stypy_param_names_list = ['cls']
    is_class.stypy_varargs_param_name = None
    is_class.stypy_kwargs_param_name = None
    is_class.stypy_call_defaults = defaults
    is_class.stypy_call_varargs = varargs
    is_class.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_class', ['cls'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_class', localization, ['cls'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_class(...)' code ##################

    str_12230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, (-1)), 'str', '\n    Shortcut to inspect.isclass\n    :param cls: Any Python object\n    :return:\n    ')
    
    # Call to isclass(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'cls' (line 99)
    cls_12233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 27), 'cls', False)
    # Processing the call keyword arguments (line 99)
    kwargs_12234 = {}
    # Getting the type of 'inspect' (line 99)
    inspect_12231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'inspect', False)
    # Obtaining the member 'isclass' of a type (line 99)
    isclass_12232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 11), inspect_12231, 'isclass')
    # Calling isclass(args, kwargs) (line 99)
    isclass_call_result_12235 = invoke(stypy.reporting.localization.Localization(__file__, 99, 11), isclass_12232, *[cls_12233], **kwargs_12234)
    
    # Assigning a type to the variable 'stypy_return_type' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type', isclass_call_result_12235)
    
    # ################# End of 'is_class(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_class' in the type store
    # Getting the type of 'stypy_return_type' (line 93)
    stypy_return_type_12236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12236)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_class'
    return stypy_return_type_12236

# Assigning a type to the variable 'is_class' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'is_class', is_class)

@norecursion
def is_old_style_class(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_old_style_class'
    module_type_store = module_type_store.open_function_context('is_old_style_class', 102, 0, False)
    
    # Passed parameters checking function
    is_old_style_class.stypy_localization = localization
    is_old_style_class.stypy_type_of_self = None
    is_old_style_class.stypy_type_store = module_type_store
    is_old_style_class.stypy_function_name = 'is_old_style_class'
    is_old_style_class.stypy_param_names_list = ['cls']
    is_old_style_class.stypy_varargs_param_name = None
    is_old_style_class.stypy_kwargs_param_name = None
    is_old_style_class.stypy_call_defaults = defaults
    is_old_style_class.stypy_call_varargs = varargs
    is_old_style_class.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_old_style_class', ['cls'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_old_style_class', localization, ['cls'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_old_style_class(...)' code ##################

    str_12237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, (-1)), 'str', '\n    Python supports two type of classes: old-style classes (those that do not inherit from object) and new-style classes\n    (those that do inherit from object). The best way to distinguish between them is to check if the class has an\n     __mro__ (method resolution order) property (only available to new-style classes). Distinguishing between both types\n     is important specially when dealing with type change or supertype change operations, as new-style classes are\n     more limited in that sense and both types cannot be mixed in one of these operations.\n    :param cls: Class to test\n    :return: bool\n    ')
    
    
    # Call to is_class(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'cls' (line 112)
    cls_12239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'cls', False)
    # Processing the call keyword arguments (line 112)
    kwargs_12240 = {}
    # Getting the type of 'is_class' (line 112)
    is_class_12238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'is_class', False)
    # Calling is_class(args, kwargs) (line 112)
    is_class_call_result_12241 = invoke(stypy.reporting.localization.Localization(__file__, 112, 11), is_class_12238, *[cls_12239], **kwargs_12240)
    
    # Applying the 'not' unary operator (line 112)
    result_not__12242 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 7), 'not', is_class_call_result_12241)
    
    # Testing if the type of an if condition is none (line 112)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 112, 4), result_not__12242):
        pass
    else:
        
        # Testing the type of an if condition (line 112)
        if_condition_12243 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 4), result_not__12242)
        # Assigning a type to the variable 'if_condition_12243' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'if_condition_12243', if_condition_12243)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 113)
        False_12244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'stypy_return_type', False_12244)
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to hasattr(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'cls' (line 114)
    cls_12246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'cls', False)
    str_12247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 28), 'str', '__mro__')
    # Processing the call keyword arguments (line 114)
    kwargs_12248 = {}
    # Getting the type of 'hasattr' (line 114)
    hasattr_12245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 114)
    hasattr_call_result_12249 = invoke(stypy.reporting.localization.Localization(__file__, 114, 15), hasattr_12245, *[cls_12246, str_12247], **kwargs_12248)
    
    # Applying the 'not' unary operator (line 114)
    result_not__12250 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 11), 'not', hasattr_call_result_12249)
    
    # Assigning a type to the variable 'stypy_return_type' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type', result_not__12250)
    
    # ################# End of 'is_old_style_class(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_old_style_class' in the type store
    # Getting the type of 'stypy_return_type' (line 102)
    stypy_return_type_12251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12251)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_old_style_class'
    return stypy_return_type_12251

# Assigning a type to the variable 'is_old_style_class' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'is_old_style_class', is_old_style_class)

@norecursion
def is_new_style_class(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_new_style_class'
    module_type_store = module_type_store.open_function_context('is_new_style_class', 117, 0, False)
    
    # Passed parameters checking function
    is_new_style_class.stypy_localization = localization
    is_new_style_class.stypy_type_of_self = None
    is_new_style_class.stypy_type_store = module_type_store
    is_new_style_class.stypy_function_name = 'is_new_style_class'
    is_new_style_class.stypy_param_names_list = ['cls']
    is_new_style_class.stypy_varargs_param_name = None
    is_new_style_class.stypy_kwargs_param_name = None
    is_new_style_class.stypy_call_defaults = defaults
    is_new_style_class.stypy_call_varargs = varargs
    is_new_style_class.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_new_style_class', ['cls'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_new_style_class', localization, ['cls'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_new_style_class(...)' code ##################

    str_12252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, (-1)), 'str', '\n    This method is a shortcut to the opposite of the previous one\n    :param cls: Class to test\n    :return: bool\n    ')
    
    
    # Call to is_old_style_class(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'cls' (line 123)
    cls_12254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 34), 'cls', False)
    # Processing the call keyword arguments (line 123)
    kwargs_12255 = {}
    # Getting the type of 'is_old_style_class' (line 123)
    is_old_style_class_12253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'is_old_style_class', False)
    # Calling is_old_style_class(args, kwargs) (line 123)
    is_old_style_class_call_result_12256 = invoke(stypy.reporting.localization.Localization(__file__, 123, 15), is_old_style_class_12253, *[cls_12254], **kwargs_12255)
    
    # Applying the 'not' unary operator (line 123)
    result_not__12257 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 11), 'not', is_old_style_class_call_result_12256)
    
    # Assigning a type to the variable 'stypy_return_type' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type', result_not__12257)
    
    # ################# End of 'is_new_style_class(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_new_style_class' in the type store
    # Getting the type of 'stypy_return_type' (line 117)
    stypy_return_type_12258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12258)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_new_style_class'
    return stypy_return_type_12258

# Assigning a type to the variable 'is_new_style_class' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'is_new_style_class', is_new_style_class)
str_12259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, (-1)), 'str', '\nCloning Python types is a key part of the implementation of the SSA algorithm. However, this is a very difficult task\nbecause some types are not meant to be easily cloned. We managed to develop ways to clone any type that can be\npresent in a stypy type store with the following functions, ensuring a proper SSA implementation.\n')

@norecursion
def __duplicate_function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__duplicate_function'
    module_type_store = module_type_store.open_function_context('__duplicate_function', 152, 0, False)
    
    # Passed parameters checking function
    __duplicate_function.stypy_localization = localization
    __duplicate_function.stypy_type_of_self = None
    __duplicate_function.stypy_type_store = module_type_store
    __duplicate_function.stypy_function_name = '__duplicate_function'
    __duplicate_function.stypy_param_names_list = ['f']
    __duplicate_function.stypy_varargs_param_name = None
    __duplicate_function.stypy_kwargs_param_name = None
    __duplicate_function.stypy_call_defaults = defaults
    __duplicate_function.stypy_call_varargs = varargs
    __duplicate_function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__duplicate_function', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__duplicate_function', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__duplicate_function(...)' code ##################

    str_12260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, (-1)), 'str', '\n    Clone an existing function\n    :param f: Function to clone\n    :return: An independent copy of the function\n    ')
    
    # Call to FunctionType(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'f' (line 158)
    f_12263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 30), 'f', False)
    # Obtaining the member 'func_code' of a type (line 158)
    func_code_12264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 30), f_12263, 'func_code')
    # Getting the type of 'f' (line 158)
    f_12265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 43), 'f', False)
    # Obtaining the member 'func_globals' of a type (line 158)
    func_globals_12266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 43), f_12265, 'func_globals')
    # Processing the call keyword arguments (line 158)
    # Getting the type of 'f' (line 158)
    f_12267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 64), 'f', False)
    # Obtaining the member 'func_name' of a type (line 158)
    func_name_12268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 64), f_12267, 'func_name')
    keyword_12269 = func_name_12268
    # Getting the type of 'f' (line 159)
    f_12270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 38), 'f', False)
    # Obtaining the member 'func_defaults' of a type (line 159)
    func_defaults_12271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 38), f_12270, 'func_defaults')
    keyword_12272 = func_defaults_12271
    # Getting the type of 'f' (line 160)
    f_12273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 38), 'f', False)
    # Obtaining the member 'func_closure' of a type (line 160)
    func_closure_12274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 38), f_12273, 'func_closure')
    keyword_12275 = func_closure_12274
    kwargs_12276 = {'closure': keyword_12275, 'name': keyword_12269, 'argdefs': keyword_12272}
    # Getting the type of 'types' (line 158)
    types_12261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'types', False)
    # Obtaining the member 'FunctionType' of a type (line 158)
    FunctionType_12262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 11), types_12261, 'FunctionType')
    # Calling FunctionType(args, kwargs) (line 158)
    FunctionType_call_result_12277 = invoke(stypy.reporting.localization.Localization(__file__, 158, 11), FunctionType_12262, *[func_code_12264, func_globals_12266], **kwargs_12276)
    
    # Assigning a type to the variable 'stypy_return_type' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'stypy_return_type', FunctionType_call_result_12277)
    
    # ################# End of '__duplicate_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__duplicate_function' in the type store
    # Getting the type of 'stypy_return_type' (line 152)
    stypy_return_type_12278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12278)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__duplicate_function'
    return stypy_return_type_12278

# Assigning a type to the variable '__duplicate_function' (line 152)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), '__duplicate_function', __duplicate_function)

@norecursion
def __duplicate_class(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__duplicate_class'
    module_type_store = module_type_store.open_function_context('__duplicate_class', 163, 0, False)
    
    # Passed parameters checking function
    __duplicate_class.stypy_localization = localization
    __duplicate_class.stypy_type_of_self = None
    __duplicate_class.stypy_type_store = module_type_store
    __duplicate_class.stypy_function_name = '__duplicate_class'
    __duplicate_class.stypy_param_names_list = ['clazz']
    __duplicate_class.stypy_varargs_param_name = None
    __duplicate_class.stypy_kwargs_param_name = None
    __duplicate_class.stypy_call_defaults = defaults
    __duplicate_class.stypy_call_varargs = varargs
    __duplicate_class.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__duplicate_class', ['clazz'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__duplicate_class', localization, ['clazz'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__duplicate_class(...)' code ##################

    str_12279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, (-1)), 'str', '\n    Clone a class object, creating a duplicate of all its members\n    :param clazz: Original class\n    :return: A clone of the class (same name, same members, same inheritance relationship, different identity\n    ')
    
    # Call to is_new_style_class(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'clazz' (line 170)
    clazz_12281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 26), 'clazz', False)
    # Processing the call keyword arguments (line 170)
    kwargs_12282 = {}
    # Getting the type of 'is_new_style_class' (line 170)
    is_new_style_class_12280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 7), 'is_new_style_class', False)
    # Calling is_new_style_class(args, kwargs) (line 170)
    is_new_style_class_call_result_12283 = invoke(stypy.reporting.localization.Localization(__file__, 170, 7), is_new_style_class_12280, *[clazz_12281], **kwargs_12282)
    
    # Testing if the type of an if condition is none (line 170)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 170, 4), is_new_style_class_call_result_12283):
        # Declaration of the 'DummyClass' class

        class DummyClass:
            pass

            @norecursion
            def __init__(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '__init__'
                module_type_store = module_type_store.open_function_context('__init__', 175, 8, False)
                # Assigning a type to the variable 'self' (line 176)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self', type_of_self)
                
                # Passed parameters checking function
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyClass.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a type to the variable 'DummyClass' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'DummyClass', DummyClass)
        
        # Assigning a Attribute to a Attribute (line 178):
        # Getting the type of 'clazz' (line 178)
        clazz_12297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 30), 'clazz')
        # Obtaining the member '__name__' of a type (line 178)
        name___12298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 30), clazz_12297, '__name__')
        # Getting the type of 'DummyClass' (line 178)
        DummyClass_12299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'DummyClass')
        # Setting the type of the member '__name__' of a type (line 178)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), DummyClass_12299, '__name__', name___12298)
        
        # Assigning a Attribute to a Attribute (line 179):
        # Getting the type of 'clazz' (line 179)
        clazz_12300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 31), 'clazz')
        # Obtaining the member '__bases__' of a type (line 179)
        bases___12301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 31), clazz_12300, '__bases__')
        # Getting the type of 'DummyClass' (line 179)
        DummyClass_12302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'DummyClass')
        # Setting the type of the member '__bases__' of a type (line 179)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), DummyClass_12302, '__bases__', bases___12301)
        
        # Assigning a Call to a Attribute (line 181):
        
        # Call to dict(...): (line 181)
        # Processing the call keyword arguments (line 181)
        kwargs_12304 = {}
        # Getting the type of 'dict' (line 181)
        dict_12303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 30), 'dict', False)
        # Calling dict(args, kwargs) (line 181)
        dict_call_result_12305 = invoke(stypy.reporting.localization.Localization(__file__, 181, 30), dict_12303, *[], **kwargs_12304)
        
        # Getting the type of 'DummyClass' (line 181)
        DummyClass_12306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'DummyClass')
        # Setting the type of the member '__dict__' of a type (line 181)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), DummyClass_12306, '__dict__', dict_call_result_12305)
        
        # Getting the type of 'clazz' (line 182)
        clazz_12307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'clazz')
        # Obtaining the member '__dict__' of a type (line 182)
        dict___12308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 22), clazz_12307, '__dict__')
        # Assigning a type to the variable 'dict___12308' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'dict___12308', dict___12308)
        # Testing if the for loop is going to be iterated (line 182)
        # Testing the type of a for loop iterable (line 182)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 182, 8), dict___12308)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 182, 8), dict___12308):
            # Getting the type of the for loop variable (line 182)
            for_loop_var_12309 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 182, 8), dict___12308)
            # Assigning a type to the variable 'member' (line 182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'member', for_loop_var_12309)
            # SSA begins for a for statement (line 182)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Subscript (line 183):
            
            # Obtaining the type of the subscript
            # Getting the type of 'member' (line 183)
            member_12310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 57), 'member')
            # Getting the type of 'clazz' (line 183)
            clazz_12311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 42), 'clazz')
            # Obtaining the member '__dict__' of a type (line 183)
            dict___12312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 42), clazz_12311, '__dict__')
            # Obtaining the member '__getitem__' of a type (line 183)
            getitem___12313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 42), dict___12312, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 183)
            subscript_call_result_12314 = invoke(stypy.reporting.localization.Localization(__file__, 183, 42), getitem___12313, member_12310)
            
            # Getting the type of 'DummyClass' (line 183)
            DummyClass_12315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'DummyClass')
            # Obtaining the member '__dict__' of a type (line 183)
            dict___12316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), DummyClass_12315, '__dict__')
            # Getting the type of 'member' (line 183)
            member_12317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 32), 'member')
            # Storing an element on a container (line 183)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 12), dict___12316, (member_12317, subscript_call_result_12314))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'DummyClass' (line 185)
        DummyClass_12318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 'DummyClass')
        # Assigning a type to the variable 'stypy_return_type' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'stypy_return_type', DummyClass_12318)
    else:
        
        # Testing the type of an if condition (line 170)
        if_condition_12284 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 4), is_new_style_class_call_result_12283)
        # Assigning a type to the variable 'if_condition_12284' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'if_condition_12284', if_condition_12284)
        # SSA begins for if statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to type(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'clazz' (line 171)
        clazz_12286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'clazz', False)
        # Obtaining the member '__name__' of a type (line 171)
        name___12287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 20), clazz_12286, '__name__')
        # Getting the type of 'clazz' (line 171)
        clazz_12288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 36), 'clazz', False)
        # Obtaining the member '__bases__' of a type (line 171)
        bases___12289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 36), clazz_12288, '__bases__')
        
        # Call to dict(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'clazz' (line 171)
        clazz_12291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 58), 'clazz', False)
        # Obtaining the member '__dict__' of a type (line 171)
        dict___12292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 58), clazz_12291, '__dict__')
        # Processing the call keyword arguments (line 171)
        kwargs_12293 = {}
        # Getting the type of 'dict' (line 171)
        dict_12290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 53), 'dict', False)
        # Calling dict(args, kwargs) (line 171)
        dict_call_result_12294 = invoke(stypy.reporting.localization.Localization(__file__, 171, 53), dict_12290, *[dict___12292], **kwargs_12293)
        
        # Processing the call keyword arguments (line 171)
        kwargs_12295 = {}
        # Getting the type of 'type' (line 171)
        type_12285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'type', False)
        # Calling type(args, kwargs) (line 171)
        type_call_result_12296 = invoke(stypy.reporting.localization.Localization(__file__, 171, 15), type_12285, *[name___12287, bases___12289, dict_call_result_12294], **kwargs_12295)
        
        # Assigning a type to the variable 'stypy_return_type' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'stypy_return_type', type_call_result_12296)
        # SSA branch for the else part of an if statement (line 170)
        module_type_store.open_ssa_branch('else')
        # Declaration of the 'DummyClass' class

        class DummyClass:
            pass

            @norecursion
            def __init__(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '__init__'
                module_type_store = module_type_store.open_function_context('__init__', 175, 8, False)
                # Assigning a type to the variable 'self' (line 176)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self', type_of_self)
                
                # Passed parameters checking function
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyClass.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a type to the variable 'DummyClass' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'DummyClass', DummyClass)
        
        # Assigning a Attribute to a Attribute (line 178):
        # Getting the type of 'clazz' (line 178)
        clazz_12297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 30), 'clazz')
        # Obtaining the member '__name__' of a type (line 178)
        name___12298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 30), clazz_12297, '__name__')
        # Getting the type of 'DummyClass' (line 178)
        DummyClass_12299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'DummyClass')
        # Setting the type of the member '__name__' of a type (line 178)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), DummyClass_12299, '__name__', name___12298)
        
        # Assigning a Attribute to a Attribute (line 179):
        # Getting the type of 'clazz' (line 179)
        clazz_12300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 31), 'clazz')
        # Obtaining the member '__bases__' of a type (line 179)
        bases___12301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 31), clazz_12300, '__bases__')
        # Getting the type of 'DummyClass' (line 179)
        DummyClass_12302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'DummyClass')
        # Setting the type of the member '__bases__' of a type (line 179)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), DummyClass_12302, '__bases__', bases___12301)
        
        # Assigning a Call to a Attribute (line 181):
        
        # Call to dict(...): (line 181)
        # Processing the call keyword arguments (line 181)
        kwargs_12304 = {}
        # Getting the type of 'dict' (line 181)
        dict_12303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 30), 'dict', False)
        # Calling dict(args, kwargs) (line 181)
        dict_call_result_12305 = invoke(stypy.reporting.localization.Localization(__file__, 181, 30), dict_12303, *[], **kwargs_12304)
        
        # Getting the type of 'DummyClass' (line 181)
        DummyClass_12306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'DummyClass')
        # Setting the type of the member '__dict__' of a type (line 181)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), DummyClass_12306, '__dict__', dict_call_result_12305)
        
        # Getting the type of 'clazz' (line 182)
        clazz_12307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'clazz')
        # Obtaining the member '__dict__' of a type (line 182)
        dict___12308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 22), clazz_12307, '__dict__')
        # Assigning a type to the variable 'dict___12308' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'dict___12308', dict___12308)
        # Testing if the for loop is going to be iterated (line 182)
        # Testing the type of a for loop iterable (line 182)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 182, 8), dict___12308)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 182, 8), dict___12308):
            # Getting the type of the for loop variable (line 182)
            for_loop_var_12309 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 182, 8), dict___12308)
            # Assigning a type to the variable 'member' (line 182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'member', for_loop_var_12309)
            # SSA begins for a for statement (line 182)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Subscript (line 183):
            
            # Obtaining the type of the subscript
            # Getting the type of 'member' (line 183)
            member_12310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 57), 'member')
            # Getting the type of 'clazz' (line 183)
            clazz_12311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 42), 'clazz')
            # Obtaining the member '__dict__' of a type (line 183)
            dict___12312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 42), clazz_12311, '__dict__')
            # Obtaining the member '__getitem__' of a type (line 183)
            getitem___12313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 42), dict___12312, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 183)
            subscript_call_result_12314 = invoke(stypy.reporting.localization.Localization(__file__, 183, 42), getitem___12313, member_12310)
            
            # Getting the type of 'DummyClass' (line 183)
            DummyClass_12315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'DummyClass')
            # Obtaining the member '__dict__' of a type (line 183)
            dict___12316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), DummyClass_12315, '__dict__')
            # Getting the type of 'member' (line 183)
            member_12317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 32), 'member')
            # Storing an element on a container (line 183)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 12), dict___12316, (member_12317, subscript_call_result_12314))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'DummyClass' (line 185)
        DummyClass_12318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 'DummyClass')
        # Assigning a type to the variable 'stypy_return_type' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'stypy_return_type', DummyClass_12318)
        # SSA join for if statement (line 170)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of '__duplicate_class(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__duplicate_class' in the type store
    # Getting the type of 'stypy_return_type' (line 163)
    stypy_return_type_12319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12319)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__duplicate_class'
    return stypy_return_type_12319

# Assigning a type to the variable '__duplicate_class' (line 163)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), '__duplicate_class', __duplicate_class)

@norecursion
def __deepest_possible_copy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__deepest_possible_copy'
    module_type_store = module_type_store.open_function_context('__deepest_possible_copy', 188, 0, False)
    
    # Passed parameters checking function
    __deepest_possible_copy.stypy_localization = localization
    __deepest_possible_copy.stypy_type_of_self = None
    __deepest_possible_copy.stypy_type_store = module_type_store
    __deepest_possible_copy.stypy_function_name = '__deepest_possible_copy'
    __deepest_possible_copy.stypy_param_names_list = ['type_inference_proxy_obj']
    __deepest_possible_copy.stypy_varargs_param_name = None
    __deepest_possible_copy.stypy_kwargs_param_name = None
    __deepest_possible_copy.stypy_call_defaults = defaults
    __deepest_possible_copy.stypy_call_varargs = varargs
    __deepest_possible_copy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__deepest_possible_copy', ['type_inference_proxy_obj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__deepest_possible_copy', localization, ['type_inference_proxy_obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__deepest_possible_copy(...)' code ##################

    str_12320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, (-1)), 'str', '\n    Create a deep copy of the passed type inference proxy, cloning all its members as best as possible to ensure that\n    deep copies are used whenever possible\n    :param type_inference_proxy_obj: Original type inference proxy\n    :return: Clone of the passed object\n    ')
    
    
    # SSA begins for try-except statement (line 197)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 199):
    
    # Call to deepcopy(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'type_inference_proxy_obj' (line 199)
    type_inference_proxy_obj_12323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 31), 'type_inference_proxy_obj', False)
    # Processing the call keyword arguments (line 199)
    kwargs_12324 = {}
    # Getting the type of 'copy' (line 199)
    copy_12321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 17), 'copy', False)
    # Obtaining the member 'deepcopy' of a type (line 199)
    deepcopy_12322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 17), copy_12321, 'deepcopy')
    # Calling deepcopy(args, kwargs) (line 199)
    deepcopy_call_result_12325 = invoke(stypy.reporting.localization.Localization(__file__, 199, 17), deepcopy_12322, *[type_inference_proxy_obj_12323], **kwargs_12324)
    
    # Assigning a type to the variable 'result' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'result', deepcopy_call_result_12325)
    # SSA branch for the except part of a try statement (line 197)
    # SSA branch for the except '<any exception>' branch of a try statement (line 197)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 202):
    
    # Call to copy(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'type_inference_proxy_obj' (line 202)
    type_inference_proxy_obj_12328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 27), 'type_inference_proxy_obj', False)
    # Processing the call keyword arguments (line 202)
    kwargs_12329 = {}
    # Getting the type of 'copy' (line 202)
    copy_12326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 17), 'copy', False)
    # Obtaining the member 'copy' of a type (line 202)
    copy_12327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 17), copy_12326, 'copy')
    # Calling copy(args, kwargs) (line 202)
    copy_call_result_12330 = invoke(stypy.reporting.localization.Localization(__file__, 202, 17), copy_12327, *[type_inference_proxy_obj_12328], **kwargs_12329)
    
    # Assigning a type to the variable 'result' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'result', copy_call_result_12330)
    # SSA join for try-except statement (line 197)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    
    # Call to supports_structural_reflection(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'type_inference_proxy_obj' (line 207)
    type_inference_proxy_obj_12332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 46), 'type_inference_proxy_obj', False)
    # Obtaining the member 'python_entity' of a type (line 207)
    python_entity_12333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 46), type_inference_proxy_obj_12332, 'python_entity')
    # Processing the call keyword arguments (line 207)
    kwargs_12334 = {}
    # Getting the type of 'supports_structural_reflection' (line 207)
    supports_structural_reflection_12331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'supports_structural_reflection', False)
    # Calling supports_structural_reflection(args, kwargs) (line 207)
    supports_structural_reflection_call_result_12335 = invoke(stypy.reporting.localization.Localization(__file__, 207, 15), supports_structural_reflection_12331, *[python_entity_12333], **kwargs_12334)
    
    # Applying the 'not' unary operator (line 207)
    result_not__12336 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 11), 'not', supports_structural_reflection_call_result_12335)
    
    # Testing if the type of an if condition is none (line 207)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 207, 8), result_not__12336):
        
        # Call to isclass(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'type_inference_proxy_obj' (line 212)
        type_inference_proxy_obj_12347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 31), 'type_inference_proxy_obj', False)
        # Obtaining the member 'python_entity' of a type (line 212)
        python_entity_12348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 31), type_inference_proxy_obj_12347, 'python_entity')
        # Processing the call keyword arguments (line 212)
        kwargs_12349 = {}
        # Getting the type of 'inspect' (line 212)
        inspect_12345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 212)
        isclass_12346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 15), inspect_12345, 'isclass')
        # Calling isclass(args, kwargs) (line 212)
        isclass_call_result_12350 = invoke(stypy.reporting.localization.Localization(__file__, 212, 15), isclass_12346, *[python_entity_12348], **kwargs_12349)
        
        # Testing if the type of an if condition is none (line 212)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 212, 12), isclass_call_result_12350):
            
            # Call to isfunction(...): (line 220)
            # Processing the call arguments (line 220)
            # Getting the type of 'type_inference_proxy_obj' (line 220)
            type_inference_proxy_obj_12368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'type_inference_proxy_obj', False)
            # Obtaining the member 'python_entity' of a type (line 220)
            python_entity_12369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 38), type_inference_proxy_obj_12368, 'python_entity')
            # Processing the call keyword arguments (line 220)
            kwargs_12370 = {}
            # Getting the type of 'inspect' (line 220)
            inspect_12366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'inspect', False)
            # Obtaining the member 'isfunction' of a type (line 220)
            isfunction_12367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 19), inspect_12366, 'isfunction')
            # Calling isfunction(args, kwargs) (line 220)
            isfunction_call_result_12371 = invoke(stypy.reporting.localization.Localization(__file__, 220, 19), isfunction_12367, *[python_entity_12369], **kwargs_12370)
            
            # Testing if the type of an if condition is none (line 220)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 220, 16), isfunction_call_result_12371):
                
                # Assigning a Call to a Attribute (line 224):
                
                # Call to deepcopy(...): (line 224)
                # Processing the call arguments (line 224)
                # Getting the type of 'type_inference_proxy_obj' (line 224)
                type_inference_proxy_obj_12381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 224)
                python_entity_12382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 57), type_inference_proxy_obj_12381, 'python_entity')
                # Processing the call keyword arguments (line 224)
                kwargs_12383 = {}
                # Getting the type of 'copy' (line 224)
                copy_12379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 224)
                deepcopy_12380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), copy_12379, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 224)
                deepcopy_call_result_12384 = invoke(stypy.reporting.localization.Localization(__file__, 224, 43), deepcopy_12380, *[python_entity_12382], **kwargs_12383)
                
                # Getting the type of 'result' (line 224)
                result_12385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 224)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), result_12385, 'python_entity', deepcopy_call_result_12384)
            else:
                
                # Testing the type of an if condition (line 220)
                if_condition_12372 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 16), isfunction_call_result_12371)
                # Assigning a type to the variable 'if_condition_12372' (line 220)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'if_condition_12372', if_condition_12372)
                # SSA begins for if statement (line 220)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Attribute (line 221):
                
                # Call to __duplicate_function(...): (line 221)
                # Processing the call arguments (line 221)
                # Getting the type of 'type_inference_proxy_obj' (line 221)
                type_inference_proxy_obj_12374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 64), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 221)
                python_entity_12375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 64), type_inference_proxy_obj_12374, 'python_entity')
                # Processing the call keyword arguments (line 221)
                kwargs_12376 = {}
                # Getting the type of '__duplicate_function' (line 221)
                duplicate_function_12373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 43), '__duplicate_function', False)
                # Calling __duplicate_function(args, kwargs) (line 221)
                duplicate_function_call_result_12377 = invoke(stypy.reporting.localization.Localization(__file__, 221, 43), duplicate_function_12373, *[python_entity_12375], **kwargs_12376)
                
                # Getting the type of 'result' (line 221)
                result_12378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 221)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 20), result_12378, 'python_entity', duplicate_function_call_result_12377)
                # SSA branch for the else part of an if statement (line 220)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Attribute (line 224):
                
                # Call to deepcopy(...): (line 224)
                # Processing the call arguments (line 224)
                # Getting the type of 'type_inference_proxy_obj' (line 224)
                type_inference_proxy_obj_12381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 224)
                python_entity_12382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 57), type_inference_proxy_obj_12381, 'python_entity')
                # Processing the call keyword arguments (line 224)
                kwargs_12383 = {}
                # Getting the type of 'copy' (line 224)
                copy_12379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 224)
                deepcopy_12380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), copy_12379, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 224)
                deepcopy_call_result_12384 = invoke(stypy.reporting.localization.Localization(__file__, 224, 43), deepcopy_12380, *[python_entity_12382], **kwargs_12383)
                
                # Getting the type of 'result' (line 224)
                result_12385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 224)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), result_12385, 'python_entity', deepcopy_call_result_12384)
                # SSA join for if statement (line 220)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 212)
            if_condition_12351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 12), isclass_call_result_12350)
            # Assigning a type to the variable 'if_condition_12351' (line 212)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'if_condition_12351', if_condition_12351)
            # SSA begins for if statement (line 212)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Type idiom detected: calculating its left and rigth part (line 213)
            # Getting the type of 'type_inference_proxy_obj' (line 213)
            type_inference_proxy_obj_12352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'type_inference_proxy_obj')
            # Obtaining the member 'instance' of a type (line 213)
            instance_12353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 19), type_inference_proxy_obj_12352, 'instance')
            # Getting the type of 'None' (line 213)
            None_12354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 56), 'None')
            
            (may_be_12355, more_types_in_union_12356) = may_be_none(instance_12353, None_12354)

            if may_be_12355:

                if more_types_in_union_12356:
                    # Runtime conditional SSA (line 213)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Call to a Attribute (line 214):
                
                # Call to __duplicate_class(...): (line 214)
                # Processing the call arguments (line 214)
                # Getting the type of 'type_inference_proxy_obj' (line 214)
                type_inference_proxy_obj_12358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 61), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 214)
                python_entity_12359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 61), type_inference_proxy_obj_12358, 'python_entity')
                # Processing the call keyword arguments (line 214)
                kwargs_12360 = {}
                # Getting the type of '__duplicate_class' (line 214)
                duplicate_class_12357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 43), '__duplicate_class', False)
                # Calling __duplicate_class(args, kwargs) (line 214)
                duplicate_class_call_result_12361 = invoke(stypy.reporting.localization.Localization(__file__, 214, 43), duplicate_class_12357, *[python_entity_12359], **kwargs_12360)
                
                # Getting the type of 'result' (line 214)
                result_12362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 214)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 20), result_12362, 'python_entity', duplicate_class_call_result_12361)

                if more_types_in_union_12356:
                    # Runtime conditional SSA for else branch (line 213)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_12355) or more_types_in_union_12356):
                
                # Assigning a Attribute to a Attribute (line 217):
                # Getting the type of 'type_inference_proxy_obj' (line 217)
                type_inference_proxy_obj_12363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 43), 'type_inference_proxy_obj')
                # Obtaining the member 'python_entity' of a type (line 217)
                python_entity_12364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 43), type_inference_proxy_obj_12363, 'python_entity')
                # Getting the type of 'result' (line 217)
                result_12365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 217)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 20), result_12365, 'python_entity', python_entity_12364)

                if (may_be_12355 and more_types_in_union_12356):
                    # SSA join for if statement (line 213)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA branch for the else part of an if statement (line 212)
            module_type_store.open_ssa_branch('else')
            
            # Call to isfunction(...): (line 220)
            # Processing the call arguments (line 220)
            # Getting the type of 'type_inference_proxy_obj' (line 220)
            type_inference_proxy_obj_12368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'type_inference_proxy_obj', False)
            # Obtaining the member 'python_entity' of a type (line 220)
            python_entity_12369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 38), type_inference_proxy_obj_12368, 'python_entity')
            # Processing the call keyword arguments (line 220)
            kwargs_12370 = {}
            # Getting the type of 'inspect' (line 220)
            inspect_12366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'inspect', False)
            # Obtaining the member 'isfunction' of a type (line 220)
            isfunction_12367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 19), inspect_12366, 'isfunction')
            # Calling isfunction(args, kwargs) (line 220)
            isfunction_call_result_12371 = invoke(stypy.reporting.localization.Localization(__file__, 220, 19), isfunction_12367, *[python_entity_12369], **kwargs_12370)
            
            # Testing if the type of an if condition is none (line 220)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 220, 16), isfunction_call_result_12371):
                
                # Assigning a Call to a Attribute (line 224):
                
                # Call to deepcopy(...): (line 224)
                # Processing the call arguments (line 224)
                # Getting the type of 'type_inference_proxy_obj' (line 224)
                type_inference_proxy_obj_12381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 224)
                python_entity_12382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 57), type_inference_proxy_obj_12381, 'python_entity')
                # Processing the call keyword arguments (line 224)
                kwargs_12383 = {}
                # Getting the type of 'copy' (line 224)
                copy_12379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 224)
                deepcopy_12380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), copy_12379, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 224)
                deepcopy_call_result_12384 = invoke(stypy.reporting.localization.Localization(__file__, 224, 43), deepcopy_12380, *[python_entity_12382], **kwargs_12383)
                
                # Getting the type of 'result' (line 224)
                result_12385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 224)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), result_12385, 'python_entity', deepcopy_call_result_12384)
            else:
                
                # Testing the type of an if condition (line 220)
                if_condition_12372 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 16), isfunction_call_result_12371)
                # Assigning a type to the variable 'if_condition_12372' (line 220)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'if_condition_12372', if_condition_12372)
                # SSA begins for if statement (line 220)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Attribute (line 221):
                
                # Call to __duplicate_function(...): (line 221)
                # Processing the call arguments (line 221)
                # Getting the type of 'type_inference_proxy_obj' (line 221)
                type_inference_proxy_obj_12374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 64), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 221)
                python_entity_12375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 64), type_inference_proxy_obj_12374, 'python_entity')
                # Processing the call keyword arguments (line 221)
                kwargs_12376 = {}
                # Getting the type of '__duplicate_function' (line 221)
                duplicate_function_12373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 43), '__duplicate_function', False)
                # Calling __duplicate_function(args, kwargs) (line 221)
                duplicate_function_call_result_12377 = invoke(stypy.reporting.localization.Localization(__file__, 221, 43), duplicate_function_12373, *[python_entity_12375], **kwargs_12376)
                
                # Getting the type of 'result' (line 221)
                result_12378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 221)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 20), result_12378, 'python_entity', duplicate_function_call_result_12377)
                # SSA branch for the else part of an if statement (line 220)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Attribute (line 224):
                
                # Call to deepcopy(...): (line 224)
                # Processing the call arguments (line 224)
                # Getting the type of 'type_inference_proxy_obj' (line 224)
                type_inference_proxy_obj_12381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 224)
                python_entity_12382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 57), type_inference_proxy_obj_12381, 'python_entity')
                # Processing the call keyword arguments (line 224)
                kwargs_12383 = {}
                # Getting the type of 'copy' (line 224)
                copy_12379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 224)
                deepcopy_12380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), copy_12379, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 224)
                deepcopy_call_result_12384 = invoke(stypy.reporting.localization.Localization(__file__, 224, 43), deepcopy_12380, *[python_entity_12382], **kwargs_12383)
                
                # Getting the type of 'result' (line 224)
                result_12385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 224)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), result_12385, 'python_entity', deepcopy_call_result_12384)
                # SSA join for if statement (line 220)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 212)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 207)
        if_condition_12337 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 8), result_not__12336)
        # Assigning a type to the variable 'if_condition_12337' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'if_condition_12337', if_condition_12337)
        # SSA begins for if statement (line 207)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 208):
        
        # Call to deepcopy(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'type_inference_proxy_obj' (line 208)
        type_inference_proxy_obj_12340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 49), 'type_inference_proxy_obj', False)
        # Obtaining the member 'python_entity' of a type (line 208)
        python_entity_12341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 49), type_inference_proxy_obj_12340, 'python_entity')
        # Processing the call keyword arguments (line 208)
        kwargs_12342 = {}
        # Getting the type of 'copy' (line 208)
        copy_12338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 35), 'copy', False)
        # Obtaining the member 'deepcopy' of a type (line 208)
        deepcopy_12339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 35), copy_12338, 'deepcopy')
        # Calling deepcopy(args, kwargs) (line 208)
        deepcopy_call_result_12343 = invoke(stypy.reporting.localization.Localization(__file__, 208, 35), deepcopy_12339, *[python_entity_12341], **kwargs_12342)
        
        # Getting the type of 'result' (line 208)
        result_12344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'result')
        # Setting the type of the member 'python_entity' of a type (line 208)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 12), result_12344, 'python_entity', deepcopy_call_result_12343)
        # SSA branch for the else part of an if statement (line 207)
        module_type_store.open_ssa_branch('else')
        
        # Call to isclass(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'type_inference_proxy_obj' (line 212)
        type_inference_proxy_obj_12347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 31), 'type_inference_proxy_obj', False)
        # Obtaining the member 'python_entity' of a type (line 212)
        python_entity_12348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 31), type_inference_proxy_obj_12347, 'python_entity')
        # Processing the call keyword arguments (line 212)
        kwargs_12349 = {}
        # Getting the type of 'inspect' (line 212)
        inspect_12345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 212)
        isclass_12346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 15), inspect_12345, 'isclass')
        # Calling isclass(args, kwargs) (line 212)
        isclass_call_result_12350 = invoke(stypy.reporting.localization.Localization(__file__, 212, 15), isclass_12346, *[python_entity_12348], **kwargs_12349)
        
        # Testing if the type of an if condition is none (line 212)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 212, 12), isclass_call_result_12350):
            
            # Call to isfunction(...): (line 220)
            # Processing the call arguments (line 220)
            # Getting the type of 'type_inference_proxy_obj' (line 220)
            type_inference_proxy_obj_12368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'type_inference_proxy_obj', False)
            # Obtaining the member 'python_entity' of a type (line 220)
            python_entity_12369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 38), type_inference_proxy_obj_12368, 'python_entity')
            # Processing the call keyword arguments (line 220)
            kwargs_12370 = {}
            # Getting the type of 'inspect' (line 220)
            inspect_12366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'inspect', False)
            # Obtaining the member 'isfunction' of a type (line 220)
            isfunction_12367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 19), inspect_12366, 'isfunction')
            # Calling isfunction(args, kwargs) (line 220)
            isfunction_call_result_12371 = invoke(stypy.reporting.localization.Localization(__file__, 220, 19), isfunction_12367, *[python_entity_12369], **kwargs_12370)
            
            # Testing if the type of an if condition is none (line 220)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 220, 16), isfunction_call_result_12371):
                
                # Assigning a Call to a Attribute (line 224):
                
                # Call to deepcopy(...): (line 224)
                # Processing the call arguments (line 224)
                # Getting the type of 'type_inference_proxy_obj' (line 224)
                type_inference_proxy_obj_12381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 224)
                python_entity_12382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 57), type_inference_proxy_obj_12381, 'python_entity')
                # Processing the call keyword arguments (line 224)
                kwargs_12383 = {}
                # Getting the type of 'copy' (line 224)
                copy_12379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 224)
                deepcopy_12380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), copy_12379, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 224)
                deepcopy_call_result_12384 = invoke(stypy.reporting.localization.Localization(__file__, 224, 43), deepcopy_12380, *[python_entity_12382], **kwargs_12383)
                
                # Getting the type of 'result' (line 224)
                result_12385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 224)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), result_12385, 'python_entity', deepcopy_call_result_12384)
            else:
                
                # Testing the type of an if condition (line 220)
                if_condition_12372 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 16), isfunction_call_result_12371)
                # Assigning a type to the variable 'if_condition_12372' (line 220)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'if_condition_12372', if_condition_12372)
                # SSA begins for if statement (line 220)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Attribute (line 221):
                
                # Call to __duplicate_function(...): (line 221)
                # Processing the call arguments (line 221)
                # Getting the type of 'type_inference_proxy_obj' (line 221)
                type_inference_proxy_obj_12374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 64), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 221)
                python_entity_12375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 64), type_inference_proxy_obj_12374, 'python_entity')
                # Processing the call keyword arguments (line 221)
                kwargs_12376 = {}
                # Getting the type of '__duplicate_function' (line 221)
                duplicate_function_12373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 43), '__duplicate_function', False)
                # Calling __duplicate_function(args, kwargs) (line 221)
                duplicate_function_call_result_12377 = invoke(stypy.reporting.localization.Localization(__file__, 221, 43), duplicate_function_12373, *[python_entity_12375], **kwargs_12376)
                
                # Getting the type of 'result' (line 221)
                result_12378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 221)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 20), result_12378, 'python_entity', duplicate_function_call_result_12377)
                # SSA branch for the else part of an if statement (line 220)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Attribute (line 224):
                
                # Call to deepcopy(...): (line 224)
                # Processing the call arguments (line 224)
                # Getting the type of 'type_inference_proxy_obj' (line 224)
                type_inference_proxy_obj_12381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 224)
                python_entity_12382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 57), type_inference_proxy_obj_12381, 'python_entity')
                # Processing the call keyword arguments (line 224)
                kwargs_12383 = {}
                # Getting the type of 'copy' (line 224)
                copy_12379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 224)
                deepcopy_12380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), copy_12379, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 224)
                deepcopy_call_result_12384 = invoke(stypy.reporting.localization.Localization(__file__, 224, 43), deepcopy_12380, *[python_entity_12382], **kwargs_12383)
                
                # Getting the type of 'result' (line 224)
                result_12385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 224)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), result_12385, 'python_entity', deepcopy_call_result_12384)
                # SSA join for if statement (line 220)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 212)
            if_condition_12351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 12), isclass_call_result_12350)
            # Assigning a type to the variable 'if_condition_12351' (line 212)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'if_condition_12351', if_condition_12351)
            # SSA begins for if statement (line 212)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Type idiom detected: calculating its left and rigth part (line 213)
            # Getting the type of 'type_inference_proxy_obj' (line 213)
            type_inference_proxy_obj_12352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'type_inference_proxy_obj')
            # Obtaining the member 'instance' of a type (line 213)
            instance_12353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 19), type_inference_proxy_obj_12352, 'instance')
            # Getting the type of 'None' (line 213)
            None_12354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 56), 'None')
            
            (may_be_12355, more_types_in_union_12356) = may_be_none(instance_12353, None_12354)

            if may_be_12355:

                if more_types_in_union_12356:
                    # Runtime conditional SSA (line 213)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Call to a Attribute (line 214):
                
                # Call to __duplicate_class(...): (line 214)
                # Processing the call arguments (line 214)
                # Getting the type of 'type_inference_proxy_obj' (line 214)
                type_inference_proxy_obj_12358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 61), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 214)
                python_entity_12359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 61), type_inference_proxy_obj_12358, 'python_entity')
                # Processing the call keyword arguments (line 214)
                kwargs_12360 = {}
                # Getting the type of '__duplicate_class' (line 214)
                duplicate_class_12357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 43), '__duplicate_class', False)
                # Calling __duplicate_class(args, kwargs) (line 214)
                duplicate_class_call_result_12361 = invoke(stypy.reporting.localization.Localization(__file__, 214, 43), duplicate_class_12357, *[python_entity_12359], **kwargs_12360)
                
                # Getting the type of 'result' (line 214)
                result_12362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 214)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 20), result_12362, 'python_entity', duplicate_class_call_result_12361)

                if more_types_in_union_12356:
                    # Runtime conditional SSA for else branch (line 213)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_12355) or more_types_in_union_12356):
                
                # Assigning a Attribute to a Attribute (line 217):
                # Getting the type of 'type_inference_proxy_obj' (line 217)
                type_inference_proxy_obj_12363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 43), 'type_inference_proxy_obj')
                # Obtaining the member 'python_entity' of a type (line 217)
                python_entity_12364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 43), type_inference_proxy_obj_12363, 'python_entity')
                # Getting the type of 'result' (line 217)
                result_12365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 217)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 20), result_12365, 'python_entity', python_entity_12364)

                if (may_be_12355 and more_types_in_union_12356):
                    # SSA join for if statement (line 213)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA branch for the else part of an if statement (line 212)
            module_type_store.open_ssa_branch('else')
            
            # Call to isfunction(...): (line 220)
            # Processing the call arguments (line 220)
            # Getting the type of 'type_inference_proxy_obj' (line 220)
            type_inference_proxy_obj_12368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'type_inference_proxy_obj', False)
            # Obtaining the member 'python_entity' of a type (line 220)
            python_entity_12369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 38), type_inference_proxy_obj_12368, 'python_entity')
            # Processing the call keyword arguments (line 220)
            kwargs_12370 = {}
            # Getting the type of 'inspect' (line 220)
            inspect_12366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'inspect', False)
            # Obtaining the member 'isfunction' of a type (line 220)
            isfunction_12367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 19), inspect_12366, 'isfunction')
            # Calling isfunction(args, kwargs) (line 220)
            isfunction_call_result_12371 = invoke(stypy.reporting.localization.Localization(__file__, 220, 19), isfunction_12367, *[python_entity_12369], **kwargs_12370)
            
            # Testing if the type of an if condition is none (line 220)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 220, 16), isfunction_call_result_12371):
                
                # Assigning a Call to a Attribute (line 224):
                
                # Call to deepcopy(...): (line 224)
                # Processing the call arguments (line 224)
                # Getting the type of 'type_inference_proxy_obj' (line 224)
                type_inference_proxy_obj_12381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 224)
                python_entity_12382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 57), type_inference_proxy_obj_12381, 'python_entity')
                # Processing the call keyword arguments (line 224)
                kwargs_12383 = {}
                # Getting the type of 'copy' (line 224)
                copy_12379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 224)
                deepcopy_12380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), copy_12379, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 224)
                deepcopy_call_result_12384 = invoke(stypy.reporting.localization.Localization(__file__, 224, 43), deepcopy_12380, *[python_entity_12382], **kwargs_12383)
                
                # Getting the type of 'result' (line 224)
                result_12385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 224)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), result_12385, 'python_entity', deepcopy_call_result_12384)
            else:
                
                # Testing the type of an if condition (line 220)
                if_condition_12372 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 16), isfunction_call_result_12371)
                # Assigning a type to the variable 'if_condition_12372' (line 220)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'if_condition_12372', if_condition_12372)
                # SSA begins for if statement (line 220)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Attribute (line 221):
                
                # Call to __duplicate_function(...): (line 221)
                # Processing the call arguments (line 221)
                # Getting the type of 'type_inference_proxy_obj' (line 221)
                type_inference_proxy_obj_12374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 64), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 221)
                python_entity_12375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 64), type_inference_proxy_obj_12374, 'python_entity')
                # Processing the call keyword arguments (line 221)
                kwargs_12376 = {}
                # Getting the type of '__duplicate_function' (line 221)
                duplicate_function_12373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 43), '__duplicate_function', False)
                # Calling __duplicate_function(args, kwargs) (line 221)
                duplicate_function_call_result_12377 = invoke(stypy.reporting.localization.Localization(__file__, 221, 43), duplicate_function_12373, *[python_entity_12375], **kwargs_12376)
                
                # Getting the type of 'result' (line 221)
                result_12378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 221)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 20), result_12378, 'python_entity', duplicate_function_call_result_12377)
                # SSA branch for the else part of an if statement (line 220)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Attribute (line 224):
                
                # Call to deepcopy(...): (line 224)
                # Processing the call arguments (line 224)
                # Getting the type of 'type_inference_proxy_obj' (line 224)
                type_inference_proxy_obj_12381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 224)
                python_entity_12382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 57), type_inference_proxy_obj_12381, 'python_entity')
                # Processing the call keyword arguments (line 224)
                kwargs_12383 = {}
                # Getting the type of 'copy' (line 224)
                copy_12379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 224)
                deepcopy_12380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), copy_12379, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 224)
                deepcopy_call_result_12384 = invoke(stypy.reporting.localization.Localization(__file__, 224, 43), deepcopy_12380, *[python_entity_12382], **kwargs_12383)
                
                # Getting the type of 'result' (line 224)
                result_12385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 224)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), result_12385, 'python_entity', deepcopy_call_result_12384)
                # SSA join for if statement (line 220)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 212)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 207)
        module_type_store = module_type_store.join_ssa_context()
        

    # SSA branch for the except part of a try statement (line 205)
    # SSA branch for the except 'Exception' branch of a try statement (line 205)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 225)
    Exception_12386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 'Exception')
    # Assigning a type to the variable 'ex' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'ex', Exception_12386)
    
    # Call to ismodule(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'type_inference_proxy_obj' (line 228)
    type_inference_proxy_obj_12389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 28), 'type_inference_proxy_obj', False)
    # Obtaining the member 'python_entity' of a type (line 228)
    python_entity_12390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 28), type_inference_proxy_obj_12389, 'python_entity')
    # Processing the call keyword arguments (line 228)
    kwargs_12391 = {}
    # Getting the type of 'inspect' (line 228)
    inspect_12387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'inspect', False)
    # Obtaining the member 'ismodule' of a type (line 228)
    ismodule_12388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 11), inspect_12387, 'ismodule')
    # Calling ismodule(args, kwargs) (line 228)
    ismodule_call_result_12392 = invoke(stypy.reporting.localization.Localization(__file__, 228, 11), ismodule_12388, *[python_entity_12390], **kwargs_12391)
    
    # Testing if the type of an if condition is none (line 228)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 228, 8), ismodule_call_result_12392):
        
        # Assigning a Call to a Attribute (line 231):
        
        # Call to copy(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'type_inference_proxy_obj' (line 231)
        type_inference_proxy_obj_12402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 45), 'type_inference_proxy_obj', False)
        # Obtaining the member 'python_entity' of a type (line 231)
        python_entity_12403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 45), type_inference_proxy_obj_12402, 'python_entity')
        # Processing the call keyword arguments (line 231)
        kwargs_12404 = {}
        # Getting the type of 'copy' (line 231)
        copy_12400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 35), 'copy', False)
        # Obtaining the member 'copy' of a type (line 231)
        copy_12401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 35), copy_12400, 'copy')
        # Calling copy(args, kwargs) (line 231)
        copy_call_result_12405 = invoke(stypy.reporting.localization.Localization(__file__, 231, 35), copy_12401, *[python_entity_12403], **kwargs_12404)
        
        # Getting the type of 'result' (line 231)
        result_12406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'result')
        # Setting the type of the member 'python_entity' of a type (line 231)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), result_12406, 'python_entity', copy_call_result_12405)
    else:
        
        # Testing the type of an if condition (line 228)
        if_condition_12393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 8), ismodule_call_result_12392)
        # Assigning a type to the variable 'if_condition_12393' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'if_condition_12393', if_condition_12393)
        # SSA begins for if statement (line 228)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 229):
        
        # Call to __clone_module(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'type_inference_proxy_obj' (line 229)
        type_inference_proxy_obj_12395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 50), 'type_inference_proxy_obj', False)
        # Obtaining the member 'python_entity' of a type (line 229)
        python_entity_12396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 50), type_inference_proxy_obj_12395, 'python_entity')
        # Processing the call keyword arguments (line 229)
        kwargs_12397 = {}
        # Getting the type of '__clone_module' (line 229)
        clone_module_12394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 35), '__clone_module', False)
        # Calling __clone_module(args, kwargs) (line 229)
        clone_module_call_result_12398 = invoke(stypy.reporting.localization.Localization(__file__, 229, 35), clone_module_12394, *[python_entity_12396], **kwargs_12397)
        
        # Getting the type of 'result' (line 229)
        result_12399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'result')
        # Setting the type of the member 'python_entity' of a type (line 229)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), result_12399, 'python_entity', clone_module_call_result_12398)
        # SSA branch for the else part of an if statement (line 228)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 231):
        
        # Call to copy(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'type_inference_proxy_obj' (line 231)
        type_inference_proxy_obj_12402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 45), 'type_inference_proxy_obj', False)
        # Obtaining the member 'python_entity' of a type (line 231)
        python_entity_12403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 45), type_inference_proxy_obj_12402, 'python_entity')
        # Processing the call keyword arguments (line 231)
        kwargs_12404 = {}
        # Getting the type of 'copy' (line 231)
        copy_12400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 35), 'copy', False)
        # Obtaining the member 'copy' of a type (line 231)
        copy_12401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 35), copy_12400, 'copy')
        # Calling copy(args, kwargs) (line 231)
        copy_call_result_12405 = invoke(stypy.reporting.localization.Localization(__file__, 231, 35), copy_12401, *[python_entity_12403], **kwargs_12404)
        
        # Getting the type of 'result' (line 231)
        result_12406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'result')
        # Setting the type of the member 'python_entity' of a type (line 231)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), result_12406, 'python_entity', copy_call_result_12405)
        # SSA join for if statement (line 228)
        module_type_store = module_type_store.join_ssa_context()
        

    # SSA join for try-except statement (line 205)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 234)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Attribute (line 235):
    
    # Call to deepcopy(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'type_inference_proxy_obj' (line 235)
    type_inference_proxy_obj_12409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 40), 'type_inference_proxy_obj', False)
    # Obtaining the member 'instance' of a type (line 235)
    instance_12410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 40), type_inference_proxy_obj_12409, 'instance')
    # Processing the call keyword arguments (line 235)
    kwargs_12411 = {}
    # Getting the type of 'copy' (line 235)
    copy_12407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 26), 'copy', False)
    # Obtaining the member 'deepcopy' of a type (line 235)
    deepcopy_12408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 26), copy_12407, 'deepcopy')
    # Calling deepcopy(args, kwargs) (line 235)
    deepcopy_call_result_12412 = invoke(stypy.reporting.localization.Localization(__file__, 235, 26), deepcopy_12408, *[instance_12410], **kwargs_12411)
    
    # Getting the type of 'result' (line 235)
    result_12413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'result')
    # Setting the type of the member 'instance' of a type (line 235)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), result_12413, 'instance', deepcopy_call_result_12412)
    # SSA branch for the except part of a try statement (line 234)
    # SSA branch for the except '<any exception>' branch of a try statement (line 234)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Attribute (line 237):
    
    # Call to copy(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'type_inference_proxy_obj' (line 237)
    type_inference_proxy_obj_12416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 36), 'type_inference_proxy_obj', False)
    # Obtaining the member 'instance' of a type (line 237)
    instance_12417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 36), type_inference_proxy_obj_12416, 'instance')
    # Processing the call keyword arguments (line 237)
    kwargs_12418 = {}
    # Getting the type of 'copy' (line 237)
    copy_12414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 26), 'copy', False)
    # Obtaining the member 'copy' of a type (line 237)
    copy_12415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 26), copy_12414, 'copy')
    # Calling copy(args, kwargs) (line 237)
    copy_call_result_12419 = invoke(stypy.reporting.localization.Localization(__file__, 237, 26), copy_12415, *[instance_12417], **kwargs_12418)
    
    # Getting the type of 'result' (line 237)
    result_12420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'result')
    # Setting the type of the member 'instance' of a type (line 237)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), result_12420, 'instance', copy_call_result_12419)
    # SSA join for try-except statement (line 234)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to hasattr(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'type_inference_proxy_obj' (line 240)
    type_inference_proxy_obj_12422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'type_inference_proxy_obj', False)
    # Getting the type of 'type_inference_proxy_obj' (line 240)
    type_inference_proxy_obj_12423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 41), 'type_inference_proxy_obj', False)
    # Obtaining the member 'contained_elements_property_name' of a type (line 240)
    contained_elements_property_name_12424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 41), type_inference_proxy_obj_12423, 'contained_elements_property_name')
    # Processing the call keyword arguments (line 240)
    kwargs_12425 = {}
    # Getting the type of 'hasattr' (line 240)
    hasattr_12421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 7), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 240)
    hasattr_call_result_12426 = invoke(stypy.reporting.localization.Localization(__file__, 240, 7), hasattr_12421, *[type_inference_proxy_obj_12422, contained_elements_property_name_12424], **kwargs_12425)
    
    # Testing if the type of an if condition is none (line 240)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 240, 4), hasattr_call_result_12426):
        pass
    else:
        
        # Testing the type of an if condition (line 240)
        if_condition_12427 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 4), hasattr_call_result_12426)
        # Assigning a type to the variable 'if_condition_12427' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'if_condition_12427', if_condition_12427)
        # SSA begins for if statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 246):
        
        # Call to getattr(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'type_inference_proxy_obj' (line 246)
        type_inference_proxy_obj_12429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 37), 'type_inference_proxy_obj', False)
        # Getting the type of 'type_inference_proxy_obj' (line 247)
        type_inference_proxy_obj_12430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 37), 'type_inference_proxy_obj', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 247)
        contained_elements_property_name_12431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 37), type_inference_proxy_obj_12430, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 246)
        kwargs_12432 = {}
        # Getting the type of 'getattr' (line 246)
        getattr_12428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 29), 'getattr', False)
        # Calling getattr(args, kwargs) (line 246)
        getattr_call_result_12433 = invoke(stypy.reporting.localization.Localization(__file__, 246, 29), getattr_12428, *[type_inference_proxy_obj_12429, contained_elements_property_name_12431], **kwargs_12432)
        
        # Assigning a type to the variable 'contained_elements' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'contained_elements', getattr_call_result_12433)
        
        # Type idiom detected: calculating its left and rigth part (line 248)
        # Getting the type of 'contained_elements' (line 248)
        contained_elements_12434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 11), 'contained_elements')
        # Getting the type of 'None' (line 248)
        None_12435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 33), 'None')
        
        (may_be_12436, more_types_in_union_12437) = may_be_none(contained_elements_12434, None_12435)

        if may_be_12436:

            if more_types_in_union_12437:
                # Runtime conditional SSA (line 248)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to setattr(...): (line 249)
            # Processing the call arguments (line 249)
            # Getting the type of 'result' (line 249)
            result_12439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'result', False)
            # Getting the type of 'type_inference_proxy_obj' (line 249)
            type_inference_proxy_obj_12440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 28), 'type_inference_proxy_obj', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 249)
            contained_elements_property_name_12441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 28), type_inference_proxy_obj_12440, 'contained_elements_property_name')
            # Getting the type of 'None' (line 250)
            None_12442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 20), 'None', False)
            # Processing the call keyword arguments (line 249)
            kwargs_12443 = {}
            # Getting the type of 'setattr' (line 249)
            setattr_12438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'setattr', False)
            # Calling setattr(args, kwargs) (line 249)
            setattr_call_result_12444 = invoke(stypy.reporting.localization.Localization(__file__, 249, 12), setattr_12438, *[result_12439, contained_elements_property_name_12441, None_12442], **kwargs_12443)
            

            if more_types_in_union_12437:
                # Runtime conditional SSA for else branch (line 248)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_12436) or more_types_in_union_12437):
            
            
            # SSA begins for try-except statement (line 252)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Call to setattr(...): (line 254)
            # Processing the call arguments (line 254)
            # Getting the type of 'result' (line 254)
            result_12446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 24), 'result', False)
            # Getting the type of 'type_inference_proxy_obj' (line 254)
            type_inference_proxy_obj_12447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 32), 'type_inference_proxy_obj', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 254)
            contained_elements_property_name_12448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 32), type_inference_proxy_obj_12447, 'contained_elements_property_name')
            
            # Call to clone(...): (line 255)
            # Processing the call keyword arguments (line 255)
            kwargs_12451 = {}
            # Getting the type of 'contained_elements' (line 255)
            contained_elements_12449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'contained_elements', False)
            # Obtaining the member 'clone' of a type (line 255)
            clone_12450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 24), contained_elements_12449, 'clone')
            # Calling clone(args, kwargs) (line 255)
            clone_call_result_12452 = invoke(stypy.reporting.localization.Localization(__file__, 255, 24), clone_12450, *[], **kwargs_12451)
            
            # Processing the call keyword arguments (line 254)
            kwargs_12453 = {}
            # Getting the type of 'setattr' (line 254)
            setattr_12445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'setattr', False)
            # Calling setattr(args, kwargs) (line 254)
            setattr_call_result_12454 = invoke(stypy.reporting.localization.Localization(__file__, 254, 16), setattr_12445, *[result_12446, contained_elements_property_name_12448, clone_call_result_12452], **kwargs_12453)
            
            # SSA branch for the except part of a try statement (line 252)
            # SSA branch for the except '<any exception>' branch of a try statement (line 252)
            module_type_store.open_ssa_branch('except')
            
            # Type idiom detected: calculating its left and rigth part (line 259)
            # Getting the type of 'dict' (line 259)
            dict_12455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 50), 'dict')
            # Getting the type of 'contained_elements' (line 259)
            contained_elements_12456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 30), 'contained_elements')
            
            (may_be_12457, more_types_in_union_12458) = may_be_subtype(dict_12455, contained_elements_12456)

            if may_be_12457:

                if more_types_in_union_12458:
                    # Runtime conditional SSA (line 259)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'contained_elements' (line 259)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'contained_elements', remove_not_subtype_from_union(contained_elements_12456, dict))
                
                # Call to set_elements_type(...): (line 261)
                # Processing the call arguments (line 261)
                # Getting the type of 'None' (line 261)
                None_12461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 45), 'None', False)
                
                # Call to dict(...): (line 261)
                # Processing the call keyword arguments (line 261)
                kwargs_12463 = {}
                # Getting the type of 'dict' (line 261)
                dict_12462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 51), 'dict', False)
                # Calling dict(args, kwargs) (line 261)
                dict_call_result_12464 = invoke(stypy.reporting.localization.Localization(__file__, 261, 51), dict_12462, *[], **kwargs_12463)
                
                # Getting the type of 'False' (line 261)
                False_12465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 59), 'False', False)
                # Processing the call keyword arguments (line 261)
                kwargs_12466 = {}
                # Getting the type of 'result' (line 261)
                result_12459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 20), 'result', False)
                # Obtaining the member 'set_elements_type' of a type (line 261)
                set_elements_type_12460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 20), result_12459, 'set_elements_type')
                # Calling set_elements_type(args, kwargs) (line 261)
                set_elements_type_call_result_12467 = invoke(stypy.reporting.localization.Localization(__file__, 261, 20), set_elements_type_12460, *[None_12461, dict_call_result_12464, False_12465], **kwargs_12466)
                
                
                
                # Call to keys(...): (line 262)
                # Processing the call keyword arguments (line 262)
                kwargs_12470 = {}
                # Getting the type of 'contained_elements' (line 262)
                contained_elements_12468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 31), 'contained_elements', False)
                # Obtaining the member 'keys' of a type (line 262)
                keys_12469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 31), contained_elements_12468, 'keys')
                # Calling keys(args, kwargs) (line 262)
                keys_call_result_12471 = invoke(stypy.reporting.localization.Localization(__file__, 262, 31), keys_12469, *[], **kwargs_12470)
                
                # Assigning a type to the variable 'keys_call_result_12471' (line 262)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 20), 'keys_call_result_12471', keys_call_result_12471)
                # Testing if the for loop is going to be iterated (line 262)
                # Testing the type of a for loop iterable (line 262)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 262, 20), keys_call_result_12471)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 262, 20), keys_call_result_12471):
                    # Getting the type of the for loop variable (line 262)
                    for_loop_var_12472 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 262, 20), keys_call_result_12471)
                    # Assigning a type to the variable 'key' (line 262)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 20), 'key', for_loop_var_12472)
                    # SSA begins for a for statement (line 262)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a Call to a Name (line 263):
                    
                    # Call to get_values_from_key(...): (line 263)
                    # Processing the call arguments (line 263)
                    # Getting the type of 'None' (line 263)
                    None_12475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 77), 'None', False)
                    # Getting the type of 'key' (line 263)
                    key_12476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 83), 'key', False)
                    # Processing the call keyword arguments (line 263)
                    kwargs_12477 = {}
                    # Getting the type of 'type_inference_proxy_obj' (line 263)
                    type_inference_proxy_obj_12473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 32), 'type_inference_proxy_obj', False)
                    # Obtaining the member 'get_values_from_key' of a type (line 263)
                    get_values_from_key_12474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 32), type_inference_proxy_obj_12473, 'get_values_from_key')
                    # Calling get_values_from_key(args, kwargs) (line 263)
                    get_values_from_key_call_result_12478 = invoke(stypy.reporting.localization.Localization(__file__, 263, 32), get_values_from_key_12474, *[None_12475, key_12476], **kwargs_12477)
                    
                    # Assigning a type to the variable 'value' (line 263)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 24), 'value', get_values_from_key_call_result_12478)
                    
                    # Call to add_key_and_value_type(...): (line 264)
                    # Processing the call arguments (line 264)
                    # Getting the type of 'None' (line 264)
                    None_12481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 54), 'None', False)
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 264)
                    tuple_12482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 61), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 264)
                    # Adding element type (line 264)
                    # Getting the type of 'key' (line 264)
                    key_12483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 61), 'key', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 61), tuple_12482, key_12483)
                    # Adding element type (line 264)
                    # Getting the type of 'value' (line 264)
                    value_12484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 66), 'value', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 61), tuple_12482, value_12484)
                    
                    # Getting the type of 'False' (line 264)
                    False_12485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 74), 'False', False)
                    # Processing the call keyword arguments (line 264)
                    kwargs_12486 = {}
                    # Getting the type of 'result' (line 264)
                    result_12479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 24), 'result', False)
                    # Obtaining the member 'add_key_and_value_type' of a type (line 264)
                    add_key_and_value_type_12480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 24), result_12479, 'add_key_and_value_type')
                    # Calling add_key_and_value_type(args, kwargs) (line 264)
                    add_key_and_value_type_call_result_12487 = invoke(stypy.reporting.localization.Localization(__file__, 264, 24), add_key_and_value_type_12480, *[None_12481, tuple_12482, False_12485], **kwargs_12486)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                

                if more_types_in_union_12458:
                    # Runtime conditional SSA for else branch (line 259)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_12457) or more_types_in_union_12458):
                # Assigning a type to the variable 'contained_elements' (line 259)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'contained_elements', remove_subtype_from_union(contained_elements_12456, dict))
                
                # Call to setattr(...): (line 267)
                # Processing the call arguments (line 267)
                # Getting the type of 'result' (line 267)
                result_12489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 28), 'result', False)
                # Getting the type of 'type_inference_proxy_obj' (line 267)
                type_inference_proxy_obj_12490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 36), 'type_inference_proxy_obj', False)
                # Obtaining the member 'contained_elements_property_name' of a type (line 267)
                contained_elements_property_name_12491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 36), type_inference_proxy_obj_12490, 'contained_elements_property_name')
                
                # Call to deepcopy(...): (line 268)
                # Processing the call arguments (line 268)
                # Getting the type of 'contained_elements' (line 268)
                contained_elements_12494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 42), 'contained_elements', False)
                # Processing the call keyword arguments (line 268)
                kwargs_12495 = {}
                # Getting the type of 'copy' (line 268)
                copy_12492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 28), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 268)
                deepcopy_12493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 28), copy_12492, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 268)
                deepcopy_call_result_12496 = invoke(stypy.reporting.localization.Localization(__file__, 268, 28), deepcopy_12493, *[contained_elements_12494], **kwargs_12495)
                
                # Processing the call keyword arguments (line 267)
                kwargs_12497 = {}
                # Getting the type of 'setattr' (line 267)
                setattr_12488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 20), 'setattr', False)
                # Calling setattr(args, kwargs) (line 267)
                setattr_call_result_12498 = invoke(stypy.reporting.localization.Localization(__file__, 267, 20), setattr_12488, *[result_12489, contained_elements_property_name_12491, deepcopy_call_result_12496], **kwargs_12497)
                

                if (may_be_12457 and more_types_in_union_12458):
                    # SSA join for if statement (line 259)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for try-except statement (line 252)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_12436 and more_types_in_union_12437):
                # SSA join for if statement (line 248)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'result' (line 270)
    result_12499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type', result_12499)
    
    # ################# End of '__deepest_possible_copy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__deepest_possible_copy' in the type store
    # Getting the type of 'stypy_return_type' (line 188)
    stypy_return_type_12500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12500)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__deepest_possible_copy'
    return stypy_return_type_12500

# Assigning a type to the variable '__deepest_possible_copy' (line 188)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 0), '__deepest_possible_copy', __deepest_possible_copy)

@norecursion
def __clone_module(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__clone_module'
    module_type_store = module_type_store.open_function_context('__clone_module', 273, 0, False)
    
    # Passed parameters checking function
    __clone_module.stypy_localization = localization
    __clone_module.stypy_type_of_self = None
    __clone_module.stypy_type_store = module_type_store
    __clone_module.stypy_function_name = '__clone_module'
    __clone_module.stypy_param_names_list = ['module']
    __clone_module.stypy_varargs_param_name = None
    __clone_module.stypy_kwargs_param_name = None
    __clone_module.stypy_call_defaults = defaults
    __clone_module.stypy_call_varargs = varargs
    __clone_module.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__clone_module', ['module'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__clone_module', localization, ['module'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__clone_module(...)' code ##################

    str_12501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, (-1)), 'str', '\n    Clone a module. This is done by deleting the loaded module and reloading it again with a different name. Later on,\n    we restore the unloaded copy.\n    :param module: Module to clone.\n    :return: Clone of the module.\n    ')
    
    # Assigning a Attribute to a Name (line 280):
    # Getting the type of 'module' (line 280)
    module_12502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 23), 'module')
    # Obtaining the member '__dict__' of a type (line 280)
    dict___12503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 23), module_12502, '__dict__')
    # Assigning a type to the variable 'original_members' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'original_members', dict___12503)
    
    
    # SSA begins for try-except statement (line 281)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    # Deleting a member
    # Getting the type of 'sys' (line 282)
    sys_12504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'sys')
    # Obtaining the member 'modules' of a type (line 282)
    modules_12505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), sys_12504, 'modules')
    
    # Obtaining the type of the subscript
    # Getting the type of 'module' (line 282)
    module_12506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'module')
    # Obtaining the member '__name__' of a type (line 282)
    name___12507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 24), module_12506, '__name__')
    # Getting the type of 'sys' (line 282)
    sys_12508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'sys')
    # Obtaining the member 'modules' of a type (line 282)
    modules_12509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), sys_12508, 'modules')
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___12510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), modules_12509, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 282)
    subscript_call_result_12511 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), getitem___12510, name___12507)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 8), modules_12505, subscript_call_result_12511)
    # SSA branch for the except part of a try statement (line 281)
    # SSA branch for the except '<any exception>' branch of a try statement (line 281)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 281)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 286)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    str_12512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 11), 'str', '_clone')
    # Getting the type of 'module' (line 287)
    module_12513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 23), 'module')
    # Obtaining the member '__name__' of a type (line 287)
    name___12514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 23), module_12513, '__name__')
    # Applying the binary operator 'in' (line 287)
    result_contains_12515 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 11), 'in', str_12512, name___12514)
    
    # Testing if the type of an if condition is none (line 287)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 287, 8), result_contains_12515):
        
        # Assigning a Attribute to a Name (line 290):
        # Getting the type of 'module' (line 290)
        module_12524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 31), 'module')
        # Obtaining the member '__name__' of a type (line 290)
        name___12525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 31), module_12524, '__name__')
        # Assigning a type to the variable 'real_module_name' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'real_module_name', name___12525)
    else:
        
        # Testing the type of an if condition (line 287)
        if_condition_12516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 287, 8), result_contains_12515)
        # Assigning a type to the variable 'if_condition_12516' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'if_condition_12516', if_condition_12516)
        # SSA begins for if statement (line 287)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 288):
        
        # Call to replace(...): (line 288)
        # Processing the call arguments (line 288)
        str_12520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 55), 'str', '_clone')
        str_12521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 65), 'str', '')
        # Processing the call keyword arguments (line 288)
        kwargs_12522 = {}
        # Getting the type of 'module' (line 288)
        module_12517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 31), 'module', False)
        # Obtaining the member '__name__' of a type (line 288)
        name___12518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 31), module_12517, '__name__')
        # Obtaining the member 'replace' of a type (line 288)
        replace_12519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 31), name___12518, 'replace')
        # Calling replace(args, kwargs) (line 288)
        replace_call_result_12523 = invoke(stypy.reporting.localization.Localization(__file__, 288, 31), replace_12519, *[str_12520, str_12521], **kwargs_12522)
        
        # Assigning a type to the variable 'real_module_name' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'real_module_name', replace_call_result_12523)
        # SSA branch for the else part of an if statement (line 287)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 290):
        # Getting the type of 'module' (line 290)
        module_12524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 31), 'module')
        # Obtaining the member '__name__' of a type (line 290)
        name___12525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 31), module_12524, '__name__')
        # Assigning a type to the variable 'real_module_name' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'real_module_name', name___12525)
        # SSA join for if statement (line 287)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 291):
    
    # Call to load_module(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'module' (line 291)
    module_12528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 32), 'module', False)
    # Obtaining the member '__name__' of a type (line 291)
    name___12529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 32), module_12528, '__name__')
    str_12530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 50), 'str', '_clone')
    # Applying the binary operator '+' (line 291)
    result_add_12531 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 32), '+', name___12529, str_12530)
    
    
    # Call to find_module(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'real_module_name' (line 291)
    real_module_name_12534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 77), 'real_module_name', False)
    # Processing the call keyword arguments (line 291)
    kwargs_12535 = {}
    # Getting the type of 'imp' (line 291)
    imp_12532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 61), 'imp', False)
    # Obtaining the member 'find_module' of a type (line 291)
    find_module_12533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 61), imp_12532, 'find_module')
    # Calling find_module(args, kwargs) (line 291)
    find_module_call_result_12536 = invoke(stypy.reporting.localization.Localization(__file__, 291, 61), find_module_12533, *[real_module_name_12534], **kwargs_12535)
    
    # Processing the call keyword arguments (line 291)
    kwargs_12537 = {}
    # Getting the type of 'imp' (line 291)
    imp_12526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'imp', False)
    # Obtaining the member 'load_module' of a type (line 291)
    load_module_12527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 16), imp_12526, 'load_module')
    # Calling load_module(args, kwargs) (line 291)
    load_module_call_result_12538 = invoke(stypy.reporting.localization.Localization(__file__, 291, 16), load_module_12527, *[result_add_12531, find_module_call_result_12536], **kwargs_12537)
    
    # Assigning a type to the variable 'clone' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'clone', load_module_call_result_12538)
    
    # Getting the type of 'original_members' (line 294)
    original_members_12539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 22), 'original_members')
    # Assigning a type to the variable 'original_members_12539' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'original_members_12539', original_members_12539)
    # Testing if the for loop is going to be iterated (line 294)
    # Testing the type of a for loop iterable (line 294)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 294, 8), original_members_12539)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 294, 8), original_members_12539):
        # Getting the type of the for loop variable (line 294)
        for_loop_var_12540 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 294, 8), original_members_12539)
        # Assigning a type to the variable 'member' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'member', for_loop_var_12540)
        # SSA begins for a for statement (line 294)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setattr(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'clone' (line 295)
        clone_12542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'clone', False)
        # Getting the type of 'member' (line 295)
        member_12543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 27), 'member', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'member' (line 295)
        member_12544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 52), 'member', False)
        # Getting the type of 'original_members' (line 295)
        original_members_12545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 35), 'original_members', False)
        # Obtaining the member '__getitem__' of a type (line 295)
        getitem___12546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 35), original_members_12545, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 295)
        subscript_call_result_12547 = invoke(stypy.reporting.localization.Localization(__file__, 295, 35), getitem___12546, member_12544)
        
        # Processing the call keyword arguments (line 295)
        kwargs_12548 = {}
        # Getting the type of 'setattr' (line 295)
        setattr_12541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 295)
        setattr_call_result_12549 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), setattr_12541, *[clone_12542, member_12543, subscript_call_result_12547], **kwargs_12548)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # SSA branch for the except part of a try statement (line 286)
    # SSA branch for the except 'Exception' branch of a try statement (line 286)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 297)
    Exception_12550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 11), 'Exception')
    # Assigning a type to the variable 'e' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'e', Exception_12550)
    
    # Assigning a Name to a Name (line 298):
    # Getting the type of 'module' (line 298)
    module_12551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'module')
    # Assigning a type to the variable 'clone' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'clone', module_12551)
    # SSA join for try-except statement (line 286)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 300):
    # Getting the type of 'module' (line 300)
    module_12552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 35), 'module')
    # Getting the type of 'sys' (line 300)
    sys_12553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'sys')
    # Obtaining the member 'modules' of a type (line 300)
    modules_12554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 4), sys_12553, 'modules')
    # Getting the type of 'module' (line 300)
    module_12555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'module')
    # Obtaining the member '__name__' of a type (line 300)
    name___12556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 16), module_12555, '__name__')
    # Storing an element on a container (line 300)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 4), modules_12554, (name___12556, module_12552))
    # Getting the type of 'clone' (line 301)
    clone_12557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 11), 'clone')
    # Assigning a type to the variable 'stypy_return_type' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'stypy_return_type', clone_12557)
    
    # ################# End of '__clone_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__clone_module' in the type store
    # Getting the type of 'stypy_return_type' (line 273)
    stypy_return_type_12558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12558)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__clone_module'
    return stypy_return_type_12558

# Assigning a type to the variable '__clone_module' (line 273)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 0), '__clone_module', __clone_module)

@norecursion
def create_duplicate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_duplicate'
    module_type_store = module_type_store.open_function_context('create_duplicate', 304, 0, False)
    
    # Passed parameters checking function
    create_duplicate.stypy_localization = localization
    create_duplicate.stypy_type_of_self = None
    create_duplicate.stypy_type_store = module_type_store
    create_duplicate.stypy_function_name = 'create_duplicate'
    create_duplicate.stypy_param_names_list = ['entity']
    create_duplicate.stypy_varargs_param_name = None
    create_duplicate.stypy_kwargs_param_name = None
    create_duplicate.stypy_call_defaults = defaults
    create_duplicate.stypy_call_varargs = varargs
    create_duplicate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_duplicate', ['entity'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_duplicate', localization, ['entity'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_duplicate(...)' code ##################

    str_12559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, (-1)), 'str', '\n    Launch the cloning procedure of a TypeInferenceProxy\n    :param entity: TypeInferenceProxy to clone\n    :return: Clone of the passed entity\n    ')
    
    
    # SSA begins for try-except statement (line 310)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to __deepest_possible_copy(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'entity' (line 311)
    entity_12561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 39), 'entity', False)
    # Processing the call keyword arguments (line 311)
    kwargs_12562 = {}
    # Getting the type of '__deepest_possible_copy' (line 311)
    deepest_possible_copy_12560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 15), '__deepest_possible_copy', False)
    # Calling __deepest_possible_copy(args, kwargs) (line 311)
    deepest_possible_copy_call_result_12563 = invoke(stypy.reporting.localization.Localization(__file__, 311, 15), deepest_possible_copy_12560, *[entity_12561], **kwargs_12562)
    
    # Assigning a type to the variable 'stypy_return_type' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'stypy_return_type', deepest_possible_copy_call_result_12563)
    # SSA branch for the except part of a try statement (line 310)
    # SSA branch for the except '<any exception>' branch of a try statement (line 310)
    module_type_store.open_ssa_branch('except')
    
    # Call to deepcopy(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'entity' (line 313)
    entity_12566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 29), 'entity', False)
    # Processing the call keyword arguments (line 313)
    kwargs_12567 = {}
    # Getting the type of 'copy' (line 313)
    copy_12564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 15), 'copy', False)
    # Obtaining the member 'deepcopy' of a type (line 313)
    deepcopy_12565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 15), copy_12564, 'deepcopy')
    # Calling deepcopy(args, kwargs) (line 313)
    deepcopy_call_result_12568 = invoke(stypy.reporting.localization.Localization(__file__, 313, 15), deepcopy_12565, *[entity_12566], **kwargs_12567)
    
    # Assigning a type to the variable 'stypy_return_type' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'stypy_return_type', deepcopy_call_result_12568)
    # SSA join for try-except statement (line 310)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'create_duplicate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_duplicate' in the type store
    # Getting the type of 'stypy_return_type' (line 304)
    stypy_return_type_12569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_12569)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_duplicate'
    return stypy_return_type_12569

# Assigning a type to the variable 'create_duplicate' (line 304)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 0), 'create_duplicate', create_duplicate)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
