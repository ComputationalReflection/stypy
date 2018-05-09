
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import inspect
2: import sys
3: import imp
4: import copy
5: import types
6: 
7: from stypy_copy import stypy_parameters_copy
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

# 'from stypy_copy import stypy_parameters_copy' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy', sys_modules_2.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_2, sys_modules_2.module_type_store, module_type_store)
    else:
        from stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'stypy_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

str_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, (-1)), 'str', '\nFile that contains helper functions to implement the type_inference_proxy.py functionality, grouped here to improve\nreadability of the code.\n')

# Assigning a Name to a Name (line 13):
# Getting the type of 'None' (line 13)
None_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 23), 'None')
# Assigning a type to the variable 'user_defined_modules' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'user_defined_modules', None_4)

# Assigning a Num to a Name (line 14):
int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'int')
# Assigning a type to the variable 'last_module_len' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'last_module_len', int_5)

@norecursion
def __init_user_defined_modules(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'stypy_parameters_copy' (line 16)
    stypy_parameters_copy_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 65), 'stypy_parameters_copy')
    # Obtaining the member 'PYTHON_EXE_PATH' of a type (line 16)
    PYTHON_EXE_PATH_7 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 65), stypy_parameters_copy_6, 'PYTHON_EXE_PATH')
    defaults = [PYTHON_EXE_PATH_7]
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

    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', '\n    Initializes the user_defined_modules variable\n    ')
    # Marking variables as global (line 20)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 20, 4), 'user_defined_modules')
    # Marking variables as global (line 21)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 21, 4), 'last_module_len')
    
    # Assigning a Call to a Name (line 25):
    
    # Call to replace(...): (line 25)
    # Processing the call arguments (line 25)
    str_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 63), 'str', '/')
    str_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 68), 'str', '\\')
    # Processing the call keyword arguments (line 25)
    kwargs_13 = {}
    # Getting the type of 'default_python_installation_path' (line 25)
    default_python_installation_path_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), 'default_python_installation_path', False)
    # Obtaining the member 'replace' of a type (line 25)
    replace_10 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 22), default_python_installation_path_9, 'replace')
    # Calling replace(args, kwargs) (line 25)
    replace_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 25, 22), replace_10, *[str_11, str_12], **kwargs_13)
    
    # Assigning a type to the variable 'normalized_path' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'normalized_path', replace_call_result_14)
    
    # Assigning a Call to a Name (line 26):
    
    # Call to items(...): (line 26)
    # Processing the call keyword arguments (line 26)
    kwargs_18 = {}
    # Getting the type of 'sys' (line 26)
    sys_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'sys', False)
    # Obtaining the member 'modules' of a type (line 26)
    modules_16 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 14), sys_15, 'modules')
    # Obtaining the member 'items' of a type (line 26)
    items_17 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 14), modules_16, 'items')
    # Calling items(args, kwargs) (line 26)
    items_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 26, 14), items_17, *[], **kwargs_18)
    
    # Assigning a type to the variable 'modules' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'modules', items_call_result_19)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'user_defined_modules' (line 29)
    user_defined_modules_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 7), 'user_defined_modules')
    # Getting the type of 'None' (line 29)
    None_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 31), 'None')
    # Applying the binary operator 'is' (line 29)
    result_is__22 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 7), 'is', user_defined_modules_20, None_21)
    
    
    
    # Call to len(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'modules' (line 29)
    modules_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 43), 'modules', False)
    # Processing the call keyword arguments (line 29)
    kwargs_25 = {}
    # Getting the type of 'len' (line 29)
    len_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 39), 'len', False)
    # Calling len(args, kwargs) (line 29)
    len_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 29, 39), len_23, *[modules_24], **kwargs_25)
    
    # Getting the type of 'last_module_len' (line 29)
    last_module_len_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 55), 'last_module_len')
    # Applying the binary operator '!=' (line 29)
    result_ne_28 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 39), '!=', len_call_result_26, last_module_len_27)
    
    # Applying the binary operator 'or' (line 29)
    result_or_keyword_29 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 7), 'or', result_is__22, result_ne_28)
    
    # Testing if the type of an if condition is none (line 29)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 29, 4), result_or_keyword_29):
        pass
    else:
        
        # Testing the type of an if condition (line 29)
        if_condition_30 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 4), result_or_keyword_29)
        # Assigning a type to the variable 'if_condition_30' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'if_condition_30', if_condition_30)
        # SSA begins for if statement (line 29)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 30):
        
        # Call to dict(...): (line 30)
        # Processing the call arguments (line 30)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 30, 36, True)
        # Calculating comprehension expression
        # Getting the type of 'modules' (line 30)
        modules_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 97), 'modules', False)
        comprehension_53 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 36), modules_52)
        # Assigning a type to the variable 'module_name' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 36), 'module_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 36), comprehension_53))
        # Assigning a type to the variable 'module_desc' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 36), 'module_desc', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 36), comprehension_53))
        
        # Evaluating a boolean operation
        
        # Getting the type of 'normalized_path' (line 31)
        normalized_path_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 36), 'normalized_path', False)
        
        # Call to str(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'module_desc' (line 31)
        module_desc_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 63), 'module_desc', False)
        # Processing the call keyword arguments (line 31)
        kwargs_38 = {}
        # Getting the type of 'str' (line 31)
        str_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 59), 'str', False)
        # Calling str(args, kwargs) (line 31)
        str_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 31, 59), str_36, *[module_desc_37], **kwargs_38)
        
        # Applying the binary operator 'notin' (line 31)
        result_contains_40 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 36), 'notin', normalized_path_35, str_call_result_39)
        
        
        str_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 80), 'str', 'built-in')
        
        # Call to str(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'module_desc' (line 32)
        module_desc_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 40), 'module_desc', False)
        # Processing the call keyword arguments (line 32)
        kwargs_44 = {}
        # Getting the type of 'str' (line 32)
        str_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 36), 'str', False)
        # Calling str(args, kwargs) (line 32)
        str_call_result_45 = invoke(stypy.reporting.localization.Localization(__file__, 32, 36), str_42, *[module_desc_43], **kwargs_44)
        
        # Applying the binary operator 'notin' (line 31)
        result_contains_46 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 80), 'notin', str_41, str_call_result_45)
        
        # Applying the binary operator 'and' (line 31)
        result_and_keyword_47 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 36), 'and', result_contains_40, result_contains_46)
        
        # Getting the type of 'module_desc' (line 33)
        module_desc_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 40), 'module_desc', False)
        # Getting the type of 'None' (line 33)
        None_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 59), 'None', False)
        # Applying the binary operator 'isnot' (line 33)
        result_is_not_50 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 40), 'isnot', module_desc_48, None_49)
        
        # Applying the binary operator 'and' (line 31)
        result_and_keyword_51 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 36), 'and', result_and_keyword_47, result_is_not_50)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 30)
        tuple_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 30)
        # Adding element type (line 30)
        # Getting the type of 'module_name' (line 30)
        module_name_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 37), 'module_name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 37), tuple_32, module_name_33)
        # Adding element type (line 30)
        # Getting the type of 'module_desc' (line 30)
        module_desc_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 50), 'module_desc', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 37), tuple_32, module_desc_34)
        
        list_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 36), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 36), list_54, tuple_32)
        # Processing the call keyword arguments (line 30)
        kwargs_55 = {}
        # Getting the type of 'dict' (line 30)
        dict_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 31), 'dict', False)
        # Calling dict(args, kwargs) (line 30)
        dict_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 30, 31), dict_31, *[list_54], **kwargs_55)
        
        # Assigning a type to the variable 'user_defined_modules' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'user_defined_modules', dict_call_result_56)
        
        # Assigning a Call to a Name (line 34):
        
        # Call to len(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'modules' (line 34)
        modules_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'modules', False)
        # Processing the call keyword arguments (line 34)
        kwargs_59 = {}
        # Getting the type of 'len' (line 34)
        len_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 26), 'len', False)
        # Calling len(args, kwargs) (line 34)
        len_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 34, 26), len_57, *[modules_58], **kwargs_59)
        
        # Assigning a type to the variable 'last_module_len' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'last_module_len', len_call_result_60)
        # SSA join for if statement (line 29)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of '__init_user_defined_modules(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__init_user_defined_modules' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_61)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__init_user_defined_modules'
    return stypy_return_type_61

# Assigning a type to the variable '__init_user_defined_modules' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '__init_user_defined_modules', __init_user_defined_modules)

@norecursion
def is_user_defined_module(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'stypy_parameters_copy' (line 37)
    stypy_parameters_copy_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 73), 'stypy_parameters_copy')
    # Obtaining the member 'PYTHON_EXE_PATH' of a type (line 37)
    PYTHON_EXE_PATH_63 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 73), stypy_parameters_copy_62, 'PYTHON_EXE_PATH')
    defaults = [PYTHON_EXE_PATH_63]
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

    str_64 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, (-1)), 'str', '\n    Determines if the passed module_name is a user created module or a Python library one.\n    :param module_name: Name of the module\n    :param default_python_installation_path: Python executable to use. Can be left blank as it is initialized by default\n     with the PYTHON_EXE_PATH parameter\n    :return: bool\n    ')
    # Marking variables as global (line 45)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 45, 4), 'user_defined_modules')
    
    # Call to __init_user_defined_modules(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'default_python_installation_path' (line 47)
    default_python_installation_path_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 32), 'default_python_installation_path', False)
    # Processing the call keyword arguments (line 47)
    kwargs_67 = {}
    # Getting the type of '__init_user_defined_modules' (line 47)
    init_user_defined_modules_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), '__init_user_defined_modules', False)
    # Calling __init_user_defined_modules(args, kwargs) (line 47)
    init_user_defined_modules_call_result_68 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), init_user_defined_modules_65, *[default_python_installation_path_66], **kwargs_67)
    
    
    # Getting the type of 'module_name' (line 49)
    module_name_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'module_name')
    # Getting the type of 'user_defined_modules' (line 49)
    user_defined_modules_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'user_defined_modules')
    # Applying the binary operator 'in' (line 49)
    result_contains_71 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 11), 'in', module_name_69, user_defined_modules_70)
    
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type', result_contains_71)
    
    # ################# End of 'is_user_defined_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_user_defined_module' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_72)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_user_defined_module'
    return stypy_return_type_72

# Assigning a type to the variable 'is_user_defined_module' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'is_user_defined_module', is_user_defined_module)

@norecursion
def is_user_defined_class(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'stypy_parameters_copy' (line 52)
    stypy_parameters_copy_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 64), 'stypy_parameters_copy')
    # Obtaining the member 'PYTHON_EXE_PATH' of a type (line 52)
    PYTHON_EXE_PATH_74 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 64), stypy_parameters_copy_73, 'PYTHON_EXE_PATH')
    defaults = [PYTHON_EXE_PATH_74]
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

    str_75 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', '\n    Determines if the passed class is a user created class or a Python library one.\n    :param cls: Class\n    :param default_python_installation_path: Python executable to use. Can be left blank as it is initialized by default\n     with the PYTHON_EXE_PATH parameter\n    :return:\n    ')
    # Marking variables as global (line 60)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 60, 4), 'user_defined_modules')
    
    
    # Call to isclass(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'cls' (line 62)
    cls_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 27), 'cls', False)
    # Processing the call keyword arguments (line 62)
    kwargs_79 = {}
    # Getting the type of 'inspect' (line 62)
    inspect_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'inspect', False)
    # Obtaining the member 'isclass' of a type (line 62)
    isclass_77 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 11), inspect_76, 'isclass')
    # Calling isclass(args, kwargs) (line 62)
    isclass_call_result_80 = invoke(stypy.reporting.localization.Localization(__file__, 62, 11), isclass_77, *[cls_78], **kwargs_79)
    
    # Applying the 'not' unary operator (line 62)
    result_not__81 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 7), 'not', isclass_call_result_80)
    
    # Testing if the type of an if condition is none (line 62)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 62, 4), result_not__81):
        pass
    else:
        
        # Testing the type of an if condition (line 62)
        if_condition_82 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 4), result_not__81)
        # Assigning a type to the variable 'if_condition_82' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'if_condition_82', if_condition_82)
        # SSA begins for if statement (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 63)
        False_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'stypy_return_type', False_83)
        # SSA join for if statement (line 62)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to __init_user_defined_modules(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'default_python_installation_path' (line 65)
    default_python_installation_path_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 32), 'default_python_installation_path', False)
    # Processing the call keyword arguments (line 65)
    kwargs_86 = {}
    # Getting the type of '__init_user_defined_modules' (line 65)
    init_user_defined_modules_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), '__init_user_defined_modules', False)
    # Calling __init_user_defined_modules(args, kwargs) (line 65)
    init_user_defined_modules_call_result_87 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), init_user_defined_modules_84, *[default_python_installation_path_85], **kwargs_86)
    
    
    # Call to is_user_defined_module(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'cls' (line 68)
    cls_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 34), 'cls', False)
    # Obtaining the member '__module__' of a type (line 68)
    module___90 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 34), cls_89, '__module__')
    # Getting the type of 'default_python_installation_path' (line 68)
    default_python_installation_path_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 50), 'default_python_installation_path', False)
    # Processing the call keyword arguments (line 68)
    kwargs_92 = {}
    # Getting the type of 'is_user_defined_module' (line 68)
    is_user_defined_module_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'is_user_defined_module', False)
    # Calling is_user_defined_module(args, kwargs) (line 68)
    is_user_defined_module_call_result_93 = invoke(stypy.reporting.localization.Localization(__file__, 68, 11), is_user_defined_module_88, *[module___90, default_python_installation_path_91], **kwargs_92)
    
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type', is_user_defined_module_call_result_93)
    
    # ################# End of 'is_user_defined_class(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_user_defined_class' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_94)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_user_defined_class'
    return stypy_return_type_94

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

    str_95 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, (-1)), 'str', '\n    Determines if an object supports structural reflection. An object supports it if it has a __dict__ property and its\n    type is dict (instead of the read-only dictproxy)\n\n    :param obj: Any Python object\n    :return: bool\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 79)
    str_96 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 24), 'str', '__dict__')
    # Getting the type of 'obj' (line 79)
    obj_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'obj')
    
    (may_be_98, more_types_in_union_99) = may_not_provide_member(str_96, obj_97)

    if may_be_98:

        if more_types_in_union_99:
            # Runtime conditional SSA (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'obj' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'obj', remove_member_provider_from_union(obj_97, '__dict__'))
        # Getting the type of 'False' (line 80)
        False_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'stypy_return_type', False_100)

        if more_types_in_union_99:
            # SSA join for if statement (line 79)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to type(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'obj' (line 82)
    obj_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'obj', False)
    # Obtaining the member '__dict__' of a type (line 82)
    dict___103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), obj_102, '__dict__')
    # Processing the call keyword arguments (line 82)
    kwargs_104 = {}
    # Getting the type of 'type' (line 82)
    type_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'type', False)
    # Calling type(args, kwargs) (line 82)
    type_call_result_105 = invoke(stypy.reporting.localization.Localization(__file__, 82, 4), type_101, *[dict___103], **kwargs_104)
    
    # Getting the type of 'dict' (line 82)
    dict_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'dict')
    # Applying the binary operator 'is' (line 82)
    result_is__107 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 4), 'is', type_call_result_105, dict_106)
    
    # Testing if the type of an if condition is none (line 82)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 82, 4), result_is__107):
        
        
        # SSA begins for try-except statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Name to a Subscript (line 86):
        # Getting the type of 'None' (line 86)
        None_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 44), 'None')
        # Getting the type of 'obj' (line 86)
        obj_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'obj')
        # Obtaining the member '__dict__' of a type (line 86)
        dict___112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), obj_111, '__dict__')
        str_113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'str', '__stypy_probe')
        # Storing an element on a container (line 86)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 12), dict___112, (str_113, None_110))
        # Deleting a member
        # Getting the type of 'obj' (line 87)
        obj_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'obj')
        # Obtaining the member '__dict__' of a type (line 87)
        dict___115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), obj_114, '__dict__')
        
        # Obtaining the type of the subscript
        str_116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 29), 'str', '__stypy_probe')
        # Getting the type of 'obj' (line 87)
        obj_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'obj')
        # Obtaining the member '__dict__' of a type (line 87)
        dict___118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), obj_117, '__dict__')
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), dict___118, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_120 = invoke(stypy.reporting.localization.Localization(__file__, 87, 16), getitem___119, str_116)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 12), dict___115, subscript_call_result_120)
        # Getting the type of 'True' (line 88)
        True_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'stypy_return_type', True_121)
        # SSA branch for the except part of a try statement (line 85)
        # SSA branch for the except '<any exception>' branch of a try statement (line 85)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'False' (line 90)
        False_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'stypy_return_type', False_122)
        # SSA join for try-except statement (line 85)
        module_type_store = module_type_store.join_ssa_context()
        
    else:
        
        # Testing the type of an if condition (line 82)
        if_condition_108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 4), result_is__107)
        # Assigning a type to the variable 'if_condition_108' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'if_condition_108', if_condition_108)
        # SSA begins for if statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 83)
        True_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'stypy_return_type', True_109)
        # SSA branch for the else part of an if statement (line 82)
        module_type_store.open_ssa_branch('else')
        
        
        # SSA begins for try-except statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Name to a Subscript (line 86):
        # Getting the type of 'None' (line 86)
        None_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 44), 'None')
        # Getting the type of 'obj' (line 86)
        obj_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'obj')
        # Obtaining the member '__dict__' of a type (line 86)
        dict___112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), obj_111, '__dict__')
        str_113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 25), 'str', '__stypy_probe')
        # Storing an element on a container (line 86)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 12), dict___112, (str_113, None_110))
        # Deleting a member
        # Getting the type of 'obj' (line 87)
        obj_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'obj')
        # Obtaining the member '__dict__' of a type (line 87)
        dict___115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), obj_114, '__dict__')
        
        # Obtaining the type of the subscript
        str_116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 29), 'str', '__stypy_probe')
        # Getting the type of 'obj' (line 87)
        obj_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'obj')
        # Obtaining the member '__dict__' of a type (line 87)
        dict___118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), obj_117, '__dict__')
        # Obtaining the member '__getitem__' of a type (line 87)
        getitem___119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 16), dict___118, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 87)
        subscript_call_result_120 = invoke(stypy.reporting.localization.Localization(__file__, 87, 16), getitem___119, str_116)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 12), dict___115, subscript_call_result_120)
        # Getting the type of 'True' (line 88)
        True_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'stypy_return_type', True_121)
        # SSA branch for the except part of a try statement (line 85)
        # SSA branch for the except '<any exception>' branch of a try statement (line 85)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'False' (line 90)
        False_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'stypy_return_type', False_122)
        # SSA join for try-except statement (line 85)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'supports_structural_reflection(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'supports_structural_reflection' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_123)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'supports_structural_reflection'
    return stypy_return_type_123

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

    str_124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, (-1)), 'str', '\n    Shortcut to inspect.isclass\n    :param cls: Any Python object\n    :return:\n    ')
    
    # Call to isclass(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'cls' (line 99)
    cls_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 27), 'cls', False)
    # Processing the call keyword arguments (line 99)
    kwargs_128 = {}
    # Getting the type of 'inspect' (line 99)
    inspect_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'inspect', False)
    # Obtaining the member 'isclass' of a type (line 99)
    isclass_126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 11), inspect_125, 'isclass')
    # Calling isclass(args, kwargs) (line 99)
    isclass_call_result_129 = invoke(stypy.reporting.localization.Localization(__file__, 99, 11), isclass_126, *[cls_127], **kwargs_128)
    
    # Assigning a type to the variable 'stypy_return_type' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type', isclass_call_result_129)
    
    # ################# End of 'is_class(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_class' in the type store
    # Getting the type of 'stypy_return_type' (line 93)
    stypy_return_type_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_130)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_class'
    return stypy_return_type_130

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

    str_131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, (-1)), 'str', '\n    Python supports two type of classes: old-style classes (those that do not inherit from object) and new-style classes\n    (those that do inherit from object). The best way to distinguish between them is to check if the class has an\n     __mro__ (method resolution order) property (only available to new-style classes). Distinguishing between both types\n     is important specially when dealing with type change or supertype change operations, as new-style classes are\n     more limited in that sense and both types cannot be mixed in one of these operations.\n    :param cls: Class to test\n    :return: bool\n    ')
    
    
    # Call to is_class(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'cls' (line 112)
    cls_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'cls', False)
    # Processing the call keyword arguments (line 112)
    kwargs_134 = {}
    # Getting the type of 'is_class' (line 112)
    is_class_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'is_class', False)
    # Calling is_class(args, kwargs) (line 112)
    is_class_call_result_135 = invoke(stypy.reporting.localization.Localization(__file__, 112, 11), is_class_132, *[cls_133], **kwargs_134)
    
    # Applying the 'not' unary operator (line 112)
    result_not__136 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 7), 'not', is_class_call_result_135)
    
    # Testing if the type of an if condition is none (line 112)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 112, 4), result_not__136):
        pass
    else:
        
        # Testing the type of an if condition (line 112)
        if_condition_137 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 4), result_not__136)
        # Assigning a type to the variable 'if_condition_137' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'if_condition_137', if_condition_137)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 113)
        False_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'stypy_return_type', False_138)
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to hasattr(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'cls' (line 114)
    cls_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'cls', False)
    str_141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 28), 'str', '__mro__')
    # Processing the call keyword arguments (line 114)
    kwargs_142 = {}
    # Getting the type of 'hasattr' (line 114)
    hasattr_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 114)
    hasattr_call_result_143 = invoke(stypy.reporting.localization.Localization(__file__, 114, 15), hasattr_139, *[cls_140, str_141], **kwargs_142)
    
    # Applying the 'not' unary operator (line 114)
    result_not__144 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 11), 'not', hasattr_call_result_143)
    
    # Assigning a type to the variable 'stypy_return_type' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type', result_not__144)
    
    # ################# End of 'is_old_style_class(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_old_style_class' in the type store
    # Getting the type of 'stypy_return_type' (line 102)
    stypy_return_type_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_145)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_old_style_class'
    return stypy_return_type_145

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

    str_146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, (-1)), 'str', '\n    This method is a shortcut to the opposite of the previous one\n    :param cls: Class to test\n    :return: bool\n    ')
    
    
    # Call to is_old_style_class(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 'cls' (line 123)
    cls_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 34), 'cls', False)
    # Processing the call keyword arguments (line 123)
    kwargs_149 = {}
    # Getting the type of 'is_old_style_class' (line 123)
    is_old_style_class_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'is_old_style_class', False)
    # Calling is_old_style_class(args, kwargs) (line 123)
    is_old_style_class_call_result_150 = invoke(stypy.reporting.localization.Localization(__file__, 123, 15), is_old_style_class_147, *[cls_148], **kwargs_149)
    
    # Applying the 'not' unary operator (line 123)
    result_not__151 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 11), 'not', is_old_style_class_call_result_150)
    
    # Assigning a type to the variable 'stypy_return_type' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type', result_not__151)
    
    # ################# End of 'is_new_style_class(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_new_style_class' in the type store
    # Getting the type of 'stypy_return_type' (line 117)
    stypy_return_type_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_152)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_new_style_class'
    return stypy_return_type_152

# Assigning a type to the variable 'is_new_style_class' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'is_new_style_class', is_new_style_class)
str_153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, (-1)), 'str', '\nCloning Python types is a key part of the implementation of the SSA algorithm. However, this is a very difficult task\nbecause some types are not meant to be easily cloned. We managed to develop ways to clone any type that can be\npresent in a stypy type store with the following functions, ensuring a proper SSA implementation.\n')

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

    str_154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, (-1)), 'str', '\n    Clone an existing function\n    :param f: Function to clone\n    :return: An independent copy of the function\n    ')
    
    # Call to FunctionType(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'f' (line 158)
    f_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 30), 'f', False)
    # Obtaining the member 'func_code' of a type (line 158)
    func_code_158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 30), f_157, 'func_code')
    # Getting the type of 'f' (line 158)
    f_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 43), 'f', False)
    # Obtaining the member 'func_globals' of a type (line 158)
    func_globals_160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 43), f_159, 'func_globals')
    # Processing the call keyword arguments (line 158)
    # Getting the type of 'f' (line 158)
    f_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 64), 'f', False)
    # Obtaining the member 'func_name' of a type (line 158)
    func_name_162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 64), f_161, 'func_name')
    keyword_163 = func_name_162
    # Getting the type of 'f' (line 159)
    f_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 38), 'f', False)
    # Obtaining the member 'func_defaults' of a type (line 159)
    func_defaults_165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 38), f_164, 'func_defaults')
    keyword_166 = func_defaults_165
    # Getting the type of 'f' (line 160)
    f_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 38), 'f', False)
    # Obtaining the member 'func_closure' of a type (line 160)
    func_closure_168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 38), f_167, 'func_closure')
    keyword_169 = func_closure_168
    kwargs_170 = {'closure': keyword_169, 'name': keyword_163, 'argdefs': keyword_166}
    # Getting the type of 'types' (line 158)
    types_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'types', False)
    # Obtaining the member 'FunctionType' of a type (line 158)
    FunctionType_156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 11), types_155, 'FunctionType')
    # Calling FunctionType(args, kwargs) (line 158)
    FunctionType_call_result_171 = invoke(stypy.reporting.localization.Localization(__file__, 158, 11), FunctionType_156, *[func_code_158, func_globals_160], **kwargs_170)
    
    # Assigning a type to the variable 'stypy_return_type' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'stypy_return_type', FunctionType_call_result_171)
    
    # ################# End of '__duplicate_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__duplicate_function' in the type store
    # Getting the type of 'stypy_return_type' (line 152)
    stypy_return_type_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_172)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__duplicate_function'
    return stypy_return_type_172

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

    str_173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, (-1)), 'str', '\n    Clone a class object, creating a duplicate of all its members\n    :param clazz: Original class\n    :return: A clone of the class (same name, same members, same inheritance relationship, different identity\n    ')
    
    # Call to is_new_style_class(...): (line 170)
    # Processing the call arguments (line 170)
    # Getting the type of 'clazz' (line 170)
    clazz_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 26), 'clazz', False)
    # Processing the call keyword arguments (line 170)
    kwargs_176 = {}
    # Getting the type of 'is_new_style_class' (line 170)
    is_new_style_class_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 7), 'is_new_style_class', False)
    # Calling is_new_style_class(args, kwargs) (line 170)
    is_new_style_class_call_result_177 = invoke(stypy.reporting.localization.Localization(__file__, 170, 7), is_new_style_class_174, *[clazz_175], **kwargs_176)
    
    # Testing if the type of an if condition is none (line 170)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 170, 4), is_new_style_class_call_result_177):
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
        clazz_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 30), 'clazz')
        # Obtaining the member '__name__' of a type (line 178)
        name___192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 30), clazz_191, '__name__')
        # Getting the type of 'DummyClass' (line 178)
        DummyClass_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'DummyClass')
        # Setting the type of the member '__name__' of a type (line 178)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), DummyClass_193, '__name__', name___192)
        
        # Assigning a Attribute to a Attribute (line 179):
        # Getting the type of 'clazz' (line 179)
        clazz_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 31), 'clazz')
        # Obtaining the member '__bases__' of a type (line 179)
        bases___195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 31), clazz_194, '__bases__')
        # Getting the type of 'DummyClass' (line 179)
        DummyClass_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'DummyClass')
        # Setting the type of the member '__bases__' of a type (line 179)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), DummyClass_196, '__bases__', bases___195)
        
        # Assigning a Call to a Attribute (line 181):
        
        # Call to dict(...): (line 181)
        # Processing the call keyword arguments (line 181)
        kwargs_198 = {}
        # Getting the type of 'dict' (line 181)
        dict_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 30), 'dict', False)
        # Calling dict(args, kwargs) (line 181)
        dict_call_result_199 = invoke(stypy.reporting.localization.Localization(__file__, 181, 30), dict_197, *[], **kwargs_198)
        
        # Getting the type of 'DummyClass' (line 181)
        DummyClass_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'DummyClass')
        # Setting the type of the member '__dict__' of a type (line 181)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), DummyClass_200, '__dict__', dict_call_result_199)
        
        # Getting the type of 'clazz' (line 182)
        clazz_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'clazz')
        # Obtaining the member '__dict__' of a type (line 182)
        dict___202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 22), clazz_201, '__dict__')
        # Assigning a type to the variable 'dict___202' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'dict___202', dict___202)
        # Testing if the for loop is going to be iterated (line 182)
        # Testing the type of a for loop iterable (line 182)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 182, 8), dict___202)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 182, 8), dict___202):
            # Getting the type of the for loop variable (line 182)
            for_loop_var_203 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 182, 8), dict___202)
            # Assigning a type to the variable 'member' (line 182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'member', for_loop_var_203)
            # SSA begins for a for statement (line 182)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Subscript (line 183):
            
            # Obtaining the type of the subscript
            # Getting the type of 'member' (line 183)
            member_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 57), 'member')
            # Getting the type of 'clazz' (line 183)
            clazz_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 42), 'clazz')
            # Obtaining the member '__dict__' of a type (line 183)
            dict___206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 42), clazz_205, '__dict__')
            # Obtaining the member '__getitem__' of a type (line 183)
            getitem___207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 42), dict___206, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 183)
            subscript_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 183, 42), getitem___207, member_204)
            
            # Getting the type of 'DummyClass' (line 183)
            DummyClass_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'DummyClass')
            # Obtaining the member '__dict__' of a type (line 183)
            dict___210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), DummyClass_209, '__dict__')
            # Getting the type of 'member' (line 183)
            member_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 32), 'member')
            # Storing an element on a container (line 183)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 12), dict___210, (member_211, subscript_call_result_208))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'DummyClass' (line 185)
        DummyClass_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 'DummyClass')
        # Assigning a type to the variable 'stypy_return_type' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'stypy_return_type', DummyClass_212)
    else:
        
        # Testing the type of an if condition (line 170)
        if_condition_178 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 4), is_new_style_class_call_result_177)
        # Assigning a type to the variable 'if_condition_178' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'if_condition_178', if_condition_178)
        # SSA begins for if statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to type(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'clazz' (line 171)
        clazz_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'clazz', False)
        # Obtaining the member '__name__' of a type (line 171)
        name___181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 20), clazz_180, '__name__')
        # Getting the type of 'clazz' (line 171)
        clazz_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 36), 'clazz', False)
        # Obtaining the member '__bases__' of a type (line 171)
        bases___183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 36), clazz_182, '__bases__')
        
        # Call to dict(...): (line 171)
        # Processing the call arguments (line 171)
        # Getting the type of 'clazz' (line 171)
        clazz_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 58), 'clazz', False)
        # Obtaining the member '__dict__' of a type (line 171)
        dict___186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 58), clazz_185, '__dict__')
        # Processing the call keyword arguments (line 171)
        kwargs_187 = {}
        # Getting the type of 'dict' (line 171)
        dict_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 53), 'dict', False)
        # Calling dict(args, kwargs) (line 171)
        dict_call_result_188 = invoke(stypy.reporting.localization.Localization(__file__, 171, 53), dict_184, *[dict___186], **kwargs_187)
        
        # Processing the call keyword arguments (line 171)
        kwargs_189 = {}
        # Getting the type of 'type' (line 171)
        type_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'type', False)
        # Calling type(args, kwargs) (line 171)
        type_call_result_190 = invoke(stypy.reporting.localization.Localization(__file__, 171, 15), type_179, *[name___181, bases___183, dict_call_result_188], **kwargs_189)
        
        # Assigning a type to the variable 'stypy_return_type' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'stypy_return_type', type_call_result_190)
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
        clazz_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 30), 'clazz')
        # Obtaining the member '__name__' of a type (line 178)
        name___192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 30), clazz_191, '__name__')
        # Getting the type of 'DummyClass' (line 178)
        DummyClass_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'DummyClass')
        # Setting the type of the member '__name__' of a type (line 178)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), DummyClass_193, '__name__', name___192)
        
        # Assigning a Attribute to a Attribute (line 179):
        # Getting the type of 'clazz' (line 179)
        clazz_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 31), 'clazz')
        # Obtaining the member '__bases__' of a type (line 179)
        bases___195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 31), clazz_194, '__bases__')
        # Getting the type of 'DummyClass' (line 179)
        DummyClass_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'DummyClass')
        # Setting the type of the member '__bases__' of a type (line 179)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), DummyClass_196, '__bases__', bases___195)
        
        # Assigning a Call to a Attribute (line 181):
        
        # Call to dict(...): (line 181)
        # Processing the call keyword arguments (line 181)
        kwargs_198 = {}
        # Getting the type of 'dict' (line 181)
        dict_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 30), 'dict', False)
        # Calling dict(args, kwargs) (line 181)
        dict_call_result_199 = invoke(stypy.reporting.localization.Localization(__file__, 181, 30), dict_197, *[], **kwargs_198)
        
        # Getting the type of 'DummyClass' (line 181)
        DummyClass_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'DummyClass')
        # Setting the type of the member '__dict__' of a type (line 181)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), DummyClass_200, '__dict__', dict_call_result_199)
        
        # Getting the type of 'clazz' (line 182)
        clazz_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'clazz')
        # Obtaining the member '__dict__' of a type (line 182)
        dict___202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 22), clazz_201, '__dict__')
        # Assigning a type to the variable 'dict___202' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'dict___202', dict___202)
        # Testing if the for loop is going to be iterated (line 182)
        # Testing the type of a for loop iterable (line 182)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 182, 8), dict___202)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 182, 8), dict___202):
            # Getting the type of the for loop variable (line 182)
            for_loop_var_203 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 182, 8), dict___202)
            # Assigning a type to the variable 'member' (line 182)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'member', for_loop_var_203)
            # SSA begins for a for statement (line 182)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Subscript (line 183):
            
            # Obtaining the type of the subscript
            # Getting the type of 'member' (line 183)
            member_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 57), 'member')
            # Getting the type of 'clazz' (line 183)
            clazz_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 42), 'clazz')
            # Obtaining the member '__dict__' of a type (line 183)
            dict___206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 42), clazz_205, '__dict__')
            # Obtaining the member '__getitem__' of a type (line 183)
            getitem___207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 42), dict___206, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 183)
            subscript_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 183, 42), getitem___207, member_204)
            
            # Getting the type of 'DummyClass' (line 183)
            DummyClass_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'DummyClass')
            # Obtaining the member '__dict__' of a type (line 183)
            dict___210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), DummyClass_209, '__dict__')
            # Getting the type of 'member' (line 183)
            member_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 32), 'member')
            # Storing an element on a container (line 183)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 12), dict___210, (member_211, subscript_call_result_208))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'DummyClass' (line 185)
        DummyClass_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 'DummyClass')
        # Assigning a type to the variable 'stypy_return_type' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'stypy_return_type', DummyClass_212)
        # SSA join for if statement (line 170)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of '__duplicate_class(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__duplicate_class' in the type store
    # Getting the type of 'stypy_return_type' (line 163)
    stypy_return_type_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_213)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__duplicate_class'
    return stypy_return_type_213

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

    str_214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, (-1)), 'str', '\n    Create a deep copy of the passed type inference proxy, cloning all its members as best as possible to ensure that\n    deep copies are used whenever possible\n    :param type_inference_proxy_obj: Original type inference proxy\n    :return: Clone of the passed object\n    ')
    
    
    # SSA begins for try-except statement (line 197)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 199):
    
    # Call to deepcopy(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'type_inference_proxy_obj' (line 199)
    type_inference_proxy_obj_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 31), 'type_inference_proxy_obj', False)
    # Processing the call keyword arguments (line 199)
    kwargs_218 = {}
    # Getting the type of 'copy' (line 199)
    copy_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 17), 'copy', False)
    # Obtaining the member 'deepcopy' of a type (line 199)
    deepcopy_216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 17), copy_215, 'deepcopy')
    # Calling deepcopy(args, kwargs) (line 199)
    deepcopy_call_result_219 = invoke(stypy.reporting.localization.Localization(__file__, 199, 17), deepcopy_216, *[type_inference_proxy_obj_217], **kwargs_218)
    
    # Assigning a type to the variable 'result' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'result', deepcopy_call_result_219)
    # SSA branch for the except part of a try statement (line 197)
    # SSA branch for the except '<any exception>' branch of a try statement (line 197)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 202):
    
    # Call to copy(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'type_inference_proxy_obj' (line 202)
    type_inference_proxy_obj_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 27), 'type_inference_proxy_obj', False)
    # Processing the call keyword arguments (line 202)
    kwargs_223 = {}
    # Getting the type of 'copy' (line 202)
    copy_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 17), 'copy', False)
    # Obtaining the member 'copy' of a type (line 202)
    copy_221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 17), copy_220, 'copy')
    # Calling copy(args, kwargs) (line 202)
    copy_call_result_224 = invoke(stypy.reporting.localization.Localization(__file__, 202, 17), copy_221, *[type_inference_proxy_obj_222], **kwargs_223)
    
    # Assigning a type to the variable 'result' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'result', copy_call_result_224)
    # SSA join for try-except statement (line 197)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    
    # Call to supports_structural_reflection(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'type_inference_proxy_obj' (line 207)
    type_inference_proxy_obj_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 46), 'type_inference_proxy_obj', False)
    # Obtaining the member 'python_entity' of a type (line 207)
    python_entity_227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 46), type_inference_proxy_obj_226, 'python_entity')
    # Processing the call keyword arguments (line 207)
    kwargs_228 = {}
    # Getting the type of 'supports_structural_reflection' (line 207)
    supports_structural_reflection_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'supports_structural_reflection', False)
    # Calling supports_structural_reflection(args, kwargs) (line 207)
    supports_structural_reflection_call_result_229 = invoke(stypy.reporting.localization.Localization(__file__, 207, 15), supports_structural_reflection_225, *[python_entity_227], **kwargs_228)
    
    # Applying the 'not' unary operator (line 207)
    result_not__230 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 11), 'not', supports_structural_reflection_call_result_229)
    
    # Testing if the type of an if condition is none (line 207)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 207, 8), result_not__230):
        
        # Call to isclass(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'type_inference_proxy_obj' (line 212)
        type_inference_proxy_obj_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 31), 'type_inference_proxy_obj', False)
        # Obtaining the member 'python_entity' of a type (line 212)
        python_entity_242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 31), type_inference_proxy_obj_241, 'python_entity')
        # Processing the call keyword arguments (line 212)
        kwargs_243 = {}
        # Getting the type of 'inspect' (line 212)
        inspect_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 212)
        isclass_240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 15), inspect_239, 'isclass')
        # Calling isclass(args, kwargs) (line 212)
        isclass_call_result_244 = invoke(stypy.reporting.localization.Localization(__file__, 212, 15), isclass_240, *[python_entity_242], **kwargs_243)
        
        # Testing if the type of an if condition is none (line 212)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 212, 12), isclass_call_result_244):
            
            # Call to isfunction(...): (line 220)
            # Processing the call arguments (line 220)
            # Getting the type of 'type_inference_proxy_obj' (line 220)
            type_inference_proxy_obj_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'type_inference_proxy_obj', False)
            # Obtaining the member 'python_entity' of a type (line 220)
            python_entity_263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 38), type_inference_proxy_obj_262, 'python_entity')
            # Processing the call keyword arguments (line 220)
            kwargs_264 = {}
            # Getting the type of 'inspect' (line 220)
            inspect_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'inspect', False)
            # Obtaining the member 'isfunction' of a type (line 220)
            isfunction_261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 19), inspect_260, 'isfunction')
            # Calling isfunction(args, kwargs) (line 220)
            isfunction_call_result_265 = invoke(stypy.reporting.localization.Localization(__file__, 220, 19), isfunction_261, *[python_entity_263], **kwargs_264)
            
            # Testing if the type of an if condition is none (line 220)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 220, 16), isfunction_call_result_265):
                
                # Assigning a Call to a Attribute (line 224):
                
                # Call to deepcopy(...): (line 224)
                # Processing the call arguments (line 224)
                # Getting the type of 'type_inference_proxy_obj' (line 224)
                type_inference_proxy_obj_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 224)
                python_entity_276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 57), type_inference_proxy_obj_275, 'python_entity')
                # Processing the call keyword arguments (line 224)
                kwargs_277 = {}
                # Getting the type of 'copy' (line 224)
                copy_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 224)
                deepcopy_274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), copy_273, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 224)
                deepcopy_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 224, 43), deepcopy_274, *[python_entity_276], **kwargs_277)
                
                # Getting the type of 'result' (line 224)
                result_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 224)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), result_279, 'python_entity', deepcopy_call_result_278)
            else:
                
                # Testing the type of an if condition (line 220)
                if_condition_266 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 16), isfunction_call_result_265)
                # Assigning a type to the variable 'if_condition_266' (line 220)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'if_condition_266', if_condition_266)
                # SSA begins for if statement (line 220)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Attribute (line 221):
                
                # Call to __duplicate_function(...): (line 221)
                # Processing the call arguments (line 221)
                # Getting the type of 'type_inference_proxy_obj' (line 221)
                type_inference_proxy_obj_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 64), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 221)
                python_entity_269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 64), type_inference_proxy_obj_268, 'python_entity')
                # Processing the call keyword arguments (line 221)
                kwargs_270 = {}
                # Getting the type of '__duplicate_function' (line 221)
                duplicate_function_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 43), '__duplicate_function', False)
                # Calling __duplicate_function(args, kwargs) (line 221)
                duplicate_function_call_result_271 = invoke(stypy.reporting.localization.Localization(__file__, 221, 43), duplicate_function_267, *[python_entity_269], **kwargs_270)
                
                # Getting the type of 'result' (line 221)
                result_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 221)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 20), result_272, 'python_entity', duplicate_function_call_result_271)
                # SSA branch for the else part of an if statement (line 220)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Attribute (line 224):
                
                # Call to deepcopy(...): (line 224)
                # Processing the call arguments (line 224)
                # Getting the type of 'type_inference_proxy_obj' (line 224)
                type_inference_proxy_obj_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 224)
                python_entity_276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 57), type_inference_proxy_obj_275, 'python_entity')
                # Processing the call keyword arguments (line 224)
                kwargs_277 = {}
                # Getting the type of 'copy' (line 224)
                copy_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 224)
                deepcopy_274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), copy_273, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 224)
                deepcopy_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 224, 43), deepcopy_274, *[python_entity_276], **kwargs_277)
                
                # Getting the type of 'result' (line 224)
                result_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 224)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), result_279, 'python_entity', deepcopy_call_result_278)
                # SSA join for if statement (line 220)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 212)
            if_condition_245 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 12), isclass_call_result_244)
            # Assigning a type to the variable 'if_condition_245' (line 212)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'if_condition_245', if_condition_245)
            # SSA begins for if statement (line 212)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Type idiom detected: calculating its left and rigth part (line 213)
            # Getting the type of 'type_inference_proxy_obj' (line 213)
            type_inference_proxy_obj_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'type_inference_proxy_obj')
            # Obtaining the member 'instance' of a type (line 213)
            instance_247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 19), type_inference_proxy_obj_246, 'instance')
            # Getting the type of 'None' (line 213)
            None_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 56), 'None')
            
            (may_be_249, more_types_in_union_250) = may_be_none(instance_247, None_248)

            if may_be_249:

                if more_types_in_union_250:
                    # Runtime conditional SSA (line 213)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Call to a Attribute (line 214):
                
                # Call to __duplicate_class(...): (line 214)
                # Processing the call arguments (line 214)
                # Getting the type of 'type_inference_proxy_obj' (line 214)
                type_inference_proxy_obj_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 61), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 214)
                python_entity_253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 61), type_inference_proxy_obj_252, 'python_entity')
                # Processing the call keyword arguments (line 214)
                kwargs_254 = {}
                # Getting the type of '__duplicate_class' (line 214)
                duplicate_class_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 43), '__duplicate_class', False)
                # Calling __duplicate_class(args, kwargs) (line 214)
                duplicate_class_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 214, 43), duplicate_class_251, *[python_entity_253], **kwargs_254)
                
                # Getting the type of 'result' (line 214)
                result_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 214)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 20), result_256, 'python_entity', duplicate_class_call_result_255)

                if more_types_in_union_250:
                    # Runtime conditional SSA for else branch (line 213)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_249) or more_types_in_union_250):
                
                # Assigning a Attribute to a Attribute (line 217):
                # Getting the type of 'type_inference_proxy_obj' (line 217)
                type_inference_proxy_obj_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 43), 'type_inference_proxy_obj')
                # Obtaining the member 'python_entity' of a type (line 217)
                python_entity_258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 43), type_inference_proxy_obj_257, 'python_entity')
                # Getting the type of 'result' (line 217)
                result_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 217)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 20), result_259, 'python_entity', python_entity_258)

                if (may_be_249 and more_types_in_union_250):
                    # SSA join for if statement (line 213)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA branch for the else part of an if statement (line 212)
            module_type_store.open_ssa_branch('else')
            
            # Call to isfunction(...): (line 220)
            # Processing the call arguments (line 220)
            # Getting the type of 'type_inference_proxy_obj' (line 220)
            type_inference_proxy_obj_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'type_inference_proxy_obj', False)
            # Obtaining the member 'python_entity' of a type (line 220)
            python_entity_263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 38), type_inference_proxy_obj_262, 'python_entity')
            # Processing the call keyword arguments (line 220)
            kwargs_264 = {}
            # Getting the type of 'inspect' (line 220)
            inspect_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'inspect', False)
            # Obtaining the member 'isfunction' of a type (line 220)
            isfunction_261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 19), inspect_260, 'isfunction')
            # Calling isfunction(args, kwargs) (line 220)
            isfunction_call_result_265 = invoke(stypy.reporting.localization.Localization(__file__, 220, 19), isfunction_261, *[python_entity_263], **kwargs_264)
            
            # Testing if the type of an if condition is none (line 220)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 220, 16), isfunction_call_result_265):
                
                # Assigning a Call to a Attribute (line 224):
                
                # Call to deepcopy(...): (line 224)
                # Processing the call arguments (line 224)
                # Getting the type of 'type_inference_proxy_obj' (line 224)
                type_inference_proxy_obj_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 224)
                python_entity_276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 57), type_inference_proxy_obj_275, 'python_entity')
                # Processing the call keyword arguments (line 224)
                kwargs_277 = {}
                # Getting the type of 'copy' (line 224)
                copy_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 224)
                deepcopy_274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), copy_273, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 224)
                deepcopy_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 224, 43), deepcopy_274, *[python_entity_276], **kwargs_277)
                
                # Getting the type of 'result' (line 224)
                result_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 224)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), result_279, 'python_entity', deepcopy_call_result_278)
            else:
                
                # Testing the type of an if condition (line 220)
                if_condition_266 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 16), isfunction_call_result_265)
                # Assigning a type to the variable 'if_condition_266' (line 220)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'if_condition_266', if_condition_266)
                # SSA begins for if statement (line 220)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Attribute (line 221):
                
                # Call to __duplicate_function(...): (line 221)
                # Processing the call arguments (line 221)
                # Getting the type of 'type_inference_proxy_obj' (line 221)
                type_inference_proxy_obj_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 64), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 221)
                python_entity_269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 64), type_inference_proxy_obj_268, 'python_entity')
                # Processing the call keyword arguments (line 221)
                kwargs_270 = {}
                # Getting the type of '__duplicate_function' (line 221)
                duplicate_function_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 43), '__duplicate_function', False)
                # Calling __duplicate_function(args, kwargs) (line 221)
                duplicate_function_call_result_271 = invoke(stypy.reporting.localization.Localization(__file__, 221, 43), duplicate_function_267, *[python_entity_269], **kwargs_270)
                
                # Getting the type of 'result' (line 221)
                result_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 221)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 20), result_272, 'python_entity', duplicate_function_call_result_271)
                # SSA branch for the else part of an if statement (line 220)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Attribute (line 224):
                
                # Call to deepcopy(...): (line 224)
                # Processing the call arguments (line 224)
                # Getting the type of 'type_inference_proxy_obj' (line 224)
                type_inference_proxy_obj_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 224)
                python_entity_276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 57), type_inference_proxy_obj_275, 'python_entity')
                # Processing the call keyword arguments (line 224)
                kwargs_277 = {}
                # Getting the type of 'copy' (line 224)
                copy_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 224)
                deepcopy_274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), copy_273, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 224)
                deepcopy_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 224, 43), deepcopy_274, *[python_entity_276], **kwargs_277)
                
                # Getting the type of 'result' (line 224)
                result_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 224)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), result_279, 'python_entity', deepcopy_call_result_278)
                # SSA join for if statement (line 220)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 212)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 207)
        if_condition_231 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 8), result_not__230)
        # Assigning a type to the variable 'if_condition_231' (line 207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'if_condition_231', if_condition_231)
        # SSA begins for if statement (line 207)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 208):
        
        # Call to deepcopy(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'type_inference_proxy_obj' (line 208)
        type_inference_proxy_obj_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 49), 'type_inference_proxy_obj', False)
        # Obtaining the member 'python_entity' of a type (line 208)
        python_entity_235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 49), type_inference_proxy_obj_234, 'python_entity')
        # Processing the call keyword arguments (line 208)
        kwargs_236 = {}
        # Getting the type of 'copy' (line 208)
        copy_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 35), 'copy', False)
        # Obtaining the member 'deepcopy' of a type (line 208)
        deepcopy_233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 35), copy_232, 'deepcopy')
        # Calling deepcopy(args, kwargs) (line 208)
        deepcopy_call_result_237 = invoke(stypy.reporting.localization.Localization(__file__, 208, 35), deepcopy_233, *[python_entity_235], **kwargs_236)
        
        # Getting the type of 'result' (line 208)
        result_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'result')
        # Setting the type of the member 'python_entity' of a type (line 208)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 12), result_238, 'python_entity', deepcopy_call_result_237)
        # SSA branch for the else part of an if statement (line 207)
        module_type_store.open_ssa_branch('else')
        
        # Call to isclass(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'type_inference_proxy_obj' (line 212)
        type_inference_proxy_obj_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 31), 'type_inference_proxy_obj', False)
        # Obtaining the member 'python_entity' of a type (line 212)
        python_entity_242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 31), type_inference_proxy_obj_241, 'python_entity')
        # Processing the call keyword arguments (line 212)
        kwargs_243 = {}
        # Getting the type of 'inspect' (line 212)
        inspect_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 15), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 212)
        isclass_240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 15), inspect_239, 'isclass')
        # Calling isclass(args, kwargs) (line 212)
        isclass_call_result_244 = invoke(stypy.reporting.localization.Localization(__file__, 212, 15), isclass_240, *[python_entity_242], **kwargs_243)
        
        # Testing if the type of an if condition is none (line 212)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 212, 12), isclass_call_result_244):
            
            # Call to isfunction(...): (line 220)
            # Processing the call arguments (line 220)
            # Getting the type of 'type_inference_proxy_obj' (line 220)
            type_inference_proxy_obj_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'type_inference_proxy_obj', False)
            # Obtaining the member 'python_entity' of a type (line 220)
            python_entity_263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 38), type_inference_proxy_obj_262, 'python_entity')
            # Processing the call keyword arguments (line 220)
            kwargs_264 = {}
            # Getting the type of 'inspect' (line 220)
            inspect_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'inspect', False)
            # Obtaining the member 'isfunction' of a type (line 220)
            isfunction_261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 19), inspect_260, 'isfunction')
            # Calling isfunction(args, kwargs) (line 220)
            isfunction_call_result_265 = invoke(stypy.reporting.localization.Localization(__file__, 220, 19), isfunction_261, *[python_entity_263], **kwargs_264)
            
            # Testing if the type of an if condition is none (line 220)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 220, 16), isfunction_call_result_265):
                
                # Assigning a Call to a Attribute (line 224):
                
                # Call to deepcopy(...): (line 224)
                # Processing the call arguments (line 224)
                # Getting the type of 'type_inference_proxy_obj' (line 224)
                type_inference_proxy_obj_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 224)
                python_entity_276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 57), type_inference_proxy_obj_275, 'python_entity')
                # Processing the call keyword arguments (line 224)
                kwargs_277 = {}
                # Getting the type of 'copy' (line 224)
                copy_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 224)
                deepcopy_274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), copy_273, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 224)
                deepcopy_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 224, 43), deepcopy_274, *[python_entity_276], **kwargs_277)
                
                # Getting the type of 'result' (line 224)
                result_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 224)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), result_279, 'python_entity', deepcopy_call_result_278)
            else:
                
                # Testing the type of an if condition (line 220)
                if_condition_266 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 16), isfunction_call_result_265)
                # Assigning a type to the variable 'if_condition_266' (line 220)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'if_condition_266', if_condition_266)
                # SSA begins for if statement (line 220)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Attribute (line 221):
                
                # Call to __duplicate_function(...): (line 221)
                # Processing the call arguments (line 221)
                # Getting the type of 'type_inference_proxy_obj' (line 221)
                type_inference_proxy_obj_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 64), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 221)
                python_entity_269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 64), type_inference_proxy_obj_268, 'python_entity')
                # Processing the call keyword arguments (line 221)
                kwargs_270 = {}
                # Getting the type of '__duplicate_function' (line 221)
                duplicate_function_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 43), '__duplicate_function', False)
                # Calling __duplicate_function(args, kwargs) (line 221)
                duplicate_function_call_result_271 = invoke(stypy.reporting.localization.Localization(__file__, 221, 43), duplicate_function_267, *[python_entity_269], **kwargs_270)
                
                # Getting the type of 'result' (line 221)
                result_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 221)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 20), result_272, 'python_entity', duplicate_function_call_result_271)
                # SSA branch for the else part of an if statement (line 220)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Attribute (line 224):
                
                # Call to deepcopy(...): (line 224)
                # Processing the call arguments (line 224)
                # Getting the type of 'type_inference_proxy_obj' (line 224)
                type_inference_proxy_obj_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 224)
                python_entity_276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 57), type_inference_proxy_obj_275, 'python_entity')
                # Processing the call keyword arguments (line 224)
                kwargs_277 = {}
                # Getting the type of 'copy' (line 224)
                copy_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 224)
                deepcopy_274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), copy_273, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 224)
                deepcopy_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 224, 43), deepcopy_274, *[python_entity_276], **kwargs_277)
                
                # Getting the type of 'result' (line 224)
                result_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 224)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), result_279, 'python_entity', deepcopy_call_result_278)
                # SSA join for if statement (line 220)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 212)
            if_condition_245 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 12), isclass_call_result_244)
            # Assigning a type to the variable 'if_condition_245' (line 212)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'if_condition_245', if_condition_245)
            # SSA begins for if statement (line 212)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Type idiom detected: calculating its left and rigth part (line 213)
            # Getting the type of 'type_inference_proxy_obj' (line 213)
            type_inference_proxy_obj_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'type_inference_proxy_obj')
            # Obtaining the member 'instance' of a type (line 213)
            instance_247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 19), type_inference_proxy_obj_246, 'instance')
            # Getting the type of 'None' (line 213)
            None_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 56), 'None')
            
            (may_be_249, more_types_in_union_250) = may_be_none(instance_247, None_248)

            if may_be_249:

                if more_types_in_union_250:
                    # Runtime conditional SSA (line 213)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Call to a Attribute (line 214):
                
                # Call to __duplicate_class(...): (line 214)
                # Processing the call arguments (line 214)
                # Getting the type of 'type_inference_proxy_obj' (line 214)
                type_inference_proxy_obj_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 61), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 214)
                python_entity_253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 61), type_inference_proxy_obj_252, 'python_entity')
                # Processing the call keyword arguments (line 214)
                kwargs_254 = {}
                # Getting the type of '__duplicate_class' (line 214)
                duplicate_class_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 43), '__duplicate_class', False)
                # Calling __duplicate_class(args, kwargs) (line 214)
                duplicate_class_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 214, 43), duplicate_class_251, *[python_entity_253], **kwargs_254)
                
                # Getting the type of 'result' (line 214)
                result_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 214)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 20), result_256, 'python_entity', duplicate_class_call_result_255)

                if more_types_in_union_250:
                    # Runtime conditional SSA for else branch (line 213)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_249) or more_types_in_union_250):
                
                # Assigning a Attribute to a Attribute (line 217):
                # Getting the type of 'type_inference_proxy_obj' (line 217)
                type_inference_proxy_obj_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 43), 'type_inference_proxy_obj')
                # Obtaining the member 'python_entity' of a type (line 217)
                python_entity_258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 43), type_inference_proxy_obj_257, 'python_entity')
                # Getting the type of 'result' (line 217)
                result_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 217)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 20), result_259, 'python_entity', python_entity_258)

                if (may_be_249 and more_types_in_union_250):
                    # SSA join for if statement (line 213)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA branch for the else part of an if statement (line 212)
            module_type_store.open_ssa_branch('else')
            
            # Call to isfunction(...): (line 220)
            # Processing the call arguments (line 220)
            # Getting the type of 'type_inference_proxy_obj' (line 220)
            type_inference_proxy_obj_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'type_inference_proxy_obj', False)
            # Obtaining the member 'python_entity' of a type (line 220)
            python_entity_263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 38), type_inference_proxy_obj_262, 'python_entity')
            # Processing the call keyword arguments (line 220)
            kwargs_264 = {}
            # Getting the type of 'inspect' (line 220)
            inspect_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'inspect', False)
            # Obtaining the member 'isfunction' of a type (line 220)
            isfunction_261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 19), inspect_260, 'isfunction')
            # Calling isfunction(args, kwargs) (line 220)
            isfunction_call_result_265 = invoke(stypy.reporting.localization.Localization(__file__, 220, 19), isfunction_261, *[python_entity_263], **kwargs_264)
            
            # Testing if the type of an if condition is none (line 220)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 220, 16), isfunction_call_result_265):
                
                # Assigning a Call to a Attribute (line 224):
                
                # Call to deepcopy(...): (line 224)
                # Processing the call arguments (line 224)
                # Getting the type of 'type_inference_proxy_obj' (line 224)
                type_inference_proxy_obj_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 224)
                python_entity_276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 57), type_inference_proxy_obj_275, 'python_entity')
                # Processing the call keyword arguments (line 224)
                kwargs_277 = {}
                # Getting the type of 'copy' (line 224)
                copy_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 224)
                deepcopy_274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), copy_273, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 224)
                deepcopy_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 224, 43), deepcopy_274, *[python_entity_276], **kwargs_277)
                
                # Getting the type of 'result' (line 224)
                result_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 224)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), result_279, 'python_entity', deepcopy_call_result_278)
            else:
                
                # Testing the type of an if condition (line 220)
                if_condition_266 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 16), isfunction_call_result_265)
                # Assigning a type to the variable 'if_condition_266' (line 220)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'if_condition_266', if_condition_266)
                # SSA begins for if statement (line 220)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Attribute (line 221):
                
                # Call to __duplicate_function(...): (line 221)
                # Processing the call arguments (line 221)
                # Getting the type of 'type_inference_proxy_obj' (line 221)
                type_inference_proxy_obj_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 64), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 221)
                python_entity_269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 64), type_inference_proxy_obj_268, 'python_entity')
                # Processing the call keyword arguments (line 221)
                kwargs_270 = {}
                # Getting the type of '__duplicate_function' (line 221)
                duplicate_function_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 43), '__duplicate_function', False)
                # Calling __duplicate_function(args, kwargs) (line 221)
                duplicate_function_call_result_271 = invoke(stypy.reporting.localization.Localization(__file__, 221, 43), duplicate_function_267, *[python_entity_269], **kwargs_270)
                
                # Getting the type of 'result' (line 221)
                result_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 221)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 20), result_272, 'python_entity', duplicate_function_call_result_271)
                # SSA branch for the else part of an if statement (line 220)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Attribute (line 224):
                
                # Call to deepcopy(...): (line 224)
                # Processing the call arguments (line 224)
                # Getting the type of 'type_inference_proxy_obj' (line 224)
                type_inference_proxy_obj_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 57), 'type_inference_proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 224)
                python_entity_276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 57), type_inference_proxy_obj_275, 'python_entity')
                # Processing the call keyword arguments (line 224)
                kwargs_277 = {}
                # Getting the type of 'copy' (line 224)
                copy_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 224)
                deepcopy_274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), copy_273, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 224)
                deepcopy_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 224, 43), deepcopy_274, *[python_entity_276], **kwargs_277)
                
                # Getting the type of 'result' (line 224)
                result_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'result')
                # Setting the type of the member 'python_entity' of a type (line 224)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 20), result_279, 'python_entity', deepcopy_call_result_278)
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
    Exception_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 'Exception')
    # Assigning a type to the variable 'ex' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'ex', Exception_280)
    
    # Call to ismodule(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'type_inference_proxy_obj' (line 228)
    type_inference_proxy_obj_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 28), 'type_inference_proxy_obj', False)
    # Obtaining the member 'python_entity' of a type (line 228)
    python_entity_284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 28), type_inference_proxy_obj_283, 'python_entity')
    # Processing the call keyword arguments (line 228)
    kwargs_285 = {}
    # Getting the type of 'inspect' (line 228)
    inspect_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'inspect', False)
    # Obtaining the member 'ismodule' of a type (line 228)
    ismodule_282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 11), inspect_281, 'ismodule')
    # Calling ismodule(args, kwargs) (line 228)
    ismodule_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 228, 11), ismodule_282, *[python_entity_284], **kwargs_285)
    
    # Testing if the type of an if condition is none (line 228)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 228, 8), ismodule_call_result_286):
        
        # Assigning a Call to a Attribute (line 231):
        
        # Call to copy(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'type_inference_proxy_obj' (line 231)
        type_inference_proxy_obj_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 45), 'type_inference_proxy_obj', False)
        # Obtaining the member 'python_entity' of a type (line 231)
        python_entity_297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 45), type_inference_proxy_obj_296, 'python_entity')
        # Processing the call keyword arguments (line 231)
        kwargs_298 = {}
        # Getting the type of 'copy' (line 231)
        copy_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 35), 'copy', False)
        # Obtaining the member 'copy' of a type (line 231)
        copy_295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 35), copy_294, 'copy')
        # Calling copy(args, kwargs) (line 231)
        copy_call_result_299 = invoke(stypy.reporting.localization.Localization(__file__, 231, 35), copy_295, *[python_entity_297], **kwargs_298)
        
        # Getting the type of 'result' (line 231)
        result_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'result')
        # Setting the type of the member 'python_entity' of a type (line 231)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), result_300, 'python_entity', copy_call_result_299)
    else:
        
        # Testing the type of an if condition (line 228)
        if_condition_287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 8), ismodule_call_result_286)
        # Assigning a type to the variable 'if_condition_287' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'if_condition_287', if_condition_287)
        # SSA begins for if statement (line 228)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 229):
        
        # Call to __clone_module(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'type_inference_proxy_obj' (line 229)
        type_inference_proxy_obj_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 50), 'type_inference_proxy_obj', False)
        # Obtaining the member 'python_entity' of a type (line 229)
        python_entity_290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 50), type_inference_proxy_obj_289, 'python_entity')
        # Processing the call keyword arguments (line 229)
        kwargs_291 = {}
        # Getting the type of '__clone_module' (line 229)
        clone_module_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 35), '__clone_module', False)
        # Calling __clone_module(args, kwargs) (line 229)
        clone_module_call_result_292 = invoke(stypy.reporting.localization.Localization(__file__, 229, 35), clone_module_288, *[python_entity_290], **kwargs_291)
        
        # Getting the type of 'result' (line 229)
        result_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'result')
        # Setting the type of the member 'python_entity' of a type (line 229)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), result_293, 'python_entity', clone_module_call_result_292)
        # SSA branch for the else part of an if statement (line 228)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 231):
        
        # Call to copy(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'type_inference_proxy_obj' (line 231)
        type_inference_proxy_obj_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 45), 'type_inference_proxy_obj', False)
        # Obtaining the member 'python_entity' of a type (line 231)
        python_entity_297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 45), type_inference_proxy_obj_296, 'python_entity')
        # Processing the call keyword arguments (line 231)
        kwargs_298 = {}
        # Getting the type of 'copy' (line 231)
        copy_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 35), 'copy', False)
        # Obtaining the member 'copy' of a type (line 231)
        copy_295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 35), copy_294, 'copy')
        # Calling copy(args, kwargs) (line 231)
        copy_call_result_299 = invoke(stypy.reporting.localization.Localization(__file__, 231, 35), copy_295, *[python_entity_297], **kwargs_298)
        
        # Getting the type of 'result' (line 231)
        result_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'result')
        # Setting the type of the member 'python_entity' of a type (line 231)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), result_300, 'python_entity', copy_call_result_299)
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
    type_inference_proxy_obj_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 40), 'type_inference_proxy_obj', False)
    # Obtaining the member 'instance' of a type (line 235)
    instance_304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 40), type_inference_proxy_obj_303, 'instance')
    # Processing the call keyword arguments (line 235)
    kwargs_305 = {}
    # Getting the type of 'copy' (line 235)
    copy_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 26), 'copy', False)
    # Obtaining the member 'deepcopy' of a type (line 235)
    deepcopy_302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 26), copy_301, 'deepcopy')
    # Calling deepcopy(args, kwargs) (line 235)
    deepcopy_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 235, 26), deepcopy_302, *[instance_304], **kwargs_305)
    
    # Getting the type of 'result' (line 235)
    result_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'result')
    # Setting the type of the member 'instance' of a type (line 235)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), result_307, 'instance', deepcopy_call_result_306)
    # SSA branch for the except part of a try statement (line 234)
    # SSA branch for the except '<any exception>' branch of a try statement (line 234)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Attribute (line 237):
    
    # Call to copy(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'type_inference_proxy_obj' (line 237)
    type_inference_proxy_obj_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 36), 'type_inference_proxy_obj', False)
    # Obtaining the member 'instance' of a type (line 237)
    instance_311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 36), type_inference_proxy_obj_310, 'instance')
    # Processing the call keyword arguments (line 237)
    kwargs_312 = {}
    # Getting the type of 'copy' (line 237)
    copy_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 26), 'copy', False)
    # Obtaining the member 'copy' of a type (line 237)
    copy_309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 26), copy_308, 'copy')
    # Calling copy(args, kwargs) (line 237)
    copy_call_result_313 = invoke(stypy.reporting.localization.Localization(__file__, 237, 26), copy_309, *[instance_311], **kwargs_312)
    
    # Getting the type of 'result' (line 237)
    result_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'result')
    # Setting the type of the member 'instance' of a type (line 237)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), result_314, 'instance', copy_call_result_313)
    # SSA join for try-except statement (line 234)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to hasattr(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'type_inference_proxy_obj' (line 240)
    type_inference_proxy_obj_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'type_inference_proxy_obj', False)
    # Getting the type of 'type_inference_proxy_obj' (line 240)
    type_inference_proxy_obj_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 41), 'type_inference_proxy_obj', False)
    # Obtaining the member 'contained_elements_property_name' of a type (line 240)
    contained_elements_property_name_318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 41), type_inference_proxy_obj_317, 'contained_elements_property_name')
    # Processing the call keyword arguments (line 240)
    kwargs_319 = {}
    # Getting the type of 'hasattr' (line 240)
    hasattr_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 7), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 240)
    hasattr_call_result_320 = invoke(stypy.reporting.localization.Localization(__file__, 240, 7), hasattr_315, *[type_inference_proxy_obj_316, contained_elements_property_name_318], **kwargs_319)
    
    # Testing if the type of an if condition is none (line 240)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 240, 4), hasattr_call_result_320):
        pass
    else:
        
        # Testing the type of an if condition (line 240)
        if_condition_321 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 4), hasattr_call_result_320)
        # Assigning a type to the variable 'if_condition_321' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'if_condition_321', if_condition_321)
        # SSA begins for if statement (line 240)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 246):
        
        # Call to getattr(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'type_inference_proxy_obj' (line 246)
        type_inference_proxy_obj_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 37), 'type_inference_proxy_obj', False)
        # Getting the type of 'type_inference_proxy_obj' (line 247)
        type_inference_proxy_obj_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 37), 'type_inference_proxy_obj', False)
        # Obtaining the member 'contained_elements_property_name' of a type (line 247)
        contained_elements_property_name_325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 37), type_inference_proxy_obj_324, 'contained_elements_property_name')
        # Processing the call keyword arguments (line 246)
        kwargs_326 = {}
        # Getting the type of 'getattr' (line 246)
        getattr_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 29), 'getattr', False)
        # Calling getattr(args, kwargs) (line 246)
        getattr_call_result_327 = invoke(stypy.reporting.localization.Localization(__file__, 246, 29), getattr_322, *[type_inference_proxy_obj_323, contained_elements_property_name_325], **kwargs_326)
        
        # Assigning a type to the variable 'contained_elements' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'contained_elements', getattr_call_result_327)
        
        # Type idiom detected: calculating its left and rigth part (line 248)
        # Getting the type of 'contained_elements' (line 248)
        contained_elements_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 11), 'contained_elements')
        # Getting the type of 'None' (line 248)
        None_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 33), 'None')
        
        (may_be_330, more_types_in_union_331) = may_be_none(contained_elements_328, None_329)

        if may_be_330:

            if more_types_in_union_331:
                # Runtime conditional SSA (line 248)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to setattr(...): (line 249)
            # Processing the call arguments (line 249)
            # Getting the type of 'result' (line 249)
            result_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 20), 'result', False)
            # Getting the type of 'type_inference_proxy_obj' (line 249)
            type_inference_proxy_obj_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 28), 'type_inference_proxy_obj', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 249)
            contained_elements_property_name_335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 28), type_inference_proxy_obj_334, 'contained_elements_property_name')
            # Getting the type of 'None' (line 250)
            None_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 20), 'None', False)
            # Processing the call keyword arguments (line 249)
            kwargs_337 = {}
            # Getting the type of 'setattr' (line 249)
            setattr_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'setattr', False)
            # Calling setattr(args, kwargs) (line 249)
            setattr_call_result_338 = invoke(stypy.reporting.localization.Localization(__file__, 249, 12), setattr_332, *[result_333, contained_elements_property_name_335, None_336], **kwargs_337)
            

            if more_types_in_union_331:
                # Runtime conditional SSA for else branch (line 248)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_330) or more_types_in_union_331):
            
            
            # SSA begins for try-except statement (line 252)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Call to setattr(...): (line 254)
            # Processing the call arguments (line 254)
            # Getting the type of 'result' (line 254)
            result_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 24), 'result', False)
            # Getting the type of 'type_inference_proxy_obj' (line 254)
            type_inference_proxy_obj_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 32), 'type_inference_proxy_obj', False)
            # Obtaining the member 'contained_elements_property_name' of a type (line 254)
            contained_elements_property_name_342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 32), type_inference_proxy_obj_341, 'contained_elements_property_name')
            
            # Call to clone(...): (line 255)
            # Processing the call keyword arguments (line 255)
            kwargs_345 = {}
            # Getting the type of 'contained_elements' (line 255)
            contained_elements_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 24), 'contained_elements', False)
            # Obtaining the member 'clone' of a type (line 255)
            clone_344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 24), contained_elements_343, 'clone')
            # Calling clone(args, kwargs) (line 255)
            clone_call_result_346 = invoke(stypy.reporting.localization.Localization(__file__, 255, 24), clone_344, *[], **kwargs_345)
            
            # Processing the call keyword arguments (line 254)
            kwargs_347 = {}
            # Getting the type of 'setattr' (line 254)
            setattr_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 16), 'setattr', False)
            # Calling setattr(args, kwargs) (line 254)
            setattr_call_result_348 = invoke(stypy.reporting.localization.Localization(__file__, 254, 16), setattr_339, *[result_340, contained_elements_property_name_342, clone_call_result_346], **kwargs_347)
            
            # SSA branch for the except part of a try statement (line 252)
            # SSA branch for the except '<any exception>' branch of a try statement (line 252)
            module_type_store.open_ssa_branch('except')
            
            # Type idiom detected: calculating its left and rigth part (line 259)
            # Getting the type of 'dict' (line 259)
            dict_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 50), 'dict')
            # Getting the type of 'contained_elements' (line 259)
            contained_elements_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 30), 'contained_elements')
            
            (may_be_351, more_types_in_union_352) = may_be_subtype(dict_349, contained_elements_350)

            if may_be_351:

                if more_types_in_union_352:
                    # Runtime conditional SSA (line 259)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'contained_elements' (line 259)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'contained_elements', remove_not_subtype_from_union(contained_elements_350, dict))
                
                # Call to set_elements_type(...): (line 261)
                # Processing the call arguments (line 261)
                # Getting the type of 'None' (line 261)
                None_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 45), 'None', False)
                
                # Call to dict(...): (line 261)
                # Processing the call keyword arguments (line 261)
                kwargs_357 = {}
                # Getting the type of 'dict' (line 261)
                dict_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 51), 'dict', False)
                # Calling dict(args, kwargs) (line 261)
                dict_call_result_358 = invoke(stypy.reporting.localization.Localization(__file__, 261, 51), dict_356, *[], **kwargs_357)
                
                # Getting the type of 'False' (line 261)
                False_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 59), 'False', False)
                # Processing the call keyword arguments (line 261)
                kwargs_360 = {}
                # Getting the type of 'result' (line 261)
                result_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 20), 'result', False)
                # Obtaining the member 'set_elements_type' of a type (line 261)
                set_elements_type_354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 20), result_353, 'set_elements_type')
                # Calling set_elements_type(args, kwargs) (line 261)
                set_elements_type_call_result_361 = invoke(stypy.reporting.localization.Localization(__file__, 261, 20), set_elements_type_354, *[None_355, dict_call_result_358, False_359], **kwargs_360)
                
                
                
                # Call to keys(...): (line 262)
                # Processing the call keyword arguments (line 262)
                kwargs_364 = {}
                # Getting the type of 'contained_elements' (line 262)
                contained_elements_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 31), 'contained_elements', False)
                # Obtaining the member 'keys' of a type (line 262)
                keys_363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 31), contained_elements_362, 'keys')
                # Calling keys(args, kwargs) (line 262)
                keys_call_result_365 = invoke(stypy.reporting.localization.Localization(__file__, 262, 31), keys_363, *[], **kwargs_364)
                
                # Assigning a type to the variable 'keys_call_result_365' (line 262)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 20), 'keys_call_result_365', keys_call_result_365)
                # Testing if the for loop is going to be iterated (line 262)
                # Testing the type of a for loop iterable (line 262)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 262, 20), keys_call_result_365)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 262, 20), keys_call_result_365):
                    # Getting the type of the for loop variable (line 262)
                    for_loop_var_366 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 262, 20), keys_call_result_365)
                    # Assigning a type to the variable 'key' (line 262)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 20), 'key', for_loop_var_366)
                    # SSA begins for a for statement (line 262)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a Call to a Name (line 263):
                    
                    # Call to get_values_from_key(...): (line 263)
                    # Processing the call arguments (line 263)
                    # Getting the type of 'None' (line 263)
                    None_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 77), 'None', False)
                    # Getting the type of 'key' (line 263)
                    key_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 83), 'key', False)
                    # Processing the call keyword arguments (line 263)
                    kwargs_371 = {}
                    # Getting the type of 'type_inference_proxy_obj' (line 263)
                    type_inference_proxy_obj_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 32), 'type_inference_proxy_obj', False)
                    # Obtaining the member 'get_values_from_key' of a type (line 263)
                    get_values_from_key_368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 32), type_inference_proxy_obj_367, 'get_values_from_key')
                    # Calling get_values_from_key(args, kwargs) (line 263)
                    get_values_from_key_call_result_372 = invoke(stypy.reporting.localization.Localization(__file__, 263, 32), get_values_from_key_368, *[None_369, key_370], **kwargs_371)
                    
                    # Assigning a type to the variable 'value' (line 263)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 24), 'value', get_values_from_key_call_result_372)
                    
                    # Call to add_key_and_value_type(...): (line 264)
                    # Processing the call arguments (line 264)
                    # Getting the type of 'None' (line 264)
                    None_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 54), 'None', False)
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 264)
                    tuple_376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 61), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 264)
                    # Adding element type (line 264)
                    # Getting the type of 'key' (line 264)
                    key_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 61), 'key', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 61), tuple_376, key_377)
                    # Adding element type (line 264)
                    # Getting the type of 'value' (line 264)
                    value_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 66), 'value', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 264, 61), tuple_376, value_378)
                    
                    # Getting the type of 'False' (line 264)
                    False_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 74), 'False', False)
                    # Processing the call keyword arguments (line 264)
                    kwargs_380 = {}
                    # Getting the type of 'result' (line 264)
                    result_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 24), 'result', False)
                    # Obtaining the member 'add_key_and_value_type' of a type (line 264)
                    add_key_and_value_type_374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 24), result_373, 'add_key_and_value_type')
                    # Calling add_key_and_value_type(args, kwargs) (line 264)
                    add_key_and_value_type_call_result_381 = invoke(stypy.reporting.localization.Localization(__file__, 264, 24), add_key_and_value_type_374, *[None_375, tuple_376, False_379], **kwargs_380)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                

                if more_types_in_union_352:
                    # Runtime conditional SSA for else branch (line 259)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_351) or more_types_in_union_352):
                # Assigning a type to the variable 'contained_elements' (line 259)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 16), 'contained_elements', remove_subtype_from_union(contained_elements_350, dict))
                
                # Call to setattr(...): (line 267)
                # Processing the call arguments (line 267)
                # Getting the type of 'result' (line 267)
                result_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 28), 'result', False)
                # Getting the type of 'type_inference_proxy_obj' (line 267)
                type_inference_proxy_obj_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 36), 'type_inference_proxy_obj', False)
                # Obtaining the member 'contained_elements_property_name' of a type (line 267)
                contained_elements_property_name_385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 36), type_inference_proxy_obj_384, 'contained_elements_property_name')
                
                # Call to deepcopy(...): (line 268)
                # Processing the call arguments (line 268)
                # Getting the type of 'contained_elements' (line 268)
                contained_elements_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 42), 'contained_elements', False)
                # Processing the call keyword arguments (line 268)
                kwargs_389 = {}
                # Getting the type of 'copy' (line 268)
                copy_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 28), 'copy', False)
                # Obtaining the member 'deepcopy' of a type (line 268)
                deepcopy_387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 28), copy_386, 'deepcopy')
                # Calling deepcopy(args, kwargs) (line 268)
                deepcopy_call_result_390 = invoke(stypy.reporting.localization.Localization(__file__, 268, 28), deepcopy_387, *[contained_elements_388], **kwargs_389)
                
                # Processing the call keyword arguments (line 267)
                kwargs_391 = {}
                # Getting the type of 'setattr' (line 267)
                setattr_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 20), 'setattr', False)
                # Calling setattr(args, kwargs) (line 267)
                setattr_call_result_392 = invoke(stypy.reporting.localization.Localization(__file__, 267, 20), setattr_382, *[result_383, contained_elements_property_name_385, deepcopy_call_result_390], **kwargs_391)
                

                if (may_be_351 and more_types_in_union_352):
                    # SSA join for if statement (line 259)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for try-except statement (line 252)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_330 and more_types_in_union_331):
                # SSA join for if statement (line 248)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 240)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'result' (line 270)
    result_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type', result_393)
    
    # ################# End of '__deepest_possible_copy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__deepest_possible_copy' in the type store
    # Getting the type of 'stypy_return_type' (line 188)
    stypy_return_type_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_394)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__deepest_possible_copy'
    return stypy_return_type_394

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

    str_395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, (-1)), 'str', '\n    Clone a module. This is done by deleting the loaded module and reloading it again with a different name. Later on,\n    we restore the unloaded copy.\n    :param module: Module to clone.\n    :return: Clone of the module.\n    ')
    
    # Assigning a Attribute to a Name (line 280):
    # Getting the type of 'module' (line 280)
    module_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 23), 'module')
    # Obtaining the member '__dict__' of a type (line 280)
    dict___397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 23), module_396, '__dict__')
    # Assigning a type to the variable 'original_members' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'original_members', dict___397)
    
    
    # SSA begins for try-except statement (line 281)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    # Deleting a member
    # Getting the type of 'sys' (line 282)
    sys_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'sys')
    # Obtaining the member 'modules' of a type (line 282)
    modules_399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), sys_398, 'modules')
    
    # Obtaining the type of the subscript
    # Getting the type of 'module' (line 282)
    module_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'module')
    # Obtaining the member '__name__' of a type (line 282)
    name___401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 24), module_400, '__name__')
    # Getting the type of 'sys' (line 282)
    sys_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'sys')
    # Obtaining the member 'modules' of a type (line 282)
    modules_403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), sys_402, 'modules')
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), modules_403, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 282)
    subscript_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), getitem___404, name___401)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 282, 8), modules_399, subscript_call_result_405)
    # SSA branch for the except part of a try statement (line 281)
    # SSA branch for the except '<any exception>' branch of a try statement (line 281)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 281)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 286)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    str_406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 11), 'str', '_clone')
    # Getting the type of 'module' (line 287)
    module_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 23), 'module')
    # Obtaining the member '__name__' of a type (line 287)
    name___408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 23), module_407, '__name__')
    # Applying the binary operator 'in' (line 287)
    result_contains_409 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 11), 'in', str_406, name___408)
    
    # Testing if the type of an if condition is none (line 287)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 287, 8), result_contains_409):
        
        # Assigning a Attribute to a Name (line 290):
        # Getting the type of 'module' (line 290)
        module_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 31), 'module')
        # Obtaining the member '__name__' of a type (line 290)
        name___419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 31), module_418, '__name__')
        # Assigning a type to the variable 'real_module_name' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'real_module_name', name___419)
    else:
        
        # Testing the type of an if condition (line 287)
        if_condition_410 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 287, 8), result_contains_409)
        # Assigning a type to the variable 'if_condition_410' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'if_condition_410', if_condition_410)
        # SSA begins for if statement (line 287)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 288):
        
        # Call to replace(...): (line 288)
        # Processing the call arguments (line 288)
        str_414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 55), 'str', '_clone')
        str_415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 65), 'str', '')
        # Processing the call keyword arguments (line 288)
        kwargs_416 = {}
        # Getting the type of 'module' (line 288)
        module_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 31), 'module', False)
        # Obtaining the member '__name__' of a type (line 288)
        name___412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 31), module_411, '__name__')
        # Obtaining the member 'replace' of a type (line 288)
        replace_413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 31), name___412, 'replace')
        # Calling replace(args, kwargs) (line 288)
        replace_call_result_417 = invoke(stypy.reporting.localization.Localization(__file__, 288, 31), replace_413, *[str_414, str_415], **kwargs_416)
        
        # Assigning a type to the variable 'real_module_name' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'real_module_name', replace_call_result_417)
        # SSA branch for the else part of an if statement (line 287)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 290):
        # Getting the type of 'module' (line 290)
        module_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 31), 'module')
        # Obtaining the member '__name__' of a type (line 290)
        name___419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 31), module_418, '__name__')
        # Assigning a type to the variable 'real_module_name' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'real_module_name', name___419)
        # SSA join for if statement (line 287)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 291):
    
    # Call to load_module(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'module' (line 291)
    module_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 32), 'module', False)
    # Obtaining the member '__name__' of a type (line 291)
    name___423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 32), module_422, '__name__')
    str_424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 50), 'str', '_clone')
    # Applying the binary operator '+' (line 291)
    result_add_425 = python_operator(stypy.reporting.localization.Localization(__file__, 291, 32), '+', name___423, str_424)
    
    
    # Call to find_module(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'real_module_name' (line 291)
    real_module_name_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 77), 'real_module_name', False)
    # Processing the call keyword arguments (line 291)
    kwargs_429 = {}
    # Getting the type of 'imp' (line 291)
    imp_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 61), 'imp', False)
    # Obtaining the member 'find_module' of a type (line 291)
    find_module_427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 61), imp_426, 'find_module')
    # Calling find_module(args, kwargs) (line 291)
    find_module_call_result_430 = invoke(stypy.reporting.localization.Localization(__file__, 291, 61), find_module_427, *[real_module_name_428], **kwargs_429)
    
    # Processing the call keyword arguments (line 291)
    kwargs_431 = {}
    # Getting the type of 'imp' (line 291)
    imp_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'imp', False)
    # Obtaining the member 'load_module' of a type (line 291)
    load_module_421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 16), imp_420, 'load_module')
    # Calling load_module(args, kwargs) (line 291)
    load_module_call_result_432 = invoke(stypy.reporting.localization.Localization(__file__, 291, 16), load_module_421, *[result_add_425, find_module_call_result_430], **kwargs_431)
    
    # Assigning a type to the variable 'clone' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'clone', load_module_call_result_432)
    
    # Getting the type of 'original_members' (line 294)
    original_members_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 22), 'original_members')
    # Assigning a type to the variable 'original_members_433' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'original_members_433', original_members_433)
    # Testing if the for loop is going to be iterated (line 294)
    # Testing the type of a for loop iterable (line 294)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 294, 8), original_members_433)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 294, 8), original_members_433):
        # Getting the type of the for loop variable (line 294)
        for_loop_var_434 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 294, 8), original_members_433)
        # Assigning a type to the variable 'member' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'member', for_loop_var_434)
        # SSA begins for a for statement (line 294)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setattr(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'clone' (line 295)
        clone_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'clone', False)
        # Getting the type of 'member' (line 295)
        member_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 27), 'member', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'member' (line 295)
        member_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 52), 'member', False)
        # Getting the type of 'original_members' (line 295)
        original_members_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 35), 'original_members', False)
        # Obtaining the member '__getitem__' of a type (line 295)
        getitem___440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 35), original_members_439, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 295)
        subscript_call_result_441 = invoke(stypy.reporting.localization.Localization(__file__, 295, 35), getitem___440, member_438)
        
        # Processing the call keyword arguments (line 295)
        kwargs_442 = {}
        # Getting the type of 'setattr' (line 295)
        setattr_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 295)
        setattr_call_result_443 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), setattr_435, *[clone_436, member_437, subscript_call_result_441], **kwargs_442)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # SSA branch for the except part of a try statement (line 286)
    # SSA branch for the except 'Exception' branch of a try statement (line 286)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 297)
    Exception_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 11), 'Exception')
    # Assigning a type to the variable 'e' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'e', Exception_444)
    
    # Assigning a Name to a Name (line 298):
    # Getting the type of 'module' (line 298)
    module_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'module')
    # Assigning a type to the variable 'clone' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'clone', module_445)
    # SSA join for try-except statement (line 286)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 300):
    # Getting the type of 'module' (line 300)
    module_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 35), 'module')
    # Getting the type of 'sys' (line 300)
    sys_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'sys')
    # Obtaining the member 'modules' of a type (line 300)
    modules_448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 4), sys_447, 'modules')
    # Getting the type of 'module' (line 300)
    module_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'module')
    # Obtaining the member '__name__' of a type (line 300)
    name___450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 16), module_449, '__name__')
    # Storing an element on a container (line 300)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 300, 4), modules_448, (name___450, module_446))
    # Getting the type of 'clone' (line 301)
    clone_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 11), 'clone')
    # Assigning a type to the variable 'stypy_return_type' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'stypy_return_type', clone_451)
    
    # ################# End of '__clone_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__clone_module' in the type store
    # Getting the type of 'stypy_return_type' (line 273)
    stypy_return_type_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_452)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__clone_module'
    return stypy_return_type_452

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

    str_453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, (-1)), 'str', '\n    Launch the cloning procedure of a TypeInferenceProxy\n    :param entity: TypeInferenceProxy to clone\n    :return: Clone of the passed entity\n    ')
    
    
    # SSA begins for try-except statement (line 310)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to __deepest_possible_copy(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'entity' (line 311)
    entity_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 39), 'entity', False)
    # Processing the call keyword arguments (line 311)
    kwargs_456 = {}
    # Getting the type of '__deepest_possible_copy' (line 311)
    deepest_possible_copy_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 15), '__deepest_possible_copy', False)
    # Calling __deepest_possible_copy(args, kwargs) (line 311)
    deepest_possible_copy_call_result_457 = invoke(stypy.reporting.localization.Localization(__file__, 311, 15), deepest_possible_copy_454, *[entity_455], **kwargs_456)
    
    # Assigning a type to the variable 'stypy_return_type' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'stypy_return_type', deepest_possible_copy_call_result_457)
    # SSA branch for the except part of a try statement (line 310)
    # SSA branch for the except '<any exception>' branch of a try statement (line 310)
    module_type_store.open_ssa_branch('except')
    
    # Call to deepcopy(...): (line 313)
    # Processing the call arguments (line 313)
    # Getting the type of 'entity' (line 313)
    entity_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 29), 'entity', False)
    # Processing the call keyword arguments (line 313)
    kwargs_461 = {}
    # Getting the type of 'copy' (line 313)
    copy_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 15), 'copy', False)
    # Obtaining the member 'deepcopy' of a type (line 313)
    deepcopy_459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 15), copy_458, 'deepcopy')
    # Calling deepcopy(args, kwargs) (line 313)
    deepcopy_call_result_462 = invoke(stypy.reporting.localization.Localization(__file__, 313, 15), deepcopy_459, *[entity_460], **kwargs_461)
    
    # Assigning a type to the variable 'stypy_return_type' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'stypy_return_type', deepcopy_call_result_462)
    # SSA join for try-except statement (line 310)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'create_duplicate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_duplicate' in the type store
    # Getting the type of 'stypy_return_type' (line 304)
    stypy_return_type_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_463)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_duplicate'
    return stypy_return_type_463

# Assigning a type to the variable 'create_duplicate' (line 304)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 0), 'create_duplicate', create_duplicate)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
